from __future__ import annotations

# ============================================================
# Standard library imports
# ============================================================
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# ============================================================
import numpy as np
import soundfile as sf
import torch

# ============================================================
# Make project root importable
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# ============================================================
from src.adapters.gsv_prompt_tokenizer import GSVPromptTokenizer
from src.adapters.gsv_semantic_codec import GSVSemanticCodecV2
from src.adapters.gsv_t2s import GSVT2S
from src.adapters.gsv_text_frontend import GSVTextFrontend
from src.interfaces.stage2_io import Stage2Inputs
from src.pipelines.stage2_inference_pipeline import Stage2InferencePipeline
from src.features.fewshot_style_features import extract_log_mel_from_path
from src.eval.voice_quality_metrics import (
    ASRConfig,
    DNSMOSConfig,
    SpeakerSimConfig,
    VoiceMetricConfig,
    VoiceQualityEvaluator,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_SEMANTIC_RATE_HZ = 25.0
DEFAULT_ACOUSTIC_RATE_HZ = 22050.0 / 256.0



def build_v66_prompt_acoustic_extras(
    prompt_wav_path: str | Path,
    *,
    device: torch.device,
    model_config: dict[str, Any],
) -> dict[str, torch.Tensor | bool]:
    """
    Build prompt_acoustic fields for v6.6 ReferenceAcousticStyleEncoder inference.
    """
    n_mels = int(model_config.get("reference_acoustic_dim", model_config.get("acoustic_dim", 80)))
    prompt_acoustic = extract_log_mel_from_path(
        prompt_wav_path,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
    ).float()
    prompt_acoustic = prompt_acoustic.unsqueeze(0).to(device)
    prompt_acoustic_lengths = torch.tensor(
        [prompt_acoustic.shape[1]],
        dtype=torch.long,
        device=device,
    )
    return {
        "prompt_acoustic": prompt_acoustic,
        "prompt_acoustic_lengths": prompt_acoustic_lengths,
        "v66_reference_style_attached": True,
    }


# ============================================================
# JSON helpers
# ============================================================

def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if torch.is_tensor(obj):
        x = obj.detach().cpu()

        if x.numel() == 1:
            return x.item()

        if x.numel() <= 128:
            return x.tolist()

        xf = x.float()
        return {
            "type": "torch.Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "mean": float(xf.mean().item()) if x.numel() > 0 else None,
            "std": float(xf.std().item()) if x.numel() > 1 else None,
            "min": float(xf.min().item()) if x.numel() > 0 else None,
            "max": float(xf.max().item()) if x.numel() > 0 else None,
        }

    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return obj.item()

        if obj.size <= 128:
            return obj.tolist()

        return {
            "type": "np.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "mean": float(obj.mean()) if obj.size > 0 else None,
            "std": float(obj.std()) if obj.size > 1 else None,
            "min": float(obj.min()) if obj.size > 0 else None,
            "max": float(obj.max()) if obj.size > 0 else None,
        }

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


# ============================================================
# CLI helpers
# ============================================================

def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()

    if s in {"1", "true", "yes", "y", "on"}:
        return True

    if s in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Cannot parse bool from: {x}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot inference for Stage2 v6.4.1. "
            "Manual inputs: target text, reference audio path, reference audio text."
        )
    )

    # --------------------------------------------------------
    # Required user inputs
    # --------------------------------------------------------
    parser.add_argument("--target_text", type=str, required=True)
    parser.add_argument("--prompt_wav_path", type=str, required=True)
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="",
        help=(
            "Reference audio transcript. "
            "Required only when --stage1_text_mode prompt_plus_target. "
            "Not used when --stage1_text_mode target_only or diagnostic_like."
        ),
    )

    parser.add_argument("--target_language", type=str, default="zh")
    parser.add_argument("--prompt_language", type=str, default="zh")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)

    # --------------------------------------------------------
    # Optional GPT-SoVITS checkpoints
    # --------------------------------------------------------
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--sovits_checkpoint_path", type=str, default=None)

    # --------------------------------------------------------
    # Stage1 text construction
    # --------------------------------------------------------
    parser.add_argument(
        "--stage1_text_mode",
        type=str,
        default="diagnostic_like",
        choices=["diagnostic_like", "target_only", "prompt_plus_target"],
        help=(
            "diagnostic_like / target_only: Stage1 uses only target_text, "
            "matching the previous diagnosis script behavior. "
            "prompt_plus_target: Stage1 uses prompt_text + target_text."
        ),
    )

    parser.add_argument(
        "--prompt_target_separator",
        type=str,
        default="",
        help="Separator inserted between prompt_text and target_text when stage1_text_mode=prompt_plus_target.",
    )

    # --------------------------------------------------------
    # Stage1 sampling args
    # --------------------------------------------------------
    parser.add_argument("--stage1_top_k", type=int, default=15)
    parser.add_argument("--stage1_top_p", type=float, default=1.0)
    parser.add_argument("--stage1_temperature", type=float, default=1.0)
    parser.add_argument("--stage1_repetition_penalty", type=float, default=1.35)
    parser.add_argument("--stage1_early_stop_num", type=int, default=-1)

    # --------------------------------------------------------
    # Stage1 output crop
    # Stage1 输出裁剪
    # --------------------------------------------------------
    parser.add_argument(
        "--crop_stage1_prompt_prefix",
        type=str2bool,
        default=True,
        help=(
            "Whether to crop Stage1 output semantic and keep only target semantic. "
            "This should be true for prompt_plus_target mode."
        ),
    )
    parser.add_argument(
        "--stage1_crop_mode",
        type=str,
        default="auto_idx",
        choices=["auto_idx", "prompt_len", "none"],
        help=(
            "auto_idx: use idx returned by GPT-SoVITS infer_panel and keep pred_semantic[:, -idx:]. "
            "prompt_len: drop the first prompt semantic length tokens as fallback. "
            "none: do not crop."
        ),
    )

    parser.add_argument(
        "--stage1_append_tail_mute_tokens",
        type=int,
        default=0,
        help="Append GPT-SoVITS mute semantic tokens after cropped Stage1 predicted semantic.",
    )

    parser.add_argument(
        "--stage1_mute_token_id",
        type=int,
        default=486,
        help="GPT-SoVITS mute semantic token id. v1/v2/v3/v4 commonly use 486.",
    )

    # --------------------------------------------------------
    # v6.4.1 continuous semantic
    # --------------------------------------------------------
    parser.add_argument("--attach_continuous_semantic", type=str2bool, default=True)
    parser.add_argument("--continuous_dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--require_continuous_semantic", type=str2bool, default=True)

    # --------------------------------------------------------
    # Stage2 length control
    # --------------------------------------------------------
    parser.add_argument(
        "--target_length_mode",
        type=str,
        default="semantic_ratio",
        choices=["inferred", "semantic_ratio", "manual"],
        help=(
            "inferred: let Stage2 infer length; "
            "semantic_ratio: use ceil(T_sem * acoustic_rate / semantic_rate); "
            "manual: use --manual_target_length."
        ),
    )
    parser.add_argument("--manual_target_length", type=int, default=None)
    parser.add_argument("--length_scale", type=float, default=1.0)

    # --------------------------------------------------------
    # Stage2 sampling args
    # --------------------------------------------------------
    parser.add_argument(
        "--sampling_start_mode",
        type=str,
        default="coarse_only",
        choices=[
            "coarse_only",
            "coarse",
            "v6_3_coarse",
            "v63_coarse",
            "bootstrap_only",
            "bootstrap",
            "v6_2_bootstrap",
            "coarse_bootstrap",
            "v6_3_bootstrap",
            "residual_refine",
            "residual",
            "v6_3",
            "v63",
            "pure_noise",
            "noise",
            "default",
            "v6_1",
        ],
    )
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--use_heun", action="store_true")
    parser.add_argument("--bootstrap_t_start", type=float, default=0.075)
    parser.add_argument("--bootstrap_noise_temperature", type=float, default=0.3)
    parser.add_argument("--return_bootstrap_mel", type=str2bool, default=True)

    # --------------------------------------------------------
    # Vocoder args
    # --------------------------------------------------------
    parser.add_argument("--vocoder_type", type=str, default="hifigan")
    parser.add_argument("--vocoder_profile", type=str, default="universal_v1")
    parser.add_argument("--hifigan_root", type=str, default=None)
    parser.add_argument("--vocoder_checkpoint_path", type=str, default=None)
    parser.add_argument("--vocoder_config_path", type=str, default=None)

    # --------------------------------------------------------
    # Runtime / output controls
    # --------------------------------------------------------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_numpy", type=str2bool, default=True)
    parser.add_argument("--save_pt", type=str2bool, default=True)
    parser.add_argument("--save_peaknorm", type=str2bool, default=True)
    parser.add_argument("--copy_prompt_wav", type=str2bool, default=True)
    parser.add_argument("--suppress_stage1_progress", type=str2bool, default=True)

    # --------------------------------------------------------
    # Optional evaluation metrics
    # 可选推理质量指标
    # --------------------------------------------------------
    parser.add_argument("--enable_metrics", type=str2bool, default=False)
    parser.add_argument("--metrics_output_json", type=str, default=None)

    parser.add_argument("--compute_dnsmos", type=str2bool, default=True)
    parser.add_argument("--compute_wer", type=str2bool, default=True)
    parser.add_argument("--compute_speaker_sim", type=str2bool, default=True)

    parser.add_argument("--dnsmos_onnx_path", type=str, default=None)
    parser.add_argument("--dnsmos_sample_rate", type=int, default=16000)
    parser.add_argument("--dnsmos_chunk_sec", type=float, default=9.01)
    parser.add_argument("--dnsmos_hop_sec", type=float, default=9.01)
    parser.add_argument("--dnsmos_use_gpu", type=str2bool, default=False)

    parser.add_argument(
        "--asr_backend",
        type=str,
        default="faster_whisper",
        choices=["faster_whisper", "whisper", "none"],
    )
    parser.add_argument("--asr_model", type=str, default="medium")
    parser.add_argument("--asr_language", type=str, default="zh")
    parser.add_argument("--asr_device", type=str, default="cuda")
    parser.add_argument("--asr_compute_type", type=str, default="float16")

    parser.add_argument(
        "--speaker_backend",
        type=str,
        default="speechbrain_ecapa",
        choices=["speechbrain_ecapa", "none"],
    )
    parser.add_argument(
        "--speaker_model_source",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
    )
    parser.add_argument(
        "--speaker_model_savedir",
        type=str,
        default="pretrained_models/speechbrain_spkrec_ecapa_voxceleb",
    )
    parser.add_argument("--speaker_device", type=str, default="cuda")

    return parser.parse_args()


# ============================================================
# Context manager
# ============================================================

@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# ============================================================
# Audio / tensor helpers
# ============================================================

def save_wave(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = wav.detach().cpu().float().numpy()
    arr = np.squeeze(arr)

    if arr.ndim > 1:
        arr = arr[0]

    sf.write(str(path), arr, int(sr))


def peak_normalize_wave(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = wav.detach().abs().max().clamp_min(eps)
    return wav / peak * 0.98


def save_wave_with_optional_peaknorm(
    path: Path,
    wav: torch.Tensor,
    sr: int,
    save_peaknorm: bool,
) -> None:
    save_wave(path, wav, sr)

    if save_peaknorm:
        peak_path = path.with_name(path.stem + "_peaknorm.wav")
        save_wave(peak_path, peak_normalize_wave(wav), sr)


def ensure_1d_long(x: torch.Tensor, name: str) -> torch.LongTensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")

    y = x.detach().cpu().long()

    while y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)

    if y.ndim != 1:
        y = y.reshape(-1)

    return y.contiguous().long()

def extract_stage1_generated_len(stage1_aux: Any) -> int | None:
    """
    Extract generated target semantic length from Stage1 aux.

    GPT-SoVITS infer_panel commonly returns:
        pred_semantic, idx

    In our current wrapper:
        pred_semantic, stage1_aux = generate_semantic(...)
    where stage1_aux is usually idx.
    """
    x = stage1_aux

    if x is None:
        return None

    if isinstance(x, (int, float)):
        value = int(x)
        return value if value > 0 else None

    if torch.is_tensor(x):
        if x.numel() <= 0:
            return None
        value = int(x.detach().cpu().view(-1)[0].item())
        return value if value > 0 else None

    if isinstance(x, (list, tuple)):
        if len(x) <= 0:
            return None
        return extract_stage1_generated_len(x[0])

    return None

def crop_stage1_semantic_for_target(
    pred_semantic: torch.Tensor,
    *,
    stage1_aux: Any,
    prompt_len: int,
    crop_enabled: bool,
    crop_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Crop prompt prefix from Stage1 semantic output.

    Preferred behavior follows GPT-SoVITS:
        pred_semantic = pred_semantic[:, -idx:]

    where idx is returned by infer_panel(...).

    Returns:
        cropped_pred_semantic
        crop_meta
    """
    if not torch.is_tensor(pred_semantic):
        raise TypeError(f"pred_semantic must be Tensor, got {type(pred_semantic)}")

    x = pred_semantic.long()

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(f"pred_semantic must be [B,T] or [T], got {tuple(x.shape)}")

    raw_len = int(x.shape[1])
    prompt_len = int(prompt_len)

    meta: dict[str, Any] = {
        "crop_enabled": bool(crop_enabled),
        "crop_mode": str(crop_mode),
        "raw_pred_semantic_len": int(raw_len),
        "prompt_semantic_len": int(prompt_len),
        "stage1_generated_len_from_aux": None,
        "cropped": False,
        "cropped_pred_semantic_len": int(raw_len),
        "num_removed_from_front": 0,
        "reason": "",
    }

    if not crop_enabled or crop_mode == "none":
        meta["reason"] = "crop_disabled"
        return x.contiguous(), meta

    mode = str(crop_mode).strip().lower()

    if mode == "auto_idx":
        generated_len = extract_stage1_generated_len(stage1_aux)
        meta["stage1_generated_len_from_aux"] = generated_len

        if generated_len is None:
            meta["reason"] = "missing_stage1_idx_no_crop"
            return x.contiguous(), meta

        generated_len = int(max(1, min(generated_len, raw_len)))

        cropped = x[:, -generated_len:].contiguous()

        meta["cropped"] = generated_len < raw_len
        meta["cropped_pred_semantic_len"] = int(cropped.shape[1])
        meta["num_removed_from_front"] = int(raw_len - cropped.shape[1])
        meta["reason"] = "cropped_by_stage1_idx"

        return cropped, meta

    if mode == "prompt_len":
        if prompt_len <= 0:
            meta["reason"] = "invalid_prompt_len_no_crop"
            return x.contiguous(), meta

        if raw_len <= prompt_len:
            meta["reason"] = "raw_len_le_prompt_len_no_crop"
            return x.contiguous(), meta

        cropped = x[:, prompt_len:].contiguous()

        meta["cropped"] = True
        meta["cropped_pred_semantic_len"] = int(cropped.shape[1])
        meta["num_removed_from_front"] = int(prompt_len)
        meta["reason"] = "cropped_by_prompt_len"

        return cropped, meta

    raise ValueError(f"Unsupported crop_mode: {crop_mode}")


def append_tail_mute_tokens(
    semantic_tokens: torch.Tensor,
    *,
    num_tokens: int,
    mute_token_id: int,
) -> torch.Tensor:
    num_tokens = int(num_tokens)

    if num_tokens <= 0:
        return semantic_tokens

    if semantic_tokens.ndim != 2:
        raise ValueError(
            f"semantic_tokens must be [B,T], got {tuple(semantic_tokens.shape)}"
        )

    B = semantic_tokens.shape[0]
    tail = torch.full(
        size=(B, num_tokens),
        fill_value=int(mute_token_id),
        dtype=semantic_tokens.dtype,
        device=semantic_tokens.device,
    )

    return torch.cat([semantic_tokens, tail], dim=1)


def ensure_batched_mel(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")

    x = x.detach().cpu().float()

    if x.ndim == 2:
        x = x.unsqueeze(0)

    if x.ndim != 3:
        raise ValueError(f"{name} must be [B,T,C] or [T,C], got {tuple(x.shape)}")

    if x.shape[-1] != 80:
        raise ValueError(f"{name} last dim must be 80, got {tuple(x.shape)}")

    return x.contiguous().float()


def tensor_short_stats(x: torch.Tensor | None) -> dict[str, Any]:
    if x is None or not torch.is_tensor(x):
        return {"has_tensor": False}

    y = x.detach().cpu().float()

    return {
        "has_tensor": True,
        "shape": list(y.shape),
        "mean": float(y.mean().item()) if y.numel() > 0 else None,
        "std": float(y.std().item()) if y.numel() > 1 else None,
        "min": float(y.min().item()) if y.numel() > 0 else None,
        "max": float(y.max().item()) if y.numel() > 0 else None,
    }


def safe_divide_time_by_duration(seconds: float | None, duration_sec: float) -> float | None:
    if seconds is None:
        return None

    try:
        seconds = float(seconds)
        duration_sec = float(duration_sec)
    except Exception:
        return None

    if not math.isfinite(seconds) or not math.isfinite(duration_sec):
        return None

    if duration_sec <= 0:
        return None

    return float(seconds) / float(duration_sec)


# ============================================================
# Pipeline / codec construction
# ============================================================

def build_pipeline(device: torch.device) -> Stage2InferencePipeline:
    frontend = GSVTextFrontend(version="v2", device=device, use_half=False)
    prompt_tokenizer = GSVPromptTokenizer(version="v2", device=device, use_half=False)
    stage1 = GSVT2S(version="v2", device=device, use_half=False)

    return Stage2InferencePipeline(
        frontend=frontend,
        prompt_tokenizer=prompt_tokenizer,
        stage1=stage1,
        device=str(device),
    )


def build_stage1_text(
    *,
    prompt_text: str,
    target_text: str,
    stage1_text_mode: str,
    separator: str,
) -> str:
    """
    Build the text sequence used by Stage1 text2semantic.

    diagnostic_like:
        Match the previous diagnosis script:
        - prompt audio is used through prompt semantic tokens;
        - prompt transcript is not used;
        - Stage1 text is target_text only.

    target_only:
        Alias of diagnostic_like.

    prompt_plus_target:
        Use reference transcript + target text.
    """
    mode = str(stage1_text_mode).strip().lower()

    if mode in {"diagnostic_like", "target_only"}:
        return str(target_text)

    if mode == "prompt_plus_target":
        prompt_text = str(prompt_text)

        if not prompt_text.strip():
            raise ValueError(
                "--prompt_text is required when --stage1_text_mode prompt_plus_target."
            )

        return f"{prompt_text}{separator}{target_text}"

    raise ValueError(f"Unsupported stage1_text_mode: {stage1_text_mode}")


# ============================================================
# Continuous semantic helpers
# ============================================================

@torch.inference_mode()
def decode_pred_semantic_to_continuous(
    codec: GSVSemanticCodecV2,
    semantic_tokens: torch.Tensor,
    *,
    sovits_checkpoint_path: str | None,
    continuous_dtype: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_1d = ensure_1d_long(semantic_tokens, "semantic_tokens")

    dtype = torch.float16 if continuous_dtype == "float16" else torch.float32

    out = codec.decode_codes_to_continuous(
        tokens_1d,
        sovits_checkpoint_path=sovits_checkpoint_path,
        return_cpu=True,
        dtype=dtype,
    )

    continuous = out.continuous.float()

    if continuous.ndim != 3:
        raise ValueError(f"continuous semantic must be [B,T,C], got {tuple(continuous.shape)}")

    if continuous.shape[0] != 1 or continuous.shape[-1] != 768:
        raise ValueError(f"continuous semantic must be [1,T,768], got {tuple(continuous.shape)}")

    if continuous.shape[1] != tokens_1d.numel():
        raise ValueError(
            f"continuous length mismatch: tokens={tokens_1d.numel()}, continuous={continuous.shape[1]}"
        )

    lengths = torch.tensor([continuous.shape[1]], dtype=torch.long)

    return continuous.to(device), lengths.to(device)


def extract_encoder_continuous_stats(pipeline: Stage2InferencePipeline) -> dict[str, Any]:
    model = getattr(pipeline, "stage2_model", None)

    if model is None:
        return {
            "continuous_semantic_encoder_has_stats": 0,
            "continuous_semantic_encoder_used": -1,
            "continuous_semantic_encoder_gate": -1.0,
            "continuous_semantic_encoder_norm_mean": -1.0,
        }

    encoder = getattr(model, "condition_encoder", None)

    if encoder is None:
        return {
            "continuous_semantic_encoder_has_stats": 0,
            "continuous_semantic_encoder_used": -1,
            "continuous_semantic_encoder_gate": -1.0,
            "continuous_semantic_encoder_norm_mean": -1.0,
        }

    stats = getattr(encoder, "_last_continuous_semantic_stats", None)

    if not isinstance(stats, dict):
        return {
            "continuous_semantic_encoder_has_stats": 0,
            "continuous_semantic_encoder_used": -1,
            "continuous_semantic_encoder_gate": -1.0,
            "continuous_semantic_encoder_norm_mean": -1.0,
        }

    return {
        "continuous_semantic_encoder_has_stats": 1,
        "continuous_semantic_encoder_used": int(stats.get("used", 0)),
        "continuous_semantic_encoder_gate": float(stats.get("gate", -1.0)),
        "continuous_semantic_encoder_norm_mean": float(stats.get("continuous_norm_mean", -1.0)),
        "continuous_semantic_encoder_len_mean": float(stats.get("continuous_len_mean", -1.0)),
        "continuous_semantic_encoder_mode": str(stats.get("mode", "")),
    }


# ============================================================
# Target length
# ============================================================

def resolve_target_lengths(
    *,
    mode: str,
    manual_target_length: int | None,
    semantic_len: int,
    length_scale: float,
    acoustic_rate_hz: float,
    semantic_rate_hz: float,
    device: torch.device,
) -> torch.Tensor | None:
    mode = str(mode).strip().lower()

    if mode == "inferred":
        return None

    if mode == "manual":
        if manual_target_length is None:
            raise ValueError("--manual_target_length is required when --target_length_mode manual")
        length = int(manual_target_length)

    elif mode == "semantic_ratio":
        ratio = float(acoustic_rate_hz) / max(float(semantic_rate_hz), 1e-6)
        length = int(math.ceil(float(semantic_len) * ratio * float(length_scale)))

    else:
        raise ValueError(f"Unsupported target_length_mode: {mode}")

    length = max(int(length), 1)
    return torch.tensor([length], dtype=torch.long, device=device)


# ============================================================
# Stage2 sampling
# ============================================================

@torch.inference_mode()
def run_stage2_to_mel(
    pipeline: Stage2InferencePipeline,
    batch: Stage2Inputs,
    *,
    target_lengths: torch.Tensor | None,
    sampling_start_mode: str,
    num_steps: int,
    temperature: float,
    guidance_scale: float,
    use_heun: bool,
    bootstrap_t_start: float,
    bootstrap_noise_temperature: float | None,
    return_bootstrap_mel: bool,
) -> dict[str, Any]:
    assert pipeline.stage2_model is not None

    batch = pipeline._move_stage2_inputs_to_device(batch)

    mode = str(sampling_start_mode).strip().lower()

    if mode in {
        "coarse_only",
        "coarse",
        "v6_3_coarse",
        "v63_coarse",
        "bootstrap_only",
    }:
        if not hasattr(pipeline.stage2_model, "sample_coarse_only"):
            raise RuntimeError("Current Stage2 model does not support sample_coarse_only(...).")

        outputs = pipeline.stage2_model.sample_coarse_only(
            batch=batch,
            target_lengths=target_lengths,
            return_intermediates=bool(return_bootstrap_mel),
        )

    elif mode in {
        "bootstrap",
        "v6_2_bootstrap",
        "coarse_bootstrap",
        "v6_3_bootstrap",
    }:
        if not hasattr(pipeline.stage2_model, "sample_with_bootstrap"):
            raise RuntimeError("Current Stage2 model does not support sample_with_bootstrap(...).")

        outputs = pipeline.stage2_model.sample_with_bootstrap(
            batch=batch,
            num_steps=int(num_steps),
            target_lengths=target_lengths,
            temperature=float(temperature),
            use_heun=bool(use_heun),
            guidance_scale=float(guidance_scale),
            bootstrap_t_start=float(bootstrap_t_start),
            bootstrap_noise_temperature=bootstrap_noise_temperature,
            return_bootstrap_mel=bool(return_bootstrap_mel),
        )

    elif mode in {
        "residual_refine",
        "residual",
        "v6_3",
        "v63",
        "v6_3_residual",
    }:
        if not hasattr(pipeline.stage2_model, "sample_with_residual_refiner"):
            raise RuntimeError("Current Stage2 model does not support sample_with_residual_refiner(...).")

        outputs = pipeline.stage2_model.sample_with_residual_refiner(
            batch=batch,
            num_steps=int(num_steps),
            target_lengths=target_lengths,
            temperature=float(temperature),
            use_heun=bool(use_heun),
            guidance_scale=float(guidance_scale),
            return_intermediates=bool(return_bootstrap_mel),
        )

    elif mode in {"pure_noise", "noise", "default", "v6_1"}:
        outputs = pipeline.stage2_model.sample(
            batch=batch,
            num_steps=int(num_steps),
            target_lengths=target_lengths,
            temperature=float(temperature),
            use_heun=bool(use_heun),
            guidance_scale=float(guidance_scale),
        )

    else:
        raise ValueError(f"Unsupported sampling_start_mode: {sampling_start_mode}")

    encoder_stats = extract_encoder_continuous_stats(pipeline)

    return {
        "outputs": outputs,
        "predicted_mel": outputs.acoustic.detach().cpu(),
        "lengths": outputs.lengths.detach().cpu() if torch.is_tensor(outputs.lengths) else None,
        "stage2_aux": outputs.aux or {},
        "encoder_continuous_stats": encoder_stats,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    total_t0 = time.perf_counter()
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    requested_device = torch.device(args.device)

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device

    prompt_wav_path = Path(args.prompt_wav_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    stage2_ckpt = Path(args.stage2_ckpt).resolve()

    if not prompt_wav_path.exists():
        raise FileNotFoundError(f"prompt_wav_path not found: {prompt_wav_path}")

    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"stage2_ckpt not found: {stage2_ckpt}")

    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_stats: dict[str, Any] = {}

    print("====================================================")
    print("Stage2 v6.4.1 zero-shot inference started")
    print(f"target_text       : {args.target_text}")
    print(f"prompt_text       : {args.prompt_text}")
    print(f"prompt_wav_path   : {prompt_wav_path}")
    print(f"output_dir        : {output_dir}")
    print(f"stage2_ckpt       : {stage2_ckpt}")
    print(f"device            : {device}")
    print(f"stage1_text_mode  : {args.stage1_text_mode}")
    print("====================================================")

    if args.prompt_language != args.target_language:
        print(
            "[WARN] prompt_language != target_language. "
            "Current script uses target_language for Stage1/Stage2 target text. "
            "prompt_language is currently recorded for metadata only."
        )


    # --------------------------------------------------------
    # 1. Build pipeline and load Stage2
    # --------------------------------------------------------
    print("[1/7] Building pipeline and loading Stage2 checkpoint...")

    pipeline_t0 = time.perf_counter()

    pipeline = build_pipeline(device)
    ckpt_meta = pipeline.load_stage2_checkpoint(stage2_ckpt, strict=True)

    runtime_stats["pipeline_load_seconds"] = float(time.perf_counter() - pipeline_t0)

    model_config = ckpt_meta.get("model_config", {})
    stage2_args = ckpt_meta.get("args", {})

    acoustic_rate_hz = float(
        model_config.get(
            "acoustic_rate_hz",
            stage2_args.get("acoustic_rate_hz", DEFAULT_ACOUSTIC_RATE_HZ),
        )
    )
    semantic_rate_hz = float(
        model_config.get(
            "semantic_rate_hz",
            stage2_args.get("semantic_rate_hz", DEFAULT_SEMANTIC_RATE_HZ),
        )
    )

    print("stage2 model_version:", ckpt_meta.get("model_version"))
    print("stage2 model_variant:", ckpt_meta.get("model_variant"))
    print("best_epoch          :", ckpt_meta.get("best_epoch"))
    print("best_val_loss       :", ckpt_meta.get("best_val_loss"))
    print("acoustic_rate_hz    :", acoustic_rate_hz)
    print("semantic_rate_hz    :", semantic_rate_hz)

    # --------------------------------------------------------
    # 2. Build prompt semantic tokens from reference audio
    # --------------------------------------------------------
    print("[2/7] Extracting prompt semantic from reference audio...")

    prompt_semantic_t0 = time.perf_counter()

    prompt_kwargs = pipeline._filter_supported_kwargs(
        pipeline.prompt_tokenizer.extract_prompt_semantic_from_wav,
        {
            "wav_path": str(prompt_wav_path),
            "sovits_checkpoint_path": args.sovits_checkpoint_path,
        },
    )
    prompt_tokens = pipeline.prompt_tokenizer.extract_prompt_semantic_from_wav(**prompt_kwargs)

    runtime_stats["prompt_semantic_seconds"] = float(time.perf_counter() - prompt_semantic_t0)
    prompt_tokens = prompt_tokens.long().to(device)

    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    prompt_lengths = torch.tensor(
        [prompt_tokens.shape[1]],
        dtype=torch.long,
        device=device,
    )

    print("prompt_tokens shape:", tuple(prompt_tokens.shape))

    # --------------------------------------------------------
    # 3. Prepare text frontend features
    # --------------------------------------------------------
    print("[3/7] Preparing frontend features...")

    stage1_text = build_stage1_text(
        prompt_text=str(args.prompt_text),
        target_text=str(args.target_text),
        stage1_text_mode=str(args.stage1_text_mode),
        separator=str(args.prompt_target_separator),
    )

    frontend_t0 = time.perf_counter()

    stage1_phoneme_ids, stage1_phoneme_lens, stage1_bert, stage1_norm_text = (
        pipeline.frontend.prepare_ids_and_bert(
            text=stage1_text,
            language=str(args.target_language),
        )
    )

    target_phoneme_ids, target_phoneme_lens, target_bert, target_norm_text = (
        pipeline.frontend.prepare_ids_and_bert(
            text=str(args.target_text),
            language=str(args.target_language),
        )
    )

    print("stage1_text      :", stage1_text)
    print("stage1_norm_text :", stage1_norm_text)
    print("target_norm_text :", target_norm_text)
    print("stage1 phoneme shape:", tuple(stage1_phoneme_ids.shape))
    print("target phoneme shape:", tuple(target_phoneme_ids.shape))

    runtime_stats["frontend_seconds"] = float(time.perf_counter() - frontend_t0)

    # --------------------------------------------------------
    # 4. Run Stage1 text2semantic
    # --------------------------------------------------------
    print("[4/7] Running Stage1 text2semantic...")

    stage1_t0 = time.perf_counter()

    with suppress_stdout_stderr(bool(args.suppress_stage1_progress)):
        pred_semantic, stage1_aux = pipeline.stage1.generate_semantic(
            phoneme_ids=stage1_phoneme_ids,
            phoneme_lens=stage1_phoneme_lens,
            bert_feature=stage1_bert,
            prompt_tokens=prompt_tokens,
            checkpoint_path=args.stage1_ckpt,
            top_k=int(args.stage1_top_k),
            top_p=float(args.stage1_top_p),
            temperature=float(args.stage1_temperature),
            early_stop_num=int(args.stage1_early_stop_num),
            repetition_penalty=float(args.stage1_repetition_penalty),
        )

    pred_semantic = pred_semantic.long().to(device)

    if pred_semantic.ndim == 1:
        pred_semantic = pred_semantic.unsqueeze(0)

    raw_pred_semantic = pred_semantic.clone()

    pred_semantic, stage1_crop_meta = crop_stage1_semantic_for_target(
        pred_semantic,
        stage1_aux=stage1_aux,
        prompt_len=int(prompt_tokens.shape[1]),
        crop_enabled=bool(args.crop_stage1_prompt_prefix),
        crop_mode=str(args.stage1_crop_mode),
    )

    pred_semantic = pred_semantic.to(device)

    pred_semantic_len_before_tail_mute = int(pred_semantic.shape[1])

    pred_semantic = append_tail_mute_tokens(
        pred_semantic,
        num_tokens=int(args.stage1_append_tail_mute_tokens),
        mute_token_id=int(args.stage1_mute_token_id),
    )

    stage1_crop_meta["tail_mute_tokens_appended"] = int(args.stage1_append_tail_mute_tokens)
    stage1_crop_meta["mute_token_id"] = int(args.stage1_mute_token_id)
    stage1_crop_meta["semantic_len_before_tail_mute"] = pred_semantic_len_before_tail_mute
    stage1_crop_meta["semantic_len_after_tail_mute"] = int(pred_semantic.shape[1])

    semantic_lengths = torch.tensor(
        [pred_semantic.shape[1]],
        dtype=torch.long,
        device=device,
    )

    print("raw_pred_semantic shape    :", tuple(raw_pred_semantic.shape))
    print("cropped_pred_semantic shape:", tuple(pred_semantic.shape))
    print("stage1_crop_meta           :", stage1_crop_meta)

    runtime_stats["stage1_seconds"] = float(time.perf_counter() - stage1_t0)

    # --------------------------------------------------------
    # 5. Optionally attach continuous semantic
    # --------------------------------------------------------
    print("[5/7] Preparing continuous semantic...")

    continuous_t0 = time.perf_counter()

    semantic_continuous = None
    semantic_continuous_lengths = None
    codec_loaded = False

    if bool(args.attach_continuous_semantic):
        codec = GSVSemanticCodecV2(
            version="v2",
            device=device,
            use_half=False,
            expected_continuous_dim=768,
        )
        codec.ensure_loaded(sovits_checkpoint_path=args.sovits_checkpoint_path)
        codec_loaded = True

        semantic_continuous, semantic_continuous_lengths = decode_pred_semantic_to_continuous(
            codec,
            pred_semantic,
            sovits_checkpoint_path=args.sovits_checkpoint_path,
            continuous_dtype=str(args.continuous_dtype),
            device=device,
        )

        print("semantic_continuous shape:", tuple(semantic_continuous.shape))

    elif bool(args.require_continuous_semantic):
        raise RuntimeError(
            "--require_continuous_semantic true but --attach_continuous_semantic false."
        )

    runtime_stats["continuous_semantic_seconds"] = float(time.perf_counter() - continuous_t0)

    v66_reference_style_extras: dict[str, Any] = {}
    if pipeline.stage2_model is not None and bool(getattr(pipeline.stage2_model, "use_reference_acoustic_style", False)):
        v66_reference_style_extras = build_v66_prompt_acoustic_extras(
            prompt_wav_path,
            device=device,
            model_config=model_config,
        )
        print("prompt_acoustic shape:", tuple(v66_reference_style_extras["prompt_acoustic"].shape))

    # --------------------------------------------------------
    # 6. Build Stage2Inputs and run Stage2
    # --------------------------------------------------------
    print("[6/7] Running Stage2 acoustic model...")

    stage2_t0 = time.perf_counter()

    target_phoneme_ids = target_phoneme_ids.to(device)
    target_phoneme_lens = target_phoneme_lens.to(device)
    target_bert = target_bert.to(device)

    if target_phoneme_ids.ndim == 1:
        target_phoneme_ids = target_phoneme_ids.unsqueeze(0)

    if target_bert.ndim == 2:
        target_bert = target_bert.unsqueeze(0)

    batch = Stage2Inputs(
        semantic_tokens=pred_semantic,
        semantic_lengths=semantic_lengths,

        semantic_source_ids=torch.tensor([1], dtype=torch.long, device=device),
        semantic_reliability=torch.tensor([0.45], dtype=torch.float32, device=device),

        semantic_continuous=semantic_continuous,
        semantic_continuous_lengths=semantic_continuous_lengths,

        phoneme_ids=target_phoneme_ids.long(),
        phoneme_lens=target_phoneme_lens.long(),
        bert_feature=target_bert.float(),

        prompt_tokens=prompt_tokens.long(),
        prompt_lengths=prompt_lengths,

        target_acoustic=None,
        target_lengths=None,

        raw_text=str(args.target_text),
        norm_text=str(target_norm_text),
        language=str(args.target_language),
        extras={
            "mode": "zero_shot_v641",
            "target_text": str(args.target_text),
            "prompt_text": str(args.prompt_text),
            "stage1_text": stage1_text,
            "stage1_norm_text": stage1_norm_text,
            "target_norm_text": target_norm_text,
            "prompt_wav_path": str(prompt_wav_path),
            "stage1_aux": stage1_aux,
            "codec_loaded": bool(codec_loaded),
            **v66_reference_style_extras,
        },
    )

    target_lengths = resolve_target_lengths(
        mode=str(args.target_length_mode),
        manual_target_length=args.manual_target_length,
        semantic_len=int(semantic_lengths[0].item()),
        length_scale=float(args.length_scale),
        acoustic_rate_hz=acoustic_rate_hz,
        semantic_rate_hz=semantic_rate_hz,
        device=device,
    )

    if target_lengths is None:
        print("target_lengths: inferred by Stage2")
    else:
        print("target_lengths:", target_lengths.detach().cpu().tolist())

    stage2_result = run_stage2_to_mel(
        pipeline=pipeline,
        batch=batch,
        target_lengths=target_lengths,
        sampling_start_mode=str(args.sampling_start_mode),
        num_steps=int(args.num_steps),
        temperature=float(args.temperature),
        guidance_scale=float(args.guidance_scale),
        use_heun=bool(args.use_heun),
        bootstrap_t_start=float(args.bootstrap_t_start),
        bootstrap_noise_temperature=args.bootstrap_noise_temperature,
        return_bootstrap_mel=bool(args.return_bootstrap_mel),
    )

    runtime_stats["stage2_seconds"] = float(time.perf_counter() - stage2_t0)

    predicted_mel = stage2_result["predicted_mel"]
    predicted_lengths = stage2_result["lengths"]
    encoder_continuous_stats = stage2_result["encoder_continuous_stats"]

    print("predicted_mel shape:", tuple(predicted_mel.shape))
    print("predicted_lengths  :", predicted_lengths.tolist() if torch.is_tensor(predicted_lengths) else None)
    print("encoder_continuous_stats:", encoder_continuous_stats)

    # --------------------------------------------------------
    # 7. Decode vocoder and save outputs
    # --------------------------------------------------------
    print("[7/7] Decoding waveform and saving outputs...")

    vocoder_t0 = time.perf_counter()

    waveform, sample_rate = pipeline.decode_mel_to_wav(
        mel=predicted_mel,
        lengths=predicted_lengths,
        vocoder_type=str(args.vocoder_type),
        vocoder_profile=str(args.vocoder_profile),
        hifigan_root=args.hifigan_root,
        vocoder_checkpoint_path=args.vocoder_checkpoint_path,
        vocoder_config_path=args.vocoder_config_path,
    )

    runtime_stats["vocoder_seconds"] = float(time.perf_counter() - vocoder_t0)

    wav_path = output_dir / "zeroshot_v641.wav"

    save_wave_with_optional_peaknorm(
        wav_path,
        waveform,
        int(sample_rate),
        save_peaknorm=bool(args.save_peaknorm),
    )

    wav_info = sf.info(str(wav_path))
    audio_duration_sec = (
        float(wav_info.frames) / float(wav_info.samplerate)
        if wav_info.samplerate > 0
        else 0.0
    )

    runtime_stats["audio_duration_sec"] = float(audio_duration_sec)

    # Generation seconds:
    #   Excludes pipeline/model loading and excludes quality metrics.
    #   Includes the actual generation path from prompt processing to vocoder.
    #
    # 生成耗时：
    #   不包含模型加载，不包含质量评估；
    #   包含 prompt semantic、frontend、stage1、continuous semantic、stage2、vocoder。
    generation_component_keys = [
        "prompt_semantic_seconds",
        "frontend_seconds",
        "stage1_seconds",
        "continuous_semantic_seconds",
        "stage2_seconds",
        "vocoder_seconds",
    ]

    generation_seconds = 0.0
    for k in generation_component_keys:
        v = runtime_stats.get(k)
        if v is not None:
            generation_seconds += float(v)

    runtime_stats["generation_seconds"] = float(generation_seconds)
    runtime_stats["generation_rtf"] = safe_divide_time_by_duration(
        generation_seconds,
        audio_duration_sec,
    )

    # Backward-compatible alias.
    # 兼容旧字段：rtf 默认指 generation_rtf。
    runtime_stats["rtf"] = runtime_stats["generation_rtf"]

    metrics_result = None

    if bool(args.enable_metrics):
        print("[metrics] Running Resonastra quality metrics...")

        metrics_t0 = time.perf_counter()

        evaluator = VoiceQualityEvaluator(
            VoiceMetricConfig(
                compute_dnsmos=bool(args.compute_dnsmos),
                compute_asr_wer=bool(args.compute_wer),
                compute_speaker_sim=bool(args.compute_speaker_sim),
                dnsmos=DNSMOSConfig(
                    onnx_path=args.dnsmos_onnx_path,
                    sample_rate=int(args.dnsmos_sample_rate),
                    chunk_sec=float(args.dnsmos_chunk_sec),
                    hop_sec=float(args.dnsmos_hop_sec),
                    use_gpu=bool(args.dnsmos_use_gpu),
                ),
                asr=ASRConfig(
                    backend=str(args.asr_backend),
                    model_name=str(args.asr_model),
                    language=str(args.asr_language),
                    device=str(args.asr_device),
                    compute_type=str(args.asr_compute_type),
                ),
                speaker=SpeakerSimConfig(
                    backend=str(args.speaker_backend),
                    model_source=str(args.speaker_model_source),
                    savedir=str(args.speaker_model_savedir),
                    device=str(args.speaker_device),
                ),
            )
        )

        metrics_result = evaluator.evaluate_file(
            generated_wav_path=wav_path,
            target_text=str(args.target_text),
            reference_wav_path=prompt_wav_path,

            # Pass generation_seconds, not total_seconds.
            # 这里传 generation_seconds，不再传 total_seconds。
            inference_seconds=generation_seconds,
            generation_seconds=generation_seconds,
        )

        metrics_seconds = float(time.perf_counter() - metrics_t0)
        runtime_stats["metrics_seconds"] = metrics_seconds
        runtime_stats["metrics_rtf"] = safe_divide_time_by_duration(
            metrics_seconds,
            audio_duration_sec,
        )

        # Attach metric runtime into metrics_result.
        # 把评估耗时也写进 metrics_result。
        metrics_result["generation_seconds"] = generation_seconds
        metrics_result["generation_rtf"] = runtime_stats["generation_rtf"]
        metrics_result["metrics_seconds"] = metrics_seconds
        metrics_result["metrics_rtf"] = runtime_stats["metrics_rtf"]


        dnsmos = metrics_result.get("dnsmos", {}) or {}
        wer = metrics_result.get("wer", {}) or {}
        spk = metrics_result.get("speaker_sim", {}) or {}

        print(
            "[metrics] "
            f"DNSMOS_OVRL={dnsmos.get('ovrl')} "
            f"DNSMOS_SIG={dnsmos.get('sig')} "
            f"WER={wer.get('wer') if isinstance(wer, dict) else None} "
            f"CER={wer.get('cer') if isinstance(wer, dict) else None} "
            f"SpeakerSim={spk.get('cosine') if isinstance(spk, dict) else None} "
            f"generation_RTF={metrics_result.get('generation_rtf')} "
            f"metrics_RTF={metrics_result.get('metrics_rtf')} "
            f"total_RTF={metrics_result.get('total_rtf')}"
        )

    # Total seconds:
    #   Full script wall-clock time from process start to after optional metrics.
    #
    # 总耗时：
    #   从脚本开始到可选 metrics 结束后的完整墙钟时间。
    total_seconds = float(time.perf_counter() - total_t0)
    runtime_stats["total_seconds"] = total_seconds
    runtime_stats["total_rtf"] = safe_divide_time_by_duration(
        total_seconds,
        audio_duration_sec,
    )

    if metrics_result is not None:
        metrics_result["total_seconds"] = total_seconds
        metrics_result["total_rtf"] = runtime_stats["total_rtf"]
        metrics_result["rtf"] = runtime_stats["generation_rtf"]

        metrics_json_path = (
            Path(args.metrics_output_json).resolve()
            if args.metrics_output_json
            else output_dir / "zeroshot_v641_metrics.json"
        )
        save_json(metrics_result, metrics_json_path)

    if bool(args.copy_prompt_wav):
        shutil.copy2(prompt_wav_path, output_dir / "prompt_reference.wav")

    mel_npy_path = None
    if bool(args.save_numpy):
        mel_npy_path = output_dir / "zeroshot_v641_mel.npy"
        np.save(mel_npy_path, predicted_mel.squeeze(0).detach().cpu().numpy())

    mel_pt_path = None
    if bool(args.save_pt):
        mel_pt_path = output_dir / "zeroshot_v641_mel.pt"
        torch.save(
            {
                "mel": predicted_mel.squeeze(0).detach().cpu(),
                "lengths": predicted_lengths.detach().cpu() if torch.is_tensor(predicted_lengths) else None,
            },
            mel_pt_path,
        )

    summary = {
        "script": "infer_zeroshot_v641.py",
        "target_text": str(args.target_text),
        "prompt_text": str(args.prompt_text),
        "prompt_wav_path": str(prompt_wav_path),
        "output_dir": str(output_dir),

        "language": {
            "target_language": str(args.target_language),
            "prompt_language": str(args.prompt_language),
        },

        "stage1": {
            "stage1_text_mode": str(args.stage1_text_mode),
            "uses_prompt_text": bool(
                str(args.stage1_text_mode).strip().lower() == "prompt_plus_target"
            ),
            "diagnostic_like_no_prompt_text": bool(
                str(args.stage1_text_mode).strip().lower() in {"diagnostic_like", "target_only"}
            ),
            "prompt_target_separator": str(args.prompt_target_separator),
            "stage1_text": stage1_text,
            "stage1_norm_text": stage1_norm_text,
            "target_norm_text": target_norm_text,
            "pred_semantic_shape": list(pred_semantic.detach().cpu().shape),
            "raw_pred_semantic_shape": list(raw_pred_semantic.detach().cpu().shape),
            "stage1_crop_meta": stage1_crop_meta,
            "semantic_lengths": semantic_lengths.detach().cpu().tolist(),
            "stage1_aux": stage1_aux,
            "stage1_top_k": int(args.stage1_top_k),
            "stage1_top_p": float(args.stage1_top_p),
            "stage1_temperature": float(args.stage1_temperature),
            "stage1_repetition_penalty": float(args.stage1_repetition_penalty),
        },

        "continuous_semantic": {
            "attach_continuous_semantic": bool(args.attach_continuous_semantic),
            "codec_loaded": bool(codec_loaded),
            "semantic_continuous_stats": tensor_short_stats(semantic_continuous),
            "semantic_continuous_lengths": (
                semantic_continuous_lengths.detach().cpu().tolist()
                if torch.is_tensor(semantic_continuous_lengths)
                else None
            ),
            "encoder_continuous_stats": encoder_continuous_stats,
        },

        "stage2": {
            "stage2_ckpt": str(stage2_ckpt),
            "stage2_checkpoint_meta": pipeline.stage2_checkpoint_meta,
            "sampling_start_mode": str(args.sampling_start_mode),
            "target_length_mode": str(args.target_length_mode),
            "target_lengths": target_lengths.detach().cpu().tolist() if torch.is_tensor(target_lengths) else None,
            "num_steps": int(args.num_steps),
            "temperature": float(args.temperature),
            "guidance_scale": float(args.guidance_scale),
            "predicted_mel_shape": list(predicted_mel.shape),
            "predicted_lengths": predicted_lengths.tolist() if torch.is_tensor(predicted_lengths) else None,
            "stage2_aux": stage2_result.get("stage2_aux", {}),
            "acoustic_rate_hz": float(acoustic_rate_hz),
            "semantic_rate_hz": float(semantic_rate_hz),
        },

        "vocoder": {
            "vocoder_type": str(args.vocoder_type),
            "vocoder_profile": str(args.vocoder_profile),
            "sample_rate": int(sample_rate),
            "hifigan_root": args.hifigan_root,
        },

        "runtime": runtime_stats,

        "metrics": metrics_result,

        "outputs": {
            "wav_path": str(wav_path),
            "peaknorm_wav_path": str(wav_path.with_name(wav_path.stem + "_peaknorm.wav"))
            if bool(args.save_peaknorm)
            else None,
            "mel_npy_path": str(mel_npy_path) if mel_npy_path is not None else None,
            "mel_pt_path": str(mel_pt_path) if mel_pt_path is not None else None,
            "prompt_reference_copy": str(output_dir / "prompt_reference.wav")
            if bool(args.copy_prompt_wav)
            else None,
        },
    }

    summary_path = output_dir / "zeroshot_v641_summary.json"
    save_json(summary, summary_path)

    print("====================================================")
    print("[DONE] Stage2 v6.4.1 zero-shot inference finished.")
    print("output wav      :", wav_path)
    if bool(args.save_peaknorm):
        print("peaknorm wav    :", wav_path.with_name(wav_path.stem + "_peaknorm.wav"))
    print("summary json    :", summary_path)
    print("semantic len    :", int(semantic_lengths[0].item()))
    print("mel shape       :", tuple(predicted_mel.shape))
    print("encoder cont    :", encoder_continuous_stats)
    print("audio duration      :", runtime_stats.get("audio_duration_sec"))
    print("generation seconds  :", runtime_stats.get("generation_seconds"))
    print("generation rtf      :", runtime_stats.get("generation_rtf"))
    print("metrics seconds     :", runtime_stats.get("metrics_seconds"))
    print("metrics rtf         :", runtime_stats.get("metrics_rtf"))
    print("total seconds       :", runtime_stats.get("total_seconds"))
    print("total rtf           :", runtime_stats.get("total_rtf"))
    print("====================================================")


if __name__ == "__main__":
    main()