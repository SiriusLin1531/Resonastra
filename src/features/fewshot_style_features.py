from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import librosa
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Audio loading
# 音频读取
# ============================================================
def load_audio_mono(
    path: str | Path,
    *,
    sample_rate: int,
) -> torch.Tensor:
    """
    EN:
    Load an audio file as mono float32 waveform in [-1, 1]-like range.

    ZH:
    以单声道 float32 波形读取音频，数值范围通常接近 [-1, 1]。
    """
    wav_path = Path(path).expanduser().resolve()

    if not wav_path.exists() or not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    wav, _ = librosa.load(str(wav_path), sr=int(sample_rate), mono=True)

    if wav.size <= 0:
        raise ValueError(f"Loaded empty audio: {wav_path}")

    wav_tensor = torch.from_numpy(wav.astype(np.float32, copy=False)).contiguous()

    if not torch.isfinite(wav_tensor).all():
        raise ValueError(f"Audio contains NaN/Inf: {wav_path}")

    return wav_tensor


# ============================================================
# Mel / acoustic features
# Mel / 声学特征
# ============================================================
def extract_log_mel(
    wav: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """
    EN:
    Extract log power-mel features aligned with the existing Stage2 preprocessor.

    Output shape:
        [T, n_mels]

    ZH:
    提取 log power-mel，格式与当前 Stage2 预处理脚本保持一致。

    输出形状：
        [T, n_mels]
    """
    if wav.ndim != 1:
        raise ValueError(f"wav must be 1-D, got shape={tuple(wav.shape)}")

    wav_np = wav.detach().cpu().float().numpy()

    mel = librosa.feature.melspectrogram(
        y=wav_np,
        sr=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        n_mels=int(n_mels),
        fmin=float(f_min),
        fmax=None if f_max is None else float(f_max),
        power=2.0,
    )

    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    mel_tensor = torch.from_numpy(mel.astype(np.float32, copy=False)).transpose(0, 1).contiguous()

    if mel_tensor.ndim != 2 or mel_tensor.shape[-1] != int(n_mels):
        raise ValueError(
            f"log mel must be [T,{n_mels}], got shape={tuple(mel_tensor.shape)}"
        )

    if mel_tensor.shape[0] <= 0:
        raise ValueError("log mel has zero frames")

    if not torch.isfinite(mel_tensor).all():
        raise ValueError("log mel contains NaN/Inf")

    return mel_tensor


def extract_log_mel_from_path(
    wav_path: str | Path,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """
    EN:
    Convenience wrapper: load audio and extract log-mel.

    ZH:
    便捷封装：读取音频并提取 log-mel。
    """
    wav = load_audio_mono(wav_path, sample_rate=int(sample_rate))
    return extract_log_mel(
        wav,
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        n_mels=int(n_mels),
        f_min=float(f_min),
        f_max=f_max,
    )


# ============================================================
# Energy features
# 能量特征
# ============================================================
def log_mel_to_log_energy(log_mel: torch.Tensor) -> torch.Tensor:
    """
    EN:
    Convert log power-mel [T, M] to a stable log-energy curve [T].

    Instead of taking a simple mean over log bins, we estimate:
        log(mean(exp(log_mel)))
    using logsumexp for numerical stability.

    ZH:
    将 log power-mel [T, M] 转换成稳定的 log-energy 曲线 [T]。

    这里使用 logsumexp 估计 log(mean(exp(log_mel)))，比直接对 log-mel 求均值更合理。
    """
    if not torch.is_tensor(log_mel):
        raise TypeError(f"log_mel must be torch.Tensor, got {type(log_mel)}")

    x = log_mel.detach().cpu().float()

    if x.ndim != 2:
        raise ValueError(f"log_mel must be [T, M], got shape={tuple(x.shape)}")

    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError(f"log_mel has invalid shape={tuple(x.shape)}")

    if not torch.isfinite(x).all():
        raise ValueError("log_mel contains NaN/Inf")

    log_energy = torch.logsumexp(x, dim=-1) - float(np.log(max(int(x.shape[-1]), 1)))
    return log_energy.contiguous()


def normalize_1d_feature(
    x: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    EN:
    Z-normalize a 1-D feature. If variance is too small, return zeros.

    ZH:
    对一维特征做 z-score 归一化；若方差过小则返回零向量。
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape={tuple(x.shape)}")

    y = x.detach().cpu().float()
    std = y.std(unbiased=False)

    if float(std.item()) < float(eps):
        return torch.zeros_like(y)

    return ((y - y.mean()) / (std + float(eps))).contiguous()


# ============================================================
# F0 features
# F0 特征
# ============================================================
def extract_f0_pyin(
    wav: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    frame_length: int,
    f0_min_hz: float = 50.0,
    f0_max_hz: float = 1100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EN:
    Extract F0 with librosa.pyin.

    Returns:
        f0_hz: [T], unvoiced frames filled with 0
        voiced_mask: [T], bool

    ZH:
    使用 librosa.pyin 提取 F0。

    返回：
        f0_hz: [T]，无声/未检测帧填 0
        voiced_mask: [T]，bool
    """
    if wav.ndim != 1:
        raise ValueError(f"wav must be 1-D, got shape={tuple(wav.shape)}")

    wav_np = wav.detach().cpu().float().numpy()

    f0, voiced_flag, _ = librosa.pyin(
        wav_np,
        fmin=float(f0_min_hz),
        fmax=float(f0_max_hz),
        sr=int(sample_rate),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
    )

    if f0 is None:
        f0_np = np.zeros((1,), dtype=np.float32)
        voiced_np = np.zeros((1,), dtype=np.bool_)
    else:
        f0_np = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        voiced_np = np.asarray(voiced_flag, dtype=np.bool_)

    f0_tensor = torch.from_numpy(f0_np).float().contiguous()
    voiced_tensor = torch.from_numpy(voiced_np).bool().contiguous()

    if f0_tensor.numel() <= 0:
        f0_tensor = torch.zeros(1, dtype=torch.float32)
        voiced_tensor = torch.zeros(1, dtype=torch.bool)

    return f0_tensor, voiced_tensor


# ============================================================
# Alignment helpers
# 对齐工具
# ============================================================
def align_1d_feature_to_length(
    x: torch.Tensor,
    *,
    target_len: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    EN:
    Resample a 1-D feature to target_len using interpolation.

    ZH:
    使用插值将一维特征重采样到 target_len。
    """
    target_len = int(target_len)

    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    if not torch.is_tensor(x):
        raise TypeError(f"x must be torch.Tensor, got {type(x)}")

    y = x.detach().cpu().float().view(-1)

    if y.numel() <= 0:
        return torch.zeros(target_len, dtype=torch.float32)

    if y.numel() == target_len:
        return y.contiguous()

    if y.numel() == 1:
        return y.repeat(target_len).contiguous()

    z = y.view(1, 1, -1)

    if mode == "nearest":
        out = F.interpolate(z, size=target_len, mode="nearest")
    else:
        out = F.interpolate(z, size=target_len, mode="linear", align_corners=False)

    return out.view(-1).float().contiguous()


def align_bool_mask_to_length(
    mask: torch.Tensor,
    *,
    target_len: int,
) -> torch.Tensor:
    """
    EN:
    Resample a boolean mask to target_len using nearest interpolation.

    ZH:
    使用 nearest 插值将 bool mask 对齐到 target_len。
    """
    aligned = align_1d_feature_to_length(mask.float(), target_len=int(target_len), mode="nearest")
    return (aligned > 0.5).bool().contiguous()


# ============================================================
# Validation / stats
# 校验与统计
# ============================================================
def ensure_finite_tensor(
    x: torch.Tensor,
    *,
    name: str,
) -> None:
    """
    EN:
    Raise ValueError if a tensor contains NaN/Inf.

    ZH:
    如果 tensor 中包含 NaN/Inf，则抛出 ValueError。
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")

    if not torch.isfinite(x.detach().float()).all():
        raise ValueError(f"{name} contains NaN/Inf")


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    """
    EN:
    Return lightweight JSON-serializable tensor stats.

    ZH:
    返回轻量级、可 JSON 序列化的 tensor 统计信息。
    """
    if not torch.is_tensor(x):
        return {"type": str(type(x))}

    y = x.detach().cpu().float()

    if y.numel() <= 0:
        return {
            "shape": list(x.shape),
            "numel": int(y.numel()),
        }

    return {
        "shape": list(x.shape),
        "numel": int(y.numel()),
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
    }


def infer_acoustic_meta_from_sample(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """
    EN:
    Read acoustic extraction settings from an existing Stage2 sample.

    ZH:
    从已有 Stage2 样本中读取声学特征提取参数。
    """
    meta = sample.get("acoustic_meta", {})

    if not isinstance(meta, dict):
        meta = {}

    return {
        "target_sr": int(meta.get("target_sr", 22050)),
        "n_fft": int(meta.get("n_fft", 1024)),
        "hop_length": int(meta.get("hop_length", 256)),
        "win_length": int(meta.get("win_length", 1024)),
        "n_mels": int(meta.get("n_mels", 80)),
        "fmin": float(meta.get("fmin", 0.0)),
        "fmax": float(meta.get("fmax", 8000.0)),
        "acoustic_rate_hz": float(meta.get("acoustic_rate_hz", 22050 / 256)),
        "mel_backend": str(meta.get("mel_backend", "librosa_power_to_log")),
    }
