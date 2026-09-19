from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch

# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.features.fewshot_style_features import (
    align_1d_feature_to_length,
    align_bool_mask_to_length,
    ensure_finite_tensor,
    extract_f0_pyin,
    extract_log_mel_from_path,
    infer_acoustic_meta_from_sample,
    load_audio_mono,
    log_mel_to_log_energy,
    tensor_stats,
)
from src.features.speaker_embedding_features import (
    SpeakerEmbeddingConfig,
    SpeakerEmbeddingExtractor,
)


STYLE_CACHE_VERSION = "v6.6.0-A"


# ============================================================
# CLI helpers
# CLI 工具
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
            "Stage2 v6.6.0-A few-shot style cache exporter. "
            "It augments existing Stage2 .pt samples with prompt_acoustic and style features."
        )
    )

    # --------------------------------------------------------
    # Paths
    # 路径参数
    # --------------------------------------------------------
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--report_json", type=str, default=None)

    # --------------------------------------------------------
    # Runtime / file control
    # 运行与文件控制
    # --------------------------------------------------------
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--copy_non_pt_files", type=str2bool, default=False)
    parser.add_argument("--recursive", type=str2bool, default=False)
    parser.add_argument("--max_files", type=int, default=None)

    # --------------------------------------------------------
    # Feature switches
    # 特征开关
    # --------------------------------------------------------
    parser.add_argument("--extract_prompt_acoustic", type=str2bool, default=True)
    parser.add_argument("--extract_energy", type=str2bool, default=True)
    parser.add_argument("--extract_f0", type=str2bool, default=False)
    parser.add_argument(
        "--extract_speaker_embedding",
        type=str2bool,
        default=False,
        help=(
            "Reserved for later v6.6 stages. Current v6.6.0-A exporter does not "
            "implement speaker embedding extraction yet."
        ),
    )

    # --------------------------------------------------------
    # Acoustic extraction defaults
    # 声学特征默认参数
    # --------------------------------------------------------
    parser.add_argument(
        "--prefer_sample_acoustic_meta",
        type=str2bool,
        default=True,
        help=(
            "If true, use acoustic_meta from each .pt sample for prompt_acoustic. "
            "Fallback to CLI acoustic args when acoustic_meta is missing."
        ),
    )
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=8000.0)

    # --------------------------------------------------------
    # F0 extraction
    # F0 提取参数
    # --------------------------------------------------------
    parser.add_argument("--f0_min_hz", type=float, default=50.0)
    parser.add_argument("--f0_max_hz", type=float, default=1100.0)

    # --------------------------------------------------------
    # Speaker embedding extraction
    # 说话人嵌入提取参数
    # --------------------------------------------------------
    parser.add_argument("--speaker_backend", type=str, default="speechbrain_ecapa")
    parser.add_argument("--speaker_model_source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument(
        "--speaker_savedir",
        type=str,
        default="pretrained_models/speechbrain_spkrec_ecapa_voxceleb",
    )
    parser.add_argument("--speaker_device", type=str, default=None)
    parser.add_argument("--speaker_sample_rate", type=int, default=16000)
    parser.add_argument("--speaker_normalize", type=str2bool, default=True)

    # --------------------------------------------------------
    # Validation policy
    # 校验策略
    # --------------------------------------------------------
    parser.add_argument(
        "--fail_on_error",
        type=str2bool,
        default=True,
        help="Raise RuntimeError if any sample fails.",
    )

    return parser.parse_args()


# ============================================================
# IO helpers
# IO 工具
# ============================================================
def save_json(obj: dict[str, Any], path: str | Path) -> None:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_pt_files(
    root: Path,
    *,
    recursive: bool,
    max_files: int | None,
) -> list[Path]:
    pattern = "**/*.pt" if recursive else "*.pt"
    files = sorted(root.glob(pattern))

    if max_files is not None:
        files = files[: int(max_files)]

    return files


def copy_non_pt_files(input_root: Path, output_root: Path) -> int:
    count = 0

    for src in sorted(input_root.iterdir()):
        if src.is_file() and src.suffix.lower() != ".pt":
            dst = output_root / src.name
            shutil.copy2(src, dst)
            count += 1

    return count


def resolve_existing_path(path_value: Any, *, field_name: str) -> Path:
    path = Path(str(path_value)).expanduser()

    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{field_name} not found: {path}")

    return path


# ============================================================
# Acoustic settings
# 声学参数
# ============================================================
def acoustic_settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_sr": int(args.sample_rate),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
        "win_length": int(args.win_length),
        "n_mels": int(args.n_mels),
        "fmin": float(args.fmin),
        "fmax": float(args.fmax),
        "acoustic_rate_hz": float(args.sample_rate) / float(args.hop_length),
        "mel_backend": "librosa_power_to_log",
    }


def choose_acoustic_settings(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.prefer_sample_acoustic_meta):
        meta = infer_acoustic_meta_from_sample(sample)
    else:
        meta = acoustic_settings_from_args(args)

    # If sample meta is incomplete or malformed, infer_acoustic_meta_from_sample already fills defaults.
    return meta


# ============================================================
# Sample augmentation
# 样本增强
# ============================================================
def augment_one_sample(
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
    speaker_extractor: SpeakerEmbeddingExtractor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    EN:
    Add v6.6 style-cache fields to one existing Stage2 .pt sample.

    ZH:
    给一个已有 Stage2 .pt 样本追加 v6.6 style-cache 字段。
    """
    out = dict(sample)
    sample_id = str(out.get("sample_id", ""))

    target_acoustic = out.get("target_acoustic")
    if not torch.is_tensor(target_acoustic):
        raise TypeError("sample is missing tensor field: target_acoustic")

    target_acoustic = target_acoustic.detach().cpu().float()

    if target_acoustic.ndim != 2:
        raise ValueError(f"target_acoustic must be [T, M], got {tuple(target_acoustic.shape)}")

    if target_acoustic.shape[0] <= 0 or target_acoustic.shape[1] <= 0:
        raise ValueError(f"target_acoustic has invalid shape={tuple(target_acoustic.shape)}")

    ensure_finite_tensor(target_acoustic, name="target_acoustic")

    acoustic_meta = choose_acoustic_settings(out, args)
    prompt_wav_path = resolve_existing_path(out.get("prompt_wav_path"), field_name="prompt_wav_path")

    prompt_acoustic = None
    prompt_energy = None

    if bool(args.extract_prompt_acoustic):
        prompt_acoustic = extract_log_mel_from_path(
            prompt_wav_path,
            sample_rate=int(acoustic_meta["target_sr"]),
            n_fft=int(acoustic_meta["n_fft"]),
            hop_length=int(acoustic_meta["hop_length"]),
            win_length=int(acoustic_meta["win_length"]),
            n_mels=int(acoustic_meta["n_mels"]),
            f_min=float(acoustic_meta["fmin"]),
            f_max=float(acoustic_meta["fmax"]),
        )

        out["prompt_acoustic"] = prompt_acoustic.detach().cpu().float()
        out["prompt_acoustic_lengths"] = torch.tensor(
            [int(prompt_acoustic.shape[0])],
            dtype=torch.long,
        )

    if bool(args.extract_energy):
        target_energy = log_mel_to_log_energy(target_acoustic)
        target_energy = align_1d_feature_to_length(
            target_energy,
            target_len=int(target_acoustic.shape[0]),
        )

        out["target_energy"] = target_energy.detach().cpu().float()

        if prompt_acoustic is None:
            existing_prompt_acoustic = out.get("prompt_acoustic")
            if torch.is_tensor(existing_prompt_acoustic):
                prompt_acoustic = existing_prompt_acoustic.detach().cpu().float()

        if prompt_acoustic is not None:
            prompt_energy = log_mel_to_log_energy(prompt_acoustic)
            out["prompt_energy"] = prompt_energy.detach().cpu().float()

    if bool(args.extract_f0):
        target_wav_path = resolve_existing_path(out.get("target_wav_path"), field_name="target_wav_path")

        target_wav = load_audio_mono(
            target_wav_path,
            sample_rate=int(acoustic_meta["target_sr"]),
        )
        target_f0, target_voiced = extract_f0_pyin(
            target_wav,
            sample_rate=int(acoustic_meta["target_sr"]),
            hop_length=int(acoustic_meta["hop_length"]),
            frame_length=int(acoustic_meta["win_length"]),
            f0_min_hz=float(args.f0_min_hz),
            f0_max_hz=float(args.f0_max_hz),
        )
        target_f0 = align_1d_feature_to_length(
            target_f0,
            target_len=int(target_acoustic.shape[0]),
        )
        target_voiced = align_bool_mask_to_length(
            target_voiced,
            target_len=int(target_acoustic.shape[0]),
        )

        out["target_f0"] = target_f0.detach().cpu().float()
        out["target_f0_voiced_mask"] = target_voiced.detach().cpu().bool()

        if prompt_acoustic is not None:
            prompt_wav = load_audio_mono(
                prompt_wav_path,
                sample_rate=int(acoustic_meta["target_sr"]),
            )
            prompt_f0, prompt_voiced = extract_f0_pyin(
                prompt_wav,
                sample_rate=int(acoustic_meta["target_sr"]),
                hop_length=int(acoustic_meta["hop_length"]),
                frame_length=int(acoustic_meta["win_length"]),
                f0_min_hz=float(args.f0_min_hz),
                f0_max_hz=float(args.f0_max_hz),
            )
            prompt_f0 = align_1d_feature_to_length(
                prompt_f0,
                target_len=int(prompt_acoustic.shape[0]),
            )
            prompt_voiced = align_bool_mask_to_length(
                prompt_voiced,
                target_len=int(prompt_acoustic.shape[0]),
            )

            out["prompt_f0"] = prompt_f0.detach().cpu().float()
            out["prompt_f0_voiced_mask"] = prompt_voiced.detach().cpu().bool()


    if bool(args.extract_speaker_embedding):
        if speaker_extractor is None:
            raise RuntimeError("extract_speaker_embedding=True but speaker_extractor is None.")

        target_wav_path = resolve_existing_path(out.get("target_wav_path"), field_name="target_wav_path")
        target_speaker_embedding = speaker_extractor.extract_from_wav(target_wav_path)
        prompt_speaker_embedding = speaker_extractor.extract_from_wav(prompt_wav_path)

        out["target_speaker_embedding"] = target_speaker_embedding.detach().cpu().float()
        out["prompt_speaker_embedding"] = prompt_speaker_embedding.detach().cpu().float()

    # --------------------------------------------------------
    # Validate newly-created tensors
    # 校验新增 tensor
    # --------------------------------------------------------
    for key in [
        "prompt_acoustic",
        "target_energy",
        "prompt_energy",
        "target_f0",
        "prompt_f0",
        "target_speaker_embedding",
        "prompt_speaker_embedding",
    ]:
        value = out.get(key)
        if torch.is_tensor(value):
            ensure_finite_tensor(value.float(), name=key)

    out["v66_style_cache_meta"] = {
        "style_cache_version": STYLE_CACHE_VERSION,
        "sample_id": sample_id,
        "acoustic_meta": acoustic_meta,
        "extract_prompt_acoustic": bool(args.extract_prompt_acoustic),
        "extract_energy": bool(args.extract_energy),
        "extract_f0": bool(args.extract_f0),
        "extract_speaker_embedding": bool(args.extract_speaker_embedding),
        "fields_added": [
            key
            for key in [
                "prompt_acoustic",
                "prompt_acoustic_lengths",
                "target_energy",
                "prompt_energy",
                "target_f0",
                "target_f0_voiced_mask",
                "prompt_f0",
                "prompt_f0_voiced_mask",
                "target_speaker_embedding",
                "prompt_speaker_embedding",
            ]
            if key in out
        ],
    }

    sample_report = {
        "sample_id": sample_id,
        "prompt_wav_path": str(prompt_wav_path),
        "target_acoustic": tensor_stats(target_acoustic),
        "prompt_acoustic": tensor_stats(out["prompt_acoustic"]) if "prompt_acoustic" in out else None,
        "target_energy": tensor_stats(out["target_energy"]) if "target_energy" in out else None,
        "prompt_energy": tensor_stats(out["prompt_energy"]) if "prompt_energy" in out else None,
    }

    if "target_f0" in out:
        sample_report["target_f0"] = tensor_stats(out["target_f0"])
        sample_report["target_f0_voiced_ratio"] = float(out["target_f0_voiced_mask"].float().mean().item())

    if "prompt_f0" in out:
        sample_report["prompt_f0"] = tensor_stats(out["prompt_f0"])
        sample_report["prompt_f0_voiced_ratio"] = float(out["prompt_f0_voiced_mask"].float().mean().item())

    if "target_speaker_embedding" in out:
        sample_report["target_speaker_embedding"] = tensor_stats(out["target_speaker_embedding"])
    if "prompt_speaker_embedding" in out:
        sample_report["prompt_speaker_embedding"] = tensor_stats(out["prompt_speaker_embedding"])

    return out, sample_report


# ============================================================
# Main
# 主流程
# ============================================================
def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input_root not found or not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    pt_files = list_pt_files(
        input_root,
        recursive=bool(args.recursive),
        max_files=args.max_files,
    )

    if len(pt_files) <= 0:
        raise FileNotFoundError(f"No .pt files found under: {input_root}")

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else output_root / "style_cache_report.json"
    )

    print("==== Stage2 v6.6.0-A few-shot style cache export started ====")
    print(f"input_root   : {input_root}")
    print(f"output_root  : {output_root}")
    print(f"num_pt       : {len(pt_files)}")
    print(f"device       : {args.device}")
    print(f"report_json  : {report_json}")
    print("---------------------------------------------------------------")
    print(f"extract_prompt_acoustic   : {bool(args.extract_prompt_acoustic)}")
    print(f"extract_energy            : {bool(args.extract_energy)}")
    print(f"extract_f0                : {bool(args.extract_f0)}")
    print(f"extract_speaker_embedding : {bool(args.extract_speaker_embedding)}")
    print("===============================================================")

    speaker_extractor = None
    if bool(args.extract_speaker_embedding):
        speaker_device = str(args.speaker_device or args.device)
        speaker_extractor = SpeakerEmbeddingExtractor(
            SpeakerEmbeddingConfig(
                backend=str(args.speaker_backend),
                model_source=str(args.speaker_model_source),
                savedir=str(args.speaker_savedir),
                device=speaker_device,
                sample_rate=int(args.speaker_sample_rate),
                normalize=bool(args.speaker_normalize),
            )
        )
        print(f"speaker_backend          : {args.speaker_backend}")
        print(f"speaker_device           : {speaker_device}")

    copied_non_pt = 0
    if bool(args.copy_non_pt_files) and not bool(args.recursive):
        copied_non_pt = copy_non_pt_files(input_root, output_root)

    ok_count = 0
    skipped_count = 0
    error_count = 0
    sample_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    t0 = time.perf_counter()

    for idx, src_path in enumerate(pt_files, start=1):
        rel_path = src_path.relative_to(input_root)
        dst_path = output_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not bool(args.overwrite):
            print(f"[{idx}/{len(pt_files)}] skip existing -> {rel_path}")
            skipped_count += 1
            continue

        try:
            sample = torch.load(str(src_path), map_location="cpu", weights_only=False)

            if not isinstance(sample, dict):
                raise TypeError(f"Expected dict sample, got {type(sample)}")

            augmented, sample_report = augment_one_sample(
                sample,
                args=args,
                speaker_extractor=speaker_extractor,
            )
            torch.save(augmented, str(dst_path))

            print(f"[{idx}/{len(pt_files)}] saved -> {rel_path}")
            ok_count += 1
            sample_reports.append({
                "input_path": str(src_path),
                "output_path": str(dst_path),
                "status": "OK",
                **sample_report,
            })

        except Exception as e:
            print(f"[{idx}/{len(pt_files)}] FAILED -> {rel_path}: {e}")
            error_count += 1
            errors.append({
                "input_path": str(src_path),
                "output_path": str(dst_path),
                "status": "ERROR",
                "error_type": type(e).__name__,
                "error": str(e),
            })

    elapsed_sec = time.perf_counter() - t0

    summary = {
        "style_cache_version": STYLE_CACHE_VERSION,
        "status": "OK" if error_count == 0 else "ERROR",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "report_json": str(report_json),
        "num_input_pt": int(len(pt_files)),
        "ok_count": int(ok_count),
        "skipped_count": int(skipped_count),
        "error_count": int(error_count),
        "copied_non_pt": int(copied_non_pt),
        "elapsed_sec": float(elapsed_sec),
        "config": {
            "device": str(args.device),
            "overwrite": bool(args.overwrite),
            "copy_non_pt_files": bool(args.copy_non_pt_files),
            "recursive": bool(args.recursive),
            "max_files": args.max_files,
            "extract_prompt_acoustic": bool(args.extract_prompt_acoustic),
            "extract_energy": bool(args.extract_energy),
            "extract_f0": bool(args.extract_f0),
            "extract_speaker_embedding": bool(args.extract_speaker_embedding),
            "prefer_sample_acoustic_meta": bool(args.prefer_sample_acoustic_meta),
            "sample_rate": int(args.sample_rate),
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "win_length": int(args.win_length),
            "n_mels": int(args.n_mels),
            "fmin": float(args.fmin),
            "fmax": float(args.fmax),
            "f0_min_hz": float(args.f0_min_hz),
            "f0_max_hz": float(args.f0_max_hz),
            "speaker_backend": str(args.speaker_backend),
            "speaker_model_source": str(args.speaker_model_source),
            "speaker_savedir": str(args.speaker_savedir),
            "speaker_device": str(args.speaker_device or args.device),
            "speaker_sample_rate": int(args.speaker_sample_rate),
            "speaker_normalize": bool(args.speaker_normalize),
        },
        "samples_preview": sample_reports[:20],
        "errors": errors[:50],
    }

    save_json(summary, report_json)

    print("===============================================================")
    print("v6.6.0-A style cache export finished")
    print(f"status        : {summary['status']}")
    print(f"ok_count      : {ok_count}")
    print(f"skipped_count : {skipped_count}")
    print(f"error_count   : {error_count}")
    print(f"elapsed_sec   : {elapsed_sec:.3f}")
    print(f"report_json   : {report_json}")
    print("===============================================================")

    if error_count > 0 and bool(args.fail_on_error):
        raise RuntimeError(
            f"style cache export finished with errors: error_count={error_count}. "
            f"See report: {report_json}"
        )


if __name__ == "__main__":
    main()
