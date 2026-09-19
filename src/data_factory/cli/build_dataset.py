from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data_factory.asr.asr_router import run_asr
from src.data_factory.export.export_dataset_manifest import (
    build_export_items_from_asr_list,
    export_dataset_manifest_jsonl,
)
from src.data_factory.export.export_stage2_manifest import (
    export_stage2_manifest_jsonl,
)
from src.data_factory.io.audio_io import copy_audio, get_audio_duration_sec
from src.data_factory.io.manifest_io import write_jsonl
from src.data_factory.schemas import ExportItem
from src.data_factory.slice.slicer_runner import run_slice
from src.data_factory.utils.path_utils import (
    clean_path,
    ensure_dir,
    prepare_work_dirs,
    scan_audio_files,
)
from src.data_factory.utils.validation import (
    assert_non_empty_audio_file_list,
    assert_raw_input_exists,
)

# ------------------------------------------------------------
# Optional runners
# 可选处理模块：第一版中先保留接口占位
# ------------------------------------------------------------
# from src.data_factory.preprocess.uvr_runner importers run_uvr
# from src.data_factory.preprocess.denoise_runner importers run_denoise


def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse CLI arguments for dataset factory v1.

    ZH:
    解析数据工厂第一版命令行参数。
    """
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Input / output
    # 输入输出
    # --------------------------------------------------------
    parser.add_argument("--raw_input_dir", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)

    # --------------------------------------------------------
    # Dataset identity
    # 数据集标识
    # --------------------------------------------------------
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--language", type=str, default="zh")

    # --------------------------------------------------------
    # Optional preprocess flags
    # 可选预处理开关
    # --------------------------------------------------------
    parser.add_argument("--enable_uvr", type=str, default="false")
    parser.add_argument("--enable_denoise", type=str, default="false")

    # --------------------------------------------------------
    # Slice parameters
    # 切片参数
    # --------------------------------------------------------
    parser.add_argument("--threshold", type=int, default=-34)
    parser.add_argument("--min_length", type=int, default=4000)
    parser.add_argument("--min_interval", type=int, default=300)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--max_sil_kept", type=int, default=500)
    parser.add_argument("--normalize_max", type=float, default=0.9)
    parser.add_argument("--alpha_mix", type=float, default=0.25)

    # --------------------------------------------------------
    # ASR parameters
    # ASR 参数
    # --------------------------------------------------------
    parser.add_argument("--asr_backend", type=str, default="auto")
    parser.add_argument("--asr_model_size", type=str, default="large-v3")
    parser.add_argument("--asr_precision", type=str, default="float32")

    # --------------------------------------------------------
    # Export options
    # 导出选项
    # --------------------------------------------------------
    parser.add_argument("--export_list", type=str, default="true")
    parser.add_argument("--export_stage2_manifest", type=str, default="true")

    return parser.parse_args()


def str2bool(x: str) -> bool:
    """
    EN:
    Convert common string values to bool.

    ZH:
    将常见字符串值转成布尔值。
    """
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def save_run_config(args: argparse.Namespace, path: Path) -> None:
    """
    EN:
    Save CLI config for reproducibility.

    ZH:
    保存本次运行配置，便于复现。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def _collect_raw_audio_files(raw_input_path: Path) -> list[Path]:
    """
    EN:
    Collect raw audio files from:
    - a directory
    - or a single file

    ZH:
    从以下输入中收集原始音频：
    - 一个目录
    - 或一条单独文件
    """
    if raw_input_path.is_file():
        return [raw_input_path.resolve()]

    if raw_input_path.is_dir():
        files = scan_audio_files(raw_input_path, recursive=True)
        return files

    raise ValueError(f"Unsupported raw input path: {raw_input_path}")


def _prepare_slice_input_dir(raw_input_path: Path, dirs: dict[str, Path]) -> Path:
    """
    EN:
    Prepare a directory that slicer_runner can consume.

    Current slicer_runner expects a directory input.
    So:
    - if raw_input_path is already a directory -> return it directly
    - if it is a single file -> copy it into a temporary staging directory

    ZH:
    为 slicer_runner 准备一个可用的输入目录。

    当前 slicer_runner 期望输入是目录，因此：
    - 如果 raw_input_path 本身就是目录 -> 直接返回
    - 如果它是一条单文件 -> 先复制到一个临时中转目录
    """
    if raw_input_path.is_dir():
        return raw_input_path.resolve()

    single_input_dir = ensure_dir(dirs["raw_index"] / "single_input_audio")
    staged_file = single_input_dir / raw_input_path.name
    copy_audio(raw_input_path, staged_file, overwrite=True)
    return single_input_dir.resolve()


def _save_raw_audio_index(
    raw_audio_files: list[Path],
    speaker_name: str,
    language: str,
    output_path: Path,
) -> Path:
    """
    EN:
    Save raw audio index into JSONL for traceability.

    ZH:
    将原始音频索引保存为 JSONL，便于追踪。
    """
    records: list[dict[str, Any]] = []

    for p in raw_audio_files:
        duration_sec = get_audio_duration_sec(p)
        records.append(
            {
                "source_audio": str(p.resolve()),
                "speaker_name": speaker_name,
                "language": language,
                "duration_sec": duration_sec,
            }
        )

    return write_jsonl(records, output_path)


def _relocate_export_clips(
    items: list[ExportItem],
    export_clips_dir: str | Path,
) -> list[ExportItem]:
    """
    EN:
    Copy final clip wav files into export/clips and rewrite wav_path accordingly.

    Why do this:
    - keep 03_clips_raw as intermediate artifacts
    - keep 06_export/clips as the final packaged dataset assets

    ZH:
    将最终 clip 音频复制到 export/clips，并重写 ExportItem 中的 wav_path。

    这样做的原因：
    - 03_clips_raw 保留中间产物
    - 06_export/clips 成为最终打包好的数据集音频目录
    """
    export_root = ensure_dir(export_clips_dir)
    relocated_items: list[ExportItem] = []

    for item in items:
        src_path = Path(item.wav_path).resolve()
        ext = src_path.suffix if src_path.suffix else ".wav"
        dst_path = export_root / f"{item.sample_id}{ext}"

        copy_audio(src_path, dst_path, overwrite=True)

        relocated_items.append(
            replace(item, wav_path=str(dst_path.resolve()))
        )

    return relocated_items


def main() -> None:
    args = parse_args()

    # --------------------------------------------------------
    # Normalize and validate raw input
    # 规范化并校验原始输入
    # --------------------------------------------------------
    raw_input_str = clean_path(args.raw_input_dir)
    work_dir_str = clean_path(args.work_dir)

    raw_input_path = assert_raw_input_exists(raw_input_str)
    raw_audio_files = _collect_raw_audio_files(raw_input_path)
    assert_non_empty_audio_file_list(raw_audio_files)

    # --------------------------------------------------------
    # Prepare working directories
    # 准备工作目录
    # --------------------------------------------------------
    dirs = prepare_work_dirs(work_dir_str)
    save_run_config(args, dirs["logs"] / "run_config.json")

    # --------------------------------------------------------
    # Save raw audio index
    # 保存原始音频索引
    # --------------------------------------------------------
    raw_index_path = dirs["raw_index"] / "raw_audio_index.jsonl"
    _save_raw_audio_index(
        raw_audio_files=raw_audio_files,
        speaker_name=args.speaker_name,
        language=args.language,
        output_path=raw_index_path,
    )

    print("====================================================")
    print("Dataset Factory v1 started")
    print(f"raw_input      : {raw_input_path}")
    print(f"work_dir       : {work_dir_str}")
    print(f"speaker_name   : {args.speaker_name}")
    print(f"language       : {args.language}")
    print(f"num_raw_files  : {len(raw_audio_files)}")
    print(f"raw_index_path : {raw_index_path}")
    print("====================================================")

    # --------------------------------------------------------
    # Prepare slicer input dir
    # 准备切片输入目录
    # --------------------------------------------------------
    current_audio_dir = _prepare_slice_input_dir(raw_input_path, dirs)

    # --------------------------------------------------------
    # Step 1: Optional UVR
    # 第一步：可选 UVR
    # --------------------------------------------------------
    if str2bool(args.enable_uvr):
        print("[Step 1] UVR enabled")
        # TODO:
        # current_audio_dir = run_uvr(
        #     input_dir=str(current_audio_dir),
        #     vocal_dir=str(dirs["uvr_vocal"]),
        #     other_dir=str(dirs["uvr_other"]),
        #     model_name="your_default_model",
        #     format="wav",
        # )
        raise NotImplementedError("UVR runner is not implemented yet.")
    else:
        print("[Step 1] UVR skipped")

    # --------------------------------------------------------
    # Step 2: Optional denoise
    # 第二步：可选降噪
    # --------------------------------------------------------
    if str2bool(args.enable_denoise):
        print("[Step 2] Denoise enabled")
        # TODO:
        # current_audio_dir = run_denoise(
        #     input_dir=str(current_audio_dir),
        #     output_dir=str(dirs["denoise"]),
        # )
        raise NotImplementedError("Denoise runner is not implemented yet.")
    else:
        print("[Step 2] Denoise skipped")

    # --------------------------------------------------------
    # Step 3: Slice long audio into short clips
    # 第三步：长音频静音切分
    # --------------------------------------------------------
    print("[Step 3] Running slicing...")
    slice_items = run_slice(
        input_dir=str(current_audio_dir),
        output_dir=str(dirs["clips_raw"]),
        speaker_name=args.speaker_name,
        language=args.language,
        threshold=args.threshold,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
        normalize_max=args.normalize_max,
        alpha_mix=args.alpha_mix,
    )
    print(f"[Step 3] Slicing finished. num_clips={len(slice_items)}")

    # --------------------------------------------------------
    # Step 4: ASR over clips
    # 第四步：对切片音频做 ASR
    # --------------------------------------------------------
    print("[Step 4] Running ASR...")
    asr_list_path = run_asr(
        input_dir=str(dirs["clips_raw"]),
        output_dir=str(dirs["asr"]),
        backend=args.asr_backend,
        language=args.language,
        model_size=args.asr_model_size,
        precision=args.asr_precision,
    )
    print(f"[Step 4] ASR finished. asr_list_path={asr_list_path}")

    # --------------------------------------------------------
    # Step 5: Build export items from ASR output
    # 第五步：根据 ASR 输出构建导出项
    # --------------------------------------------------------
    print("[Step 5] Building export items...")
    export_items = build_export_items_from_asr_list(
        asr_list_path=asr_list_path,
        slice_items=slice_items,
        speaker_name=args.speaker_name,
        language=args.language,
    )
    print(f"[Step 5] Built export items. num_items={len(export_items)}")

    # --------------------------------------------------------
    # Step 6: Relocate final clips into export/clips
    # 第六步：将最终 clip 复制到 export/clips
    # --------------------------------------------------------
    print("[Step 6] Relocating clips to export/clips...")
    export_clips_dir = ensure_dir(dirs["export"] / "clips")
    export_items = _relocate_export_clips(
        items=export_items,
        export_clips_dir=export_clips_dir,
    )
    print(f"[Step 6] Relocation finished. export_clips_dir={export_clips_dir}")

    # --------------------------------------------------------
    # Step 7: Export standard dataset manifest
    # 第七步：导出标准 manifest.jsonl
    # --------------------------------------------------------
    print("[Step 7] Exporting manifest.jsonl...")
    dataset_manifest_path = dirs["export"] / "manifest.jsonl"
    export_dataset_manifest_jsonl(
        items=export_items,
        output_path=dataset_manifest_path,
        export_format="jsonl",
    )
    print(f"[Step 7] Saved -> {dataset_manifest_path}")

    # --------------------------------------------------------
    # Step 8: Optional export GPT-SoVITS .list
    # 第八步：可选导出 GPT-SoVITS 兼容 .list
    # --------------------------------------------------------
    if str2bool(args.export_list):
        print("[Step 8] Exporting dataset.list...")
        dataset_list_path = dirs["export"] / "dataset.list"
        export_dataset_manifest_jsonl(
            items=export_items,
            output_path=dataset_list_path,
            export_format="list",
        )
        print(f"[Step 8] Saved -> {dataset_list_path}")
    else:
        print("[Step 8] Export .list skipped")

    # --------------------------------------------------------
    # Step 9: Optional export stage2 manifest
    # 第九步：可选导出 stage2_manifest.jsonl
    # --------------------------------------------------------
    if str2bool(args.export_stage2_manifest):
        print("[Step 9] Exporting stage2_manifest.jsonl...")
        stage2_manifest_path = dirs["export"] / "stage2_manifest.jsonl"
        export_stage2_manifest_jsonl(
            items=export_items,
            output_path=stage2_manifest_path,
            prompt_mode="self",
        )
        print(f"[Step 9] Saved -> {stage2_manifest_path}")
    else:
        print("[Step 9] Export stage2 manifest skipped")

    print("====================================================")
    print("[DONE] Dataset Factory v1 finished successfully.")
    print(f"work_dir = {Path(work_dir_str).resolve()}")
    print("====================================================")


if __name__ == "__main__":
    main()