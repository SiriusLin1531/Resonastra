from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from pathlib import Path
from typing import Iterable, Any


def assert_raw_input_exists(raw_input: str | Path) -> Path:
    """
    EN:
    Validate the raw input path and return its resolved Path object.

    Accepted input:
    - an existing directory containing raw audio files
    - or an existing single audio file

    ZH:
    检查原始输入路径是否合法，并返回其规范化后的 Path 对象。

    可接受输入：
    - 一个存在的原始音频目录
    - 或一条存在的单独音频文件
    """
    p = Path(raw_input).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Raw input path does not exist: {p}")

    if not p.is_dir() and not p.is_file():
        raise ValueError(f"Raw input path is neither a file nor a directory: {p}")

    return p


def assert_non_empty_audio_file_list(audio_files: Iterable[Any]) -> None:
    """
    EN:
    Ensure the scanned audio file list is not empty.

    ZH:
    确保扫描出来的音频文件列表非空。
    """
    audio_files = list(audio_files)
    if len(audio_files) == 0:
        raise ValueError("No audio files were found in the input path.")


def assert_slice_items_non_empty(slice_items: Iterable[Any]) -> None:
    """
    EN:
    Ensure slicing produced at least one clip.

    ZH:
    确保静音切分后至少产出一条短音频。
    """
    slice_items = list(slice_items)
    if len(slice_items) == 0:
        raise ValueError("Slicing produced zero clips. Please check slicing parameters or input audio quality.")


def assert_asr_output_exists(asr_output_path: str | Path) -> Path:
    """
    EN:
    Validate that the ASR output file exists and is non-empty.

    ZH:
    检查 ASR 输出文件存在且非空。
    """
    p = Path(asr_output_path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"ASR output file not found: {p}")

    if not p.is_file():
        raise ValueError(f"ASR output path is not a file: {p}")

    if p.stat().st_size == 0:
        raise ValueError(f"ASR output file is empty: {p}")

    return p


def assert_manifest_items_non_empty(items: Iterable[Any], name: str = "manifest items") -> None:
    """
    EN:
    Ensure exported manifest-like items are not empty.

    ZH:
    确保导出的 manifest 类数据项非空。
    """
    items = list(items)
    if len(items) == 0:
        raise ValueError(f"{name} is empty.")


def assert_directory_exists(dir_path: str | Path, name: str = "directory") -> Path:
    """
    EN:
    Validate that one directory exists.

    ZH:
    检查某个目录是否存在。
    """
    p = Path(dir_path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")

    if not p.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {p}")

    return p