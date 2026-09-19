from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from pathlib import Path
from typing import Iterable


# ============================================================
# Constants
# 常量定义
# ============================================================
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}


def clean_path(path_str: str) -> str:
    """
    EN:
    Normalize a path string copied from file explorer / terminal.

    This function is intentionally similar in spirit to the original
    GPT-SoVITS utility behavior:
    - strip trailing slashes
    - replace '/' and '\\' into current OS separator
    - strip surrounding spaces / quotes / invisible marks

    ZH:
    规范化用户从资源管理器或终端复制来的路径字符串。

    本函数和原 GPT-SoVITS 中的路径清理思路一致：
    - 去掉末尾多余斜杠
    - 将 '/' 和 '\\' 统一替换成当前操作系统分隔符
    - 去掉前后空格、引号、不可见符号
    """
    if path_str is None:
        raise ValueError("path_str cannot be None")

    path_str = str(path_str)

    # --------------------------------------------------------
    # Strip common trailing separators first
    # 先去掉末尾多余分隔符
    # --------------------------------------------------------
    while path_str.endswith(("\\", "/")):
        path_str = path_str[:-1]

    # --------------------------------------------------------
    # Normalize slash direction
    # 统一斜杠方向
    # --------------------------------------------------------
    path_str = path_str.replace("/", str(Path("/"))).replace("\\", str(Path("/")))

    # --------------------------------------------------------
    # Remove quotes / spaces / hidden marks
    # 去掉引号、空格、隐藏字符
    # --------------------------------------------------------
    path_str = path_str.strip(" '\n\"\u202a")

    # --------------------------------------------------------
    # Convert again with Path for OS-safe formatting
    # 再用 Path 做一次系统安全格式化
    # --------------------------------------------------------
    return str(Path(path_str))


def ensure_dir(path: str | Path) -> Path:
    """
    EN:
    Ensure directory exists and return Path object.

    ZH:
    确保目录存在，并返回 Path 对象。
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def scan_audio_files(root: str | Path, recursive: bool = True) -> list[Path]:
    """
    EN:
    Scan audio files under a directory.

    Parameters:
    - root: input directory
    - recursive: whether to scan recursively

    Returns:
    - sorted list of audio file paths

    ZH:
    扫描目录下的音频文件。

    参数：
    - root: 输入目录
    - recursive: 是否递归扫描子目录

    返回：
    - 按路径排序后的音频文件列表
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Audio root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Audio root is not a directory: {root_path}")

    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = root_path.rglob("*")
    else:
        pattern_iter = root_path.glob("*")

    audio_files = [
        p.resolve()
        for p in pattern_iter
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    audio_files = sorted(audio_files, key=lambda x: str(x).lower())
    return audio_files


def prepare_work_dirs(work_dir: str | Path) -> dict[str, Path]:
    """
    EN:
    Create and return the standard working directory structure for dataset factory v1.

    Current planned structure:
        work_dir/
        ├─ 00_raw_index/
        ├─ 01_uvr_vocal/
        ├─ 01_uvr_other/
        ├─ 02_denoise/
        ├─ 03_clips_raw/
        ├─ 04_asr/
        ├─ 05_label/
        ├─ 06_export/
        └─ logs/

    ZH:
    创建并返回数据工厂第一版标准工作目录结构。

    当前规划结构：
        work_dir/
        ├─ 00_raw_index/
        ├─ 01_uvr_vocal/
        ├─ 01_uvr_other/
        ├─ 02_denoise/
        ├─ 03_clips_raw/
        ├─ 04_asr/
        ├─ 05_label/
        ├─ 06_export/
        └─ logs/
    """
    root = ensure_dir(work_dir)

    dirs = {
        "root": root,
        "raw_index": ensure_dir(root / "00_raw_index"),
        "uvr_vocal": ensure_dir(root / "01_uvr_vocal"),
        "uvr_other": ensure_dir(root / "01_uvr_other"),
        "denoise": ensure_dir(root / "02_denoise"),
        "clips_raw": ensure_dir(root / "03_clips_raw"),
        "asr": ensure_dir(root / "04_asr"),
        "label": ensure_dir(root / "05_label"),
        "export": ensure_dir(root / "06_export"),
        "logs": ensure_dir(root / "logs"),
    }

    return dirs