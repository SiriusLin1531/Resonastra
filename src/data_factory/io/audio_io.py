from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import shutil
from pathlib import Path

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import ffmpeg
import librosa
import numpy as np
import soundfile as sf

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data_factory.utils.path_utils import clean_path


def load_audio_mono(file: str | Path, sr: int) -> np.ndarray:
    """
    EN:
    Load audio into mono float32 waveform using ffmpeg,
    aligned with the spirit of GPT-SoVITS tools.my_utils.load_audio().

    Returns:
        np.ndarray, shape=(T,), dtype=float32

    ZH:
    使用 ffmpeg 读取音频，并转为单声道 float32 waveform，
    设计思路对齐 GPT-SoVITS 的 tools.my_utils.load_audio()。

    返回：
        np.ndarray，形状为 (T,)，dtype=float32
    """
    file = clean_path(str(file))
    file_path = Path(file).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        out, _ = (
            ffmpeg.input(str(file_path), threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with ffmpeg: {file_path}") from e

    audio = np.frombuffer(out, np.float32).flatten()
    return audio


def get_audio_duration_sec(file: str | Path) -> float:
    """
    EN:
    Get audio duration in seconds.

    First try soundfile.info() for efficiency.
    If that fails (e.g. some compressed formats), fall back to librosa.

    ZH:
    获取音频时长（秒）。

    优先使用 soundfile.info()，速度更快；
    如果失败（例如部分压缩格式不支持），则退回到 librosa。
    """
    file_path = Path(clean_path(str(file))).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        info = sf.info(str(file_path))
        if info.samplerate > 0 and info.frames >= 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    try:
        audio, sr = librosa.load(str(file_path), sr=None, mono=True)
        if sr <= 0:
            raise ValueError(f"Invalid sample rate from librosa: {sr}")
        return float(len(audio)) / float(sr)
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {file_path}") from e


def copy_audio(src: str | Path, dst: str | Path, overwrite: bool = True) -> Path:
    """
    EN:
    Copy one audio file to target path.

    Parameters:
    - src: source audio file
    - dst: destination audio file
    - overwrite: whether to overwrite if destination exists

    Returns:
    - resolved Path of destination

    ZH:
    复制一条音频文件到目标路径。

    参数：
    - src: 源音频文件
    - dst: 目标音频文件
    - overwrite: 若目标存在，是否覆盖

    返回：
    - 目标文件的绝对路径 Path
    """
    src_path = Path(clean_path(str(src))).resolve()
    dst_path = Path(clean_path(str(dst))).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source audio not found: {src_path}")
    if not src_path.is_file():
        raise ValueError(f"Source path is not a file: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        if not overwrite:
            return dst_path
        if src_path.resolve() == dst_path.resolve():
            return dst_path

    shutil.copy2(src_path, dst_path)
    return dst_path.resolve()