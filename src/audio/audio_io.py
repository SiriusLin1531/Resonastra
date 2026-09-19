from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioDecodeConfig:
    """
    EN:
    Configuration for decoding compressed audio bytes with ffmpeg.

    ZH:
    使用 ffmpeg 解码压缩音频 bytes 的配置。
    """

    target_sample_rate: int = 22050
    mono: bool = True
    ffmpeg_path: str = "ffmpeg"
    input_format: str | None = None
    output_dtype: str = "float32"


def find_ffmpeg(ffmpeg_path: str = "ffmpeg") -> str:
    """
    EN:
    Resolve ffmpeg executable.

    ZH:
    查找 ffmpeg 可执行文件。
    """
    resolved = shutil.which(ffmpeg_path)
    if resolved is None:
        raise FileNotFoundError(
            f"ffmpeg executable not found: {ffmpeg_path}. "
            "Please make sure ffmpeg is installed and available in PATH."
        )
    return resolved


def decode_audio_bytes_ffmpeg(
    audio_bytes: bytes,
    config: AudioDecodeConfig | None = None,
) -> tuple[torch.Tensor, int]:
    """
    EN:
    Decode compressed audio bytes into a mono waveform tensor using ffmpeg.

    Return:
        waveform: FloatTensor with shape (1, T), roughly in [-1, 1]
        sample_rate: target sample rate

    Notes:
        - This function intentionally avoids torchcodec and Hugging Face Audio decoding.
        - It is suitable for MP3 bytes extracted from Emilia WebDataset tar shards.

    ZH:
    使用 ffmpeg 将压缩音频 bytes 解码为单声道 waveform tensor。

    返回：
        waveform: FloatTensor，形状 (1, T)，数值大致在 [-1, 1]
        sample_rate: 目标采样率

    说明：
        - 该函数刻意绕开 torchcodec 和 Hugging Face Audio 自动解码。
        - 适合处理从 Emilia WebDataset tar 中取出的 MP3 bytes。
    """
    if config is None:
        config = AudioDecodeConfig()

    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise TypeError(f"audio_bytes must be bytes, got {type(audio_bytes)}")

    if len(audio_bytes) <= 0:
        raise ValueError("audio_bytes is empty.")

    ffmpeg = find_ffmpeg(config.ffmpeg_path)

    cmd: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if config.input_format:
        cmd += ["-f", str(config.input_format)]

    cmd += [
        "-i",
        "pipe:0",
    ]

    if config.mono:
        cmd += ["-ac", "1"]

    cmd += [
        "-ar",
        str(int(config.target_sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        input=bytes(audio_bytes),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            "ffmpeg failed to decode audio bytes.\n"
            f"returncode={proc.returncode}\n"
            f"stderr:\n{stderr}"
        )

    if not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            "ffmpeg produced empty audio output.\n"
            f"stderr:\n{stderr}"
        )

    wav_np = np.frombuffer(proc.stdout, dtype=np.float32).copy()

    if wav_np.size <= 0:
        raise RuntimeError("Decoded waveform is empty.")

    waveform = torch.from_numpy(wav_np).float().view(1, -1)

    # Defensive cleanup for rare invalid values.
    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Do not hard normalize here. Keep original loudness for mel inspection.
    waveform = waveform.clamp(-1.0, 1.0)

    return waveform, int(config.target_sample_rate)


def save_waveform(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
    peak_normalize: bool = False,
) -> None:
    """
    EN:
    Save waveform tensor as wav file.

    Args:
        path: output wav path
        waveform: Tensor with shape (T,), (1, T), or (C, T)
        sample_rate: sample rate
        peak_normalize: if true, peak normalize to 0.98 before saving

    ZH:
    将 waveform tensor 保存为 wav 文件。

    参数：
        path: 输出 wav 路径
        waveform: 形状可以是 (T,), (1, T), 或 (C, T)
        sample_rate: 采样率
        peak_normalize: 如果为 true，保存前峰值归一化到 0.98
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wav = waveform.detach().cpu().float()

    if wav.ndim == 2:
        # (C, T) -> (T, C)
        if wav.shape[0] <= wav.shape[1]:
            wav = wav.transpose(0, 1)
    elif wav.ndim == 1:
        pass
    else:
        wav = wav.view(-1)

    if peak_normalize:
        peak = wav.abs().max().clamp_min(1e-8)
        wav = wav / peak * 0.98

    arr = wav.numpy()
    sf.write(str(path), arr, int(sample_rate))


def waveform_duration_sec(
    waveform: torch.Tensor,
    sample_rate: int,
) -> float:
    """
    EN:
    Compute waveform duration in seconds.

    ZH:
    计算 waveform 时长，单位秒。
    """
    if waveform.ndim == 1:
        num_samples = waveform.numel()
    elif waveform.ndim == 2:
        num_samples = waveform.shape[-1]
    else:
        num_samples = waveform.view(-1).numel()

    return float(num_samples) / float(sample_rate)