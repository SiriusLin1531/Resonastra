from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from pathlib import Path

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import numpy as np
import soundfile as sf

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data_factory.schemas import SliceItem
from src.data_factory.utils.path_utils import ensure_dir, scan_audio_files
from src.data_factory.utils.validation import (
    assert_directory_exists,
    assert_non_empty_audio_file_list,
    assert_slice_items_non_empty,
)

# ============================================================
# GPT-SoVITS reference imports
# 参考原 GPT-SoVITS 的切分实现
# ============================================================
# EN:
# We intentionally align with the original project here:
# - load_audio() from tools.my_utils
# - Slicer from tools.slicer2
#
# ZH:
# 这里有意对齐原项目：
# - 使用 tools.my_utils.load_audio
# - 使用 tools.slicer2.Slicer
try:
    from tools.my_utils import load_audio
    from tools.slicer2 import Slicer
except ImportError as e:
    raise ImportError(
        "Failed to importers GPT-SoVITS slicing dependencies: tools.my_utils / tools.slicer2. "
        "Please make sure your original GPT-SoVITS project files are available in the current project."
    ) from e


def _safe_normalize_chunk(
    chunk: np.ndarray,
    normalize_max: float,
    alpha_mix: float,
) -> np.ndarray:
    """
    EN:
    Apply normalization in a way similar to the original slice_audio.py logic.

    Original logic roughly does:
        tmp_max = abs(chunk).max()
        if tmp_max > 1:
            chunk /= tmp_max
        chunk = (chunk / tmp_max * (normalize_max * alpha_mix)) + (1 - alpha_mix) * chunk

    Here we add one safety guard for zero-energy segments.

    ZH:
    以接近原 slice_audio.py 的方式对切片做归一化。

    原始逻辑大致是：
        tmp_max = abs(chunk).max()
        if tmp_max > 1:
            chunk /= tmp_max
        chunk = (chunk / tmp_max * (normalize_max * alpha_mix)) + (1 - alpha_mix) * chunk

    这里额外加了一个零能量片段保护，避免除零。
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    tmp_max = float(np.max(np.abs(chunk))) if chunk.size > 0 else 0.0

    if tmp_max <= 1e-8:
        return chunk

    if tmp_max > 1.0:
        chunk = chunk / tmp_max

    # Mix original chunk with normalized chunk, aligned to original behavior
    normalized = chunk / max(float(np.max(np.abs(chunk))), 1e-8)
    chunk = (normalized * (normalize_max * alpha_mix)) + ((1.0 - alpha_mix) * chunk)

    # Final clip to safe range
    chunk = np.clip(chunk, -1.0, 1.0)
    return chunk.astype(np.float32)


def run_slice(
    input_dir: str,
    output_dir: str,
    speaker_name: str,
    language: str,
    threshold: int = -34,
    min_length: int = 4000,
    min_interval: int = 300,
    hop_size: int = 10,
    max_sil_kept: int = 500,
    normalize_max: float = 0.9,
    alpha_mix: float = 0.25,
    sample_rate: int = 32000,
) -> list[SliceItem]:
    """
    EN:
    Slice raw long audio files into short clips using the GPT-SoVITS-style silence slicer.

    Parameters are intentionally aligned with the original project:
    - sample_rate = 32000
    - threshold / min_length / min_interval / hop_size / max_sil_kept
      follow the original slicing pipeline.

    Output:
    - save clipped wav files into output_dir
    - return a list of SliceItem

    ZH:
    使用接近 GPT-SoVITS 原项目的静音切片逻辑，将长音频切成短音频。

    参数设计刻意对齐原项目：
    - sample_rate = 32000
    - threshold / min_length / min_interval / hop_size / max_sil_kept
      与原切片流程一致

    输出：
    - 将切好的 wav 保存到 output_dir
    - 返回 SliceItem 列表
    """
    input_path = assert_directory_exists(input_dir, name="input_dir")
    output_root = ensure_dir(output_dir)

    audio_files = scan_audio_files(input_path, recursive=True)
    assert_non_empty_audio_file_list(audio_files)

    slicer = Slicer(
        sr=sample_rate,
        threshold=int(threshold),
        min_length=int(min_length),
        min_interval=int(min_interval),
        hop_size=int(hop_size),
        max_sil_kept=int(max_sil_kept),
    )

    slice_items: list[SliceItem] = []
    running_index = 1

    for src_audio in audio_files:
        audio = load_audio(str(src_audio), sample_rate)

        # EN:
        # Original slicer returns iterable items shaped like:
        #   [chunk, start, end]
        # where start/end are positions aligned to the slicer hop-based timeline
        # and in the original slice_audio.py they are directly written into filenames.
        #
        # ZH:
        # 原 slicer 返回的每一项形如：
        #   [chunk, start, end]
        # 其中 start/end 会直接被原 slice_audio.py 写进文件名。
        # 这里我们将其理解为基于 sample_rate 的时间位置。
        sliced = slicer.slice(audio)

        for item in sliced:
            if len(item) != 3:
                # Defensive check in case upstream implementation changes
                # 防御性检查，防止上游实现变化
                continue

            chunk, start, end = item
            chunk = np.asarray(chunk, dtype=np.float32)

            if chunk.size == 0:
                continue

            duration_sec = float(chunk.shape[0]) / float(sample_rate)
            if duration_sec <= 0:
                continue

            sample_id = f"{running_index:06d}"
            clip_path = output_root / f"{sample_id}.wav"

            chunk = _safe_normalize_chunk(
                chunk=chunk,
                normalize_max=normalize_max,
                alpha_mix=alpha_mix,
            )

            # Save as 16-bit PCM WAV for broad compatibility
            # 以 PCM_16 保存，兼容性更高
            sf.write(
                file=str(clip_path),
                data=chunk,
                samplerate=sample_rate,
                subtype="PCM_16",
            )

            start_sec = float(start) / float(sample_rate)
            end_sec = float(end) / float(sample_rate)

            slice_items.append(
                SliceItem(
                    sample_id=sample_id,
                    wav_path=str(clip_path.resolve()),
                    source_audio=str(src_audio.resolve()),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                    speaker_name=speaker_name,
                    language=language,
                )
            )

            running_index += 1

    assert_slice_items_non_empty(slice_items)
    return slice_items