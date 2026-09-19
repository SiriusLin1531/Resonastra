from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RawAudioItem:
    """
    EN:
    Metadata for one raw long-form source audio file.

    ZH:
    一条原始长音频文件的元信息。
    """

    source_audio: str
    speaker_name: str
    language: str

    def to_dict(self) -> dict[str, Any]:
        """
        EN:
        Convert dataclass to plain dict.

        ZH:
        将 dataclass 转换为普通字典。
        """
        return asdict(self)


@dataclass
class SliceItem:
    """
    EN:
    Metadata for one sliced audio clip produced from a raw audio file.

    ZH:
    从原始长音频切分出来的一条短音频片段的元信息。
    """

    sample_id: str
    wav_path: str
    source_audio: str
    start_sec: float
    end_sec: float
    duration_sec: float
    speaker_name: str
    language: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ASRItem:
    """
    EN:
    Slice item after automatic speech recognition.

    ZH:
    经过自动语音识别后的切片数据项。
    """

    sample_id: str
    wav_path: str
    source_audio: str
    start_sec: float
    end_sec: float
    duration_sec: float
    speaker_name: str
    language: str

    # --------------------------------------------------------
    # ASR related fields
    # ASR 相关字段
    # --------------------------------------------------------
    text: str
    asr_source: str
    text_status: str = "auto"   # auto / manual_corrected / rejected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportItem:
    """
    EN:
    Final export item for dataset manifest / .list / stage2 manifest.

    ZH:
    最终导出用的数据项，可用于：
    - 标准 manifest.jsonl
    - GPT-SoVITS 兼容 .list
    - 你自己的 stage2_manifest.jsonl
    """

    sample_id: str
    wav_path: str
    text: str
    language: str
    speaker_name: str
    source_audio: str
    start_sec: float
    end_sec: float
    duration_sec: float
    asr_source: str
    text_status: str = "auto"

    # --------------------------------------------------------
    # Optional extensible metadata
    # 可扩展元数据
    # --------------------------------------------------------
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_rejected_text_status(text_status: str) -> bool:
    """
    EN:
    Helper to judge whether one item is marked as rejected.

    ZH:
    判断某条样本是否被标记为拒绝/丢弃。
    """
    return str(text_status).strip().lower() == "rejected"