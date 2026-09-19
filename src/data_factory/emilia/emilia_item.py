from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EmiliaRawItem:
    """
    EN:
    A lightweight raw item parsed from Emilia WebDataset tar shard.

    This object is intentionally kept close to the original Emilia metadata.
    It does not decode MP3 and does not contain VoiceLab features yet.

    ZH:
    从 Emilia WebDataset tar shard 中解析出的轻量原始样本。

    这个对象刻意保持接近 Emilia 原始 metadata。
    它不解码 MP3，也不包含 VoiceLab 特征。
    """

    sample_id: str
    text: str
    language: str
    speaker: str | None
    duration_sec: float
    dnsmos: float | None

    source: str = "emilia_hf"
    audio_format: str = "mp3"

    # Hugging Face / WebDataset location info
    key: str | None = None
    tar_file: str | None = None
    tar_path: str | None = None
    url: str | None = None
    shard_index: int | None = None

    # Size info
    mp3_num_bytes: int | None = None

    # Keep selected raw metadata for debugging.
    raw_json: dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, Any]:
        """
        EN:
        Convert to JSONL-safe manifest record.

        ZH:
        转成可写入 JSONL 的 manifest 记录。
        """
        d = asdict(self)

        # Make sure raw_json is always a dict.
        if d.get("raw_json") is None:
            d["raw_json"] = {}

        return d