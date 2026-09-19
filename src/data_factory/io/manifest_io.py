from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def _record_to_dict(record: Any) -> dict[str, Any]:
    """
    EN:
    Convert one record into a plain dict.

    Supported input types:
    - dict
    - dataclass instance
    - any object with `to_dict()` method

    ZH:
    将单条记录转换成普通 dict。

    支持输入类型：
    - dict
    - dataclass 实例
    - 带有 `to_dict()` 方法的对象
    """
    if isinstance(record, dict):
        return record

    if is_dataclass(record):
        return asdict(record)

    if hasattr(record, "to_dict") and callable(record.to_dict):
        return record.to_dict()

    raise TypeError(f"Unsupported record type: {type(record)}")


def write_jsonl(records: Iterable[Any], path: str | Path) -> Path:
    """
    EN:
    Write records into a JSONL file.
    One line = one JSON object.

    ZH:
    将记录写入 JSONL 文件。
    一行 = 一条 JSON 对象。
    """
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            data = _record_to_dict(record)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    EN:
    Read a JSONL file into a list of dicts.

    ZH:
    读取 JSONL 文件并返回 dict 列表。
    """
    input_path = Path(path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_idx} in {input_path}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_idx} in {input_path} is not a JSON object.")
            records.append(obj)

    return records


def write_gsv_list(records: Iterable[Any], path: str | Path) -> Path:
    """
    EN:
    Write GPT-SoVITS compatible `.list` file.

    Expected fields in each record:
    - wav_path
    - speaker_name
    - language
    - text

    Output line format:
        wav_path|speaker_name|language|text

    ZH:
    写入 GPT-SoVITS 兼容 `.list` 文件。

    每条记录要求包含字段：
    - wav_path
    - speaker_name
    - language
    - text

    输出格式：
        wav_path|speaker_name|language|text
    """
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            data = _record_to_dict(record)

            required_keys = ["wav_path", "speaker_name", "language", "text"]
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing required key for .list export: {key}")

            wav_path = str(data["wav_path"]).strip()
            speaker_name = str(data["speaker_name"]).strip()
            language = str(data["language"]).strip()
            text = str(data["text"]).strip()

            line = f"{wav_path}|{speaker_name}|{language}|{text}"
            f.write(line + "\n")

    return output_path


def read_gsv_list(path: str | Path) -> list[dict[str, str]]:
    """
    EN:
    Read GPT-SoVITS compatible `.list` file.

    Input line format:
        wav_path|speaker_name|language|text

    ZH:
    读取 GPT-SoVITS 兼容 `.list` 文件。

    输入格式：
        wav_path|speaker_name|language|text
    """
    input_path = Path(path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f".list file not found: {input_path}")

    records: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("|", maxsplit=3)
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid .list format at line {line_idx} in {input_path}: expected 4 fields, got {len(parts)}"
                )

            wav_path, speaker_name, language, text = parts
            records.append(
                {
                    "wav_path": wav_path.strip(),
                    "speaker_name": speaker_name.strip(),
                    "language": language.strip(),
                    "text": text.strip(),
                }
            )

    return records