from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import webdataset as wds
from huggingface_hub import HfApi, hf_hub_download

from .emilia_item import EmiliaRawItem


def _parse_json_field(x: Any) -> dict[str, Any]:
    """
    EN:
    Robust parser for WebDataset json field.

    ZH:
    稳健解析 WebDataset 中的 json 字段。
    """
    if isinstance(x, dict):
        return x

    if isinstance(x, bytes):
        return json.loads(x.decode("utf-8"))

    if isinstance(x, str):
        return json.loads(x)

    return {}


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    EN:
    Safe float conversion.

    ZH:
    安全 float 转换。
    """
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_optional_float(x: Any) -> float | None:
    """
    EN:
    Safe optional float conversion.

    ZH:
    安全可选 float 转换。
    """
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def list_emilia_tar_files(
    repo_id: str = "amphion/Emilia-Dataset",
    language: str = "ZH",
    token: bool | str = True,
) -> list[str]:
    """
    EN:
    List Emilia tar files for one language split.

    Example:
        Emilia/ZH/ZH-B000000.tar

    ZH:
    列出某个语言子集的 Emilia tar 文件。
    """
    api = HfApi(token=token)

    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )

    lang = language.strip().upper()
    prefix = f"Emilia/{lang}/"

    tar_files = [
        f for f in files
        if f.startswith(prefix) and f.endswith(".tar")
    ]

    tar_files.sort()
    return tar_files


def download_emilia_tar(
    *,
    repo_id: str,
    tar_file: str,
    cache_dir: str | Path,
    token: bool | str = True,
    force_download: bool = False,
) -> Path:
    """
    EN:
    Download one Emilia tar shard to local_dir.

    ZH:
    下载一个 Emilia tar shard 到本地 cache 目录。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=tar_file,
        token=token,
        local_dir=str(cache_dir),
        force_download=force_download,
    )

    return Path(local_path)


def parse_emilia_sample(
    sample: dict[str, Any],
    *,
    tar_file: str,
    tar_path: str | Path,
    shard_index: int | None = None,
) -> EmiliaRawItem:
    """
    EN:
    Convert one WebDataset sample to EmiliaRawItem.

    ZH:
    将一个 WebDataset sample 转成 EmiliaRawItem。
    """
    meta = _parse_json_field(sample.get("json"))

    key = sample.get("__key__")
    url = sample.get("__url__")

    sample_id = str(
        meta.get("id")
        or meta.get("sample_id")
        or key
        or ""
    )

    text = str(
        meta.get("text")
        or meta.get("transcript")
        or ""
    )

    language = str(
        meta.get("language")
        or meta.get("lang")
        or ""
    )

    speaker = meta.get("speaker") or meta.get("speaker_id")
    if speaker is not None:
        speaker = str(speaker)

    duration_sec = _safe_float(
        meta.get("duration")
        or meta.get("duration_sec")
        or meta.get("dur"),
        default=0.0,
    )

    dnsmos = _safe_optional_float(meta.get("dnsmos"))

    mp3 = sample.get("mp3")
    mp3_num_bytes = len(mp3) if isinstance(mp3, bytes) else None

    return EmiliaRawItem(
        sample_id=sample_id,
        text=text,
        language=language,
        speaker=speaker,
        duration_sec=duration_sec,
        dnsmos=dnsmos,
        source="emilia_hf",
        audio_format="mp3",
        key=str(key) if key is not None else None,
        tar_file=str(tar_file),
        tar_path=str(Path(tar_path).as_posix()),
        url=str(url) if url is not None else None,
        shard_index=shard_index,
        mp3_num_bytes=mp3_num_bytes,
        raw_json=meta,
    )


def iter_emilia_items_from_local_tar(
    *,
    tar_path: str | Path,
    tar_file: str | None = None,
    shard_index: int | None = None,
) -> Iterator[EmiliaRawItem]:
    """
    EN:
    Iterate EmiliaRawItem from one local tar shard using WebDataset.

    Important:
    On Windows, WebDataset may misread backslash paths.
    We always pass a forward-slash path.

    ZH:
    使用 WebDataset 从一个本地 tar shard 中迭代 EmiliaRawItem。

    注意：
    Windows 下 WebDataset 可能错误解析反斜杠路径。
    因此这里统一传入 forward-slash 路径。
    """
    tar_path = Path(tar_path)

    if tar_file is None:
        tar_file = tar_path.name

    if not tar_path.exists():
        raise FileNotFoundError(f"tar_path not found: {tar_path}")

    # Use relative/forward-slash path when possible.
    tar_url = tar_path.as_posix()

    dataset = wds.WebDataset(
        [tar_url],
        shardshuffle=False,
    )

    for sample in dataset:
        yield parse_emilia_sample(
            sample,
            tar_file=str(tar_file),
            tar_path=tar_path,
            shard_index=shard_index,
        )