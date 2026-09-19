from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from pathlib import Path

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data_factory.io.manifest_io import (
    read_gsv_list,
    write_gsv_list,
    write_jsonl,
)
from src.data_factory.schemas import ExportItem, SliceItem, is_rejected_text_status
from src.data_factory.utils.validation import (
    assert_asr_output_exists,
    assert_manifest_items_non_empty,
)


def _normalize_wav_path(path: str) -> str:
    """
    EN:
    Normalize a wav path into resolved absolute string form.

    This is important because:
    - slice_items store absolute paths
    - ASR .list paths should also be normalized to the same form
    so we can join them reliably.

    ZH:
    将 wav 路径规范化为绝对路径字符串。

    这样做很重要，因为：
    - slice_items 中保存的是绝对路径
    - ASR .list 中的路径也要归一成同样格式
    这样两者才能稳定对齐。
    """
    return str(Path(path).resolve())


def _build_slice_lookup(slice_items: list[SliceItem]) -> dict[str, SliceItem]:
    """
    EN:
    Build a lookup from normalized wav_path -> SliceItem.

    ZH:
    构建一个从规范化 wav_path -> SliceItem 的映射表。
    """
    lookup: dict[str, SliceItem] = {}
    for item in slice_items:
        key = _normalize_wav_path(item.wav_path)
        lookup[key] = item
    return lookup


def _infer_asr_source_from_path(asr_list_path: str | Path) -> str:
    """
    EN:
    Infer a human-readable ASR source name from the ASR list file path.

    This is only a best-effort first-version heuristic.

    ZH:
    从 ASR list 文件路径中推断一个可读的 ASR 来源名称。

    这里只是第一版的启发式推断。
    """
    p = str(Path(asr_list_path).resolve()).lower()

    if "funasr" in p or "damo" in p:
        return "funasr"
    if "faster" in p and "whisper" in p:
        return "faster_whisper"

    return "unknown"


def build_export_items_from_asr_list(
    asr_list_path: str | Path,
    slice_items: list[SliceItem],
    speaker_name: str,
    language: str,
) -> list[ExportItem]:
    """
    EN:
    Build final export items by joining:
    - ASR .list output
    - slice metadata produced by slicer_runner

    Input ASR .list format follows GPT-SoVITS convention:
        wav_path|speaker_name_or_folder|LANG|text

    We mainly trust:
    - wav_path
    - text

    Metadata such as source_audio / time spans are recovered from slice_items.

    ZH:
    通过关联以下两部分数据，构建最终导出项：
    - ASR 输出的 .list
    - slicer_runner 产出的切片元数据

    输入的 ASR .list 格式遵循 GPT-SoVITS 习惯：
        wav_path|speaker_name_or_folder|LANG|text

    我们主要信任：
    - wav_path
    - text

    而 source_audio / 起止时间等元信息则从 slice_items 中恢复。
    """
    asr_list_path = assert_asr_output_exists(asr_list_path)
    slice_lookup = _build_slice_lookup(slice_items)
    asr_rows = read_gsv_list(asr_list_path)

    export_items: list[ExportItem] = []
    asr_source = _infer_asr_source_from_path(asr_list_path)

    for idx, row in enumerate(asr_rows, start=1):
        wav_path = _normalize_wav_path(row["wav_path"])

        if wav_path not in slice_lookup:
            raise KeyError(
                f"ASR row wav_path not found in slice_items: {wav_path}\n"
                f"This usually means slice output and ASR input are not aligned."
            )

        slice_item = slice_lookup[wav_path]

        # EN:
        # Keep the original requested dataset language as the main language field,
        # because this is what your downstream training pipeline currently expects.
        #
        # We still preserve ASR-detected language in metadata for traceability.
        #
        # ZH:
        # 主 language 字段优先保留你构建数据集时指定的语言，
        # 因为这更符合你当前下游训练链的预期。
        #
        # 同时把 ASR 输出中的语言保存在 metadata 中，便于追踪。
        detected_lang = str(row.get("language", "")).strip().lower()
        text = str(row.get("text", "")).strip()

        export_items.append(
            ExportItem(
                sample_id=slice_item.sample_id,
                wav_path=slice_item.wav_path,
                text=text,
                language=slice_item.language or language,
                speaker_name=slice_item.speaker_name or speaker_name,
                source_audio=slice_item.source_audio,
                start_sec=slice_item.start_sec,
                end_sec=slice_item.end_sec,
                duration_sec=slice_item.duration_sec,
                asr_source=asr_source,
                text_status="auto",
                metadata={
                    "detected_language": detected_lang,
                    "asr_list_path": str(asr_list_path),
                    "slice_index": idx,
                },
            )
        )

    assert_manifest_items_non_empty(export_items, name="export_items")
    return export_items


def export_dataset_manifest_jsonl(
    items: list[ExportItem],
    output_path: str | Path,
    export_format: str = "jsonl",
) -> Path:
    """
    EN:
    Export final dataset items into one of:
    - JSONL manifest
    - GPT-SoVITS compatible .list

    Supported values:
    - export_format="jsonl"
    - export_format="list"

    ZH:
    将最终导出项写成以下任一格式：
    - JSONL manifest
    - GPT-SoVITS 兼容 .list

    支持：
    - export_format="jsonl"
    - export_format="list"
    """
    assert_manifest_items_non_empty(items, name="export_items")

    export_format = str(export_format).strip().lower()
    output_path = Path(output_path).resolve()

    # --------------------------------------------------------
    # Filter rejected items
    # 过滤掉被标记为 rejected 的样本
    # --------------------------------------------------------
    valid_items = [item for item in items if not is_rejected_text_status(item.text_status)]
    assert_manifest_items_non_empty(valid_items, name="valid_export_items")

    if export_format == "jsonl":
        return write_jsonl(valid_items, output_path)

    if export_format == "list":
        # EN:
        # write_gsv_list only needs:
        # wav_path | speaker_name | language | text
        #
        # ZH:
        # write_gsv_list 只需要：
        # wav_path | speaker_name | language | text
        return write_gsv_list(valid_items, output_path)

    raise ValueError(f"Unsupported export_format: {export_format}")