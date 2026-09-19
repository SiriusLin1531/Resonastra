from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data_factory.io.audio_io import get_audio_duration_sec
from src.data_factory.io.manifest_io import read_gsv_list, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse arguments for rebuilding corrected manifests.

    ZH:
    解析“从修正后的 .list 反向重建 corrected manifest”的参数。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_manifest", type=str, required=True)
    parser.add_argument("--corrected_list", type=str, required=True)

    # Optional but strongly recommended for reporting and precise diff
    parser.add_argument("--original_list", type=str, default=None)

    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--output_stage2_manifest", type=str, required=True)
    parser.add_argument("--output_report", type=str, required=True)

    parser.add_argument("--duration_tolerance_sec", type=float, default=0.05)
    parser.add_argument("--prompt_mode", type=str, default="self")

    return parser.parse_args()


def normalize_path_str(path: str | Path) -> str:
    """
    EN:
    Normalize path to resolved absolute string.

    ZH:
    将路径规范化为绝对路径字符串。
    """
    return str(Path(path).resolve())


def natural_sort_key(text: str) -> list[Any]:
    """
    EN:
    Build a natural sort key from a string.

    Example:
        000001      -> [1]
        000001_00   -> [1, 0]
        000010      -> [10]

    This helps keep output order intuitive without renumbering sample_id.

    ZH:
    从字符串构造自然排序键。

    例如：
        000001      -> [1]
        000001_00   -> [1, 0]
        000010      -> [10]

    这样可以在不重编号 sample_id 的前提下，让输出顺序更符合直觉。
    """
    text = str(text)
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []

    for part in parts:
        if part == "":
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())

    return key


def row_natural_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    EN:
    Natural sort key for manifest-like rows.

    Priority:
    1. sample_id if present
    2. stem of wav_path
    3. wav_path full string

    ZH:
    为 manifest 类记录构造自然排序键。

    优先级：
    1. 若有 sample_id，则优先按 sample_id 排
    2. 否则按 wav_path 的 stem 排
    3. 再用完整 wav_path 作为兜底
    """
    sample_id = str(row.get("sample_id", "")).strip()
    wav_path = normalize_path_str(row.get("wav_path", "")) if row.get("wav_path") else ""
    wav_stem = Path(wav_path).stem if wav_path else ""

    primary = sample_id if sample_id else wav_stem
    return (
        tuple(natural_sort_key(primary)),
        tuple(natural_sort_key(wav_stem)),
        wav_path.lower(),
    )


def sort_manifest_rows_naturally(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    EN:
    Return a new list sorted in natural order.

    ZH:
    返回按自然顺序排序后的新列表。
    """
    return sorted(rows, key=row_natural_sort_key)


def index_manifest_by_wav_path(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    EN:
    Index manifest rows by normalized wav_path.

    ZH:
    用规范化后的 wav_path 为键，建立 manifest 索引。
    """
    out: dict[str, dict[str, Any]] = {}
    for row in records:
        wav_path = normalize_path_str(row["wav_path"])
        out[wav_path] = row
    return out


def index_list_by_wav_path(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """
    EN:
    Index .list rows by normalized wav_path.

    ZH:
    用规范化后的 wav_path 为键，建立 .list 索引。
    """
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        wav_path = normalize_path_str(row["wav_path"])
        out[wav_path] = {
            "wav_path": wav_path,
            "speaker_name": str(row["speaker_name"]).strip(),
            "language": str(row["language"]).strip(),
            "text": str(row["text"]).strip(),
        }
    return out


def infer_parent_wav_path_for_generated_clip(
    wav_path: str,
    original_manifest_map: dict[str, dict[str, Any]],
) -> str | None:
    """
    EN:
    Try to infer a parent wav path for generated clips created by split operations.

    Typical pattern from subfix split:
        000001.wav -> 000001_00.wav

    Strategy:
    - if stem ends with _<digits>, strip the suffix and see whether the base wav exists
    - otherwise return None

    ZH:
    尝试为“新生成的切片音频”推断一个父样本路径。

    subfix split 常见模式：
        000001.wav -> 000001_00.wav

    策略：
    - 如果 stem 以 _<数字> 结尾，则去掉这一段，看看基础 wav 是否存在
    - 否则返回 None
    """
    p = Path(wav_path).resolve()
    stem = p.stem

    if "_" not in stem:
        return None

    base_stem, suffix = stem.rsplit("_", 1)
    if not suffix.isdigit():
        return None

    candidate = p.with_name(base_stem + p.suffix)
    candidate_norm = normalize_path_str(candidate)

    if candidate_norm in original_manifest_map:
        return candidate_norm

    return None


def choose_edit_type(
    text_changed: bool,
    duration_changed: bool,
    path_known_in_original: bool,
) -> str:
    """
    EN:
    Assign a coarse edit type for the rebuilt row.

    ZH:
    为重建后的记录分配一个粗粒度编辑类型。
    """
    if not path_known_in_original:
        return "new_generated_clip"
    if duration_changed:
        return "audio_changed_same_path"
    if text_changed:
        return "text_edited"
    return "unchanged"


def build_corrected_manifest_rows(
    original_manifest_rows: list[dict[str, Any]],
    corrected_list_rows: list[dict[str, str]],
    original_list_rows: list[dict[str, str]] | None,
    duration_tolerance_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    EN:
    Core rebuild logic.

    Rebuild corrected manifest rows from:
    - original manifest.jsonl
    - corrected dataset.list
    - optional original dataset.before_proofread.list

    Conservative metadata strategy:
    - if wav path is unchanged and audio duration is unchanged:
        keep original source_audio / start_sec / end_sec
    - if wav path is unchanged but duration changed:
        keep source_audio, but set start_sec / end_sec = None
    - if wav path is new (e.g. split-created new file):
        try to infer parent and inherit source_audio
        but set start_sec / end_sec = None

    ZH:
    核心重建逻辑。

    根据：
    - 原始 manifest.jsonl
    - 修正后的 dataset.list
    - 可选的原始 dataset.before_proofread.list

    来重建 corrected manifest。

    保守 metadata 策略：
    - 如果 wav 路径没变且音频时长没变：
        保留原 source_audio / start_sec / end_sec
    - 如果 wav 路径没变但时长变了：
        保留 source_audio，但将 start_sec / end_sec 设为 None
    - 如果 wav 路径是新生成的（如 split 新增文件）：
        尝试推断父样本并继承 source_audio
        但 start_sec / end_sec 设为 None
    """
    original_manifest_map = index_manifest_by_wav_path(original_manifest_rows)
    corrected_list_map = index_list_by_wav_path(corrected_list_rows)
    original_list_map = index_list_by_wav_path(original_list_rows) if original_list_rows is not None else {}

    corrected_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "counts": {
            "unchanged": 0,
            "text_edited": 0,
            "audio_changed_same_path": 0,
            "new_generated_clip": 0,
            "deleted_or_missing_from_corrected_list": 0,
        },
        "unchanged": [],
        "text_edited": [],
        "audio_changed_same_path": [],
        "new_generated_clip": [],
        "deleted_or_missing_from_corrected_list": [],
    }

    corrected_wav_paths_seen: set[str] = set()

    for corrected_row in corrected_list_rows:
        wav_path = normalize_path_str(corrected_row["wav_path"])
        corrected_wav_paths_seen.add(wav_path)

        text = str(corrected_row["text"]).strip()
        speaker_name = str(corrected_row["speaker_name"]).strip()
        language = str(corrected_row["language"]).strip()

        duration_sec = get_audio_duration_sec(wav_path)

        if wav_path in original_manifest_map:
            original_row = deepcopy(original_manifest_map[wav_path])

            original_text = str(original_row.get("text", "")).strip()
            original_duration = original_row.get("duration_sec", None)
            if original_duration is None:
                original_duration = duration_sec
            original_duration = float(original_duration)

            text_changed = (text != original_text)
            duration_changed = abs(float(duration_sec) - float(original_duration)) > float(duration_tolerance_sec)

            edit_type = choose_edit_type(
                text_changed=text_changed,
                duration_changed=duration_changed,
                path_known_in_original=True,
            )

            metadata = deepcopy(original_row.get("metadata", {}))
            metadata.update(
                {
                    "correction_type": edit_type,
                    "rebuild_source": "corrected_list",
                    "original_wav_path": wav_path,
                }
            )

            rebuilt = deepcopy(original_row)
            rebuilt["wav_path"] = wav_path
            rebuilt["text"] = text
            rebuilt["speaker_name"] = speaker_name
            rebuilt["language"] = language or original_row.get("language", "")
            rebuilt["duration_sec"] = float(duration_sec)
            rebuilt["text_status"] = "manual_corrected" if edit_type != "unchanged" else original_row.get(
                "text_status", "auto"
            )
            rebuilt["metadata"] = metadata

            # Conservative strategy for audio-changed samples
            if duration_changed:
                rebuilt["start_sec"] = None
                rebuilt["end_sec"] = None

            corrected_rows.append(rebuilt)
            report["counts"][edit_type] += 1
            report[edit_type].append(
                {
                    "sample_id": rebuilt.get("sample_id"),
                    "wav_path": wav_path,
                    "text_before": original_text,
                    "text_after": text,
                    "duration_before": original_duration,
                    "duration_after": duration_sec,
                }
            )

        else:
            # New generated clip, often caused by split
            parent_wav_path = infer_parent_wav_path_for_generated_clip(
                wav_path=wav_path,
                original_manifest_map=original_manifest_map,
            )
            parent_row = deepcopy(original_manifest_map[parent_wav_path]) if parent_wav_path else None

            if parent_row is not None:
                source_audio = parent_row.get("source_audio")
                asr_source = parent_row.get("asr_source", "proofread_rebuild")
                parent_sample_id = parent_row.get("sample_id")
            else:
                source_audio = None
                asr_source = "proofread_rebuild"
                parent_sample_id = None

            rebuilt = {
                "sample_id": Path(wav_path).stem,
                "wav_path": wav_path,
                "text": text,
                "language": language,
                "speaker_name": speaker_name,
                "source_audio": source_audio,
                "start_sec": None,
                "end_sec": None,
                "duration_sec": float(duration_sec),
                "asr_source": asr_source,
                "text_status": "manual_corrected",
                "metadata": {
                    "correction_type": "new_generated_clip",
                    "rebuild_source": "corrected_list",
                    "inferred_parent_wav_path": parent_wav_path,
                    "inferred_parent_sample_id": parent_sample_id,
                },
            }

            corrected_rows.append(rebuilt)
            report["counts"]["new_generated_clip"] += 1
            report["new_generated_clip"].append(
                {
                    "sample_id": rebuilt.get("sample_id"),
                    "wav_path": wav_path,
                    "text_after": text,
                    "duration_after": duration_sec,
                    "inferred_parent_wav_path": parent_wav_path,
                    "inferred_parent_sample_id": parent_sample_id,
                }
            )

    # Anything in original manifest but missing from corrected list
    for wav_path, original_row in original_manifest_map.items():
        if wav_path not in corrected_wav_paths_seen:
            report["counts"]["deleted_or_missing_from_corrected_list"] += 1
            report["deleted_or_missing_from_corrected_list"].append(
                {
                    "sample_id": original_row.get("sample_id"),
                    "wav_path": wav_path,
                    "text_before": original_row.get("text", ""),
                }
            )

    return corrected_rows, report


def build_stage2_rows(
    corrected_manifest_rows: list[dict[str, Any]],
    prompt_mode: str = "self",
) -> list[dict[str, Any]]:
    """
    EN:
    Build stage2 manifest rows from corrected manifest rows.

    Current supported mode:
    - self:
        prompt_wav_path = target_wav_path = wav_path

    ZH:
    从 corrected manifest 构建 stage2 manifest。

    当前支持：
    - self:
        prompt_wav_path = target_wav_path = wav_path
    """
    prompt_mode = str(prompt_mode).strip().lower()

    if prompt_mode != "self":
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    stage2_rows: list[dict[str, Any]] = []

    for row in corrected_manifest_rows:
        wav_path = normalize_path_str(row["wav_path"])
        text = str(row["text"]).strip()

        stage2_rows.append(
            {
                "sample_id": str(row.get("sample_id", Path(wav_path).stem)),
                "raw_text": text,
                "language": str(row.get("language", "")).strip(),
                "prompt_wav_path": wav_path,
                "target_wav_path": wav_path,
            }
        )

    return stage2_rows


def save_json(obj: dict[str, Any], path: Path) -> None:
    """
    EN:
    Save JSON file.

    ZH:
    保存 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    original_manifest_path = Path(args.original_manifest).resolve()
    corrected_list_path = Path(args.corrected_list).resolve()
    output_manifest_path = Path(args.output_manifest).resolve()
    output_stage2_manifest_path = Path(args.output_stage2_manifest).resolve()
    output_report_path = Path(args.output_report).resolve()

    if args.original_list is not None:
        original_list_path = Path(args.original_list).resolve()
    else:
        # Auto-fallback: sibling file dataset.before_proofread.list beside corrected list
        sibling_guess = corrected_list_path.with_name("dataset.before_proofread.list")
        original_list_path = sibling_guess if sibling_guess.exists() else None

    if not original_manifest_path.exists():
        raise FileNotFoundError(f"original_manifest not found: {original_manifest_path}")
    if not corrected_list_path.exists():
        raise FileNotFoundError(f"corrected_list not found: {corrected_list_path}")
    if original_list_path is not None and not original_list_path.exists():
        raise FileNotFoundError(f"original_list not found: {original_list_path}")

    original_manifest_rows = read_jsonl(original_manifest_path)
    corrected_list_rows = read_gsv_list(corrected_list_path)
    original_list_rows = read_gsv_list(original_list_path) if original_list_path is not None else None

    corrected_manifest_rows, report = build_corrected_manifest_rows(
        original_manifest_rows=original_manifest_rows,
        corrected_list_rows=corrected_list_rows,
        original_list_rows=original_list_rows,
        duration_tolerance_sec=float(args.duration_tolerance_sec),
    )

    # --------------------------------------------------------
    # Natural sort output rows before writing
    # 在写出之前按自然顺序排序
    # --------------------------------------------------------
    corrected_manifest_rows = sort_manifest_rows_naturally(corrected_manifest_rows)

    stage2_rows = build_stage2_rows(
        corrected_manifest_rows=corrected_manifest_rows,
        prompt_mode=args.prompt_mode,
    )
    stage2_rows = sort_manifest_rows_naturally(stage2_rows)

    write_jsonl(corrected_manifest_rows, output_manifest_path)
    write_jsonl(stage2_rows, output_stage2_manifest_path)

    report_payload = {
        "original_manifest": str(original_manifest_path),
        "corrected_list": str(corrected_list_path),
        "original_list": str(original_list_path) if original_list_path is not None else None,
        "output_manifest": str(output_manifest_path),
        "output_stage2_manifest": str(output_stage2_manifest_path),
        "duration_tolerance_sec": float(args.duration_tolerance_sec),
        "prompt_mode": args.prompt_mode,
        "report": report,
    }
    save_json(report_payload, output_report_path)

    print("====================================================")
    print("Rebuild corrected manifest finished")
    print(f"original_manifest       : {original_manifest_path}")
    print(f"corrected_list          : {corrected_list_path}")
    print(f"original_list           : {original_list_path}")
    print(f"output_manifest         : {output_manifest_path}")
    print(f"output_stage2_manifest  : {output_stage2_manifest_path}")
    print(f"output_report           : {output_report_path}")
    print("====================================================")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print("====================================================")


if __name__ == "__main__":
    main()