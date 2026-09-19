from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import json
import math
from pathlib import Path
from typing import Any

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data_factory.io.manifest_io import write_jsonl
from src.data_factory.schemas import ExportItem, is_rejected_text_status
from src.data_factory.utils.validation import assert_manifest_items_non_empty

try:
    from src.data_factory.io.audio_io import get_audio_duration_sec
except Exception:
    get_audio_duration_sec = None


# ============================================================
# Basic helpers
# 基础工具函数
# ============================================================
def _choose_final_text(item: ExportItem) -> str:
    """
    EN:
    Choose the final text used for downstream model training.

    First-version rule:
    - use item.text directly as the final text.

    Future extensibility:
    - if later you add manual_text / corrected_text in metadata,
      this function is the right place to switch logic.

    ZH:
    选择最终用于下游模型训练的文本。

    第一版规则：
    - 直接使用 item.text 作为最终文本。

    未来可扩展：
    - 如果后面你在 metadata 中加入 manual_text / corrected_text，
      这里就是最合适的切换逻辑。
    """
    return str(item.text).strip()


def _normalize_prompt_mode(prompt_mode: str) -> str:
    """
    EN:
    Normalize prompt mode.

    ZH:
    规范化 prompt 模式。
    """
    mode = str(prompt_mode).strip().lower()

    aliases = {
        "self_prompt": "self",
        "self-prompt": "self",
        "same": "self",
        "same_as_target": "self",
        "speaker-pool": "speaker_pool",
        "speaker_pool_prompt": "speaker_pool",
        "pool": "speaker_pool",
        "fixed": "fixed_reference",
        "fixed_ref": "fixed_reference",
        "fixed-reference": "fixed_reference",
        "fixed_prompt": "fixed_reference",
    }

    return aliases.get(mode, mode)


def _safe_float(x: Any, default: float = math.nan) -> float:
    """
    EN:
    Defensive float conversion.

    ZH:
    防御式 float 转换。
    """
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _is_finite(x: float) -> bool:
    """
    EN:
    Whether x is finite.

    ZH:
    判断是否是有限数值。
    """
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _normalize_path(path: str | Path) -> str:
    """
    EN:
    Normalize a path into absolute resolved string form.

    ZH:
    将路径规范化为绝对路径字符串。
    """
    return str(Path(path).resolve())


def _path_exists(path: str | Path) -> bool:
    """
    EN:
    Check whether a path exists and is a file.

    ZH:
    检查路径是否存在且是文件。
    """
    try:
        p = Path(path).resolve()
        return p.exists() and p.is_file()
    except Exception:
        return False


def _duration_from_item_or_audio(item: ExportItem) -> float:
    """
    EN:
    Return duration from item.duration_sec first.
    If unavailable, try to read audio duration from wav_path.

    ZH:
    优先使用 item.duration_sec。
    如果不可用，则尝试从 wav_path 读取音频时长。
    """
    duration_sec = _safe_float(getattr(item, "duration_sec", math.nan))

    if _is_finite(duration_sec) and duration_sec > 0:
        return float(duration_sec)

    if get_audio_duration_sec is None:
        return math.nan

    try:
        wav_path = Path(item.wav_path).resolve()
        if not wav_path.exists():
            return math.nan
        return float(get_audio_duration_sec(wav_path))
    except Exception:
        return math.nan


def _is_valid_prompt_duration(
    duration_sec: float,
    *,
    min_prompt_sec: float,
    max_prompt_sec: float,
) -> bool:
    """
    EN:
    Whether a clip duration is valid as GPT-SoVITS-style prompt audio.

    ZH:
    判断某条音频时长是否适合作为 GPT-SoVITS 风格 prompt 音频。
    """
    if not _is_finite(duration_sec):
        return False

    return (
        float(duration_sec) >= float(min_prompt_sec)
        and float(duration_sec) <= float(max_prompt_sec)
    )


def _prompt_score(
    duration_sec: float,
    *,
    prefer_prompt_sec: float,
) -> tuple[float, float]:
    """
    EN:
    Score a prompt candidate.

    Lower is better:
    1. distance to prefer_prompt_sec
    2. negative duration, so longer candidate wins when distances tie

    ZH:
    给 prompt 候选打分。

    越小越好：
    1. 距离 prefer_prompt_sec 越近越好
    2. 距离相同则更长的候选优先
    """
    d = float(duration_sec)

    return (
        abs(d - float(prefer_prompt_sec)),
        -d,
    )


def _speaker_key(item: ExportItem) -> str:
    """
    EN:
    Speaker grouping key.

    ZH:
    说话人分组键。
    """
    speaker = str(getattr(item, "speaker_name", "")).strip()

    if speaker:
        return speaker

    # Fallback:
    # If speaker_name is missing, treat everything as one few-shot speaker.
    # 若 speaker_name 缺失，则视为同一个 few-shot 说话人。
    return "__default_speaker__"


def _save_json(obj: dict[str, Any], path: str | Path) -> None:
    """
    EN:
    Save JSON object.

    ZH:
    保存 JSON 对象。
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Stage2 record builders
# Stage2 记录构建函数
# ============================================================
def _build_stage2_record(
    item: ExportItem,
    *,
    prompt_wav_path: str | Path,
) -> dict[str, Any]:
    """
    EN:
    Build one stage2 manifest row.

    Keep only the fields required by current preprocessing pipeline:
        sample_id
        raw_text
        language
        prompt_wav_path
        target_wav_path

    ZH:
    构建一条 stage2 manifest 记录。

    仅保留当前预处理链所需字段：
        sample_id
        raw_text
        language
        prompt_wav_path
        target_wav_path
    """
    final_text = _choose_final_text(item)

    return {
        "sample_id": str(item.sample_id),
        "raw_text": final_text,
        "language": str(item.language).strip(),
        "prompt_wav_path": _normalize_path(prompt_wav_path),
        "target_wav_path": _normalize_path(item.wav_path),
    }


def _build_stage2_record_self_prompt(item: ExportItem) -> dict[str, Any]:
    """
    EN:
    Build one stage2 manifest record using self-prompt:

        prompt_wav_path = target_wav_path = item.wav_path

    ZH:
    使用 self-prompt 构建一条 stage2 manifest 记录：

        prompt_wav_path = target_wav_path = item.wav_path
    """
    return _build_stage2_record(
        item,
        prompt_wav_path=item.wav_path,
    )


# ============================================================
# Prompt candidate selection
# prompt 候选选择
# ============================================================
def _build_prompt_candidate_pools(
    items: list[ExportItem],
    *,
    min_prompt_sec: float,
    max_prompt_sec: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    EN:
    Build prompt candidate pools grouped by speaker.

    A valid prompt candidate must:
    - not be rejected
    - have a valid wav file
    - have duration within [min_prompt_sec, max_prompt_sec]

    ZH:
    按 speaker 构建 prompt 候选池。

    合格 prompt 候选必须：
    - 未被 rejected
    - wav 文件存在
    - 时长处于 [min_prompt_sec, max_prompt_sec]
    """
    pools: dict[str, list[dict[str, Any]]] = {}
    rejected_candidates: list[dict[str, Any]] = []

    for item in items:
        if is_rejected_text_status(item.text_status):
            continue

        wav_path = _normalize_path(item.wav_path)
        speaker = _speaker_key(item)
        duration_sec = _duration_from_item_or_audio(item)

        reason = ""

        if not _path_exists(wav_path):
            reason = "wav_not_found"
        elif not _is_valid_prompt_duration(
            duration_sec,
            min_prompt_sec=min_prompt_sec,
            max_prompt_sec=max_prompt_sec,
        ):
            reason = "duration_out_of_prompt_range"

        candidate = {
            "sample_id": str(item.sample_id),
            "speaker_name": speaker,
            "wav_path": wav_path,
            "duration_sec": (
                float(duration_sec)
                if _is_finite(duration_sec)
                else None
            ),
        }

        if reason:
            candidate["reject_reason"] = reason
            rejected_candidates.append(candidate)
            continue

        pools.setdefault(speaker, []).append(candidate)

    return pools, rejected_candidates


def _select_prompt_from_pool(
    item: ExportItem,
    *,
    pool: list[dict[str, Any]],
    min_prompt_sec: float,
    max_prompt_sec: float,
    prefer_prompt_sec: float,
    allow_self_prompt: bool,
) -> tuple[dict[str, Any], str]:
    """
    EN:
    Select one prompt candidate from a speaker-specific pool.

    Policy:
    1. If allow_self_prompt and current target clip is valid, use itself.
    2. Otherwise choose candidate closest to prefer_prompt_sec.
    3. If no candidate exists, raise error.

    Return:
        (candidate, selection_reason)

    ZH:
    从同说话人候选池中选择一条 prompt。

    策略：
    1. 如果 allow_self_prompt=True 且当前 target 本身合格，优先使用自己。
    2. 否则选择时长最接近 prefer_prompt_sec 的候选。
    3. 如果没有候选，则报错。

    返回：
        (candidate, selection_reason)
    """
    target_wav_path = _normalize_path(item.wav_path)
    target_duration = _duration_from_item_or_audio(item)

    if (
        bool(allow_self_prompt)
        and _path_exists(target_wav_path)
        and _is_valid_prompt_duration(
            target_duration,
            min_prompt_sec=min_prompt_sec,
            max_prompt_sec=max_prompt_sec,
        )
    ):
        return (
            {
                "sample_id": str(item.sample_id),
                "speaker_name": _speaker_key(item),
                "wav_path": target_wav_path,
                "duration_sec": float(target_duration),
            },
            "self_prompt_valid",
        )

    if not pool:
        raise ValueError(
            "No valid prompt candidate found for speaker="
            f"{_speaker_key(item)!r}. "
            "Please provide longer clips or use prompt_mode=fixed_reference."
        )

    sorted_pool = sorted(
        pool,
        key=lambda x: (
            _prompt_score(
                float(x.get("duration_sec") or 0.0),
                prefer_prompt_sec=prefer_prompt_sec,
            ),
            str(x.get("sample_id", "")),
        ),
    )

    return sorted_pool[0], "selected_from_speaker_pool"


def _validate_fixed_reference(
    fixed_prompt_wav_path: str | Path | None,
    *,
    min_prompt_sec: float,
    max_prompt_sec: float,
    strict_prompt_duration: bool,
) -> dict[str, Any]:
    """
    EN:
    Validate fixed reference prompt audio.

    ZH:
    校验固定参考 prompt 音频。
    """
    if fixed_prompt_wav_path is None or str(fixed_prompt_wav_path).strip() == "":
        raise ValueError(
            "prompt_mode=fixed_reference requires fixed_prompt_wav_path."
        )

    wav_path = _normalize_path(fixed_prompt_wav_path)

    if not _path_exists(wav_path):
        raise FileNotFoundError(f"fixed_prompt_wav_path not found: {wav_path}")

    duration_sec = math.nan

    if get_audio_duration_sec is not None:
        try:
            duration_sec = float(get_audio_duration_sec(wav_path))
        except Exception:
            duration_sec = math.nan

    duration_ok = _is_valid_prompt_duration(
        duration_sec,
        min_prompt_sec=min_prompt_sec,
        max_prompt_sec=max_prompt_sec,
    )

    if bool(strict_prompt_duration) and not duration_ok:
        raise ValueError(
            f"fixed_prompt_wav_path duration is invalid: {duration_sec}. "
            f"Expected [{min_prompt_sec}, {max_prompt_sec}] seconds. "
            f"path={wav_path}"
        )

    return {
        "sample_id": "__fixed_reference__",
        "speaker_name": "__fixed_reference__",
        "wav_path": wav_path,
        "duration_sec": (
            float(duration_sec)
            if _is_finite(duration_sec)
            else None
        ),
        "duration_ok": bool(duration_ok),
    }


# ============================================================
# Report helpers
# 报告辅助函数
# ============================================================
def _build_report_base(
    *,
    prompt_mode: str,
    output_path: str | Path,
    min_prompt_sec: float,
    max_prompt_sec: float,
    prefer_prompt_sec: float,
) -> dict[str, Any]:
    """
    EN:
    Build prompt selection report base.

    ZH:
    构建 prompt 选择报告的基础结构。
    """
    return {
        "prompt_mode": prompt_mode,
        "output_path": str(Path(output_path).resolve()),
        "config": {
            "min_prompt_sec": float(min_prompt_sec),
            "max_prompt_sec": float(max_prompt_sec),
            "prefer_prompt_sec": float(prefer_prompt_sec),
        },
        "num_records": 0,
        "num_self_prompt": 0,
        "num_speaker_pool_prompt": 0,
        "num_fixed_reference_prompt": 0,
        "records": [],
        "rejected_prompt_candidates": [],
    }


def _append_selection_report_record(
    report: dict[str, Any],
    *,
    item: ExportItem,
    prompt_candidate: dict[str, Any],
    selection_reason: str,
) -> None:
    """
    EN:
    Append one prompt-selection detail row.

    ZH:
    追加一条 prompt 选择详情。
    """
    target_wav_path = _normalize_path(item.wav_path)
    prompt_wav_path = _normalize_path(prompt_candidate["wav_path"])

    if selection_reason == "self_prompt_valid":
        report["num_self_prompt"] += 1
    elif selection_reason == "selected_from_speaker_pool":
        report["num_speaker_pool_prompt"] += 1
    elif selection_reason == "fixed_reference":
        report["num_fixed_reference_prompt"] += 1

    report["records"].append(
        {
            "sample_id": str(item.sample_id),
            "speaker_name": _speaker_key(item),
            "target_wav_path": target_wav_path,
            "target_duration_sec": (
                float(_duration_from_item_or_audio(item))
                if _is_finite(_duration_from_item_or_audio(item))
                else None
            ),
            "prompt_wav_path": prompt_wav_path,
            "prompt_sample_id": str(prompt_candidate.get("sample_id", "")),
            "prompt_duration_sec": prompt_candidate.get("duration_sec"),
            "selection_reason": selection_reason,
        }
    )


# ============================================================
# Public API
# 对外 API
# ============================================================
def export_stage2_manifest_jsonl(
    items: list[ExportItem],
    output_path: str | Path,
    prompt_mode: str = "self",
    *,
    fixed_prompt_wav_path: str | Path | None = None,
    prompt_selection_report_path: str | Path | None = None,
    min_prompt_sec: float = 3.0,
    max_prompt_sec: float = 10.0,
    prefer_prompt_sec: float = 6.0,
    allow_self_prompt: bool = True,
    strict_prompt_duration: bool = True,
) -> Path:
    """
    EN:
    Export dataset items into VoiceLab Stage2 training manifest format.

    Supported prompt modes:
    - self
        prompt_wav_path = target_wav_path

    - speaker_pool
        If target clip is a valid prompt, use itself.
        Otherwise select a valid prompt clip from the same speaker pool.

    - fixed_reference
        Use one fixed prompt audio for all target samples.

    Output record format:
        {
            "sample_id": "...",
            "raw_text": "...",
            "language": "...",
            "prompt_wav_path": "...",
            "target_wav_path": "..."
        }

    ZH:
    将数据导出成 VoiceLab 第二阶段训练链使用的 manifest 格式。

    支持的 prompt 模式：
    - self
        prompt_wav_path = target_wav_path

    - speaker_pool
        如果当前 target 本身适合作为 prompt，则使用自己；
        否则从同说话人候选池中选择一条合格 prompt。

    - fixed_reference
        所有 target 样本共用同一条固定参考音频作为 prompt。

    输出格式：
        {
            "sample_id": "...",
            "raw_text": "...",
            "language": "...",
            "prompt_wav_path": "...",
            "target_wav_path": "..."
        }
    """
    assert_manifest_items_non_empty(items, name="export_items")

    prompt_mode = _normalize_prompt_mode(prompt_mode)
    output_path = Path(output_path).resolve()

    valid_items = [
        item
        for item in items
        if not is_rejected_text_status(item.text_status)
    ]
    assert_manifest_items_non_empty(valid_items, name="valid_export_items")

    report = _build_report_base(
        prompt_mode=prompt_mode,
        output_path=output_path,
        min_prompt_sec=min_prompt_sec,
        max_prompt_sec=max_prompt_sec,
        prefer_prompt_sec=prefer_prompt_sec,
    )

    records: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Mode 1: self prompt
    # 模式 1：自提示
    # --------------------------------------------------------
    if prompt_mode == "self":
        for item in valid_items:
            records.append(_build_stage2_record_self_prompt(item))

            _append_selection_report_record(
                report,
                item=item,
                prompt_candidate={
                    "sample_id": str(item.sample_id),
                    "speaker_name": _speaker_key(item),
                    "wav_path": _normalize_path(item.wav_path),
                    "duration_sec": (
                        float(_duration_from_item_or_audio(item))
                        if _is_finite(_duration_from_item_or_audio(item))
                        else None
                    ),
                },
                selection_reason="self_prompt_valid",
            )

    # --------------------------------------------------------
    # Mode 2: speaker pool prompt
    # 模式 2：同说话人候选池
    # --------------------------------------------------------
    elif prompt_mode == "speaker_pool":
        pools, rejected_candidates = _build_prompt_candidate_pools(
            valid_items,
            min_prompt_sec=float(min_prompt_sec),
            max_prompt_sec=float(max_prompt_sec),
        )

        report["rejected_prompt_candidates"] = rejected_candidates

        for item in valid_items:
            speaker = _speaker_key(item)
            pool = pools.get(speaker, [])

            prompt_candidate, selection_reason = _select_prompt_from_pool(
                item,
                pool=pool,
                min_prompt_sec=float(min_prompt_sec),
                max_prompt_sec=float(max_prompt_sec),
                prefer_prompt_sec=float(prefer_prompt_sec),
                allow_self_prompt=bool(allow_self_prompt),
            )

            records.append(
                _build_stage2_record(
                    item,
                    prompt_wav_path=prompt_candidate["wav_path"],
                )
            )

            _append_selection_report_record(
                report,
                item=item,
                prompt_candidate=prompt_candidate,
                selection_reason=selection_reason,
            )

    # --------------------------------------------------------
    # Mode 3: fixed reference prompt
    # 模式 3：固定参考音频
    # --------------------------------------------------------
    elif prompt_mode == "fixed_reference":
        fixed_candidate = _validate_fixed_reference(
            fixed_prompt_wav_path,
            min_prompt_sec=float(min_prompt_sec),
            max_prompt_sec=float(max_prompt_sec),
            strict_prompt_duration=bool(strict_prompt_duration),
        )

        for item in valid_items:
            records.append(
                _build_stage2_record(
                    item,
                    prompt_wav_path=fixed_candidate["wav_path"],
                )
            )

            _append_selection_report_record(
                report,
                item=item,
                prompt_candidate=fixed_candidate,
                selection_reason="fixed_reference",
            )

    else:
        raise ValueError(
            f"Unsupported prompt_mode: {prompt_mode}. "
            "Supported: self, speaker_pool, fixed_reference."
        )

    report["num_records"] = int(len(records))

    output = write_jsonl(records, output_path)

    if prompt_selection_report_path is not None:
        _save_json(report, prompt_selection_report_path)

    return output