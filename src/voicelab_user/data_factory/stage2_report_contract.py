from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SPLIT_FILTER_CONTRACT_VERSION = "voicelab_stage2_split_filter_contract_v1"
FILTER_REPORT_SCHEMA_VERSION = "voicelab_stage2_filter_report_v2"


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _count_pt_files(path: Path | None) -> int:
    if path is None or not Path(path).is_dir():
        return 0
    return sum(1 for item in Path(path).glob("*.pt") if item.is_file())


def _as_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _normalized_path(value: Any) -> str:
    if value is None or not str(value).strip():
        return ""
    return str(Path(str(value)).expanduser().resolve(strict=False)).casefold()


def _same_path(left: Any, right: Any) -> bool:
    return bool(_normalized_path(left)) and _normalized_path(left) == _normalized_path(right)


def _same_path_set(left: Iterable[Any], right: Iterable[Any]) -> bool:
    return {
        _normalized_path(item)
        for item in left
        if _normalized_path(item)
    } == {
        _normalized_path(item)
        for item in right
        if _normalized_path(item)
    }


def _mtime(path: Path | None) -> float | None:
    if path is None or not Path(path).exists():
        return None
    return float(Path(path).stat().st_mtime)


def _completion_ratio(found: int, expected: int | None) -> float | None:
    if expected is None or expected <= 0:
        return None
    return min(float(found) / float(expected), 1.0)


@dataclass(frozen=True, slots=True)
class Stage2SplitFilterState:
    contract_version: str
    source_count: int | None
    usable_count: int | None
    found_count: int
    train_count: int
    val_count: int
    rejected_reported_count: int
    rejected_existing_count: int
    style_cache_count: int
    completion_ratio: float | None
    split_report_valid: bool
    filter_report_valid: bool
    split_done: bool
    filter_done: bool
    completed: bool
    split_mode: str
    filter_mode: str
    requested_filter_mode: str | None
    split_report_freshness_ok: bool
    count_coherence_ok: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_stage2_split_filter_state(
    artifacts: Any,
    *,
    requested_filter_mode: str | None = None,
) -> Stage2SplitFilterState:
    """Validate Stage2 split/filter reports against current filesystem state.

    The pipeline order remains split -> filter. In move/delete filter modes the
    train/val directories intentionally contain fewer files after quarantine,
    so completion must use the post-filter usable count rather than the original
    style-cache source count.
    """

    style_count = _count_pt_files(artifacts.style_cache_root)
    train_count = _count_pt_files(artifacts.train_pt_root)
    val_count = _count_pt_files(artifacts.val_pt_root)
    found_count = train_count + val_count

    split_report_path = Path(artifacts.split_report_path)
    filter_report_path = Path(artifacts.filter_report_json_path)
    split_report = _read_json_dict(split_report_path)
    filter_report = _read_json_dict(filter_report_path)
    reasons: list[str] = []

    split_total = _as_nonnegative_int(split_report.get("total_files"))
    split_train = _as_nonnegative_int(split_report.get("train_files"))
    split_val = _as_nonnegative_int(split_report.get("val_files"))
    split_mode = str(split_report.get("mode") or "")

    split_paths_ok = (
        _same_path(split_report.get("data_root"), artifacts.style_cache_root)
        and _same_path(split_report.get("train_out"), artifacts.train_pt_root)
        and _same_path(split_report.get("val_out"), artifacts.val_pt_root)
    )
    split_counts_ok = bool(
        split_total is not None
        and split_total > 0
        and split_train is not None
        and split_val is not None
        and split_train + split_val == split_total
    )
    split_report_valid = bool(
        split_report_path.is_file()
        and split_paths_ok
        and split_counts_ok
        and split_mode in {"copy", "move", "hardlink"}
    )

    if split_report_path.is_file() and not split_paths_ok:
        reasons.append("split_report_paths_mismatch")
    if split_report_path.is_file() and not split_counts_ok:
        reasons.append("split_report_counts_incoherent")
    if split_report_path.is_file() and split_mode not in {"copy", "move", "hardlink"}:
        reasons.append("split_report_mode_invalid")

    source_count = split_total if split_report_valid else (style_count or None)

    filter_mode = str(
        filter_report.get("mode")
        or filter_report.get("filter_mode")
        or ""
    )
    filter_total = _as_nonnegative_int(filter_report.get("num_total"))
    filter_ok = _as_nonnegative_int(filter_report.get("num_ok"))
    filter_bad = _as_nonnegative_int(filter_report.get("num_bad"))
    filter_roots = filter_report.get("roots")
    if not isinstance(filter_roots, list):
        filter_roots = []

    mode_ok = filter_mode in {"move", "copy", "delete", "report_only"}
    if requested_filter_mode is not None:
        mode_ok = mode_ok and filter_mode == str(requested_filter_mode)

    roots_ok = _same_path_set(
        filter_roots,
        [artifacts.train_pt_root, artifacts.val_pt_root],
    )
    filter_counts_ok = bool(
        filter_total is not None
        and filter_ok is not None
        and filter_bad is not None
        and filter_ok + filter_bad == filter_total
        and (source_count is None or filter_total == source_count)
    )

    split_mtime = _mtime(split_report_path)
    filter_mtime = _mtime(filter_report_path)
    split_report_freshness_ok = bool(
        split_mtime is not None
        and filter_mtime is not None
        and filter_mtime >= split_mtime
    )

    bad_samples = filter_report.get("bad_samples")
    if not isinstance(bad_samples, list):
        bad_samples = []
    rejected_existing_count = 0
    for row in bad_samples:
        if not isinstance(row, dict):
            continue
        rejected_path = row.get("rejected_path")
        if rejected_path and Path(str(rejected_path)).is_file():
            rejected_existing_count += 1

    rejected_reported_count = filter_bad or 0

    post_filter_counts_ok = False
    if filter_counts_ok:
        if filter_mode in {"move", "delete"}:
            post_filter_counts_ok = found_count == int(filter_ok or 0)
        else:
            post_filter_counts_ok = found_count == int(filter_total or 0)

        if filter_mode in {"move", "copy"} and rejected_reported_count > 0:
            post_filter_counts_ok = (
                post_filter_counts_ok
                and rejected_existing_count >= rejected_reported_count
            )

    count_coherence_ok = bool(
        source_count is not None
        and found_count + rejected_reported_count >= source_count
    )

    filter_report_valid = bool(
        filter_report_path.is_file()
        and split_report_valid
        and mode_ok
        and roots_ok
        and filter_counts_ok
        and split_report_freshness_ok
        and post_filter_counts_ok
        and count_coherence_ok
    )

    if filter_report_path.is_file() and not mode_ok:
        reasons.append("filter_report_mode_mismatch")
    if filter_report_path.is_file() and not roots_ok:
        reasons.append("filter_report_roots_mismatch")
    if filter_report_path.is_file() and not filter_counts_ok:
        reasons.append("filter_report_counts_incoherent")
    if filter_report_path.is_file() and not split_report_freshness_ok:
        reasons.append("filter_report_older_than_split_report")
    if filter_report_path.is_file() and not post_filter_counts_ok:
        reasons.append("filter_report_filesystem_counts_mismatch")
    if filter_report_path.is_file() and not count_coherence_ok:
        reasons.append("train_val_rejected_less_than_source")

    if filter_report_valid and filter_mode in {"move", "delete"}:
        usable_count = filter_ok
    else:
        usable_count = source_count

    split_done = bool(
        split_report_valid
        and usable_count is not None
        and usable_count > 0
        and found_count >= usable_count
    )
    filter_done = filter_report_valid
    completed = bool(split_done and filter_done)

    return Stage2SplitFilterState(
        contract_version=SPLIT_FILTER_CONTRACT_VERSION,
        source_count=source_count,
        usable_count=usable_count,
        found_count=found_count,
        train_count=train_count,
        val_count=val_count,
        rejected_reported_count=rejected_reported_count,
        rejected_existing_count=rejected_existing_count,
        style_cache_count=style_count,
        completion_ratio=_completion_ratio(found_count, usable_count),
        split_report_valid=split_report_valid,
        filter_report_valid=filter_report_valid,
        split_done=split_done,
        filter_done=filter_done,
        completed=completed,
        split_mode=split_mode,
        filter_mode=filter_mode,
        requested_filter_mode=requested_filter_mode,
        split_report_freshness_ok=split_report_freshness_ok,
        count_coherence_ok=count_coherence_ok,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def invalidate_filter_reports(
    artifacts: Any,
    *,
    reason: str,
) -> list[str]:
    """Rename existing filter reports so a subsequent run must regenerate them."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved: list[str] = []
    for path_value in (
        artifacts.filter_report_json_path,
        artifacts.filter_report_csv_path,
    ):
        path = Path(path_value)
        if not path.exists():
            continue
        stale_path = path.with_name(
            f"{path.stem}.stale_{timestamp}{path.suffix}"
        )
        path.replace(stale_path)
        moved.append(str(stale_path))

    if moved:
        marker = Path(artifacts.rejected_pt_root) / "filter_report_invalidation.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": "voicelab_filter_report_invalidation_v1",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "reason": str(reason),
                    "moved_reports": moved,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return moved


__all__ = [
    "FILTER_REPORT_SCHEMA_VERSION",
    "SPLIT_FILTER_CONTRACT_VERSION",
    "Stage2SplitFilterState",
    "analyze_stage2_split_filter_state",
    "invalidate_filter_reports",
]
