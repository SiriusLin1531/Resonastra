from __future__ import annotations

"""Reconcile stale DataFactory Runtime State v2 records.

A Gradio page refresh can interrupt the UI event chain after the durable
``running`` state has already been written. The real subprocess may then finish
(or disappear) without the UI-side finalize callback writing a trustworthy
terminal state.

UX-FIX-1 recovery rules:
- keep a real active task active while a matching subprocess is visible;
- recover the refresh-specific synthetic ``failed / missing result`` state;
- if artifacts prove completion, converge to ``succeeded``;
- if neither process nor artifacts can be confirmed, keep a finite recovery
  grace window so the browser Timer can resume checking;
- after the grace window, converge to ``failed / reconciled_interrupted``;
- never rewrite an intentional ``stopped`` or a conclusive terminal result.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .progress import (
    ACTIVE_RUNTIME_STATUSES,
    TERMINAL_RUNTIME_STATUSES,
    _atomic_write_json,
    now_iso,
    runtime_state_path,
)


DEFAULT_STALE_GRACE_SECONDS = 120.0


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _seconds_since(value: Any) -> float | None:
    timestamp = _parse_time(value)
    if timestamp is None:
        return None
    return max(0.0, (datetime.now() - timestamp).total_seconds())


def _stale_elapsed_seconds(state: dict[str, Any]) -> float | None:
    # A refresh-recovered task gets a new finite grace window from the moment
    # the inconclusive finalize was observed, not from the original task start.
    recovery_started_at = state.get("recovery_started_at")
    if recovery_started_at:
        return _seconds_since(recovery_started_at)
    return _seconds_since(state.get("started_at"))


def runtime_elapsed_seconds(state: dict[str, Any] | None) -> float | None:
    """Return elapsed time, freezing terminal tasks at their recorded end time."""

    payload = state if isinstance(state, dict) else {}
    started = _parse_time(payload.get("started_at"))
    if started is None:
        return None

    status = str(payload.get("status") or "").strip().lower()
    if status in TERMINAL_RUNTIME_STATUSES:
        ended = _parse_time(payload.get("finished_at")) or _parse_time(
            payload.get("updated_at")
        )
        if ended is None:
            return None
    else:
        ended = datetime.now()

    return max(0.0, (ended - started).total_seconds())


def _command_matches_work_dir(command_line: str, work_dir: str | Path) -> bool:
    lowered = str(command_line or "").lower()
    resolved = Path(work_dir).expanduser().resolve(strict=False)
    variants = {str(resolved).lower(), resolved.as_posix().lower()}
    return any(value and value in lowered for value in variants)


def has_matching_process(
    process_infos: Iterable[dict[str, Any]] | None,
    *,
    work_dir: str | Path,
) -> bool:
    for item in process_infos or []:
        if not isinstance(item, dict):
            continue
        if _command_matches_work_dir(str(item.get("CommandLine") or ""), work_dir):
            return True
    return False


def _stage_map(status_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in status_payload.get("stages") or []:
        if isinstance(item, dict) and item.get("name"):
            result[str(item["name"])] = item
    return result


def _stage_done(stage: dict[str, Any] | None) -> bool:
    if not isinstance(stage, dict):
        return False
    status = str(stage.get("status") or "").strip().lower()
    if status in {"succeeded", "success", "done", "ready"}:
        return True
    return bool(stage.get("done"))


def operation_completed(
    operation: str,
    status_payload: dict[str, Any] | None,
) -> bool:
    payload = status_payload if isinstance(status_payload, dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    stages = _stage_map(payload)
    op = str(operation or "").strip().lower()

    if op == "prepare":
        return bool(summary.get("prepare_done")) or _stage_done(stages.get("prepare"))
    if op == "proofread_rebuild":
        return bool(summary.get("proofread_done")) or _stage_done(stages.get("proofread"))
    if op == "stage1":
        return bool(summary.get("stage1_done")) or _stage_done(stages.get("stage1_dataset"))
    if op in {"stage2_generate", "stage2_resume"}:
        return bool(summary.get("all_stage2_done"))
    return False


def _is_refresh_missing_result_failure(state: dict[str, Any]) -> bool:
    if str(state.get("status") or "").strip().lower() != "failed":
        return False

    result_status = str(state.get("result_status") or "").strip().lower()
    if result_status not in {"", "missing"}:
        return False

    message = str(state.get("message") or "")
    return (
        "无法识别任务结果状态" in message
        or "未返回可识别的结果状态" in message
    )


def _write_repaired(work_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    _atomic_write_json(runtime_state_path(work_dir), payload)
    return payload


def _recover_as_running(
    state: dict[str, Any],
    *,
    work_dir: str | Path,
    reason: str,
) -> dict[str, Any]:
    timestamp = now_iso()
    recovery_started_at = (
        state.get("recovery_started_at")
        or state.get("finished_at")
        or state.get("updated_at")
        or timestamp
    )
    repaired = {
        **state,
        "status": "running",
        "updated_at": timestamp,
        "finished_at": None,
        "message": "页面刷新后正在恢复任务状态；将继续根据实际进程与产物自动确认结果。",
        "result_status": None,
        "reconciled": True,
        "reconcile_reason": reason,
        "recovery_started_at": recovery_started_at,
    }
    return _write_repaired(work_dir, repaired)


def _repair_succeeded(
    state: dict[str, Any],
    *,
    work_dir: str | Path,
) -> dict[str, Any]:
    timestamp = now_iso()
    repaired = {
        **state,
        "status": "succeeded",
        "updated_at": timestamp,
        # A prior refresh-disconnected finalize may have written a false early
        # finished_at. Reconciliation owns the trustworthy terminal decision,
        # so its timestamp becomes the terminal time.
        "finished_at": timestamp,
        "message": "任务产物已完成；页面恢复时已自动修正遗留运行状态。",
        "result_status": "reconciled_succeeded",
        "reconciled": True,
        "reconcile_reason": "artifacts_complete_without_matching_process",
    }
    return _write_repaired(work_dir, repaired)


def _repair_interrupted(
    state: dict[str, Any],
    *,
    work_dir: str | Path,
) -> dict[str, Any]:
    timestamp = now_iso()
    repaired = {
        **state,
        "status": "failed",
        "updated_at": timestamp,
        "finished_at": timestamp,
        "message": "任务已失去运行进程且产物未完成；可能因页面刷新、WebUI 重启或进程异常退出而中断。",
        "result_status": "reconciled_interrupted",
        "reconciled": True,
        "reconcile_reason": "stale_running_without_process",
    }
    return _write_repaired(work_dir, repaired)


def reconcile_runtime_state(
    *,
    work_dir: str | Path,
    runtime_state: dict[str, Any] | None,
    status_payload: dict[str, Any] | None,
    process_infos: Iterable[dict[str, Any]] | None,
    stale_grace_seconds: float = DEFAULT_STALE_GRACE_SECONDS,
) -> tuple[dict[str, Any], bool]:
    """Return ``(state, changed)`` and persist a repaired lifecycle when needed."""

    state = dict(runtime_state or {})
    status = str(state.get("status") or "").strip().lower()
    refresh_missing_failure = _is_refresh_missing_result_failure(state)

    if status not in ACTIVE_RUNTIME_STATUSES and not refresh_missing_failure:
        return state, False

    if has_matching_process(process_infos, work_dir=work_dir):
        if refresh_missing_failure:
            repaired = _recover_as_running(
                state,
                work_dir=work_dir,
                reason="refresh_missing_result_with_matching_process",
            )
            return repaired, True
        return state, False

    operation = str(state.get("operation") or "").strip()
    if operation_completed(operation, status_payload):
        return _repair_succeeded(state, work_dir=work_dir), True

    grace = max(0.0, float(stale_grace_seconds))

    if refresh_missing_failure:
        # An inconclusive finalize produced by a disconnected/refreshed page is
        # not proof that the backend failed. Re-open a finite recovery window so
        # Gradio can reactivate its Timer and keep polling for process/artifacts.
        recovery_age = _seconds_since(
            state.get("recovery_started_at")
            or state.get("finished_at")
            or state.get("updated_at")
        )
        if recovery_age is None or recovery_age < grace:
            repaired = _recover_as_running(
                state,
                work_dir=work_dir,
                reason="refresh_missing_result_recovery_grace",
            )
            return repaired, True
        return _repair_interrupted(state, work_dir=work_dir), True

    elapsed = _stale_elapsed_seconds(state)
    if elapsed is None or elapsed < grace:
        return state, False

    return _repair_interrupted(state, work_dir=work_dir), True


__all__ = [
    "DEFAULT_STALE_GRACE_SECONDS",
    "has_matching_process",
    "operation_completed",
    "runtime_elapsed_seconds",
    "reconcile_runtime_state",
]
