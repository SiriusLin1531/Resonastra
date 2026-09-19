from __future__ import annotations

"""UI-independent orchestration for the DataFactory live monitor.

The Gradio page owns Timer components and event chaining. This module owns only
the monitor data flow:

Runtime State v2 + quarantine-aware artifact scan + optional process snapshot
    -> stale-state reconciliation
    -> current-task markdown
    -> current-step progress HTML
    -> Stage2 six-step live HTML
    -> diagnostic payload
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .progress import (
    LiveProgressSnapshot,
    begin_runtime_task,
    build_live_monitor_payload,
    build_live_progress_snapshot,
    finish_runtime_task,
    format_current_task_markdown,
    format_live_progress_html,
    read_runtime_state,
    runtime_is_active,
)
from .runtime_reconcile import reconcile_runtime_state, runtime_elapsed_seconds
from .stage2_live_steps import (
    Stage2LiveStepsSnapshot,
    build_stage2_live_steps_snapshot,
    format_stage2_live_steps_html,
)
from .status_manager_v2 import scan_unified_data_factory_status_v2


@dataclass(frozen=True)
class DataFactoryLiveMonitorView:
    work_dir: str
    runtime_state: dict[str, Any]
    status_payload: dict[str, Any]
    progress: LiveProgressSnapshot
    stage2_steps: Stage2LiveStepsSnapshot
    current_task_markdown: str
    progress_html: str
    stage2_steps_html: str
    diagnostic_payload: dict[str, Any]
    monitor_active: bool


def _resolved(work_dir: str | Path) -> Path:
    return Path(work_dir).expanduser().resolve(strict=False)


def _process_list(process_infos: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [item for item in (process_infos or []) if isinstance(item, dict)]


def build_live_monitor_view(
    work_dir: str | Path,
    *,
    process_infos: Iterable[dict[str, Any]] | None = None,
    status_payload: dict[str, Any] | None = None,
) -> DataFactoryLiveMonitorView:
    """Build one monitor refresh and repair stale active Runtime State when safe."""

    active_work_dir = _resolved(work_dir)
    runtime_state = read_runtime_state(active_work_dir)
    scanned_status = (
        status_payload
        if isinstance(status_payload, dict)
        else scan_unified_data_factory_status_v2(
            active_work_dir,
            write_status_json=False,
        )
    )
    processes = _process_list(process_infos)

    runtime_state, reconciled = reconcile_runtime_state(
        work_dir=active_work_dir,
        runtime_state=runtime_state,
        status_payload=scanned_status,
        process_infos=processes,
    )

    progress = build_live_progress_snapshot(
        work_dir=active_work_dir,
        runtime_state=runtime_state,
        status_payload=scanned_status,
        process_infos=processes,
    )
    # progress.py remains the stable DF-UI-3 module. UX-FIX-1 freezes elapsed
    # time here instead of rewriting that large core module: active states use
    # now-started_at, terminal states use finished_at-started_at.
    progress = replace(
        progress,
        elapsed_seconds=runtime_elapsed_seconds(runtime_state),
    )
    stage2_steps = build_stage2_live_steps_snapshot(scanned_status, progress)

    diagnostic_payload = build_live_monitor_payload(
        work_dir=active_work_dir,
        runtime_state=runtime_state,
        status_payload=scanned_status,
        process_infos=processes,
    )
    # Keep diagnostic JSON consistent with the user-facing frozen snapshot.
    diagnostic_payload["progress"] = progress.to_dict()
    diagnostic_payload["process_snapshot"] = {
        "num_processes": len(processes),
        "matched_process_detected": bool(progress.process_detected),
        "matched_script": progress.process_script or None,
    }
    diagnostic_payload["stage2_live_steps"] = stage2_steps.to_dict()
    diagnostic_payload["runtime_reconcile"] = {
        "changed": bool(reconciled),
        "reconciled": bool(runtime_state.get("reconciled")),
        "reason": runtime_state.get("reconcile_reason"),
    }

    return DataFactoryLiveMonitorView(
        work_dir=str(active_work_dir),
        runtime_state=runtime_state,
        status_payload=scanned_status,
        progress=progress,
        stage2_steps=stage2_steps,
        current_task_markdown=format_current_task_markdown(progress),
        progress_html=format_live_progress_html(progress),
        stage2_steps_html=format_stage2_live_steps_html(stage2_steps),
        diagnostic_payload=diagnostic_payload,
        monitor_active=runtime_is_active(runtime_state),
    )


def begin_live_task(
    work_dir: str | Path,
    *,
    operation: str,
    process_infos: Iterable[dict[str, Any]] | None = None,
) -> DataFactoryLiveMonitorView:
    active_work_dir = _resolved(work_dir)
    begin_runtime_task(active_work_dir, operation=operation)
    return build_live_monitor_view(active_work_dir, process_infos=process_infos)


def finish_live_task(
    work_dir: str | Path,
    *,
    operation: str,
    result_payload: Any,
    process_infos: Iterable[dict[str, Any]] | None = None,
) -> DataFactoryLiveMonitorView:
    active_work_dir = _resolved(work_dir)
    current = read_runtime_state(active_work_dir)
    if str(current.get("status") or "").strip().lower() != "stopped":
        finish_runtime_task(
            active_work_dir,
            operation=operation,
            result_payload=result_payload,
        )
    return build_live_monitor_view(active_work_dir, process_infos=process_infos)


__all__ = [
    "DataFactoryLiveMonitorView",
    "build_live_monitor_view",
    "begin_live_task",
    "finish_live_task",
]
