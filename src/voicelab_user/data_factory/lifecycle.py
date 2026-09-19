from __future__ import annotations

"""Thin callback lifecycle wrappers for DataFactory UI v4.

This module deliberately does not implement any DataFactory business logic.
It resolves the active work directory, writes Runtime State v2 before/after a
stable callback, and delegates the actual operation to the existing legacy/v2
callbacks.

The wrappers preserve the existing Gradio input/output protocol. A crucial
race rule is also enforced: when the user stops a running child process, the
blocked callback may later return a failure caused by that termination. In
that case an already-written ``stopped`` Runtime State v2 must win and must not
be overwritten as ``failed``.
"""

from pathlib import Path
from typing import Any, Callable, Sequence

from .progress import (
    begin_runtime_task,
    finish_runtime_task,
    mark_runtime_stopped,
    read_runtime_state,
)


Callback = Callable[..., Any]
WorkDirResolver = Callable[[str, str, str], Path]


def _developer_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        value = result[1]
        if isinstance(value, dict):
            return value
    if isinstance(result, dict):
        return result
    return {}


def _runtime_already_stopped(work_dir: str | Path) -> bool:
    state = read_runtime_state(work_dir)
    return str(state.get("status") or "").strip().lower() == "stopped"


def _finish_if_not_stopped(
    work_dir: str | Path,
    *,
    operation: str,
    result_payload: Any,
) -> dict[str, Any]:
    if _runtime_already_stopped(work_dir):
        return read_runtime_state(work_dir)
    return finish_runtime_task(
        work_dir,
        operation=operation,
        result_payload=result_payload,
    )


def run_callback_with_lifecycle(
    callback: Callback,
    callback_args: Sequence[Any],
    *,
    work_dir: str | Path,
    operation: str,
) -> Any:
    """Run one stable callback with additive Runtime State v2 lifecycle."""

    active_work_dir = Path(work_dir).expanduser().resolve(strict=False)
    begin_runtime_task(active_work_dir, operation=operation)

    try:
        result = callback(*callback_args)
    except BaseException as exc:
        _finish_if_not_stopped(
            active_work_dir,
            operation=operation,
            result_payload={
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    _finish_if_not_stopped(
        active_work_dir,
        operation=operation,
        result_payload=_developer_payload(result),
    )
    return result


def run_stop_with_lifecycle(
    callback: Callback,
    callback_args: Sequence[Any],
    *,
    work_dir: str | Path,
) -> Any:
    """Delegate stable Stop and then synchronize Runtime State v2."""

    active_work_dir = Path(work_dir).expanduser().resolve(strict=False)
    previous_state = read_runtime_state(active_work_dir)
    result = callback(*callback_args)
    mark_runtime_stopped(
        active_work_dir,
        previous_state=previous_state,
        stop_payload=_developer_payload(result),
    )
    return result


def resolve_callback_work_dir(
    resolver: WorkDirResolver,
    *,
    work_dir_output: Any,
    work_dir: Any,
    speaker_name: Any,
) -> Path:
    return resolver(
        str(work_dir_output or ""),
        str(work_dir or ""),
        str(speaker_name or ""),
    )


def prepare_lifecycle_click(
    callback: Callback,
    resolver: WorkDirResolver,
    *args: Any,
) -> Any:
    # Prepare callback contract:
    # speaker_name -> index 5, work_dir -> index 7.
    if len(args) < 8:
        raise ValueError("Prepare lifecycle wrapper received an incomplete callback contract.")
    active_work_dir = resolve_callback_work_dir(
        resolver,
        work_dir_output="",
        work_dir=args[7],
        speaker_name=args[5],
    )
    return run_callback_with_lifecycle(
        callback,
        args,
        work_dir=active_work_dir,
        operation="prepare",
    )


def proofread_rebuild_lifecycle_click(
    callback: Callback,
    resolver: WorkDirResolver,
    *args: Any,
) -> Any:
    # Shared work-dir callback contract starts with output/workdir/speaker.
    if len(args) < 3:
        raise ValueError("Proofread lifecycle wrapper received an incomplete callback contract.")
    active_work_dir = resolve_callback_work_dir(
        resolver,
        work_dir_output=args[0],
        work_dir=args[1],
        speaker_name=args[2],
    )
    return run_callback_with_lifecycle(
        callback,
        args,
        work_dir=active_work_dir,
        operation="proofread_rebuild",
    )


def stage1_lifecycle_click(
    callback: Callback,
    resolver: WorkDirResolver,
    *args: Any,
) -> Any:
    if len(args) < 3:
        raise ValueError("Stage1 lifecycle wrapper received an incomplete callback contract.")
    active_work_dir = resolve_callback_work_dir(
        resolver,
        work_dir_output=args[0],
        work_dir=args[1],
        speaker_name=args[2],
    )
    return run_callback_with_lifecycle(
        callback,
        args,
        work_dir=active_work_dir,
        operation="stage1",
    )


def stage2_generate_lifecycle_click(
    callback: Callback,
    resolver: WorkDirResolver,
    *args: Any,
) -> Any:
    if len(args) < 3:
        raise ValueError("Stage2 lifecycle wrapper received an incomplete callback contract.")
    active_work_dir = resolve_callback_work_dir(
        resolver,
        work_dir_output=args[0],
        work_dir=args[1],
        speaker_name=args[2],
    )
    return run_callback_with_lifecycle(
        callback,
        args,
        work_dir=active_work_dir,
        operation="stage2_generate",
    )


def stage2_resume_lifecycle_click(
    callback: Callback,
    resolver: WorkDirResolver,
    *args: Any,
) -> Any:
    if len(args) < 3:
        raise ValueError("Stage2 resume lifecycle wrapper received an incomplete callback contract.")
    active_work_dir = resolve_callback_work_dir(
        resolver,
        work_dir_output=args[0],
        work_dir=args[1],
        speaker_name=args[2],
    )
    return run_callback_with_lifecycle(
        callback,
        args,
        work_dir=active_work_dir,
        operation="stage2_resume",
    )


def stop_lifecycle_click(
    callback: Callback,
    resolver: WorkDirResolver,
    *args: Any,
) -> Any:
    if len(args) < 3:
        raise ValueError("Stop lifecycle wrapper received an incomplete callback contract.")
    active_work_dir = resolve_callback_work_dir(
        resolver,
        work_dir_output=args[0],
        work_dir=args[1],
        speaker_name=args[2],
    )
    return run_stop_with_lifecycle(
        callback,
        args,
        work_dir=active_work_dir,
    )


__all__ = [
    "run_callback_with_lifecycle",
    "run_stop_with_lifecycle",
    "resolve_callback_work_dir",
    "prepare_lifecycle_click",
    "proofread_rebuild_lifecycle_click",
    "stage1_lifecycle_click",
    "stage2_generate_lifecycle_click",
    "stage2_resume_lifecycle_click",
    "stop_lifecycle_click",
]
