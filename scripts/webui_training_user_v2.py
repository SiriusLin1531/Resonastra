from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import webui_training_user as legacy
from src.voicelab_user.training import (
    delete_checkpoint,
    generate_user_profile,
    mark_best_checkpoint,
    restore_automatic_best,
    scan_training_outputs,
)
from src.voicelab_user.training.data_contract import (
    validate_stage1_ready,
    validate_stage2_ready,
)
from src.voicelab_user.training.experience import (
    extract_training_error_excerpt,
    format_checkpoint_visual_html,
    format_failure_summary_markdown,
    format_latest_run_markdown,
    select_latest_run,
    update_terminal_notification_state,
)
from src.voicelab_user.training.progress import (
    format_gpu_markdown,
    query_gpu_snapshot,
)
from src.voicelab_user.training.service import (
    DEFAULT_STAGE2_BASE_CKPT_RELATIVE,
    create_training_service,
)


TERMINAL_STATUSES = {"succeeded", "failed", "stopped", "stop_failed"}
ACTIVE_MONITOR_STATUSES = {"pending", "running", "stopping"}
_LEGACY_ACTIVE_DEFAULTS_PAYLOAD = legacy._active_defaults_payload
LIVE_MONITOR_INTERVAL_SECONDS = 5.0


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


def _read_json_dict(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _training_ui_defaults() -> dict[str, dict[str, int]]:
    service = create_training_service()
    return {
        "stage1": {
            "epochs": int(service.stage1_defaults.epochs),
            "batch_size": int(service.stage1_defaults.batch_size),
        },
        "stage2": {
            "epochs": int(service.stage2_defaults.epochs),
            "batch_size": int(service.stage2_defaults.batch_size),
        },
    }


def _active_defaults_payload_v2() -> dict[str, Any]:
    payload = _LEGACY_ACTIVE_DEFAULTS_PAYLOAD()
    payload["schema_version"] = "voicelab_training_ui_defaults_view_v2"
    policy = payload.get("user_mode_policy")
    if not isinstance(policy, dict):
        policy = {}
        payload["user_mode_policy"] = policy
    policy.update(
        {
            "editable_in_user_ui": True,
            "editable_fields": {
                "stage1": ["epochs", "batch_size"],
                "stage2": ["epochs", "batch_size"],
            },
            "override_scope": "this_run_only",
            "yaml_defaults_are_modified": False,
            "base_ckpt_editable": False,
            "terminal_default_open": False,
            "reason": (
                "用户版仅开放 epochs / batch_size 作为单次 run override；"
                "其余训练参数继续使用 YAML 默认值。"
            ),
        }
    )
    return payload


def _format_parameter_summary_v2(
    defaults_payload: dict[str, Any] | None = None,
) -> str:
    """Keep legacy 14-output payload compatible; this value is diagnostic-only in v2 UI."""
    payload = defaults_payload or _active_defaults_payload_v2()
    defaults = payload.get("active_defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    return json.dumps(defaults, ensure_ascii=False, default=str)


def _friendly_problem(problem: Any) -> str:
    text = str(problem or "").strip()
    if not text:
        return "数据尚未准备完成"

    lowered = text.lower()
    if "stage1 数据未完成" in lowered:
        return "Stage1 数据尚未准备完成"
    if "stage2 数据未完成" in lowered:
        return "Stage2 数据尚未准备完成"
    if "train/val split" in lowered or "split 未通过" in lowered:
        return "Stage2 训练/验证划分尚未通过完整性校验"
    if "过滤报告" in text or "filter" in lowered:
        return "Stage2 异常样本过滤尚未通过完整性校验"
    if "缺少路径" in text or "缺少路径组" in text:
        return "所需训练数据产物不完整"
    if "split/filter contract" in lowered:
        return "Stage2 数据契约尚未满足训练要求"
    return text


def _dedupe_problems(problems: list[Any]) -> list[str]:
    result: list[str] = []
    for problem in problems:
        friendly = _friendly_problem(problem)
        if friendly and friendly not in result:
            result.append(friendly)
    return result


def _format_data_readiness_panel(contract: dict[str, Any]) -> str:
    readiness = contract.get("readiness")
    if not isinstance(readiness, dict):
        readiness = {}

    stage1_ready = bool(readiness.get("stage1_ready"))
    stage2_ready = bool(readiness.get("stage2_ready"))
    all_ready = bool(readiness.get("all_training_data_ready"))

    return "\n".join(
        [
            "### 数据准备",
            "",
            f"- Stage1 数据：{'✅ 已就绪' if stage1_ready else '⬜ 未就绪'}",
            f"- Stage2 数据：{'✅ 已就绪' if stage2_ready else '⬜ 未就绪'}",
            f"- 完整训练数据：{'✅ 已就绪' if all_ready else '⬜ 尚未全部就绪'}",
        ]
    )


def _format_training_entry_panel(contract: dict[str, Any]) -> str:
    stage1_ok, stage1_problems = validate_stage1_ready(contract)
    stage2_ok, stage2_problems = validate_stage2_ready(contract)

    lines = [
        "### 训练入口",
        "",
        f"- Stage1 Training：{'✅ 可以启动' if stage1_ok else '❌ 暂不可启动'}",
        f"- Stage2 Training：{'✅ 可以启动' if stage2_ok else '❌ 暂不可启动'}",
    ]

    stage1_friendly = _dedupe_problems(list(stage1_problems))
    stage2_friendly = _dedupe_problems(list(stage2_problems))
    if stage1_friendly:
        lines.extend(["", "**Stage1：** " + "；".join(stage1_friendly)])
    if stage2_friendly:
        lines.extend(["", "**Stage2：** " + "；".join(stage2_friendly)])
    return "\n".join(lines)


# Keep legacy callback protocol, but replace verbose user-facing formatter outputs.
legacy._active_defaults_payload = _active_defaults_payload_v2
legacy._format_parameter_summary = _format_parameter_summary_v2
legacy._format_training_ready_panel = _format_data_readiness_panel
legacy._format_validation_summary = _format_training_entry_panel


def _write_training_state_copies(
    state: dict[str, Any],
    *,
    stage_state_path: str | Path,
) -> list[str]:
    written: list[str] = []
    targets = [Path(stage_state_path).expanduser().resolve(strict=False)]
    run_state_text = str(state.get("run_state_path") or "").strip()
    if run_state_text:
        targets.append(Path(run_state_text).expanduser().resolve(strict=False))

    seen: set[str] = set()
    for target in targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        _atomic_write_json(target, state)
        written.append(key)
    return written


def _refresh_panels(
    work_dir: str | Path,
    *,
    refresh_status: bool,
):
    return legacy._scan_contract_and_panels(work_dir, refresh_status=refresh_status)


def _success_outputs(
    *,
    status_md: str,
    work_dir: Path,
    refresh_status: bool,
    contract_update_key: str | None = None,
    contract_update_value: Any = None,
    checkpoint_update_key: str | None = None,
    checkpoint_update_value: Any = None,
    profile_value: dict[str, Any] | None = None,
):
    (
        contract,
        panel_md,
        validation_md,
        parameter_md,
        run_status_md,
        log_md,
        defaults_payload,
        checkpoint_md,
        stage1_update,
        stage2_update,
        checkpoint_json,
        scanned_profile_json,
    ) = _refresh_panels(work_dir, refresh_status=refresh_status)

    if contract_update_key:
        contract[contract_update_key] = contract_update_value
    if checkpoint_update_key:
        checkpoint_json[checkpoint_update_key] = checkpoint_update_value

    return (
        status_md,
        contract,
        panel_md,
        validation_md,
        parameter_md,
        run_status_md,
        log_md,
        checkpoint_md,
        stage1_update,
        stage2_update,
        checkpoint_json,
        profile_value if profile_value is not None else scanned_profile_json,
        str(work_dir),
        defaults_payload,
    )


def _empty_progress_snapshot(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_started",
        "current_epoch": None,
        "total_epochs": None,
        "completed_epochs": 0,
        "current_step": None,
        "total_steps": None,
        "global_step": None,
        "phase": "waiting",
        "percent": 0.0,
        "latest_epoch_line": "",
        "source_log_path": "",
    }


def _read_monitor_stage_states(active_work_dir: str | Path) -> dict[str, dict[str, Any]]:
    stage_states: dict[str, dict[str, Any]] = {}
    for stage in ("stage1", "stage2"):
        stage_state_path = legacy._stage_run_state_path(active_work_dir, stage)
        state = _read_json_dict(stage_state_path)
        if state:
            state.setdefault("run_state_path", str(stage_state_path))
        stage_states[stage] = state
    return stage_states


def _monitor_should_run(stage_states: dict[str, dict[str, Any]]) -> bool:
    for state in stage_states.values():
        status = str(state.get("status") or "").strip().lower()
        if status in ACTIVE_MONITOR_STATUSES:
            return True
    return False


def _new_terminal_transition(
    stage_states: dict[str, dict[str, Any]],
    notification_state: dict[str, Any] | None,
) -> bool:
    previous = dict(notification_state or {})
    if not bool(previous.get("initialized")):
        return False

    seen_raw = previous.get("seen_terminal_tokens")
    seen = {
        str(item)
        for item in seen_raw
        if str(item)
    } if isinstance(seen_raw, list) else set()

    for stage in ("stage1", "stage2"):
        state = stage_states.get(stage) or {}
        run_id = str(state.get("run_id") or "").strip()
        status = str(state.get("status") or "unknown").strip().lower()
        if not run_id or status not in TERMINAL_STATUSES:
            continue
        if f"{stage}:{run_id}:{status}" not in seen:
            return True
    return False


def _timer_component_update(active: bool):
    timer_type = getattr(gr, "Timer", None)
    if timer_type is None:
        return None
    return timer_type(
        value=LIVE_MONITOR_INTERVAL_SECONDS,
        active=bool(active),
    )


def sync_live_monitor_timer(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        stage_states = _read_monitor_stage_states(active_work_dir)
        return _timer_component_update(_monitor_should_run(stage_states))
    except Exception:
        return _timer_component_update(False)


def _live_progress_bar_html(
    stage_label: str,
    view: dict[str, Any],
    snapshot: dict[str, Any],
) -> str:
    try:
        percent = float(snapshot.get("percent") or 0.0)
    except (TypeError, ValueError):
        percent = 0.0
    percent = max(0.0, min(percent, 100.0))

    status = html.escape(str(view.get("status") or snapshot.get("status") or "not_started"))
    phase = html.escape(str(snapshot.get("phase") or "waiting"))
    run_id = html.escape(str(view.get("run_id") or "-"))
    current_epoch = snapshot.get("current_epoch")
    total_epochs = snapshot.get("total_epochs")
    epoch_text = "-"
    if current_epoch is not None:
        epoch_text = str(current_epoch)
        if total_epochs is not None:
            epoch_text += f" / {total_epochs}"

    return "".join(
        [
            '<div style="margin:8px 0 14px 0;">',
            '<div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:5px;">',
            f'<strong>{html.escape(stage_label)}</strong>',
            f'<span>{percent:.1f}%</span>',
            "</div>",
            '<div style="height:16px;background:#e5e7eb;border-radius:8px;overflow:hidden;">',
            f'<div style="height:100%;width:{percent:.2f}%;background:linear-gradient(90deg,#3b82f6,#22c55e);"></div>',
            "</div>",
            '<div style="font-size:12px;margin-top:5px;opacity:.8;">',
            f'status=<code>{status}</code> · phase=<code>{phase}</code> · '
            f'epoch=<code>{html.escape(epoch_text)}</code> · run=<code>{run_id}</code>',
            "</div>",
            "</div>",
        ]
    )


def _format_live_progress_markdown(
    views: dict[str, dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "### Epoch / Step 详情",
        "",
        "| Stage | 状态 | 最新 epoch | phase | step | global_step | 进度 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]

    for stage in ("stage1", "stage2"):
        view = views[stage]
        snapshot = snapshots[stage]
        current_epoch = snapshot.get("current_epoch")
        total_epochs = snapshot.get("total_epochs")
        epoch_text = "-"
        if current_epoch is not None:
            epoch_text = str(current_epoch)
            if total_epochs is not None:
                epoch_text += f" / {total_epochs}"

        current_step = snapshot.get("current_step")
        total_steps = snapshot.get("total_steps")
        step_text = "-"
        if current_step is not None:
            step_text = str(current_step)
            if total_steps is not None:
                step_text += f" / {total_steps}"

        try:
            percent = float(snapshot.get("percent") or 0.0)
        except (TypeError, ValueError):
            percent = 0.0

        lines.append(
            f"| {stage} | `{view.get('status') or snapshot.get('status') or 'not_started'}` | "
            f"`{epoch_text}` | `{snapshot.get('phase') or 'waiting'}` | `{step_text}` | "
            f"`{snapshot.get('global_step') if snapshot.get('global_step') is not None else '-'}` | "
            f"`{percent:.1f}%` |"
        )

    return "\n".join(lines)


def _emit_training_notifications(
    events: list[dict[str, Any]],
    stage_errors: dict[str, dict[str, Any]],
) -> None:
    for event in events:
        stage = str(event.get("stage") or "training")
        run_id = str(event.get("run_id") or "-")
        status = str(event.get("status") or "unknown")
        try:
            if status == "succeeded":
                info_fn = getattr(gr, "Info", None)
                if callable(info_fn):
                    info_fn(f"{stage} 训练完成：run_id={run_id}")
            else:
                error_payload = stage_errors.get(stage) or {}
                summary = str(error_payload.get("summary") or "请查看训练失败摘要。")
                warning_fn = getattr(gr, "Warning", None)
                if callable(warning_fn):
                    warning_fn(f"{stage} 训练失败：{summary}")
        except Exception:
            pass


def _refresh_live_monitor_core(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    notification_state: dict[str, Any] | None,
    *,
    force_checkpoint_scan: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        service = create_training_service()

        views: dict[str, dict[str, Any]] = {}
        snapshots: dict[str, dict[str, Any]] = {}
        stage_states = _read_monitor_stage_states(active_work_dir)
        stage_errors: dict[str, dict[str, Any]] = {}

        for stage in ("stage1", "stage2"):
            view = service.get_training_progress(active_work_dir, stage)
            snapshot = view.get("progress_snapshot")
            if not isinstance(snapshot, dict):
                snapshot = _empty_progress_snapshot(stage)
            views[stage] = view
            snapshots[stage] = snapshot
            stage_errors[stage] = extract_training_error_excerpt(
                stage_states.get(stage) or {}
            )

        monitor_active = _monitor_should_run(stage_states)
        terminal_transition = _new_terminal_transition(stage_states, notification_state)
        latest_run = select_latest_run(stage_states)
        next_notification_state, events = update_terminal_notification_state(
            stage_states,
            notification_state,
        )
        _emit_training_notifications(events, stage_errors)

        if monitor_active:
            gpu_snapshot = query_gpu_snapshot()
            gpu_md = format_gpu_markdown(gpu_snapshot)
            gpu_payload: dict[str, Any] = gpu_snapshot.to_dict()
        else:
            gpu_md = "### GPU / 显存\n\n当前无运行中的训练任务，GPU 自动监控已暂停。"
            gpu_payload = {
                "available": False,
                "paused": True,
                "reason": "no_active_training",
            }

        should_scan_checkpoints = bool(force_checkpoint_scan or terminal_transition)
        checkpoint_scan: dict[str, Any] | None = None
        if should_scan_checkpoints:
            checkpoint_scan = scan_training_outputs(active_work_dir)
            checkpoint_html: Any = format_checkpoint_visual_html(checkpoint_scan)
        else:
            checkpoint_html = gr.update()

        progress_html = "".join(
            [
                _live_progress_bar_html("Stage1", views["stage1"], snapshots["stage1"]),
                _live_progress_bar_html("Stage2", views["stage2"], snapshots["stage2"]),
            ]
        )
        progress_md = _format_live_progress_markdown(views, snapshots)
        latest_run_md = format_latest_run_markdown(latest_run)
        failure_md = format_failure_summary_markdown(stage_errors)
        monitor_json = {
            "schema_version": "voicelab_training_live_monitor_v2",
            "refreshed_at": _now_iso(),
            "work_dir": str(active_work_dir),
            "monitor_active": monitor_active,
            "refresh_policy": {
                "interval_seconds": LIVE_MONITOR_INTERVAL_SECONDS,
                "active_statuses": sorted(ACTIVE_MONITOR_STATUSES),
                "gpu_only_while_training": True,
                "checkpoint_scan_mode": "manual_or_terminal_transition",
                "checkpoint_scan_performed": should_scan_checkpoints,
            },
            "stage1": views["stage1"],
            "stage2": views["stage2"],
            "latest_run": latest_run,
            "failure_summary": stage_errors,
            "notification_events": events,
            "notification_state": next_notification_state,
            "gpu": gpu_payload,
            "checkpoint_scan": checkpoint_scan,
        }
        outputs = (
            progress_html,
            progress_md,
            gpu_md,
            latest_run_md,
            failure_md,
            checkpoint_html,
            monitor_json,
            next_notification_state,
        )
        return outputs, monitor_active
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        outputs = (
            "<div>Training Live Monitor 暂不可用。</div>",
            f"### Epoch / Step 详情\n\n监控刷新失败：`{error_text}`",
            "### GPU / 显存\n\n本轮未能刷新 GPU 状态。",
            "### 最近一次训练\n\n本轮未能刷新最近训练。",
            "### 训练失败摘要\n\n本轮未能提取失败信息。",
            gr.update(),
            {
                "schema_version": "voicelab_training_live_monitor_v2",
                "refreshed_at": _now_iso(),
                "status": "failed",
                "error": error_text,
            },
            dict(notification_state or {}),
        )
        return outputs, False


def refresh_live_monitor_click(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    notification_state: dict[str, Any] | None,
):
    outputs, _monitor_active = _refresh_live_monitor_core(
        resolved_work_dir,
        work_dir,
        speaker_name,
        notification_state,
        force_checkpoint_scan=True,
    )
    return outputs


def refresh_live_monitor_tick(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    notification_state: dict[str, Any] | None,
):
    outputs, monitor_active = _refresh_live_monitor_core(
        resolved_work_dir,
        work_dir,
        speaker_name,
        notification_state,
        force_checkpoint_scan=False,
    )
    return (*outputs, _timer_component_update(monitor_active))


def launch_training_click_v2(
    stage: str,
    epochs: Any,
    batch_size: Any,
    open_training_terminal: bool,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        ui_overrides = {"epochs": epochs, "batch_size": batch_size}
        service = create_training_service()

        if stage == "stage1":
            state = service.launch_stage1_training(
                active_work_dir,
                refresh_status=refresh_status,
                ui_overrides=ui_overrides,
                show_terminal_progress=bool(open_training_terminal),
            )
        elif stage == "stage2":
            state = service.launch_stage2_training(
                active_work_dir,
                refresh_status=refresh_status,
                ui_overrides=ui_overrides,
                show_terminal_progress=bool(open_training_terminal),
            )
        else:
            raise ValueError(f"Unsupported training stage: {stage!r}")

        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _refresh_panels(active_work_dir, refresh_status=False)

        contract[f"latest_{stage}_run_state"] = state
        effective = state.get("effective_parameters")
        if not isinstance(effective, dict):
            effective = {}
        resolved_override = state.get("ui_override")
        if not isinstance(resolved_override, dict):
            resolved_override = {}
        contract[f"latest_{stage}_ui_override"] = resolved_override

        lines = [
            f"## {stage} Training 已启动",
            "",
            f"- run：`{state.get('run_id')}`",
            f"- epochs：`{effective.get('epochs')}`",
            f"- batch_size：`{effective.get('batch_size')}`",
            f"- 独立日志终端：{'已开启' if state.get('terminal_started') else '关闭'}",
        ]
        if state.get("terminal_error"):
            lines.append(f"- 日志终端提示：`{state.get('terminal_error')}`")

        return (
            "\n".join(lines),
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        return legacy._failure_outputs(
            status_md="\n".join(
                [
                    f"## {stage} Training 启动失败",
                    "",
                    f"**错误类型：** `{type(exc).__name__}`",
                    f"**错误消息：** {exc}",
                ]
            ),
            error=exc,
            resolved_work_dir=resolved_work_dir or work_dir or "",
        )


def stop_training_click_v2(
    stage: str,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        stage_state_path = legacy._stage_run_state_path(active_work_dir, stage)
        state = _read_json_dict(stage_state_path)
        if not state:
            raise FileNotFoundError(f"未找到 {stage} training run_state")

        current_status = str(state.get("status") or "unknown")
        if current_status in TERMINAL_STATUSES:
            return _success_outputs(
                status_md="\n".join(
                    [
                        f"## {stage} Training 已结束",
                        "",
                        f"- run：`{state.get('run_id')}`",
                        f"- 状态：`{current_status}`",
                        "- 无需再次停止。",
                    ]
                ),
                work_dir=active_work_dir,
                refresh_status=refresh_status,
                contract_update_key=f"latest_{stage}_stop_result",
                contract_update_value={"no_op": True, "status": current_status},
            )

        state.update(
            {
                "status": "stopping",
                "message": f"{stage} training stop requested by user.",
                "stop_requested": True,
                "stop_requested_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )
        written_paths = _write_training_state_copies(
            state,
            stage_state_path=stage_state_path,
        )

        child_pid = state.get("child_pid")
        worker_pid = state.get("worker_pid") or state.get("pid")
        kill_attempts: list[dict[str, Any]] = []

        if child_pid and legacy._is_pid_running(child_pid):
            result = legacy._kill_pid(child_pid)
            result["target"] = "child"
            kill_attempts.append(result)
        elif worker_pid and legacy._is_pid_running(worker_pid):
            result = legacy._kill_pid(worker_pid)
            result["target"] = "worker"
            kill_attempts.append(result)
        else:
            kill_attempts.append(
                {
                    "target": "none",
                    "succeeded": True,
                    "message": "No running child/worker process was found.",
                }
            )

        latest = state
        run_state_text = str(state.get("run_state_path") or "").strip()
        for _ in range(10):
            time.sleep(0.1)
            candidate = _read_json_dict(run_state_text or stage_state_path)
            if candidate:
                latest = candidate
            if str(latest.get("status") or "") in TERMINAL_STATUSES:
                break

        if str(latest.get("status") or "") not in TERMINAL_STATUSES:
            process_still_running = bool(
                (child_pid and legacy._is_pid_running(child_pid))
                or (worker_pid and legacy._is_pid_running(worker_pid))
            )
            if not process_still_running:
                latest.update(
                    {
                        "status": "stopped",
                        "message": "Training process stopped after user request.",
                        "ended_at": latest.get("ended_at") or _now_iso(),
                        "child_pid": None,
                    }
                )
            latest["stop_requested"] = True
            latest["stop_result"] = {"attempts": kill_attempts}
            latest["updated_at"] = _now_iso()
            written_paths = _write_training_state_copies(
                latest,
                stage_state_path=stage_state_path,
            )

        return _success_outputs(
            status_md="\n".join(
                [
                    f"## {stage} Training 停止请求已执行",
                    "",
                    f"- run：`{latest.get('run_id')}`",
                    f"- 当前状态：`{latest.get('status')}`",
                ]
            ),
            work_dir=active_work_dir,
            refresh_status=refresh_status,
            contract_update_key=f"latest_{stage}_stop_result",
            contract_update_value={
                "state": latest,
                "kill_attempts": kill_attempts,
                "written_paths": written_paths,
            },
        )
    except Exception as exc:
        return legacy._failure_outputs(
            status_md="\n".join(
                [
                    "## 停止训练失败",
                    "",
                    f"**错误类型：** `{type(exc).__name__}`",
                    f"**错误消息：** {exc}",
                ]
            ),
            error=exc,
            resolved_work_dir=resolved_work_dir or work_dir or "",
        )


def mark_best_checkpoint_click_v2(
    stage: str,
    checkpoint_value: str,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        meta = mark_best_checkpoint(
            active_work_dir,
            stage=stage,
            checkpoint_value=checkpoint_value,
            copy_mode=True,
        )
        source_name = Path(str(meta.get("source_checkpoint") or "")).name or "-"
        metric_value = meta.get("validation_metric_value")
        metric_text = "N/A" if metric_value is None else f"{float(metric_value):.6f}"
        metric_label = "Val Loss / Token" if stage == "stage1" else "Val Loss"
        epoch_text = meta.get("checkpoint_epoch") if meta.get("checkpoint_epoch") is not None else "N/A"
        return _success_outputs(
            status_md="\n".join(
                [
                    f"## {stage} Active Best 已手动锁定",
                    "",
                    f"- checkpoint：`{source_name}`",
                    f"- source run：`{meta.get('source_run_id') or '-'}`",
                    f"- epoch：`{epoch_text}`",
                    f"- {metric_label}：`{metric_text}`",
                    "- 后续训练产生的新 Trainer Best 不会自动覆盖当前选择。",
                ]
            ),
            work_dir=active_work_dir,
            refresh_status=refresh_status,
            checkpoint_update_key="latest_mark_best",
            checkpoint_update_value=meta,
        )
    except Exception as exc:
        return legacy._failure_outputs(
            status_md="\n".join(
                [
                    "## 标记 Active Best 失败",
                    "",
                    f"**错误类型：** `{type(exc).__name__}`",
                    f"**错误消息：** {exc}",
                ]
            ),
            error=exc,
            resolved_work_dir=resolved_work_dir or work_dir or "",
        )


def restore_automatic_best_click(
    stage: str,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        meta = restore_automatic_best(active_work_dir, stage=stage)
        source_name = Path(str(meta.get("source_checkpoint") or "")).name or "-"
        metric_value = meta.get("validation_metric_value")
        metric_text = "N/A" if metric_value is None else f"{float(metric_value):.6f}"
        metric_label = "Val Loss / Token" if stage == "stage1" else "Val Loss"
        epoch_text = meta.get("checkpoint_epoch") if meta.get("checkpoint_epoch") is not None else "N/A"
        return _success_outputs(
            status_md="\n".join(
                [
                    f"## {stage} 已恢复 Automatic Best",
                    "",
                    f"- checkpoint：`{source_name}`",
                    f"- source run：`{meta.get('source_run_id') or '-'}`",
                    f"- epoch：`{epoch_text}`",
                    f"- {metric_label}：`{metric_text}`",
                    "- 后续成功训练产生的新 Trainer Best 可以继续自动更新 Active Best。",
                ]
            ),
            work_dir=active_work_dir,
            refresh_status=refresh_status,
            checkpoint_update_key="latest_restore_automatic_best",
            checkpoint_update_value=meta,
        )
    except Exception as exc:
        return legacy._failure_outputs(
            status_md="\n".join(
                [
                    "## 恢复 Automatic Best 失败",
                    "",
                    f"**错误类型：** `{type(exc).__name__}`",
                    f"**错误消息：** {exc}",
                ]
            ),
            error=exc,
            resolved_work_dir=resolved_work_dir or work_dir or "",
        )


def _checkpoint_metric_text(stage: str, item: dict[str, Any]) -> tuple[str, str]:
    label = "Val Loss / Token" if stage == "stage1" else "Val Loss"
    value = item.get("validation_metric_value")
    if value is None:
        return label, "N/A"
    try:
        return label, f"{float(value):.6f}"
    except (TypeError, ValueError):
        return label, "N/A"


def _checkpoint_item_from_scan(
    stage: str,
    checkpoint_value: str,
    checkpoint_scan: dict[str, Any] | None,
) -> dict[str, Any]:
    if not checkpoint_value or not isinstance(checkpoint_scan, dict):
        return {}
    stages = checkpoint_scan.get("stages")
    if not isinstance(stages, dict):
        return {}
    stage_payload = stages.get(stage)
    if not isinstance(stage_payload, dict):
        return {}
    checkpoints = stage_payload.get("checkpoints")
    if not isinstance(checkpoints, list):
        return {}
    text = str(checkpoint_value)
    for item in checkpoints:
        if not isinstance(item, dict):
            continue
        candidates = {
            str(item.get("label") or ""),
            str(item.get("path") or ""),
            str(item.get("relative_path") or ""),
            str(item.get("name") or ""),
        }
        if text in candidates:
            return dict(item)
    return {}


def checkpoint_delete_selection_change(
    stage: str,
    checkpoint_value: str,
    checkpoint_scan: dict[str, Any] | None,
):
    if not checkpoint_value:
        return "选择一个历史 checkpoint 后可查看删除保护状态。", False

    item = _checkpoint_item_from_scan(stage, checkpoint_value, checkpoint_scan)
    if not item:
        return "所选 checkpoint 的扫描信息已过期，请先点击“刷新 checkpoint 列表”。", False

    metric_label, metric_text = _checkpoint_metric_text(stage, item)
    epoch_text = item.get("checkpoint_epoch") if item.get("checkpoint_epoch") is not None else "N/A"
    size_mb = float(item.get("size_bytes") or 0) / 1024.0 / 1024.0
    lines = [
        f"**run：** `{item.get('run_id') or '-'}`  ",
        f"**checkpoint：** `{item.get('name') or '-'}`  ",
        f"**epoch：** `{epoch_text}`  ",
        f"**{metric_label}：** `{metric_text}`  ",
        f"**大小：** `{size_mb:.1f} MB`",
    ]

    if not item.get("deletion_allowed"):
        lines.extend(
            [
                "",
                f"❌ **当前不可删除：** {item.get('deletion_block_reason') or '该 checkpoint 当前受保护。'}",
            ]
        )
    elif item.get("is_trainer_best"):
        lines.extend(
            [
                "",
                "⚠️ **高风险历史 checkpoint：** 这是该 run 的 Trainer Best。删除后该 run 的训练历史仍保留，但最佳模型文件本身不可恢复。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "✅ 当前 checkpoint 可删除。仅删除模型文件，history / summary / run_state / logs 会保留。",
            ]
        )

    return "\n".join(lines), False


def _checkpoint_visual_from_scan(checkpoint_scan: dict[str, Any] | None):
    if not isinstance(checkpoint_scan, dict):
        return gr.update()
    return format_checkpoint_visual_html(checkpoint_scan)


def _reset_delete_confirmations():
    return False, False


def delete_checkpoint_click_v2(
    stage: str,
    checkpoint_value: str,
    confirmed: bool,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        result = delete_checkpoint(
            active_work_dir,
            stage=stage,
            checkpoint_value=checkpoint_value,
            confirmed=bool(confirmed),
        )

        released_mb = float(result.get("released_bytes") or 0) / 1024.0 / 1024.0
        metric_label, metric_text = _checkpoint_metric_text(stage, result)
        epoch_text = result.get("checkpoint_epoch") if result.get("checkpoint_epoch") is not None else "N/A"
        warning_line = (
            "- 注意：本次删除的是该历史 run 的 Trainer Best 模型文件。"
            if result.get("was_trainer_best")
            else ""
        )
        status_lines = [
            f"## {stage} checkpoint 已永久删除",
            "",
            f"- run：`{result.get('run_id') or '-'}`",
            f"- checkpoint：`{result.get('checkpoint_name') or '-'}`",
            f"- epoch：`{epoch_text}`",
            f"- {metric_label}：`{metric_text}`",
            f"- 已释放空间：`{released_mb:.1f} MB`",
            "- history / summary / run_state / logs 已保留。",
        ]
        if warning_line:
            status_lines.append(warning_line)

        base_outputs = _success_outputs(
            status_md="\n".join(status_lines),
            work_dir=active_work_dir,
            refresh_status=refresh_status,
            checkpoint_update_key="latest_checkpoint_deletion",
            checkpoint_update_value=result,
        )
        refreshed_scan = base_outputs[10] if isinstance(base_outputs[10], dict) else {}
        checkpoint_html = format_checkpoint_visual_html(refreshed_scan)
        result_preview = "\n".join(
            [
                "### 删除完成",
                "",
                f"已删除 `{result.get('checkpoint_name') or '-'}`，释放约 `{released_mb:.1f} MB`。",
                "删除确认已重置；如需继续删除，请重新选择 checkpoint 并再次确认。",
            ]
        )
        return (*base_outputs, checkpoint_html, False, result_preview)
    except Exception as exc:
        base_outputs = legacy._failure_outputs(
            status_md="\n".join(
                [
                    "## Checkpoint 删除未执行",
                    "",
                    f"**原因：** {exc}",
                ]
            ),
            error=exc,
            resolved_work_dir=resolved_work_dir or work_dir or "",
        )
        return (
            *base_outputs,
            gr.update(),
            False,
            "\n".join(
                [
                    "### 删除未执行",
                    "",
                    f"❌ {exc}",
                    "请根据提示调整 Active Best、等待训练结束或重新选择 checkpoint。",
                ]
            ),
        )


def generate_profile_click_v2(
    profile_name: str,
    force_default_stage1: bool,
    force_default_stage2: bool,
    overwrite_profile: bool,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
):
    try:
        active_work_dir = legacy._resolve_active_work_dir(
            resolved_work_dir,
            work_dir,
            speaker_name,
        )
        profile = generate_user_profile(
            active_work_dir,
            profile_name=profile_name or speaker_name or "default_user",
            include_stage1=True,
            include_stage2=True,
            force_default_stage1=bool(force_default_stage1),
            force_default_stage2=bool(force_default_stage2),
            overwrite=bool(overwrite_profile),
        )
        contract = (
            profile.get("inference_contract")
            if isinstance(profile.get("inference_contract"), dict)
            else {}
        )
        stage1_source = "zero-shot default" if contract.get("stage1_fallback") else "few-shot active best"
        stage2_source = "zero-shot default" if contract.get("stage2_fallback") else "few-shot active best"
        return _success_outputs(
            status_md="\n".join(
                [
                    "## Voice Profile 已生成",
                    "",
                    f"- profile：`{profile.get('profile_name')}`",
                    f"- 类型：`{profile.get('profile_type')}`",
                    f"- Stage1 来源：`{stage1_source}`",
                    f"- Stage2 来源：`{stage2_source}`",
                    f"- 可用于推理：`{profile.get('ready_for_inference')}`",
                ]
            ),
            work_dir=active_work_dir,
            refresh_status=refresh_status,
            profile_value=profile,
        )
    except Exception as exc:
        return legacy._failure_outputs(
            status_md="\n".join(
                [
                    "## Voice Profile 生成失败",
                    "",
                    f"**错误类型：** `{type(exc).__name__}`",
                    f"**错误消息：** {exc}",
                ]
            ),
            error=exc,
            resolved_work_dir=resolved_work_dir or work_dir or "",
        )


def create_demo() -> gr.Blocks:
    ui_defaults = _training_ui_defaults()
    stage1_ui_defaults = ui_defaults["stage1"]
    stage2_ui_defaults = ui_defaults["stage2"]

    with gr.Blocks(title="Resonastra · Training") as demo:
        gr.Markdown(
            f"""
# Resonastra · Training

从 DataFactory 工作目录读取训练数据，完成 Stage1 / Stage2 few-shot 训练，并管理 Active Best 与 Voice Profile。

- 用户可覆盖本次 run 的 `epochs` 与 `batch_size`，不会修改 YAML 默认配置。
- Live Monitor 仅在训练运行时每 {LIVE_MONITOR_INTERVAL_SECONDS:.0f} 秒自动刷新；空闲时自动停止。
- 独立训练日志终端默认关闭，可按需开启。
- Stage2 base checkpoint 由项目配置锁定，用户版无需手动设置。
- 历史 checkpoint 支持单文件安全删除；Active Best source 与正在训练 run 会自动保护。
""".strip()
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. 输入")
                speaker_name = gr.Textbox(
                    label="说话人 / 角色名",
                    value="",
                    info="工作目录留空时使用 user_data/{speaker_name}_factory。",
                )
                work_dir = gr.Textbox(
                    label="DataFactory 工作目录",
                    value="",
                )
                refresh_status = gr.Checkbox(
                    label="扫描时刷新 DataFactory 状态",
                    value=True,
                )

                gr.Markdown("## 2. 训练操作")
                with gr.Accordion("本次训练参数（仅当前 run）", open=True):
                    gr.Markdown("默认值读取 `configs/training_default.yaml`，修改后不会写回配置。")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Stage1")
                            stage1_epochs = gr.Number(
                                label="Stage1 epochs",
                                value=stage1_ui_defaults["epochs"],
                                precision=0,
                                minimum=1,
                                step=1,
                            )
                            stage1_batch_size = gr.Number(
                                label="Stage1 batch_size",
                                value=stage1_ui_defaults["batch_size"],
                                precision=0,
                                minimum=1,
                                step=1,
                            )
                        with gr.Column():
                            gr.Markdown("### Stage2")
                            stage2_epochs = gr.Number(
                                label="Stage2 epochs",
                                value=stage2_ui_defaults["epochs"],
                                precision=0,
                                minimum=1,
                                step=1,
                            )
                            stage2_batch_size = gr.Number(
                                label="Stage2 batch_size",
                                value=stage2_ui_defaults["batch_size"],
                                precision=0,
                                minimum=1,
                                step=1,
                            )
                    open_training_terminal = gr.Checkbox(
                        label="打开独立训练日志终端（可选）",
                        value=False,
                        info="默认关闭；开启后仅作为只读日志窗口，不拥有训练进程。",
                    )

                scan_button = gr.Button("扫描训练数据", variant="primary")
                with gr.Row():
                    start_stage1_button = gr.Button("启动 Stage1 Training", variant="primary")
                    start_stage2_button = gr.Button("启动 Stage2 Training", variant="primary")
                with gr.Row():
                    refresh_run_button = gr.Button("刷新训练状态", variant="secondary")
                    stop_stage1_button = gr.Button("停止 Stage1 Training", variant="stop")
                    stop_stage2_button = gr.Button("停止 Stage2 Training", variant="stop")

                gr.Markdown("## 3. Checkpoint / Active Best")
                refresh_checkpoint_button = gr.Button("刷新 checkpoint 列表", variant="secondary")
                stage1_checkpoint_dropdown = gr.Dropdown(
                    label="Stage1 历史 checkpoint",
                    choices=[],
                    value=None,
                    info="选择历史 checkpoint 后可手动锁定为 Active Best，或在确认后永久删除该单个模型文件。",
                )
                with gr.Row():
                    mark_stage1_best_button = gr.Button("手动锁定 Stage1 Active Best", variant="secondary")
                    restore_stage1_auto_button = gr.Button("恢复 Stage1 Automatic Best", variant="secondary")
                with gr.Accordion("删除 Stage1 checkpoint", open=False):
                    stage1_delete_preview = gr.Markdown("选择一个历史 checkpoint 后可查看删除保护状态。")
                    stage1_delete_confirm = gr.Checkbox(
                        label="我确认永久删除当前选中的 Stage1 checkpoint 模型文件",
                        value=False,
                    )
                    delete_stage1_checkpoint_button = gr.Button(
                        "永久删除选中的 Stage1 checkpoint",
                        variant="stop",
                    )

                stage2_checkpoint_dropdown = gr.Dropdown(
                    label="Stage2 历史 checkpoint",
                    choices=[],
                    value=None,
                    info="选择历史 checkpoint 后可手动锁定为 Active Best，或在确认后永久删除该单个模型文件。",
                )
                with gr.Row():
                    mark_stage2_best_button = gr.Button("手动锁定 Stage2 Active Best", variant="secondary")
                    restore_stage2_auto_button = gr.Button("恢复 Stage2 Automatic Best", variant="secondary")
                with gr.Accordion("删除 Stage2 checkpoint", open=False):
                    stage2_delete_preview = gr.Markdown("选择一个历史 checkpoint 后可查看删除保护状态。")
                    stage2_delete_confirm = gr.Checkbox(
                        label="我确认永久删除当前选中的 Stage2 checkpoint 模型文件",
                        value=False,
                    )
                    delete_stage2_checkpoint_button = gr.Button(
                        "永久删除选中的 Stage2 checkpoint",
                        variant="stop",
                    )

                with gr.Accordion("生成 Voice Profile v2", open=True):
                    profile_name = gr.Textbox(
                        label="profile_name",
                        value="",
                    )
                    gr.Markdown("默认使用当前 Active Best；缺少 few-shot checkpoint 的 Stage 自动回退 zero-shot 默认模型。")
                    with gr.Row():
                        force_default_stage1 = gr.Checkbox(
                            label="Stage1 强制使用 zero-shot",
                            value=False,
                        )
                        force_default_stage2 = gr.Checkbox(
                            label="Stage2 强制使用 zero-shot",
                            value=False,
                        )
                    overwrite_profile = gr.Checkbox(label="覆盖已有 profile", value=True)
                    generate_profile_button = gr.Button("生成 Voice Profile v2", variant="primary")

                with gr.Accordion("使用说明", open=False):
                    gr.Markdown(
                        """
- `epochs / batch_size` 只影响当前 run。
- Live Monitor 在训练运行时自动刷新，训练结束后停止高频轮询。
- 独立日志终端为可选只读 viewer，关闭终端不会停止训练。
- `Trainer Best` 为训练脚本按验证指标产生的本次最佳模型。
- `Active Best` 为生成 Profile 时实际使用的 few-shot checkpoint。
- 手动锁定后，新训练不会覆盖 Active Best；恢复 Automatic 后重新跟随最新 Trainer Best。
- checkpoint 删除一次只处理一个模型文件；选择变化后确认框会自动重置。
- 当前 Active Best source、正在训练 run 的 checkpoint 不能删除。
- 删除历史 Trainer Best 是允许的高风险操作，会显示额外警告；训练 metadata 和日志不会随模型文件删除。
""".strip()
                    )

            with gr.Column(scale=1):
                gr.Markdown("## 状态")
                resolved_work_dir = gr.Textbox(
                    label="当前工作目录",
                    value="",
                    interactive=False,
                )
                status_markdown = gr.Markdown("等待扫描。")
                training_status_markdown = gr.Markdown(
                    "### 数据准备\n\n尚未扫描 DataFactory 数据。"
                )
                validation_markdown = gr.Markdown(
                    "### 训练入口\n\n尚未执行训练入口校验。"
                )

                with gr.Accordion("Training Live Monitor", open=True):
                    live_monitor_refresh_button = gr.Button("立即刷新监控", variant="secondary")
                    latest_run_markdown = gr.Markdown(
                        "### 最近一次训练\n\n等待首次刷新。"
                    )
                    live_progress_html = gr.HTML(
                        value="<div>当前无训练进度。</div>",
                        label="Training progress",
                    )
                    live_progress_markdown = gr.Markdown(
                        "### Epoch / Step 详情\n\n当前无运行中的训练任务。"
                    )
                    gpu_markdown = gr.Markdown(
                        "### GPU / 显存\n\n当前无运行中的训练任务。"
                    )
                    failure_summary_markdown = gr.Markdown(
                        "### 训练失败摘要\n\n当前未检测到失败信息。"
                    )

                with gr.Accordion("Checkpoint / Active Best", open=True):
                    checkpoint_visual_html = gr.HTML(
                        value="<div>等待 checkpoint 状态首次刷新。</div>",
                        label="Checkpoint / Active Best",
                    )
                    checkpoint_status_markdown = gr.Markdown("暂无 checkpoint 状态。")

                # Preserve the legacy 14-output protocol without exposing verbose tables/logs.
                parameter_summary_state = gr.State("")
                run_status_state = gr.State("")
                log_state = gr.State("")

                with gr.Accordion("诊断信息", open=False):
                    gr.Markdown("仅用于排错；正常训练无需查看以下内部状态。")
                    contract_json = gr.JSON(label="Training Interface")
                    defaults_json = gr.JSON(label="Training Defaults")
                    live_monitor_json = gr.JSON(label="Live Monitor")
                    checkpoint_json = gr.JSON(label="Checkpoint Scan")
                    profile_json = gr.JSON(label="Voice Profile")

        notification_state = gr.State({})

        outputs = [
            status_markdown,
            contract_json,
            training_status_markdown,
            validation_markdown,
            parameter_summary_state,
            run_status_state,
            log_state,
            checkpoint_status_markdown,
            stage1_checkpoint_dropdown,
            stage2_checkpoint_dropdown,
            checkpoint_json,
            profile_json,
            resolved_work_dir,
            defaults_json,
        ]
        live_monitor_inputs = [
            resolved_work_dir,
            work_dir,
            speaker_name,
            notification_state,
        ]
        live_monitor_outputs = [
            live_progress_html,
            live_progress_markdown,
            gpu_markdown,
            latest_run_markdown,
            failure_summary_markdown,
            checkpoint_visual_html,
            live_monitor_json,
            notification_state,
        ]

        timer_type = getattr(gr, "Timer", None)
        live_monitor_timer = None
        if timer_type is not None:
            live_monitor_timer = timer_type(
                value=LIVE_MONITOR_INTERVAL_SECONDS,
                active=False,
            )

        scan_event = scan_button.click(
            fn=legacy.scan_training_data_click,
            inputs=[work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        stage1_start_event = start_stage1_button.click(
            fn=launch_training_click_v2,
            inputs=[
                gr.State("stage1"),
                stage1_epochs,
                stage1_batch_size,
                open_training_terminal,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        stage2_start_event = start_stage2_button.click(
            fn=launch_training_click_v2,
            inputs=[
                gr.State("stage2"),
                stage2_epochs,
                stage2_batch_size,
                open_training_terminal,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        refresh_run_event = refresh_run_button.click(
            fn=legacy.refresh_training_run_status_click,
            inputs=[resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        stage1_stop_event = stop_stage1_button.click(
            fn=stop_training_click_v2,
            inputs=[
                gr.State("stage1"),
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        stage2_stop_event = stop_stage2_button.click(
            fn=stop_training_click_v2,
            inputs=[
                gr.State("stage2"),
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )

        refresh_checkpoint_event = refresh_checkpoint_button.click(
            fn=legacy.refresh_checkpoints_click,
            inputs=[resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        refresh_checkpoint_event.then(
            fn=_checkpoint_visual_from_scan,
            inputs=[checkpoint_json],
            outputs=[checkpoint_visual_html],
            queue=False,
        )
        refresh_checkpoint_event.then(
            fn=_reset_delete_confirmations,
            inputs=None,
            outputs=[stage1_delete_confirm, stage2_delete_confirm],
            queue=False,
        )

        mark_stage1_event = mark_stage1_best_button.click(
            fn=mark_best_checkpoint_click_v2,
            inputs=[
                gr.State("stage1"),
                stage1_checkpoint_dropdown,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        mark_stage1_event.then(
            fn=_checkpoint_visual_from_scan,
            inputs=[checkpoint_json],
            outputs=[checkpoint_visual_html],
            queue=False,
        )

        mark_stage2_event = mark_stage2_best_button.click(
            fn=mark_best_checkpoint_click_v2,
            inputs=[
                gr.State("stage2"),
                stage2_checkpoint_dropdown,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        mark_stage2_event.then(
            fn=_checkpoint_visual_from_scan,
            inputs=[checkpoint_json],
            outputs=[checkpoint_visual_html],
            queue=False,
        )

        restore_stage1_event = restore_stage1_auto_button.click(
            fn=restore_automatic_best_click,
            inputs=[
                gr.State("stage1"),
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        restore_stage1_event.then(
            fn=_checkpoint_visual_from_scan,
            inputs=[checkpoint_json],
            outputs=[checkpoint_visual_html],
            queue=False,
        )

        restore_stage2_event = restore_stage2_auto_button.click(
            fn=restore_automatic_best_click,
            inputs=[
                gr.State("stage2"),
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )
        restore_stage2_event.then(
            fn=_checkpoint_visual_from_scan,
            inputs=[checkpoint_json],
            outputs=[checkpoint_visual_html],
            queue=False,
        )

        stage1_checkpoint_dropdown.change(
            fn=checkpoint_delete_selection_change,
            inputs=[gr.State("stage1"), stage1_checkpoint_dropdown, checkpoint_json],
            outputs=[stage1_delete_preview, stage1_delete_confirm],
            queue=False,
        )
        stage2_checkpoint_dropdown.change(
            fn=checkpoint_delete_selection_change,
            inputs=[gr.State("stage2"), stage2_checkpoint_dropdown, checkpoint_json],
            outputs=[stage2_delete_preview, stage2_delete_confirm],
            queue=False,
        )

        delete_stage1_checkpoint_button.click(
            fn=delete_checkpoint_click_v2,
            inputs=[
                gr.State("stage1"),
                stage1_checkpoint_dropdown,
                stage1_delete_confirm,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=[
                *outputs,
                checkpoint_visual_html,
                stage1_delete_confirm,
                stage1_delete_preview,
            ],
        )
        delete_stage2_checkpoint_button.click(
            fn=delete_checkpoint_click_v2,
            inputs=[
                gr.State("stage2"),
                stage2_checkpoint_dropdown,
                stage2_delete_confirm,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=[
                *outputs,
                checkpoint_visual_html,
                stage2_delete_confirm,
                stage2_delete_preview,
            ],
        )

        generate_profile_button.click(
            fn=generate_profile_click_v2,
            inputs=[
                profile_name,
                force_default_stage1,
                force_default_stage2,
                overwrite_profile,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )

        live_monitor_refresh_event = live_monitor_refresh_button.click(
            fn=refresh_live_monitor_click,
            inputs=live_monitor_inputs,
            outputs=live_monitor_outputs,
            queue=False,
        )

        if live_monitor_timer is not None:
            live_monitor_timer.tick(
                fn=refresh_live_monitor_tick,
                inputs=live_monitor_inputs,
                outputs=[*live_monitor_outputs, live_monitor_timer],
                queue=False,
            )

            timer_sync_inputs = [resolved_work_dir, work_dir, speaker_name]
            for event in (
                scan_event,
                stage1_start_event,
                stage2_start_event,
                refresh_run_event,
                stage1_stop_event,
                stage2_stop_event,
                live_monitor_refresh_event,
            ):
                event.then(
                    fn=sync_live_monitor_timer,
                    inputs=timer_sync_inputs,
                    outputs=live_monitor_timer,
                    queue=False,
                )

            load_method = getattr(demo, "load", None)
            if callable(load_method):
                load_event = load_method(
                    fn=refresh_live_monitor_click,
                    inputs=live_monitor_inputs,
                    outputs=live_monitor_outputs,
                    queue=False,
                )
                load_event.then(
                    fn=sync_live_monitor_timer,
                    inputs=timer_sync_inputs,
                    outputs=live_monitor_timer,
                    queue=False,
                )
        else:
            gr.Markdown(
                "当前 Gradio 版本不提供 `gr.Timer`；可使用“立即刷新监控”手动刷新。"
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resonastra · Training")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--auto-port", action="store_true")
    parser.add_argument("--auto-port-tries", type=int, default=20)
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        port = legacy._find_available_port(
            args.host,
            port,
            max_tries=int(args.auto_port_tries),
        )
    demo = create_demo()
    try:
        demo.queue(default_concurrency_limit=4)
    except TypeError:
        demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=bool(args.inbrowser),
    )


if __name__ == "__main__":
    main()
