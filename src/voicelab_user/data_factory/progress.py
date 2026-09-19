from __future__ import annotations

"""DataFactory Runtime State v2 and user-facing live progress model.

The legacy ``data_factory_runtime_state.json`` is intentionally left untouched
because the stable Stop implementation owns that v1 file. DF-UI-3 writes a
parallel lifecycle state file, ``data_factory_runtime_state_v2.json``. This
keeps Stop backward compatible while giving the v4 UI a durable source for
running / succeeded / failed / stopped / blocked task lifecycle.

Progress rules are conservative:
- use a percentage only when the unified status exposes real ``found`` and
  ``expected`` item counts;
- if only the Stage2 pipeline position is known, show ``流程位置 N/6`` rather
  than inventing a percentage;
- if only a processed count is known, show that count without a percentage.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable


RUNTIME_STATE_SCHEMA_VERSION = "voicelab_data_factory_runtime_state_v2"
RUNTIME_STATE_FILENAME = "data_factory_runtime_state_v2.json"
LIVE_MONITOR_SCHEMA_VERSION = "voicelab_data_factory_live_monitor_v1"

ACTIVE_RUNTIME_STATUSES = {"running", "stopping"}
TERMINAL_RUNTIME_STATUSES = {"succeeded", "failed", "stopped", "blocked"}


OPERATION_META: dict[str, dict[str, Any]] = {
    "prepare": {
        "label": "音频处理",
        "resumable": False,
    },
    "proofread_rebuild": {
        "label": "文本确认",
        "resumable": False,
    },
    "stage1": {
        "label": "Stage1 训练数据",
        "resumable": False,
    },
    "stage2_generate": {
        "label": "Stage2 训练数据",
        "resumable": True,
    },
    "stage2_resume": {
        "label": "Stage2 训练数据（继续）",
        "resumable": True,
    },
}


STAGE2_STEPS: tuple[tuple[str, str], ...] = (
    ("stage2_manifest", "准备 Stage2 数据"),
    ("stage2_pt", "生成声学训练数据"),
    ("continuous_semantic", "生成语义缓存"),
    ("style_cache", "生成风格 / 音高特征"),
    ("train_val_split", "划分训练 / 验证集"),
    ("filter", "检查并隔离异常样本"),
)


PROCESS_STEP_MAP: dict[str, dict[str, Any]] = {
    "prepare_fewshot_dataset.py": {
        "operation": "prepare",
        "label": "处理原始音频与识别文本",
    },
    "check_fewshot_dataset_health.py": {
        "operation": "prepare",
        "label": "检查数据质量",
    },
    "rebuild_corrected_manifest.py": {
        "operation": "proofread_rebuild",
        "label": "重建校对后的文本清单",
    },
    "build_stage1_fewshot_dataset.py": {
        "operation": "stage1",
        "label": "生成 Stage1 特征与语义缓存",
    },
    "export_fewshot_stage2_manifest.py": {
        "operation": "stage2",
        "stage_key": "stage2_manifest",
        "label": "准备 Stage2 数据",
        "position": 1,
        "total": 6,
    },
    "preprocess_stage2_fm_dataset.py": {
        "operation": "stage2",
        "stage_key": "stage2_pt",
        "label": "生成声学训练数据",
        "position": 2,
        "total": 6,
    },
    "export_stage2_continuous_semantic_cache.py": {
        "operation": "stage2",
        "stage_key": "continuous_semantic",
        "label": "生成语义缓存",
        "position": 3,
        "total": 6,
    },
    "export_fewshot_style_cache.py": {
        "operation": "stage2",
        "stage_key": "style_cache",
        "label": "生成风格 / 音高特征",
        "position": 4,
        "total": 6,
    },
    "split_stage2_pt_dataset.py": {
        "operation": "stage2",
        "stage_key": "train_val_split",
        "label": "划分训练 / 验证集",
        "position": 5,
        "total": 6,
    },
    "filter_v662_bad_style_samples.py": {
        "operation": "stage2",
        "stage_key": "filter",
        "label": "检查并隔离异常样本",
        "position": 6,
        "total": 6,
    },
}


@dataclass(frozen=True)
class LiveProgressSnapshot:
    runtime_status: str
    operation: str
    operation_label: str
    current_step: str
    stage_key: str
    progress_mode: str
    found: int | None
    expected: int | None
    percent: float | None
    position: int | None
    total_positions: int | None
    elapsed_seconds: float | None
    resumable: bool
    process_detected: bool
    process_script: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> datetime:
    return datetime.now()


def now_iso() -> str:
    return _now().isoformat(timespec="seconds")


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


def runtime_state_path(work_dir: str | Path) -> Path:
    return (
        Path(work_dir).expanduser().resolve(strict=False)
        / RUNTIME_STATE_FILENAME
    )


def read_runtime_state(work_dir: str | Path) -> dict[str, Any]:
    return _read_json_dict(runtime_state_path(work_dir))


def operation_label(operation: str) -> str:
    meta = OPERATION_META.get(str(operation), {})
    return str(meta.get("label") or operation or "DataFactory 任务")


def operation_resumable(operation: str) -> bool:
    meta = OPERATION_META.get(str(operation), {})
    return bool(meta.get("resumable"))


def begin_runtime_task(
    work_dir: str | Path,
    *,
    operation: str,
) -> dict[str, Any]:
    resolved = Path(work_dir).expanduser().resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso()
    payload = {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "status": "running",
        "operation": str(operation),
        "operation_label": operation_label(operation),
        "work_dir": str(resolved),
        "resumable": operation_resumable(operation),
        "started_at": timestamp,
        "updated_at": timestamp,
        "finished_at": None,
        "message": "任务已启动。",
        "result_status": None,
    }
    _atomic_write_json(runtime_state_path(resolved), payload)
    return payload


def _normalize_result_status(result_payload: Any) -> tuple[str, str]:
    if not isinstance(result_payload, dict):
        return "failed", "任务结束，但未返回可识别的结果状态。"

    raw = str(result_payload.get("status") or "").strip().lower()
    message = str(result_payload.get("message") or "").strip()

    if raw in {"succeeded", "success", "done", "ok"}:
        return "succeeded", message or "任务已完成。"
    if raw in {"cancelled", "canceled", "stopped"}:
        return "stopped", message or "任务已停止。"
    if raw == "blocked":
        return "blocked", message or "任务未执行；当前条件不允许启动。"
    if raw in {"failed", "timed_out", "timeout", "error"}:
        return "failed", message or "任务执行失败。"

    # Unified callbacks may wrap a base status under another key. Treat a
    # clearly completed unified status as success only if the outer payload did
    # not explicitly report failure.
    if isinstance(result_payload.get("unified_status"), dict):
        return "succeeded", message or "任务已结束并刷新数据状态。"

    return "failed", message or f"无法识别任务结果状态：{raw or 'missing'}"


def finish_runtime_task(
    work_dir: str | Path,
    *,
    operation: str,
    result_payload: Any,
) -> dict[str, Any]:
    resolved = Path(work_dir).expanduser().resolve(strict=False)
    current = read_runtime_state(resolved)
    status, message = _normalize_result_status(result_payload)
    timestamp = now_iso()

    started_at = current.get("started_at") if isinstance(current, dict) else None
    payload = {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "status": status,
        "operation": str(operation),
        "operation_label": operation_label(operation),
        "work_dir": str(resolved),
        "resumable": operation_resumable(operation),
        "started_at": started_at or timestamp,
        "updated_at": timestamp,
        "finished_at": timestamp,
        "message": message,
        "result_status": (
            str(result_payload.get("status") or "")
            if isinstance(result_payload, dict)
            else None
        ),
    }
    _atomic_write_json(runtime_state_path(resolved), payload)
    return payload


def mark_runtime_stopped(
    work_dir: str | Path,
    *,
    previous_state: dict[str, Any] | None = None,
    stop_payload: Any = None,
) -> dict[str, Any]:
    resolved = Path(work_dir).expanduser().resolve(strict=False)
    previous = (
        dict(previous_state)
        if isinstance(previous_state, dict) and previous_state
        else read_runtime_state(resolved)
    )

    operation = str(previous.get("operation") or "").strip()
    if not operation:
        operation = _infer_operation_from_stop_payload(stop_payload) or "unknown"

    stopped_count = _stop_success_count(stop_payload)
    prior_status = str(previous.get("status") or "").strip().lower()

    # A Stop click with no active lifecycle record and no killed process should
    # not fabricate a stopped task.
    if stopped_count <= 0 and prior_status not in ACTIVE_RUNTIME_STATUSES:
        return previous

    timestamp = now_iso()
    payload = {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "status": "stopped",
        "operation": operation,
        "operation_label": operation_label(operation),
        "work_dir": str(resolved),
        "resumable": operation_resumable(operation),
        "started_at": previous.get("started_at") or timestamp,
        "updated_at": timestamp,
        "finished_at": timestamp,
        "message": (
            f"已停止 {stopped_count} 个匹配的任务子进程。"
            if stopped_count > 0
            else "已记录用户停止请求。"
        ),
        "result_status": "stopped",
        "stop_success_count": stopped_count,
    }
    _atomic_write_json(runtime_state_path(resolved), payload)
    return payload


def _stop_success_count(stop_payload: Any) -> int:
    if not isinstance(stop_payload, dict):
        return 0
    total = 0
    for key in ("kill_results", "stage1_kill_results"):
        values = stop_payload.get(key)
        if not isinstance(values, list):
            continue
        total += sum(
            1 for item in values
            if isinstance(item, dict) and bool(item.get("succeeded"))
        )
    return total


def _infer_operation_from_stop_payload(stop_payload: Any) -> str:
    if not isinstance(stop_payload, dict):
        return ""
    command_lines: list[str] = []
    for key in ("matched_processes", "stage1_matched_processes"):
        values = stop_payload.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                command_lines.append(str(item.get("CommandLine") or ""))

    for command_line in command_lines:
        step = process_step_from_command(command_line)
        operation = str(step.get("operation") or "")
        if operation == "stage2":
            return "stage2_resume"
        if operation:
            return operation
    return ""


def runtime_is_active(state: dict[str, Any] | None) -> bool:
    return str((state or {}).get("status") or "").strip().lower() in ACTIVE_RUNTIME_STATUSES


def _stage_map(status_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    payload = status_payload if isinstance(status_payload, dict) else {}
    result: dict[str, dict[str, Any]] = {}
    for stage in payload.get("stages") or []:
        if isinstance(stage, dict) and stage.get("name"):
            result[str(stage["name"])] = stage
    return result


def process_step_from_command(command_line: str) -> dict[str, Any]:
    lowered = str(command_line or "").lower()
    for script_name, metadata in PROCESS_STEP_MAP.items():
        if script_name.lower() in lowered:
            return {"script": script_name, **metadata}
    return {}


def _work_dir_matches_command(command_line: str, work_dir: str | Path) -> bool:
    lowered = str(command_line or "").lower()
    resolved = Path(work_dir).expanduser().resolve(strict=False)
    variants = {
        str(resolved).lower(),
        resolved.as_posix().lower(),
    }
    return any(value and value in lowered for value in variants)


def detect_active_process_step(
    process_infos: Iterable[dict[str, Any]] | None,
    *,
    work_dir: str | Path,
) -> dict[str, Any]:
    for process in process_infos or []:
        if not isinstance(process, dict):
            continue
        command_line = str(process.get("CommandLine") or "")
        if not _work_dir_matches_command(command_line, work_dir):
            continue
        step = process_step_from_command(command_line)
        if step:
            result = dict(step)
            result["pid"] = process.get("ProcessId")
            result["command_line"] = command_line
            return result
    return {}


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_progress(stage: dict[str, Any] | None) -> tuple[int | None, int | None, float | None]:
    if not isinstance(stage, dict):
        return None, None, None
    counts = stage.get("counts")
    if not isinstance(counts, dict):
        counts = {}

    expected = _safe_int(counts.get("expected"))
    found = _safe_int(counts.get("found"))
    if found is None:
        # Stage1 / prepare expose named counts instead of generic found.
        for key in (
            "frontend_pt_count",
            "semantic_pt_count",
            "manifest_rows",
            "corrected_rows",
            "train",
        ):
            candidate = _safe_int(counts.get(key))
            if candidate is not None:
                found = candidate
                break

    percent: float | None = None
    if expected is not None and expected > 0 and found is not None:
        percent = max(0.0, min(float(found) / float(expected) * 100.0, 100.0))
    return found, expected, percent


def _stage2_position(stage_map: dict[str, dict[str, Any]]) -> tuple[int | None, str, str]:
    for index, (stage_key, label) in enumerate(STAGE2_STEPS, start=1):
        stage = stage_map.get(stage_key) or {}
        if not bool(stage.get("done")):
            return index, stage_key, label
    if STAGE2_STEPS:
        key, label = STAGE2_STEPS[-1]
        return len(STAGE2_STEPS), key, label
    return None, "", ""


def _elapsed_seconds(state: dict[str, Any] | None) -> float | None:
    text = str((state or {}).get("started_at") or "").strip()
    if not text:
        return None
    try:
        started = datetime.fromisoformat(text)
    except ValueError:
        return None
    try:
        return max(0.0, (_now() - started).total_seconds())
    except TypeError:
        return None


def build_live_progress_snapshot(
    *,
    work_dir: str | Path,
    runtime_state: dict[str, Any] | None,
    status_payload: dict[str, Any] | None,
    process_infos: Iterable[dict[str, Any]] | None = None,
) -> LiveProgressSnapshot:
    state = runtime_state if isinstance(runtime_state, dict) else {}
    stage_map = _stage_map(status_payload)
    runtime_status = str(state.get("status") or "idle").strip().lower()
    operation = str(state.get("operation") or "").strip()
    op_label = str(state.get("operation_label") or operation_label(operation))
    resumable = bool(state.get("resumable")) or operation_resumable(operation)

    process_step = detect_active_process_step(process_infos, work_dir=work_dir)
    process_detected = bool(process_step)
    process_script = str(process_step.get("script") or "")
    current_step = str(process_step.get("label") or "")
    stage_key = str(process_step.get("stage_key") or "")
    position = _safe_int(process_step.get("position"))
    total_positions = _safe_int(process_step.get("total"))

    if operation in {"stage2_generate", "stage2_resume"}:
        if not stage_key:
            position, stage_key, inferred_label = _stage2_position(stage_map)
            current_step = current_step or inferred_label
            total_positions = len(STAGE2_STEPS)
        stage = stage_map.get(stage_key)
        found, expected, percent = _count_progress(stage)
        if percent is not None:
            progress_mode = "items"
        elif position is not None:
            progress_mode = "position"
        elif found is not None:
            progress_mode = "count"
        else:
            progress_mode = "indeterminate"
    elif operation == "stage1":
        stage_key = "stage1_dataset"
        current_step = current_step or "生成 Stage1 训练数据"
        found, expected, percent = _count_progress(stage_map.get(stage_key))
        progress_mode = "items" if percent is not None else "count" if found is not None else "indeterminate"
        position = None
        total_positions = None
    elif operation == "prepare":
        stage_key = "prepare"
        current_step = current_step or "处理原始音频与识别文本"
        found, expected, percent = _count_progress(stage_map.get(stage_key))
        progress_mode = "items" if percent is not None else "count" if found is not None else "indeterminate"
        position = None
        total_positions = None
    elif operation == "proofread_rebuild":
        stage_key = "proofread"
        current_step = current_step or "重建校对后的文本清单"
        found, expected, percent = _count_progress(stage_map.get(stage_key))
        progress_mode = "items" if percent is not None else "count" if found is not None else "indeterminate"
        position = None
        total_positions = None
    else:
        found = expected = None
        percent = None
        progress_mode = "idle"
        current_step = current_step or ""
        position = None
        total_positions = None

    elapsed = _elapsed_seconds(state)

    if runtime_status == "running":
        if process_detected:
            message = "已检测到运行中的任务子进程。"
        else:
            message = "任务已登记为运行中；当前未捕获到子进程，可能正在启动或切换步骤。"
    elif runtime_status == "stopping":
        message = "正在停止当前任务。"
    elif runtime_status == "stopped":
        message = "任务已停止。" + (" 可以从已有 Stage2 产物继续。" if resumable else "")
    elif runtime_status == "blocked":
        message = str(state.get("message") or "任务未执行；当前条件不允许启动。")
    elif runtime_status == "succeeded":
        message = "任务已完成。"
    elif runtime_status == "failed":
        message = str(state.get("message") or "任务执行失败。")
    else:
        message = "当前没有运行中的 DataFactory 任务。"

    return LiveProgressSnapshot(
        runtime_status=runtime_status,
        operation=operation,
        operation_label=op_label,
        current_step=current_step,
        stage_key=stage_key,
        progress_mode=progress_mode,
        found=found,
        expected=expected,
        percent=percent,
        position=position,
        total_positions=total_positions,
        elapsed_seconds=elapsed,
        resumable=resumable,
        process_detected=process_detected,
        process_script=process_script,
        message=message,
    )


def _duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_current_task_markdown(snapshot: LiveProgressSnapshot) -> str:
    status_meta = {
        "running": ("⏳", "正在运行"),
        "stopping": ("⏳", "正在停止"),
        "stopped": ("⏸", "已停止"),
        "blocked": ("🛡️", "未执行"),
        "succeeded": ("✅", "已完成"),
        "failed": ("❌", "失败"),
        "idle": ("⬜", "当前无任务"),
    }
    icon, label = status_meta.get(snapshot.runtime_status, ("⬜", snapshot.runtime_status or "当前无任务"))

    if snapshot.runtime_status == "idle" or not snapshot.operation:
        return "### 当前任务\n\n⬜ 当前没有运行中的 DataFactory 任务。"

    lines = [
        "### 当前任务",
        "",
        f"**{icon} {label}** · {snapshot.operation_label}",
    ]
    if snapshot.current_step:
        lines.append(f"- 当前步骤：**{snapshot.current_step}**")
    if snapshot.elapsed_seconds is not None:
        lines.append(f"- 已运行：`{_duration_text(snapshot.elapsed_seconds)}`")
    if snapshot.runtime_status == "stopped" and snapshot.resumable:
        lines.append("- 下一步：可点击 **继续生成 Stage2 训练数据**。")
    lines.extend(["", snapshot.message])
    return "\n".join(lines)


def format_live_progress_html(snapshot: LiveProgressSnapshot) -> str:
    title = escape(snapshot.operation_label or "DataFactory")
    step = escape(snapshot.current_step or "等待任务")

    parts = [
        '<div class="vl-live-progress-card">',
        '<div class="vl-live-progress-head">',
        f"<strong>{title}</strong>",
        f"<span>{step}</span>",
        "</div>",
    ]

    if snapshot.progress_mode == "items" and snapshot.percent is not None:
        percent = max(0.0, min(float(snapshot.percent), 100.0))
        parts.extend(
            [
                '<div class="vl-live-progress-track">',
                f'<div class="vl-live-progress-fill" style="width:{percent:.2f}%"></div>',
                "</div>",
                '<div class="vl-live-progress-meta">',
                f"<span>{snapshot.found} / {snapshot.expected}</span>",
                f"<span>{percent:.1f}%</span>",
                "</div>",
            ]
        )
    elif snapshot.progress_mode == "position" and snapshot.position is not None:
        parts.append(
            '<div class="vl-live-progress-position">'
            f"流程位置 {snapshot.position}/{snapshot.total_positions or '?'}"
            "</div>"
        )
    elif snapshot.progress_mode == "count" and snapshot.found is not None:
        parts.append(
            '<div class="vl-live-progress-position">'
            f"已处理 / 生成 {snapshot.found} 项"
            "</div>"
        )
    else:
        parts.append(
            '<div class="vl-live-progress-position">等待可可靠统计的进度信息</div>'
        )

    parts.append("</div>")
    return "".join(parts)


def build_live_monitor_payload(
    *,
    work_dir: str | Path,
    runtime_state: dict[str, Any] | None,
    status_payload: dict[str, Any] | None,
    process_infos: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot = build_live_progress_snapshot(
        work_dir=work_dir,
        runtime_state=runtime_state,
        status_payload=status_payload,
        process_infos=process_infos,
    )
    return {
        "schema_version": LIVE_MONITOR_SCHEMA_VERSION,
        "refreshed_at": now_iso(),
        "work_dir": str(Path(work_dir).expanduser().resolve(strict=False)),
        "monitor_active": runtime_is_active(runtime_state),
        "runtime_state_path": str(runtime_state_path(work_dir)),
        "runtime_state": dict(runtime_state or {}),
        "progress": snapshot.to_dict(),
    }


__all__ = [
    "RUNTIME_STATE_SCHEMA_VERSION",
    "RUNTIME_STATE_FILENAME",
    "LIVE_MONITOR_SCHEMA_VERSION",
    "ACTIVE_RUNTIME_STATUSES",
    "TERMINAL_RUNTIME_STATUSES",
    "OPERATION_META",
    "STAGE2_STEPS",
    "PROCESS_STEP_MAP",
    "LiveProgressSnapshot",
    "runtime_state_path",
    "read_runtime_state",
    "operation_label",
    "operation_resumable",
    "begin_runtime_task",
    "finish_runtime_task",
    "mark_runtime_stopped",
    "runtime_is_active",
    "process_step_from_command",
    "detect_active_process_step",
    "build_live_progress_snapshot",
    "format_current_task_markdown",
    "format_live_progress_html",
    "build_live_monitor_payload",
]
