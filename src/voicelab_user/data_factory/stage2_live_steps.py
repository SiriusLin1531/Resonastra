from __future__ import annotations

"""User-facing six-step Stage2 live-state presentation model.

DF-UI-4 converts the existing artifact scanner state plus DF-UI-3
runtime/process progress into six simultaneously visible user-facing steps.

This module is deliberately presentation-only:
- no subprocess execution;
- no filesystem scanning;
- no quarantine decisions;
- no training readiness decisions;
- no Gradio imports.

State precedence for the current Stage2 step is conservative:

    running process/runtime overlay
        > stopped / failed runtime overlay
        > artifact done
        > artifact partial
        > blocked / missing artifact state

A percentage is shown only when the scanner provides a real ``found`` and
positive ``expected`` count. Stage position is never converted into a fake
percentage.

DF-UI-4-5 keeps exact counts/percentages in the snapshot and diagnostics while
making the HTML presentation quieter for completed steps. Current, partial,
stopped, and failed steps keep their useful progress detail and receive concise
next-action guidance.
"""

from dataclasses import asdict, dataclass
from html import escape
from typing import Any

from .progress import LiveProgressSnapshot, STAGE2_STEPS


STEP_STATE_DONE = "done"
STEP_STATE_RUNNING = "running"
STEP_STATE_PARTIAL = "partial"
STEP_STATE_STOPPED = "stopped"
STEP_STATE_FAILED = "failed"
STEP_STATE_WAITING = "waiting"
STEP_STATE_NOT_STARTED = "not_started"


_STATE_META: dict[str, tuple[str, str]] = {
    STEP_STATE_DONE: ("✅", "已完成"),
    STEP_STATE_RUNNING: ("⏳", "正在处理"),
    STEP_STATE_PARTIAL: ("⚠️", "部分完成"),
    STEP_STATE_STOPPED: ("⏸", "已停止，可继续"),
    STEP_STATE_FAILED: ("❌", "需要处理"),
    STEP_STATE_WAITING: ("⬜", "等待前置步骤"),
    STEP_STATE_NOT_STARTED: ("⬜", "尚未开始"),
}


@dataclass(frozen=True)
class Stage2LiveStep:
    index: int
    key: str
    label: str
    state: str
    icon: str
    state_label: str
    detail: str
    guidance: str
    source_status: str
    found: int | None
    expected: int | None
    percent: float | None
    is_current: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2LiveStepsSnapshot:
    operation: str
    runtime_status: str
    current_stage_key: str
    active: bool
    has_artifacts: bool
    all_done: bool
    steps: tuple[Stage2LiveStep, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["steps"] = [step.to_dict() for step in self.steps]
        return value


def _stage_map(status_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    payload = status_payload if isinstance(status_payload, dict) else {}
    result: dict[str, dict[str, Any]] = {}
    for stage in payload.get("stages") or []:
        if isinstance(stage, dict) and stage.get("name"):
            result[str(stage["name"])] = stage
    return result


def _summary(status_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = status_payload if isinstance(status_payload, dict) else {}
    value = payload.get("summary")
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_pair(stage: dict[str, Any]) -> tuple[int | None, int | None, float | None]:
    counts = stage.get("counts")
    if not isinstance(counts, dict):
        return None, None, None

    found = _safe_int(counts.get("found"))
    expected = _safe_int(counts.get("expected"))
    percent: float | None = None
    if found is not None and expected is not None and expected > 0:
        percent = max(0.0, min(float(found) / float(expected) * 100.0, 100.0))
    return found, expected, percent


def _manifest_detail(stage: dict[str, Any]) -> str:
    counts = stage.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    rows = _safe_int(counts.get("stage2_manifest_fewshot_rows"))
    if rows is None:
        return ""
    return f"{rows} 条样本"


def _split_detail(stage: dict[str, Any]) -> str:
    counts = stage.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    train = _safe_int(counts.get("train"))
    val = _safe_int(counts.get("val"))
    found, expected, percent = _count_pair(stage)

    parts: list[str] = []
    if train is not None or val is not None:
        parts.append(f"train {train or 0} · val {val or 0}")
    if found is not None and expected is not None and expected > 0:
        parts.append(f"{found}/{expected} · {percent:.1f}%")
    return " · ".join(parts)


def _filter_detail(stage: dict[str, Any]) -> str:
    counts = stage.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    ok = _safe_int(counts.get("ok"))
    bad = _safe_int(counts.get("bad"))
    source = _safe_int(counts.get("source"))

    if ok is not None or bad is not None:
        return f"可用 {ok or 0} · 隔离 {bad or 0}"
    if source is not None:
        return f"检查 {source} 项"
    return ""


def _generic_detail(stage: dict[str, Any]) -> str:
    found, expected, percent = _count_pair(stage)
    if found is not None and expected is not None and expected > 0:
        return f"{found}/{expected} · {percent:.1f}%"
    if found is not None:
        return f"已生成 {found} 项"
    return ""


def _step_detail(stage_key: str, stage: dict[str, Any]) -> str:
    if stage_key == "stage2_manifest":
        return _manifest_detail(stage)
    if stage_key == "train_val_split":
        return _split_detail(stage)
    if stage_key == "filter":
        return _filter_detail(stage)
    return _generic_detail(stage)


def _artifact_state(stage: dict[str, Any]) -> str:
    if bool(stage.get("done")) or str(stage.get("status") or "") == "done":
        return STEP_STATE_DONE
    if bool(stage.get("partial")) or str(stage.get("status") or "") == "partial":
        return STEP_STATE_PARTIAL
    if str(stage.get("status") or "") == "blocked":
        return STEP_STATE_WAITING
    return STEP_STATE_NOT_STARTED


def _guidance_for_state(state: str) -> str:
    if state == STEP_STATE_PARTIAL:
        return "已有部分产物，可继续生成。"
    if state == STEP_STATE_STOPPED:
        return "点击“继续生成 Stage2 训练数据”从现有产物继续。"
    if state == STEP_STATE_FAILED:
        return "请先查看诊断信息；确认问题后可重试或继续生成。"
    return ""


def _has_stage2_artifacts(stage_map: dict[str, dict[str, Any]]) -> bool:
    for stage_key, _ in STAGE2_STEPS:
        stage = stage_map.get(stage_key) or {}
        if bool(stage.get("done")) or bool(stage.get("partial")):
            return True
        status = str(stage.get("status") or "")
        if status in {"done", "partial"}:
            return True
        counts = stage.get("counts")
        if isinstance(counts, dict):
            # Only produced/final counts count as Stage2 artifacts. Do not use
            # expected/source counts, which can be non-zero before this step
            # has produced anything.
            for key in (
                "found",
                "stage2_manifest_fewshot_rows",
                "train",
                "val",
                "ok",
                "bad",
            ):
                value = _safe_int(counts.get(key))
                if value is not None and value > 0:
                    return True
    return False


def build_stage2_live_steps_snapshot(
    status_payload: dict[str, Any] | None,
    progress: LiveProgressSnapshot,
) -> Stage2LiveStepsSnapshot:
    stage_map = _stage_map(status_payload)
    summary = _summary(status_payload)

    operation = str(progress.operation or "")
    runtime_status = str(progress.runtime_status or "idle")
    is_stage2_operation = operation in {"stage2_generate", "stage2_resume"}
    current_stage_key = str(progress.stage_key or "") if is_stage2_operation else ""

    steps: list[Stage2LiveStep] = []
    for index, (stage_key, label) in enumerate(STAGE2_STEPS, start=1):
        stage = stage_map.get(stage_key) or {}
        source_status = str(stage.get("status") or "missing")
        state = _artifact_state(stage)
        is_current = bool(current_stage_key and current_stage_key == stage_key)

        # Runtime overlay applies only to the current Stage2 step. It wins over
        # artifact state while the process is active/stopped/failed because the
        # user is asking "what is happening now?" rather than only "what files
        # exist?".
        if is_current and is_stage2_operation:
            if runtime_status in {"running", "stopping"}:
                state = STEP_STATE_RUNNING
            elif runtime_status == "stopped" and state != STEP_STATE_DONE:
                state = STEP_STATE_STOPPED
            elif runtime_status == "failed" and state != STEP_STATE_DONE:
                state = STEP_STATE_FAILED

        icon, state_label = _STATE_META[state]
        found, expected, percent = _count_pair(stage)
        steps.append(
            Stage2LiveStep(
                index=index,
                key=stage_key,
                label=label,
                state=state,
                icon=icon,
                state_label=state_label,
                detail=_step_detail(stage_key, stage),
                guidance=_guidance_for_state(state),
                source_status=source_status,
                found=found,
                expected=expected,
                percent=percent,
                is_current=is_current,
            )
        )

    all_done = bool(summary.get("all_stage2_done")) or all(
        step.state == STEP_STATE_DONE for step in steps
    )

    return Stage2LiveStepsSnapshot(
        operation=operation,
        runtime_status=runtime_status,
        current_stage_key=current_stage_key,
        active=bool(is_stage2_operation and runtime_status in {"running", "stopping"}),
        has_artifacts=_has_stage2_artifacts(stage_map),
        all_done=all_done,
        steps=tuple(steps),
    )


def _current_step(snapshot: Stage2LiveStepsSnapshot) -> Stage2LiveStep | None:
    for step in snapshot.steps:
        if step.is_current:
            return step
    return None


def _panel_header(snapshot: Stage2LiveStepsSnapshot) -> tuple[str, str]:
    current = _current_step(snapshot)
    done_count = sum(step.state == STEP_STATE_DONE for step in snapshot.steps)

    if snapshot.all_done:
        return "✅ Stage2 数据已准备完成", "6/6"
    if snapshot.active:
        suffix = f"当前：{current.label}" if current is not None else f"{done_count}/6 已完成"
        return "⏳ 正在准备 Stage2 数据", suffix
    if (
        snapshot.runtime_status == "stopped"
        and snapshot.operation in {"stage2_generate", "stage2_resume"}
    ):
        suffix = f"停在：{current.label}" if current is not None else f"{done_count}/6 已完成"
        return "⏸ Stage2 已停止，可继续", suffix
    if (
        snapshot.runtime_status == "failed"
        and snapshot.operation in {"stage2_generate", "stage2_resume"}
    ):
        suffix = f"问题步骤：{current.label}" if current is not None else f"{done_count}/6 已完成"
        return "❌ Stage2 需要处理", suffix
    if snapshot.has_artifacts:
        return "⚠️ Stage2 数据尚未完成", f"{done_count}/6 已完成"
    return "⬜ Stage2 尚未开始", "0/6"


def _compact_done_detail(step: Stage2LiveStep) -> str:
    """Keep completed rows informative without repeating 100% on every line."""

    if step.key in {"stage2_manifest", "filter"}:
        return step.detail
    if step.key == "train_val_split":
        if " · " in step.detail:
            # Current split detail is "train N · val M · found/expected · 100%".
            # Only the train/val composition is useful once the step is done.
            parts = step.detail.split(" · ")
            if len(parts) >= 2:
                return " · ".join(parts[:2])
        return step.detail
    if step.found is not None:
        return f"{step.found} 项"
    return ""


def _display_detail(step: Stage2LiveStep) -> str:
    if step.state == STEP_STATE_DONE:
        return _compact_done_detail(step)
    if step.state in {
        STEP_STATE_RUNNING,
        STEP_STATE_PARTIAL,
        STEP_STATE_STOPPED,
        STEP_STATE_FAILED,
    }:
        return step.detail
    return ""


def format_stage2_live_steps_html(snapshot: Stage2LiveStepsSnapshot) -> str:
    """Render a quiet-at-rest, attention-forward six-step Stage2 panel."""

    if not snapshot.steps:
        return '<div class="vl-stage2-steps-empty">暂无 Stage2 状态。</div>'

    header, badge = _panel_header(snapshot)
    rows: list[str] = [
        '<div class="vl-stage2-steps">',
        '<div class="vl-stage2-steps-title">'
        f'<strong>{escape(header)}</strong><span>{escape(badge)}</span>'
        '</div>',
    ]
    for step in snapshot.steps:
        current_class = " vl-stage2-step-current" if step.is_current else ""
        attention_class = (
            " vl-stage2-step-attention"
            if step.state in {STEP_STATE_PARTIAL, STEP_STATE_STOPPED, STEP_STATE_FAILED}
            else ""
        )
        detail = _display_detail(step)
        detail_html = (
            f'<div class="vl-stage2-step-detail">{escape(detail)}</div>'
            if detail
            else ""
        )
        guidance_html = (
            f'<div class="vl-stage2-step-guidance">{escape(step.guidance)}</div>'
            if step.guidance
            else ""
        )
        rows.append(
            f'<div class="vl-stage2-step vl-stage2-step-{escape(step.state)}'
            f'{current_class}{attention_class}">'
            '<div class="vl-stage2-step-main">'
            f'<span class="vl-stage2-step-index">{step.index}</span>'
            f'<span class="vl-stage2-step-icon">{escape(step.icon)}</span>'
            f'<span class="vl-stage2-step-label">{escape(step.label)}</span>'
            f'<span class="vl-stage2-step-state">{escape(step.state_label)}</span>'
            '</div>'
            f'{detail_html}'
            f'{guidance_html}'
            '</div>'
        )
    rows.append('</div>')
    return "".join(rows)


__all__ = [
    "STEP_STATE_DONE",
    "STEP_STATE_RUNNING",
    "STEP_STATE_PARTIAL",
    "STEP_STATE_STOPPED",
    "STEP_STATE_FAILED",
    "STEP_STATE_WAITING",
    "STEP_STATE_NOT_STARTED",
    "Stage2LiveStep",
    "Stage2LiveStepsSnapshot",
    "build_stage2_live_steps_snapshot",
    "format_stage2_live_steps_html",
]
