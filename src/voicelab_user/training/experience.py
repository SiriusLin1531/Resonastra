from __future__ import annotations

import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


TERMINAL_STATUSES = frozenset({"succeeded", "failed", "stopped", "stop_failed"})
NOTIFIABLE_STATUSES = frozenset({"succeeded", "failed", "stop_failed"})

_ERROR_PATTERN = re.compile(
    r"(?:traceback \(most recent call last\)|"
    r"cuda out of memory|outofmemoryerror|"
    r"runtimeerror|valueerror|typeerror|keyerror|indexerror|"
    r"filenotfounderror|modulenotfounderror|importerror|assertionerror|"
    r"exception|\[failed\]|\bfatal\b|\berror\b|returncode\s*=\s*[1-9])",
    re.IGNORECASE,
)
_TRACEBACK_START = re.compile(r"traceback \(most recent call last\)", re.IGNORECASE)


def _read_tail_lines(
    path: str | Path,
    *,
    max_bytes: int = 768 * 1024,
    max_lines: int = 800,
) -> list[str]:
    target = Path(path).expanduser().resolve(strict=False)
    if not target.is_file():
        return []

    try:
        size = target.stat().st_size
        with target.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-int(max_bytes), os.SEEK_END)
                handle.readline()
            raw = handle.read()
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []

    return text.splitlines()[-max(1, int(max_lines)) :]


def _state_sort_key(state: Mapping[str, Any]) -> tuple[str, str]:
    timestamp = ""
    for key in ("updated_at", "ended_at", "started_at", "created_at"):
        value = str(state.get(key) or "").strip()
        if value:
            timestamp = value
            break
    return timestamp, str(state.get("run_id") or "")


def select_latest_run(
    stage_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for stage in ("stage1", "stage2"):
        raw = stage_states.get(stage)
        if not isinstance(raw, Mapping) or not raw.get("run_id"):
            continue
        item = dict(raw)
        item["stage"] = stage
        candidates.append(item)

    if not candidates:
        return {}
    candidates.sort(key=_state_sort_key, reverse=True)
    return candidates[0]


def extract_training_error_excerpt(
    state: Mapping[str, Any],
    *,
    max_output_lines: int = 28,
) -> dict[str, Any]:
    status = str(state.get("status") or "unknown").lower()
    if status not in {"failed", "stop_failed"}:
        return {
            "found": False,
            "status": status,
            "summary": "",
            "lines": [],
            "source_log_path": "",
        }

    ordered_paths = [
        value
        for value in (
            state.get("stderr_log_path"),
            state.get("stdout_log_path"),
        )
        if value
    ]

    fallback_message = str(state.get("message") or "").strip()
    problems = state.get("problems")
    if not isinstance(problems, Sequence) or isinstance(problems, (str, bytes)):
        problems = []

    for path_value in ordered_paths:
        lines = _read_tail_lines(path_value)
        if not lines:
            continue

        traceback_indexes = [
            index
            for index, line in enumerate(lines)
            if _TRACEBACK_START.search(str(line))
        ]
        if traceback_indexes:
            start = traceback_indexes[-1]
            excerpt = [str(line).rstrip() for line in lines[start:]]
            excerpt = excerpt[-max(1, int(max_output_lines)) :]
            nonempty = [line.strip() for line in excerpt if line.strip()]
            summary = nonempty[-1] if nonempty else "Python traceback"
            return {
                "found": True,
                "status": status,
                "summary": summary,
                "lines": excerpt,
                "source_log_path": str(path_value),
                "mode": "traceback",
            }

        matched_indexes = [
            index
            for index, line in enumerate(lines)
            if _ERROR_PATTERN.search(str(line))
        ]
        if matched_indexes:
            selected: list[str] = []
            for index in matched_indexes[-8:]:
                window_start = max(0, index - 1)
                window_end = min(len(lines), index + 2)
                selected.extend(str(line).rstrip() for line in lines[window_start:window_end])

            deduped: list[str] = []
            previous: str | None = None
            for line in selected:
                if line == previous:
                    continue
                previous = line
                deduped.append(line)
            deduped = deduped[-max(1, int(max_output_lines)) :]
            matching_lines = [line.strip() for line in deduped if _ERROR_PATTERN.search(line)]
            summary = matching_lines[-1] if matching_lines else "Training failed"
            return {
                "found": True,
                "status": status,
                "summary": summary,
                "lines": deduped,
                "source_log_path": str(path_value),
                "mode": "matched_error_lines",
            }

    fallback_lines = [str(item).strip() for item in problems if str(item).strip()]
    if fallback_message:
        fallback_lines.append(fallback_message)
    return {
        "found": bool(fallback_lines),
        "status": status,
        "summary": fallback_lines[-1] if fallback_lines else "Training failed; no critical log line was detected.",
        "lines": fallback_lines,
        "source_log_path": "",
        "mode": "run_state_fallback",
    }


def format_latest_run_markdown(latest: Mapping[str, Any]) -> str:
    if not latest:
        return "### 最近一次训练\n\n尚未发现 Stage1 / Stage2 训练记录。"

    effective = latest.get("effective_parameters")
    if not isinstance(effective, Mapping):
        effective = {}

    return "\n".join(
        [
            "### 最近一次训练",
            "",
            f"- **Stage：** `{latest.get('stage') or '-'}`",
            f"- **状态：** `{latest.get('status') or 'unknown'}`",
            f"- **run：** `{latest.get('run_id') or '-'}`",
            f"- **epochs / batch_size：** `{effective.get('epochs')}` / `{effective.get('batch_size')}`",
            f"- **开始：** `{latest.get('started_at') or latest.get('created_at') or '-'}`",
            f"- **完成 / 更新：** `{latest.get('ended_at') or latest.get('updated_at') or '-'}`",
        ]
    )


def format_failure_summary_markdown(
    stage_errors: Mapping[str, Mapping[str, Any]],
) -> str:
    failed = [
        (stage, payload)
        for stage, payload in stage_errors.items()
        if isinstance(payload, Mapping) and payload.get("found")
    ]
    if not failed:
        return "### 训练失败摘要\n\n当前最近 run 未检测到失败信息。"

    lines = ["### 训练失败摘要"]
    for stage, payload in failed:
        lines.extend(
            [
                "",
                f"#### {stage}",
                f"**关键错误：** `{payload.get('summary') or 'unknown'}`",
            ]
        )
        excerpt_lines = payload.get("lines")
        if isinstance(excerpt_lines, Sequence) and not isinstance(excerpt_lines, (str, bytes)):
            cleaned = [str(line) for line in excerpt_lines if str(line).strip()]
            if cleaned:
                lines.extend(["", "```text", *cleaned, "```"])
    return "\n".join(lines)


def _selection_label(mode: Any, exists: bool) -> tuple[str, str]:
    mode_text = str(mode or "").lower()
    if not exists:
        return "Zero-shot fallback", "未生成 few-shot active best"
    if mode_text == "manual":
        return "Manual lock", "用户手动锁定；新 trainer best 不会自动覆盖"
    if mode_text == "automatic":
        return "Automatic", "跟随最新成功训练产生的 Trainer Best"
    return mode_text or "Active", "已存在 active best"


def _checkpoint_name(value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        return Path(text).name or fallback
    except Exception:
        return fallback


def _metric_ui_name(stage: str) -> str:
    return "Val Loss / Token" if stage == "stage1" else "Val Loss"


def _metric_text(payload: Mapping[str, Any]) -> str:
    try:
        value = float(payload.get("validation_metric_value"))
    except (TypeError, ValueError):
        return "N/A"
    if value != value or value in {float("inf"), float("-inf")}:
        return "N/A"
    return f"{value:.6f}"


def _epoch_text(payload: Mapping[str, Any]) -> str:
    value = payload.get("checkpoint_epoch")
    return "N/A" if value is None else str(value)


def format_checkpoint_visual_html(scan_payload: Mapping[str, Any]) -> str:
    stages = scan_payload.get("stages")
    if not isinstance(stages, Mapping):
        return "<div>尚未发现 checkpoint 扫描结果。</div>"

    cards: list[str] = []
    for stage in ("stage1", "stage2"):
        item = stages.get(stage)
        if not isinstance(item, Mapping):
            item = {}
        exists = bool(item.get("best_checkpoint_exists"))
        badge, explanation = _selection_label(item.get("selection_mode"), exists)
        active = item.get("active_best")
        if not isinstance(active, Mapping):
            active = {}
        checkpoints = item.get("checkpoints")
        if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
            checkpoints = []
        trainer_best = next(
            (
                candidate
                for candidate in checkpoints
                if isinstance(candidate, Mapping)
                and str(candidate.get("name") or "").startswith("best_model")
            ),
            {},
        )

        active_name = _checkpoint_name(
            active.get("source_checkpoint"),
            "zero-shot default",
        )
        trainer_run = str(trainer_best.get("run_id") or "-")
        trainer_name = str(trainer_best.get("name") or "-")
        active_metric = _metric_text(active) if active else "N/A"
        trainer_metric = _metric_text(trainer_best) if trainer_best else "N/A"
        metric_label = _metric_ui_name(stage)

        cards.append(
            "".join(
                [
                    '<div style="border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:14px;min-width:0;">',
                    '<div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">',
                    f'<strong>{html.escape(stage.upper())}</strong>',
                    f'<span style="padding:2px 8px;border-radius:999px;border:1px solid rgba(128,128,128,.4);">{html.escape(badge)}</span>',
                    "</div>",
                    f'<div style="margin-top:10px;font-size:13px;">Runs: <strong>{int(item.get("num_runs") or 0)}</strong> · '
                    f'Checkpoints: <strong>{int(item.get("num_checkpoints") or 0)}</strong></div>',
                    f'<div style="margin-top:6px;font-size:13px;">Active best: <strong>{"Yes" if exists else "No"}</strong></div>',
                    f'<div style="margin-top:6px;font-size:12px;opacity:.8;">{html.escape(explanation)}</div>',
                    '<div style="margin-top:12px;font-size:12px;font-weight:600;">Active Best</div>',
                    f'<div style="margin-top:4px;font-size:12px;">run: <code>{html.escape(str(active.get("source_run_id") or "-"))}</code></div>',
                    f'<div style="margin-top:4px;font-size:12px;">checkpoint: <code>{html.escape(active_name)}</code></div>',
                    f'<div style="margin-top:4px;font-size:12px;">epoch: <code>{html.escape(_epoch_text(active))}</code></div>',
                    f'<div style="margin-top:4px;font-size:12px;">{html.escape(metric_label)}: <code>{html.escape(active_metric)}</code></div>',
                    '<div style="margin-top:12px;font-size:12px;font-weight:600;">Latest Trainer Best</div>',
                    f'<div style="margin-top:4px;font-size:12px;">run: <code>{html.escape(trainer_run)}</code></div>',
                    f'<div style="margin-top:4px;font-size:12px;">checkpoint: <code>{html.escape(trainer_name)}</code></div>',
                    f'<div style="margin-top:4px;font-size:12px;">epoch: <code>{html.escape(_epoch_text(trainer_best))}</code></div>',
                    f'<div style="margin-top:4px;font-size:12px;">{html.escape(metric_label)}: <code>{html.escape(trainer_metric)}</code></div>',
                    "</div>",
                ]
            )
        )

    return "".join(
        [
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;">',
            *cards,
            "</div>",
        ]
    )


def update_terminal_notification_state(
    stage_states: Mapping[str, Mapping[str, Any]],
    previous_state: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    previous = dict(previous_state or {})
    initialized = bool(previous.get("initialized"))
    seen_raw = previous.get("seen_terminal_tokens")
    seen = {
        str(item)
        for item in seen_raw
        if str(item)
    } if isinstance(seen_raw, Sequence) and not isinstance(seen_raw, (str, bytes)) else set()

    current_tokens: list[str] = []
    events: list[dict[str, Any]] = []
    for stage in ("stage1", "stage2"):
        state = stage_states.get(stage)
        if not isinstance(state, Mapping):
            continue
        run_id = str(state.get("run_id") or "").strip()
        status = str(state.get("status") or "unknown").lower()
        if not run_id or status not in TERMINAL_STATUSES:
            continue
        token = f"{stage}:{run_id}:{status}"
        current_tokens.append(token)
        if initialized and status in NOTIFIABLE_STATUSES and token not in seen:
            events.append(
                {
                    "stage": stage,
                    "run_id": run_id,
                    "status": status,
                    "token": token,
                }
            )
        seen.add(token)

    ordered_seen = sorted(seen)[-100:]
    return {
        **previous,
        "schema_version": "voicelab_training_notification_state_v1",
        "initialized": True,
        "seen_terminal_tokens": ordered_seen,
        "last_checked_at": datetime.now().isoformat(timespec="seconds"),
        "current_terminal_tokens": current_tokens,
    }, events


__all__ = [
    "TERMINAL_STATUSES",
    "NOTIFIABLE_STATUSES",
    "select_latest_run",
    "extract_training_error_excerpt",
    "format_latest_run_markdown",
    "format_failure_summary_markdown",
    "format_checkpoint_visual_html",
    "update_terminal_notification_state",
]
