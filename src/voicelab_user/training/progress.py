from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


_STAGE_RATIO_RE = re.compile(r"\[epoch\s+(\d+)\s*/\s*(\d+)\]", re.IGNORECASE)
_STAGE2_STEP_RE = re.compile(
    r"\[epoch=(\d+)\s+step=(\d+)\s*/\s*(\d+)(?:\s+global_step=(\d+))?\]",
    re.IGNORECASE,
)
_STAGE1_STEP_RE = re.compile(
    r"\[train\]\[epoch=(\d+)\s+step=(\d+)\]",
    re.IGNORECASE,
)

_TERMINAL_STATUSES = {"succeeded", "failed", "stopped", "stop_failed"}


@dataclass(frozen=True, slots=True)
class TrainingProgressSnapshot:
    stage: str
    status: str
    current_epoch: int | None
    total_epochs: int | None
    completed_epochs: int
    current_step: int | None
    total_steps: int | None
    global_step: int | None
    phase: str
    percent: float
    latest_epoch_line: str
    source_log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GPUDeviceSnapshot:
    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    temperature_c: float | None

    @property
    def memory_percent(self) -> float | None:
        if self.memory_used_mb is None or self.memory_total_mb in {None, 0}:
            return None
        return max(
            0.0,
            min(float(self.memory_used_mb) / float(self.memory_total_mb) * 100.0, 100.0),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memory_percent"] = self.memory_percent
        return payload


@dataclass(frozen=True, slots=True)
class GPUSnapshot:
    available: bool
    devices: tuple[GPUDeviceSnapshot, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "devices": [item.to_dict() for item in self.devices],
            "error": self.error,
        }


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_tail_lines(
    path: str | Path,
    *,
    max_bytes: int = 512 * 1024,
    max_lines: int = 2500,
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

    lines = text.splitlines()
    return lines[-int(max_lines):]


def _iter_log_lines(paths: Sequence[str | Path]) -> Iterable[tuple[str, str]]:
    for path in paths:
        for line in _read_tail_lines(path):
            yield str(path), line


def parse_training_progress(
    *,
    stage: str,
    status: str,
    log_paths: Sequence[str | Path],
    total_epochs_hint: int | None = None,
) -> TrainingProgressSnapshot:
    current_epoch: int | None = None
    total_epochs = _safe_int(total_epochs_hint)
    completed_epochs = 0
    current_step: int | None = None
    total_steps: int | None = None
    global_step: int | None = None
    phase = "waiting"
    latest_epoch_line = ""
    source_log_path = ""

    for path_text, line in _iter_log_lines(log_paths):
        line_text = str(line).strip()
        if not line_text:
            continue

        ratio_match = _STAGE_RATIO_RE.search(line_text)
        if ratio_match:
            epoch_value = int(ratio_match.group(1))
            total_value = int(ratio_match.group(2))
            current_epoch = epoch_value
            total_epochs = total_value
            latest_epoch_line = line_text
            source_log_path = path_text

            lower = line_text.lower()
            if "training started" in lower:
                phase = "train"
                current_step = None
                total_steps = None
            elif "training finished" in lower:
                phase = "validation_pending"
            elif "validation started" in lower:
                phase = "validation"
            elif "validation finished" in lower:
                phase = "epoch_complete"
                completed_epochs = max(completed_epochs, epoch_value)
            elif line_text.startswith("[epoch ") or line_text.startswith("[Epoch "):
                # Stage1 emits one compact [epoch x/y] summary only after the
                # epoch has completed. Stage2 ratio lines include explicit
                # phase text handled above.
                if str(stage).lower() == "stage1":
                    phase = "epoch_complete"
                    completed_epochs = max(completed_epochs, epoch_value)

        step_match = _STAGE2_STEP_RE.search(line_text)
        if step_match:
            current_epoch = int(step_match.group(1))
            current_step = int(step_match.group(2))
            total_steps = int(step_match.group(3))
            if step_match.group(4) is not None:
                global_step = int(step_match.group(4))
            phase = "train"
            latest_epoch_line = line_text
            source_log_path = path_text
            continue

        stage1_step_match = _STAGE1_STEP_RE.search(line_text)
        if stage1_step_match:
            current_epoch = int(stage1_step_match.group(1))
            current_step = int(stage1_step_match.group(2))
            total_steps = None
            phase = "train"
            latest_epoch_line = line_text
            source_log_path = path_text
            continue

        lower = line_text.lower()
        if "[failed]" in lower or "traceback (most recent call last)" in lower:
            phase = "failed"
            source_log_path = path_text
        elif "training summary" in lower or "status     : ok" in lower:
            phase = "completed"

    normalized_status = str(status or "unknown").lower()
    if normalized_status == "succeeded":
        phase = "completed"
        if total_epochs is not None:
            current_epoch = total_epochs
            completed_epochs = total_epochs
    elif normalized_status in {"failed", "stop_failed"}:
        phase = "failed"
    elif normalized_status == "stopped":
        phase = "stopped"

    percent = 0.0
    if total_epochs is not None and total_epochs > 0:
        if normalized_status == "succeeded":
            percent = 100.0
        elif (
            phase == "train"
            and current_epoch is not None
            and current_step is not None
            and total_steps is not None
            and total_steps > 0
        ):
            completed_before = max(current_epoch - 1, 0)
            intra_epoch = max(0.0, min(float(current_step) / float(total_steps), 1.0))
            percent = (completed_before + intra_epoch) / float(total_epochs) * 100.0
        else:
            percent = float(max(completed_epochs, 0)) / float(total_epochs) * 100.0

    percent = max(0.0, min(percent, 100.0))

    return TrainingProgressSnapshot(
        stage=str(stage),
        status=str(status or "unknown"),
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        completed_epochs=int(completed_epochs),
        current_step=current_step,
        total_steps=total_steps,
        global_step=global_step,
        phase=phase,
        percent=round(percent, 2),
        latest_epoch_line=latest_epoch_line,
        source_log_path=source_log_path,
    )


def snapshot_from_run_state(state: dict[str, Any], *, stage: str) -> TrainingProgressSnapshot:
    effective = state.get("effective_parameters")
    if not isinstance(effective, dict):
        effective = {}

    total_epochs_hint = _safe_int(effective.get("epochs"))
    log_paths = [
        value
        for value in (
            state.get("stdout_log_path"),
            state.get("stderr_log_path"),
        )
        if value
    ]
    return parse_training_progress(
        stage=stage,
        status=str(state.get("status") or "unknown"),
        log_paths=log_paths,
        total_epochs_hint=total_epochs_hint,
    )


def query_gpu_snapshot(*, timeout_seconds: float = 2.0) -> GPUSnapshot:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return GPUSnapshot(
            available=False,
            devices=(),
            error="nvidia-smi not found in PATH",
        )

    command = [
        executable,
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = int(subprocess.CREATE_NO_WINDOW)

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(float(timeout_seconds), 0.1),
            creationflags=creationflags,
        )
    except Exception as exc:
        return GPUSnapshot(
            available=False,
            devices=(),
            error=f"{type(exc).__name__}: {exc}",
        )

    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip()
        return GPUSnapshot(
            available=False,
            devices=(),
            error=error_text or f"nvidia-smi returncode={completed.returncode}",
        )

    devices: list[GPUDeviceSnapshot] = []
    for row in completed.stdout.splitlines():
        parts = [item.strip() for item in row.split(",")]
        if len(parts) < 6:
            continue
        index = _safe_int(parts[0])
        if index is None:
            continue
        devices.append(
            GPUDeviceSnapshot(
                index=index,
                name=parts[1],
                utilization_percent=_safe_float(parts[2]),
                memory_used_mb=_safe_float(parts[3]),
                memory_total_mb=_safe_float(parts[4]),
                temperature_c=_safe_float(parts[5]),
            )
        )

    if not devices:
        return GPUSnapshot(
            available=False,
            devices=(),
            error="nvidia-smi returned no parseable GPU rows",
        )

    return GPUSnapshot(available=True, devices=tuple(devices), error=None)


def format_gpu_markdown(snapshot: GPUSnapshot) -> str:
    lines = ["### GPU / 显存"]
    if not snapshot.available:
        lines.extend(["", f"GPU 状态不可用：`{snapshot.error or 'unknown'}`"])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| GPU | 利用率 | 显存 | 显存占用 | 温度 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in snapshot.devices:
        used_gb = None if item.memory_used_mb is None else item.memory_used_mb / 1024.0
        total_gb = None if item.memory_total_mb is None else item.memory_total_mb / 1024.0
        util_text = "?" if item.utilization_percent is None else f"{item.utilization_percent:.0f}%"
        memory_text = (
            "?"
            if used_gb is None or total_gb is None
            else f"{used_gb:.2f} / {total_gb:.2f} GB"
        )
        memory_percent_text = "?" if item.memory_percent is None else f"{item.memory_percent:.1f}%"
        temperature_text = "?" if item.temperature_c is None else f"{item.temperature_c:.0f} °C"
        lines.append(
            f"| `{item.index}` {item.name} | {util_text} | {memory_text} | {memory_percent_text} | {temperature_text} |"
        )
    return "\n".join(lines)


def format_progress_markdown(snapshots: Sequence[TrainingProgressSnapshot]) -> str:
    lines = [
        "### 训练进度监控",
        "",
        "| Stage | 状态 | 最新 epoch | 阶段 | step | 进度 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for item in snapshots:
        epoch_text = "-"
        if item.current_epoch is not None:
            epoch_text = str(item.current_epoch)
            if item.total_epochs is not None:
                epoch_text += f" / {item.total_epochs}"
        step_text = "-"
        if item.current_step is not None:
            step_text = str(item.current_step)
            if item.total_steps is not None:
                step_text += f" / {item.total_steps}"
        lines.append(
            f"| {item.stage} | `{item.status}` | `{epoch_text}` | `{item.phase}` | `{step_text}` | `{item.percent:.1f}%` |"
        )
        if item.latest_epoch_line:
            lines.append(
                f"| ↳ {item.stage} latest |  |  | `{item.latest_epoch_line[:180]}` |  |  |"
            )
    return "\n".join(lines)


__all__ = [
    "TrainingProgressSnapshot",
    "GPUDeviceSnapshot",
    "GPUSnapshot",
    "parse_training_progress",
    "snapshot_from_run_state",
    "query_gpu_snapshot",
    "format_gpu_markdown",
    "format_progress_markdown",
]
