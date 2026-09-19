from __future__ import annotations

"""DF-UI-6 user-facing DataFactory status overview presentation.

This module deliberately sits above the existing status/readiness layers. It
consumes an already-built quarantine-aware unified status payload and presents
only the information a normal user needs near the top of the right column:

- which DataFactory project/workdir is currently selected;
- whether Stage1 training data is ready;
- whether Stage2 training data is ready;
- whether both are ready together.

It does NOT scan files, write status JSON, build training artifacts, decide
quarantine validity, or replace ``training.data_contract``. Stage1 and Stage2
readiness are kept independent: Stage1 is sourced from ``summary.stage1_done``
and Stage2 from ``summary.all_stage2_done``. ``all_training_data_done`` is used
only as an aggregate consistency signal, never as a gate for either stage.
"""

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingReadinessItem:
    key: str
    label: str
    ready: bool
    icon: str
    state_label: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataFactoryStatusOverview:
    work_dir: str
    project_name: str
    project_selected: bool
    stage1_ready: bool
    stage2_ready: bool
    all_training_data_ready: bool
    aggregate_consistent: bool
    readiness: tuple[TrainingReadinessItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_dir": self.work_dir,
            "project_name": self.project_name,
            "project_selected": self.project_selected,
            "stage1_ready": self.stage1_ready,
            "stage2_ready": self.stage2_ready,
            "all_training_data_ready": self.all_training_data_ready,
            "aggregate_consistent": self.aggregate_consistent,
            "readiness": [item.to_dict() for item in self.readiness],
        }


def _summary(status_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = status_payload if isinstance(status_payload, dict) else {}
    value = payload.get("summary")
    return value if isinstance(value, dict) else {}


def _work_dir(status_payload: dict[str, Any] | None) -> str:
    payload = status_payload if isinstance(status_payload, dict) else {}
    value = str(payload.get("work_dir") or "").strip()
    return value


def _readiness_item(*, key: str, label: str, ready: bool) -> TrainingReadinessItem:
    if ready:
        return TrainingReadinessItem(
            key=key,
            label=label,
            ready=True,
            icon="✅",
            state_label="可进入训练",
            detail="训练数据已准备完成。",
        )
    return TrainingReadinessItem(
        key=key,
        label=label,
        ready=False,
        icon="⬜",
        state_label="尚未就绪",
        detail="继续完成对应训练数据即可。",
    )


def build_status_overview(
    status_payload: dict[str, Any] | None,
) -> DataFactoryStatusOverview:
    """Build a presentation-only status summary from unified status."""

    summary = _summary(status_payload)
    work_dir = _work_dir(status_payload)
    project_selected = bool(work_dir)
    project_name = Path(work_dir).name if work_dir else "尚未选择项目"

    stage1_ready = bool(summary.get("stage1_done"))
    stage2_ready = bool(summary.get("all_stage2_done"))
    independent_all_ready = bool(stage1_ready and stage2_ready)

    aggregate_value = summary.get("all_training_data_done")
    aggregate_present = isinstance(aggregate_value, bool)
    aggregate_ready = bool(aggregate_value) if aggregate_present else independent_all_ready
    aggregate_consistent = (
        aggregate_ready == independent_all_ready
        if aggregate_present
        else True
    )

    readiness = (
        _readiness_item(key="stage1", label="Stage1", ready=stage1_ready),
        _readiness_item(key="stage2", label="Stage2", ready=stage2_ready),
    )

    return DataFactoryStatusOverview(
        work_dir=work_dir,
        project_name=project_name,
        project_selected=project_selected,
        stage1_ready=stage1_ready,
        stage2_ready=stage2_ready,
        # The user-facing aggregate is computed from the two independent source
        # signals. A stale aggregate flag must never hide an independently ready
        # Stage1 or Stage2 state.
        all_training_data_ready=independent_all_ready,
        aggregate_consistent=aggregate_consistent,
        readiness=readiness,
    )


def format_status_overview_html(
    overview_or_payload: DataFactoryStatusOverview | dict[str, Any] | None,
) -> str:
    """Render current project + independent Stage1/Stage2 readiness."""

    overview = (
        overview_or_payload
        if isinstance(overview_or_payload, DataFactoryStatusOverview)
        else build_status_overview(overview_or_payload)
    )

    project_value = (
        escape(overview.project_name)
        if overview.project_selected
        else "尚未选择项目"
    )
    work_dir_html = (
        f'<div class="vl-status-workdir">{escape(overview.work_dir)}</div>'
        if overview.work_dir
        else '<div class="vl-status-workdir">载入或创建工作目录后显示。</div>'
    )

    rows = [
        '<div class="vl-status-overview">',
        '<div class="vl-status-overview-title">当前项目</div>',
        f'<div class="vl-status-project-name">{project_value}</div>',
        work_dir_html,
        '<div class="vl-status-readiness-title">训练数据状态</div>',
        '<div class="vl-status-readiness-grid">',
    ]

    for item in overview.readiness:
        ready_class = "ready" if item.ready else "pending"
        rows.extend(
            [
                f'<div class="vl-status-readiness-item vl-status-readiness-{ready_class}">',
                '<div class="vl-status-readiness-main">',
                f'<span>{escape(item.icon)}</span>',
                f'<strong>{escape(item.label)}</strong>',
                f'<span class="vl-status-readiness-state">{escape(item.state_label)}</span>',
                '</div>',
                f'<div class="vl-status-readiness-detail">{escape(item.detail)}</div>',
                '</div>',
            ]
        )

    rows.append('</div>')
    if overview.all_training_data_ready:
        rows.append(
            '<div class="vl-status-all-ready">✅ Stage1 与 Stage2 均已准备完成。</div>'
        )
    rows.append('</div>')
    return "".join(rows)


__all__ = [
    "TrainingReadinessItem",
    "DataFactoryStatusOverview",
    "build_status_overview",
    "format_status_overview_html",
]
