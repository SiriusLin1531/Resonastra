from __future__ import annotations

"""User-facing DataFactory flow state model.

This module is deliberately presentation-only. It consumes the existing
quarantine-aware unified DataFactory status payload and converts internal
artifact stages into four user-facing phases:

1. 音频处理
2. 文本确认
3. Stage1 训练数据
4. Stage2 训练数据

It does NOT scan files, run subprocesses, decide Stage2 quarantine validity,
or replace the Training Input Contract. Those responsibilities stay in the
existing status / pipeline / training-contract layers.
"""

from dataclasses import asdict, dataclass
from html import escape
from typing import Any, Iterable


FLOW_STATUS_NOT_STARTED = "not_started"
FLOW_STATUS_BLOCKED = "blocked"
FLOW_STATUS_WAITING_USER = "waiting_user"
FLOW_STATUS_PARTIAL = "partial"
FLOW_STATUS_DONE = "done"


_STAGE2_INTERNAL_NAMES = (
    "stage2_manifest",
    "stage2_pt",
    "continuous_semantic",
    "style_cache",
    "train_val_split",
    "filter",
)


@dataclass(frozen=True)
class UserFlowStage:
    key: str
    title: str
    status: str
    icon: str
    status_label: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendedAction:
    action_id: str
    label: str
    reason: str
    priority: str = "secondary"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataFactoryUserSnapshot:
    work_dir: str
    stages: tuple[UserFlowStage, ...]
    recommended_actions: tuple[RecommendedAction, ...]
    summary: str

    def stage(self, key: str) -> UserFlowStage | None:
        for item in self.stages:
            if item.key == key:
                return item
        return None

    @property
    def recommended_action_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.recommended_actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_dir": self.work_dir,
            "stages": [item.to_dict() for item in self.stages],
            "recommended_actions": [
                item.to_dict() for item in self.recommended_actions
            ],
            "summary": self.summary,
        }


def _stage_map(status_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage in status_payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name") or "").strip()
        if name:
            result[name] = stage
    return result


def _summary(status_payload: dict[str, Any]) -> dict[str, Any]:
    value = status_payload.get("summary")
    return value if isinstance(value, dict) else {}


def _internal_status(stage: dict[str, Any] | None) -> str:
    if not isinstance(stage, dict):
        return "missing"
    return str(stage.get("status") or "missing").strip().lower()


def _is_done(stage: dict[str, Any] | None) -> bool:
    if not isinstance(stage, dict):
        return False
    return bool(stage.get("done")) or _internal_status(stage) == "done"


def _is_partial(stage: dict[str, Any] | None) -> bool:
    if not isinstance(stage, dict):
        return False
    return bool(stage.get("partial")) or _internal_status(stage) == "partial"


def _numeric_progress(values: Iterable[Any]) -> bool:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _stage_has_artifact_progress(stage: dict[str, Any] | None) -> bool:
    if not isinstance(stage, dict):
        return False
    if _is_done(stage) or _is_partial(stage):
        return True

    counts = stage.get("counts")
    if isinstance(counts, dict) and _numeric_progress(counts.values()):
        return True

    ratio = stage.get("ratio")
    return isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0


def _user_status_meta(status: str) -> tuple[str, str]:
    mapping = {
        FLOW_STATUS_DONE: ("✅", "已完成"),
        FLOW_STATUS_WAITING_USER: ("⚠️", "等待确认"),
        FLOW_STATUS_PARTIAL: ("⚠️", "部分完成"),
        FLOW_STATUS_BLOCKED: ("⬜", "等待前置步骤"),
        FLOW_STATUS_NOT_STARTED: ("⬜", "尚未开始"),
    }
    return mapping.get(status, ("⬜", "尚未开始"))


def _make_stage(
    *,
    key: str,
    title: str,
    status: str,
    detail: str,
) -> UserFlowStage:
    icon, label = _user_status_meta(status)
    return UserFlowStage(
        key=key,
        title=title,
        status=status,
        icon=icon,
        status_label=label,
        detail=detail,
    )


def build_user_flow_snapshot(
    status_payload: dict[str, Any] | None,
) -> DataFactoryUserSnapshot:
    """Convert unified internal status into a four-stage user snapshot.

    This function intentionally trusts the supplied status payload. In normal
    DataFactory v4 usage that payload comes from the already-installed
    quarantine-aware v2 scanner.
    """

    payload = status_payload if isinstance(status_payload, dict) else {}
    stages = _stage_map(payload)
    summary = _summary(payload)

    prepare_stage = stages.get("prepare")
    proofread_stage = stages.get("proofread")
    stage1_stage = stages.get("stage1_dataset")

    prepare_done = bool(summary.get("prepare_done")) or _is_done(prepare_stage)
    proofread_done = bool(summary.get("proofread_done")) or _is_done(proofread_stage)
    stage1_done = bool(summary.get("stage1_done")) or _is_done(stage1_stage)
    stage2_done = bool(summary.get("all_stage2_done"))

    # ------------------------------
    # 1. Audio processing
    # ------------------------------
    if prepare_done:
        prepare_status = FLOW_STATUS_DONE
        prepare_detail = "音频处理与基础识别产物已就绪。"
    elif _is_partial(prepare_stage) or _stage_has_artifact_progress(prepare_stage):
        prepare_status = FLOW_STATUS_PARTIAL
        prepare_detail = "检测到部分音频处理产物，可再次执行数据准备补齐。"
    else:
        prepare_status = FLOW_STATUS_NOT_STARTED
        prepare_detail = "选择原始音频后即可开始准备数据。"

    # ------------------------------
    # 2. Text confirmation
    # ------------------------------
    if proofread_done:
        proofread_status = FLOW_STATUS_DONE
        proofread_detail = "文本确认结果已就绪。"
    elif prepare_done:
        proofread_status = FLOW_STATUS_WAITING_USER
        proofread_detail = "可以启动人工校对，也可以选择无需校对直接继续。"
    else:
        proofread_status = FLOW_STATUS_BLOCKED
        proofread_detail = "完成音频处理后即可确认文本。"

    # ------------------------------
    # 3. Stage1 data
    # ------------------------------
    if stage1_done:
        stage1_status = FLOW_STATUS_DONE
        stage1_detail = "Stage1 训练数据已完成。"
    elif _is_partial(stage1_stage) or _stage_has_artifact_progress(stage1_stage):
        stage1_status = FLOW_STATUS_PARTIAL
        stage1_detail = "检测到部分 Stage1 数据，可再次生成以补齐。"
    elif proofread_done:
        stage1_status = FLOW_STATUS_NOT_STARTED
        stage1_detail = "文本确认已完成，可独立生成 Stage1 训练数据。"
    else:
        stage1_status = FLOW_STATUS_BLOCKED
        stage1_detail = "完成文本确认后即可生成 Stage1 训练数据。"

    # ------------------------------
    # 4. Stage2 data
    # ------------------------------
    stage2_internal = [stages.get(name) for name in _STAGE2_INTERNAL_NAMES]
    stage2_has_progress = any(
        _stage_has_artifact_progress(stage) for stage in stage2_internal
    )

    if stage2_done:
        stage2_status = FLOW_STATUS_DONE
        stage2_detail = "Stage2 训练数据已完成。"
    elif stage2_has_progress:
        stage2_status = FLOW_STATUS_PARTIAL
        stage2_detail = "检测到部分 Stage2 产物，可从已有结果继续生成。"
    elif proofread_done:
        stage2_status = FLOW_STATUS_NOT_STARTED
        stage2_detail = "文本确认已完成，可独立生成 Stage2 训练数据。"
    else:
        stage2_status = FLOW_STATUS_BLOCKED
        stage2_detail = "完成文本确认后即可生成 Stage2 训练数据。"

    user_stages = (
        _make_stage(
            key="prepare",
            title="音频处理",
            status=prepare_status,
            detail=prepare_detail,
        ),
        _make_stage(
            key="proofread",
            title="文本确认",
            status=proofread_status,
            detail=proofread_detail,
        ),
        _make_stage(
            key="stage1",
            title="Stage1 训练数据",
            status=stage1_status,
            detail=stage1_detail,
        ),
        _make_stage(
            key="stage2",
            title="Stage2 训练数据",
            status=stage2_status,
            detail=stage2_detail,
        ),
    )

    actions: list[RecommendedAction] = []

    if not prepare_done:
        actions.append(
            RecommendedAction(
                action_id="prepare",
                label="开始准备数据",
                reason="先完成原始音频处理与基础识别。",
                priority="primary",
            )
        )
        recommendation_summary = "当前应先完成原始音频的数据准备。"
    elif not proofread_done:
        actions.extend(
            [
                RecommendedAction(
                    action_id="launch_proofread",
                    label="启动人工校对器",
                    reason="需要人工检查或修改识别文本时使用。",
                    priority="primary",
                ),
                RecommendedAction(
                    action_id="skip_proofread",
                    label="无需校对，继续",
                    reason="确认自动识别结果无需人工修改时使用。",
                ),
                RecommendedAction(
                    action_id="confirm_proofread",
                    label="我已完成校对",
                    reason="完成外部校对后返回此页面确认。",
                ),
            ]
        )
        recommendation_summary = "音频处理已完成，下一步需要确认文本。"
    else:
        if not stage1_done:
            actions.append(
                RecommendedAction(
                    action_id="stage1_generate",
                    label="生成 Stage1 训练数据",
                    reason="Stage1 数据尚未完成。",
                    priority="primary",
                )
            )

        if not stage2_done:
            if stage2_has_progress:
                actions.append(
                    RecommendedAction(
                        action_id="stage2_resume",
                        label="继续生成 Stage2 训练数据",
                        reason="已检测到部分 Stage2 产物，优先从现有进度继续。",
                        priority="primary" if stage1_done else "secondary",
                    )
                )
            else:
                actions.append(
                    RecommendedAction(
                        action_id="stage2_generate",
                        label="生成 Stage2 训练数据",
                        reason="Stage2 数据尚未开始。",
                        priority="primary" if stage1_done else "secondary",
                    )
                )

        if stage1_done and stage2_done:
            recommendation_summary = "Stage1 与 Stage2 数据均已完成。"
        elif stage1_done:
            recommendation_summary = (
                "Stage1 数据已完成；如只训练 Stage1，可保留当前结果。"
                "如还需要 Stage2，请继续准备 Stage2 数据。"
            )
        elif stage2_done:
            recommendation_summary = (
                "Stage2 数据已完成；如只训练 Stage2，可保留当前结果。"
                "如还需要 Stage1，请继续准备 Stage1 数据。"
            )
        else:
            recommendation_summary = (
                "文本确认已完成。Stage1 与 Stage2 可以按需要独立生成。"
            )

    return DataFactoryUserSnapshot(
        work_dir=str(payload.get("work_dir") or ""),
        stages=user_stages,
        recommended_actions=tuple(actions),
        summary=recommendation_summary,
    )


def format_user_flow_cards(
    snapshot_or_payload: DataFactoryUserSnapshot | dict[str, Any] | None,
) -> str:
    """Render four user-facing flow cards as lightweight HTML."""

    snapshot = (
        snapshot_or_payload
        if isinstance(snapshot_or_payload, DataFactoryUserSnapshot)
        else build_user_flow_snapshot(snapshot_or_payload)
    )

    parts = ['<div class="vl-flow-grid">']
    for index, stage in enumerate(snapshot.stages, start=1):
        parts.extend(
            [
                f'<div class="vl-flow-card vl-flow-{escape(stage.status)}">',
                '<div class="vl-flow-card-head">',
                f'<span class="vl-flow-index">{index}</span>',
                f'<span class="vl-flow-title">{escape(stage.title)}</span>',
                f'<span class="vl-flow-state">{escape(stage.icon)} {escape(stage.status_label)}</span>',
                "</div>",
                f'<div class="vl-flow-detail">{escape(stage.detail)}</div>',
                "</div>",
            ]
        )
    parts.append("</div>")
    return "".join(parts)


def format_recommended_actions_markdown(
    snapshot_or_payload: DataFactoryUserSnapshot | dict[str, Any] | None,
) -> str:
    snapshot = (
        snapshot_or_payload
        if isinstance(snapshot_or_payload, DataFactoryUserSnapshot)
        else build_user_flow_snapshot(snapshot_or_payload)
    )

    lines = ["### 当前推荐操作", "", snapshot.summary]
    if not snapshot.recommended_actions:
        lines.extend(["", "✅ 当前没有必须继续的数据处理操作。"])
        return "\n".join(lines)

    lines.append("")
    for action in snapshot.recommended_actions:
        marker = "**推荐：** " if action.priority == "primary" else ""
        lines.append(f"- {marker}**{action.label}** — {action.reason}")
    return "\n".join(lines)


__all__ = [
    "FLOW_STATUS_NOT_STARTED",
    "FLOW_STATUS_BLOCKED",
    "FLOW_STATUS_WAITING_USER",
    "FLOW_STATUS_PARTIAL",
    "FLOW_STATUS_DONE",
    "UserFlowStage",
    "RecommendedAction",
    "DataFactoryUserSnapshot",
    "build_user_flow_snapshot",
    "format_user_flow_cards",
    "format_recommended_actions_markdown",
]
