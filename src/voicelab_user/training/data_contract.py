from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.voicelab_user.data_factory.status_manager_v2 import (
    scan_unified_data_factory_status_v2,
)


TRAINING_INTERFACE_SCHEMA_VERSION = "voicelab_training_input_contract_v2"
TRAINING_INTERFACE_FILENAME = "training_interface.json"


# ============================================================
# Helpers
# ============================================================
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return p


def _stage_paths(stage: dict[str, Any]) -> dict[str, str]:
    paths = stage.get("paths")
    return paths if isinstance(paths, dict) else {}


def _stage_counts(stage: dict[str, Any]) -> dict[str, Any]:
    counts = stage.get("counts")
    return counts if isinstance(counts, dict) else {}


def _stage_report_contract(stage: dict[str, Any]) -> dict[str, Any]:
    value = stage.get("report_contract")
    return value if isinstance(value, dict) else {}


# ============================================================
# Status loading
# ============================================================
def load_data_factory_status(
    work_dir: str | Path,
    *,
    refresh: bool = True,
) -> dict[str, Any]:
    """Load quarantine-aware unified DataFactory status.

    Training UI is a separate process from DataFactory UI, so this function
    calls the v2 scanner directly instead of relying on DataFactory's WebUI
    launcher having installed a runtime monkeypatch first.
    """

    return scan_unified_data_factory_status_v2(
        work_dir,
        write_status_json=refresh,
    )


# ============================================================
# Contract builder
# ============================================================
def build_training_input_contract(
    work_dir: str | Path,
    *,
    refresh_status: bool = True,
) -> dict[str, Any]:
    """Convert quarantine-aware DataFactory status into Training UI contract."""

    work_dir = Path(work_dir).expanduser().resolve(strict=False)

    status = load_data_factory_status(
        work_dir,
        refresh=refresh_status,
    )

    stage_map = {
        stage.get("name"): stage
        for stage in (status.get("stages") or [])
        if isinstance(stage, dict) and stage.get("name")
    }

    summary = status.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    stage1_stage = stage_map.get("stage1_dataset", {})
    stage2_pt_stage = stage_map.get("stage2_pt", {})
    split_stage = stage_map.get("train_val_split", {})
    filter_stage = stage_map.get("filter", {})

    split_filter_contract = summary.get("stage2_split_filter_contract")
    if not isinstance(split_filter_contract, dict):
        split_filter_contract = _stage_report_contract(split_stage)
    if not split_filter_contract:
        split_filter_contract = _stage_report_contract(filter_stage)

    stage2_ready = bool(summary.get("all_stage2_done"))
    stage1_ready = bool(summary.get("stage1_done"))

    contract_path = work_dir / TRAINING_INTERFACE_FILENAME

    contract = {
        "schema_version": TRAINING_INTERFACE_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "work_dir": str(work_dir),
        "data_factory_status_path": str(
            status.get("status_json_path") or ""
        ),
        "training_interface_path": str(contract_path),
        "status_scanner": "quarantine_aware_v2",
        "readiness": {
            "stage1_ready": stage1_ready,
            "stage2_ready": stage2_ready,
            "all_training_data_ready": bool(
                summary.get("all_training_data_done")
            ),
        },
        "stage1": {
            "ready": stage1_ready,
            "paths": _stage_paths(stage1_stage),
            "counts": _stage_counts(stage1_stage),
        },
        "stage2": {
            "ready": stage2_ready,
            "pt_paths": _stage_paths(stage2_pt_stage),
            "pt_counts": _stage_counts(stage2_pt_stage),
            "split_paths": _stage_paths(split_stage),
            "split_counts": _stage_counts(split_stage),
            "filter_paths": _stage_paths(filter_stage),
            "filter_counts": _stage_counts(filter_stage),
            "split_filter_contract": split_filter_contract,
            "source_count": summary.get("stage2_source_count"),
            "usable_count": summary.get("stage2_usable_count"),
            "rejected_count": summary.get("stage2_rejected_count"),
        },
        "recommended_actions": [],
        "source_status": {
            "summary": summary,
            "next_actions": status.get("next_actions") or [],
        },
    }

    actions: list[str] = []

    if contract["readiness"]["stage1_ready"]:
        actions.append("stage1_training_ready")
    else:
        actions.append("stage1_training_data_missing")

    if contract["readiness"]["stage2_ready"]:
        actions.append("stage2_training_ready")
    else:
        actions.append("stage2_training_data_missing")

    contract["recommended_actions"] = actions

    return contract


# ============================================================
# Validation
# ============================================================
def validate_stage1_ready(contract: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []

    stage1 = contract.get("stage1") or {}

    if not stage1.get("ready"):
        problems.append("Stage1 数据未完成")

    paths = stage1.get("paths") or {}

    for key in [
        "train_manifest",
        "val_manifest",
        "frontend_root",
        "semantic_root",
    ]:
        if not paths.get(key):
            problems.append(f"Stage1 缺少路径: {key}")

    return len(problems) == 0, problems


def validate_stage2_ready(contract: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []

    stage2 = contract.get("stage2") or {}
    report_contract = stage2.get("split_filter_contract")
    if not isinstance(report_contract, dict):
        report_contract = {}

    if not stage2.get("ready"):
        problems.append("Stage2 数据未完成")

        reasons = report_contract.get("reasons")
        if isinstance(reasons, (list, tuple)):
            for reason in reasons:
                reason_text = str(reason).strip()
                if reason_text:
                    problems.append(
                        f"Stage2 split/filter contract: {reason_text}"
                    )

        if report_contract:
            if not report_contract.get("split_done"):
                problems.append("Stage2 train/val split 未通过完整性校验")
            if not report_contract.get("filter_done"):
                problems.append("Stage2 异常样本过滤报告未通过完整性校验")

    for group_name in ["pt_paths", "split_paths", "filter_paths"]:
        group = stage2.get(group_name) or {}
        if not group:
            problems.append(f"Stage2 缺少路径组: {group_name}")

    return len(problems) == 0, list(dict.fromkeys(problems))


# ============================================================
# Persistence / formatting
# ============================================================
def write_training_interface_json(
    work_dir: str | Path,
    *,
    refresh_status: bool = True,
) -> Path:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    contract = build_training_input_contract(
        work_dir,
        refresh_status=refresh_status,
    )

    return _write_json(
        work_dir / TRAINING_INTERFACE_FILENAME,
        contract,
    )


def format_training_contract_markdown(contract: dict[str, Any]) -> str:
    readiness = contract.get("readiness") or {}
    stage2 = contract.get("stage2") or {}

    return "\n".join(
        [
            "### Training Input Contract",
            "",
            f"- work_dir: `{contract.get('work_dir')}`",
            f"- status scanner: `{contract.get('status_scanner')}`",
            f"- Stage1 ready: `{readiness.get('stage1_ready')}`",
            f"- Stage2 ready: `{readiness.get('stage2_ready')}`",
            f"- All training data ready: `{readiness.get('all_training_data_ready')}`",
            f"- Stage2 source / rejected / usable: "
            f"`{stage2.get('source_count')}` / "
            f"`{stage2.get('rejected_count')}` / "
            f"`{stage2.get('usable_count')}`",
            "",
            "Recommended actions:",
            *[
                f"- {item}"
                for item in contract.get("recommended_actions") or []
            ],
        ]
    )


__all__ = [
    "TRAINING_INTERFACE_SCHEMA_VERSION",
    "TRAINING_INTERFACE_FILENAME",
    "load_data_factory_status",
    "build_training_input_contract",
    "validate_stage1_ready",
    "validate_stage2_ready",
    "write_training_interface_json",
    "format_training_contract_markdown",
]
