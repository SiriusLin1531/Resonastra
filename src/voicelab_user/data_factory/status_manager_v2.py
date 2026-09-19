from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .adapter import create_data_factory_adapter
from .stage2_report_contract import analyze_stage2_split_filter_state


ScanFunction = Callable[..., dict[str, Any]]


def _stage_by_name(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in payload.get("stages") or []:
        if isinstance(stage, dict) and stage.get("name") == name:
            return stage
    return None


def _write_payload(payload: dict[str, Any]) -> None:
    path_value = payload.get("status_json_path")
    if not path_value:
        return
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def build_quarantine_aware_scan(
    original_scan: ScanFunction,
) -> ScanFunction:
    """Wrap the legacy unified scanner with Stage2 quarantine semantics."""

    def scan_unified_data_factory_status(
        work_dir: str | Path,
        *,
        write_status_json: bool = True,
    ) -> dict[str, Any]:
        payload = original_scan(work_dir, write_status_json=False)

        adapter = create_data_factory_adapter()
        resolved_work_dir = adapter.resolve_path(work_dir)
        artifacts = adapter.expected_artifacts(resolved_work_dir)
        state = analyze_stage2_split_filter_state(artifacts)

        split_stage = _stage_by_name(payload, "train_val_split")
        if split_stage is not None:
            split_stage["status"] = (
                "done"
                if state.split_done
                else "partial"
                if state.found_count > 0
                else "missing"
            )
            split_stage["done"] = bool(state.split_done)
            split_stage["partial"] = bool(
                not state.split_done and state.found_count > 0
            )
            split_stage["ratio"] = state.completion_ratio
            split_stage["counts"] = {
                "source": state.source_count,
                "expected": state.usable_count,
                "usable": state.usable_count,
                "train": state.train_count,
                "val": state.val_count,
                "found": state.found_count,
                "rejected": state.rejected_reported_count,
            }
            split_stage["message"] = (
                "Stage2 split 已完成："
                f"source {state.source_count}, "
                f"rejected {state.rejected_reported_count}, "
                f"usable {state.usable_count}, "
                f"train {state.train_count}, val {state.val_count}。"
                if state.split_done
                else "Stage2 split / filter 报告尚未形成一致状态。"
            )
            split_stage["report_contract"] = state.to_dict()

        filter_stage = _stage_by_name(payload, "filter")
        if filter_stage is not None:
            filter_stage["status"] = (
                "done"
                if state.filter_done
                else "blocked"
                if not state.split_done
                else "missing"
            )
            filter_stage["done"] = bool(state.filter_done)
            filter_stage["counts"] = {
                "source": state.source_count,
                "ok": state.usable_count,
                "bad": state.rejected_reported_count,
                "rejected_existing": state.rejected_existing_count,
            }
            filter_stage["message"] = (
                "异常样本报告有效，且与当前 split 产物一致。"
                if state.filter_done
                else (
                    "异常样本报告缺失、过期或与当前 split / 文件数量不一致。"
                )
            )
            filter_stage["report_contract"] = state.to_dict()

        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = {}
            payload["summary"] = summary

        summary["split_done"] = bool(state.split_done)
        summary["filter_done"] = bool(state.filter_done)
        summary["stage2_source_count"] = state.source_count
        summary["stage2_usable_count"] = state.usable_count
        summary["stage2_rejected_count"] = state.rejected_reported_count
        summary["stage2_split_filter_contract"] = state.to_dict()

        stage2_prefix_done = all(
            bool(summary.get(key))
            for key in (
                "stage2_manifest_done",
                "stage2_pt_done",
                "continuous_done",
                "style_done",
            )
        )
        summary["all_stage2_done"] = bool(
            stage2_prefix_done and state.split_done and state.filter_done
        )
        summary["all_training_data_done"] = bool(
            summary.get("stage1_done") and summary["all_stage2_done"]
        )

        next_actions = [
            str(item)
            for item in payload.get("next_actions") or []
            if "继续生成 Stage2" not in str(item)
            and "Stage1 / Stage2 数据产物均已就绪" not in str(item)
        ]
        if stage2_prefix_done and not state.filter_done:
            next_actions.append(
                "点击“继续生成 Stage2 训练数据”刷新 split / 异常样本报告。"
            )
        if summary["all_training_data_done"]:
            next_actions.append(
                "Stage1 / Stage2 数据产物均已就绪，可进入训练阶段。"
            )
        payload["next_actions"] = list(dict.fromkeys(next_actions))

        if write_status_json:
            _write_payload(payload)
        return payload

    scan_unified_data_factory_status.__name__ = (
        "scan_unified_data_factory_status"
    )
    setattr(scan_unified_data_factory_status, "_voicelab_quarantine_aware", True)
    return scan_unified_data_factory_status


def get_quarantine_aware_status_scanner() -> ScanFunction:
    """Return a quarantine-aware scanner without requiring WebUI patch order.

    Training UI runs in a separate entrypoint from DataFactory UI. It therefore
    must not depend on DataFactory's v3 launcher having installed the runtime
    monkeypatch first.
    """

    from . import status_manager as legacy_status

    current = legacy_status.scan_unified_data_factory_status
    if getattr(current, "_voicelab_quarantine_aware", False):
        return current
    return build_quarantine_aware_scan(current)


def scan_unified_data_factory_status_v2(
    work_dir: str | Path,
    *,
    write_status_json: bool = True,
) -> dict[str, Any]:
    """Scan with Stage2 quarantine semantics in any process/entrypoint."""

    scanner = get_quarantine_aware_status_scanner()
    return scanner(work_dir, write_status_json=write_status_json)


def install_status_manager_v2() -> None:
    """Patch the legacy status module before WebUI imports its functions."""

    from . import status_manager as legacy_status

    current = legacy_status.scan_unified_data_factory_status
    if getattr(current, "_voicelab_quarantine_aware", False):
        return

    legacy_status.scan_unified_data_factory_status = (
        get_quarantine_aware_status_scanner()
    )


__all__ = [
    "build_quarantine_aware_scan",
    "get_quarantine_aware_status_scanner",
    "scan_unified_data_factory_status_v2",
    "install_status_manager_v2",
]
