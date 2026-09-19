from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from .stage2_report_contract import (
    analyze_stage2_split_filter_state,
    invalidate_filter_reports,
)


ResumeFunction = Callable[..., Any]


def _prefix_is_complete(legacy: Any, artifacts: Any) -> tuple[bool, dict[str, int]]:
    fewshot_rows = legacy._count_jsonl_rows(
        artifacts.stage2_manifest_fewshot_path
    ) or 0
    stage2_count = legacy._count_pt_files(artifacts.stage2_pt_root) or 0
    continuous_count = legacy._count_pt_files(
        artifacts.continuous_semantic_root
    ) or 0
    style_count = legacy._count_pt_files(artifacts.style_cache_root) or 0

    counts = {
        "fewshot_rows": int(fewshot_rows),
        "stage2_pt": int(stage2_count),
        "continuous": int(continuous_count),
        "style": int(style_count),
    }
    complete = bool(
        fewshot_rows > 0
        and stage2_count >= fewshot_rows
        and continuous_count >= stage2_count
        and style_count >= continuous_count
    )
    return complete, counts


def _overwrite_requested(kwargs: dict[str, Any]) -> bool:
    return any(
        bool(kwargs.get(key))
        for key in (
            "stage2_overwrite",
            "continuous_overwrite",
            "style_overwrite",
            "split_overwrite",
        )
    )


def build_quarantine_aware_resume(
    original_resume: ResumeFunction,
) -> ResumeFunction:
    """Wrap legacy resume with validated split/filter short-circuit logic.

    A completed move-mode quarantine intentionally leaves train+val smaller
    than style_cache. The legacy resume interprets that as an incomplete split
    and recreates train/val. This wrapper validates the split/filter reports and
    current files first. When the complete artifact chain is coherent it returns
    a normal successful resume result with skipped steps, without invoking the
    destructive split command.
    """

    def run_resumable_postprocess_pipeline(*args: Any, **kwargs: Any) -> Any:
        if args:
            return original_resume(*args, **kwargs)

        from . import resume_pipeline as legacy

        work_dir = kwargs.get("work_dir")
        if work_dir is None or not str(work_dir).strip():
            return original_resume(**kwargs)

        requested_filter_mode = (
            "move"
            if bool(kwargs.get("auto_reject_bad_samples"))
            else "report_only"
        )
        pipeline = legacy.create_data_factory_user_pipeline_service(
            show_terminal_progress=bool(
                kwargs.get("show_terminal_progress", True)
            )
        )
        adapter = pipeline.adapter
        resolved_work_dir = adapter.resolve_path(work_dir)
        artifacts = adapter.expected_artifacts(resolved_work_dir)
        state = analyze_stage2_split_filter_state(
            artifacts,
            requested_filter_mode=requested_filter_mode,
        )

        if bool(kwargs.get("split_overwrite")):
            invalidated = invalidate_filter_reports(
                artifacts,
                reason=(
                    "split_overwrite requested; the previous filter report "
                    "must not be reused after train/val is regenerated"
                ),
            )
            result = original_resume(**kwargs)
            try:
                details = result.developer_details
                if isinstance(details, dict):
                    details["split_overwrite_filter_invalidation"] = {
                        "invalidated_reports": invalidated,
                        "pre_run_state": state.to_dict(),
                    }
            except Exception:
                pass
            return result

        prefix_complete, prefix_counts = _prefix_is_complete(
            legacy,
            artifacts,
        )
        can_short_circuit = bool(
            state.completed
            and prefix_complete
            and not _overwrite_requested(kwargs)
        )
        if not can_short_circuit:
            return original_resume(**kwargs)

        started_at = legacy.now_iso()
        steps = [
            legacy._make_skipped_step(
                name="export_corrected_fewshot_stage2_manifest",
                display_name="导出 few-shot Stage2 manifest",
                message="检测到已有完整 few-shot Stage2 manifest，自动跳过。",
                report_path=artifacts.fewshot_manifest_conversion_report_path,
                extra={"contract_resume_skip": True},
            ),
            legacy._make_skipped_step(
                name="preprocess_stage2_pt_dataset",
                display_name="Stage2 .pt 预处理",
                message="检测到已有完整 Stage2 .pt，自动跳过。",
                extra={"contract_resume_skip": True},
            ),
            legacy._make_skipped_step(
                name="export_continuous_semantic_cache",
                display_name="导出 continuous semantic cache",
                message="检测到已有完整 continuous cache，自动跳过。",
                extra={"contract_resume_skip": True},
            ),
            legacy._make_skipped_step(
                name="export_fewshot_style_cache",
                display_name="导出 Style / F0 / Speaker cache",
                message="检测到已有完整 style cache，自动跳过。",
                extra={"contract_resume_skip": True},
            ),
            legacy._make_skipped_step(
                name="split_stage2_pt_dataset",
                display_name="切分 train / val",
                message=(
                    "split/filter 报告与当前文件一致；隔离后的 train/val "
                    "已完成，自动跳过 split。"
                ),
                report_path=artifacts.split_report_path,
                extra={
                    "contract_resume_skip": True,
                    "source": state.source_count,
                    "usable": state.usable_count,
                    "train": state.train_count,
                    "val": state.val_count,
                    "rejected": state.rejected_reported_count,
                },
            ),
            legacy._make_skipped_step(
                name="filter_v662_bad_style_samples",
                display_name="过滤异常样本",
                message="过滤报告有效且不早于 split report，自动跳过。",
                report_path=artifacts.filter_report_json_path,
                extra={
                    "contract_resume_skip": True,
                    "filter_mode": state.filter_mode,
                },
            ),
        ]

        return pipeline._build_user_result(
            operation="data_factory_user_postprocess",
            status=legacy.DATA_FACTORY_STATUS_SUCCEEDED,
            message=(
                "训练数据已完整存在。已验证隔离后的 train/val 与 "
                "split/filter 报告一致，未重跑 split。"
            ),
            work_dir=resolved_work_dir,
            started_at=started_at,
            result_json_filename="data_factory_user_postprocess_result.json",
            steps=steps,
            current_stage="一键生成训练数据",
            current_step="completed",
            output_summary=legacy._output_summary_from_artifacts(
                artifacts,
                bool(kwargs.get("enable_speaker_embedding", True)),
                bool(kwargs.get("auto_reject_bad_samples", False)),
            ),
            developer_details={
                "auto_resume_enabled": True,
                "resume_contract": state.to_dict(),
                "prefix_counts": prefix_counts,
                "resume_decisions": [
                    {
                        "step_name": "split_stage2_pt_dataset",
                        "action": "skip",
                        "reason": (
                            "validated_split_and_filter_outputs_exist"
                        ),
                        "details": state.to_dict(),
                    },
                    {
                        "step_name": "filter_v662_bad_style_samples",
                        "action": "skip",
                        "reason": "validated_filter_report_is_current",
                        "details": {
                            "mode": state.filter_mode,
                            "freshness_ok": (
                                state.split_report_freshness_ok
                            ),
                        },
                    },
                ],
            },
            extra={
                "auto_reject_bad_samples": bool(
                    kwargs.get("auto_reject_bad_samples", False)
                ),
                "filter_mode": requested_filter_mode,
                "enable_speaker_embedding": bool(
                    kwargs.get("enable_speaker_embedding", True)
                ),
                "auto_resume_enabled": True,
                "quarantine_aware_resume": True,
                "num_skipped_steps": len(steps),
                "num_executed_steps": 0,
            },
        )

    run_resumable_postprocess_pipeline.__name__ = (
        "run_resumable_postprocess_pipeline"
    )
    return run_resumable_postprocess_pipeline


def install_resume_pipeline_v2() -> None:
    """Install the runtime wrapper before WebUI binds the resume callback."""

    from . import resume_pipeline as legacy_resume
    import src.voicelab_user.data_factory as package

    current = legacy_resume.run_resumable_postprocess_pipeline
    if getattr(current, "_voicelab_quarantine_aware", False):
        wrapped = current
    else:
        wrapped = build_quarantine_aware_resume(current)
        setattr(wrapped, "_voicelab_quarantine_aware", True)
        legacy_resume.run_resumable_postprocess_pipeline = wrapped

    package.run_resumable_postprocess_pipeline = wrapped

    base_module = sys.modules.get("scripts.webui_data_factory_user")
    if base_module is not None:
        setattr(
            base_module,
            "run_resumable_postprocess_pipeline",
            wrapped,
        )


__all__ = [
    "build_quarantine_aware_resume",
    "install_resume_pipeline_v2",
]
