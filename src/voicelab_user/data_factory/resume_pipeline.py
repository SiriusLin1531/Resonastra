from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Local imports
# 本地导入
# ============================================================
from .result import DATA_FACTORY_STATUS_FAILED, DATA_FACTORY_STATUS_SUCCEEDED
from .user_pipeline import (
    UserPipelineResult,
    UserPipelineStepResult,
    _failure_reason_from_result,
    _failed_step_from_result,
    _step_from_data_factory_result,
    create_data_factory_user_pipeline_service,
    now_iso,
)


PathLike = str | Path


# ============================================================
# Artifact checks
# 产物检查
# ============================================================
def _path_exists(path: Optional[Path]) -> bool:
    return path is not None and Path(path).exists()


def _file_has_content(path: Optional[Path]) -> bool:
    return path is not None and Path(path).is_file() and Path(path).stat().st_size > 0


def _count_jsonl_rows(path: Optional[Path]) -> Optional[int]:
    if path is None or not Path(path).is_file():
        return None

    count = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_pt_files(path: Optional[Path]) -> Optional[int]:
    if path is None or not Path(path).is_dir():
        return None
    return len(list(Path(path).glob("*.pt")))


def _is_output_newer_or_same(input_path: Optional[Path], output_path: Optional[Path]) -> bool:
    if input_path is None or output_path is None:
        return False
    if not Path(input_path).exists() or not Path(output_path).exists():
        return False
    return Path(output_path).stat().st_mtime >= Path(input_path).stat().st_mtime


def _read_json_dict(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        import json

        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _make_skipped_step(
    *,
    name: str,
    display_name: str,
    message: str,
    report_path: Optional[Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> UserPipelineStepResult:
    return UserPipelineStepResult(
        name=name,
        display_name=display_name,
        status=DATA_FACTORY_STATUS_SUCCEEDED,
        message=message,
        report_path=report_path,
        extra={
            "auto_resume_skipped": True,
            **(extra or {}),
        },
    )


# ============================================================
# Resumable postprocess pipeline
# 断点续跑后处理流程
# ============================================================
def run_resumable_postprocess_pipeline(
    *,
    work_dir: PathLike,
    show_terminal_progress: bool = True,

    # few-shot stage2 manifest
    prompt_mode: str = "speaker_pool",
    min_prompt_sec: float = 3.0,
    max_prompt_sec: float = 10.0,
    prefer_prompt_sec: float = 6.0,
    allow_self_prompt: bool = True,
    fixed_prompt_wav_path: Optional[PathLike] = None,
    strict_prompt_duration: bool = True,
    skip_invalid_rows: bool = False,
    allow_empty_text: bool = False,

    # Stage2 .pt preprocess
    stage2_device: str = "cuda",
    stage2_use_half: bool = False,
    stage2_overwrite: bool = False,
    target_sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,

    # continuous semantic cache
    continuous_device: str = "cuda",
    continuous_use_half: bool = False,
    continuous_dtype: str = "float32",
    continuous_require_predicted: bool = False,
    continuous_overwrite: bool = False,
    continuous_max_files: Optional[int] = None,
    continuous_copy_non_pt_files: bool = False,
    sovits_checkpoint_path: Optional[PathLike] = None,

    # style / F0 / speaker cache
    style_device: str = "cuda",
    style_overwrite: bool = False,
    style_copy_non_pt_files: bool = False,
    style_recursive: bool = False,
    style_max_files: Optional[int] = None,
    extract_prompt_acoustic: bool = True,
    extract_energy: bool = True,
    extract_f0: bool = True,
    enable_speaker_embedding: bool = True,
    prefer_sample_acoustic_meta: bool = True,
    f0_min_hz: float = 50.0,
    f0_max_hz: float = 1100.0,
    speaker_backend: str = "speechbrain_ecapa",
    speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
    speaker_savedir: PathLike = "pretrained_models/speechbrain_spkrec_ecapa_voxceleb",
    speaker_device: Optional[str] = None,
    speaker_sample_rate: int = 16000,
    speaker_normalize: bool = True,

    # split
    split_train_ratio: Optional[float] = 0.9,
    split_val_count: Optional[int] = None,
    split_shuffle: bool = True,
    split_seed: int = 2026,
    split_mode: str = "copy",
    split_overwrite: bool = False,

    # filter
    auto_reject_bad_samples: bool = False,
    filter_recursive: bool = False,
    require_target_f0: bool = True,
    require_target_voiced_mask: bool = True,
    reject_all_unvoiced_target_f0: bool = True,
    require_prompt_f0: bool = False,
    speaker_dim: int = 192,

    # runtime
    timeout_seconds: Optional[float] = None,
) -> UserPipelineResult:
    """Run postprocess with artifact-based auto resume.

    中文说明：
        基于工作目录中的既有产物自动断点续跑。

        策略：
        - 若某一步的关键产物已经完整存在，则自动跳过；
        - 若用户显式勾选 overwrite，则对应步骤仍会重跑；
        - 对部分完成的 .pt 目录，保持 overwrite=False 调用原脚本，让原脚本自行补齐缺失样本；
        - 任一步失败仍立即停止。
    """

    pipeline = create_data_factory_user_pipeline_service(
        show_terminal_progress=bool(show_terminal_progress),
    )
    adapter = pipeline.adapter
    service = pipeline.service

    started_at = now_iso()
    resolved_work_dir = adapter.resolve_path(work_dir)
    artifacts = adapter.expected_artifacts(resolved_work_dir)
    steps: list[UserPipelineStepResult] = []
    resume_decisions: list[dict[str, Any]] = []

    def add_decision(
        *,
        step_name: str,
        action: str,
        reason: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        resume_decisions.append(
            {
                "step_name": step_name,
                "action": action,
                "reason": reason,
                "details": details or {},
            }
        )

    def stop_with_result(
        *,
        message: str,
        failed_step: Optional[str],
        failure_reason: Optional[str],
        failure_hint: Optional[str],
    ) -> UserPipelineResult:
        return pipeline._build_user_result(
            operation="data_factory_user_postprocess",
            status=DATA_FACTORY_STATUS_FAILED,
            message=message,
            work_dir=resolved_work_dir,
            started_at=started_at,
            result_json_filename="data_factory_user_postprocess_result.json",
            steps=steps,
            current_stage="一键生成训练数据",
            current_step=failed_step,
            failed_step=failed_step,
            failure_reason=failure_reason,
            failure_hint=failure_hint,
            output_summary=_output_summary_from_artifacts(artifacts, enable_speaker_embedding, auto_reject_bad_samples),
            developer_details={
                "auto_resume_enabled": True,
                "resume_decisions": resume_decisions,
            },
        )

    try:
        if artifacts.manifest_corrected_path is None or not artifacts.manifest_corrected_path.exists():
            raise FileNotFoundError(
                "manifest.corrected.jsonl not found. "
                "请先完成人工校对并重建 corrected manifest，或点击跳过人工校对。"
            )

        # ----------------------------------------------------
        # 1. Export corrected few-shot Stage2 manifest
        # ----------------------------------------------------
        fewshot_rows = _count_jsonl_rows(artifacts.stage2_manifest_fewshot_path)
        fewshot_complete = bool(
            fewshot_rows
            and fewshot_rows > 0
            and _is_output_newer_or_same(
                artifacts.manifest_corrected_path,
                artifacts.stage2_manifest_fewshot_path,
            )
        )

        if fewshot_complete:
            add_decision(
                step_name="export_corrected_fewshot_stage2_manifest",
                action="skip",
                reason="stage2_manifest.fewshot.jsonl 已存在且不早于 manifest.corrected.jsonl。",
                details={"fewshot_rows": fewshot_rows},
            )
            steps.append(
                _make_skipped_step(
                    name="export_corrected_fewshot_stage2_manifest",
                    display_name="导出 few-shot Stage2 manifest",
                    message="检测到已有 few-shot Stage2 manifest，自动跳过。",
                    report_path=artifacts.fewshot_manifest_conversion_report_path,
                    extra={"fewshot_rows": fewshot_rows},
                )
            )
        else:
            add_decision(
                step_name="export_corrected_fewshot_stage2_manifest",
                action="run",
                reason="few-shot Stage2 manifest 缺失、为空或已过期。",
            )
            result = service.export_corrected_fewshot_stage2_manifest(
                work_dir=resolved_work_dir,
                prompt_mode=prompt_mode,
                min_prompt_sec=min_prompt_sec,
                max_prompt_sec=max_prompt_sec,
                prefer_prompt_sec=prefer_prompt_sec,
                allow_self_prompt=allow_self_prompt,
                fixed_prompt_wav_path=fixed_prompt_wav_path,
                strict_prompt_duration=strict_prompt_duration,
                skip_invalid_rows=skip_invalid_rows,
                allow_empty_text=allow_empty_text,
                timeout_seconds=timeout_seconds,
            )
            step = _step_from_data_factory_result(
                name="export_corrected_fewshot_stage2_manifest",
                display_name="导出 few-shot Stage2 manifest",
                result=result,
            )
            steps.append(step)
            if result.status != DATA_FACTORY_STATUS_SUCCEEDED:
                reason, hint = _failure_reason_from_result(result)
                return stop_with_result(
                    message="导出 few-shot Stage2 manifest 失败。",
                    failed_step="export_corrected_fewshot_stage2_manifest",
                    failure_reason=reason,
                    failure_hint=hint,
                )

        artifacts = adapter.expected_artifacts(resolved_work_dir)
        expected_stage2_items = _count_jsonl_rows(artifacts.stage2_manifest_fewshot_path) or 0

        # ----------------------------------------------------
        # 2. Stage2 .pt preprocessing
        # ----------------------------------------------------
        stage2_pt_count = _count_pt_files(artifacts.stage2_pt_root) or 0
        stage2_complete = bool(
            expected_stage2_items > 0
            and stage2_pt_count >= expected_stage2_items
            and not bool(stage2_overwrite)
        )

        if stage2_complete:
            add_decision(
                step_name="preprocess_stage2_pt_dataset",
                action="skip",
                reason="Stage2 .pt 数量已达到 stage2_manifest.fewshot.jsonl 行数。",
                details={"expected": expected_stage2_items, "found": stage2_pt_count},
            )
            steps.append(
                _make_skipped_step(
                    name="preprocess_stage2_pt_dataset",
                    display_name="Stage2 .pt 预处理",
                    message=f"检测到 {stage2_pt_count}/{expected_stage2_items} 个 .pt，自动跳过。",
                    extra={"expected": expected_stage2_items, "found": stage2_pt_count},
                )
            )
        else:
            add_decision(
                step_name="preprocess_stage2_pt_dataset",
                action="run",
                reason="Stage2 .pt 缺失、数量不足，或用户要求覆盖。",
                details={"expected": expected_stage2_items, "found": stage2_pt_count, "overwrite": bool(stage2_overwrite)},
            )
            result = service.preprocess_stage2_pt_dataset(
                work_dir=resolved_work_dir,
                output_dir=None,
                device=stage2_device,
                use_half=stage2_use_half,
                overwrite=stage2_overwrite,
                target_sr=target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                timeout_seconds=timeout_seconds,
            )
            step = _step_from_data_factory_result(
                name="preprocess_stage2_pt_dataset",
                display_name="Stage2 .pt 预处理",
                result=result,
            )
            steps.append(step)
            if result.status != DATA_FACTORY_STATUS_SUCCEEDED:
                reason, hint = _failure_reason_from_result(result)
                return stop_with_result(
                    message="Stage2 .pt 预处理失败。",
                    failed_step="preprocess_stage2_pt_dataset",
                    failure_reason=reason,
                    failure_hint=hint,
                )

        artifacts = adapter.expected_artifacts(resolved_work_dir)
        stage2_pt_count = _count_pt_files(artifacts.stage2_pt_root) or 0

        # ----------------------------------------------------
        # 3. Continuous semantic cache
        # ----------------------------------------------------
        continuous_count = _count_pt_files(artifacts.continuous_semantic_root) or 0
        continuous_complete = bool(
            stage2_pt_count > 0
            and continuous_count >= stage2_pt_count
            and not bool(continuous_overwrite)
        )

        if continuous_complete:
            add_decision(
                step_name="export_continuous_semantic_cache",
                action="skip",
                reason="continuous cache 数量已达到 Stage2 .pt 数量。",
                details={"expected": stage2_pt_count, "found": continuous_count},
            )
            steps.append(
                _make_skipped_step(
                    name="export_continuous_semantic_cache",
                    display_name="导出 continuous semantic cache",
                    message=f"检测到 {continuous_count}/{stage2_pt_count} 个 continuous .pt，自动跳过。",
                    extra={"expected": stage2_pt_count, "found": continuous_count},
                )
            )
        else:
            add_decision(
                step_name="export_continuous_semantic_cache",
                action="run",
                reason="continuous cache 缺失、数量不足，或用户要求覆盖。",
                details={"expected": stage2_pt_count, "found": continuous_count, "overwrite": bool(continuous_overwrite)},
            )
            result = service.export_continuous_semantic_cache(
                work_dir=resolved_work_dir,
                input_root=None,
                output_root=None,
                sovits_checkpoint_path=sovits_checkpoint_path,
                device=continuous_device,
                use_half=continuous_use_half,
                continuous_dtype=continuous_dtype,
                require_predicted=continuous_require_predicted,
                overwrite=continuous_overwrite,
                max_files=continuous_max_files,
                copy_non_pt_files=continuous_copy_non_pt_files,
                timeout_seconds=timeout_seconds,
            )
            step = _step_from_data_factory_result(
                name="export_continuous_semantic_cache",
                display_name="导出 continuous semantic cache",
                result=result,
            )
            steps.append(step)
            if result.status != DATA_FACTORY_STATUS_SUCCEEDED:
                reason, hint = _failure_reason_from_result(result)
                return stop_with_result(
                    message="Continuous semantic cache 导出失败。",
                    failed_step="export_continuous_semantic_cache",
                    failure_reason=reason,
                    failure_hint=hint,
                )

        artifacts = adapter.expected_artifacts(resolved_work_dir)
        continuous_count = _count_pt_files(artifacts.continuous_semantic_root) or 0

        # ----------------------------------------------------
        # 4. Style / F0 / speaker cache
        # ----------------------------------------------------
        style_count = _count_pt_files(artifacts.style_cache_root) or 0
        style_complete = bool(
            continuous_count > 0
            and style_count >= continuous_count
            and not bool(style_overwrite)
        )

        if style_complete:
            add_decision(
                step_name="export_fewshot_style_cache",
                action="skip",
                reason="style cache 数量已达到 continuous cache 数量。",
                details={"expected": continuous_count, "found": style_count},
            )
            steps.append(
                _make_skipped_step(
                    name="export_fewshot_style_cache",
                    display_name="导出 Style / F0 / Speaker cache",
                    message=f"检测到 {style_count}/{continuous_count} 个 style cache .pt，自动跳过。",
                    extra={"expected": continuous_count, "found": style_count},
                )
            )
        else:
            add_decision(
                step_name="export_fewshot_style_cache",
                action="run",
                reason="style cache 缺失、数量不足，或用户要求覆盖。",
                details={"expected": continuous_count, "found": style_count, "overwrite": bool(style_overwrite)},
            )
            result = service.export_fewshot_style_cache(
                work_dir=resolved_work_dir,
                input_root=None,
                output_root=None,
                device=style_device,
                overwrite=style_overwrite,
                copy_non_pt_files=style_copy_non_pt_files,
                recursive=style_recursive,
                max_files=style_max_files,
                extract_prompt_acoustic=extract_prompt_acoustic,
                extract_energy=extract_energy,
                extract_f0=extract_f0,
                extract_speaker_embedding=enable_speaker_embedding,
                prefer_sample_acoustic_meta=prefer_sample_acoustic_meta,
                sample_rate=target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                f0_min_hz=f0_min_hz,
                f0_max_hz=f0_max_hz,
                speaker_backend=speaker_backend,
                speaker_model_source=speaker_model_source,
                speaker_savedir=speaker_savedir,
                speaker_device=speaker_device,
                speaker_sample_rate=speaker_sample_rate,
                speaker_normalize=speaker_normalize,
                fail_on_error=True,
                timeout_seconds=timeout_seconds,
            )
            step = _step_from_data_factory_result(
                name="export_fewshot_style_cache",
                display_name="导出 Style / F0 / Speaker cache",
                result=result,
            )
            steps.append(step)
            if result.status != DATA_FACTORY_STATUS_SUCCEEDED:
                reason, hint = _failure_reason_from_result(result)
                return stop_with_result(
                    message="Style / F0 / Speaker cache 导出失败。",
                    failed_step="export_fewshot_style_cache",
                    failure_reason=reason,
                    failure_hint=hint,
                )

        artifacts = adapter.expected_artifacts(resolved_work_dir)
        style_count = _count_pt_files(artifacts.style_cache_root) or 0

        # ----------------------------------------------------
        # 5. Train / val split
        # ----------------------------------------------------
        train_count = _count_pt_files(artifacts.train_pt_root) or 0
        val_count = _count_pt_files(artifacts.val_pt_root) or 0
        split_complete = bool(
            style_count > 0
            and train_count + val_count >= style_count
            and _file_has_content(artifacts.split_report_path)
            and not bool(split_overwrite)
        )

        if split_complete:
            add_decision(
                step_name="split_stage2_pt_dataset",
                action="skip",
                reason="train/val 数量已覆盖 style cache，且 split_report.json 存在。",
                details={"expected": style_count, "train": train_count, "val": val_count},
            )
            steps.append(
                _make_skipped_step(
                    name="split_stage2_pt_dataset",
                    display_name="切分 train / val",
                    message=f"检测到 train {train_count} + val {val_count} / {style_count}，自动跳过。",
                    report_path=artifacts.split_report_path,
                    extra={"expected": style_count, "train": train_count, "val": val_count},
                )
            )
        else:
            add_decision(
                step_name="split_stage2_pt_dataset",
                action="run",
                reason="train/val 缺失、数量不足、缺少 split report，或用户要求覆盖。",
                details={"expected": style_count, "train": train_count, "val": val_count, "overwrite": bool(split_overwrite)},
            )
            result = service.split_stage2_pt_dataset(
                work_dir=resolved_work_dir,
                data_root=None,
                train_out=None,
                val_out=None,
                train_ratio=split_train_ratio,
                val_count=split_val_count,
                shuffle=split_shuffle,
                seed=split_seed,
                mode=split_mode,
                overwrite=split_overwrite,
                report_path=None,
                timeout_seconds=timeout_seconds,
            )
            step = _step_from_data_factory_result(
                name="split_stage2_pt_dataset",
                display_name="切分 train / val",
                result=result,
            )
            steps.append(step)
            if result.status != DATA_FACTORY_STATUS_SUCCEEDED:
                reason, hint = _failure_reason_from_result(result)
                return stop_with_result(
                    message="Train / val split 失败。",
                    failed_step="split_stage2_pt_dataset",
                    failure_reason=reason,
                    failure_hint=hint,
                )

        artifacts = adapter.expected_artifacts(resolved_work_dir)

        # ----------------------------------------------------
        # 6. Bad style sample filter
        # ----------------------------------------------------
        filter_mode = "move" if bool(auto_reject_bad_samples) else "report_only"
        filter_report = _read_json_dict(artifacts.filter_report_json_path)
        existing_filter_mode = str(filter_report.get("mode") or filter_report.get("filter_mode") or "")
        filter_complete = bool(
            _file_has_content(artifacts.filter_report_json_path)
            and (
                not existing_filter_mode
                or existing_filter_mode == filter_mode
                or filter_mode == "report_only"
            )
        )

        if filter_complete:
            add_decision(
                step_name="filter_v662_bad_style_samples",
                action="skip",
                reason="异常样本过滤报告已存在且模式兼容。",
                details={"requested_mode": filter_mode, "existing_mode": existing_filter_mode},
            )
            steps.append(
                _make_skipped_step(
                    name="filter_v662_bad_style_samples",
                    display_name="过滤异常样本",
                    message="检测到异常样本过滤报告，自动跳过。",
                    report_path=artifacts.filter_report_json_path,
                    extra={"requested_mode": filter_mode, "existing_mode": existing_filter_mode},
                )
            )
        else:
            add_decision(
                step_name="filter_v662_bad_style_samples",
                action="run",
                reason="异常样本过滤报告缺失或模式不兼容。",
                details={"requested_mode": filter_mode, "existing_mode": existing_filter_mode},
            )
            result = service.filter_v662_bad_style_samples(
                work_dir=resolved_work_dir,
                roots=None,
                rejected_dir=None,
                recursive=filter_recursive,
                mode=filter_mode,
                require_target_f0=require_target_f0,
                require_target_voiced_mask=require_target_voiced_mask,
                reject_all_unvoiced_target_f0=reject_all_unvoiced_target_f0,
                require_prompt_f0=require_prompt_f0,
                check_speaker_fields=enable_speaker_embedding,
                speaker_dim=speaker_dim,
                report_json=None,
                report_csv=None,
                timeout_seconds=timeout_seconds,
            )
            step = _step_from_data_factory_result(
                name="filter_v662_bad_style_samples",
                display_name="过滤异常样本",
                result=result,
            )
            steps.append(step)
            if result.status != DATA_FACTORY_STATUS_SUCCEEDED:
                reason, hint = _failure_reason_from_result(result)
                return stop_with_result(
                    message="异常样本过滤失败。",
                    failed_step="filter_v662_bad_style_samples",
                    failure_reason=reason,
                    failure_hint=hint,
                )

        artifacts = adapter.expected_artifacts(resolved_work_dir)
        return pipeline._build_user_result(
            operation="data_factory_user_postprocess",
            status=DATA_FACTORY_STATUS_SUCCEEDED,
            message="训练数据一键生成完成。已自动跳过检测到的完整产物。",
            work_dir=resolved_work_dir,
            started_at=started_at,
            result_json_filename="data_factory_user_postprocess_result.json",
            steps=steps,
            current_stage="一键生成训练数据",
            current_step="completed",
            output_summary=_output_summary_from_artifacts(artifacts, enable_speaker_embedding, auto_reject_bad_samples),
            developer_details={
                "step_result_json_paths": [
                    str(step.result_json_path)
                    for step in steps
                    if step.result_json_path is not None
                ],
                "auto_resume_enabled": True,
                "resume_decisions": resume_decisions,
            },
            extra={
                "auto_reject_bad_samples": bool(auto_reject_bad_samples),
                "filter_mode": filter_mode,
                "enable_speaker_embedding": bool(enable_speaker_embedding),
                "auto_resume_enabled": True,
                "num_skipped_steps": sum(
                    1 for step in steps if step.extra.get("auto_resume_skipped")
                ),
                "num_executed_steps": sum(
                    1 for step in steps if not step.extra.get("auto_resume_skipped")
                ),
            },
        )

    except Exception as exc:
        return pipeline._build_exception_result(
            operation="data_factory_user_postprocess",
            work_dir=resolved_work_dir,
            started_at=started_at,
            result_json_filename="data_factory_user_postprocess_result.json",
            stage="一键生成训练数据",
            step_name="resumable_postprocess_pipeline",
            exc=exc,
        )


def _output_summary_from_artifacts(
    artifacts: Any,
    enable_speaker_embedding: bool,
    auto_reject_bad_samples: bool,
) -> dict[str, Any]:
    filter_mode = "move" if bool(auto_reject_bad_samples) else "report_only"
    return {
        "stage2_manifest_fewshot": artifacts.stage2_manifest_fewshot_path,
        "stage2_pt_root": artifacts.stage2_pt_root,
        "continuous_semantic_root": artifacts.continuous_semantic_root,
        "style_cache_root": artifacts.style_cache_root,
        "train_pt_root": artifacts.train_pt_root,
        "val_pt_root": artifacts.val_pt_root,
        "rejected_pt_root": artifacts.rejected_pt_root,
        "split_report": artifacts.split_report_path,
        "filter_report_json": artifacts.filter_report_json_path,
        "filter_report_csv": artifacts.filter_report_csv_path,
        "filter_mode": filter_mode,
        "speaker_embedding_enabled": bool(enable_speaker_embedding),
    }


__all__ = ["run_resumable_postprocess_pipeline"]
