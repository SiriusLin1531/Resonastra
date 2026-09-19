from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Local imports
# 本地导入
# ============================================================
from .adapter import DataFactoryAdapter, create_data_factory_adapter
from .request import DataFactoryRequest, DataFactoryStepSelection
from .result import (
    DATA_FACTORY_STATUS_FAILED,
    DATA_FACTORY_STATUS_PARTIAL,
    DATA_FACTORY_STATUS_SUCCEEDED,
    DataFactoryErrorInfo,
    DataFactoryResult,
    DataFactoryStepResult,
    result_to_dict,
)
from .service import (
    DataFactoryExecutionOptions,
    DataFactoryService,
    create_data_factory_service,
)


PathLike = str | Path

USER_PIPELINE_SCHEMA_VERSION = "voicelab_data_factory_user_pipeline_v1"


# ============================================================
# JSON helpers
# JSON 工具
# ============================================================
def now_iso() -> str:
    """Return local ISO timestamp."""

    return datetime.now().isoformat(timespec="seconds")


def _to_jsonable(value: Any) -> Any:
    """Convert dataclasses / Path / list / dict into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(x) for x in value]

    return value


def write_json(obj: Any, path: PathLike) -> Path:
    """Write a JSON object and return resolved path."""

    resolved = Path(path).expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resolved


# ============================================================
# User pipeline result structures
# 用户版 pipeline 结果结构
# ============================================================
@dataclass(frozen=True)
class UserPipelineStepResult:
    """Compact user-facing result for one pipeline step.

    中文说明：
        用户版 pipeline 的单步摘要结果。

        它不是 DataFactoryStepResult 的替代品，而是面向用户 UI 的
        简洁包装。完整开发者信息仍然可以从 result_json_path /
        stdout_tail / stderr_tail 中查看。
    """

    name: str
    display_name: str
    status: str
    message: str

    data_factory_status: Optional[str] = None
    result_json_path: Optional[Path] = None
    report_path: Optional[Path] = None

    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None

    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total_seconds: Optional[float] = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserPipelineResult:
    """Top-level result for user-mode data factory orchestration.

    中文说明：
        用户版一键编排结果。

        UI 层默认只需要显示：
        - status
        - message
        - current_stage
        - current_step
        - failed_step
        - failure_reason
        - output_summary

        开发者详情可以从 developer_details 和 steps 里展开。
    """

    schema_version: str
    operation: str
    status: str
    message: str

    work_dir: Path

    started_at: str
    finished_at: Optional[str] = None
    total_seconds: Optional[float] = None

    current_stage: Optional[str] = None
    current_step: Optional[str] = None

    failed_step: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_hint: Optional[str] = None

    steps: list[UserPipelineStepResult] = field(default_factory=list)

    output_summary: dict[str, Any] = field(default_factory=dict)
    developer_details: dict[str, Any] = field(default_factory=dict)

    result_json_path: Optional[Path] = None
    error: Optional[dict[str, Any]] = None
    extra: dict[str, Any] = field(default_factory=dict)


def user_pipeline_result_to_dict(result: UserPipelineResult) -> dict[str, Any]:
    """Convert UserPipelineResult to JSON-safe dict."""

    return _to_jsonable(result)


def write_user_pipeline_result_json(
    result: UserPipelineResult,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Write UserPipelineResult JSON and return resolved path."""

    if output_path is None:
        if result.result_json_path is not None:
            output_path = result.result_json_path
        else:
            output_path = result.work_dir / f"{result.operation}.json"

    return write_json(user_pipeline_result_to_dict(result), output_path)


# ============================================================
# Conversion helpers
# 转换工具
# ============================================================
def _tail_from_result(result: DataFactoryResult, *, stream: str) -> str:
    """Collect stdout/stderr tails from DataFactoryResult steps."""

    chunks: list[str] = []

    for step in result.steps or []:
        value = step.stdout_tail if stream == "stdout" else step.stderr_tail
        if value:
            chunks.append(f"===== {step.name} =====\n{value}")

    return "\n\n".join(chunks)


def _failure_reason_from_result(result: DataFactoryResult) -> tuple[Optional[str], Optional[str]]:
    """Extract compact failure reason and hint from DataFactoryResult."""

    if result.error is not None:
        return result.error.message, result.error.hint

    for step in result.steps or []:
        if step.status != DATA_FACTORY_STATUS_SUCCEEDED:
            return step.message, None

    return None, None


def _failed_step_from_result(result: DataFactoryResult) -> Optional[str]:
    """Find failed step name from DataFactoryResult."""

    if result.error is not None and result.error.stage:
        return result.error.stage

    for step in result.steps or []:
        if step.status != DATA_FACTORY_STATUS_SUCCEEDED:
            return step.name

    return None


def _first_report_path(result: DataFactoryResult) -> Optional[Path]:
    """Return the first available report_path from DataFactoryResult steps."""

    for step in result.steps or []:
        if step.report_path is not None:
            return step.report_path
    return None


def _step_from_data_factory_result(
    *,
    name: str,
    display_name: str,
    result: DataFactoryResult,
) -> UserPipelineStepResult:
    """Convert DataFactoryResult into one user pipeline step result."""

    stdout_tail = _tail_from_result(result, stream="stdout")
    stderr_tail = _tail_from_result(result, stream="stderr")

    return UserPipelineStepResult(
        name=name,
        display_name=display_name,
        status=result.status,
        message=result.message,
        data_factory_status=result.status,
        result_json_path=result.artifacts.result_json_path,
        report_path=_first_report_path(result),
        stdout_tail=stdout_tail or None,
        stderr_tail=stderr_tail or None,
        started_at=result.timing.started_at,
        finished_at=result.timing.finished_at,
        total_seconds=result.timing.total_seconds,
        extra={
            "data_factory_result": result_to_dict(result),
        },
    )


def _step_from_exception(
    *,
    name: str,
    display_name: str,
    exc: Exception,
) -> UserPipelineStepResult:
    """Build a failed user step from exception."""

    return UserPipelineStepResult(
        name=name,
        display_name=display_name,
        status=DATA_FACTORY_STATUS_FAILED,
        message=str(exc),
        stderr_tail=traceback.format_exc(),
        extra={
            "error_type": type(exc).__name__,
        },
    )


# ============================================================
# User pipeline service
# 用户版数据工厂编排服务
# ============================================================
class DataFactoryUserPipelineService:
    """User-mode orchestration layer for VoiceLab DataFactory.

    中文说明：
        用户版数据工厂编排层。

        这一层不重写底层数据处理逻辑，只负责把 P10-1 到 P10-10 已经
        完成的 DataFactoryService 方法组合成用户版的一键流程。

        当前提供：
        - run_prepare_pipeline()
        - launch_proofread()
        - rebuild_after_proofread()
        - skip_proofread()
        - run_postprocess_pipeline()
    """

    def __init__(
        self,
        *,
        service: Optional[DataFactoryService] = None,
        adapter: Optional[DataFactoryAdapter] = None,
        show_terminal_progress: bool = True,
    ) -> None:
        self.adapter = adapter if adapter is not None else create_data_factory_adapter()

        self.service = (
            service
            if service is not None
            else create_data_factory_service(
                adapter=self.adapter,
                show_terminal_progress=bool(show_terminal_progress),
            )
        )

        self.show_terminal_progress = bool(show_terminal_progress)


    # --------------------------------------------------------
    # Internal helpers
    # 内部工具
    # --------------------------------------------------------
    def _resolve_work_dir(self, work_dir: PathLike) -> Path:
        return self.adapter.resolve_path(work_dir)

    def _result_path(self, work_dir: Path, filename: str) -> Path:
        return work_dir / filename

    def _build_user_result(
        self,
        *,
        operation: str,
        status: str,
        message: str,
        work_dir: Path,
        started_at: str,
        result_json_filename: str,
        steps: Optional[list[UserPipelineStepResult]] = None,
        current_stage: Optional[str] = None,
        current_step: Optional[str] = None,
        failed_step: Optional[str] = None,
        failure_reason: Optional[str] = None,
        failure_hint: Optional[str] = None,
        output_summary: Optional[dict[str, Any]] = None,
        developer_details: Optional[dict[str, Any]] = None,
        error: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> UserPipelineResult:
        finished_at = now_iso()
        result_json_path = self._result_path(work_dir, result_json_filename)

        result = UserPipelineResult(
            schema_version=USER_PIPELINE_SCHEMA_VERSION,
            operation=operation,
            status=status,
            message=message,
            work_dir=work_dir,
            started_at=started_at,
            finished_at=finished_at,
            total_seconds=None,
            current_stage=current_stage,
            current_step=current_step,
            failed_step=failed_step,
            failure_reason=failure_reason,
            failure_hint=failure_hint,
            steps=steps or [],
            output_summary=output_summary or {},
            developer_details=developer_details or {},
            result_json_path=result_json_path,
            error=error,
            extra=extra or {},
        )

        # Compute total seconds after object creation.
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(finished_at)
            result = replace(
                result,
                total_seconds=float((end_dt - start_dt).total_seconds()),
            )
        except Exception:
            pass

        write_user_pipeline_result_json(result, result_json_path)
        return result

    def _build_exception_result(
        self,
        *,
        operation: str,
        work_dir: Path,
        started_at: str,
        result_json_filename: str,
        stage: str,
        step_name: str,
        exc: Exception,
    ) -> UserPipelineResult:
        step = _step_from_exception(
            name=step_name,
            display_name=stage,
            exc=exc,
        )

        return self._build_user_result(
            operation=operation,
            status=DATA_FACTORY_STATUS_FAILED,
            message=f"{stage} failed.",
            work_dir=work_dir,
            started_at=started_at,
            result_json_filename=result_json_filename,
            steps=[step],
            current_stage=stage,
            current_step=step_name,
            failed_step=step_name,
            failure_reason=str(exc),
            failure_hint="请展开开发者详情或查看日志文件，确认输入路径和上游产物是否存在。",
            error={
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    # --------------------------------------------------------
    # Stage 1: prepare pipeline
    # 阶段一：数据工厂执行
    # --------------------------------------------------------
    def run_prepare_pipeline(
        self,
        request: DataFactoryRequest,
    ) -> UserPipelineResult:
        """Run user-mode prepare pipeline.

        中文说明：
            用户版第一阶段。

            内部执行：
            1. prepare dataset
            2. manifest health check
            3. export initial stage2 manifest
            4. stage2 manifest health check

            用户版不会暴露 Build Plan；但底层 service 仍会生成 request /
            command_plan / result 文件，方便开发者排查。
        """

        started_at = now_iso()

        # Force user-mode stage-1 step selection.
        # 强制用户版第一阶段只运行轻量数据工厂四步。
        user_request = replace(
            request,
            dry_run=False,
            steps=DataFactoryStepSelection(
                run_prepare=True,
                run_manifest_health_check=True,
                run_export_fewshot_stage2_manifest=True,
                run_stage2_manifest_health_check=True,
                run_stage2_pt_preprocess=False,
                run_continuous_semantic_cache=False,
                run_style_cache=False,
                run_split_train_val=False,
                run_filter_bad_samples=False,
            ),
        )

        work_dir = self.adapter.resolve_work_dir(user_request)

        try:
            result = self.service.run(user_request, execute=True)
            step = _step_from_data_factory_result(
                name="prepare_pipeline",
                display_name="执行数据工厂",
                result=result,
            )

            failure_reason, failure_hint = _failure_reason_from_result(result)
            failed_step = _failed_step_from_result(result)

            status = result.status
            message = (
                "数据工厂执行完成。"
                if status == DATA_FACTORY_STATUS_SUCCEEDED
                else "数据工厂执行失败。"
            )

            artifacts = result.artifacts

            return self._build_user_result(
                operation="data_factory_user_prepare",
                status=status,
                message=message,
                work_dir=work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_prepare_result.json",
                steps=[step],
                current_stage="数据工厂执行",
                current_step="prepare_pipeline",
                failed_step=failed_step,
                failure_reason=failure_reason,
                failure_hint=failure_hint,
                output_summary={
                    "work_dir": artifacts.work_dir,
                    "manifest_jsonl": artifacts.manifest_jsonl_path,
                    "dataset_list": artifacts.dataset_list_path,
                    "stage2_manifest_jsonl": artifacts.stage2_manifest_jsonl_path,
                    "prepare_report": artifacts.prepare_report_path,
                    "manifest_health_report": artifacts.manifest_health_report_path,
                    "stage2_manifest_health_report": artifacts.stage2_manifest_health_report_path,
                },
                developer_details={
                    "data_factory_result_json": artifacts.result_json_path,
                    "command_plan_json": artifacts.command_plan_json_path,
                    "request_json": artifacts.request_json_path,
                },
            )

        except Exception as exc:
            return self._build_exception_result(
                operation="data_factory_user_prepare",
                work_dir=work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_prepare_result.json",
                stage="数据工厂执行",
                step_name="prepare_pipeline",
                exc=exc,
            )

    # --------------------------------------------------------
    # Stage 2: proofread bridge
    # 阶段二：人工校对
    # --------------------------------------------------------
    def launch_proofread(
        self,
        work_dir: PathLike,
        *,
        webui_port_subfix: int = 9871,
        is_share: str = "False",
        g_batch: int = 10,
        backup_suffix: str = "before_proofread",
        overwrite_backup: bool = False,
    ) -> dict[str, Any]:
        """Launch legacy proofread WebUI through DataFactoryService.

        中文说明：
            启动旧版人工校对器。

            这个方法暂时直接复用 DataFactoryService.launch_proofread()，
            返回字典 payload。用户版 UI 只需要显示 proofread_url / pid /
            dataset_list 即可。
        """

        return self.service.launch_proofread(
            work_dir=work_dir,
            webui_port_subfix=webui_port_subfix,
            is_share=is_share,
            g_batch=g_batch,
            backup_suffix=backup_suffix,
            overwrite_backup=overwrite_backup,
        )

    def rebuild_after_proofread(
        self,
        work_dir: PathLike,
        *,
        duration_tolerance_sec: float = 0.05,
        prompt_mode: str = "self",
        timeout_seconds: Optional[float] = 600,
    ) -> UserPipelineResult:
        """Rebuild corrected manifest after manual proofreading."""

        started_at = now_iso()
        resolved_work_dir = self._resolve_work_dir(work_dir)

        try:
            result = self.service.rebuild_corrected_manifest(
                work_dir=resolved_work_dir,
                duration_tolerance_sec=duration_tolerance_sec,
                prompt_mode=prompt_mode,
                timeout_seconds=timeout_seconds,
            )

            step = _step_from_data_factory_result(
                name="rebuild_corrected_manifest",
                display_name="重建校对后 manifest",
                result=result,
            )

            failure_reason, failure_hint = _failure_reason_from_result(result)
            failed_step = _failed_step_from_result(result)

            artifacts = result.artifacts
            status = result.status
            message = (
                "校对后 manifest 重建完成。"
                if status == DATA_FACTORY_STATUS_SUCCEEDED
                else "校对后 manifest 重建失败。"
            )

            return self._build_user_result(
                operation="data_factory_user_rebuild_after_proofread",
                status=status,
                message=message,
                work_dir=resolved_work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_rebuild_after_proofread_result.json",
                steps=[step],
                current_stage="人工校对",
                current_step="rebuild_corrected_manifest",
                failed_step=failed_step,
                failure_reason=failure_reason,
                failure_hint=failure_hint,
                output_summary={
                    "manifest_corrected": artifacts.manifest_corrected_path,
                    "stage2_manifest_corrected": artifacts.stage2_manifest_corrected_path,
                    "correction_report": artifacts.correction_report_path,
                },
                developer_details={
                    "data_factory_result_json": artifacts.result_json_path,
                },
            )

        except Exception as exc:
            return self._build_exception_result(
                operation="data_factory_user_rebuild_after_proofread",
                work_dir=resolved_work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_rebuild_after_proofread_result.json",
                stage="人工校对",
                step_name="rebuild_corrected_manifest",
                exc=exc,
            )

    def skip_proofread(
        self,
        work_dir: PathLike,
        *,
        overwrite: bool = True,
    ) -> UserPipelineResult:
        """Skip proofreading by copying original manifest files to corrected manifest files.

        中文说明：
            跳过人工校对。

            根据你的决策 9，跳过校对时仍然生成 corrected manifest，
            这样后续一键后处理可以始终从 manifest.corrected.jsonl 开始。

            执行：
                manifest.jsonl -> manifest.corrected.jsonl
                stage2_manifest.jsonl -> stage2_manifest.corrected.jsonl
                correction_report.json 写入 proofread_skipped=true
        """

        started_at = now_iso()
        resolved_work_dir = self._resolve_work_dir(work_dir)

        try:
            artifacts = self.adapter.expected_artifacts(resolved_work_dir)

            if artifacts.manifest_jsonl_path is None or not artifacts.manifest_jsonl_path.exists():
                raise FileNotFoundError(
                    f"manifest.jsonl not found: {artifacts.manifest_jsonl_path}"
                )

            if artifacts.manifest_corrected_path is None:
                raise ValueError("manifest_corrected_path is missing in artifacts.")

            if artifacts.correction_report_path is None:
                raise ValueError("correction_report_path is missing in artifacts.")

            if artifacts.manifest_corrected_path.exists() and not overwrite:
                raise FileExistsError(
                    f"manifest.corrected.jsonl already exists: {artifacts.manifest_corrected_path}"
                )

            artifacts.manifest_corrected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                str(artifacts.manifest_jsonl_path),
                str(artifacts.manifest_corrected_path),
            )

            stage2_copied = False
            stage2_source_missing = False

            if (
                artifacts.stage2_manifest_jsonl_path is not None
                and artifacts.stage2_manifest_jsonl_path.exists()
                and artifacts.stage2_manifest_corrected_path is not None
            ):
                if artifacts.stage2_manifest_corrected_path.exists() and not overwrite:
                    raise FileExistsError(
                        "stage2_manifest.corrected.jsonl already exists: "
                        f"{artifacts.stage2_manifest_corrected_path}"
                    )

                artifacts.stage2_manifest_corrected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    str(artifacts.stage2_manifest_jsonl_path),
                    str(artifacts.stage2_manifest_corrected_path),
                )
                stage2_copied = True
            else:
                stage2_source_missing = True

            report = {
                "proofread_skipped": True,
                "overwrite": bool(overwrite),
                "source_manifest": str(artifacts.manifest_jsonl_path),
                "output_manifest": str(artifacts.manifest_corrected_path),
                "source_stage2_manifest": (
                    None
                    if artifacts.stage2_manifest_jsonl_path is None
                    else str(artifacts.stage2_manifest_jsonl_path)
                ),
                "output_stage2_manifest": (
                    None
                    if artifacts.stage2_manifest_corrected_path is None
                    else str(artifacts.stage2_manifest_corrected_path)
                ),
                "stage2_manifest_copied": bool(stage2_copied),
                "stage2_source_missing": bool(stage2_source_missing),
                "created_at": now_iso(),
            }
            write_json(report, artifacts.correction_report_path)

            step = UserPipelineStepResult(
                name="skip_proofread",
                display_name="跳过人工校对",
                status=DATA_FACTORY_STATUS_SUCCEEDED,
                message="已跳过人工校对，并生成 corrected manifest。",
                report_path=artifacts.correction_report_path,
                extra=report,
            )

            return self._build_user_result(
                operation="data_factory_user_skip_proofread",
                status=DATA_FACTORY_STATUS_SUCCEEDED,
                message="已跳过人工校对，后续流程将使用 manifest.corrected.jsonl。",
                work_dir=resolved_work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_skip_proofread_result.json",
                steps=[step],
                current_stage="人工校对",
                current_step="skip_proofread",
                output_summary={
                    "manifest_corrected": artifacts.manifest_corrected_path,
                    "stage2_manifest_corrected": artifacts.stage2_manifest_corrected_path,
                    "correction_report": artifacts.correction_report_path,
                },
                developer_details={
                    "proofread_skipped": True,
                    "correction_report": artifacts.correction_report_path,
                },
            )

        except Exception as exc:
            return self._build_exception_result(
                operation="data_factory_user_skip_proofread",
                work_dir=resolved_work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_skip_proofread_result.json",
                stage="人工校对",
                step_name="skip_proofread",
                exc=exc,
            )

    # --------------------------------------------------------
    # Stage 3: one-click postprocess
    # 阶段三：一键生成训练数据
    # --------------------------------------------------------
    def run_postprocess_pipeline(
        self,
        work_dir: PathLike,
        *,
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
        """Run one-click postprocess pipeline.

        中文说明：
            用户版第三阶段：一键生成训练数据。

            顺序执行：
            1. export corrected few-shot Stage2 manifest
            2. Stage2 .pt preprocessing
            3. continuous semantic cache
            4. style / F0 / speaker cache
            5. train / val split
            6. bad style sample filter

            任何一步失败都会立即停止。
        """

        started_at = now_iso()
        resolved_work_dir = self._resolve_work_dir(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        steps: list[UserPipelineStepResult] = []

        def stop_with_result(
            *,
            message: str,
            failed_step: Optional[str],
            failure_reason: Optional[str],
            failure_hint: Optional[str],
        ) -> UserPipelineResult:
            return self._build_user_result(
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
                output_summary={
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
                },
            )

        try:
            if artifacts.manifest_corrected_path is None or not artifacts.manifest_corrected_path.exists():
                raise FileNotFoundError(
                    "manifest.corrected.jsonl not found. "
                    "请先完成人工校对并重建 corrected manifest，或点击跳过人工校对。"
                )

            # ------------------------------------------------
            # 1. Export corrected few-shot Stage2 manifest
            # ------------------------------------------------
            result = self.service.export_corrected_fewshot_stage2_manifest(
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

            # ------------------------------------------------
            # 2. Stage2 .pt preprocessing
            # ------------------------------------------------
            result = self.service.preprocess_stage2_pt_dataset(
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

            # ------------------------------------------------
            # 3. Continuous semantic cache
            # ------------------------------------------------
            result = self.service.export_continuous_semantic_cache(
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

            # ------------------------------------------------
            # 4. Style / F0 / speaker cache
            # ------------------------------------------------
            result = self.service.export_fewshot_style_cache(
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

            # ------------------------------------------------
            # 5. Train / val split
            # ------------------------------------------------
            result = self.service.split_stage2_pt_dataset(
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

            # ------------------------------------------------
            # 6. Bad style sample filter
            # ------------------------------------------------
            filter_mode = "move" if bool(auto_reject_bad_samples) else "report_only"

            result = self.service.filter_v662_bad_style_samples(
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

            # Refresh artifacts after all steps.
            artifacts = self.adapter.expected_artifacts(resolved_work_dir)

            return self._build_user_result(
                operation="data_factory_user_postprocess",
                status=DATA_FACTORY_STATUS_SUCCEEDED,
                message="训练数据一键生成完成。",
                work_dir=resolved_work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_postprocess_result.json",
                steps=steps,
                current_stage="一键生成训练数据",
                current_step="completed",
                output_summary={
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
                },
                developer_details={
                    "step_result_json_paths": [
                        str(step.result_json_path)
                        for step in steps
                        if step.result_json_path is not None
                    ],
                },
                extra={
                    "auto_reject_bad_samples": bool(auto_reject_bad_samples),
                    "filter_mode": filter_mode,
                    "enable_speaker_embedding": bool(enable_speaker_embedding),
                },
            )

        except Exception as exc:
            return self._build_exception_result(
                operation="data_factory_user_postprocess",
                work_dir=resolved_work_dir,
                started_at=started_at,
                result_json_filename="data_factory_user_postprocess_result.json",
                stage="一键生成训练数据",
                step_name="postprocess_pipeline",
                exc=exc,
            )


def create_data_factory_user_pipeline_service(
    *,
    service: Optional[DataFactoryService] = None,
    adapter: Optional[DataFactoryAdapter] = None,
    show_terminal_progress: bool = True,
) -> DataFactoryUserPipelineService:
    """Factory helper for DataFactoryUserPipelineService."""

    return DataFactoryUserPipelineService(
        service=service,
        adapter=adapter,
        show_terminal_progress=show_terminal_progress,
    )


__all__ = [
    "USER_PIPELINE_SCHEMA_VERSION",
    "UserPipelineStepResult",
    "UserPipelineResult",
    "user_pipeline_result_to_dict",
    "write_user_pipeline_result_json",
    "DataFactoryUserPipelineService",
    "create_data_factory_user_pipeline_service",
]