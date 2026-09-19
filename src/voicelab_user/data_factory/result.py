from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DATA_FACTORY_RESULT_SCHEMA_VERSION = "voicelab_data_factory_result_v1"

DATA_FACTORY_STATUS_PREPARED = "prepared"
DATA_FACTORY_STATUS_RUNNING = "running"
DATA_FACTORY_STATUS_SUCCEEDED = "succeeded"
DATA_FACTORY_STATUS_FAILED = "failed"
DATA_FACTORY_STATUS_PARTIAL = "partial"


@dataclass(frozen=True)
class DataFactoryArtifactPaths:
    """Paths produced or referenced by one data factory run.

    中文说明：
        一次数据工厂运行产生或引用的关键路径。
    """

    work_dir: Path

    request_json_path: Optional[Path] = None
    result_json_path: Optional[Path] = None
    command_plan_json_path: Optional[Path] = None

    # prepare_fewshot_dataset.py outputs
    raw_audio_index_path: Optional[Path] = None
    prepare_report_path: Optional[Path] = None

    manifest_jsonl_path: Optional[Path] = None
    dataset_list_path: Optional[Path] = None
    stage2_manifest_jsonl_path: Optional[Path] = None

    export_clips_dir: Optional[Path] = None

    manifest_before_proofread_path: Optional[Path] = None
    dataset_before_proofread_path: Optional[Path] = None
    stage2_manifest_before_proofread_path: Optional[Path] = None

    manifest_corrected_path: Optional[Path] = None
    stage2_manifest_corrected_path: Optional[Path] = None
    correction_report_path: Optional[Path] = None

    # few-shot stage2 manifest outputs
    stage2_manifest_fewshot_path: Optional[Path] = None
    prompt_selection_report_path: Optional[Path] = None
    fewshot_manifest_conversion_report_path: Optional[Path] = None

    # health reports
    manifest_health_report_path: Optional[Path] = None
    stage2_manifest_health_report_path: Optional[Path] = None

    # Later P10/P11 heavy outputs
    stage2_pt_root: Optional[Path] = None
    stage2_pt_health_report_path: Optional[Path] = None
    continuous_semantic_root: Optional[Path] = None
    style_cache_root: Optional[Path] = None
    train_pt_root: Optional[Path] = None
    val_pt_root: Optional[Path] = None
    split_report_path: Optional[Path] = None

    rejected_pt_root: Optional[Path] = None
    filter_report_json_path: Optional[Path] = None
    filter_report_csv_path: Optional[Path] = None


@dataclass(frozen=True)
class DataFactoryTiming:
    """Timing information for a data factory run."""

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    prepare_seconds: Optional[float] = None
    manifest_health_seconds: Optional[float] = None
    export_fewshot_stage2_seconds: Optional[float] = None
    stage2_manifest_health_seconds: Optional[float] = None

    total_seconds: Optional[float] = None


@dataclass(frozen=True)
class DataFactoryStepResult:
    """Result for one data factory pipeline step."""

    name: str
    status: str
    command: Optional[list[str]] = None
    returncode: Optional[int] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    report_path: Optional[Path] = None
    message: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataFactoryErrorInfo:
    """Structured error information for failed runs."""

    error_type: str
    message: str
    stage: Optional[str] = None
    traceback_text: Optional[str] = None
    hint: Optional[str] = None


@dataclass(frozen=True)
class DataFactoryResult:
    """Top-level result returned by data factory service / WebUI."""

    schema_version: str
    status: str
    message: str

    artifacts: DataFactoryArtifactPaths

    speaker_name: Optional[str] = None
    language: str = "zh"

    timing: DataFactoryTiming = field(default_factory=DataFactoryTiming)
    steps: list[DataFactoryStepResult] = field(default_factory=list)

    error: Optional[DataFactoryErrorInfo] = None
    extra: dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    """Return local ISO timestamp."""

    return datetime.now().isoformat(timespec="seconds")


def _to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths to JSON-safe values."""

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return value


def result_to_dict(result: DataFactoryResult) -> dict[str, Any]:
    """Convert DataFactoryResult to a JSON-serializable dictionary."""

    return _to_jsonable(result)


def write_data_factory_result_json(
    result: DataFactoryResult,
    output_path: Optional[Path] = None,
) -> Path:
    """Write data factory result JSON and return path."""

    if output_path is not None:
        resolved = Path(output_path).resolve()
    elif result.artifacts.result_json_path is not None:
        resolved = result.artifacts.result_json_path
    else:
        resolved = result.artifacts.work_dir / "data_factory_result.json"

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(result_to_dict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resolved


def make_prepared_data_factory_result(
    *,
    work_dir: Path,
    message: str = "Data factory run prepared.",
    request_json_path: Optional[Path] = None,
    result_json_path: Optional[Path] = None,
    command_plan_json_path: Optional[Path] = None,
    speaker_name: Optional[str] = None,
    language: str = "zh",
    extra: Optional[dict[str, Any]] = None,
) -> DataFactoryResult:
    """Create a prepared result before running heavy steps."""

    artifacts = DataFactoryArtifactPaths(
        work_dir=work_dir,
        request_json_path=request_json_path,
        result_json_path=result_json_path or (work_dir / "data_factory_result.json"),
        command_plan_json_path=command_plan_json_path,
    )

    return DataFactoryResult(
        schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
        status=DATA_FACTORY_STATUS_PREPARED,
        message=message,
        artifacts=artifacts,
        speaker_name=speaker_name,
        language=language,
        timing=DataFactoryTiming(started_at=now_iso()),
        extra=extra or {},
    )