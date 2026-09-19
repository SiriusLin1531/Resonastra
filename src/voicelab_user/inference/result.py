"""
Result and artifact data structures for VoiceLab user-edition inference.

P5-2 defines a stable result contract shared by future CLI, Gradio WebUI, and
service layers.  This module intentionally has no dependency on GPT-SoVITS,
Stage2 acoustic models, vocoders, torch, or metrics backends.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..runtime.path_resolver import PathLike, resolve_project_path


RESULT_STATUS_PREPARED = "prepared"
RESULT_STATUS_RUNNING = "running"
RESULT_STATUS_SUCCEEDED = "succeeded"
RESULT_STATUS_FAILED = "failed"
RESULT_STATUS_NOT_IMPLEMENTED = "not_implemented"
RESULT_STATUS_CANCELLED = "cancelled"


@dataclass(frozen=True)
class InferenceArtifactPaths:
    """Paths produced or referenced by one inference run."""

    run_dir: Path

    request_json_path: Optional[Path] = None
    internal_summary_json_path: Optional[Path] = None
    result_json_path: Optional[Path] = None

    prompt_copy_path: Optional[Path] = None
    output_wav_path: Optional[Path] = None
    output_peaknorm_wav_path: Optional[Path] = None

    stage1_semantic_path: Optional[Path] = None
    stage2_acoustic_path: Optional[Path] = None
    mel_preview_path: Optional[Path] = None

    metrics_json_path: Optional[Path] = None
    log_path: Optional[Path] = None


@dataclass(frozen=True)
class InferenceTiming:
    """Timing fields for user-edition inference."""

    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    prepare_seconds: Optional[float] = None
    stage1_seconds: Optional[float] = None
    stage2_seconds: Optional[float] = None
    vocoder_seconds: Optional[float] = None
    metrics_seconds: Optional[float] = None
    total_seconds: Optional[float] = None
    rtf: Optional[float] = None


@dataclass(frozen=True)
class InferenceMetricSummary:
    """Optional quality metrics shown to users and saved for debugging."""

    dnsmos_ovrl: Optional[float] = None
    dnsmos_sig: Optional[float] = None
    dnsmos_bak: Optional[float] = None
    speaker_sim: Optional[float] = None
    wer: Optional[float] = None
    cer: Optional[float] = None
    rtf: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceErrorInfo:
    """Structured error information for failed runs."""

    error_type: str
    message: str
    stage: Optional[str] = None
    traceback_text: Optional[str] = None
    hint: Optional[str] = None


@dataclass(frozen=True)
class UserInferenceResult:
    """Top-level result contract returned by service/CLI/WebUI."""

    schema_version: str
    status: str
    message: str
    artifacts: InferenceArtifactPaths

    profile_name: Optional[str] = None
    inference_mode: Optional[str] = None
    language: str = "zh"

    timing: InferenceTiming = field(default_factory=InferenceTiming)
    metrics: Optional[InferenceMetricSummary] = None
    error: Optional[InferenceErrorInfo] = None
    extra: dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    """Return a compact local ISO timestamp."""

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


def result_to_dict(result: UserInferenceResult) -> dict[str, Any]:
    """Convert UserInferenceResult to a JSON-serializable dictionary."""

    return _to_jsonable(result)


def write_result_json(result: UserInferenceResult, output_path: Optional[PathLike] = None) -> Path:
    """Write a result JSON file and return its resolved path.

    If output_path is not provided, result.artifacts.result_json_path is used.
    If that is also empty, the file defaults to <run_dir>/result.json.
    """

    if output_path is not None:
        resolved = resolve_project_path(output_path)
    elif result.artifacts.result_json_path is not None:
        resolved = result.artifacts.result_json_path
    else:
        resolved = result.artifacts.run_dir / "result.json"

    if resolved is None:
        raise ValueError("result JSON output path must not be empty.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(result_to_dict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resolved


def make_prepared_result(
    *,
    run_dir: Path,
    message: str = "Inference run prepared.",
    request_json_path: Optional[Path] = None,
    internal_summary_json_path: Optional[Path] = None,
    prompt_copy_path: Optional[Path] = None,
    profile_name: Optional[str] = None,
    inference_mode: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> UserInferenceResult:
    """Create a prepared result before heavy inference starts."""

    artifacts = InferenceArtifactPaths(
        run_dir=run_dir,
        request_json_path=request_json_path,
        internal_summary_json_path=internal_summary_json_path,
        result_json_path=run_dir / "result.json",
        prompt_copy_path=prompt_copy_path,
    )
    return UserInferenceResult(
        schema_version="voicelab_user_inference_result_v1",
        status=RESULT_STATUS_PREPARED,
        message=message,
        artifacts=artifacts,
        profile_name=profile_name,
        inference_mode=inference_mode,
        timing=InferenceTiming(started_at=now_iso()),
        extra=extra or {},
    )


def make_not_implemented_result(
    *,
    run_dir: Path,
    message: str,
    request_json_path: Optional[Path] = None,
    internal_summary_json_path: Optional[Path] = None,
    prompt_copy_path: Optional[Path] = None,
    profile_name: Optional[str] = None,
    inference_mode: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> UserInferenceResult:
    """Create a result for an explicit not-yet-connected inference hook."""

    artifacts = InferenceArtifactPaths(
        run_dir=run_dir,
        request_json_path=request_json_path,
        internal_summary_json_path=internal_summary_json_path,
        result_json_path=run_dir / "result.json",
        prompt_copy_path=prompt_copy_path,
    )
    return UserInferenceResult(
        schema_version="voicelab_user_inference_result_v1",
        status=RESULT_STATUS_NOT_IMPLEMENTED,
        message=message,
        artifacts=artifacts,
        profile_name=profile_name,
        inference_mode=inference_mode,
        timing=InferenceTiming(started_at=now_iso(), finished_at=now_iso()),
        error=InferenceErrorInfo(
            error_type="NotImplementedError",
            message=message,
            stage="inference_hook",
            hint="Connect GPT-SoVITS, Stage2, and vocoder in a later P5 step.",
        ),
        extra=extra or {},
    )


def make_failed_result(
    *,
    run_dir: Path,
    message: str,
    error_type: str,
    stage: Optional[str] = None,
    traceback_text: Optional[str] = None,
    hint: Optional[str] = None,
    request_json_path: Optional[Path] = None,
    internal_summary_json_path: Optional[Path] = None,
    prompt_copy_path: Optional[Path] = None,
    profile_name: Optional[str] = None,
    inference_mode: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> UserInferenceResult:
    """Create a structured failed result."""

    artifacts = InferenceArtifactPaths(
        run_dir=run_dir,
        request_json_path=request_json_path,
        internal_summary_json_path=internal_summary_json_path,
        result_json_path=run_dir / "result.json",
        prompt_copy_path=prompt_copy_path,
    )
    return UserInferenceResult(
        schema_version="voicelab_user_inference_result_v1",
        status=RESULT_STATUS_FAILED,
        message=message,
        artifacts=artifacts,
        profile_name=profile_name,
        inference_mode=inference_mode,
        timing=InferenceTiming(finished_at=now_iso()),
        error=InferenceErrorInfo(
            error_type=error_type,
            message=message,
            stage=stage,
            traceback_text=traceback_text,
            hint=hint,
        ),
        extra=extra or {},
    )


__all__ = [
    "RESULT_STATUS_PREPARED",
    "RESULT_STATUS_RUNNING",
    "RESULT_STATUS_SUCCEEDED",
    "RESULT_STATUS_FAILED",
    "RESULT_STATUS_NOT_IMPLEMENTED",
    "RESULT_STATUS_CANCELLED",
    "InferenceArtifactPaths",
    "InferenceTiming",
    "InferenceMetricSummary",
    "InferenceErrorInfo",
    "UserInferenceResult",
    "now_iso",
    "result_to_dict",
    "write_result_json",
    "make_prepared_result",
    "make_not_implemented_result",
    "make_failed_result",
]
