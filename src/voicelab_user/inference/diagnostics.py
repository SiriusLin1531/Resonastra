from __future__ import annotations

"""Read-only diagnostics presentation for VoiceLab inference UI.

IF-UI-3 keeps raw developer information out of the normal-user result area but
makes it available inside one collapsed diagnostics section.

This module is intentionally read-only:
- no model loading;
- no subprocess execution;
- no profile mutation;
- no checkpoint resolution;
- no artifact writes.

It consumes the existing ``UserInferenceResult`` contract and, when available,
reads the service-generated ``developer_command.json`` referenced by
``result.extra['developer_command_json_path']``.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional


DIAGNOSTICS_SCHEMA_VERSION = "voicelab_inference_ui_diagnostics_v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _obj_get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _read_json_dict(path_value: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Read a referenced diagnostic JSON file without raising into the UI."""

    text = _text(path_value)
    if not text:
        return {}, None

    path = Path(text).expanduser().resolve(strict=False)
    if not path.is_file():
        return {}, f"file not found: {path}"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostics must stay observable.
        return {}, f"{exc.__class__.__name__}: {exc}"

    if not isinstance(payload, dict):
        return {}, "top-level JSON value is not an object"
    return payload, None


def build_inference_diagnostics_payload(
    result: Any,
    *,
    selected_profile: Optional[str] = None,
) -> dict[str, Any]:
    """Build complete diagnostic payload from the stable result contract."""

    extra = _as_dict(_obj_get(result, "extra", {}))
    artifacts = _obj_get(result, "artifacts")
    error = _obj_get(result, "error")
    timing = _obj_get(result, "timing")
    metrics = _obj_get(result, "metrics")

    command_json_path = extra.get("developer_command_json_path")
    developer_command, command_load_error = _read_json_dict(command_json_path)

    payload: dict[str, Any] = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "status": _text(_obj_get(result, "status")),
        "profile": {
            "selected_value": _text(selected_profile),
            "resolved_profile_name": _text(_obj_get(result, "profile_name")),
            "inference_mode": _text(_obj_get(result, "inference_mode")),
            "language": _text(_obj_get(result, "language")),
        },
        "artifacts": _to_jsonable(artifacts),
        "timing": _to_jsonable(timing),
        "metrics": _to_jsonable(metrics),
        "error": _to_jsonable(error),
        "backend": {
            "returncode": extra.get("returncode"),
            "timeout": bool(extra.get("timeout")),
            "developer_script": extra.get("developer_script"),
            "developer_command_json_path": command_json_path,
            "developer_summary_json_path": extra.get("developer_summary_json_path"),
            "developer_raw_summary_json_path": extra.get("developer_raw_summary_json_path"),
            "developer_raw_metrics_json_path": extra.get("developer_raw_metrics_json_path"),
            "developer_summary_loaded": extra.get("developer_summary_loaded"),
            "stdout_tail": extra.get("stdout_tail"),
            "stderr_tail": extra.get("stderr_tail"),
        },
        "developer_command": developer_command,
    }

    if command_load_error:
        payload["developer_command_load_error"] = command_load_error

    # Keep any adapter/service extras that are not already promoted above. This
    # makes diagnostics forward-compatible while normal-user formatting remains
    # intentionally strict and path-free.
    payload["result_extra"] = _to_jsonable(extra)
    return payload


def _code_block(text: Any, *, language: str = "text", max_chars: int = 8000) -> str:
    raw = _text(text)
    if not raw:
        return "`(empty)`"
    if len(raw) > max_chars:
        raw = raw[-max_chars:]
        raw = "[tail only]\n" + raw
    return f"```{language}\n{raw}\n```"


def format_inference_diagnostics_markdown(payload: Any) -> str:
    """Render developer-oriented diagnostics for the collapsed UI area."""

    data = _as_dict(payload)
    if not data:
        return "尚无推理诊断信息。完成一次推理后，这里会显示命令、原始路径和错误详情。"

    profile = _as_dict(data.get("profile"))
    artifacts = _as_dict(data.get("artifacts"))
    backend = _as_dict(data.get("backend"))
    error = _as_dict(data.get("error"))
    command = _as_dict(data.get("developer_command"))

    lines = [
        "### 推理诊断",
        "",
        f"- Status: `{data.get('status') or ''}`",
        f"- Selected profile: `{profile.get('selected_value') or ''}`",
        f"- Resolved profile: `{profile.get('resolved_profile_name') or ''}`",
        f"- Return code: `{backend.get('returncode')}`",
        f"- Timeout: `{backend.get('timeout')}`",
    ]

    lines.extend(
        [
            "",
            "#### 原始路径",
            "",
            f"- Run directory: `{artifacts.get('run_dir') or ''}`",
            f"- Output WAV: `{artifacts.get('output_wav_path') or ''}`",
            f"- Result JSON: `{artifacts.get('result_json_path') or ''}`",
            f"- Metrics JSON: `{artifacts.get('metrics_json_path') or ''}`",
            f"- Developer command JSON: `{backend.get('developer_command_json_path') or ''}`",
            f"- Developer summary JSON: `{backend.get('developer_summary_json_path') or ''}`",
        ]
    )

    display_command = command.get("display_command")
    if display_command:
        lines.extend(["", "#### Developer command", "", _code_block(display_command)])
    elif data.get("developer_command_load_error"):
        lines.extend(
            [
                "",
                "#### Developer command",
                "",
                f"读取失败：`{data.get('developer_command_load_error')}`",
            ]
        )

    if error:
        lines.extend(
            [
                "",
                "#### Error",
                "",
                f"- Type: `{error.get('error_type') or ''}`",
                f"- Stage: `{error.get('stage') or ''}`",
                f"- Message: `{error.get('message') or ''}`",
                f"- Hint: {error.get('hint') or ''}",
            ]
        )
        if error.get("traceback_text"):
            lines.extend(["", "**Traceback / raw error tail**", "", _code_block(error.get("traceback_text"))])

    if backend.get("stdout_tail"):
        lines.extend(["", "#### stdout tail", "", _code_block(backend.get("stdout_tail"))])
    if backend.get("stderr_tail"):
        lines.extend(["", "#### stderr tail", "", _code_block(backend.get("stderr_tail"))])

    return "\n".join(lines)


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "build_inference_diagnostics_payload",
    "format_inference_diagnostics_markdown",
]
