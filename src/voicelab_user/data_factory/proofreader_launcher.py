from __future__ import annotations

"""Resource-safe launcher for the external DataFactory proofreader.

This helper mirrors the stable DataFactory proofreader command construction but
closes the parent process' stdout/stderr file objects immediately after Popen.
The child keeps its inherited OS handles, so logging continues normally without
leaking Python file objects in the long-lived DataFactory WebUI process.
"""

import subprocess
from pathlib import Path
from typing import Any

from .proofreader_runtime import record_proofreader_launch


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    import json

    path = Path(path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def launch_proofreader_resource_safe(
    adapter: Any,
    work_dir: str | Path,
    *,
    webui_port_subfix: int = 9871,
    is_share: str = "False",
    g_batch: int = 10,
    backup_suffix: str = "before_proofread",
    overwrite_backup: bool = False,
) -> dict[str, Any]:
    """Launch the proofreader and persist its runtime identity.

    ``adapter`` is intentionally duck-typed to avoid coupling this small helper
    to the large service module. It must provide the stable DataFactory adapter
    methods used by the existing service implementation.
    """

    resolved_work_dir = adapter.resolve_path(work_dir)
    artifacts = adapter.expected_artifacts(resolved_work_dir)

    dataset_list_path = artifacts.dataset_list_path
    if dataset_list_path is None or not dataset_list_path.exists():
        raise FileNotFoundError(
            f"dataset.list not found. Please run prepare first: {dataset_list_path}"
        )

    command = adapter.build_launch_proofread_command(
        artifacts,
        webui_port_subfix=int(webui_port_subfix),
        is_share=str(is_share),
        g_batch=int(g_batch),
        backup_suffix=str(backup_suffix),
        overwrite_backup=bool(overwrite_backup),
    )

    logs_dir = resolved_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = logs_dir / "proofread_stdout.log"
    stderr_log_path = logs_dir / "proofread_stderr.log"

    stdout_f = stdout_log_path.open("a", encoding="utf-8", errors="replace")
    stderr_f = stderr_log_path.open("a", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(
            [str(value) for value in command.argv],
            cwd=str(command.cwd),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        # The child has inherited/duplicated the OS handles by this point.
        # The long-lived DataFactory process must release its own file objects.
        stdout_f.close()
        stderr_f.close()

    proofread_url = f"http://127.0.0.1:{int(webui_port_subfix)}"
    payload: dict[str, Any] = {
        "status": "launched",
        "message": "Proofread WebUI launched.",
        "pid": int(process.pid),
        "proofread_url": proofread_url,
        "work_dir": str(resolved_work_dir),
        "dataset_list": str(dataset_list_path),
        "command": [str(value) for value in command.argv],
        "printable": command.to_printable(),
        "proofread_launch_report": str(command.expected_report_path),
        "stdout_log_path": str(stdout_log_path),
        "stderr_log_path": str(stderr_log_path),
        "expected_output_paths": {
            str(key): str(value)
            for key, value in command.expected_output_paths.items()
        },
    }

    bridge_report_path = (
        resolved_work_dir / "06_export" / "proofread_bridge_launch.json"
    )
    _write_json(bridge_report_path, payload)
    payload["bridge_report_path"] = str(bridge_report_path)

    runtime_state = record_proofreader_launch(
        resolved_work_dir,
        launcher_pid=int(process.pid),
        dataset_list=dataset_list_path,
        port=int(webui_port_subfix),
        command=payload["command"],
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        launch_report_path=bridge_report_path,
    )
    payload["proofreader_runtime"] = runtime_state
    payload["url"] = runtime_state.get("url")
    return payload


__all__ = ["launch_proofreader_resource_safe"]
