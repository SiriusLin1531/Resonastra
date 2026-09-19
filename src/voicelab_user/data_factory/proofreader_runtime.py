from __future__ import annotations

"""Persistent runtime lifecycle for the external DataFactory proofreader.

The proofreader is a long-lived auxiliary WebUI, not a DataFactory build task.
It therefore owns an independent runtime state instead of reusing Runtime State
v2 used by Prepare / Stage1 / Stage2.

The safety rule is strict: a PID alone is never sufficient to stop a process.
A process must also match the current work directory's ``dataset.list`` and a
known proofreader command signature before it is eligible for termination.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


PROOFREADER_RUNTIME_SCHEMA_VERSION = "voicelab_proofreader_runtime_v1"
PROOFREADER_RUNTIME_FILENAME = "proofreader_runtime_state.json"
PROOFREADER_STOP_REPORT_FILENAME = "proofreader_stop_report.json"

LAUNCHER_SIGNATURE = "launch_proofread.py"
SERVER_SIGNATURE = "tools.subfix_webui"

ProcessInfo = dict[str, Any]
KillProcessTree = Callable[[int], dict[str, Any]]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def proofreader_state_path(work_dir: str | Path) -> Path:
    return _resolved(work_dir) / PROOFREADER_RUNTIME_FILENAME


def proofreader_stop_report_path(work_dir: str | Path) -> Path:
    return _resolved(work_dir) / PROOFREADER_STOP_REPORT_FILENAME


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = _resolved(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path


def read_proofreader_state(work_dir: str | Path) -> dict[str, Any]:
    path = proofreader_state_path(work_dir)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_proofreader_state(
    work_dir: str | Path,
    payload: dict[str, Any],
) -> Path:
    state = dict(payload)
    state["schema_version"] = PROOFREADER_RUNTIME_SCHEMA_VERSION
    state["work_dir"] = str(_resolved(work_dir))
    state["updated_at"] = _now_iso()
    path = proofreader_state_path(work_dir)
    _atomic_write_json(path, state)
    return path


def _query_python_processes() -> list[ProcessInfo]:
    """Return Windows python/pythonw process metadata.

    ``ParentProcessId`` is intentionally included so diagnostics can distinguish
    the launcher from the real ``tools.subfix_webui`` server process.
    """

    if os.name != "nt":
        return []

    ps_script = r"""
$items = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe'
} | Select-Object ProcessId,ParentProcessId,Name,CommandLine
$items | ConvertTo-Json -Compress -Depth 4
""".strip()

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_script,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []

    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _path_variants(path: str | Path) -> tuple[str, ...]:
    resolved = _resolved(path)
    values = {
        str(resolved).lower(),
        resolved.as_posix().lower(),
    }
    return tuple(value for value in values if value)


def _command_matches_dataset(command_line: str, dataset_list: Path) -> bool:
    lower = str(command_line or "").lower()
    return any(value in lower for value in _path_variants(dataset_list))


def proofreader_process_kind(
    process_info: ProcessInfo,
    *,
    dataset_list: str | Path,
) -> str | None:
    """Return ``launcher`` / ``server`` only for the current dataset session."""

    command_line = str(process_info.get("CommandLine") or "")
    lower = command_line.lower()
    if not lower or not _command_matches_dataset(command_line, _resolved(dataset_list)):
        return None
    if LAUNCHER_SIGNATURE in lower:
        return "launcher"
    if SERVER_SIGNATURE in lower:
        return "server"
    return None


def find_proofreader_processes(
    work_dir: str | Path,
    *,
    dataset_list: str | Path | None = None,
    process_infos: Iterable[ProcessInfo] | None = None,
) -> list[ProcessInfo]:
    work_dir = _resolved(work_dir)
    dataset = _resolved(dataset_list or (work_dir / "06_export" / "dataset.list"))
    processes = list(_query_python_processes() if process_infos is None else process_infos)

    matches: list[ProcessInfo] = []
    for process in processes:
        kind = proofreader_process_kind(process, dataset_list=dataset)
        if kind is None:
            continue
        item = dict(process)
        item["proofreader_kind"] = kind
        matches.append(item)
    return matches


def _first_pid(processes: list[ProcessInfo], kind: str) -> int | None:
    for process in processes:
        if process.get("proofreader_kind") != kind:
            continue
        try:
            return int(process.get("ProcessId"))
        except (TypeError, ValueError):
            continue
    return None


def record_proofreader_launch(
    work_dir: str | Path,
    *,
    launcher_pid: int,
    dataset_list: str | Path,
    port: int,
    command: list[str] | None = None,
    stdout_log_path: str | Path | None = None,
    stderr_log_path: str | Path | None = None,
    launch_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the launcher identity immediately after ``Popen`` succeeds."""

    work_dir = _resolved(work_dir)
    dataset = _resolved(dataset_list)
    now = _now_iso()
    payload: dict[str, Any] = {
        "status": "starting",
        "launcher_pid": int(launcher_pid),
        "server_pid": None,
        "dataset_list": str(dataset),
        "host": "127.0.0.1",
        "port": int(port),
        "url": f"http://127.0.0.1:{int(port)}",
        "command": list(command or []),
        "started_at": now,
        "last_verified_at": now,
        "stopped_at": None,
        "stdout_log_path": None if stdout_log_path is None else str(_resolved(stdout_log_path)),
        "stderr_log_path": None if stderr_log_path is None else str(_resolved(stderr_log_path)),
        "launch_report_path": None if launch_report_path is None else str(_resolved(launch_report_path)),
        "stop_report_path": None,
        "last_error": None,
    }
    state_path = write_proofreader_state(work_dir, payload)
    payload["runtime_state_path"] = str(state_path)
    return payload


def inspect_proofreader_runtime(
    work_dir: str | Path,
    *,
    dataset_list: str | Path | None = None,
    process_infos: Iterable[ProcessInfo] | None = None,
    write_state: bool = True,
) -> dict[str, Any]:
    """Reconcile persisted proofreader state with the real Windows process list."""

    work_dir = _resolved(work_dir)
    previous = read_proofreader_state(work_dir)
    dataset = _resolved(
        dataset_list
        or previous.get("dataset_list")
        or (work_dir / "06_export" / "dataset.list")
    )
    matches = find_proofreader_processes(
        work_dir,
        dataset_list=dataset,
        process_infos=process_infos,
    )

    launcher_pid = _first_pid(matches, "launcher")
    server_pid = _first_pid(matches, "server")
    now = _now_iso()

    payload = dict(previous)
    payload.update(
        {
            "dataset_list": str(dataset),
            "launcher_pid": launcher_pid,
            "server_pid": server_pid,
            "matched_processes": matches,
            "process_alive": bool(matches),
            "last_verified_at": now,
        }
    )

    if matches:
        payload["status"] = "running"
        payload["stopped_at"] = None
        payload["last_error"] = None
    else:
        previous_status = str(previous.get("status") or "").strip().lower()
        if previous_status in {"running", "starting", "stopping"}:
            payload["status"] = "exited"
        elif previous_status in {"stopped", "exited", "failed"}:
            payload["status"] = previous_status
        else:
            payload["status"] = "not_started"

    if payload.get("port") is not None and not payload.get("url"):
        payload["url"] = f"http://127.0.0.1:{int(payload['port'])}"

    if write_state:
        state_path = write_proofreader_state(work_dir, payload)
        payload["runtime_state_path"] = str(state_path)
    else:
        payload["runtime_state_path"] = str(proofreader_state_path(work_dir))
    return payload


def _kill_process_tree(pid: int) -> dict[str, Any]:
    completed = subprocess.run(
        ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "pid": int(pid),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "succeeded": completed.returncode == 0,
    }


def stop_proofreader_runtime(
    work_dir: str | Path,
    *,
    dataset_list: str | Path | None = None,
    process_infos: Iterable[ProcessInfo] | None = None,
    kill_process_tree: KillProcessTree | None = None,
) -> dict[str, Any]:
    """Stop only the proofreader process tree belonging to ``work_dir``.

    The function is intentionally idempotent. If no matching proofreader exists,
    it returns ``status=stopped`` without treating that condition as an error.
    """

    work_dir = _resolved(work_dir)
    inspected = inspect_proofreader_runtime(
        work_dir,
        dataset_list=dataset_list,
        process_infos=process_infos,
        write_state=False,
    )
    matches = list(inspected.get("matched_processes") or [])
    killer = kill_process_tree or _kill_process_tree

    stopping = dict(inspected)
    stopping["status"] = "stopping" if matches else "stopped"
    write_proofreader_state(work_dir, stopping)

    kill_results: list[dict[str, Any]] = []
    # Killing the launcher tree first normally terminates its subfix child too.
    ordered = sorted(
        matches,
        key=lambda item: 0 if item.get("proofreader_kind") == "launcher" else 1,
    )
    attempted: set[int] = set()
    for process in ordered:
        try:
            pid = int(process.get("ProcessId"))
        except (TypeError, ValueError):
            continue
        if pid in attempted:
            continue
        attempted.add(pid)
        kill_results.append(killer(pid))
        # In production, taskkill /T on the launcher covers the server child;
        # do not target the server separately unless no launcher matched.
        if process.get("proofreader_kind") == "launcher":
            break

    if process_infos is None:
        after = inspect_proofreader_runtime(
            work_dir,
            dataset_list=dataset_list,
            process_infos=None,
            write_state=False,
        )
        still_running = bool(after.get("process_alive"))
        # If launcher-tree termination left an orphan server, terminate only the
        # verified server belonging to the same dataset.
        if still_running:
            server_matches = [
                item
                for item in (after.get("matched_processes") or [])
                if item.get("proofreader_kind") == "server"
            ]
            for process in server_matches:
                try:
                    pid = int(process.get("ProcessId"))
                except (TypeError, ValueError):
                    continue
                if pid in attempted:
                    continue
                attempted.add(pid)
                kill_results.append(killer(pid))
            after = inspect_proofreader_runtime(
                work_dir,
                dataset_list=dataset_list,
                process_infos=None,
                write_state=False,
            )
            still_running = bool(after.get("process_alive"))
    else:
        # Injected process snapshots are immutable test fixtures; infer success
        # from the kill results rather than pretending to re-query the OS.
        still_running = bool(matches) and not any(
            bool(item.get("succeeded")) for item in kill_results
        )

    now = _now_iso()
    status = "failed" if still_running else "stopped"
    report = {
        "schema_version": "voicelab_proofreader_stop_report_v1",
        "status": status,
        "work_dir": str(work_dir),
        "dataset_list": inspected.get("dataset_list"),
        "created_at": now,
        "matched_processes": matches,
        "kill_results": kill_results,
        "still_running": still_running,
    }
    report_path = proofreader_stop_report_path(work_dir)
    _atomic_write_json(report_path, report)

    final_state = dict(inspected)
    final_state.update(
        {
            "status": status,
            "process_alive": still_running,
            "still_running": still_running,
            "launcher_pid": None if not still_running else inspected.get("launcher_pid"),
            "server_pid": None if not still_running else inspected.get("server_pid"),
            "stopped_at": None if still_running else now,
            "stop_report_path": str(report_path),
            "last_error": "Matching proofreader process remained alive after stop." if still_running else None,
            "kill_results": kill_results,
        }
    )
    state_path = write_proofreader_state(work_dir, final_state)
    final_state["runtime_state_path"] = str(state_path)
    final_state["stop_report_path"] = str(report_path)
    return final_state


__all__ = [
    "PROOFREADER_RUNTIME_SCHEMA_VERSION",
    "PROOFREADER_RUNTIME_FILENAME",
    "PROOFREADER_STOP_REPORT_FILENAME",
    "proofreader_state_path",
    "proofreader_stop_report_path",
    "read_proofreader_state",
    "write_proofreader_state",
    "record_proofreader_launch",
    "proofreader_process_kind",
    "find_proofreader_processes",
    "inspect_proofreader_runtime",
    "stop_proofreader_runtime",
]
