from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LoggedProcessResult:
    command: tuple[str, ...]
    returncode: int
    log_path: Path
    elapsed_seconds: float
    timed_out: bool = False
    terminal_requested: bool = False
    terminal_started: bool = False
    terminal_error: str | None = None


def _quote_powershell_single(value: str | Path) -> str:
    return str(value).replace("'", "''")


def launch_log_tail_terminal(
    log_path: str | Path,
    *,
    title: str,
    tail_lines: int = 80,
    keep_open: bool = True,
    extra_log_paths: Sequence[str | Path] = (),
) -> subprocess.Popen[bytes] | None:
    """Open a Windows PowerShell window that tails one or more UTF-8 logs.

    The terminal is viewer-only: closing it does not stop the underlying task.

    Important Windows detail:
    this helper intentionally launches PowerShell directly with
    CREATE_NEW_CONSOLE instead of routing through ``cmd.exe /c start``. Training
    titles can contain characters such as ``|``; CMD treats those characters as
    shell metacharacters even when they belong to the PowerShell command string,
    which previously produced errors such as ``'run' is not recognized``.
    """

    if os.name != "nt":
        return None

    resolved_logs = [
        Path(log_path).expanduser().resolve(strict=False),
        *[
            Path(value).expanduser().resolve(strict=False)
            for value in extra_log_paths
        ],
    ]
    deduped_logs: list[Path] = []
    seen: set[str] = set()
    for item in resolved_logs:
        key = str(item).casefold()
        if key in seen:
            continue
        seen.add(key)
        item.parent.mkdir(parents=True, exist_ok=True)
        deduped_logs.append(item)

    safe_logs = [_quote_powershell_single(item) for item in deduped_logs]
    safe_title = str(title).replace('"', "'")
    no_exit = ["-NoExit"] if keep_open else []
    ps_array = ", ".join(f"'{value}'" for value in safe_logs)

    ps_command = (
        f'$Host.UI.RawUI.WindowTitle = "{safe_title}"; '
        f"$paths = @({ps_array}); "
        "Write-Host 'Resonastra log viewer'; "
        "Write-Host 'This window only displays logs. The WebUI/Worker still manages the task.'; "
        "Write-Host ''; "
        "$paths | ForEach-Object { Write-Host ('  ' + $_) }; "
        "Write-Host ''; "
        "while (($paths | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -gt 0) "
        "{ Start-Sleep -Milliseconds 250 }; "
        f"Get-Content -LiteralPath $paths -Tail {max(1, int(tail_lines))} -Wait"
    )

    creationflags = 0
    if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
        creationflags |= int(subprocess.CREATE_NEW_CONSOLE)
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= int(subprocess.CREATE_NEW_PROCESS_GROUP)

    return subprocess.Popen(
        [
            "powershell.exe",
            *no_exit,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_command,
        ],
        cwd=str(deduped_logs[0].parent),
        shell=False,
        creationflags=creationflags,
    )


def terminate_process_tree(process: subprocess.Popen[object]) -> None:
    """Terminate one process and its descendants when possible."""

    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_logged_process(
    command: Sequence[str | Path],
    *,
    cwd: str | Path,
    log_path: str | Path,
    timeout_seconds: float | None = None,
    show_terminal_progress: bool = False,
    terminal_title: str = "Resonastra Progress",
    env: Mapping[str, str] | None = None,
) -> LoggedProcessResult:
    """Run a subprocess with merged stdout/stderr written live to one log.

    The optional terminal is only a log viewer. Failure to launch that viewer
    is recorded in the result but does not prevent the real subprocess from
    running.
    """

    resolved_log = Path(log_path).expanduser().resolve(strict=False)
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    resolved_cwd = Path(cwd).expanduser().resolve(strict=False)
    command_text = tuple(str(item) for item in command)

    process_env = os.environ.copy()
    process_env["PYTHONUNBUFFERED"] = "1"
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})

    started = time.monotonic()
    timed_out = False
    returncode = -1
    terminal_requested = bool(show_terminal_progress)
    terminal_started = False
    terminal_error: str | None = None

    # Reset the log before opening the tail terminal, so the terminal always
    # follows the current run rather than an older file with the same name.
    with resolved_log.open(
        "w",
        encoding="utf-8",
        errors="replace",
        buffering=1,
    ) as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Resonastra command: {' '.join(command_text)}\n")
        log_file.write(f"Working directory: {resolved_cwd}\n")
        log_file.write("stdout/stderr mode: merged\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()

    if terminal_requested:
        try:
            terminal_process = launch_log_tail_terminal(
                resolved_log,
                title=terminal_title,
                tail_lines=80,
                keep_open=True,
            )
            terminal_started = terminal_process is not None
            if terminal_process is None and os.name != "nt":
                terminal_error = "Terminal progress windows are only available on Windows."
        except Exception as exc:
            terminal_error = f"{type(exc).__name__}: {exc}"
            with resolved_log.open(
                "a",
                encoding="utf-8",
                errors="replace",
                buffering=1,
            ) as log_file:
                log_file.write(
                    "[Resonastra] Terminal viewer could not be opened; "
                    "the main process will continue.\n"
                )
                log_file.write(f"[Resonastra] terminal_error={terminal_error}\n")
                log_file.flush()

    with resolved_log.open(
        "a",
        encoding="utf-8",
        errors="replace",
        buffering=1,
    ) as log_file:
        process = subprocess.Popen(
            list(command_text),
            cwd=str(resolved_cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env,
        )
        try:
            returncode = int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(process)
            returncode = -124
            log_file.write(
                "\n[Resonastra] Process timed out and was terminated.\n"
            )
            log_file.flush()

        elapsed_seconds = round(time.monotonic() - started, 4)
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(
            f"[Resonastra] Process finished with returncode={returncode}.\n"
        )
        log_file.write(
            f"[Resonastra] elapsed_seconds={elapsed_seconds:.4f}.\n"
        )
        if timed_out:
            log_file.write("[Resonastra] timed_out=true.\n")
        if terminal_error:
            log_file.write(f"[Resonastra] terminal_error={terminal_error}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()

    return LoggedProcessResult(
        command=command_text,
        returncode=returncode,
        log_path=resolved_log,
        elapsed_seconds=elapsed_seconds,
        timed_out=timed_out,
        terminal_requested=terminal_requested,
        terminal_started=terminal_started,
        terminal_error=terminal_error,
    )


__all__ = [
    "LoggedProcessResult",
    "launch_log_tail_terminal",
    "terminate_process_tree",
    "run_logged_process",
]
