from __future__ import annotations

import json
import os
import re
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .adapter import (
    DataFactoryAdapter,
    DataFactoryCommand,
    DataFactoryCommandPlan,
    create_data_factory_adapter,
)
from .config import DataFactoryConfig
from .request import DataFactoryRequest
from .result import (
    DATA_FACTORY_RESULT_SCHEMA_VERSION,
    DATA_FACTORY_STATUS_FAILED,
    DATA_FACTORY_STATUS_PARTIAL,
    DATA_FACTORY_STATUS_PREPARED,
    DATA_FACTORY_STATUS_SUCCEEDED,
    DataFactoryArtifactPaths,
    DataFactoryErrorInfo,
    DataFactoryResult,
    DataFactoryStepResult,
    DataFactoryTiming,
    write_data_factory_result_json,
)


PathLike = str | Path

TAIL_CHAR_LIMIT = 6000
TIMEOUT_RETURNCODE = -124


DEFAULT_TERMINAL_PROGRESS_COMMAND_NAMES = frozenset(
    {
        "prepare",
        "preprocess_stage2_pt",
        "export_continuous_semantic_cache",
        "export_fewshot_style_cache",
    }
)


@dataclass(frozen=True)
class DataFactoryExecutionOptions:
    """Execution options for subprocess running.

    中文说明：
        数据工厂子进程执行选项。

        P10-11-3 主要用于用户版 UI：
        - 写入实时日志文件；
        - 可选打开终端窗口查看日志；
        - UI 仍然等待真实任务结束。
    """

    show_terminal_progress: bool = False
    log_subprocess_output: bool = True
    terminal_tail_lines: int = 80
    terminal_progress_command_names: frozenset[str] = DEFAULT_TERMINAL_PROGRESS_COMMAND_NAMES


def now_iso() -> str:
    """Return local ISO timestamp."""

    return datetime.now().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, Path, list and dict values to JSON-safe values."""

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    return value


def write_json(obj: Any, path: Path) -> Path:
    """Write JSON object and return path."""

    path = Path(path).resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def request_to_dict(request: DataFactoryRequest) -> dict[str, Any]:
    """Convert DataFactoryRequest to a stable JSON dictionary.

    中文说明：
        将 DataFactoryRequest 转成稳定 JSON。

        这里显式保留 steps，避免 command_plan 中只看到命令，
        但看不到用户原始勾选了哪些步骤。
    """

    return {
        "schema_version": request.schema_version,
        "raw_input_dir": str(request.raw_input_dir),
        "work_dir": None if request.work_dir is None else str(request.work_dir),
        "speaker_name": str(request.speaker_name),
        "language": str(request.language),

        "asr_backend": str(request.asr_backend),
        "asr_model_size": str(request.asr_model_size),
        "asr_precision": str(request.asr_precision),

        "enable_uvr": bool(request.enable_uvr),
        "enable_denoise": bool(request.enable_denoise),

        "threshold": int(request.threshold),
        "min_length": int(request.min_length),
        "min_interval": int(request.min_interval),
        "hop_size": int(request.hop_size),
        "max_sil_kept": int(request.max_sil_kept),
        "normalize_max": float(request.normalize_max),
        "alpha_mix": float(request.alpha_mix),

        "export_list": bool(request.export_list),
        "export_stage2_manifest": bool(request.export_stage2_manifest),

        "prompt_mode": str(request.prompt_mode),
        "min_prompt_sec": float(request.min_prompt_sec),
        "max_prompt_sec": float(request.max_prompt_sec),
        "prefer_prompt_sec": float(request.prefer_prompt_sec),
        "allow_self_prompt": bool(request.allow_self_prompt),

        "overwrite_work_dir": bool(request.overwrite_work_dir),
        "dry_run": bool(request.dry_run),
        "timeout_seconds": request.timeout_seconds,

        "steps": {
            "run_prepare": bool(request.steps.run_prepare),
            "run_manifest_health_check": bool(request.steps.run_manifest_health_check),
            "run_export_fewshot_stage2_manifest": bool(
                request.steps.run_export_fewshot_stage2_manifest
            ),
            "run_stage2_manifest_health_check": bool(
                request.steps.run_stage2_manifest_health_check
            ),
            "run_stage2_pt_preprocess": bool(request.steps.run_stage2_pt_preprocess),
            "run_continuous_semantic_cache": bool(
                request.steps.run_continuous_semantic_cache
            ),
            "run_style_cache": bool(request.steps.run_style_cache),
            "run_split_train_val": bool(request.steps.run_split_train_val),
            "run_filter_bad_samples": bool(request.steps.run_filter_bad_samples),
        },
    }


def _tail_text(value: Optional[str], limit: int = TAIL_CHAR_LIMIT) -> Optional[str]:
    """Return tail text for stdout/stderr storage."""

    if value is None:
        return None

    text = str(value)
    if len(text) <= limit:
        return text

    return text[-limit:]


def _safe_filename(value: str) -> str:
    """Return filesystem-safe filename part."""

    text = str(value).strip() or "command"
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("._") or "command"


def _read_text_tail(path: Path, limit: int = TAIL_CHAR_LIMIT) -> str:
    """Read last characters from UTF-8 text file."""

    path = Path(path)
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8", errors="replace")
    return _tail_text(text, limit=limit) or ""


def _build_command_log_path(
    *,
    log_dir: Path,
    command: DataFactoryCommand,
) -> Path:
    """Build a stable log path for one command."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _safe_filename(command.name)
    return Path(log_dir) / f"{timestamp}_{name}.log"


def _quote_powershell_single(value: Path) -> str:
    """Quote a path for PowerShell single-quoted string."""

    return str(value).replace("'", "''")


def _launch_log_tail_terminal(
    *,
    log_path: Path,
    title: str,
    tail_lines: int = 80,
) -> None:
    """Open a Windows terminal that tails a log file.

    中文说明：
        打开一个新的 Windows 终端窗口，实时查看 log 文件。

        这个终端只负责显示日志；真实任务仍由 service 当前进程控制。
        使用 powershell -NoExit，因此任务结束后终端仍保留。
    """

    if os.name != "nt":
        return

    log_path = Path(log_path).resolve(strict=False)
    safe_log = _quote_powershell_single(log_path)
    safe_title = str(title).replace('"', "'")

    ps_command = (
        f"$Host.UI.RawUI.WindowTitle = \"{safe_title}\"; "
        f"Write-Host 'Resonastra log: {safe_log}'; "
        f"Write-Host 'Close this window manually when you no longer need it.'; "
        f"Write-Host ''; "
        f"while (-not (Test-Path -LiteralPath '{safe_log}')) "
        f"{{ Start-Sleep -Milliseconds 300 }}; "
        f"Get-Content -LiteralPath '{safe_log}' -Tail {int(tail_lines)} -Wait"
    )

    subprocess.Popen(
        [
            "cmd.exe",
            "/c",
            "start",
            "Resonastra · Data Factory Log",
            "powershell",
            "-NoExit",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_command,
        ],
        cwd=str(log_path.parent),
        shell=False,
    )


def _command_to_step_result(command: DataFactoryCommand) -> DataFactoryStepResult:
    """Convert planned command to DataFactoryStepResult.

    中文说明：
        P10-3/P10-4 dry-run 不执行命令，所以 step status 固定为 planned。
    """

    return DataFactoryStepResult(
        name=command.name,
        status="planned",
        command=[str(x) for x in command.argv],
        returncode=None,
        stdout_tail=None,
        stderr_tail=None,
        report_path=command.expected_report_path,
        message="Command planned but not executed.",
        extra={
            "printable": command.to_printable(),
            "expected_output_paths": {
                str(k): str(v) for k, v in command.expected_output_paths.items()
            },
        },
    )


def _executed_command_to_step_result(
    command: DataFactoryCommand,
    *,
    returncode: int,
    stdout: Optional[str],
    stderr: Optional[str],
    elapsed_seconds: float,
    timed_out: bool = False,
    combined_log_path: Optional[Path] = None,
) -> DataFactoryStepResult:
    """Convert executed command to DataFactoryStepResult."""

    if timed_out:
        status = "timeout"
        message = "Command timed out."
    elif returncode == 0:
        status = "succeeded"
        message = "Command executed successfully."
    else:
        status = "failed"
        message = f"Command failed with returncode={returncode}."

    extra = {
        "printable": command.to_printable(),
        "elapsed_seconds": float(elapsed_seconds),
        "timed_out": bool(timed_out),
        "expected_output_paths": {
            str(k): str(v) for k, v in command.expected_output_paths.items()
        },
    }

    if combined_log_path is not None:
        extra["combined_log_path"] = str(combined_log_path)

    return DataFactoryStepResult(
        name=command.name,
        status=status,
        command=[str(x) for x in command.argv],
        returncode=int(returncode),
        stdout_tail=_tail_text(stdout),
        stderr_tail=_tail_text(stderr),
        report_path=command.expected_report_path,
        message=message,
        extra=extra,
    )



def _augment_plan_dict(
    plan: DataFactoryCommandPlan,
    *,
    command_plan_json_path: Path,
    execute: bool,
) -> dict[str, Any]:
    """Return command plan dict with extra protocol metadata."""

    data = plan.to_dict()
    data["schema_version"] = "voicelab_data_factory_command_plan_v1"
    data["command_plan_json_path"] = str(command_plan_json_path)
    data["generated_at"] = now_iso()
    data["execute"] = bool(execute)
    data["note"] = (
        "Data factory command plan generated by DataFactoryService. "
        "If execute=false, no subprocess was executed."
    )
    return data


def _result_status_from_steps(
    steps: list[DataFactoryStepResult],
    *,
    planned_count: int,
) -> str:
    """Infer top-level result status from executed steps."""

    if planned_count <= 0:
        return DATA_FACTORY_STATUS_PREPARED

    if not steps:
        return DATA_FACTORY_STATUS_FAILED

    if len(steps) == planned_count and all(step.returncode == 0 for step in steps):
        return DATA_FACTORY_STATUS_SUCCEEDED

    if any(step.returncode == 0 for step in steps):
        return DATA_FACTORY_STATUS_PARTIAL

    return DATA_FACTORY_STATUS_FAILED


def _result_message_from_status(status: str) -> str:
    """Return user-facing result message."""

    if status == DATA_FACTORY_STATUS_SUCCEEDED:
        return "Data factory execution finished successfully."

    if status == DATA_FACTORY_STATUS_PARTIAL:
        return (
            "Data factory execution stopped after a failure. "
            "Some earlier steps may have completed successfully."
        )

    if status == DATA_FACTORY_STATUS_FAILED:
        return "Data factory execution failed."

    return "Data factory result prepared."


class DataFactoryService:
    """User-edition data factory service.

    中文说明：
        用户版数据工厂 service。

        P10-3:
            - dry-run result only

        P10-4:
            - explicit execution mode
            - run(request) 默认仍是 dry-run
            - run(request, execute=True) 或 execute(request) 才真正执行命令
    """

    def __init__(
        self,
        *,
        adapter: Optional[DataFactoryAdapter] = None,
        project_root: Optional[PathLike] = None,
        python_executable: Optional[PathLike] = None,
        config: Optional[DataFactoryConfig] = None,
        execution_options: Optional[DataFactoryExecutionOptions] = None,
        show_terminal_progress: Optional[bool] = None,
    ) -> None:
        self.adapter = (
            adapter
            if adapter is not None
            else create_data_factory_adapter(
                project_root=project_root,
                python_executable=python_executable,
                config=config,
            )
        )

        if execution_options is None:
            execution_options = DataFactoryExecutionOptions()

        if show_terminal_progress is not None:
            execution_options = replace(
                execution_options,
                show_terminal_progress=bool(show_terminal_progress),
            )

        self.execution_options = execution_options


    # ------------------------------------------------------------------
    # Protocol preparation
    # 协议文件准备
    # ------------------------------------------------------------------
    def _build_plan_and_prepare_paths(
        self,
        request: DataFactoryRequest,
        *,
        execute: bool,
    ) -> tuple[DataFactoryCommandPlan, DataFactoryArtifactPaths, Path, Path, Path]:
        """Build command plan, create work_dir, and persist request/plan JSON."""

        plan = self.adapter.build_command_plan(request)
        artifacts = plan.artifacts

        work_dir = Path(artifacts.work_dir).resolve(strict=False)
        work_dir.mkdir(parents=True, exist_ok=True)

        request_json_path = artifacts.request_json_path or (work_dir / "data_factory_request.json")
        result_json_path = artifacts.result_json_path or (work_dir / "data_factory_result.json")
        command_plan_json_path = work_dir / "data_factory_command_plan.json"

        artifacts = replace(
            artifacts,
            request_json_path=request_json_path,
            result_json_path=result_json_path,
            command_plan_json_path=command_plan_json_path,
        )

        write_json(request_to_dict(request), request_json_path)
        write_json(
            _augment_plan_dict(
                plan,
                command_plan_json_path=command_plan_json_path,
                execute=execute,
            ),
            command_plan_json_path,
        )

        return plan, artifacts, request_json_path, command_plan_json_path, result_json_path

    # ------------------------------------------------------------------
    # Dry-run mode
    # Dry-run 模式
    # ------------------------------------------------------------------
    def build_dry_run_result(
        self,
        request: DataFactoryRequest,
    ) -> DataFactoryResult:
        """Build and persist dry-run result.

        中文说明：
            构建并保存 dry-run 结果。
            该函数不会执行 command plan 中的命令。
        """

        started_at = now_iso()
        t0 = time.perf_counter()

        plan, artifacts, request_json_path, command_plan_json_path, result_json_path = (
            self._build_plan_and_prepare_paths(
                request,
                execute=False,
            )
        )

        step_results = [
            _command_to_step_result(command)
            for command in plan.commands
        ]

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=DATA_FACTORY_STATUS_PREPARED,
            message=(
                "Data factory dry-run result prepared. "
                "No heavy job was executed."
            ),
            artifacts=artifacts,
            speaker_name=request.speaker_name,
            language=request.language,
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=step_results,
            error=None,
            extra={
                "dry_run_only": True,
                "execute": False,
                "num_planned_commands": len(plan.commands),
                "command_plan_json_path": str(command_plan_json_path),
                "request_json_path": str(request_json_path),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    # ------------------------------------------------------------------
    # Execution mode
    # 执行模式
    # ------------------------------------------------------------------
    def execute(
        self,
        request: DataFactoryRequest,
    ) -> DataFactoryResult:
        """Execute planned data factory commands sequentially.

        中文说明：
            按 command plan 顺序执行命令。

            注意：
                只有显式调用 execute(request)，或 run(request, execute=True)，
                才会进入这里。
        """

        started_at = now_iso()
        t0 = time.perf_counter()

        plan, artifacts, request_json_path, command_plan_json_path, result_json_path = (
            self._build_plan_and_prepare_paths(
                request,
                execute=True,
            )
        )

        step_results: list[DataFactoryStepResult] = []

        if request.dry_run:
            result = DataFactoryResult(
                schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
                status=DATA_FACTORY_STATUS_PREPARED,
                message=(
                    "Execution was requested, but DataFactoryRequest.dry_run=True. "
                    "No heavy job was executed."
                ),
                artifacts=artifacts,
                speaker_name=request.speaker_name,
                language=request.language,
                timing=DataFactoryTiming(
                    started_at=started_at,
                    finished_at=now_iso(),
                    total_seconds=float(time.perf_counter() - t0),
                ),
                steps=[
                    _command_to_step_result(command)
                    for command in plan.commands
                ],
                error=None,
                extra={
                    "dry_run_only": True,
                    "execute": False,
                    "execution_blocked_by_request_dry_run": True,
                    "num_planned_commands": len(plan.commands),
                    "command_plan_json_path": str(command_plan_json_path),
                    "request_json_path": str(request_json_path),
                    "result_json_path": str(result_json_path),
                },
            )
            write_data_factory_result_json(result, result_json_path)
            return result

        for command in plan.commands:
            step_result = self._run_one_command(
                command,
                timeout_seconds=request.timeout_seconds,
                log_dir=artifacts.work_dir / "logs" / "commands",
            )
            step_results.append(step_result)

            # Stop on first failure / timeout.
            # 遇到第一个失败或超时即停止。
            if step_result.returncode != 0:
                break

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        status = _result_status_from_steps(
            step_results,
            planned_count=len(plan.commands),
        )

        error: Optional[DataFactoryErrorInfo] = None
        if status in {DATA_FACTORY_STATUS_FAILED, DATA_FACTORY_STATUS_PARTIAL}:
            failed_step = next(
                (step for step in step_results if step.returncode != 0),
                None,
            )
            error = DataFactoryErrorInfo(
                error_type="DataFactoryCommandExecutionError",
                message=(
                    "One data factory command failed."
                    if failed_step is not None
                    else "Data factory execution did not complete."
                ),
                stage=None if failed_step is None else failed_step.name,
                traceback_text=None,
                hint=(
                    "Open data_factory_result.json and inspect the failed step's "
                    "stderr_tail/stdout_tail. You can also copy the printable command "
                    "from data_factory_command_plan.json and run it manually."
                ),
            )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=_result_message_from_status(status),
            artifacts=artifacts,
            speaker_name=request.speaker_name,
            language=request.language,
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=step_results,
            error=error,
            extra={
                "dry_run_only": False,
                "execute": True,
                "num_planned_commands": len(plan.commands),
                "num_executed_commands": len(step_results),
                "command_plan_json_path": str(command_plan_json_path),
                "request_json_path": str(request_json_path),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def _run_one_command(
        self,
        command: DataFactoryCommand,
        *,
        timeout_seconds: Optional[float],
        log_dir: Optional[PathLike] = None,
        show_terminal_progress: Optional[bool] = None,
    ) -> DataFactoryStepResult:
        """Run one planned command and capture stdout/stderr.

        中文说明：
            P10-11-3 后支持两种模式：

            1. 旧模式：
                subprocess.run(capture_output=True)

            2. 日志模式：
                subprocess.Popen(stdout=log_file, stderr=STDOUT)
                同时可选打开终端窗口 tail 日志文件。

            用户版 UI 默认使用日志模式 + 终端 tail。
        """

        options = self.execution_options
        should_log = bool(options.log_subprocess_output) or bool(options.show_terminal_progress)

        if show_terminal_progress is None:
            show_terminal_progress = bool(options.show_terminal_progress)

        should_open_terminal = (
            bool(show_terminal_progress)
            and command.name in set(options.terminal_progress_command_names)
        )

        if not should_log:
            return self._run_one_command_capture(
                command,
                timeout_seconds=timeout_seconds,
            )

        resolved_log_dir = (
            Path(log_dir).resolve(strict=False)
            if log_dir is not None
            else Path(command.cwd).resolve(strict=False) / "logs" / "commands"
        )
        resolved_log_dir.mkdir(parents=True, exist_ok=True)

        combined_log_path = _build_command_log_path(
            log_dir=resolved_log_dir,
            command=command,
        )

        if should_open_terminal:
            _launch_log_tail_terminal(
                log_path=combined_log_path,
                title=f"Resonastra · Data Factory - {command.name}",
                tail_lines=options.terminal_tail_lines,
            )

        t0 = time.perf_counter()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        with combined_log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write(f"Resonastra DataFactory command: {command.name}\n")
            log_file.write(f"Started at: {now_iso()}\n")
            log_file.write(f"CWD: {command.cwd}\n")
            log_file.write(f"Command: {command.to_printable()}\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.flush()

            try:
                process = subprocess.Popen(
                    [str(x) for x in command.argv],
                    cwd=str(command.cwd),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )

                timed_out = False
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    returncode = TIMEOUT_RETURNCODE
                    log_file.write("\n" + "=" * 80 + "\n")
                    log_file.write(
                        f"Command timed out after {timeout_seconds} seconds.\n"
                    )
                    log_file.write("=" * 80 + "\n")
                    log_file.flush()

                elapsed = time.perf_counter() - t0

                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write(f"Finished at: {now_iso()}\n")
                log_file.write(f"Return code: {returncode}\n")
                log_file.write(f"Elapsed seconds: {elapsed:.3f}\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()

                tail = _read_text_tail(combined_log_path)

                return _executed_command_to_step_result(
                    command,
                    returncode=int(returncode),
                    stdout=tail,
                    stderr="" if int(returncode) == 0 else tail,
                    elapsed_seconds=float(elapsed),
                    timed_out=bool(timed_out),
                    combined_log_path=combined_log_path,
                )

            except Exception:
                elapsed = time.perf_counter() - t0
                tb = traceback.format_exc()

                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write("Command launcher failed before normal completion.\n")
                log_file.write(tb)
                log_file.write("=" * 80 + "\n")
                log_file.flush()

                tail = _read_text_tail(combined_log_path)

                return _executed_command_to_step_result(
                    command,
                    returncode=1,
                    stdout=tail,
                    stderr=tail,
                    elapsed_seconds=float(elapsed),
                    timed_out=False,
                    combined_log_path=combined_log_path,
                )

    def _run_one_command_capture(
        self,
        command: DataFactoryCommand,
        *,
        timeout_seconds: Optional[float],
    ) -> DataFactoryStepResult:
        """Original capture-output execution path."""

        t0 = time.perf_counter()

        try:
            completed = subprocess.run(
                [str(x) for x in command.argv],
                cwd=str(command.cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )

            elapsed = time.perf_counter() - t0
            return _executed_command_to_step_result(
                command,
                returncode=int(completed.returncode),
                stdout=completed.stdout,
                stderr=completed.stderr,
                elapsed_seconds=float(elapsed),
                timed_out=False,
            )

        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0

            stdout = exc.stdout
            stderr = exc.stderr

            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")

            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            timeout_text = (
                f"Command timed out after {timeout_seconds} seconds."
                if timeout_seconds is not None
                else "Command timed out."
            )

            stderr_text = "\n".join(
                part for part in [str(stderr or ""), timeout_text] if part.strip()
            )

            return _executed_command_to_step_result(
                command,
                returncode=TIMEOUT_RETURNCODE,
                stdout=str(stdout or ""),
                stderr=stderr_text,
                elapsed_seconds=float(elapsed),
                timed_out=True,
            )


    def launch_proofread(
        self,
        work_dir: str | Path,
        *,
        webui_port_subfix: int = 9871,
        is_share: str = "False",
        g_batch: int = 10,
        backup_suffix: str = "before_proofread",
        overwrite_backup: bool = False,
    ) -> dict[str, Any]:
        """Launch legacy proofread WebUI without blocking the main UI.

        中文说明：
            启动旧版人工校对器，但不阻塞当前数据工厂 UI。

            该函数使用 subprocess.Popen。
        """

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        if artifacts.dataset_list_path is None or not artifacts.dataset_list_path.exists():
            raise FileNotFoundError(
                f"dataset.list not found. Please run prepare first: {artifacts.dataset_list_path}"
            )

        command = self.adapter.build_launch_proofread_command(
            artifacts,
            webui_port_subfix=webui_port_subfix,
            is_share=is_share,
            g_batch=g_batch,
            backup_suffix=backup_suffix,
            overwrite_backup=overwrite_backup,
        )

        logs_dir = resolved_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        stdout_log_path = logs_dir / "proofread_stdout.log"
        stderr_log_path = logs_dir / "proofread_stderr.log"

        stdout_f = stdout_log_path.open("a", encoding="utf-8", errors="replace")
        stderr_f = stderr_log_path.open("a", encoding="utf-8", errors="replace")

        process = subprocess.Popen(
            [str(x) for x in command.argv],
            cwd=str(command.cwd),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        proofread_url = f"http://127.0.0.1:{int(webui_port_subfix)}"

        payload = {
            "status": "launched",
            "message": "Proofread WebUI launched. Please open the proofread URL and edit dataset.list.",
            "pid": int(process.pid),
            "proofread_url": proofread_url,
            "work_dir": str(resolved_work_dir),
            "dataset_list": str(artifacts.dataset_list_path),
            "command": [str(x) for x in command.argv],
            "printable": command.to_printable(),
            "proofread_launch_report": str(command.expected_report_path),
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
            "expected_output_paths": {
                str(k): str(v) for k, v in command.expected_output_paths.items()
            },
        }

        report_path = resolved_work_dir / "06_export" / "proofread_bridge_launch.json"
        write_json(payload, report_path)
        payload["bridge_report_path"] = str(report_path)
        return payload


    def rebuild_corrected_manifest(
        self,
        work_dir: str | Path,
        *,
        duration_tolerance_sec: float = 0.05,
        prompt_mode: str = "self",
        timeout_seconds: Optional[float] = 600,
    ) -> DataFactoryResult:
        """Rebuild corrected manifest files after manual proofreading.

        中文说明：
            人工校对完成后，根据校对后的 dataset.list 重建：

            - manifest.corrected.jsonl
            - stage2_manifest.corrected.jsonl
            - correction_report.json
        """

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        command = self.adapter.build_rebuild_corrected_manifest_command(
            artifacts,
            duration_tolerance_sec=duration_tolerance_sec,
            prompt_mode=prompt_mode,
        )

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        result_json_path = resolved_work_dir / "data_factory_rebuild_corrected_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Corrected manifest files rebuilt successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Failed to rebuild corrected manifest files."
            error = DataFactoryErrorInfo(
                error_type="RebuildCorrectedManifestError",
                message=step_result.message or "rebuild_corrected_manifest failed.",
                stage="rebuild_corrected_manifest",
                traceback_text=None,
                hint=(
                    "Check dataset.list after proofreading, manifest.jsonl, "
                    "dataset.before_proofread.list, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "rebuild_corrected_manifest",
                "work_dir": str(resolved_work_dir),
                "duration_tolerance_sec": float(duration_tolerance_sec),
                "prompt_mode": str(prompt_mode),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def export_corrected_fewshot_stage2_manifest(
        self,
        work_dir: str | Path,
        *,
        prompt_mode: str = "speaker_pool",
        min_prompt_sec: float = 3.0,
        max_prompt_sec: float = 10.0,
        prefer_prompt_sec: float = 6.0,
        allow_self_prompt: bool = True,
        fixed_prompt_wav_path: Optional[str | Path] = None,
        strict_prompt_duration: bool = True,
        skip_invalid_rows: bool = False,
        allow_empty_text: bool = False,
        timeout_seconds: Optional[float] = 600,
    ) -> DataFactoryResult:
        """Export final few-shot Stage2 manifest from manifest.corrected.jsonl.

        中文说明：
            从人工校对后的 manifest.corrected.jsonl 导出最终的：

                stage2_manifest.fewshot.jsonl

            这是 Stage2 .pt 预处理前的最后一个 manifest 导出步骤。
        """

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        if artifacts.manifest_corrected_path is None or not artifacts.manifest_corrected_path.exists():
            raise FileNotFoundError(
                "manifest.corrected.jsonl not found. "
                f"Please run rebuild corrected manifest first: {artifacts.manifest_corrected_path}"
            )

        command = self.adapter.build_export_corrected_fewshot_stage2_manifest_command(
            artifacts,
            prompt_mode=prompt_mode,
            min_prompt_sec=min_prompt_sec,
            max_prompt_sec=max_prompt_sec,
            prefer_prompt_sec=prefer_prompt_sec,
            allow_self_prompt=allow_self_prompt,
            fixed_prompt_wav_path=fixed_prompt_wav_path,
            strict_prompt_duration=strict_prompt_duration,
            skip_invalid_rows=skip_invalid_rows,
            allow_empty_text=allow_empty_text,
        )

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        result_json_path = resolved_work_dir / "data_factory_export_corrected_fewshot_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Corrected few-shot Stage2 manifest exported successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Failed to export corrected few-shot Stage2 manifest."
            error = DataFactoryErrorInfo(
                error_type="ExportCorrectedFewshotStage2ManifestError",
                message=step_result.message or "export_corrected_fewshot_stage2_manifest failed.",
                stage="export_corrected_fewshot_stage2_manifest",
                traceback_text=None,
                hint=(
                    "Check manifest.corrected.jsonl, prompt duration settings, "
                    "prompt_mode, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "export_corrected_fewshot_stage2_manifest",
                "work_dir": str(resolved_work_dir),
                "prompt_mode": str(prompt_mode),
                "min_prompt_sec": float(min_prompt_sec),
                "max_prompt_sec": float(max_prompt_sec),
                "prefer_prompt_sec": float(prefer_prompt_sec),
                "allow_self_prompt": bool(allow_self_prompt),
                "strict_prompt_duration": bool(strict_prompt_duration),
                "skip_invalid_rows": bool(skip_invalid_rows),
                "allow_empty_text": bool(allow_empty_text),
                "fixed_prompt_wav_path": (
                    None if fixed_prompt_wav_path is None else str(fixed_prompt_wav_path)
                ),
                "stage2_manifest_fewshot_path": str(artifacts.stage2_manifest_fewshot_path),
                "prompt_selection_report_path": str(artifacts.prompt_selection_report_path),
                "fewshot_manifest_conversion_report_path": str(
                    artifacts.fewshot_manifest_conversion_report_path
                ),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def preprocess_stage2_pt_dataset(
        self,
        work_dir: str | Path,
        *,
        output_dir: Optional[str | Path] = None,
        device: str = "cuda",
        use_half: bool = False,
        overwrite: bool = False,
        target_sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        timeout_seconds: Optional[float] = None,
    ) -> DataFactoryResult:
        """Run Stage2 .pt offline preprocessing.

        中文说明：
            调用 scripts/preprocess_stage2_fm_dataset.py，把：

                stage2_manifest.fewshot.jsonl

            预处理成 Stage2 训练需要的 `.pt` 样本目录。
        """

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        if artifacts.stage2_manifest_fewshot_path is None or not artifacts.stage2_manifest_fewshot_path.exists():
            raise FileNotFoundError(
                "stage2_manifest.fewshot.jsonl not found. "
                f"Please export corrected few-shot Stage2 manifest first: {artifacts.stage2_manifest_fewshot_path}"
            )

        command = self.adapter.build_preprocess_stage2_pt_command(
            artifacts,
            output_dir=output_dir,
            device=device,
            use_half=use_half,
            overwrite=overwrite,
            target_sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        resolved_output_dir = Path(
            command.expected_output_paths["stage2_pt_root"]
        ).resolve(strict=False)

        result_json_path = resolved_work_dir / "data_factory_preprocess_stage2_pt_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Stage2 .pt preprocessing finished successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Stage2 .pt preprocessing failed."
            error = DataFactoryErrorInfo(
                error_type="PreprocessStage2PtError",
                message=step_result.message or "preprocess_stage2_pt_dataset failed.",
                stage="preprocess_stage2_pt",
                traceback_text=None,
                hint=(
                    "Check stage2_manifest.fewshot.jsonl, prompt/target wav paths, "
                    "GPU/CUDA availability, GSV compatibility modules, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
            stage2_pt_root=resolved_output_dir,
            stage2_pt_health_report_path=resolved_output_dir / "fewshot_health_pt_report.json",
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "preprocess_stage2_pt_dataset",
                "work_dir": str(resolved_work_dir),
                "manifest": str(artifacts.stage2_manifest_fewshot_path),
                "stage2_pt_root": str(resolved_output_dir),
                "device": str(device),
                "use_half": bool(use_half),
                "overwrite": bool(overwrite),
                "target_sr": int(target_sr),
                "n_fft": int(n_fft),
                "hop_length": int(hop_length),
                "win_length": int(win_length),
                "n_mels": int(n_mels),
                "fmin": float(fmin),
                "fmax": float(fmax),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def export_continuous_semantic_cache(
        self,
        work_dir: str | Path,
        *,
        input_root: Optional[str | Path] = None,
        output_root: Optional[str | Path] = None,
        sovits_checkpoint_path: Optional[str | Path] = None,
        device: str = "cuda",
        use_half: bool = False,
        continuous_dtype: str = "float32",
        require_predicted: bool = False,
        overwrite: bool = False,
        max_files: Optional[int] = None,
        copy_non_pt_files: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> DataFactoryResult:
        """Export continuous semantic cache from Stage2 .pt samples."""

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        default_input_root = artifacts.stage2_pt_root
        resolved_input_root = (
            self.adapter.resolve_path(input_root)
            if input_root is not None and str(input_root).strip()
            else default_input_root
        )

        if resolved_input_root is None or not resolved_input_root.exists():
            raise FileNotFoundError(
                f"Stage2 .pt input_root not found: {resolved_input_root}"
            )

        command = self.adapter.build_export_continuous_semantic_cache_command(
            artifacts,
            input_root=resolved_input_root,
            output_root=output_root,
            sovits_checkpoint_path=sovits_checkpoint_path,
            device=device,
            use_half=use_half,
            continuous_dtype=continuous_dtype,
            require_predicted=require_predicted,
            overwrite=overwrite,
            max_files=max_files,
            copy_non_pt_files=copy_non_pt_files,
        )

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        resolved_output_root = Path(
            command.expected_output_paths["continuous_semantic_root"]
        ).resolve(strict=False)

        result_json_path = resolved_work_dir / "data_factory_continuous_semantic_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Continuous semantic cache exported successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Continuous semantic cache export failed."
            error = DataFactoryErrorInfo(
                error_type="ContinuousSemanticCacheError",
                message=step_result.message or "export_continuous_semantic_cache failed.",
                stage="export_continuous_semantic_cache",
                traceback_text=None,
                hint=(
                    "Check Stage2 .pt input_root, semantic_tokens fields, "
                    "GSVSemanticCodecV2 loading, CUDA/device, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
            continuous_semantic_root=resolved_output_root,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "export_continuous_semantic_cache",
                "work_dir": str(resolved_work_dir),
                "input_root": str(resolved_input_root),
                "continuous_semantic_root": str(resolved_output_root),
                "continuous_dtype": str(continuous_dtype),
                "device": str(device),
                "use_half": bool(use_half),
                "overwrite": bool(overwrite),
                "require_predicted": bool(require_predicted),
                "copy_non_pt_files": bool(copy_non_pt_files),
                "max_files": max_files,
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def export_fewshot_style_cache(
        self,
        work_dir: str | Path,
        *,
        input_root: Optional[str | Path] = None,
        output_root: Optional[str | Path] = None,
        device: str = "cuda",
        overwrite: bool = False,
        copy_non_pt_files: bool = False,
        recursive: bool = False,
        max_files: Optional[int] = None,
        extract_prompt_acoustic: bool = True,
        extract_energy: bool = True,
        extract_f0: bool = True,
        extract_speaker_embedding: bool = False,
        prefer_sample_acoustic_meta: bool = True,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        f0_min_hz: float = 50.0,
        f0_max_hz: float = 1100.0,
        speaker_backend: str = "speechbrain_ecapa",
        speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        speaker_savedir: str | Path = "pretrained_models/speechbrain_spkrec_ecapa_voxceleb",
        speaker_device: Optional[str] = None,
        speaker_sample_rate: int = 16000,
        speaker_normalize: bool = True,
        fail_on_error: bool = True,
        timeout_seconds: Optional[float] = None,
    ) -> DataFactoryResult:
        """Export style/F0/speaker cache from Stage2 .pt samples."""

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        default_input_root = (
            artifacts.continuous_semantic_root
            if artifacts.continuous_semantic_root is not None
            else artifacts.stage2_pt_root
        )

        resolved_input_root = (
            self.adapter.resolve_path(input_root)
            if input_root is not None and str(input_root).strip()
            else default_input_root
        )

        if resolved_input_root is None or not resolved_input_root.exists():
            raise FileNotFoundError(
                f"Style cache input_root not found: {resolved_input_root}"
            )

        command = self.adapter.build_export_fewshot_style_cache_command(
            artifacts,
            input_root=resolved_input_root,
            output_root=output_root,
            device=device,
            overwrite=overwrite,
            copy_non_pt_files=copy_non_pt_files,
            recursive=recursive,
            max_files=max_files,
            extract_prompt_acoustic=extract_prompt_acoustic,
            extract_energy=extract_energy,
            extract_f0=extract_f0,
            extract_speaker_embedding=extract_speaker_embedding,
            prefer_sample_acoustic_meta=prefer_sample_acoustic_meta,
            sample_rate=sample_rate,
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
            fail_on_error=fail_on_error,
        )

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        resolved_output_root = Path(
            command.expected_output_paths["style_cache_root"]
        ).resolve(strict=False)

        result_json_path = resolved_work_dir / "data_factory_style_cache_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Style/F0/speaker cache exported successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Style/F0/speaker cache export failed."
            error = DataFactoryErrorInfo(
                error_type="FewshotStyleCacheError",
                message=step_result.message or "export_fewshot_style_cache failed.",
                stage="export_fewshot_style_cache",
                traceback_text=None,
                hint=(
                    "Check input_root .pt fields, prompt_wav_path/target_wav_path, "
                    "F0 extraction settings, speaker embedding model path, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
            style_cache_root=resolved_output_root,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "export_fewshot_style_cache",
                "work_dir": str(resolved_work_dir),
                "input_root": str(resolved_input_root),
                "style_cache_root": str(resolved_output_root),
                "device": str(device),
                "overwrite": bool(overwrite),
                "extract_prompt_acoustic": bool(extract_prompt_acoustic),
                "extract_energy": bool(extract_energy),
                "extract_f0": bool(extract_f0),
                "extract_speaker_embedding": bool(extract_speaker_embedding),
                "speaker_backend": str(speaker_backend),
                "speaker_model_source": str(speaker_model_source),
                "speaker_savedir": str(speaker_savedir),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def split_stage2_pt_dataset(
        self,
        work_dir: str | Path,
        *,
        data_root: Optional[str | Path] = None,
        train_out: Optional[str | Path] = None,
        val_out: Optional[str | Path] = None,
        train_ratio: Optional[float] = 0.9,
        val_count: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 2026,
        mode: str = "copy",
        overwrite: bool = False,
        report_path: Optional[str | Path] = None,
        timeout_seconds: Optional[float] = None,
    ) -> DataFactoryResult:
        """Split final Stage2 .pt cache directory into train/val sets."""

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        default_data_root = artifacts.style_cache_root
        resolved_data_root = (
            self.adapter.resolve_path(data_root)
            if data_root is not None and str(data_root).strip()
            else default_data_root
        )

        if resolved_data_root is None or not resolved_data_root.exists():
            raise FileNotFoundError(f"Split data_root not found: {resolved_data_root}")

        command = self.adapter.build_split_stage2_pt_dataset_command(
            artifacts,
            data_root=resolved_data_root,
            train_out=train_out,
            val_out=val_out,
            train_ratio=train_ratio,
            val_count=val_count,
            shuffle=shuffle,
            seed=seed,
            mode=mode,
            overwrite=overwrite,
            report_path=report_path,
        )

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        resolved_train_out = Path(command.expected_output_paths["train_pt_root"]).resolve(strict=False)
        resolved_val_out = Path(command.expected_output_paths["val_pt_root"]).resolve(strict=False)
        resolved_report_path = Path(command.expected_output_paths["split_report"]).resolve(strict=False)

        result_json_path = resolved_work_dir / "data_factory_split_stage2_pt_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Stage2 .pt train/val split finished successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Stage2 .pt train/val split failed."
            error = DataFactoryErrorInfo(
                error_type="SplitStage2PtDatasetError",
                message=step_result.message or "split_stage2_pt_dataset failed.",
                stage="split_stage2_pt_dataset",
                traceback_text=None,
                hint=(
                    "Check data_root, train_out/val_out, overwrite behavior, "
                    "train_ratio/val_count, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
            train_pt_root=resolved_train_out,
            val_pt_root=resolved_val_out,
            split_report_path=resolved_report_path,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "split_stage2_pt_dataset",
                "work_dir": str(resolved_work_dir),
                "data_root": str(resolved_data_root),
                "train_pt_root": str(resolved_train_out),
                "val_pt_root": str(resolved_val_out),
                "train_ratio": train_ratio,
                "val_count": val_count,
                "shuffle": bool(shuffle),
                "seed": int(seed),
                "mode": str(mode),
                "overwrite": bool(overwrite),
                "split_report_path": str(resolved_report_path),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

    def filter_v662_bad_style_samples(
        self,
        work_dir: str | Path,
        *,
        roots: Optional[list[str | Path]] = None,
        rejected_dir: Optional[str | Path] = None,
        recursive: bool = False,
        mode: str = "move",
        require_target_f0: bool = True,
        require_target_voiced_mask: bool = True,
        reject_all_unvoiced_target_f0: bool = True,
        require_prompt_f0: bool = False,
        check_speaker_fields: bool = True,
        speaker_dim: int = 192,
        report_json: Optional[str | Path] = None,
        report_csv: Optional[str | Path] = None,
        timeout_seconds: Optional[float] = None,
    ) -> DataFactoryResult:
        """Filter bad v6.6.2 style-cache samples from train/val roots."""

        started_at = now_iso()
        t0 = time.perf_counter()

        resolved_work_dir = self.adapter.resolve_path(work_dir)
        artifacts = self.adapter.expected_artifacts(resolved_work_dir)

        command = self.adapter.build_filter_v662_bad_style_samples_command(
            artifacts,
            roots=roots,
            rejected_dir=rejected_dir,
            recursive=recursive,
            mode=mode,
            require_target_f0=require_target_f0,
            require_target_voiced_mask=require_target_voiced_mask,
            reject_all_unvoiced_target_f0=reject_all_unvoiced_target_f0,
            require_prompt_f0=require_prompt_f0,
            check_speaker_fields=check_speaker_fields,
            speaker_dim=speaker_dim,
            report_json=report_json,
            report_csv=report_csv,
        )

        # Explicit existence check for all roots.
        for root in command.expected_output_paths.get("roots_list", []) or []:
            if not Path(root).exists():
                raise FileNotFoundError(f"Filter root not found: {root}")

        step_result = self._run_one_command(
            command,
            timeout_seconds=timeout_seconds,
            log_dir=resolved_work_dir / "logs" / "commands",
        )

        total_seconds = time.perf_counter() - t0
        finished_at = now_iso()

        resolved_rejected_dir = Path(command.expected_output_paths["rejected_dir"]).resolve(strict=False)
        resolved_report_json = Path(command.expected_output_paths["filter_report_json"]).resolve(strict=False)
        resolved_report_csv = Path(command.expected_output_paths["filter_report_csv"]).resolve(strict=False)

        result_json_path = resolved_work_dir / "data_factory_filter_v662_result.json"

        if step_result.returncode == 0:
            status = DATA_FACTORY_STATUS_SUCCEEDED
            message = "Bad style sample filtering finished successfully."
            error = None
        else:
            status = DATA_FACTORY_STATUS_FAILED
            message = "Bad style sample filtering failed."
            error = DataFactoryErrorInfo(
                error_type="FilterV662BadStyleSamplesError",
                message=step_result.message or "filter_v662_bad_style_samples failed.",
                stage="filter_v662_bad_style_samples",
                traceback_text=None,
                hint=(
                    "Check train/val roots, style cache fields, F0/speaker checks, "
                    "mode behavior, and stderr_tail."
                ),
            )

        artifacts = replace(
            artifacts,
            result_json_path=result_json_path,
            rejected_pt_root=resolved_rejected_dir,
            filter_report_json_path=resolved_report_json,
            filter_report_csv_path=resolved_report_csv,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=status,
            message=message,
            artifacts=artifacts,
            speaker_name=None,
            language="zh",
            timing=DataFactoryTiming(
                started_at=started_at,
                finished_at=finished_at,
                total_seconds=float(total_seconds),
            ),
            steps=[step_result],
            error=error,
            extra={
                "operation": "filter_v662_bad_style_samples",
                "work_dir": str(resolved_work_dir),
                "rejected_dir": str(resolved_rejected_dir),
                "recursive": bool(recursive),
                "mode": str(mode),
                "require_target_f0": bool(require_target_f0),
                "require_target_voiced_mask": bool(require_target_voiced_mask),
                "reject_all_unvoiced_target_f0": bool(reject_all_unvoiced_target_f0),
                "require_prompt_f0": bool(require_prompt_f0),
                "check_speaker_fields": bool(check_speaker_fields),
                "speaker_dim": int(speaker_dim),
                "filter_report_json_path": str(resolved_report_json),
                "filter_report_csv_path": str(resolved_report_csv),
                "result_json_path": str(result_json_path),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result


    # ------------------------------------------------------------------
    # Public run wrapper
    # 公开运行入口
    # ------------------------------------------------------------------
    def run(
        self,
        request: DataFactoryRequest,
        *,
        execute: bool = False,
    ) -> DataFactoryResult:
        """Run service.

        中文说明：
            默认行为仍然是 dry-run。

            - run(request)
                只生成 request / command_plan / result

            - run(request, execute=True)
                真实执行 command plan

            - execute(request)
                等价于 run(request, execute=True)
        """

        try:
            if execute:
                return self.execute(request)

            return self.build_dry_run_result(request)

        except Exception as exc:
            return self._build_failure_result(request, exc)

    def _build_failure_result(
        self,
        request: DataFactoryRequest,
        exc: Exception,
    ) -> DataFactoryResult:
        """Build failure result when protocol generation or execution setup fails."""

        fallback_root = (
            self.adapter.project_root
            / "outputs"
            / "data_factory_failed"
        ).resolve(strict=False)
        fallback_root.mkdir(parents=True, exist_ok=True)

        result_json_path = fallback_root / "data_factory_result.json"

        artifacts = DataFactoryArtifactPaths(
            work_dir=fallback_root,
            result_json_path=result_json_path,
        )

        result = DataFactoryResult(
            schema_version=DATA_FACTORY_RESULT_SCHEMA_VERSION,
            status=DATA_FACTORY_STATUS_FAILED,
            message="Data factory service failed before a normal result could be produced.",
            artifacts=artifacts,
            speaker_name=request.speaker_name,
            language=request.language,
            timing=DataFactoryTiming(
                started_at=now_iso(),
                finished_at=now_iso(),
            ),
            steps=[],
            error=DataFactoryErrorInfo(
                error_type=type(exc).__name__,
                message=str(exc),
                stage="data_factory_service",
                traceback_text=traceback.format_exc(),
                hint=(
                    "Check DataFactoryRequest fields, especially raw_input_dir, "
                    "speaker_name, language, prompt_mode, UVR/denoise flags, "
                    "and config defaults."
                ),
            ),
            extra={
                "dry_run_only": True,
                "request_preview": request_to_dict(request),
            },
        )

        write_data_factory_result_json(result, result_json_path)
        return result

def create_data_factory_service(
        *,
        adapter: Optional[DataFactoryAdapter] = None,
        project_root: Optional[PathLike] = None,
        python_executable: Optional[PathLike] = None,
        config: Optional[DataFactoryConfig] = None,
        execution_options: Optional[DataFactoryExecutionOptions] = None,
        show_terminal_progress: Optional[bool] = None,
) -> DataFactoryService:
    """Factory helper for future WebUI."""

    return DataFactoryService(
        adapter=adapter,
        project_root=project_root,
        python_executable=python_executable,
        config=config,
        execution_options=execution_options,
        show_terminal_progress=show_terminal_progress,
    )


__all__ = [
    "DataFactoryService",
    "create_data_factory_service",
    "request_to_dict",
    "DataFactoryExecutionOptions",
]