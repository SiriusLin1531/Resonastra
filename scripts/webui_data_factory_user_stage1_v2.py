from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

# ============================================================
# Project bootstrap
# 项目路径引导
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Reuse the stable Stage1/Stage2 UI and replace only Stage1 run
# 复用稳定 UI，仅替换 Stage1 长任务运行链
# ============================================================
from scripts import webui_data_factory_user_stage1 as legacy
from src.voicelab_user.runtime.terminal_progress import run_logged_process


SHOW_TERMINAL_LABEL = "显示终端进度窗口"
STAGE1_TERMINAL_TITLE = "Resonastra Stage1 Dataset Builder"
HARDENED_STAGE1_BUILDER = "scripts/build_stage1_fewshot_dataset_hardened.py"
HARDENED_PRUNE_FLAG = "--prune_orphans"


def _stage1_command_hardened(
    *,
    active_work_dir: Path,
    speaker_name: str,
    device: str,
    use_half: bool,
    overwrite: bool,
    validate_dataset: bool,
    train_ratio: float,
) -> tuple[list[str], dict[str, Path]]:
    """Build the release-hardened Stage1 command used by the user UI.

    The legacy command builder remains the source of all canonical Stage1
    arguments. RH-1B replaces only the executable script with the hardened
    wrapper. When the user explicitly selects "覆盖已有 Stage1 cache", the
    wrapper also receives ``--prune_orphans`` so stale cache files that are no
    longer referenced by the newly written train/val manifests are removed only
    after the canonical builder succeeds.
    """

    cmd, paths = legacy._stage1_command(
        active_work_dir=active_work_dir,
        speaker_name=speaker_name,
        device=device,
        use_half=use_half,
        overwrite=overwrite,
        validate_dataset=validate_dataset,
        train_ratio=train_ratio,
    )

    legacy_script = "scripts/build_stage1_fewshot_dataset.py"
    try:
        script_index = cmd.index(legacy_script)
    except ValueError as exc:
        raise RuntimeError(
            "RH-1B could not locate the canonical Stage1 builder in the UI command."
        ) from exc

    cmd = list(cmd)
    cmd[script_index] = HARDENED_STAGE1_BUILDER
    if bool(overwrite) and HARDENED_PRUNE_FLAG not in cmd:
        cmd.append(HARDENED_PRUNE_FLAG)
    return cmd, paths


def _read_stage1_hardening_report(build_report: Path) -> dict[str, Any]:
    """Read the RH-1B fields persisted by the hardened wrapper."""

    path = Path(build_report).expanduser().resolve(strict=False)
    if not path.is_file():
        return {
            "available": False,
            "build_report": str(path),
            "reason": "build_report_missing",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "build_report": str(path),
            "reason": "build_report_unreadable",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(value, dict):
        return {
            "available": False,
            "build_report": str(path),
            "reason": "build_report_not_object",
        }
    return {
        "available": True,
        "build_report": str(path),
        "release_hardening": value.get("release_hardening"),
        "cache_cleanup": value.get("cache_cleanup"),
        "cache_contract": value.get("cache_contract"),
        "report_status": value.get("status"),
    }


def _inspect_existing_stage1_artifacts(
    output_root: str | Path,
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    """Detect meaningful existing Stage1 output without treating empty dirs as data.

    The overwrite checkbox is an explicit user authorization to rebuild an
    existing Stage1 dataset. Any file already present under ``07_stage1_ft`` is
    therefore treated as an existing Stage1 artifact, including partial output
    from a failed or interrupted build. Empty directories are intentionally not
    considered existing data so a fresh first build remains allowed.
    """

    root = Path(output_root).expanduser().resolve(strict=False)
    limit = max(1, int(sample_limit))

    if root.is_file():
        return {
            "exists": True,
            "output_root": str(root),
            "examples": ["<output_root_is_file>"],
            "sample_limit": limit,
        }
    if not root.is_dir():
        return {
            "exists": False,
            "output_root": str(root),
            "examples": [],
            "sample_limit": limit,
        }

    examples: list[str] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            label = candidate.relative_to(root).as_posix()
        except ValueError:
            label = str(candidate)
        examples.append(label)
        if len(examples) >= limit:
            break

    return {
        "exists": bool(examples),
        "output_root": str(root),
        "examples": examples,
        "sample_limit": limit,
    }


def run_stage1_dataset_unified_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    stage1_device: str,
    stage1_use_half: bool,
    stage1_overwrite: bool,
    stage1_validate_dataset: bool,
    stage1_train_ratio: Any,
    timeout_seconds: Any,
    show_terminal_progress: bool,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    """Build Stage1 data with live merged logging and an optional terminal.

    RH-1B routes the real user-facing Stage1 build through the hardened wrapper.
    The optional PowerShell window only tails the merged UTF-8 log. Closing that
    window does not stop the builder; the existing "停止当前任务" action still
    owns task termination.
    """

    try:
        active_work_dir = legacy._resolve_active_work_dir(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        active_work_dir.mkdir(parents=True, exist_ok=True)
        normalized_train_ratio = legacy._normalize_train_ratio(
            stage1_train_ratio,
            stage_name="Stage1",
        )

        cmd, paths = _stage1_command_hardened(
            active_work_dir=active_work_dir,
            speaker_name=speaker_name,
            device=stage1_device,
            use_half=stage1_use_half,
            overwrite=stage1_overwrite,
            validate_dataset=stage1_validate_dataset,
            train_ratio=normalized_train_ratio,
        )

        if not paths["input_manifest"].is_file():
            raise FileNotFoundError(
                f"Stage1 input manifest not found: {paths['input_manifest']}. "
                "请先执行数据工厂，并完成或跳过人工校对。"
            )

        existing_stage1 = _inspect_existing_stage1_artifacts(paths["output_root"])
        if bool(existing_stage1.get("exists")) and not bool(stage1_overwrite):
            status_payload, panel_md, output_md, step_md = legacy._scan_unified(
                active_work_dir
            )
            stage1_status = legacy.scan_stage1_status(active_work_dir)
            message = (
                "检测到当前项目已经存在 Stage1 训练数据或缓存。"
                "为避免意外覆盖，本次没有启动 Stage1 生成任务。"
            )
            dev_json: dict[str, Any] = {
                "status": "blocked",
                "message": message,
                "operation": "build_stage1_fewshot_dataset",
                "reason": "stage1_output_exists_overwrite_not_confirmed",
                "process_started": False,
                "overwrite_requested": False,
                "existing_stage1_artifacts": existing_stage1,
                "stage1_status": stage1_status,
                "unified_status": status_payload,
                "release_hardening_request": {
                    "rh_1b": True,
                    "builder": HARDENED_STAGE1_BUILDER,
                    "overwrite_requested": False,
                    "prune_orphans": False,
                    "strict_cache_contract": True,
                    "overwrite_preflight_guard": True,
                },
            }
            status_lines = [
                "## 当前状态：未执行",
                "",
                f"**消息：** {message}",
                "",
                "### 下一步",
                "如需根据当前数据重新生成或更新 Stage1 数据，请勾选"
                "“覆盖已有 Stage1 cache”后再次点击“生成 Stage1 训练数据”。",
                "",
                "**Stage1 builder：** 未启动。",
            ]
            return (
                "\n".join(status_lines),
                dev_json,
                output_md,
                step_md,
                str(active_work_dir),
                panel_md,
            )

        paths["stdout_log"].parent.mkdir(parents=True, exist_ok=True)
        terminal_title = (
            f"{STAGE1_TERMINAL_TITLE} - "
            f"{str(speaker_name).strip() or 'speaker'}"
        )
        command_payload: dict[str, Any] = {
            "schema_version": "voicelab_stage1_build_command_v4",
            "command": cmd,
            "cwd": str(PROJECT_ROOT),
            "paths": {
                key: str(value)
                for key, value in paths.items()
            },
            "split_request": {
                "train_ratio": normalized_train_ratio,
                "val_ratio": round(1.0 - normalized_train_ratio, 12),
                "seed_mode": "builder_generated_random",
                "max_items": None,
                "process_all_items": True,
            },
            "execution_request": {
                "runner": "run_logged_process",
                "log_mode": "merged_stdout_stderr",
                "show_terminal_progress": bool(show_terminal_progress),
                "terminal_title": terminal_title,
                "timeout_seconds": legacy.base._as_timeout(timeout_seconds),
            },
            "release_hardening": {
                "rh_1b": True,
                "builder": HARDENED_STAGE1_BUILDER,
                "overwrite_requested": bool(stage1_overwrite),
                "prune_orphans": bool(stage1_overwrite),
                "strict_cache_contract": True,
                "overwrite_preflight_guard": True,
            },
            "created_at": legacy.base.now_iso(),
        }
        legacy.base._write_json(paths["command_json"], command_payload)

        # Preserve the historical stderr path as a compatibility pointer while
        # the actual Stage1 output is streamed into one merged log.
        paths["stderr_log"].write_text(
            "Stage1 stderr is merged into the live log below:\n"
            f"{paths['stdout_log']}\n",
            encoding="utf-8",
            errors="replace",
        )

        process_result = run_logged_process(
            cmd,
            cwd=PROJECT_ROOT,
            log_path=paths["stdout_log"],
            timeout_seconds=legacy.base._as_timeout(timeout_seconds),
            show_terminal_progress=bool(show_terminal_progress),
            terminal_title=terminal_title,
        )

        command_payload["execution_result"] = {
            "returncode": int(process_result.returncode),
            "elapsed_seconds": float(process_result.elapsed_seconds),
            "timed_out": bool(process_result.timed_out),
            "merged_log": str(process_result.log_path),
            "terminal_requested": bool(process_result.terminal_requested),
            "terminal_started": bool(process_result.terminal_started),
            "terminal_error": process_result.terminal_error,
            "finished_at": legacy.base.now_iso(),
        }
        hardening_report = _read_stage1_hardening_report(paths["build_report"])
        command_payload["release_hardening_result"] = hardening_report
        legacy.base._write_json(paths["command_json"], command_payload)

        status_payload, panel_md, output_md, step_md = legacy._scan_unified(
            active_work_dir
        )
        stage1_status = legacy.scan_stage1_status(active_work_dir)
        succeeded = (
            process_result.returncode == 0
            and not process_result.timed_out
            and bool(stage1_status.get("done"))
            and (
                not hardening_report.get("available")
                or (hardening_report.get("release_hardening") or {}).get("rh_1b") == "OK"
            )
        )

        if process_result.timed_out:
            status_name = "超时"
            message = "Stage1 训练数据生成超时，builder 进程已终止。"
        elif succeeded:
            status_name = "成功"
            message = "Stage1 训练数据生成完成，Stage1 cache 一致性校验通过。"
        else:
            status_name = "失败"
            message = "Stage1 训练数据生成或 cache 一致性校验失败。"

        dev_json: dict[str, Any] = {
            "status": (
                "timed_out"
                if process_result.timed_out
                else "succeeded"
                if succeeded
                else "failed"
            ),
            "operation": "build_stage1_fewshot_dataset",
            "runner": "run_logged_process",
            "returncode": int(process_result.returncode),
            "total_seconds": float(process_result.elapsed_seconds),
            "timed_out": bool(process_result.timed_out),
            "command": cmd,
            "command_json": str(paths["command_json"]),
            "split_request": command_payload["split_request"],
            "execution_request": command_payload["execution_request"],
            "execution_result": command_payload["execution_result"],
            "release_hardening_request": command_payload["release_hardening"],
            "release_hardening_result": hardening_report,
            "merged_log": str(paths["stdout_log"]),
            "stderr_compatibility_note": str(paths["stderr_log"]),
            "stage1_status": stage1_status,
            "unified_status": status_payload,
        }

        if process_result.terminal_requested:
            if process_result.terminal_started:
                terminal_status = "已打开 PowerShell 实时日志窗口。"
            elif process_result.terminal_error:
                terminal_status = (
                    "终端窗口未能打开，但 Stage1 主任务已继续执行。"
                )
            else:
                terminal_status = "终端窗口未启动。"
        else:
            terminal_status = "已关闭终端进度窗口，仅写入实时日志文件。"

        overwrite_status = (
            "已启用：允许重建已有 Stage1，并在成功构建后清理新 manifest 未引用的旧 cache。"
            if stage1_overwrite
            else "未启用：仅允许在 Stage1 输出尚不存在时首次生成。"
        )
        status_lines = [
            f"## 当前状态：{status_name}",
            "",
            f"**消息：** {message}",
            f"**工作目录：** `{active_work_dir}`",
            f"**Stage1 训练集比例：** `{normalized_train_ratio:.3f}`",
            f"**Stage1 验证集比例：** `{1.0 - normalized_train_ratio:.3f}`",
            "**Stage1 split seed：** 由 builder 自动随机生成并记录到 `metadata/split.json`。",
            "**处理样本范围：** 全部输入样本。",
            f"**覆盖已有 Stage1 cache：** {overwrite_status}",
            "**RH-1B strict cache contract：** 已启用。",
            f"**终端进度：** {terminal_status}",
            f"**Stage1 输出目录：** `{paths['output_root']}`",
            f"**构建报告：** `{paths['build_report']}`",
            f"**实时合并日志：** `{paths['stdout_log']}`",
            f"**stderr 兼容说明：** `{paths['stderr_log']}`",
            f"**耗时：** {process_result.elapsed_seconds:.2f} 秒",
        ]

        if process_result.terminal_error:
            status_lines.extend(
                [
                    "",
                    "### 终端窗口提示",
                    f"`{process_result.terminal_error}`",
                    "该问题不影响 builder 的执行结果，请直接查看实时合并日志。",
                ]
            )

        if not succeeded:
            status_lines.extend(
                [
                    "",
                    "### 失败信息",
                    f"**returncode：** `{process_result.returncode}`",
                    f"**timed_out：** `{process_result.timed_out}`",
                    "请查看实时合并日志、build_report.json 与诊断信息中的 RH-1B cache contract。",
                ]
            )
            if not stage1_overwrite:
                status_lines.append(
                    "若本次失败后已留下 Stage1 产物，再次运行前请勾选“覆盖已有 Stage1 cache”。"
                )

        return (
            "\n".join(status_lines),
            dev_json,
            output_md,
            step_md,
            str(active_work_dir),
            panel_md,
        )

    except Exception as exc:
        status_md, dev_json, output_md, step_md = legacy.base._format_exception(
            exc,
            title="Stage1 训练数据生成失败",
        )
        return (
            status_md,
            dev_json,
            output_md,
            step_md,
            work_dir_output or work_dir or "",
            "暂无统一状态。",
        )


def _create_demo_with_stage1_terminal_binding():
    """Reuse the stable UI while extending only the Stage1 button inputs.

    The legacy UI already owns the shared "显示终端进度窗口" component, but
    its Stage1 button was created before Stage1 terminal support existed. During
    UI construction we capture that existing checkbox and append it only to the
    Stage1 click binding. All other click bindings are delegated unchanged.
    """

    captured: dict[str, Any] = {}
    original_checkbox = legacy.gr.Checkbox
    original_button_click = legacy.gr.Button.click
    original_stage1_callback = legacy.run_stage1_dataset_unified_click

    def checkbox_factory(*args: Any, **kwargs: Any):
        component = original_checkbox(*args, **kwargs)
        label = kwargs.get("label")
        if label is None and args:
            label = args[0]
        if label == SHOW_TERMINAL_LABEL:
            captured["show_terminal_progress"] = component
        return component

    def patched_button_click(
        button: Any,
        *args: Any,
        **kwargs: Any,
    ):
        fn: Callable[..., Any] | None
        if "fn" in kwargs:
            fn = kwargs.get("fn")
        elif args:
            fn = args[0]
        else:
            fn = None

        if fn is run_stage1_dataset_unified_click:
            terminal_component = captured.get("show_terminal_progress")
            if terminal_component is None:
                raise RuntimeError(
                    "Stage1 terminal binding could not find the existing "
                    f"'{SHOW_TERMINAL_LABEL}' checkbox."
                )

            if "inputs" in kwargs:
                inputs = list(kwargs.get("inputs") or [])
                kwargs["inputs"] = [*inputs, terminal_component]
            else:
                mutable_args = list(args)
                while len(mutable_args) < 2:
                    mutable_args.append(None)
                inputs = list(mutable_args[1] or [])
                mutable_args[1] = [*inputs, terminal_component]
                args = tuple(mutable_args)

        return original_button_click(button, *args, **kwargs)

    legacy.run_stage1_dataset_unified_click = run_stage1_dataset_unified_click
    legacy.gr.Checkbox = checkbox_factory
    legacy.gr.Button.click = patched_button_click
    try:
        return legacy.create_demo()
    finally:
        legacy.gr.Button.click = original_button_click
        legacy.gr.Checkbox = original_checkbox
        legacy.run_stage1_dataset_unified_click = original_stage1_callback


def create_demo():
    return _create_demo_with_stage1_terminal_binding()


def parse_args() -> argparse.Namespace:
    return legacy.parse_args()


def main() -> None:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        selected_port = legacy.base._find_available_port(
            args.host,
            port,
            max_tries=int(args.auto_port_tries),
        )
        if selected_port != port:
            print(
                f"[INFO] Requested port {port} is occupied. "
                f"Using available port {selected_port} instead."
            )
        port = selected_port

    demo = create_demo()
    try:
        demo.queue(default_concurrency_limit=8)
    except TypeError:
        demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=bool(args.inbrowser),
    )


if __name__ == "__main__":
    main()
