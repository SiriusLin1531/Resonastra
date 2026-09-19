from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Project bootstrap
# 项目路径引导
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Third-party imports
# 第三方导入
# ============================================================
import gradio as gr

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.voicelab_user.training import (
    build_training_input_contract,
    checkpoint_choices,
    format_checkpoint_status_markdown,
    generate_user_profile,
    mark_best_checkpoint,
    scan_training_outputs,
    validate_stage1_ready,
    validate_stage2_ready,
    write_training_interface_json,
)
from src.voicelab_user.training.config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    load_training_default_config,
)
from src.voicelab_user.training.service import (
    DEFAULT_STAGE2_BASE_CKPT_RELATIVE,
    TRAINING_RUN_DIRNAME,
    TRAINING_RUN_STATE_FILENAME,
    create_training_service,
)


# ============================================================
# Generic helpers
# 通用工具
# ============================================================
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _blank_to_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_json_dict(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, int(port))) != 0


def _find_available_port(host: str, start_port: int, *, max_tries: int = 20) -> int:
    for port in range(int(start_port), int(start_port) + int(max_tries)):
        if _is_port_available(host, port):
            return port
    raise OSError(f"No empty port found in range {start_port}-{int(start_port) + int(max_tries) - 1}.")


def _resolve_work_dir(work_dir: str, speaker_name: str) -> Path:
    text = _blank_to_none(work_dir)
    if text is not None:
        return Path(text).expanduser().resolve(strict=False)

    speaker = str(speaker_name or "").strip()
    if not speaker:
        raise ValueError("请先填写说话人 / 角色名，或填写工作目录。")
    return (PROJECT_ROOT / "user_data" / f"{speaker}_factory").resolve(strict=False)


def _resolve_active_work_dir(resolved_work_dir: str, work_dir: str, speaker_name: str) -> Path:
    return _resolve_work_dir(resolved_work_dir or work_dir, speaker_name)


def _safe_contract(work_dir: str | Path, *, refresh_status: bool = True) -> dict[str, Any]:
    return build_training_input_contract(
        work_dir,
        refresh_status=bool(refresh_status),
    )


def _contract_paths(contract: dict[str, Any]) -> dict[str, Any]:
    stage1 = contract.get("stage1") if isinstance(contract.get("stage1"), dict) else {}
    stage2 = contract.get("stage2") if isinstance(contract.get("stage2"), dict) else {}
    stage1_paths = stage1.get("paths") if isinstance(stage1.get("paths"), dict) else {}
    pt_paths = stage2.get("pt_paths") if isinstance(stage2.get("pt_paths"), dict) else {}
    split_paths = stage2.get("split_paths") if isinstance(stage2.get("split_paths"), dict) else {}
    filter_paths = stage2.get("filter_paths") if isinstance(stage2.get("filter_paths"), dict) else {}

    return {
        "stage1_output_root": stage1_paths.get("output_root"),
        "stage1_train_manifest": stage1_paths.get("train_manifest"),
        "stage1_val_manifest": stage1_paths.get("val_manifest"),
        "stage1_frontend_root": stage1_paths.get("frontend_root"),
        "stage1_semantic_root": stage1_paths.get("semantic_root"),
        "stage2_pt_root": pt_paths.get("stage2_pt_root"),
        "stage2_train_pt_root": split_paths.get("train_pt_root"),
        "stage2_val_pt_root": split_paths.get("val_pt_root"),
        "stage2_split_report": split_paths.get("split_report"),
        "stage2_filter_report": filter_paths.get("filter_report_json"),
    }


def _tail_file(path: str | Path, *, max_lines: int = 80) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"[log read failed] {type(exc).__name__}: {exc}"
    return "\n".join(lines[-int(max_lines):])


def _is_pid_running(pid: Any) -> Optional[bool]:
    if pid is None:
        return None
    try:
        pid_int = int(pid)
    except Exception:
        return None

    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid_int}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            return None
        out = completed.stdout.strip().lower()
        return bool(out and "no tasks" not in out and str(pid_int) in out)

    try:
        os.kill(pid_int, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return None


def _kill_pid(pid: Any) -> dict[str, Any]:
    try:
        pid_int = int(pid)
    except Exception as exc:
        return {"succeeded": False, "error": f"Invalid PID: {pid!r}; {exc}"}

    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid_int), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "pid": pid_int,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "succeeded": completed.returncode == 0,
        }

    try:
        os.kill(pid_int, 15)
        return {"pid": pid_int, "succeeded": True, "signal": 15}
    except Exception as exc:
        return {"pid": pid_int, "succeeded": False, "error": f"{type(exc).__name__}: {exc}"}


# ============================================================
# Parameter visibility / advanced isolation
# 参数展示与高级模式隔离
# ============================================================
def _active_defaults_payload() -> dict[str, Any]:
    service = create_training_service()
    config_path = (PROJECT_ROOT / DEFAULT_TRAINING_CONFIG_PATH).resolve(strict=False)
    raw_config = load_training_default_config(config_path)
    return {
        "schema_version": "voicelab_training_ui_defaults_view_v1",
        "created_at": _now_iso(),
        "config_path": str(config_path),
        "config_exists": config_path.is_file(),
        "user_mode_policy": {
            "editable_in_user_ui": False,
            "base_ckpt_editable": False,
            "stage2_base_ckpt_relative": DEFAULT_STAGE2_BASE_CKPT_RELATIVE.as_posix(),
            "reason": "用户版使用已验证默认参数；完整参数仅在开发者详情中查看。",
        },
        "active_defaults": {
            "stage1": asdict(service.stage1_defaults),
            "stage2": asdict(service.stage2_defaults),
        },
        "raw_config": raw_config,
    }


def _format_parameter_summary(defaults_payload: dict[str, Any] | None = None) -> str:
    payload = defaults_payload or _active_defaults_payload()
    defaults = payload.get("active_defaults") if isinstance(payload.get("active_defaults"), dict) else {}
    stage1 = defaults.get("stage1") if isinstance(defaults.get("stage1"), dict) else {}
    stage2 = defaults.get("stage2") if isinstance(defaults.get("stage2"), dict) else {}

    return "\n".join(
        [
            "### 训练参数摘要（用户模式）",
            "",
            "用户版 Training UI 使用项目内置默认参数，不提供超参数编辑入口。",
            "",
            "| 阶段 | 参数摘要 |",
            "|---|---|",
            "| Stage1 | "
            f"epochs=`{stage1.get('epochs')}`，batch_size=`{stage1.get('batch_size')}`，"
            f"lr=`{stage1.get('lr')}`，use_amp=`{stage1.get('use_amp')}`，"
            f"trainable_scope=`{stage1.get('trainable_scope')}`，last_n_layers=`{stage1.get('last_n_layers')}` |",
            "| Stage2 | "
            f"epochs=`{stage2.get('epochs')}`，batch_size=`{stage2.get('batch_size')}`，"
            f"lr=`{stage2.get('lr')}`，use_amp=`{stage2.get('use_amp')}`，"
            f"trainable_scope=`{stage2.get('trainable_scope')}` |",
            "| Stage2 style / LoRA | "
            f"continuous=`{stage2.get('require_continuous_semantic')}`，"
            f"v66_style=`{stage2.get('require_v66_style_fields')}`，"
            f"bootstrapper_lora=`{stage2.get('use_bootstrapper_lora')}`，"
            f"attention_lora=`{stage2.get('use_bootstrapper_attention_lora')}`，"
            f"energy_loss=`{stage2.get('v66_energy_loss_weight')}`，"
            f"f0_loss=`{stage2.get('v66_f0_loss_weight')}`，"
            f"speaker_loss=`{stage2.get('v66_speaker_loss_weight')}` |",
            "| Stage2 base_ckpt | "
            f"`{DEFAULT_STAGE2_BASE_CKPT_RELATIVE.as_posix()}`，项目相对路径，用户版不可修改 |",
            "",
            f"配置来源：`{payload.get('config_path')}`",
        ]
    )


# ============================================================
# Formatting helpers
# 格式化工具
# ============================================================
def _format_training_ready_panel(contract: dict[str, Any]) -> str:
    readiness = contract.get("readiness") if isinstance(contract.get("readiness"), dict) else {}
    source = contract.get("source_status") if isinstance(contract.get("source_status"), dict) else {}
    next_actions = source.get("next_actions") if isinstance(source.get("next_actions"), list) else []
    paths = _contract_paths(contract)

    stage1_ready = bool(readiness.get("stage1_ready"))
    stage2_ready = bool(readiness.get("stage2_ready"))
    all_ready = bool(readiness.get("all_training_data_ready"))

    lines = [
        "### Training UI 数据状态",
        "",
        f"**工作目录：** `{contract.get('work_dir')}`",
        f"**training_interface：** `{contract.get('training_interface_path')}`",
        f"**DataFactory 状态文件：** `{contract.get('data_factory_status_path')}`",
        "",
        "| 训练阶段 | 状态 | 关键输入 |",
        "|---|---:|---|",
        f"| Stage1 Training | {'✅' if stage1_ready else '⬜'} | train_manifest：`{paths.get('stage1_train_manifest')}` |",
        f"| Stage2 Training | {'✅' if stage2_ready else '⬜'} | train_pt_root：`{paths.get('stage2_train_pt_root')}`，val_pt_root：`{paths.get('stage2_val_pt_root')}` |",
        f"| 完整训练数据 | {'✅' if all_ready else '⬜'} | Stage1 + Stage2 |",
    ]

    if next_actions:
        lines.extend(["", "### DataFactory 下一步建议"])
        for item in next_actions:
            lines.append(f"- {item}")

    return "\n".join(lines)


def _format_validation_summary(contract: dict[str, Any]) -> str:
    stage1_ok, stage1_problems = validate_stage1_ready(contract)
    stage2_ok, stage2_problems = validate_stage2_ready(contract)

    lines = [
        "### 训练入口校验",
        "",
        f"- Stage1：{'可启动' if stage1_ok else '不可启动'}",
        f"- Stage2：{'可启动' if stage2_ok else '不可启动'}",
    ]

    if stage1_problems:
        lines.extend(["", "#### Stage1 问题"])
        lines.extend([f"- {item}" for item in stage1_problems])

    if stage2_problems:
        lines.extend(["", "#### Stage2 问题"])
        lines.extend([f"- {item}" for item in stage2_problems])

    return "\n".join(lines)


def _stage_run_state_path(work_dir: str | Path, stage: str) -> Path:
    return Path(work_dir).expanduser().resolve(strict=False) / TRAINING_RUN_DIRNAME / stage / TRAINING_RUN_STATE_FILENAME


def _read_stage_run_state(work_dir: str | Path, stage: str) -> dict[str, Any]:
    path = _stage_run_state_path(work_dir, stage)
    payload = _read_json_dict(path)
    if payload:
        payload["run_state_path"] = str(path)
    return payload


def _format_run_status_panel(work_dir: str | Path) -> str:
    lines = [
        "### 训练运行状态",
        "",
        f"**工作目录：** `{work_dir}`",
        "",
        "| Stage | 状态 | PID | 进程 | 输出目录 |",
        "|---|---:|---:|---:|---|",
    ]

    for stage in ["stage1", "stage2"]:
        state = _read_stage_run_state(work_dir, stage)
        if not state:
            lines.append(f"| {stage} | ⬜ 未启动 |  |  |  |")
            continue
        pid = state.get("pid")
        running = _is_pid_running(pid)
        running_icon = "✅ 运行中" if running else "⬜ 未运行" if running is False else "未知"
        status = state.get("status") or "unknown"
        output_dir = state.get("output_dir") or ""
        lines.append(f"| {stage} | `{status}` | `{pid}` | {running_icon} | `{output_dir}` |")

    return "\n".join(lines)


def _format_log_preview(work_dir: str | Path, *, max_lines: int = 80) -> str:
    chunks = []
    for stage in ["stage1", "stage2"]:
        state = _read_stage_run_state(work_dir, stage)
        if not state:
            chunks.append(f"### {stage} 日志\n\n尚未启动。")
            continue
        stdout_path = state.get("stdout_log_path") or ""
        stderr_path = state.get("stderr_log_path") or ""
        stdout_tail = _tail_file(stdout_path, max_lines=max_lines)
        stderr_tail = _tail_file(stderr_path, max_lines=max_lines)
        chunks.append(
            "\n".join(
                [
                    f"### {stage} 日志",
                    "",
                    f"stdout：`{stdout_path}`",
                    "",
                    "```text",
                    stdout_tail or "暂无 stdout 输出。",
                    "```",
                    "",
                    f"stderr：`{stderr_path}`",
                    "",
                    "```text",
                    stderr_tail or "暂无 stderr 输出。",
                    "```",
                ]
            )
        )
    return "\n\n".join(chunks)


# ============================================================
# Checkpoint helpers
# Checkpoint 辅助函数
# ============================================================
def _empty_checkpoint_updates() -> tuple[str, Any, Any, dict[str, Any], dict[str, Any]]:
    return (
        "暂无 checkpoint 状态。",
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        {},
        {},
    )


def _scan_checkpoint_panels(work_dir: str | Path) -> tuple[str, Any, Any, dict[str, Any], dict[str, Any]]:
    scan_payload = scan_training_outputs(work_dir)
    checkpoint_md = format_checkpoint_status_markdown(scan_payload)
    stage1_choices = checkpoint_choices(scan_payload, "stage1")
    stage2_choices = checkpoint_choices(scan_payload, "stage2")
    return (
        checkpoint_md,
        gr.update(choices=stage1_choices, value=stage1_choices[0] if stage1_choices else None),
        gr.update(choices=stage2_choices, value=stage2_choices[0] if stage2_choices else None),
        scan_payload,
        {},
    )


def _scan_contract_and_panels(
    work_dir: str | Path,
    *,
    refresh_status: bool,
) -> tuple[dict[str, Any], str, str, str, str, str, dict[str, Any], str, Any, Any, dict[str, Any], dict[str, Any]]:
    contract = _safe_contract(work_dir, refresh_status=refresh_status)
    written_path = write_training_interface_json(work_dir, refresh_status=refresh_status)
    contract["training_interface_path"] = str(written_path)
    defaults_payload = _active_defaults_payload()
    panel_md = _format_training_ready_panel(contract)
    validation_md = _format_validation_summary(contract)
    parameter_md = _format_parameter_summary(defaults_payload)
    run_status_md = _format_run_status_panel(work_dir)
    log_md = _format_log_preview(work_dir)
    checkpoint_md, stage1_update, stage2_update, checkpoint_json, profile_json = _scan_checkpoint_panels(work_dir)
    return (
        contract,
        panel_md,
        validation_md,
        parameter_md,
        run_status_md,
        log_md,
        defaults_payload,
        checkpoint_md,
        stage1_update,
        stage2_update,
        checkpoint_json,
        profile_json,
    )


def _failure_outputs(
    *,
    status_md: str,
    error: Exception,
    resolved_work_dir: str,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    return (
        status_md,
        {"status": "failed", "error_type": type(error).__name__, "message": str(error)},
        "暂无训练数据状态。",
        "暂无校验结果。",
        "暂无参数摘要。",
        "暂无训练运行状态。",
        "暂无日志。",
        "暂无 checkpoint 状态。",
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        {},
        {},
        resolved_work_dir,
        {},
    )


# ============================================================
# Click handlers
# 按钮事件
# ============================================================
def scan_training_data_click(
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        resolved = _resolve_work_dir(work_dir, speaker_name)
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(resolved, refresh_status=refresh_status)

        status_md = "\n".join(
            [
                "## 当前状态：训练输入已扫描",
                "",
                f"**工作目录：** `{resolved}`",
                f"**Training Interface：** `{contract.get('training_interface_path')}`",
                "",
                "用户版采用自动默认训练参数；完整超参数在“开发者详情”中查看。",
            ]
        )
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(resolved),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(
            [
                "## 当前状态：扫描失败",
                "",
                f"**错误类型：** `{type(exc).__name__}`",
                f"**错误消息：** {exc}",
            ]
        )
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=work_dir or "")


def launch_stage1_training_click(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        service = create_training_service()
        state = service.launch_stage1_training(active_work_dir, refresh_status=refresh_status)
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=False)
        contract["latest_stage1_run_state"] = state
        status_md = "\n".join(
            [
                "## Stage1 Training 已启动",
                "",
                f"**PID：** `{state.get('pid')}`",
                f"**run_state：** `{state.get('run_state_path')}`",
                f"**output_dir：** `{state.get('output_dir')}`",
                f"**stdout：** `{state.get('stdout_log_path')}`",
                f"**stderr：** `{state.get('stderr_log_path')}`",
            ]
        )
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## Stage1 Training 启动失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


def launch_stage2_training_click(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        service = create_training_service()
        state = service.launch_stage2_training(active_work_dir, refresh_status=refresh_status)
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=False)
        contract["latest_stage2_run_state"] = state
        status_md = "\n".join(
            [
                "## Stage2 Training 已启动",
                "",
                f"**PID：** `{state.get('pid')}`",
                f"**run_state：** `{state.get('run_state_path')}`",
                f"**output_dir：** `{state.get('output_dir')}`",
                f"**stdout：** `{state.get('stdout_log_path')}`",
                f"**stderr：** `{state.get('stderr_log_path')}`",
                "",
                f"**固定 base_ckpt：** `{DEFAULT_STAGE2_BASE_CKPT_RELATIVE.as_posix()}`",
            ]
        )
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## Stage2 Training 启动失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


def refresh_training_run_status_click(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=refresh_status)
        status_md = "\n".join(["## 当前状态：训练运行状态已刷新", "", f"**工作目录：** `{active_work_dir}`"])
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## 训练运行状态刷新失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


def stop_training_click(
    stage: str,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        state_path = _stage_run_state_path(active_work_dir, stage)
        state = _read_json_dict(state_path)
        if not state:
            raise FileNotFoundError(f"未找到 {stage} training run_state: {state_path}")

        pid = state.get("pid")
        kill_result = _kill_pid(pid)
        state["status"] = "stopped" if kill_result.get("succeeded") else "stop_failed"
        state["message"] = f"{stage} training stop requested."
        state["stop_result"] = kill_result
        state["updated_at"] = _now_iso()
        _write_json(state_path, state)

        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=refresh_status)
        contract[f"latest_{stage}_stop_result"] = kill_result
        status_md = "\n".join(
            [
                f"## {stage} Training 停止请求已执行",
                "",
                f"**PID：** `{pid}`",
                f"**成功：** `{kill_result.get('succeeded')}`",
                f"**run_state：** `{state_path}`",
            ]
        )
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## 停止训练失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


def refresh_checkpoints_click(
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=refresh_status)
        status_md = "\n".join(["## Checkpoint 状态已刷新", "", f"**工作目录：** `{active_work_dir}`"])
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## Checkpoint 刷新失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


def mark_best_checkpoint_click(
    stage: str,
    checkpoint_value: str,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        meta = mark_best_checkpoint(
            active_work_dir,
            stage=stage,
            checkpoint_value=checkpoint_value,
            copy_mode=True,
        )
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=refresh_status)
        checkpoint_json["latest_mark_best"] = meta
        status_md = "\n".join(
            [
                f"## {stage} best checkpoint 已标记",
                "",
                f"**source：** `{meta.get('source_checkpoint')}`",
                f"**best：** `{meta.get('best_checkpoint')}`",
            ]
        )
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile_json,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## 标记 best checkpoint 失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


def generate_profile_click(
    profile_name: str,
    include_stage1: bool,
    include_stage2: bool,
    overwrite_profile: bool,
    resolved_work_dir: str,
    work_dir: str,
    speaker_name: str,
    refresh_status: bool,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str, Any, Any, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    try:
        active_work_dir = _resolve_active_work_dir(resolved_work_dir, work_dir, speaker_name)
        profile = generate_user_profile(
            active_work_dir,
            profile_name=profile_name or speaker_name or "default_user",
            include_stage1=include_stage1,
            include_stage2=include_stage2,
            overwrite=overwrite_profile,
        )
        (
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            defaults_payload,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            _profile_json,
        ) = _scan_contract_and_panels(active_work_dir, refresh_status=refresh_status)
        status_md = "\n".join(
            [
                "## User Profile 已生成",
                "",
                f"**profile_name：** `{profile.get('profile_name')}`",
                f"**profile_root：** `{profile.get('profile_root')}`",
                f"**manifest：** `{profile.get('profile_manifest_path')}`",
                f"**ready_for_inference：** `{profile.get('ready_for_inference')}`",
            ]
        )
        return (
            status_md,
            contract,
            panel_md,
            validation_md,
            parameter_md,
            run_status_md,
            log_md,
            checkpoint_md,
            stage1_update,
            stage2_update,
            checkpoint_json,
            profile,
            str(active_work_dir),
            defaults_payload,
        )
    except Exception as exc:
        status_md = "\n".join(["## User Profile 生成失败", "", f"**错误类型：** `{type(exc).__name__}`", f"**错误消息：** {exc}"])
        return _failure_outputs(status_md=status_md, error=exc, resolved_work_dir=resolved_work_dir or work_dir or "")


# ============================================================
# UI
# UI 定义
# ============================================================
def create_demo() -> gr.Blocks:
    with gr.Blocks(title="VoiceLab Training UI - 用户版") as demo:
        gr.Markdown(
            """
# VoiceLab Training UI - 用户版

这个页面负责从 DataFactory 工作目录读取训练输入契约，并启动 Stage1 / Stage2 few-shot 训练。

用户版采用**自动默认训练参数**：用户只需要选择工作目录、扫描数据、启动训练。完整超参数放在开发者详情中查看，不在用户模式中编辑。

Stage2 base_ckpt 固定为项目相对路径：`user_profiles/default_zh/stage2/best_model.pt`，用户版 UI 不提供修改入口。
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. 输入")
                speaker_name = gr.Textbox(
                    label="说话人 / 角色名",
                    value="",
                    info="当工作目录留空时，用它推断 user_data/{speaker_name}_factory。",
                )
                work_dir = gr.Textbox(
                    label="DataFactory 工作目录",
                    value="",
                )
                refresh_status = gr.Checkbox(
                    label="扫描时刷新 DataFactory 统一状态",
                    value=True,
                    info="开启后会重新扫描工作目录，并刷新 data_factory_unified_status.json。",
                )

                gr.Markdown("## 2. 训练操作")
                scan_button = gr.Button("扫描训练数据 / 生成 training_interface.json", variant="primary")
                with gr.Row():
                    start_stage1_button = gr.Button("启动 Stage1 Training", variant="primary")
                    start_stage2_button = gr.Button("启动 Stage2 Training", variant="primary")
                with gr.Row():
                    refresh_run_button = gr.Button("刷新训练状态 / 日志", variant="secondary")
                    stop_stage1_button = gr.Button("停止 Stage1 Training", variant="stop")
                    stop_stage2_button = gr.Button("停止 Stage2 Training", variant="stop")

                gr.Markdown("## 3. Checkpoint / Profile 管理")
                refresh_checkpoint_button = gr.Button("刷新 checkpoint 列表", variant="secondary")
                stage1_checkpoint_dropdown = gr.Dropdown(
                    label="Stage1 checkpoint",
                    choices=[],
                    value=None,
                    info="刷新 checkpoint 后选择一个 Stage1 checkpoint 标记为 best。",
                )
                stage2_checkpoint_dropdown = gr.Dropdown(
                    label="Stage2 checkpoint",
                    choices=[],
                    value=None,
                    info="刷新 checkpoint 后选择一个 Stage2 checkpoint 标记为 best。",
                )
                with gr.Row():
                    mark_stage1_best_button = gr.Button("标记 Stage1 best", variant="secondary")
                    mark_stage2_best_button = gr.Button("标记 Stage2 best", variant="secondary")

                with gr.Accordion("生成 User Profile", open=True):
                    profile_name = gr.Textbox(
                        label="profile_name",
                        value="",
                        info="生成到 user_profiles/{profile_name}/，供未来 Inference UI 自动读取。",
                    )
                    with gr.Row():
                        include_stage1_profile = gr.Checkbox(label="包含 Stage1 checkpoint", value=True)
                        include_stage2_profile = gr.Checkbox(label="包含 Stage2 checkpoint", value=True)
                        overwrite_profile = gr.Checkbox(label="覆盖已有 profile", value=True)
                    generate_profile_button = gr.Button("生成 User Profile", variant="primary")

                with gr.Accordion("用户模式说明", open=False):
                    gr.Markdown(
                        """
- 训练参数来自 `configs/training_default.yaml`。
- 用户版不显示也不允许编辑学习率、LoRA、loss weight 等高级参数。
- Stage2 base_ckpt 固定为项目相对路径，避免打包后出现本机盘符路径错误。
- Checkpoint 标记 best 后会生成 `best_model.pt`，User Profile 会复制 best checkpoint 到 `user_profiles/<profile_name>/`。
"""
                    )

            with gr.Column(scale=1):
                gr.Markdown("## 状态")
                status_markdown = gr.Markdown("等待扫描。")

                gr.Markdown("## 训练数据状态")
                training_status_markdown = gr.Markdown("暂无训练数据状态。")

                gr.Markdown("## 训练参数摘要")
                parameter_summary_markdown = gr.Markdown(_format_parameter_summary())

                gr.Markdown("## 校验结果")
                validation_markdown = gr.Markdown("暂无校验结果。")

                gr.Markdown("## 训练运行状态")
                run_status_markdown = gr.Markdown("暂无训练运行状态。")

                gr.Markdown("## Checkpoint / Output 状态")
                checkpoint_status_markdown = gr.Markdown("暂无 checkpoint 状态。")

                gr.Markdown("## 日志预览")
                log_markdown = gr.Markdown("暂无日志。")

                resolved_work_dir = gr.Textbox(label="当前工作目录", value="", interactive=False)

                with gr.Accordion("开发者详情：Training Interface JSON", open=False):
                    contract_json = gr.JSON(label="training_interface.json")

                with gr.Accordion("开发者详情：当前训练默认参数", open=False):
                    defaults_json = gr.JSON(label="active training defaults")

                with gr.Accordion("开发者详情：Checkpoint Scan JSON", open=False):
                    checkpoint_json = gr.JSON(label="checkpoint scan")

                with gr.Accordion("开发者详情：User Profile JSON", open=False):
                    profile_json = gr.JSON(label="profile manifest")

        outputs = [
            status_markdown,
            contract_json,
            training_status_markdown,
            validation_markdown,
            parameter_summary_markdown,
            run_status_markdown,
            log_markdown,
            checkpoint_status_markdown,
            stage1_checkpoint_dropdown,
            stage2_checkpoint_dropdown,
            checkpoint_json,
            profile_json,
            resolved_work_dir,
            defaults_json,
        ]

        scan_button.click(
            fn=scan_training_data_click,
            inputs=[work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        start_stage1_button.click(
            fn=launch_stage1_training_click,
            inputs=[resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        start_stage2_button.click(
            fn=launch_stage2_training_click,
            inputs=[resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        refresh_run_button.click(
            fn=refresh_training_run_status_click,
            inputs=[resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        stop_stage1_button.click(
            fn=stop_training_click,
            inputs=[gr.State("stage1"), resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        stop_stage2_button.click(
            fn=stop_training_click,
            inputs=[gr.State("stage2"), resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        refresh_checkpoint_button.click(
            fn=refresh_checkpoints_click,
            inputs=[resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        mark_stage1_best_button.click(
            fn=mark_best_checkpoint_click,
            inputs=[gr.State("stage1"), stage1_checkpoint_dropdown, resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        mark_stage2_best_button.click(
            fn=mark_best_checkpoint_click,
            inputs=[gr.State("stage2"), stage2_checkpoint_dropdown, resolved_work_dir, work_dir, speaker_name, refresh_status],
            outputs=outputs,
        )
        generate_profile_button.click(
            fn=generate_profile_click,
            inputs=[
                profile_name,
                include_stage1_profile,
                include_stage2_profile,
                overwrite_profile,
                resolved_work_dir,
                work_dir,
                speaker_name,
                refresh_status,
            ],
            outputs=outputs,
        )

    return demo


# ============================================================
# CLI
# 命令行入口
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceLab Training UI - 用户版")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--auto-port", action="store_true")
    parser.add_argument("--auto-port-tries", type=int, default=20)
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        selected_port = _find_available_port(args.host, port, max_tries=int(args.auto_port_tries))
        if selected_port != port:
            print(f"[INFO] Requested port {port} is occupied. Using available port {selected_port} instead.")
        port = selected_port

    demo = create_demo()
    try:
        demo.queue(default_concurrency_limit=4)
    except TypeError:
        demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=bool(args.inbrowser),
    )


if __name__ == "__main__":
    main()
