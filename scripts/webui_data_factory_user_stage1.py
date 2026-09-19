from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
# Reuse existing UI callbacks and unified status manager
# 复用现有 UI 回调与统一状态管理器
# ============================================================
from scripts import webui_data_factory_user as base
from src.voicelab_user.data_factory.status_manager import (
    format_unified_output_summary,
    format_unified_status_markdown,
    format_unified_step_summary,
    scan_stage1_status,
    scan_unified_data_factory_status,
    stage1_paths_from_work_dir,
)


# ============================================================
# Unified status helpers
# 统一状态辅助函数
# ============================================================
def _scan_unified(
    active_work_dir: str | Path,
) -> tuple[dict[str, Any], str, str, str]:
    """Scan once and return JSON + panel + output summary + step summary."""

    status_payload = scan_unified_data_factory_status(
        active_work_dir,
        write_status_json=True,
    )
    panel_md = format_unified_status_markdown(status_payload)
    output_md = format_unified_output_summary(status_payload)
    step_md = format_unified_step_summary(status_payload)
    return status_payload, panel_md, output_md, step_md


def _resolve_active_work_dir(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> Path:
    return base._resolve_active_work_dir(
        work_dir_output=work_dir_output,
        work_dir=work_dir,
        speaker_name=speaker_name,
    )


def _stage1_command_paths(
    active_work_dir: str | Path,
) -> dict[str, Path]:
    """Return Stage1 command paths, extending Stage1 status paths."""

    work_dir = Path(active_work_dir).expanduser().resolve(strict=False)
    paths = stage1_paths_from_work_dir(work_dir)
    paths.update(
        {
            "work_dir": work_dir,
            "input_manifest": work_dir / "06_export" / "dataset.list",
            "wav_root": work_dir / "06_export" / "clips",
            "stdout_log": work_dir / "logs" / "stage1_build_stdout.log",
            "stderr_log": work_dir / "logs" / "stage1_build_stderr.log",
            "command_json": work_dir / "logs" / "stage1_build_command.json",
        }
    )
    return paths


def _normalize_train_ratio(value: Any, *, stage_name: str) -> float:
    ratio = base._as_float(value, 0.9)
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"{stage_name} train_ratio 必须大于 0 且小于 1，当前值为 {ratio}。"
        )
    return float(ratio)


def _stage1_command(
    *,
    active_work_dir: Path,
    speaker_name: str,
    device: str,
    use_half: bool,
    overwrite: bool,
    validate_dataset: bool,
    train_ratio: float,
) -> tuple[list[str], dict[str, Path]]:
    """Build the user-mode Stage1 dataset command.

    The user UI intentionally passes only train_ratio. The builder generates
    and records an effective random split seed, and processes all items because
    max_items is a developer-only CLI compatibility option.
    """

    paths = _stage1_command_paths(active_work_dir)
    cmd = [
        sys.executable,
        "scripts/build_stage1_fewshot_dataset.py",
        "--input_manifest",
        str(paths["input_manifest"]),
        "--output_root",
        str(paths["output_root"]),
        "--wav_root",
        str(paths["wav_root"]),
        "--language",
        "zh",
        "--speaker_id",
        str(speaker_name).strip() or "speaker",
        "--device",
        str(device).strip() or "cuda",
        "--train_ratio",
        str(float(train_ratio)),
        "--target_tail_silence_sec",
        "0.0",
    ]
    if bool(use_half):
        cmd.append("--use_half")
    if bool(overwrite):
        cmd.append("--overwrite")
    if bool(validate_dataset):
        cmd.append("--validate_dataset")
    return cmd, paths


# ============================================================
# Status-wrapped click handlers
# 状态统一后的点击事件
# ============================================================
def inspect_work_dir_unified_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        status_payload, panel_md, output_md, step_md = _scan_unified(
            active_work_dir
        )
        status_md = "\n".join(
            [
                "## 当前状态：已载入工作目录",
                "",
                f"**工作目录：** `{active_work_dir}`",
                "",
                "已使用统一状态管理器刷新 Stage1 / Stage2 数据产物状态。",
            ]
        )
        return (
            status_md,
            status_payload,
            output_md,
            step_md,
            str(active_work_dir),
            panel_md,
        )
    except Exception as exc:
        status_md, dev_json, output_md, step_md = base._format_exception(
            exc,
            title="载入/扫描工作目录失败",
        )
        return (
            status_md,
            dev_json,
            output_md,
            step_md,
            work_dir_output or work_dir or "",
            "暂无统一状态。",
        )


def _wrap_base_five_output_with_unified_status(
    base_result: tuple[str, dict[str, Any], str, str, str],
    *,
    work_dir_fallback: str,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    status_md, dev_json, old_output_md, old_step_md, active_work_dir = (
        base_result
    )
    active_work_dir = active_work_dir or work_dir_fallback
    try:
        status_payload, panel_md, output_md, step_md = _scan_unified(
            active_work_dir
        )
        dev_json = (
            dev_json
            if isinstance(dev_json, dict)
            else {"base_result": dev_json}
        )
        dev_json["unified_status"] = status_payload
        return (
            status_md,
            dev_json,
            output_md,
            step_md,
            active_work_dir,
            panel_md,
        )
    except Exception as exc:
        dev_json = (
            dev_json
            if isinstance(dev_json, dict)
            else {"base_result": dev_json}
        )
        dev_json["unified_status_error"] = f"{type(exc).__name__}: {exc}"
        return (
            status_md,
            dev_json,
            old_output_md,
            old_step_md,
            active_work_dir,
            "暂无统一状态。",
        )


def run_prepare_unified_click(
    *args: Any,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    base_result = base.run_prepare_click(*args)
    work_dir_fallback = str(args[7]) if len(args) > 7 else ""
    return _wrap_base_five_output_with_unified_status(
        base_result,
        work_dir_fallback=work_dir_fallback,
    )


def rebuild_after_proofread_unified_click(
    *args: Any,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    base_result = base.rebuild_after_proofread_click(*args)
    work_dir_fallback = str(args[1]) if len(args) > 1 else ""
    return _wrap_base_five_output_with_unified_status(
        base_result,
        work_dir_fallback=work_dir_fallback,
    )


def skip_proofread_unified_click(
    *args: Any,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    base_result = base.skip_proofread_click(*args)
    work_dir_fallback = str(args[1]) if len(args) > 1 else ""
    return _wrap_base_five_output_with_unified_status(
        base_result,
        work_dir_fallback=work_dir_fallback,
    )


def run_stage2_postprocess_unified_click(
    *args: Any,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    base_result = base.run_postprocess_click(*args)
    work_dir_fallback = str(args[1]) if len(args) > 1 else ""
    return _wrap_base_five_output_with_unified_status(
        base_result,
        work_dir_fallback=work_dir_fallback,
    )


def run_stage2_resume_unified_click(
    *args: Any,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    (
        status_md,
        dev_json,
        old_output_md,
        old_step_md,
        active_work_dir,
        old_panel,
    ) = base.run_resume_postprocess_click(*args)
    active_work_dir = active_work_dir or (
        str(args[1]) if len(args) > 1 else ""
    )
    try:
        status_payload, panel_md, output_md, step_md = _scan_unified(
            active_work_dir
        )
        dev_json = (
            dev_json
            if isinstance(dev_json, dict)
            else {"base_result": dev_json}
        )
        dev_json["unified_status"] = status_payload
        return (
            status_md,
            dev_json,
            output_md,
            step_md,
            active_work_dir,
            panel_md,
        )
    except Exception as exc:
        dev_json = (
            dev_json
            if isinstance(dev_json, dict)
            else {"base_result": dev_json}
        )
        dev_json["unified_status_error"] = f"{type(exc).__name__}: {exc}"
        return (
            status_md,
            dev_json,
            old_output_md,
            old_step_md,
            active_work_dir,
            old_panel,
        )


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
) -> tuple[str, dict[str, Any], str, str, str, str]:
    started = time.time()
    try:
        active_work_dir = _resolve_active_work_dir(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        active_work_dir.mkdir(parents=True, exist_ok=True)
        normalized_train_ratio = _normalize_train_ratio(
            stage1_train_ratio,
            stage_name="Stage1",
        )

        cmd, paths = _stage1_command(
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

        paths["stdout_log"].parent.mkdir(parents=True, exist_ok=True)
        command_payload = {
            "schema_version": "voicelab_stage1_build_command_v2",
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
            "created_at": base.now_iso(),
        }
        base._write_json(paths["command_json"], command_payload)

        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=base._as_timeout(timeout_seconds),
        )
        paths["stdout_log"].write_text(
            completed.stdout or "",
            encoding="utf-8",
            errors="replace",
        )
        paths["stderr_log"].write_text(
            completed.stderr or "",
            encoding="utf-8",
            errors="replace",
        )

        status_payload, panel_md, output_md, step_md = _scan_unified(
            active_work_dir
        )
        stage1_status = scan_stage1_status(active_work_dir)
        succeeded = completed.returncode == 0 and bool(
            stage1_status.get("done")
        )
        elapsed = time.time() - started

        dev_json: dict[str, Any] = {
            "status": "succeeded" if succeeded else "failed",
            "operation": "build_stage1_fewshot_dataset",
            "returncode": int(completed.returncode),
            "total_seconds": round(elapsed, 3),
            "command": cmd,
            "command_json": str(paths["command_json"]),
            "split_request": command_payload["split_request"],
            "stdout_log": str(paths["stdout_log"]),
            "stderr_log": str(paths["stderr_log"]),
            "stage1_status": stage1_status,
            "unified_status": status_payload,
        }

        status_lines = [
            f"## 当前状态：{'成功' if succeeded else '失败'}",
            "",
            (
                "**消息：** Stage1 训练数据生成完成。"
                if succeeded
                else "**消息：** Stage1 训练数据生成失败。"
            ),
            f"**工作目录：** `{active_work_dir}`",
            f"**Stage1 训练集比例：** `{normalized_train_ratio:.3f}`",
            f"**Stage1 验证集比例：** `{1.0 - normalized_train_ratio:.3f}`",
            "**Stage1 split seed：** 由 builder 自动随机生成并记录到 `metadata/split.json`。",
            "**处理样本范围：** 全部输入样本。",
            f"**Stage1 输出目录：** `{paths['output_root']}`",
            f"**构建报告：** `{paths['build_report']}`",
            f"**stdout 日志：** `{paths['stdout_log']}`",
            f"**stderr 日志：** `{paths['stderr_log']}`",
            f"**耗时：** {elapsed:.2f} 秒",
        ]
        if not succeeded:
            status_lines.extend(
                [
                    "",
                    "### 失败信息",
                    f"**returncode：** `{completed.returncode}`",
                    "请查看 stderr 日志和 build_report.json。",
                ]
            )

        return (
            "\n".join(status_lines),
            dev_json,
            output_md,
            step_md,
            str(active_work_dir),
            panel_md,
        )

    except subprocess.TimeoutExpired as exc:
        status_md, dev_json, output_md, step_md = base._format_exception(
            exc,
            title="Stage1 训练数据生成超时",
        )
        return (
            status_md,
            dev_json,
            output_md,
            step_md,
            work_dir_output or work_dir or "",
            "暂无统一状态。",
        )
    except Exception as exc:
        status_md, dev_json, output_md, step_md = base._format_exception(
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


def _find_stage1_processes(
    active_work_dir: str | Path,
) -> list[dict[str, Any]]:
    work_dir = Path(active_work_dir).resolve(strict=False)
    paths = _stage1_command_paths(work_dir)
    patterns = [
        str(work_dir).lower(),
        str(paths["output_root"]).lower(),
        str(paths["input_manifest"]).lower(),
    ]
    matches: list[dict[str, Any]] = []
    for process in base._query_python_processes():
        command_line = str(process.get("CommandLine") or "")
        lower = command_line.lower()
        if "build_stage1_fewshot_dataset.py" not in lower:
            continue
        if any(pattern in lower for pattern in patterns):
            matches.append(process)
    return matches


def stop_current_task_unified_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
) -> tuple[str, dict[str, Any], str, str, str, str]:
    (
        status_md,
        dev_json,
        old_output_md,
        old_step_md,
        active_work_dir,
        old_panel,
    ) = base.stop_current_task_click(
        work_dir_output,
        work_dir,
        speaker_name,
    )

    try:
        active = _resolve_active_work_dir(
            active_work_dir,
            work_dir,
            speaker_name,
        )
        stage1_matches = _find_stage1_processes(active)
        stage1_kill_results = []
        for process in stage1_matches:
            pid = process.get("ProcessId")
            if pid is None:
                continue
            stage1_kill_results.append(
                base._kill_process_tree(int(pid))
            )

        stopped_stage1 = sum(
            1
            for item in stage1_kill_results
            if item.get("succeeded")
        )
        status_payload, panel_md, output_md, step_md = _scan_unified(active)
        dev_json = (
            dev_json
            if isinstance(dev_json, dict)
            else {"base_result": dev_json}
        )
        dev_json["stage1_matched_processes"] = stage1_matches
        dev_json["stage1_kill_results"] = stage1_kill_results
        dev_json["unified_status"] = status_payload

        if stopped_stage1 > 0:
            status_md = (
                f"{status_md}\n\n### Stage1 停止结果\n"
                f"- 成功停止 Stage1 子进程数：`{stopped_stage1}`"
            )

        return (
            status_md,
            dev_json,
            output_md,
            step_md,
            str(active),
            panel_md,
        )
    except Exception as exc:
        dev_json = (
            dev_json
            if isinstance(dev_json, dict)
            else {"base_result": dev_json}
        )
        dev_json["unified_stop_error"] = f"{type(exc).__name__}: {exc}"
        return (
            status_md,
            dev_json,
            old_output_md,
            old_step_md,
            active_work_dir,
            old_panel,
        )


# ============================================================
# UI
# UI 定义
# ============================================================
def create_demo() -> gr.Blocks:
    with gr.Blocks(title="VoiceLab 数据工厂 - 用户版") as demo:
        gr.Markdown(
            """
# VoiceLab 数据工厂 - 用户版

这个页面用于把原始中文语音转换为 VoiceLab Stage1 / Stage2 训练数据。

推荐流程：

1. 填写原始音频目录和说话人信息；
2. 点击 **执行数据工厂**；
3. 选择 **启动人工校对器** 或 **跳过人工校对**；
4. 点击 **生成 Stage1 训练数据**；
5. 首次生成 Stage2 点击 **一键生成 Stage2 训练数据**；已有工作目录或中断后恢复点击 **继续生成 Stage2 训练数据**。

右侧“工作目录状态”由统一状态管理器生成，统一显示 Stage1、Stage2、cache、train/val 和 filter 状态。
当前项目主线只针对中文语音生成，语言框暂时固定为 `zh`，保留作为后续多语言扩展接口。
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. 基础输入")

                audio_source_mode = gr.Radio(
                    label="原始音频来源",
                    choices=[
                        "本地目录路径",
                        "上传音频文件",
                        "上传音频文件夹",
                    ],
                    value="本地目录路径",
                    info=(
                        "大数据集建议使用本地目录路径；小样本或测试数据"
                        "可以使用上传入口。"
                    ),
                )

                raw_input_dir = gr.Textbox(
                    label="本地原始音频目录",
                    value="",
                    info=(
                        "填写已存在的本地音频目录。适合十几分钟、数小时"
                        "或更大的数据集。"
                    ),
                    visible=True,
                )

                uploaded_audio_files = gr.File(
                    label="上传音频文件",
                    file_count="multiple",
                    file_types=base.AUDIO_FILE_TYPES,
                    type="filepath",
                    visible=False,
                )
                uploaded_audio_folder = gr.File(
                    label="上传音频文件夹",
                    file_count="directory",
                    type="filepath",
                    visible=False,
                )
                upload_hint = gr.Markdown(
                    """
上传模式会先把音频复制到：

`work_dir/00_uploaded_raw_audio/`

然后再把该目录作为原始音频目录传给数据工厂。大数据集不建议使用浏览器上传。
""",
                    visible=False,
                )

                speaker_name = gr.Textbox(
                    label="说话人 / 角色名",
                    value="",
                )
                language = gr.Dropdown(
                    label="语言",
                    choices=["zh"],
                    value="zh",
                    info=(
                        "当前版本仅开放中文 zh。后续多语言版本可在这里扩展。"
                    ),
                )
                work_dir = gr.Textbox(
                    label="工作目录（可选）",
                    value="",
                    info=(
                        "留空时自动使用 user_data/{speaker_name}_factory。"
                        "页面刷新后可填入旧工作目录再点击载入/扫描。"
                    ),
                )

                with gr.Row():
                    load_work_dir_button = gr.Button(
                        "载入/扫描工作目录",
                        variant="secondary",
                    )
                    stop_current_task_button = gr.Button(
                        "停止当前任务",
                        variant="stop",
                    )

                with gr.Accordion("常用设置", open=True):
                    overwrite_work_dir = gr.Checkbox(
                        label="覆盖已有数据工厂工作目录",
                        value=False,
                    )
                    overwrite_uploaded_audio_cache = gr.Checkbox(
                        label="覆盖已上传音频缓存",
                        value=True,
                        info=(
                            "仅清理 work_dir/00_uploaded_raw_audio，"
                            "不会删除原始本地音频目录。"
                        ),
                    )
                    timeout_seconds = gr.Number(
                        label="单步超时秒数，0 表示不限制",
                        value=0,
                        precision=0,
                    )
                    stage1_train_ratio = gr.Number(
                        label="Stage1 训练集比例 train_ratio",
                        value=0.9,
                        precision=3,
                        info=(
                            "必须大于 0 且小于 1。验证集比例由 builder "
                            "自动计算；split seed 自动随机生成并记录。"
                        ),
                    )
                    stage2_train_ratio = gr.Number(
                        label="Stage2 训练集比例 train_ratio",
                        value=0.9,
                        precision=3,
                        info="必须大于 0 且小于 1。",
                    )
                    enable_speaker_embedding = gr.State(True)
                    auto_reject_bad_samples = gr.Checkbox(
                        label="自动隔离异常样本",
                        value=False,
                        info=(
                            "关闭时只生成异常报告，不移动样本；开启后会把"
                            "异常样本移动到 rejected 目录。"
                        ),
                    )
                    show_terminal_progress = gr.Checkbox(
                        label="显示终端进度窗口",
                        value=True,
                        info=(
                            "数据工厂和 Stage2 长任务运行时打开终端窗口"
                            "实时查看日志；任务结束后窗口会保留。"
                        ),
                    )

                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("### ASR 设置")
                    asr_backend = gr.Dropdown(
                        label="ASR 后端",
                        choices=[
                            "auto",
                            "faster_whisper",
                            "whisper",
                            "none",
                        ],
                        value="auto",
                    )
                    asr_model_size = gr.Textbox(
                        label="ASR 模型大小",
                        value="large-v3",
                    )
                    asr_precision = gr.Textbox(
                        label="ASR 精度",
                        value="float32",
                    )

                    gr.Markdown("### 音频切分参数")
                    with gr.Row():
                        threshold = gr.Number(
                            label="threshold",
                            value=-34,
                            precision=0,
                        )
                        min_length = gr.Number(
                            label="min_length",
                            value=4000,
                            precision=0,
                        )
                        min_interval = gr.Number(
                            label="min_interval",
                            value=300,
                            precision=0,
                        )
                    with gr.Row():
                        hop_size = gr.Number(
                            label="hop_size",
                            value=10,
                            precision=0,
                        )
                        max_sil_kept = gr.Number(
                            label="max_sil_kept",
                            value=500,
                            precision=0,
                        )
                    with gr.Row():
                        normalize_max = gr.Number(
                            label="normalize_max",
                            value=0.9,
                        )
                        alpha_mix = gr.Number(
                            label="alpha_mix",
                            value=0.25,
                        )

                    gr.Markdown("### Prompt 设置")
                    with gr.Row():
                        prompt_mode = gr.Dropdown(
                            label="prompt_mode",
                            choices=[
                                "self",
                                "speaker_pool",
                                "fixed_reference",
                            ],
                            value="speaker_pool",
                        )
                        allow_self_prompt = gr.Checkbox(
                            label="allow_self_prompt",
                            value=True,
                        )
                    with gr.Row():
                        min_prompt_sec = gr.Number(
                            label="min_prompt_sec",
                            value=3.0,
                        )
                        max_prompt_sec = gr.Number(
                            label="max_prompt_sec",
                            value=10.0,
                        )
                        prefer_prompt_sec = gr.Number(
                            label="prefer_prompt_sec",
                            value=6.0,
                        )

                    gr.Markdown("### Stage1 参数")
                    with gr.Row():
                        stage1_device = gr.Dropdown(
                            label="Stage1 device",
                            choices=["cuda", "cpu"],
                            value="cuda",
                        )
                        stage1_use_half = gr.Checkbox(
                            label="Stage1 use_half",
                            value=True,
                        )
                    with gr.Row():
                        stage1_overwrite = gr.Checkbox(
                            label="覆盖已有 Stage1 cache",
                            value=False,
                        )
                        stage1_validate_dataset = gr.Checkbox(
                            label="构建后验证 Stage1 dataset",
                            value=True,
                        )
                    gr.Markdown(
                        "Stage1 split seed 默认自动随机生成；"
                        "Stage1 默认处理全部输入样本。"
                    )

                    gr.Markdown("### Stage2 / Cache 参数")
                    with gr.Row():
                        stage2_device = gr.Dropdown(
                            label="device",
                            choices=["cuda", "cpu"],
                            value="cuda",
                        )
                        stage2_use_half = gr.Checkbox(
                            label="Stage2 use_half",
                            value=False,
                        )
                    with gr.Row():
                        stage2_overwrite = gr.Checkbox(
                            label="覆盖已有 Stage2 .pt",
                            value=False,
                        )
                        continuous_overwrite = gr.Checkbox(
                            label="覆盖已有 continuous cache",
                            value=False,
                        )
                        style_overwrite = gr.Checkbox(
                            label="覆盖已有 style cache",
                            value=False,
                        )
                        split_overwrite = gr.Checkbox(
                            label="覆盖已有 train/val",
                            value=False,
                        )
                    continuous_dtype = gr.Dropdown(
                        label="continuous dtype",
                        choices=["float16", "float32"],
                        value="float32",
                    )

                    gr.Markdown("### Mel / F0 参数")
                    with gr.Row():
                        target_sr = gr.Number(
                            label="target_sr",
                            value=22050,
                            precision=0,
                        )
                        n_fft = gr.Number(
                            label="n_fft",
                            value=1024,
                            precision=0,
                        )
                        hop_length = gr.Number(
                            label="hop_length",
                            value=256,
                            precision=0,
                        )
                        win_length = gr.Number(
                            label="win_length",
                            value=1024,
                            precision=0,
                        )
                    with gr.Row():
                        n_mels = gr.Number(
                            label="n_mels",
                            value=80,
                            precision=0,
                        )
                        fmin = gr.Number(label="fmin", value=0.0)
                        fmax = gr.Number(label="fmax", value=8000.0)
                    with gr.Row():
                        f0_min_hz = gr.Number(
                            label="f0_min_hz",
                            value=50.0,
                        )
                        f0_max_hz = gr.Number(
                            label="f0_max_hz",
                            value=1100.0,
                        )

                gr.Markdown("## 2. 执行流程")
                run_prepare_button = gr.Button(
                    "执行数据工厂",
                    variant="primary",
                )
                with gr.Row():
                    launch_proofread_button = gr.Button(
                        "启动人工校对器",
                        variant="secondary",
                    )
                    rebuild_after_proofread_button = gr.Button(
                        "我已完成校对",
                        variant="secondary",
                    )
                    skip_proofread_button = gr.Button(
                        "跳过人工校对",
                        variant="secondary",
                    )
                run_stage1_button = gr.Button(
                    "生成 Stage1 训练数据",
                    variant="primary",
                )
                with gr.Row():
                    run_postprocess_button = gr.Button(
                        "一键生成 Stage2 训练数据",
                        variant="primary",
                    )
                    resume_postprocess_button = gr.Button(
                        "继续生成 Stage2 训练数据",
                        variant="secondary",
                    )

                with gr.Accordion("人工校对设置", open=False):
                    proofread_port = gr.Number(
                        label="校对器端口",
                        value=9871,
                        precision=0,
                    )
                    proofread_g_batch = gr.Number(
                        label="校对批大小 g_batch",
                        value=10,
                        precision=0,
                    )
                    proofread_overwrite_backup = gr.Checkbox(
                        label="覆盖已有校对备份",
                        value=False,
                    )
                    duration_tolerance_sec = gr.Number(
                        label="重建 corrected manifest 时长容差",
                        value=0.05,
                    )
                    rebuild_timeout_seconds = gr.Number(
                        label="重建超时秒数",
                        value=600,
                        precision=0,
                    )

            with gr.Column(scale=1):
                gr.Markdown("## 状态")
                status_markdown = gr.Markdown("等待开始。")
                gr.Markdown("## 工作目录状态")
                work_dir_status_markdown = gr.Markdown("暂无统一状态。")
                gr.Markdown("## 输出摘要")
                output_summary_markdown = gr.Markdown("暂无输出。")
                gr.Markdown("## 步骤摘要")
                step_summary_markdown = gr.Markdown("暂无步骤记录。")
                work_dir_output = gr.Textbox(
                    label="当前工作目录",
                    value="",
                    interactive=False,
                )
                proofread_status = gr.Markdown("人工校对器尚未启动。")
                proofread_json = gr.JSON(
                    label="人工校对启动信息",
                    visible=False,
                )
                with gr.Accordion("开发者详情", open=False):
                    developer_json = gr.JSON(label="用户级 result JSON")

        # ----------------------------------------------------
        # Bind events
        # 绑定事件
        # ----------------------------------------------------
        unified_outputs = [
            status_markdown,
            developer_json,
            output_summary_markdown,
            step_summary_markdown,
            work_dir_output,
            work_dir_status_markdown,
        ]

        audio_source_mode.change(
            fn=base.audio_source_mode_change,
            inputs=[audio_source_mode],
            outputs=[
                raw_input_dir,
                uploaded_audio_files,
                uploaded_audio_folder,
                upload_hint,
            ],
        )

        load_work_dir_button.click(
            fn=inspect_work_dir_unified_click,
            inputs=[work_dir_output, work_dir, speaker_name],
            outputs=unified_outputs,
        )

        stop_current_task_button.click(
            fn=stop_current_task_unified_click,
            inputs=[work_dir_output, work_dir, speaker_name],
            outputs=unified_outputs,
        )

        run_prepare_button.click(
            fn=run_prepare_unified_click,
            inputs=[
                audio_source_mode,
                raw_input_dir,
                uploaded_audio_files,
                uploaded_audio_folder,
                overwrite_uploaded_audio_cache,
                speaker_name,
                language,
                work_dir,
                asr_backend,
                asr_model_size,
                asr_precision,
                threshold,
                min_length,
                min_interval,
                hop_size,
                max_sil_kept,
                normalize_max,
                alpha_mix,
                prompt_mode,
                min_prompt_sec,
                max_prompt_sec,
                prefer_prompt_sec,
                allow_self_prompt,
                overwrite_work_dir,
                timeout_seconds,
                show_terminal_progress,
            ],
            outputs=unified_outputs,
        )

        launch_proofread_button.click(
            fn=base.launch_proofread_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
                proofread_port,
                proofread_g_batch,
                proofread_overwrite_backup,
            ],
            outputs=[
                proofread_status,
                proofread_json,
                work_dir_output,
            ],
        )

        rebuild_after_proofread_button.click(
            fn=rebuild_after_proofread_unified_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
                duration_tolerance_sec,
                rebuild_timeout_seconds,
            ],
            outputs=unified_outputs,
        )

        skip_proofread_button.click(
            fn=skip_proofread_unified_click,
            inputs=[work_dir_output, work_dir, speaker_name],
            outputs=unified_outputs,
        )

        run_stage1_button.click(
            fn=run_stage1_dataset_unified_click,
            inputs=[
                work_dir_output,
                work_dir,
                speaker_name,
                stage1_device,
                stage1_use_half,
                stage1_overwrite,
                stage1_validate_dataset,
                stage1_train_ratio,
                timeout_seconds,
            ],
            outputs=unified_outputs,
        )

        postprocess_inputs = [
            work_dir_output,
            work_dir,
            speaker_name,
            stage2_device,
            stage2_use_half,
            stage2_overwrite,
            continuous_dtype,
            continuous_overwrite,
            style_overwrite,
            enable_speaker_embedding,
            auto_reject_bad_samples,
            stage2_train_ratio,
            split_overwrite,
            timeout_seconds,
            prompt_mode,
            min_prompt_sec,
            max_prompt_sec,
            prefer_prompt_sec,
            allow_self_prompt,
            target_sr,
            n_fft,
            hop_length,
            win_length,
            n_mels,
            fmin,
            fmax,
            f0_min_hz,
            f0_max_hz,
            show_terminal_progress,
        ]

        run_postprocess_button.click(
            fn=run_stage2_postprocess_unified_click,
            inputs=postprocess_inputs,
            outputs=unified_outputs,
        )
        resume_postprocess_button.click(
            fn=run_stage2_resume_unified_click,
            inputs=postprocess_inputs,
            outputs=unified_outputs,
        )

    return demo


# ============================================================
# CLI
# 命令行入口
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resonastra 数据工厂用户版 WebUI - Stage1/Stage2"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--auto-port", action="store_true")
    parser.add_argument("--auto-port-tries", type=int, default=20)
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        selected_port = base._find_available_port(
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
