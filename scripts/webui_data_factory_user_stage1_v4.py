from __future__ import annotations

"""Resonastra DataFactory user UI v4.

DF-UI-1 established the user-facing layout and preserved stable callbacks.
DF-UI-2 added four user flow cards and recommended actions.
DF-UI-3 added Runtime State v2 and a real live current-task monitor.
DF-UI-4 added six simultaneously visible Stage2 live-step states.
DF-UI-5 completed the real Prepare -> Stage1 -> Stage2 -> Stop -> Resume path.
DF-UI-6 consolidates user status and keeps detailed diagnostics collapsed.

Important DF-UI-3 concurrency rule:
long DataFactory callbacks are synchronous. A single callback that performs
``begin -> backend -> finish`` cannot activate a browser Timer until it returns,
which would make the "live" monitor completion-only. v4 therefore uses a
split Gradio event chain:

    short begin event -> stable long callback -> short finalize event

The begin event writes Runtime State v2 and activates the Timer before the long
callback runs. The Timer can then refresh concurrently while the queued backend
operation is still running. Stable DataFactory / Stage1 / Stage2 business
callbacks and their component input/output contracts remain unchanged.
"""

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

# Import v3 first for compatibility side effects: quarantine-aware status and
# resumable-pipeline runtime patches plus repaired legacy source loading.
from scripts import webui_data_factory_user_stage1_v3 as compat_v3  # noqa: F401
from scripts import webui_data_factory_user_stage1 as legacy
from scripts import webui_data_factory_user_stage1_v2 as stage1_v2
from src.voicelab_user.data_factory.lifecycle import stop_lifecycle_click
from src.voicelab_user.data_factory.live_monitor import (
    begin_live_task,
    build_live_monitor_view,
    finish_live_task,
)
from src.voicelab_user.data_factory.status_presentation import (
    format_status_overview_html,
)
from src.voicelab_user.data_factory.user_experience import (
    format_recommended_actions_markdown,
    format_user_flow_cards,
)


base = legacy.base
LIVE_MONITOR_INTERVAL_SECONDS = 5.0

# Stop still delegates to the stable process-kill implementation, then writes
# Runtime State v2. Long-task lifecycle itself is split into begin/backend/end
# Gradio events below so the Timer can run while the backend callback blocks.
stop_lifecycle_callback = partial(
    stop_lifecycle_click,
    legacy.stop_current_task_unified_click,
    legacy._resolve_active_work_dir,
)


DF_UI_V4_CSS = r"""
.vl-page-title { margin-bottom: 2px !important; }
.vl-page-subtitle { color: var(--body-text-color-subdued); margin-top: 0 !important; }
.vl-section-note { color: var(--body-text-color-subdued); font-size: 13px; }
.vl-settings-note {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 10px 12px;
    margin: 4px 0 10px 0;
    background: var(--background-fill-secondary);
}
.vl-flow-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 12px 0;
}
.vl-flow-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px 14px;
    background: var(--background-fill-secondary);
    min-height: 92px;
}
.vl-flow-card-head {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.vl-flow-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 999px;
    border: 1px solid var(--border-color-primary);
    font-weight: 700;
}
.vl-flow-title { font-weight: 700; }
.vl-flow-state { white-space: nowrap; font-size: 13px; }
.vl-flow-detail { color: var(--body-text-color-subdued); font-size: 13px; line-height: 1.5; }
.vl-flow-done { border-style: solid; }
.vl-flow-waiting_user, .vl-flow-partial { border-width: 2px; }
.vl-status-overview {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px 14px;
    background: var(--background-fill-secondary);
    margin: 8px 0 12px 0;
}
.vl-status-overview-title,
.vl-status-readiness-title {
    color: var(--body-text-color-subdued);
    font-size: 12px;
    font-weight: 700;
}
.vl-status-project-name {
    font-size: 15px;
    font-weight: 700;
    margin-top: 3px;
}
.vl-status-workdir {
    color: var(--body-text-color-subdued);
    font-size: 11px;
    line-height: 1.4;
    overflow-wrap: anywhere;
    margin: 2px 0 10px 0;
}
.vl-status-readiness-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 6px;
}
.vl-status-readiness-item {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 9px 10px;
}
.vl-status-readiness-main {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    gap: 6px;
    align-items: center;
}
.vl-status-readiness-state {
    text-align: right;
    color: var(--body-text-color-subdued);
    font-size: 11px;
}
.vl-status-readiness-detail {
    color: var(--body-text-color-subdued);
    font-size: 11px;
    margin-top: 4px;
}
.vl-status-all-ready {
    margin-top: 9px;
    font-size: 12px;
    font-weight: 600;
}
.vl-live-progress-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px 14px;
    background: var(--background-fill-secondary);
    margin: 6px 0 10px 0;
}
.vl-live-progress-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 9px;
}
.vl-live-progress-head span {
    color: var(--body-text-color-subdued);
    font-size: 13px;
    text-align: right;
}
.vl-live-progress-track {
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--border-color-primary);
}
.vl-live-progress-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--primary-500);
}
.vl-live-progress-meta {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 12px;
    margin-top: 6px;
    color: var(--body-text-color-subdued);
}
.vl-live-progress-position {
    font-size: 13px;
    color: var(--body-text-color-subdued);
    padding: 5px 0 2px 0;
}
.vl-stage2-steps {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 10px 12px;
    background: var(--background-fill-secondary);
    margin: 6px 0 10px 0;
}
.vl-stage2-steps-title {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    padding: 2px 2px 8px 2px;
}
.vl-stage2-steps-title span {
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.vl-stage2-step {
    border-top: 1px solid var(--border-color-primary);
    padding: 8px 2px;
}
.vl-stage2-step-main {
    display: grid;
    grid-template-columns: 24px 24px minmax(0, 1fr) auto;
    gap: 6px;
    align-items: center;
}
.vl-stage2-step-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 999px;
    border: 1px solid var(--border-color-primary);
    font-size: 12px;
    font-weight: 700;
}
.vl-stage2-step-icon { text-align: center; }
.vl-stage2-step-label { font-size: 13px; font-weight: 600; }
.vl-stage2-step-state {
    white-space: nowrap;
    font-size: 12px;
    color: var(--body-text-color-subdued);
}
.vl-stage2-step-detail,
.vl-stage2-step-guidance {
    margin: 4px 0 0 54px;
    font-size: 12px;
    color: var(--body-text-color-subdued);
}
.vl-stage2-step-guidance { font-weight: 600; }
.vl-stage2-step-current {
    border-left: 3px solid var(--primary-500);
    padding-left: 7px;
}
.vl-stage2-step-attention:not(.vl-stage2-step-current) {
    border-left: 3px solid var(--border-color-primary);
    padding-left: 7px;
}
.vl-stage2-steps-empty {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 10px 12px;
    color: var(--body-text-color-subdued);
    background: var(--background-fill-secondary);
    margin: 6px 0 10px 0;
}
.vl-proofread-status:empty { display: none; }
@media (max-width: 900px) {
    .vl-flow-grid,
    .vl-status-readiness-grid { grid-template-columns: 1fr; }
    .vl-stage2-step-main { grid-template-columns: 24px 24px minmax(0, 1fr); }
    .vl-stage2-step-state { grid-column: 3; }
}
"""


def _extract_unified_status(developer_payload: Any) -> dict[str, Any]:
    if not isinstance(developer_payload, dict):
        return {}

    nested = developer_payload.get("unified_status")
    if isinstance(nested, dict):
        return nested

    if isinstance(developer_payload.get("summary"), dict) and isinstance(
        developer_payload.get("stages"), list
    ):
        return developer_payload

    return {}


def refresh_user_experience_from_result(
    developer_payload: Any,
) -> tuple[Any, Any, Any]:
    status_payload = _extract_unified_status(developer_payload)
    if not status_payload:
        return gr.update(), gr.update(), gr.update()

    return (
        format_user_flow_cards(status_payload),
        format_recommended_actions_markdown(status_payload),
        format_status_overview_html(status_payload),
    )


def _attach_user_experience_refresh(
    event: Any,
    *,
    developer_json: Any,
    flow_cards_html: Any,
    recommended_actions_markdown: Any,
    status_overview_html: Any,
) -> Any:
    return event.then(
        fn=refresh_user_experience_from_result,
        inputs=[developer_json],
        outputs=[
            flow_cards_html,
            recommended_actions_markdown,
            status_overview_html,
        ],
        queue=False,
    )


def _resolve_active_monitor_work_dir(
    work_dir_output: Any,
    work_dir: Any,
    speaker_name: Any,
) -> Path:
    return legacy._resolve_active_work_dir(
        str(work_dir_output or ""),
        str(work_dir or ""),
        str(speaker_name or ""),
    )


def _resolve_prepare_monitor_work_dir(
    work_dir: Any,
    speaker_name: Any,
) -> Path:
    # Prepare itself does not consume work_dir_output, so a stale result from a
    # previously loaded project must not override the currently edited input.
    return legacy._resolve_active_work_dir(
        "",
        str(work_dir or ""),
        str(speaker_name or ""),
    )


def _query_process_snapshot() -> list[dict[str, Any]]:
    try:
        return list(base._query_python_processes())
    except Exception:
        return []


def _timer_component_update(active: bool):
    timer_type = getattr(gr, "Timer", None)
    if timer_type is None:
        return None
    return timer_type(
        value=LIVE_MONITOR_INTERVAL_SECONDS,
        active=bool(active),
    )


def _monitor_outputs(view: Any) -> tuple[str, str, str, dict[str, Any]]:
    return (
        view.current_task_markdown,
        view.progress_html,
        view.stage2_steps_html,
        view.diagnostic_payload,
    )


def begin_prepare_monitor(
    work_dir: str,
    speaker_name: str,
):
    active_work_dir = _resolve_prepare_monitor_work_dir(work_dir, speaker_name)
    view = begin_live_task(
        active_work_dir,
        operation="prepare",
        process_infos=_query_process_snapshot(),
    )
    return (*_monitor_outputs(view), _timer_component_update(True))


def begin_active_monitor(
    operation: str,
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
):
    active_work_dir = _resolve_active_monitor_work_dir(
        work_dir_output,
        work_dir,
        speaker_name,
    )
    view = begin_live_task(
        active_work_dir,
        operation=str(operation),
        process_infos=_query_process_snapshot(),
    )
    return (*_monitor_outputs(view), _timer_component_update(True))


def finalize_active_monitor(
    operation: str,
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    developer_payload: Any,
):
    active_work_dir = _resolve_active_monitor_work_dir(
        work_dir_output,
        work_dir,
        speaker_name,
    )
    view = finish_live_task(
        active_work_dir,
        operation=str(operation),
        result_payload=developer_payload,
        process_infos=_query_process_snapshot(),
    )
    return (*_monitor_outputs(view), _timer_component_update(view.monitor_active))


def refresh_live_monitor(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
):
    try:
        active_work_dir = _resolve_active_monitor_work_dir(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        view = build_live_monitor_view(
            active_work_dir,
            process_infos=_query_process_snapshot(),
        )
        return _monitor_outputs(view)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return (
            "### 当前任务\n\n⚠️ 本轮监控刷新失败。",
            "<div class=\"vl-live-progress-card\">监控暂不可用。</div>",
            '<div class="vl-stage2-steps-empty">Stage2 状态暂不可用。</div>',
            {
                "schema_version": "voicelab_data_factory_live_monitor_v1",
                "status": "failed",
                "error": error,
            },
        )


def refresh_live_monitor_tick(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
):
    try:
        active_work_dir = _resolve_active_monitor_work_dir(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        view = build_live_monitor_view(
            active_work_dir,
            process_infos=_query_process_snapshot(),
        )
        return (
            *_monitor_outputs(view),
            _timer_component_update(view.monitor_active),
        )
    except Exception as exc:
        current_md, progress_html, stage2_steps_html, payload = refresh_live_monitor(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        payload["timer_error"] = f"{type(exc).__name__}: {exc}"
        return (
            current_md,
            progress_html,
            stage2_steps_html,
            payload,
            _timer_component_update(False),
        )


def sync_live_monitor_timer(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
):
    try:
        active_work_dir = _resolve_active_monitor_work_dir(
            work_dir_output,
            work_dir,
            speaker_name,
        )
        view = build_live_monitor_view(
            active_work_dir,
            process_infos=_query_process_snapshot(),
        )
        return _timer_component_update(view.monitor_active)
    except Exception:
        return _timer_component_update(False)


def create_demo() -> gr.Blocks:
    initial_flow_cards = format_user_flow_cards({})
    initial_recommendation = format_recommended_actions_markdown({})
    initial_status_overview = format_status_overview_html({})

    with gr.Blocks(
        title="Resonastra · Data Factory",
        css=DF_UI_V4_CSS,
    ) as demo:
        gr.Markdown("# Resonastra · Data Factory", elem_classes=["vl-page-title"])
        gr.Markdown(
            "将角色语音转换为可用于 Stage1 / Stage2 训练的数据。",
            elem_classes=["vl-page-subtitle"],
        )
        gr.Markdown(
            "Stage1 与 Stage2 数据可以按需要独立生成。页面优先显示当前数据准备阶段；"
            "高级参数和诊断信息默认保持收起。",
            elem_classes=["vl-section-note"],
        )

        with gr.Row(equal_height=False):
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
""".strip(),
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
                        "当前版本仅开放中文 zh。保留该选项作为后续多语言扩展接口。"
                    ),
                )

                with gr.Accordion("已有项目 / 工作目录", open=False):
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

                gr.Markdown("## 2. 执行流程")
                gr.Markdown("### 数据准备阶段")
                flow_cards_html = gr.HTML(initial_flow_cards)

                gr.Markdown("### 可用操作")
                run_prepare_button = gr.Button(
                    "开始准备数据",
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
                        "无需校对，继续",
                        variant="secondary",
                    )
                run_stage1_button = gr.Button(
                    "生成 Stage1 训练数据",
                    variant="primary",
                )
                with gr.Row():
                    run_postprocess_button = gr.Button(
                        "生成 Stage2 训练数据",
                        variant="primary",
                    )
                    resume_postprocess_button = gr.Button(
                        "继续生成 Stage2 训练数据",
                        variant="secondary",
                    )

                gr.Markdown("## 3. 设置")
                gr.Markdown(
                    "常用设置与高级设置保留原版本参数结构；正常情况下使用默认值即可。",
                    elem_classes=["vl-settings-note"],
                )

                with gr.Accordion("常用设置", open=False):
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

                    stage1_train_ratio = gr.Slider(
                        label="Stage1 训练集比例 train_ratio",
                        minimum=0.50,
                        maximum=0.98,
                        step=0.01,
                        value=0.90,
                        info=(
                            "训练集比例 0.50–0.98；验证集比例自动使用 1 - train_ratio。"
                            "split seed 由 builder 自动随机生成并记录。"
                        ),
                    )
                    stage2_train_ratio = gr.Slider(
                        label="Stage2 训练集比例 train_ratio",
                        minimum=0.50,
                        maximum=0.98,
                        step=0.01,
                        value=0.90,
                        info="训练集比例 0.50–0.98；验证集比例自动使用 1 - train_ratio。",
                    )

                    auto_reject_bad_samples = gr.State(True)
                    enable_speaker_embedding = gr.State(True)

                    show_terminal_progress = gr.Checkbox(
                        label="显示终端进度窗口",
                        value=True,
                        info=(
                            "数据工厂、Stage1 和 Stage2 长任务运行时打开终端窗口"
                            "实时查看日志；终端仅作为进度查看器。"
                        ),
                    )

                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("### ASR 设置")
                    asr_backend = gr.Dropdown(
                        label="ASR 后端",
                        choices=["auto", "faster_whisper", "whisper", "none"],
                        value="auto",
                    )
                    asr_model_size = gr.Textbox(label="ASR 模型大小", value="large-v3")
                    asr_precision = gr.Textbox(label="ASR 精度", value="float32")

                    gr.Markdown("### 音频切分参数")
                    with gr.Row():
                        threshold = gr.Number(label="threshold", value=-34, precision=0)
                        min_length = gr.Number(label="min_length", value=4000, precision=0)
                        min_interval = gr.Number(label="min_interval", value=300, precision=0)
                    with gr.Row():
                        hop_size = gr.Number(label="hop_size", value=10, precision=0)
                        max_sil_kept = gr.Number(label="max_sil_kept", value=500, precision=0)
                    with gr.Row():
                        normalize_max = gr.Number(label="normalize_max", value=0.9)
                        alpha_mix = gr.Number(label="alpha_mix", value=0.25)

                    gr.Markdown("### Prompt 设置")
                    with gr.Row():
                        prompt_mode = gr.Dropdown(
                            label="prompt_mode",
                            choices=["self", "speaker_pool"],
                            value="speaker_pool",
                        )
                        allow_self_prompt = gr.Checkbox(
                            label="allow_self_prompt",
                            value=True,
                        )
                    with gr.Row():
                        min_prompt_sec = gr.Number(label="min_prompt_sec", value=3.0)
                        max_prompt_sec = gr.Number(label="max_prompt_sec", value=10.0)
                        prefer_prompt_sec = gr.Number(label="prefer_prompt_sec", value=6.0)

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
                        target_sr = gr.Number(label="target_sr", value=22050, precision=0)
                        n_fft = gr.Number(label="n_fft", value=1024, precision=0)
                        hop_length = gr.Number(label="hop_length", value=256, precision=0)
                        win_length = gr.Number(label="win_length", value=1024, precision=0)
                    with gr.Row():
                        n_mels = gr.Number(label="n_mels", value=80, precision=0)
                        fmin = gr.Number(label="fmin", value=0.0)
                        fmax = gr.Number(label="fmax", value=8000.0)
                    with gr.Row():
                        f0_min_hz = gr.Number(label="f0_min_hz", value=50.0)
                        f0_max_hz = gr.Number(label="f0_max_hz", value=1100.0)

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
                recommended_actions_markdown = gr.Markdown(initial_recommendation)
                status_overview_html = gr.HTML(initial_status_overview)

                with gr.Accordion("当前任务", open=True):
                    with gr.Row():
                        live_monitor_refresh_button = gr.Button(
                            "立即刷新",
                            variant="secondary",
                        )
                    current_task_markdown = gr.Markdown(
                        "### 当前任务\n\n⬜ 当前没有运行中的 DataFactory 任务。"
                    )
                    live_progress_html = gr.HTML(
                        '<div class="vl-live-progress-card">等待任务。</div>'
                    )
                    stage2_live_steps_html = gr.HTML(
                        '<div class="vl-stage2-steps-empty">暂无 Stage2 状态。</div>'
                    )

                proofread_status = gr.Markdown(
                    "",
                    elem_classes=["vl-proofread-status"],
                )

                # DF-UI-6 keeps exactly one outer diagnostics accordion. The
                # legacy callback outputs remain real Gradio components, but
                # they no longer compete with the user-facing status overview.
                with gr.Accordion("诊断信息", open=False):
                    gr.Markdown("### 最近一次操作")
                    status_markdown = gr.Markdown("等待开始。")
                    gr.Markdown("### 工作目录状态")
                    work_dir_status_markdown = gr.Markdown("暂无统一状态。")
                    gr.Markdown("### 输出摘要")
                    output_summary_markdown = gr.Markdown("暂无输出。")
                    gr.Markdown("### 步骤摘要")
                    step_summary_markdown = gr.Markdown("暂无步骤记录。")
                    work_dir_output = gr.Textbox(
                        label="当前工作目录",
                        value="",
                        interactive=False,
                    )
                    proofread_json = gr.JSON(
                        label="人工校对启动信息",
                        visible=False,
                    )
                    developer_json = gr.JSON(label="DataFactory Result JSON")
                    live_monitor_json = gr.JSON(label="DataFactory Live Monitor JSON")

        unified_outputs = [
            status_markdown,
            developer_json,
            output_summary_markdown,
            step_summary_markdown,
            work_dir_output,
            work_dir_status_markdown,
        ]
        live_monitor_outputs = [
            current_task_markdown,
            live_progress_html,
            stage2_live_steps_html,
            live_monitor_json,
        ]
        monitor_work_dir_inputs = [
            work_dir_output,
            work_dir,
            speaker_name,
        ]

        timer_type = getattr(gr, "Timer", None)
        live_monitor_timer = None
        if timer_type is not None:
            live_monitor_timer = timer_type(
                value=LIVE_MONITOR_INTERVAL_SECONDS,
                active=False,
            )

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

        load_event = load_work_dir_button.click(
            fn=legacy.inspect_work_dir_unified_click,
            inputs=[work_dir_output, work_dir, speaker_name],
            outputs=unified_outputs,
        )
        load_refresh_event = _attach_user_experience_refresh(
            load_event,
            developer_json=developer_json,
            flow_cards_html=flow_cards_html,
            recommended_actions_markdown=recommended_actions_markdown,
            status_overview_html=status_overview_html,
        )
        load_refresh_event.then(
            fn=refresh_live_monitor,
            inputs=monitor_work_dir_inputs,
            outputs=live_monitor_outputs,
            queue=False,
        )

        stop_event = stop_current_task_button.click(
            fn=stop_lifecycle_callback,
            inputs=[work_dir_output, work_dir, speaker_name],
            outputs=unified_outputs,
        )
        stop_refresh_event = _attach_user_experience_refresh(
            stop_event,
            developer_json=developer_json,
            flow_cards_html=flow_cards_html,
            recommended_actions_markdown=recommended_actions_markdown,
            status_overview_html=status_overview_html,
        )
        stop_monitor_event = stop_refresh_event.then(
            fn=refresh_live_monitor,
            inputs=monitor_work_dir_inputs,
            outputs=live_monitor_outputs,
            queue=False,
        )

        prepare_inputs = [
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
        ]

        if live_monitor_timer is not None:
            prepare_begin_outputs = [*live_monitor_outputs, live_monitor_timer]
        else:
            prepare_begin_outputs = live_monitor_outputs

        def _begin_prepare_ui(work_dir_value: str, speaker_value: str):
            result = begin_prepare_monitor(work_dir_value, speaker_value)
            return result if live_monitor_timer is not None else result[:4]

        prepare_begin_event = run_prepare_button.click(
            fn=_begin_prepare_ui,
            inputs=[work_dir, speaker_name],
            outputs=prepare_begin_outputs,
            queue=False,
        )
        prepare_backend_event = prepare_begin_event.then(
            fn=legacy.run_prepare_unified_click,
            inputs=prepare_inputs,
            outputs=unified_outputs,
        )

        # Launching the external proofreader does not itself represent a build
        # lifecycle; the rebuild after the user finishes proofreading does.
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
            outputs=[proofread_status, proofread_json, work_dir_output],
        )

        rebuild_inputs = [
            work_dir_output,
            work_dir,
            speaker_name,
            duration_tolerance_sec,
            rebuild_timeout_seconds,
        ]
        stage1_inputs = [
            work_dir_output,
            work_dir,
            speaker_name,
            stage1_device,
            stage1_use_half,
            stage1_overwrite,
            stage1_validate_dataset,
            stage1_train_ratio,
            timeout_seconds,
            show_terminal_progress,
        ]
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

        def _begin_active_ui(
            operation: str,
            current_work_dir: str,
            configured_work_dir: str,
            speaker: str,
        ):
            result = begin_active_monitor(
                operation,
                current_work_dir,
                configured_work_dir,
                speaker,
            )
            return result if live_monitor_timer is not None else result[:4]

        begin_outputs = (
            [*live_monitor_outputs, live_monitor_timer]
            if live_monitor_timer is not None
            else live_monitor_outputs
        )

        rebuild_begin_event = rebuild_after_proofread_button.click(
            fn=_begin_active_ui,
            inputs=[
                gr.State("proofread_rebuild"),
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=begin_outputs,
            queue=False,
        )
        rebuild_backend_event = rebuild_begin_event.then(
            fn=legacy.rebuild_after_proofread_unified_click,
            inputs=rebuild_inputs,
            outputs=unified_outputs,
        )

        # Skip proofread is a short file copy/update and remains direct.
        skip_event = skip_proofread_button.click(
            fn=legacy.skip_proofread_unified_click,
            inputs=[work_dir_output, work_dir, speaker_name],
            outputs=unified_outputs,
        )
        _attach_user_experience_refresh(
            skip_event,
            developer_json=developer_json,
            flow_cards_html=flow_cards_html,
            recommended_actions_markdown=recommended_actions_markdown,
            status_overview_html=status_overview_html,
        )

        stage1_begin_event = run_stage1_button.click(
            fn=_begin_active_ui,
            inputs=[
                gr.State("stage1"),
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=begin_outputs,
            queue=False,
        )
        stage1_backend_event = stage1_begin_event.then(
            fn=stage1_v2.run_stage1_dataset_unified_click,
            inputs=stage1_inputs,
            outputs=unified_outputs,
        )

        stage2_generate_begin_event = run_postprocess_button.click(
            fn=_begin_active_ui,
            inputs=[
                gr.State("stage2_generate"),
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=begin_outputs,
            queue=False,
        )
        stage2_generate_backend_event = stage2_generate_begin_event.then(
            fn=legacy.run_stage2_postprocess_unified_click,
            inputs=postprocess_inputs,
            outputs=unified_outputs,
        )

        stage2_resume_begin_event = resume_postprocess_button.click(
            fn=_begin_active_ui,
            inputs=[
                gr.State("stage2_resume"),
                work_dir_output,
                work_dir,
                speaker_name,
            ],
            outputs=begin_outputs,
            queue=False,
        )
        stage2_resume_backend_event = stage2_resume_begin_event.then(
            fn=legacy.run_stage2_resume_unified_click,
            inputs=postprocess_inputs,
            outputs=unified_outputs,
        )

        def _finalize_ui(
            operation: str,
            current_work_dir: str,
            configured_work_dir: str,
            speaker: str,
            developer_payload: Any,
        ):
            result = finalize_active_monitor(
                operation,
                current_work_dir,
                configured_work_dir,
                speaker,
                developer_payload,
            )
            return result if live_monitor_timer is not None else result[:4]

        finalize_outputs = begin_outputs
        backend_events = [
            (prepare_backend_event, "prepare"),
            (rebuild_backend_event, "proofread_rebuild"),
            (stage1_backend_event, "stage1"),
            (stage2_generate_backend_event, "stage2_generate"),
            (stage2_resume_backend_event, "stage2_resume"),
        ]
        finalized_events = []
        for backend_event, operation in backend_events:
            finalize_event = backend_event.then(
                fn=_finalize_ui,
                inputs=[
                    gr.State(operation),
                    work_dir_output,
                    work_dir,
                    speaker_name,
                    developer_json,
                ],
                outputs=finalize_outputs,
                queue=False,
            )
            finalized_events.append(finalize_event)
            _attach_user_experience_refresh(
                finalize_event,
                developer_json=developer_json,
                flow_cards_html=flow_cards_html,
                recommended_actions_markdown=recommended_actions_markdown,
                status_overview_html=status_overview_html,
            )

        live_monitor_refresh_event = live_monitor_refresh_button.click(
            fn=refresh_live_monitor,
            inputs=monitor_work_dir_inputs,
            outputs=live_monitor_outputs,
            queue=False,
        )

        if live_monitor_timer is not None:
            live_monitor_timer.tick(
                fn=refresh_live_monitor_tick,
                inputs=monitor_work_dir_inputs,
                outputs=[*live_monitor_outputs, live_monitor_timer],
                queue=False,
            )

            timer_sync_events = [
                load_event,
                stop_monitor_event,
                live_monitor_refresh_event,
                *finalized_events,
            ]
            for event in timer_sync_events:
                event.then(
                    fn=sync_live_monitor_timer,
                    inputs=monitor_work_dir_inputs,
                    outputs=live_monitor_timer,
                    queue=False,
                )

            load_method = getattr(demo, "load", None)
            if callable(load_method):
                page_load_event = load_method(
                    fn=refresh_live_monitor,
                    inputs=monitor_work_dir_inputs,
                    outputs=live_monitor_outputs,
                    queue=False,
                )
                page_load_event.then(
                    fn=sync_live_monitor_timer,
                    inputs=monitor_work_dir_inputs,
                    outputs=live_monitor_timer,
                    queue=False,
                )
        else:
            gr.Markdown(
                "当前 Gradio 版本不提供 `gr.Timer`；可使用“立即刷新”手动查看当前任务。"
            )

    return demo


def parse_args() -> argparse.Namespace:
    return legacy.parse_args()


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
