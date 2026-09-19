from __future__ import annotations

"""VoiceLab user inference UI v2 — IF-UI-1 + IF-UI-2.

The accepted inference backend remains unchanged:

    UI
      -> UserInferenceRequest
          -> UserInferenceService
              -> DeveloperInferenceAdapter
                  -> scripts/infer_zeroshot_v641.py

IF-UI-1 established the v2 user-flow shell without switching the formal
launcher.  IF-UI-2 adds presentation-only userization:
- compact selected-profile status instead of registry/path-heavy text;
- friendly success/failure/result language;
- user-facing DNSMOS / Speaker Sim / WER / CER / RTF presentation;
- raw Profile Registry JSON stays inside one collapsed diagnostics area;
- normal result text omits raw run paths, JSON paths, commands, and traceback;
- the original legacy callback is still re-exported unchanged for regression
  compatibility, while the v2 UI uses a 20-input presentation-aware wrapper.

No request/service/adapter/developer-inference/profile-resolution contract is
changed in this file.  ``launch_infer_ui.bat`` is intentionally not switched
yet.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve(strict=False).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import webui_user_infer as legacy
from src.voicelab_user.inference.presentation import (
    format_inference_result_markdown,
    format_profile_presentation_markdown,
    format_user_exception_markdown,
)


# -----------------------------------------------------------------------------
# Stable compatibility references
# -----------------------------------------------------------------------------
# Keep these exact aliases available so IF-UI v2 never erases the accepted v1
# callback/profile contracts.  The v2 Blocks intentionally use the new wrappers
# further below for presentation only.
run_user_inference = legacy.run_user_inference
refresh_profiles_click = legacy.refresh_profiles_click
set_active_profile_click = legacy.set_active_profile_click

REAL_INFERENCE_MODE = "Run real inference"


IF_UI_V2_CSS = r"""
.vl-infer-title { margin-bottom: 2px !important; }
.vl-infer-subtitle {
    margin-top: 0 !important;
    color: var(--body-text-color-subdued);
}
.vl-infer-note {
    margin-top: 0 !important;
    color: var(--body-text-color-subdued);
    font-size: 13px;
}
.vl-infer-profile-summary {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 10px 12px;
    background: var(--background-fill-secondary);
    margin: 4px 0 10px 0;
}
.vl-infer-result-heading { margin-bottom: 4px !important; }
.vl-infer-result-summary {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 10px 12px;
    background: var(--background-fill-secondary);
}
"""


# -----------------------------------------------------------------------------
# IF-UI-2 profile presentation callbacks
# -----------------------------------------------------------------------------


def _profile_summary(
    payload: Optional[dict[str, Any]],
    selected_value: Optional[str],
) -> str:
    return format_profile_presentation_markdown(
        payload or {},
        selected_value,
        default_choice=legacy.NO_PROFILE_CHOICE,
    )


def refresh_profiles_user_click() -> tuple[Any, str, dict[str, Any]]:
    """Refresh profile choices and return compact user-facing profile status."""

    payload = legacy._safe_scan_profiles()
    choices, value = legacy._profile_dropdown_choices(payload)
    return (
        gr.update(choices=choices, value=value),
        _profile_summary(payload, value),
        payload,
    )


def set_default_profile_user_click(
    profile_choice_value: str,
) -> tuple[Any, str, dict[str, Any]]:
    """Set active profile while keeping the UI message user-facing."""

    if legacy._is_default_profile_choice(profile_choice_value):
        payload = legacy._safe_scan_profiles()
        choices, value = legacy._profile_dropdown_choices(payload)
        summary = _profile_summary(payload, value)
        summary += "\n\n> 当前使用系统默认配置；未修改默认声音角色。"
        return gr.update(choices=choices, value=value), summary, payload

    if legacy.set_active_profile is None:
        payload = legacy._safe_scan_profiles()
        choices, value = legacy._profile_dropdown_choices(payload)
        summary = _profile_summary(payload, value)
        summary += "\n\n> ⚠️ 无法修改默认角色，请检查 Profile Registry。"
        return gr.update(choices=choices, value=value), summary, payload

    try:
        active_payload = legacy.set_active_profile(profile_choice_value)
    except Exception as exc:  # noqa: BLE001 - UI boundary only.
        payload = legacy._safe_scan_profiles()
        choices, value = legacy._profile_dropdown_choices(payload)
        summary = _profile_summary(payload, value)
        summary += (
            "\n\n> ⚠️ 设置默认角色失败："
            f"{exc.__class__.__name__}。请刷新角色状态后重试。"
        )
        return gr.update(choices=choices, value=value), summary, payload

    payload = legacy._safe_scan_profiles()
    choices, value = legacy._profile_dropdown_choices(payload)
    summary = _profile_summary(payload, value)
    active_name = str(active_payload.get("active_profile_name") or "").strip()
    if active_name:
        summary += f"\n\n> ✅ 已将 **{active_name}** 设为默认声音角色。"
    return gr.update(choices=choices, value=value), summary, payload


def profile_selection_user_change(profile_choice_value: str) -> str:
    """Update visible profile summary when the dropdown selection changes."""

    payload = legacy._safe_scan_profiles()
    return _profile_summary(payload, profile_choice_value)


# -----------------------------------------------------------------------------
# IF-UI-2 inference callback
# -----------------------------------------------------------------------------


def run_user_inference_v2(
    prompt_wav,
    prompt_text: str,
    target_text: str,
    mode: str,
    profile_choice_value: str,
    output_dir: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    device: str,
    use_fixed_seed: bool,
    seed_value: Optional[float],
    temperature: float,
    top_p: float,
    top_k: float,
    length_scale: float,
    enable_metrics: bool,
    compute_dnsmos: bool,
    compute_speaker_sim: bool,
    compute_wer: bool,
    timeout_seconds: Optional[float],
):
    """Run the stable backend but render its result for normal users.

    Important compatibility rule:
        The argument count and order intentionally match the accepted legacy
        ``run_user_inference`` callback exactly (20 inputs).
    """

    if legacy.IMPORT_ERROR is not None:
        return (
            format_user_exception_markdown(
                title="推理组件加载失败",
                message="VoiceLab 推理服务暂时无法启动。",
                guidance="请检查运行环境后重新启动 Inference UI。",
            ),
            None,
        )

    prompt_wav_path = legacy._file_to_path(prompt_wav)
    if not prompt_wav_path:
        return (
            format_user_exception_markdown(
                title="缺少参考音频",
                message="请先上传或选择一段参考音频。",
                guidance="参考音频用于提供说话人音色与风格信息。",
            ),
            None,
        )

    prompt_text = str(prompt_text or "").strip()
    target_text = str(target_text or "").strip()

    if not prompt_text:
        return (
            format_user_exception_markdown(
                title="缺少参考文本",
                message="请输入参考音频对应的文字内容。",
            ),
            None,
        )

    if not target_text:
        return (
            format_user_exception_markdown(
                title="缺少生成文本",
                message="请输入希望 VoiceLab 生成的目标文本。",
            ),
            None,
        )

    dry_run = legacy._bool_from_mode(mode)
    seed_int = legacy._normalize_seed_value(use_fixed_seed, seed_value)
    top_k_int = int(top_k) if top_k is not None else None

    try:
        (
            resolved_stage1_ckpt,
            resolved_stage2_ckpt,
            _resolved_profile,
            _profile_md,
        ) = legacy._resolve_profile_checkpoint_overrides(
            profile_choice_value,
            stage1_ckpt,
            stage2_ckpt,
        )

        request = legacy.UserInferenceRequest(
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            target_text=target_text,
            output_dir=legacy._optional_text(output_dir),
            stage1_ckpt=resolved_stage1_ckpt,
            stage2_ckpt=resolved_stage2_ckpt,
            device=legacy._optional_text(device),
            seed=seed_int,
            temperature=float(temperature) if temperature is not None else None,
            top_p=float(top_p) if top_p is not None else None,
            top_k=top_k_int,
            length_scale=float(length_scale) if length_scale is not None else None,
            enable_metrics=bool(enable_metrics),
            compute_dnsmos=bool(compute_dnsmos),
            compute_speaker_sim=bool(compute_speaker_sim),
            compute_wer=bool(compute_wer),
        )

        normalized_timeout_seconds = legacy._normalize_timeout_seconds(timeout_seconds)
        service = legacy.create_user_inference_service(
            project_root=str(PROJECT_ROOT),
            python_executable=None,
            inference_timeout_seconds=normalized_timeout_seconds,
        )

        result = service.run(request, dry_run=dry_run)
        return (
            format_inference_result_markdown(result),
            legacy._result_output_audio_path(result),
        )

    except Exception as exc:  # noqa: BLE001 - UI boundary, traceback intentionally hidden.
        return (
            format_user_exception_markdown(
                title="语音生成未完成",
                message=f"推理服务返回异常：{exc.__class__.__name__}。",
                guidance="请检查参考音频、声音角色和高级设置后重试。",
            ),
            None,
        )


# -----------------------------------------------------------------------------
# UI construction
# -----------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    """Build the IF-UI-2 user-facing Gradio shell."""

    (
        profile_initial_choices,
        profile_initial_value,
        _profile_initial_legacy_md,
        profile_initial_json,
    ) = legacy._initial_profile_ui_state()
    profile_initial_summary = _profile_summary(
        profile_initial_json,
        profile_initial_value,
    )

    with gr.Blocks(
        title="VoiceLab 语音生成 - 用户版 v2",
        css=IF_UI_V2_CSS,
    ) as demo:
        gr.Markdown(
            "# VoiceLab 语音生成",
            elem_classes=["vl-infer-title"],
        )
        gr.Markdown(
            "选择声音角色、提供参考音频并输入目标文本，即可生成语音。",
            elem_classes=["vl-infer-subtitle"],
        )
        gr.Markdown(
            "普通使用场景默认执行真实推理；模型覆盖、采样参数和诊断信息均保持收起。",
            elem_classes=["vl-infer-note"],
        )

        with gr.Row(equal_height=False):
            # -----------------------------------------------------------------
            # Left: user inputs and the single primary action.
            # -----------------------------------------------------------------
            with gr.Column(scale=6):
                gr.Markdown("## 1. 声音角色")
                profile_choice = gr.Dropdown(
                    label="当前声音角色",
                    choices=profile_initial_choices,
                    value=profile_initial_value,
                    info="默认优先使用当前默认角色；没有默认角色时选择最新可用角色。",
                )
                profile_summary_markdown = gr.Markdown(
                    value=profile_initial_summary,
                    elem_classes=["vl-infer-profile-summary"],
                )

                with gr.Accordion("管理声音角色", open=False):
                    with gr.Row():
                        refresh_profiles_button = gr.Button(
                            "刷新角色列表",
                            variant="secondary",
                        )
                        set_active_profile_button = gr.Button(
                            "设为默认角色",
                            variant="secondary",
                        )
                    gr.Markdown(
                        "默认角色会在下次打开 Inference UI 时优先选中。"
                    )

                gr.Markdown("## 2. 参考音频")
                prompt_wav = gr.Audio(
                    label="参考音频 / Prompt WAV",
                    type="filepath",
                )
                prompt_text = gr.Textbox(
                    label="参考音频文本 / Prompt Text",
                    lines=3,
                    placeholder="请输入参考音频对应文本，例如：这是参考音频。",
                )

                gr.Markdown("## 3. 生成文本")
                target_text = gr.Textbox(
                    label="目标文本 / Target Text",
                    lines=5,
                    placeholder="请输入需要合成的文本。",
                )

                # Dry-run remains supported by the stable backend, but normal
                # users always use real inference from the v2 interface.
                mode = gr.State(REAL_INFERENCE_MODE)

                run_button = gr.Button(
                    value="生成语音",
                    variant="primary",
                    size="lg",
                )

                # Advanced controls retain IF-UI-1 / legacy names, order,
                # choices, ranges, and defaults.
                with gr.Accordion("高级选项 / Advanced Options", open=False):
                    gr.Markdown(
                        """
高级 checkpoint override 会覆盖上方 Profile 中的 checkpoint。
普通用户一般不需要填写；Training UI 生成的 User Profile 会自动提供 Stage1 / Stage2 best_model.pt。
""".strip()
                    )
                    output_dir = gr.Textbox(
                        label="输出目录 / Output Directory",
                        value="",
                        placeholder="留空则自动写入 outputs/inference_runs。",
                    )
                    stage1_ckpt = gr.Textbox(
                        label="Stage1 checkpoint override",
                        value="",
                        placeholder="可选：few-shot Stage1 checkpoint。留空则使用 Profile / 默认 Stage1。",
                    )
                    stage2_ckpt = gr.Textbox(
                        label="Stage2 checkpoint override",
                        value="",
                        placeholder="可选：few-shot Stage2 checkpoint。留空则使用 Profile / 默认 Stage2。",
                    )

                    with gr.Row():
                        device = gr.Radio(
                            label="设备 / Device",
                            choices=["cuda", "cpu"],
                            value="cuda",
                        )
                        use_fixed_seed = gr.Checkbox(
                            label="固定随机种子 / Fix seed",
                            value=False,
                            info="默认不固定 seed，以保持每次推理具有随机性。",
                        )
                        seed_value = gr.Number(
                            label="Seed value",
                            value=0,
                            precision=0,
                            info="仅在勾选 Fix seed 时生效。",
                        )

                    with gr.Row():
                        temperature = gr.Slider(
                            label="Stage1 temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                        )
                        top_p = gr.Slider(
                            label="Stage1 top_p",
                            minimum=0.1,
                            maximum=1.0,
                            value=1.0,
                            step=0.01,
                        )
                        top_k = gr.Slider(
                            label="Stage1 top_k",
                            minimum=1,
                            maximum=100,
                            value=15,
                            step=1,
                        )

                    length_scale = gr.Slider(
                        label="Stage2 length_scale",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                    )

                    timeout_seconds = gr.Number(
                        label="推理超时秒数 / Timeout Seconds",
                        value=600,
                        precision=1,
                        info="默认 600 秒。填 0 表示不限制超时。",
                    )

                with gr.Accordion("质量指标 / Metrics", open=False):
                    gr.Markdown(
                        "开启后可在生成结果中查看 DNSMOS、音色相似度、WER/CER 等验收指标。"
                    )
                    enable_metrics = gr.Checkbox(
                        label="启用质量指标 / Enable metrics",
                        value=False,
                    )
                    compute_dnsmos = gr.Checkbox(
                        label="DNSMOS",
                        value=True,
                    )
                    compute_speaker_sim = gr.Checkbox(
                        label="Speaker Sim",
                        value=True,
                    )
                    compute_wer = gr.Checkbox(
                        label="WER / CER",
                        value=False,
                    )

            # -----------------------------------------------------------------
            # Right: audio first, then normal-user summary, then diagnostics.
            # -----------------------------------------------------------------
            with gr.Column(scale=5):
                gr.Markdown(
                    "## 生成结果",
                    elem_classes=["vl-infer-result-heading"],
                )
                output_audio = gr.Audio(
                    label="生成音频 / Generated Audio",
                    type="filepath",
                )
                output_markdown = gr.Markdown(
                    value="尚未生成语音。",
                    elem_classes=["vl-infer-result-summary"],
                )

                with gr.Accordion("诊断信息", open=False):
                    gr.Markdown(
                        "这里保留 Profile Registry 原始状态。推理命令、原始路径和 traceback 仍由底层 result/service contract 保存，不在普通结果区展示。"
                    )
                    profile_json = gr.JSON(
                        label="Profile Registry JSON",
                        value=profile_initial_json,
                    )

        # Profile user-facing callbacks retain the accepted three outputs.
        refresh_profiles_button.click(
            fn=refresh_profiles_user_click,
            inputs=[],
            outputs=[
                profile_choice,
                profile_summary_markdown,
                profile_json,
            ],
        )
        set_active_profile_button.click(
            fn=set_default_profile_user_click,
            inputs=[profile_choice],
            outputs=[
                profile_choice,
                profile_summary_markdown,
                profile_json,
            ],
        )
        profile_choice.change(
            fn=profile_selection_user_change,
            inputs=[profile_choice],
            outputs=[profile_summary_markdown],
        )

        # Keep the exact accepted 20-input order and two normal UI outputs.
        run_inputs = [
            prompt_wav,
            prompt_text,
            target_text,
            mode,
            profile_choice,
            output_dir,
            stage1_ckpt,
            stage2_ckpt,
            device,
            use_fixed_seed,
            seed_value,
            temperature,
            top_p,
            top_k,
            length_scale,
            enable_metrics,
            compute_dnsmos,
            compute_speaker_sim,
            compute_wer,
            timeout_seconds,
        ]
        run_outputs = [
            output_markdown,
            output_audio,
        ]
        run_button.click(
            fn=run_user_inference_v2,
            inputs=run_inputs,
            outputs=run_outputs,
        )

    return demo


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    """Keep the stable legacy CLI flags/defaults."""

    return legacy.parse_args()


def main() -> int:
    args = parse_args()

    print("====================================================")
    print("VoiceLab User Inference WebUI v2 (IF-UI-2)")
    print("====================================================")
    print(f"project_root : {PROJECT_ROOT}")
    print(f"host         : {args.host}")
    print(f"port         : {args.port}")
    print(f"share        : {args.share}")
    print("====================================================")

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=int(args.port),
        share=bool(args.share),
        inbrowser=bool(args.inbrowser),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
