from __future__ import annotations

"""Resonastra user inference UI v4 — IF-UI-4 final pre-launch candidate.

v4 is intentionally a presentation-only shell over the already accepted v3
runtime path.  It does not rebuild or fork the inference backend.

Stable chain remains:
    v4 UI
      -> v3.run_user_inference_v3(...)
          -> UserInferenceService
              -> DeveloperInferenceAdapter
                  -> scripts/infer_zeroshot_v641.py

IF-UI-4 final polish:
- keep Profile -> reference audio -> target text -> Generate as the main path;
- keep Audio first on the result side;
- keep the truthful running -> final state from v3;
- remove backend/subprocess implementation wording from the normal-user running
  state while preserving the same diagnostic behavior;
- keep one collapsed diagnostics area;
- keep all 20 inference inputs and all Advanced/Metrics defaults/ranges/order;
- keep the formal launcher on the accepted v1 entry until final switch.
"""

import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve(strict=False).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import webui_user_infer as legacy
from scripts import webui_user_infer_v2 as v2
from scripts import webui_user_infer_v3 as v3


REAL_INFERENCE_MODE = v3.REAL_INFERENCE_MODE

# Compatibility aliases: v4 does not erase any previously accepted callback.
run_user_inference = legacy.run_user_inference
run_user_inference_v2 = v2.run_user_inference_v2
run_user_inference_v3 = v3.run_user_inference_v3
refresh_profiles_user_click = v3.refresh_profiles_user_click
set_default_profile_user_click = v3.set_default_profile_user_click
profile_selection_user_change = v3.profile_selection_user_change


IF_UI_V4_CSS = v3.IF_UI_V3_CSS + r"""
.vl-infer-hero-note {
    margin-top: 2px !important;
    margin-bottom: 14px !important;
    color: var(--body-text-color-subdued);
    font-size: 13px;
}
.vl-infer-step-note {
    color: var(--body-text-color-subdued);
    font-size: 12px;
    margin-top: -4px !important;
}
.vl-infer-result-empty {
    color: var(--body-text-color-subdued);
}
"""


def _running_user_markdown(profile_choice_value: str) -> str:
    """Final user-facing running state without implementation jargon."""

    profile = str(profile_choice_value or "").strip()
    lines = [
        "## ⏳ 正在生成语音",
        "",
        "Resonastra 正在根据参考音频和目标文本生成结果。完成后会自动更新播放器和质量信息。",
        "",
        "请保持当前页面打开，生成完成后即可直接试听。",
    ]
    if profile and not legacy._is_default_profile_choice(profile):
        lines.extend(["", f"**声音角色：** {profile.split(' |', 1)[0]}"])
    return "\n".join(lines)


def run_user_inference_v4(
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
) -> Iterator[tuple[Any, Any, Any, Any, Any]]:
    """Delegate the accepted v3 runtime path and polish only visible running text.

    The exact 20-input signature is preserved.  Every final result, audio path,
    diagnostic payload, diagnostic markdown, and button update comes directly
    from v3.  Only a genuine first running-state markdown is replaced.
    """

    generator = v3.run_user_inference_v3(
        prompt_wav,
        prompt_text,
        target_text,
        mode,
        profile_choice_value,
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
    )

    first_yield = True
    for state in generator:
        if first_yield and isinstance(state, tuple) and len(state) == 5:
            first_yield = False
            markdown = str(state[0] or "")
            # Invalid-input responses do not yield a running state in v3.  Only
            # replace the text when v3 truthfully reported that inference began.
            if "⏳ 正在生成" in markdown:
                yield (
                    _running_user_markdown(profile_choice_value),
                    state[1],
                    state[2],
                    state[3],
                    state[4],
                )
                continue
        else:
            first_yield = False
        yield state


def build_demo() -> gr.Blocks:
    (
        profile_initial_choices,
        profile_initial_value,
        _profile_initial_legacy_md,
        profile_initial_json,
    ) = legacy._initial_profile_ui_state()
    profile_initial_summary = v2._profile_summary(
        profile_initial_json,
        profile_initial_value,
    )

    with gr.Blocks(
        title="Resonastra 语音生成",
        css=IF_UI_V4_CSS,
    ) as demo:
        gr.Markdown("# Resonastra 语音生成", elem_classes=["vl-infer-title"])
        gr.Markdown(
            "选择声音角色、提供参考音频并输入目标文本，即可生成语音。",
            elem_classes=["vl-infer-subtitle"],
        )
        gr.Markdown(
            "常用操作都在主界面；模型覆盖、采样参数和诊断信息默认收起。",
            elem_classes=["vl-infer-hero-note"],
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                gr.Markdown("## 1. 选择声音角色")
                profile_choice = gr.Dropdown(
                    label="声音角色",
                    choices=profile_initial_choices,
                    value=profile_initial_value,
                    info="优先选择默认角色；也可以临时选择其他已就绪角色。",
                )
                profile_summary_markdown = gr.Markdown(
                    value=profile_initial_summary,
                    elem_classes=["vl-infer-profile-summary"],
                )

                with gr.Accordion("管理声音角色", open=False):
                    with gr.Row():
                        refresh_profiles_button = gr.Button("刷新角色列表", variant="secondary")
                        set_active_profile_button = gr.Button("设为默认角色", variant="secondary")
                    gr.Markdown("设为默认角色后，下次打开 Inference UI 会优先选中该角色。")

                gr.Markdown("## 2. 添加参考音频")
                gr.Markdown(
                    "参考文本应尽量与参考音频实际内容一致。",
                    elem_classes=["vl-infer-step-note"],
                )
                prompt_wav = gr.Audio(label="参考音频 / Prompt WAV", type="filepath")
                prompt_text = gr.Textbox(
                    label="参考音频文本 / Prompt Text",
                    lines=3,
                    placeholder="请输入参考音频中实际说出的文字。",
                )

                gr.Markdown("## 3. 输入需要生成的文本")
                target_text = gr.Textbox(
                    label="目标文本 / Target Text",
                    lines=5,
                    placeholder="请输入希望生成的语音内容。",
                )
                mode = gr.State(REAL_INFERENCE_MODE)
                run_button = gr.Button(value="生成语音", variant="primary", size="lg")

                # Exact v3 component types/defaults/ranges/order are retained.
                with gr.Accordion("高级选项 / Advanced Options", open=False):
                    gr.Markdown(
                        "高级 checkpoint override 会覆盖 Profile checkpoint。普通用户一般不需要填写。"
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
                        device = gr.Radio(label="设备 / Device", choices=["cuda", "cpu"], value="cuda")
                        use_fixed_seed = gr.Checkbox(
                            label="固定随机种子 / Fix seed",
                            value=False,
                            info="默认不固定 seed，以保持每次推理具有随机性。",
                        )
                        seed_value = gr.Number(
                            label="Seed value", value=0, precision=0, info="仅在勾选 Fix seed 时生效。"
                        )
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Stage1 temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.05
                        )
                        top_p = gr.Slider(
                            label="Stage1 top_p", minimum=0.1, maximum=1.0, value=1.0, step=0.01
                        )
                        top_k = gr.Slider(
                            label="Stage1 top_k", minimum=1, maximum=100, value=15, step=1
                        )
                    length_scale = gr.Slider(
                        label="Stage2 length_scale", minimum=0.5, maximum=2.0, value=1.0, step=0.05
                    )
                    timeout_seconds = gr.Number(
                        label="推理超时秒数 / Timeout Seconds",
                        value=600,
                        precision=1,
                        info="默认 600 秒。填 0 表示不限制超时。",
                    )

                with gr.Accordion("质量指标 / Metrics", open=False):
                    gr.Markdown("如需验收生成质量，可启用 DNSMOS、音色相似度和 WER/CER。")
                    enable_metrics = gr.Checkbox(label="启用质量指标 / Enable metrics", value=False)
                    compute_dnsmos = gr.Checkbox(label="DNSMOS", value=True)
                    compute_speaker_sim = gr.Checkbox(label="Speaker Sim", value=True)
                    compute_wer = gr.Checkbox(label="WER / CER", value=False)

            with gr.Column(scale=5):
                gr.Markdown("## 生成结果", elem_classes=["vl-infer-result-heading"])
                output_audio = gr.Audio(label="生成音频 / Generated Audio", type="filepath")
                output_markdown = gr.Markdown(
                    value="生成后的音频和结果信息会显示在这里。",
                    elem_classes=["vl-infer-result-summary", "vl-infer-runtime"],
                )

                with gr.Accordion("诊断信息", open=False):
                    gr.Markdown(
                        "仅在排查问题时展开。这里集中保存 Profile Registry、原始路径、Developer command、stdout/stderr 和错误详情。",
                        elem_classes=["vl-infer-diagnostics-note"],
                    )
                    profile_json = gr.JSON(label="Profile Registry JSON", value=profile_initial_json)
                    inference_diagnostics_markdown = gr.Markdown(
                        "尚无推理诊断信息。完成一次推理后会显示本次运行详情。"
                    )
                    inference_diagnostics_json = gr.JSON(label="Inference Diagnostics JSON", value={})

        refresh_profiles_button.click(
            fn=refresh_profiles_user_click,
            inputs=[],
            outputs=[profile_choice, profile_summary_markdown, profile_json],
        )
        set_active_profile_button.click(
            fn=set_default_profile_user_click,
            inputs=[profile_choice],
            outputs=[profile_choice, profile_summary_markdown, profile_json],
        )
        profile_choice.change(
            fn=profile_selection_user_change,
            inputs=[profile_choice],
            outputs=[profile_summary_markdown],
        )

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
            inference_diagnostics_markdown,
            inference_diagnostics_json,
            run_button,
        ]
        run_button.click(
            fn=run_user_inference_v4,
            inputs=run_inputs,
            outputs=run_outputs,
        )

    return demo


def parse_args():
    return legacy.parse_args()


def main() -> int:
    args = parse_args()
    print("====================================================")
    print("Resonastra User Inference WebUI v4 (IF-UI-4 candidate)")
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
