from __future__ import annotations

"""VoiceLab user inference UI v3 — IF-UI-3 runtime + diagnostics shell.

This version is a side-by-side migration target.  It does NOT replace the
formal launcher yet and does NOT modify the accepted inference backend.

Architecture remains:
    Gradio UI
      -> UserInferenceRequest
          -> UserInferenceService
              -> DeveloperInferenceAdapter
                  -> scripts/infer_zeroshot_v641.py

IF-UI-3 adds only UI-boundary behavior:
- a truthful two-state runtime presentation: running -> final result;
- no fabricated Stage1/Stage2 percentage (backend subprocess is synchronous);
- Generate button disabled while the synchronous service call is active;
- one collapsed diagnostics area containing Profile Registry plus inference
  command/raw paths/stdout/stderr/error information;
- 20 inference inputs stay in the same positional order and use the same
  defaults/ranges as the accepted v2 UI.
"""

import sys
import traceback
from pathlib import Path
from typing import Any, Iterator, Optional

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve(strict=False).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import webui_user_infer as legacy
from scripts import webui_user_infer_v2 as v2
from src.voicelab_user.inference.diagnostics import (
    build_inference_diagnostics_payload,
    format_inference_diagnostics_markdown,
)
from src.voicelab_user.inference.presentation import (
    format_inference_result_markdown,
    format_user_exception_markdown,
)


REAL_INFERENCE_MODE = v2.REAL_INFERENCE_MODE

IF_UI_V3_CSS = v2.IF_UI_V2_CSS + r"""
.vl-infer-runtime {
    border-left: 4px solid var(--border-color-primary);
}
.vl-infer-diagnostics-note {
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
"""


# -----------------------------------------------------------------------------
# Compatibility aliases
# -----------------------------------------------------------------------------
run_user_inference = legacy.run_user_inference
run_user_inference_v2 = v2.run_user_inference_v2
refresh_profiles_user_click = v2.refresh_profiles_user_click
set_default_profile_user_click = v2.set_default_profile_user_click
profile_selection_user_change = v2.profile_selection_user_change


# -----------------------------------------------------------------------------
# Runtime / diagnostics helpers
# -----------------------------------------------------------------------------


def _running_markdown(profile_choice_value: str) -> str:
    profile = str(profile_choice_value or "").strip()
    lines = [
        "## ⏳ 正在生成",
        "",
        "VoiceLab 正在执行真实推理。完成后会自动更新音频和质量结果。",
        "",
        "当前 backend 是同步 subprocess，因此这里只显示真实的运行状态，不显示虚假的阶段百分比。",
    ]
    if profile and not legacy._is_default_profile_choice(profile):
        lines.extend(["", f"**当前声音角色：** {profile.split(' |', 1)[0]}"])
    return "\n".join(lines)


def _waiting_diagnostics_markdown() -> str:
    return "推理正在运行。命令、stdout/stderr 和最终错误信息会在任务结束后写入这里。"


def _exception_diagnostics_payload(
    exc: BaseException,
    *,
    selected_profile: str,
) -> dict[str, Any]:
    return {
        "schema_version": "voicelab_inference_ui_diagnostics_v1",
        "status": "failed",
        "profile": {
            "selected_value": str(selected_profile or ""),
            "resolved_profile_name": "",
            "inference_mode": "",
            "language": "zh",
        },
        "artifacts": {},
        "timing": {},
        "metrics": {},
        "error": {
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "stage": "inference_ui_boundary",
            "traceback_text": traceback.format_exc(),
            "hint": "Check profile/input paths and advanced inference settings.",
        },
        "backend": {
            "returncode": None,
            "timeout": False,
            "stdout_tail": "",
            "stderr_tail": "",
        },
        "developer_command": {},
        "result_extra": {},
    }


def _final_invalid_input(
    *,
    title: str,
    message: str,
    guidance: str,
) -> tuple[str, None, str, dict[str, Any], Any]:
    return (
        format_user_exception_markdown(
            title=title,
            message=message,
            guidance=guidance,
        ),
        None,
        "尚未启动 backend，因此没有本次推理诊断信息。",
        {},
        gr.update(value="生成语音", interactive=True),
    )


def run_user_inference_v3(
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
    """Generator callback: truthful running state followed by final result.

    The 20 input arguments intentionally match legacy/v2 exactly.
    Outputs:
        1. normal-user result markdown
        2. generated audio
        3. collapsed diagnostic markdown
        4. collapsed diagnostic JSON
        5. Generate button update
    """

    if legacy.IMPORT_ERROR is not None:
        yield _final_invalid_input(
            title="推理组件加载失败",
            message="VoiceLab 推理服务暂时无法启动。",
            guidance="请检查运行环境后重新启动 Inference UI。",
        )
        return

    prompt_wav_path = legacy._file_to_path(prompt_wav)
    if not prompt_wav_path:
        yield _final_invalid_input(
            title="缺少参考音频",
            message="请先上传或选择一段参考音频。",
            guidance="参考音频用于提供说话人音色与风格信息。",
        )
        return

    prompt_text = str(prompt_text or "").strip()
    target_text = str(target_text or "").strip()
    if not prompt_text:
        yield _final_invalid_input(
            title="缺少参考文本",
            message="请输入参考音频对应的文字内容。",
            guidance="参考文本应与参考音频内容一致。",
        )
        return
    if not target_text:
        yield _final_invalid_input(
            title="缺少生成文本",
            message="请输入希望 VoiceLab 生成的目标文本。",
            guidance="输入目标文本后重新点击生成语音。",
        )
        return

    # This is the only truthful live state available without changing the
    # synchronous adapter subprocess contract.
    yield (
        _running_markdown(profile_choice_value),
        None,
        _waiting_diagnostics_markdown(),
        {},
        gr.update(value="生成中...", interactive=False),
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

        service = legacy.create_user_inference_service(
            project_root=str(PROJECT_ROOT),
            python_executable=None,
            inference_timeout_seconds=legacy._normalize_timeout_seconds(timeout_seconds),
        )
        result = service.run(request, dry_run=dry_run)

        diagnostic_payload = build_inference_diagnostics_payload(
            result,
            selected_profile=profile_choice_value,
        )
        yield (
            format_inference_result_markdown(result),
            legacy._result_output_audio_path(result),
            format_inference_diagnostics_markdown(diagnostic_payload),
            diagnostic_payload,
            gr.update(value="生成语音", interactive=True),
        )

    except Exception as exc:  # noqa: BLE001 - raw detail goes to diagnostics only.
        diagnostic_payload = _exception_diagnostics_payload(
            exc,
            selected_profile=profile_choice_value,
        )
        yield (
            format_user_exception_markdown(
                title="语音生成未完成",
                message=f"推理服务返回异常：{exc.__class__.__name__}。",
                guidance="请检查参考音频、声音角色和高级设置后重试。",
            ),
            None,
            format_inference_diagnostics_markdown(diagnostic_payload),
            diagnostic_payload,
            gr.update(value="生成语音", interactive=True),
        )


# -----------------------------------------------------------------------------
# UI construction
# -----------------------------------------------------------------------------


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
        title="VoiceLab 语音生成 - 用户版 v3",
        css=IF_UI_V3_CSS,
    ) as demo:
        gr.Markdown("# VoiceLab 语音生成", elem_classes=["vl-infer-title"])
        gr.Markdown(
            "选择声音角色、提供参考音频并输入目标文本，即可生成语音。",
            elem_classes=["vl-infer-subtitle"],
        )
        gr.Markdown(
            "真实推理期间会显示运行状态；开发者命令和原始错误只保留在折叠诊断信息中。",
            elem_classes=["vl-infer-note"],
        )

        with gr.Row(equal_height=False):
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
                        refresh_profiles_button = gr.Button("刷新角色列表", variant="secondary")
                        set_active_profile_button = gr.Button("设为默认角色", variant="secondary")
                    gr.Markdown("默认角色会在下次打开 Inference UI 时优先选中。")

                gr.Markdown("## 2. 参考音频")
                prompt_wav = gr.Audio(label="参考音频 / Prompt WAV", type="filepath")
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
                mode = gr.State(REAL_INFERENCE_MODE)
                run_button = gr.Button(value="生成语音", variant="primary", size="lg")

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
                    gr.Markdown("开启后可在生成结果中查看 DNSMOS、音色相似度、WER/CER 等验收指标。")
                    enable_metrics = gr.Checkbox(label="启用质量指标 / Enable metrics", value=False)
                    compute_dnsmos = gr.Checkbox(label="DNSMOS", value=True)
                    compute_speaker_sim = gr.Checkbox(label="Speaker Sim", value=True)
                    compute_wer = gr.Checkbox(label="WER / CER", value=False)

            with gr.Column(scale=5):
                gr.Markdown("## 生成结果", elem_classes=["vl-infer-result-heading"])
                output_audio = gr.Audio(label="生成音频 / Generated Audio", type="filepath")
                output_markdown = gr.Markdown(
                    value="尚未生成语音。",
                    elem_classes=["vl-infer-result-summary", "vl-infer-runtime"],
                )

                with gr.Accordion("诊断信息", open=False):
                    gr.Markdown(
                        "普通用户无需展开。这里集中保存 Profile Registry、原始路径、Developer command、stdout/stderr 和错误详情。",
                        elem_classes=["vl-infer-diagnostics-note"],
                    )
                    profile_json = gr.JSON(label="Profile Registry JSON", value=profile_initial_json)
                    inference_diagnostics_markdown = gr.Markdown("尚无推理诊断信息。")
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
            fn=run_user_inference_v3,
            inputs=run_inputs,
            outputs=run_outputs,
        )

    return demo


def parse_args():
    return legacy.parse_args()


def main() -> int:
    args = parse_args()
    print("====================================================")
    print("VoiceLab User Inference WebUI v3 (IF-UI-3)")
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
