"""
VoiceLab user-edition Gradio inference UI.

中文说明：
    这是 VoiceLab 用户版推理 WebUI 的入口。
    它不直接 import GPT-SoVITS / Stage2 / vocoder / torch。
    真正推理通过：

        Gradio UI
            -> UserInferenceRequest
                -> UserInferenceService
                    -> DeveloperInferenceAdapter
                        -> scripts/infer_zeroshot_v641.py

    完成。

English notes:
    This is the Gradio WebUI entrypoint for VoiceLab user-edition inference.
    It keeps the UI layer lightweight. Heavy inference is delegated to
    UserInferenceService and DeveloperInferenceAdapter.

Execution modes:
执行模式：
    - dry-run:
        Only prepares config/output directory and writes JSON files.
        只准备配置和输出目录，不加载大模型。

    - real inference:
        Calls existing developer inference script through subprocess.
        通过 subprocess 调用现有稳定开发者推理脚本。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import gradio as gr


# -----------------------------------------------------------------------------
# Project bootstrap
# 项目路径引导
# -----------------------------------------------------------------------------


def _bootstrap_project_path() -> Path:
    """Add repository root to sys.path and return it.

    中文说明：
        本脚本位于 scripts/webui_user_infer.py。
        因此项目根目录是 scripts/ 的父目录。
    """

    project_root = Path(__file__).resolve(strict=False).parents[1]
    project_root_text = str(project_root)

    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    return project_root


PROJECT_ROOT = _bootstrap_project_path()


try:
    from src.voicelab_user.inference.service import (
        UserInferenceRequest,
        create_user_inference_service,
    )
    from src.voicelab_user.inference.profile_registry import (
        format_profile_registry_markdown,
        profile_choices,
        resolve_inference_profile,
        scan_user_profiles,
        set_active_profile,
    )
except Exception as _import_exc:  # noqa: BLE001 - shown inside UI.
    UserInferenceRequest = None
    create_user_inference_service = None
    format_profile_registry_markdown = None
    profile_choices = None
    resolve_inference_profile = None
    scan_user_profiles = None
    set_active_profile = None
    IMPORT_ERROR = _import_exc
else:
    IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Small helpers
# 小工具函数
# -----------------------------------------------------------------------------


NO_PROFILE_CHOICE = "使用默认配置 / Use default config"


def _optional_text(value: Optional[str]) -> Optional[str]:
    """Normalize optional text from UI.

    规范化 UI 中的可选文本。
    """

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _bool_from_mode(mode: str) -> bool:
    """Return dry_run bool from UI mode string.

    根据 UI 模式字符串返回 dry_run。
    """

    return str(mode).strip().lower().startswith("dry")


def _normalize_timeout_seconds(value: Optional[float]) -> Optional[float]:
    """Normalize timeout value from Gradio UI."""

    if value is None:
        return None

    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None

    if timeout <= 0:
        return None

    return timeout


def _normalize_seed_value(
    use_fixed_seed: bool,
    seed_value: Optional[float],
) -> Optional[int]:
    """Normalize seed controls from Gradio UI."""

    if not bool(use_fixed_seed):
        return None

    if seed_value is None:
        return 0

    try:
        seed_int = int(seed_value)
    except (TypeError, ValueError):
        return 0

    if seed_int < 0:
        return 0

    return seed_int


def _format_metric_number(value: Optional[float]) -> str:
    """Format metric number for markdown display."""

    if value is None:
        return "`None`"

    try:
        return f"`{float(value):.4f}`"
    except (TypeError, ValueError):
        return f"`{value}`"


def _metric_reason(section: Any, *, missing_reason: str = "not computed") -> str:
    """Return a short reason for missing metric values."""

    if not isinstance(section, dict):
        return missing_reason

    error = section.get("error")
    if error:
        return f"failed: {error}"

    enabled = section.get("enabled")
    if enabled is False:
        return "disabled"

    return missing_reason


def _format_metric_with_reason(
    value: Optional[float],
    section: Any,
    *,
    missing_reason: str = "not computed",
) -> str:
    """Format metric value and include reason when value is None."""

    if value is not None:
        return _format_metric_number(value)

    reason = _metric_reason(section, missing_reason=missing_reason)
    return f"`None` ({reason})"


def _extract_metrics_extra(result) -> dict[str, Any]:
    """Extract raw metrics dict from InferenceMetricSummary.extra."""

    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return {}

    extra = getattr(metrics, "extra", None)
    return extra if isinstance(extra, dict) else {}


def _file_to_path(value: Any) -> Optional[str]:
    """Extract path text from Gradio file/audio input."""

    if value is None:
        return None

    if isinstance(value, (str, Path)):
        return str(value)

    if isinstance(value, dict):
        for key in ("path", "name"):
            item = value.get(key)
            if item:
                return str(item)

    name = getattr(value, "name", None)
    if name:
        return str(name)

    return str(value)


# -----------------------------------------------------------------------------
# Profile registry helpers
# Profile 注册表工具
# -----------------------------------------------------------------------------


def _safe_scan_profiles() -> dict[str, Any]:
    if scan_user_profiles is None:
        return {
            "profiles": [],
            "num_profiles": 0,
            "error": "profile_registry import failed",
        }
    try:
        return scan_user_profiles(require_ready=False)
    except Exception as exc:  # noqa: BLE001 - shown in UI.
        return {
            "profiles": [],
            "num_profiles": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _profile_dropdown_choices(scan_payload: Optional[dict[str, Any]] = None) -> tuple[list[str], str]:
    payload = scan_payload or _safe_scan_profiles()
    choices = [NO_PROFILE_CHOICE]

    if profile_choices is not None:
        try:
            choices.extend(profile_choices(payload))
        except Exception:
            for item in payload.get("profiles") or []:
                label = str(item.get("label") or item.get("profile_name") or "")
                if label:
                    choices.append(label)
    else:
        for item in payload.get("profiles") or []:
            label = str(item.get("label") or item.get("profile_name") or "")
            if label:
                choices.append(label)

    # Prefer the active profile if one is already set; otherwise prefer latest READY.
    value = NO_PROFILE_CHOICE
    for item in payload.get("profiles") or []:
        if item.get("is_active"):
            label = str(item.get("label") or item.get("profile_name") or "")
            if label in choices:
                return choices, label
    for item in payload.get("profiles") or []:
        if item.get("ready_for_inference"):
            label = str(item.get("label") or item.get("profile_name") or "")
            if label in choices:
                value = label
                break
    return choices, value


def _format_profile_markdown(scan_payload: Optional[dict[str, Any]] = None) -> str:
    payload = scan_payload or _safe_scan_profiles()
    if format_profile_registry_markdown is None:
        return "### 声音角色 / Profile\n\nprofile_registry 导入失败。"
    try:
        return format_profile_registry_markdown(payload)
    except Exception as exc:  # noqa: BLE001 - shown in UI.
        return f"### 声音角色 / Profile\n\nProfile 扫描失败：`{type(exc).__name__}: {exc}`"


def _initial_profile_ui_state() -> tuple[list[str], str, str, dict[str, Any]]:
    payload = _safe_scan_profiles()
    choices, value = _profile_dropdown_choices(payload)
    return choices, value, _format_profile_markdown(payload), payload


def _is_default_profile_choice(value: Optional[str]) -> bool:
    text = str(value or "").strip()
    return not text or text == NO_PROFILE_CHOICE


def _resolve_profile_checkpoint_overrides(
    profile_choice_value: Optional[str],
    stage1_ckpt_override: Optional[str],
    stage2_ckpt_override: Optional[str],
) -> tuple[Optional[str], Optional[str], dict[str, Any], str]:
    """Resolve checkpoint overrides from selected profile plus manual advanced overrides.

    中文说明：
        用户版默认从 profile_manifest.json 自动解析 stage1/stage2 best_model.pt。
        高级选项中的 checkpoint override 仍保留，并且优先级高于 profile。
    """

    manual_stage1 = _optional_text(stage1_ckpt_override)
    manual_stage2 = _optional_text(stage2_ckpt_override)

    if _is_default_profile_choice(profile_choice_value):
        lines = [
            "### Profile",
            "",
            "- 使用默认推理配置。",
            "- 如果高级 checkpoint override 为空，则走 `configs/user_inference_default.yaml` 中的默认 profile。",
        ]
        return manual_stage1, manual_stage2, {}, "\n".join(lines)

    if resolve_inference_profile is None:
        raise RuntimeError("profile_registry is not available.")

    # If the user manually provides a Stage2 override, allow loading a profile even
    # when its own stage2 best checkpoint is missing. Otherwise require Stage2.
    require_stage2 = manual_stage2 is None
    resolved_profile = resolve_inference_profile(
        str(profile_choice_value),
        require_stage2=require_stage2,
    )

    profile_stage1 = resolved_profile.get("stage1_ckpt")
    profile_stage2 = resolved_profile.get("stage2_ckpt")

    final_stage1 = manual_stage1 or (str(profile_stage1) if profile_stage1 else None)
    final_stage2 = manual_stage2 or (str(profile_stage2) if profile_stage2 else None)

    lines = [
        "### Profile",
        "",
        f"- 当前声音角色：`{resolved_profile.get('profile_name')}`",
        f"- ready_for_inference：`{resolved_profile.get('ready_for_inference')}`",
        f"- Stage1 checkpoint：`{final_stage1 or 'default_gpt_sovits_v2'}`",
        f"- Stage2 checkpoint：`{final_stage2 or ''}`",
    ]

    if manual_stage1:
        lines.append("- Stage1 使用高级 override，覆盖 profile。")
    if manual_stage2:
        lines.append("- Stage2 使用高级 override，覆盖 profile。")

    return final_stage1, final_stage2, resolved_profile, "\n".join(lines)


def refresh_profiles_click() -> tuple[Any, str, dict[str, Any]]:
    payload = _safe_scan_profiles()
    choices, value = _profile_dropdown_choices(payload)
    return gr.update(choices=choices, value=value), _format_profile_markdown(payload), payload


def set_active_profile_click(profile_choice_value: str) -> tuple[Any, str, dict[str, Any]]:
    if _is_default_profile_choice(profile_choice_value):
        payload = _safe_scan_profiles()
        choices, value = _profile_dropdown_choices(payload)
        md = _format_profile_markdown(payload) + "\n\n> 未设置 active profile：当前选择为默认配置。"
        return gr.update(choices=choices, value=value), md, payload

    if set_active_profile is None:
        payload = _safe_scan_profiles()
        choices, value = _profile_dropdown_choices(payload)
        md = _format_profile_markdown(payload) + "\n\n> 设置 active profile 失败：profile_registry 不可用。"
        return gr.update(choices=choices, value=value), md, payload

    try:
        active_payload = set_active_profile(profile_choice_value)
    except Exception as exc:  # noqa: BLE001 - shown in UI.
        payload = _safe_scan_profiles()
        choices, value = _profile_dropdown_choices(payload)
        md = _format_profile_markdown(payload) + f"\n\n> 设置 active profile 失败：`{type(exc).__name__}: {exc}`"
        return gr.update(choices=choices, value=value), md, payload

    payload = _safe_scan_profiles()
    choices, value = _profile_dropdown_choices(payload)
    md = _format_profile_markdown(payload) + f"\n\n> 已设为当前声音角色：`{active_payload.get('active_profile_name')}`"
    return gr.update(choices=choices, value=value), md, payload


# -----------------------------------------------------------------------------
# Result formatting
# 结果格式化
# -----------------------------------------------------------------------------


def _status_markdown(result) -> str:
    """Build compact markdown summary for UI."""

    lines: list[str] = []

    lines.append(f"## 状态 / Status: `{result.status}`")
    lines.append("")
    lines.append(f"**Message:** {result.message}")
    lines.append("")
    lines.append(f"**Run directory:** `{result.artifacts.run_dir}`")

    if result.artifacts.result_json_path is not None:
        lines.append(f"**Result JSON:** `{result.artifacts.result_json_path}`")

    if result.artifacts.request_json_path is not None:
        lines.append(f"**User request JSON:** `{result.artifacts.request_json_path}`")

    if result.artifacts.internal_summary_json_path is not None:
        lines.append(f"**Internal summary JSON:** `{result.artifacts.internal_summary_json_path}`")

    if result.artifacts.output_wav_path is not None:
        lines.append(f"**Output WAV:** `{result.artifacts.output_wav_path}`")

    if result.artifacts.output_peaknorm_wav_path is not None:
        lines.append(f"**Peak-normalized WAV:** `{result.artifacts.output_peaknorm_wav_path}`")

    if result.artifacts.metrics_json_path is not None:
        lines.append(f"**Metrics JSON:** `{result.artifacts.metrics_json_path}`")

    if result.timing is not None:
        timing_items = []
        if result.timing.total_seconds is not None:
            timing_items.append(f"total_seconds={result.timing.total_seconds}")
        if result.timing.rtf is not None:
            timing_items.append(f"rtf={result.timing.rtf}")
        if timing_items:
            lines.append("")
            lines.append("### Timing")
            lines.append("`" + ", ".join(timing_items) + "`")

    if result.metrics is not None:
        metrics_extra = _extract_metrics_extra(result)

        dnsmos_info = metrics_extra.get("dnsmos")
        speaker_info = metrics_extra.get("speaker_sim")
        wer_info = metrics_extra.get("wer")
        asr_info = metrics_extra.get("asr")

        lines.append("")
        lines.append("### Metrics")
        lines.append(
            "- DNSMOS OVRL: "
            + _format_metric_with_reason(
                result.metrics.dnsmos_ovrl,
                dnsmos_info,
                missing_reason="DNSMOS not computed",
            )
        )
        lines.append(
            "- DNSMOS SIG: "
            + _format_metric_with_reason(
                result.metrics.dnsmos_sig,
                dnsmos_info,
                missing_reason="DNSMOS not computed",
            )
        )
        lines.append(
            "- Speaker Sim: "
            + _format_metric_with_reason(
                result.metrics.speaker_sim,
                speaker_info,
                missing_reason="speaker similarity not computed",
            )
        )

        wer_missing_reason = "WER/CER disabled or not computed"
        if isinstance(asr_info, dict) and asr_info.get("error"):
            wer_missing_reason = f"ASR failed: {asr_info.get('error')}"
        elif wer_info is None:
            wer_missing_reason = "WER/CER disabled"

        lines.append(
            "- WER: "
            + _format_metric_with_reason(
                result.metrics.wer,
                wer_info,
                missing_reason=wer_missing_reason,
            )
        )
        lines.append(
            "- CER: "
            + _format_metric_with_reason(
                result.metrics.cer,
                wer_info,
                missing_reason=wer_missing_reason,
            )
        )

        if isinstance(asr_info, dict):
            asr_text = str(asr_info.get("text", "")).strip()
            asr_error = asr_info.get("error")

            if asr_text:
                lines.append("")
                lines.append("### ASR Transcript")
                lines.append(f"`{asr_text}`")

            if asr_error:
                lines.append("")
                lines.append("### ASR Error")
                lines.append("```text")
                lines.append(str(asr_error))
                lines.append("```")

    if result.error is not None:
        lines.append("")
        lines.append("### Error")
        lines.append(f"- Type: `{result.error.error_type}`")
        lines.append(f"- Stage: `{result.error.stage}`")
        lines.append(f"- Hint: {result.error.hint}")
        lines.append("")
        lines.append("```text")
        lines.append(result.error.message)
        lines.append("```")

    developer_command_path = (result.extra or {}).get("developer_command_json_path")
    if developer_command_path:
        lines.append("")
        lines.append(f"**Developer command JSON:** `{developer_command_path}`")

    developer_summary_path = (result.extra or {}).get("developer_summary_json_path")
    if developer_summary_path:
        lines.append(f"**Developer summary JSON:** `{developer_summary_path}`")

    return "\n".join(lines)


def _result_output_audio_path(result) -> Optional[str]:
    """Return best audio output path for Gradio Audio component."""

    if result.artifacts.output_wav_path is not None:
        return str(result.artifacts.output_wav_path)

    if result.artifacts.output_peaknorm_wav_path is not None:
        return str(result.artifacts.output_peaknorm_wav_path)

    return None


# -----------------------------------------------------------------------------
# Inference callback
# 推理回调
# -----------------------------------------------------------------------------


def run_user_inference(
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

    """Run user inference from Gradio inputs.

    从 Gradio 输入执行用户版推理。
    """

    if IMPORT_ERROR is not None:
        message = (
            "## Import Error\n\n"
            "无法导入 VoiceLab 用户版推理服务。\n\n"
            "```text\n"
            f"{IMPORT_ERROR!r}\n"
            "```"
        )
        return message, None

    prompt_wav_path = _file_to_path(prompt_wav)
    if not prompt_wav_path:
        return "## 输入错误\n\n请先上传或选择参考音频。", None

    prompt_text = str(prompt_text or "").strip()
    target_text = str(target_text or "").strip()

    if not prompt_text:
        return "## 输入错误\n\n请输入参考音频文本 prompt_text。", None

    if not target_text:
        return "## 输入错误\n\n请输入需要合成的目标文本 target_text。", None

    dry_run = _bool_from_mode(mode)

    seed_int = _normalize_seed_value(use_fixed_seed, seed_value)
    top_k_int = int(top_k) if top_k is not None else None

    try:
        resolved_stage1_ckpt, resolved_stage2_ckpt, resolved_profile, profile_md = _resolve_profile_checkpoint_overrides(
            profile_choice_value,
            stage1_ckpt,
            stage2_ckpt,
        )

        request = UserInferenceRequest(
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            target_text=target_text,
            output_dir=_optional_text(output_dir),
            stage1_ckpt=resolved_stage1_ckpt,
            stage2_ckpt=resolved_stage2_ckpt,
            device=_optional_text(device),
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

        normalized_timeout_seconds = _normalize_timeout_seconds(timeout_seconds)

        service = create_user_inference_service(
            project_root=str(PROJECT_ROOT),
            python_executable=None,
            inference_timeout_seconds=normalized_timeout_seconds,
        )

        result = service.run(request, dry_run=dry_run)
        result_md = _status_markdown(result)
        if resolved_profile:
            result_md = profile_md + "\n\n---\n\n" + result_md
        return result_md, _result_output_audio_path(result)

    except Exception as exc:  # noqa: BLE001 - UI boundary.
        tb = traceback.format_exc()
        message = (
            "## 推理服务异常 / Inference Service Error\n\n"
            f"**Error:** `{exc.__class__.__name__}`\n\n"
            "```text\n"
            f"{tb}\n"
            "```"
        )
        return message, None


# -----------------------------------------------------------------------------
# UI construction
# UI 构建
# -----------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    """Build Gradio demo.

    构建 Gradio 界面。
    """

    profile_initial_choices, profile_initial_value, profile_initial_md, profile_initial_json = _initial_profile_ui_state()

    with gr.Blocks(title="VoiceLab User Inference") as demo:
        gr.Markdown(
            """
# VoiceLab 用户版推理 UI

这是 VoiceLab 用户版推理 WebUI。

- 默认使用 **Dry-run**，只检查配置并生成输出目录，不加载模型。
- 选择 **Run real inference** 后，才会通过 adapter 调用现有稳定推理脚本。
- 第一版暂时只面向中文推理链。
- 声音角色来自 `user_profiles/*/profile_manifest.json`；Training UI 生成 User Profile 后，Inference UI 会自动扫描并使用对应 best checkpoint。
            """.strip()
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("声音角色 / Voice Profile", open=True):
                    profile_choice = gr.Dropdown(
                        label="声音角色 / Profile",
                        choices=profile_initial_choices,
                        value=profile_initial_value,
                        info="默认选择当前 active profile；没有 active 时选择最新 READY profile。",
                    )
                    with gr.Row():
                        refresh_profiles_button = gr.Button("刷新 Profile 列表", variant="secondary")
                        set_active_profile_button = gr.Button("设为当前角色", variant="secondary")
                    profile_markdown = gr.Markdown(profile_initial_md)
                    with gr.Accordion("Profile Registry JSON", open=False):
                        profile_json = gr.JSON(label="profile registry", value=profile_initial_json)

                prompt_wav = gr.Audio(
                    label="参考音频 / Prompt WAV",
                    type="filepath",
                )
                prompt_text = gr.Textbox(
                    label="参考音频文本 / Prompt Text",
                    lines=3,
                    placeholder="请输入参考音频对应文本，例如：这是参考音频。",
                )
                target_text = gr.Textbox(
                    label="目标文本 / Target Text",
                    lines=5,
                    placeholder="请输入需要合成的文本。",
                )

                mode = gr.Radio(
                    label="执行模式 / Execution Mode",
                    choices=[
                        "Dry-run only",
                        "Run real inference",
                    ],
                    value="Dry-run only",
                )

                run_button = gr.Button(
                    value="开始 / Run",
                    variant="primary",
                )

            with gr.Column(scale=1):
                output_markdown = gr.Markdown(
                    label="结果 / Result",
                    value="等待运行。",
                )
                output_audio = gr.Audio(
                    label="生成音频 / Generated Audio",
                    type="filepath",
                )

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

        refresh_profiles_button.click(
            fn=refresh_profiles_click,
            inputs=[],
            outputs=[
                profile_choice,
                profile_markdown,
                profile_json,
            ],
        )
        set_active_profile_button.click(
            fn=set_active_profile_click,
            inputs=[profile_choice],
            outputs=[
                profile_choice,
                profile_markdown,
                profile_json,
            ],
        )

        run_button.click(
            fn=run_user_inference,
            inputs=[
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
            ],
            outputs=[
                output_markdown,
                output_audio,
            ],
        )

    return demo


# -----------------------------------------------------------------------------
# CLI for launching UI
# 启动 UI 的命令行入口
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Resonastra inference Gradio UI.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link. Not recommended for offline user package.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open browser automatically.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("====================================================")
    print("VoiceLab User Inference WebUI")
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
