from __future__ import annotations

"""Presentation-only helpers for the VoiceLab inference user interface.

This module consumes existing Profile Registry and UserInferenceResult data. It
never resolves checkpoints, mutates profiles, starts subprocesses, loads models,
or imports Gradio.

Normal-user formatters intentionally omit filesystem paths, artifact JSON paths,
developer commands, raw exception messages, and tracebacks. Those values remain
available in the underlying backend contracts for diagnostics.
"""

from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_PROFILE_CHOICE = "使用默认配置 / Use default config"


@dataclass(frozen=True)
class ProfilePresentation:
    profile_name: str
    selected: bool
    is_default_choice: bool
    is_active: bool
    ready_for_inference: bool
    stage1_ready: bool
    stage2_ready: bool
    status_icon: str
    status_label: str
    guidance: str


@dataclass(frozen=True)
class MetricPresentation:
    key: str
    label: str
    value_text: str
    state_text: str
    help_text: str


@dataclass(frozen=True)
class InferenceResultPresentation:
    status: str
    status_icon: str
    status_label: str
    profile_name: str
    total_seconds_text: str
    rtf_text: str
    metrics: tuple[MetricPresentation, ...]
    metrics_available: bool
    asr_text: str


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _obj_get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_items(scan_payload: Any) -> list[dict[str, Any]]:
    items = _as_dict(scan_payload).get("profiles") or []
    return [item for item in items if isinstance(item, dict)]


def _is_default_choice(value: Optional[str], default_choice: str) -> bool:
    text = _text(value)
    return not text or text == _text(default_choice)


def _matches_profile(item: dict[str, Any], selected_value: str) -> bool:
    selected = _text(selected_value)
    profile_name = _text(item.get("profile_name"))
    label = _text(item.get("label"))
    if selected in {profile_name, label}:
        return True
    return bool(profile_name and selected.startswith(profile_name + " |"))


def build_profile_presentation(
    scan_payload: Any,
    selected_value: Optional[str],
    *,
    default_choice: str = DEFAULT_PROFILE_CHOICE,
) -> ProfilePresentation:
    """Build compact selected-profile state without exposing checkpoint paths."""

    selected_text = _text(selected_value)
    if _is_default_choice(selected_text, default_choice):
        return ProfilePresentation(
            profile_name="系统默认配置",
            selected=True,
            is_default_choice=True,
            is_active=False,
            ready_for_inference=True,
            stage1_ready=True,
            stage2_ready=True,
            status_icon="✅",
            status_label="使用默认模型配置",
            guidance="可以直接生成语音；如需使用训练后的角色，请从角色列表选择。",
        )

    selected_item = next(
        (item for item in _profile_items(scan_payload) if _matches_profile(item, selected_text)),
        None,
    )
    if selected_item is None:
        return ProfilePresentation(
            profile_name=selected_text or "未选择声音角色",
            selected=bool(selected_text),
            is_default_choice=False,
            is_active=False,
            ready_for_inference=False,
            stage1_ready=False,
            stage2_ready=False,
            status_icon="⚠️",
            status_label="角色信息不可用",
            guidance="请刷新角色列表，或选择其他可用角色。",
        )

    stage1_ready = bool(selected_item.get("stage1_ready"))
    stage2_ready = bool(selected_item.get("stage2_ready"))
    ready = bool(selected_item.get("ready_for_inference"))
    profile_name = _text(selected_item.get("profile_name")) or selected_text

    if ready:
        status_icon = "✅"
        status_label = "可用于生成"
        guidance = "模型已准备完成，可以直接生成语音。"
    else:
        status_icon = "⚠️"
        status_label = "尚未就绪"
        missing = [
            label
            for label, flag in (("Stage1", stage1_ready), ("Stage2", stage2_ready))
            if not flag
        ]
        guidance = (
            "请先完成 " + " / ".join(missing) + " 模型准备，或选择其他已就绪角色。"
            if missing
            else "当前角色暂不可用于推理，请刷新角色状态或选择其他已就绪角色。"
        )

    return ProfilePresentation(
        profile_name=profile_name,
        selected=True,
        is_default_choice=False,
        is_active=bool(selected_item.get("is_active")),
        ready_for_inference=ready,
        stage1_ready=stage1_ready,
        stage2_ready=stage2_ready,
        status_icon=status_icon,
        status_label=status_label,
        guidance=guidance,
    )


def format_profile_presentation_markdown(
    scan_payload: Any,
    selected_value: Optional[str],
    *,
    default_choice: str = DEFAULT_PROFILE_CHOICE,
) -> str:
    view = build_profile_presentation(
        scan_payload,
        selected_value,
        default_choice=default_choice,
    )

    badges = [f"`{view.status_label}`"]
    if view.is_active:
        badges.append("`默认角色`")

    lines = [f"### {view.status_icon} {view.profile_name}", "", " ".join(badges)]
    if not view.is_default_choice:
        lines.extend(
            [
                "",
                f"- Stage1 模型：{'✅ 已就绪' if view.stage1_ready else '⬜ 未就绪'}",
                f"- Stage2 模型：{'✅ 已就绪' if view.stage2_ready else '⬜ 未就绪'}",
            ]
        )
    lines.extend(["", view.guidance])
    return "\n".join(lines)


def _metric_state(value: Optional[float], section: Any, *, unavailable: str) -> str:
    if value is not None:
        return "已计算"
    payload = _as_dict(section)
    if payload.get("error"):
        return "计算失败"
    if payload.get("enabled") is False:
        return "未启用"
    return unavailable


def _wer_metric_state(
    value: Optional[float],
    wer_section: Any,
    asr_section: Any,
) -> str:
    """Return WER/CER state with ASR failure taking precedence.

    WER/CER depend on ASR output.  If ASR itself failed, showing "未启用" from a
    secondary WER section is misleading even when that section contains
    ``enabled=False``.  A concrete ASR failure therefore wins over the generic
    disabled/unavailable state.  Raw ASR error text is never exposed here.
    """

    if value is not None:
        return "已计算"

    asr_payload = _as_dict(asr_section)
    if asr_payload.get("error"):
        return "识别失败"

    wer_payload = _as_dict(wer_section)
    if wer_payload.get("error"):
        return "计算失败"
    if wer_payload.get("enabled") is False:
        return "未启用"
    return "未计算"


def _number_text(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _error_rate_text(value: Optional[float]) -> str:
    # VoiceLab compute_error_rate returns edit_distance / reference_length.
    return "—" if value is None else f"{value * 100.0:.2f}%"


def _status_presentation(status: str) -> tuple[str, str]:
    return {
        "prepared": ("⬜", "已准备"),
        "running": ("⏳", "正在生成"),
        "succeeded": ("✅", "生成完成"),
        "failed": ("❌", "生成失败"),
        "cancelled": ("⏸", "已取消"),
        "not_implemented": ("⚠️", "当前功能不可用"),
    }.get(_text(status).lower(), ("⚠️", _text(status) or "状态未知"))


def build_inference_result_presentation(result: Any) -> InferenceResultPresentation:
    """Build user-facing result data from the stable result contract."""

    status = _text(_obj_get(result, "status"))
    status_icon, status_label = _status_presentation(status)
    profile_name = _text(_obj_get(result, "profile_name"))

    timing = _obj_get(result, "timing")
    total_seconds = _float_or_none(_obj_get(timing, "total_seconds"))
    timing_rtf = _float_or_none(_obj_get(timing, "rtf"))

    metrics = _obj_get(result, "metrics")
    extra = _as_dict(_obj_get(metrics, "extra", {})) if metrics is not None else {}
    dnsmos = _as_dict(extra.get("dnsmos"))
    speaker = _as_dict(extra.get("speaker_sim"))
    wer_section = _as_dict(extra.get("wer"))
    asr = _as_dict(extra.get("asr"))

    dnsmos_ovrl = _float_or_none(_obj_get(metrics, "dnsmos_ovrl")) if metrics is not None else None
    dnsmos_sig = _float_or_none(_obj_get(metrics, "dnsmos_sig")) if metrics is not None else None
    speaker_sim = _float_or_none(_obj_get(metrics, "speaker_sim")) if metrics is not None else None
    wer = _float_or_none(_obj_get(metrics, "wer")) if metrics is not None else None
    cer = _float_or_none(_obj_get(metrics, "cer")) if metrics is not None else None
    metric_rtf = _float_or_none(_obj_get(metrics, "rtf")) if metrics is not None else None

    rows = (
        MetricPresentation(
            "dnsmos_ovrl",
            "综合自然度（DNSMOS OVRL）",
            _number_text(dnsmos_ovrl),
            _metric_state(dnsmos_ovrl, dnsmos, unavailable="未计算"),
            "语音整体自然度与质量评分。",
        ),
        MetricPresentation(
            "dnsmos_sig",
            "语音信号质量（DNSMOS SIG）",
            _number_text(dnsmos_sig),
            _metric_state(dnsmos_sig, dnsmos, unavailable="未计算"),
            "更关注语音主体本身的清晰与失真情况。",
        ),
        MetricPresentation(
            "speaker_sim",
            "音色相似度（Speaker Sim）",
            _number_text(speaker_sim),
            _metric_state(speaker_sim, speaker, unavailable="未计算"),
            "生成语音与参考说话人的相似程度。",
        ),
        MetricPresentation(
            "wer",
            "词错误率（WER）",
            _error_rate_text(wer),
            _wer_metric_state(wer, wer_section, asr),
            "文本识别错误率，越低越好；中文场景当前实现使用字符代理。",
        ),
        MetricPresentation(
            "cer",
            "字符错误率（CER）",
            _error_rate_text(cer),
            _wer_metric_state(cer, wer_section, asr),
            "字符级文本识别错误率，越低越好。",
        ),
    )

    rtf = timing_rtf if timing_rtf is not None else metric_rtf
    return InferenceResultPresentation(
        status=status,
        status_icon=status_icon,
        status_label=status_label,
        profile_name=profile_name,
        total_seconds_text=f"{total_seconds:.2f} 秒" if total_seconds is not None else "—",
        rtf_text=f"{rtf:.3f}" if rtf is not None else "—",
        metrics=rows,
        metrics_available=metrics is not None,
        asr_text=_text(asr.get("text")),
    )


def format_inference_result_markdown(result: Any) -> str:
    """Render normal-user result text with no developer/raw diagnostic payload."""

    view = build_inference_result_presentation(result)
    lines = [f"## {view.status_icon} {view.status_label}", ""]

    if view.profile_name:
        lines.extend([f"**声音角色：** {view.profile_name}", ""])

    if view.status == "succeeded":
        lines.append("语音已生成，可以直接在上方播放器试听。")
        lines.extend(
            [
                "",
                "### 生成性能",
                "",
                f"- 总耗时：**{view.total_seconds_text}**",
                f"- 实时系数（RTF）：**{view.rtf_text}**",
            ]
        )

        if view.metrics_available:
            lines.extend(
                [
                    "",
                    "### 质量指标",
                    "",
                    "| 指标 | 结果 | 状态 |",
                    "|---|---:|---|",
                ]
            )
            for metric in view.metrics:
                lines.append(f"| {metric.label} | **{metric.value_text}** | {metric.state_text} |")
            lines.extend(["", "<details>", "<summary>指标说明</summary>", ""])
            for metric in view.metrics:
                lines.append(f"- **{metric.label}**：{metric.help_text}")
            lines.append("</details>")
        else:
            lines.extend(
                [
                    "",
                    "### 质量指标",
                    "",
                    "本次未生成质量指标。如需要验收 DNSMOS、音色相似度或 WER/CER，可在左侧「质量指标」中启用。",
                ]
            )

        if view.asr_text:
            lines.extend(["", "### 识别文本", "", view.asr_text])

    elif view.status == "failed":
        # Do not echo result.message/error.message/hint here: backend errors may
        # contain absolute paths, commands, or implementation details.
        lines.extend(
            [
                "本次语音生成未完成。",
                "",
                "**建议：** 检查参考音频、声音角色和高级设置后重试；详细错误信息已保留在本次推理结果的诊断数据中。",
            ]
        )
    elif view.status == "cancelled":
        lines.append("本次语音生成已取消，可以调整输入后重新生成。")
    elif view.status == "prepared":
        lines.append("推理请求已经准备完成。")
    else:
        lines.append("当前推理状态暂不可用，请检查设置后重试。")

    return "\n".join(lines)


def format_user_exception_markdown(
    *,
    title: str,
    message: str,
    guidance: str = "请检查输入和声音角色后重试。",
) -> str:
    """Render a controlled UI-boundary exception without traceback text."""

    return "\n".join(
        [
            f"## ❌ {title}",
            "",
            message,
            "",
            f"**建议：** {guidance}",
        ]
    )


__all__ = [
    "DEFAULT_PROFILE_CHOICE",
    "ProfilePresentation",
    "MetricPresentation",
    "InferenceResultPresentation",
    "build_profile_presentation",
    "format_profile_presentation_markdown",
    "build_inference_result_presentation",
    "format_inference_result_markdown",
    "format_user_exception_markdown",
]
