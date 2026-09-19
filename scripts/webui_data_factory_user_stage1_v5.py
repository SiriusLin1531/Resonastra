from __future__ import annotations

"""VoiceLab DataFactory user UI v5 compatibility shell.

This shell keeps the accepted v4 business/event chain while applying the
post-freeze user-trial fixes and release-hardening lifecycle guards.
"""

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from scripts import webui_data_factory_user_stage1_v4 as v4
from src.voicelab_user.data_factory.proofreader_launcher import (
    launch_proofreader_resource_safe,
)
from src.voicelab_user.data_factory.proofreader_runtime import (
    inspect_proofreader_runtime,
    stop_proofreader_runtime,
)

legacy = v4.legacy
base = v4.base
DF_UI_V5_CSS = v4.DF_UI_V4_CSS
PROOFREADER_REFRESH_INTERVAL_SECONDS = 5.0

_ORIGINAL_INSPECT_WORK_DIR_UNIFIED_CLICK = legacy.inspect_work_dir_unified_click
_ORIGINAL_LAUNCH_PROOFREAD_CLICK = base.launch_proofread_click
_ORIGINAL_REBUILD_AFTER_PROOFREAD_CLICK = base.rebuild_after_proofread_click
_ORIGINAL_SKIP_PROOFREAD_CLICK = base.skip_proofread_click

# RH-1A separation rule: generic DataFactory Stop owns build tasks only.
# The proofreader is stopped exclusively by its dedicated lifecycle control.
base.DATA_FACTORY_PROCESS_SCRIPT_NAMES.discard("launch_proofread.py")

# RH-1B process ownership rule: the generic Stop control must recognize both
# the canonical Stage1 builder and the release-hardened wrapper. The wrapper
# launches the canonical builder as a child, so taskkill /T on either matched
# parent process keeps Stop behavior compatible with the v4 lifecycle.
base.DATA_FACTORY_PROCESS_SCRIPT_NAMES.update(
    {
        "build_stage1_fewshot_dataset.py",
        "build_stage1_fewshot_dataset_hardened.py",
    }
)

_INJECTED_CLOSE_PROOFREAD_BUTTON: Any = None


def inspect_current_selection_unified_click(work_dir_output: str, work_dir: str, speaker_name: str):
    del work_dir_output
    return _ORIGINAL_INSPECT_WORK_DIR_UNIFIED_CLICK("", str(work_dir or ""), str(speaker_name or ""))


def _resolve_active_work_dir(work_dir_output: Any, work_dir: Any, speaker_name: Any) -> Path:
    return base._resolve_active_work_dir(
        work_dir_output=str(work_dir_output or ""),
        work_dir=str(work_dir or ""),
        speaker_name=str(speaker_name or ""),
    )


def _proofreader_user_markdown(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "").strip().lower()
    url = str(payload.get("url") or payload.get("proofread_url") or "").strip()
    if status in {"running", "starting", "launched"}:
        lines = ["### 人工校对", "", "🟢 人工校对器正在运行。"]
        if url:
            lines.extend(["", f"校对器地址：{url}"])
        lines.extend(["", "请在校对器中保存修改。完成后回到本页面点击 **已保存并完成校对**。"])
        return "\n".join(lines)
    if status == "failed":
        return "\n".join(["### 人工校对", "", "⚠️ 人工校对器未能正常关闭或启动。", "请展开右侧 **诊断信息** 查看详细信息。"])
    if status == "exited":
        return "\n".join(["### 人工校对", "", "⚪ 上一次人工校对器已经退出。", "如需继续修改，请重新启动人工校对器。"])
    if status == "not_loaded":
        return "\n".join(["### 人工校对", "", "⚪ 尚未载入数据准备项目。"])
    return "\n".join(["### 人工校对", "", "⚪ 人工校对器当前未运行。"])


def _blank_proofreader_selection(work_dir_output: Any, work_dir: Any, speaker_name: Any) -> bool:
    return not any(str(value or "").strip() for value in (work_dir_output, work_dir, speaker_name))


def refresh_proofreader_user_click(work_dir_output: str, work_dir: str, speaker_name: str):
    if _blank_proofreader_selection(work_dir_output, work_dir, speaker_name):
        payload = {"status": "not_loaded"}
        return _proofreader_user_markdown(payload), payload
    try:
        active_work_dir = _resolve_active_work_dir(work_dir_output, work_dir, speaker_name)
        payload = inspect_proofreader_runtime(active_work_dir)
        return _proofreader_user_markdown(payload), payload
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
        return _proofreader_user_markdown(payload), payload


def refresh_current_selection_proofreader_click(work_dir: str, speaker_name: str):
    return refresh_proofreader_user_click("", work_dir, speaker_name)


def stop_proofreader_user_click(work_dir_output: str, work_dir: str, speaker_name: str):
    if _blank_proofreader_selection(work_dir_output, work_dir, speaker_name):
        payload = {"status": "not_loaded"}
        return _proofreader_user_markdown(payload), payload
    try:
        active_work_dir = _resolve_active_work_dir(work_dir_output, work_dir, speaker_name)
        payload = stop_proofreader_runtime(active_work_dir)
        return _proofreader_user_markdown(payload), payload
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
        return _proofreader_user_markdown(payload), payload


def launch_proofread_user_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    proofread_port: Any,
    proofread_g_batch: Any,
    proofread_overwrite_backup: bool,
):
    try:
        active_work_dir = _resolve_active_work_dir(work_dir_output, work_dir, speaker_name)
        current = inspect_proofreader_runtime(active_work_dir)
        if bool(current.get("process_alive")):
            current["duplicate_launch_prevented"] = True
            return _proofreader_user_markdown(current), current, str(active_work_dir)

        pipeline = base.create_data_factory_user_pipeline_service()
        payload = launch_proofreader_resource_safe(
            pipeline.adapter,
            active_work_dir,
            webui_port_subfix=base._as_int(proofread_port, 9871),
            g_batch=base._as_int(proofread_g_batch, 10),
            overwrite_backup=bool(proofread_overwrite_backup),
        )
        diagnostics = dict(payload)
        diagnostics["status"] = "starting"
        diagnostics["url"] = payload.get("url") or payload.get("proofread_url")
        return _proofreader_user_markdown(diagnostics), diagnostics, str(active_work_dir)
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
        return _proofreader_user_markdown(payload), payload, work_dir_output or work_dir or ""


def _stop_proofreader_for_transition(active_work_dir: Path) -> dict[str, Any]:
    result = stop_proofreader_runtime(active_work_dir)
    if bool(result.get("still_running")) or str(result.get("status")) == "failed":
        raise RuntimeError(
            "人工校对器仍在运行，未继续修改 corrected manifest。"
            "请先点击“关闭人工校对器”，确认关闭后再重试。"
        )
    return result


def rebuild_after_proofread_user_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
    duration_tolerance_sec: Any,
    rebuild_timeout_seconds: Any,
):
    active_work_dir = _resolve_active_work_dir(work_dir_output, work_dir, speaker_name)
    try:
        stop_result = _stop_proofreader_for_transition(active_work_dir)
    except Exception as exc:  # noqa: BLE001
        status_md, dev_json, output_md, step_md = base._format_exception(exc, title="人工校对器尚未关闭")
        if isinstance(dev_json, dict):
            dev_json["proofreader_runtime"] = inspect_proofreader_runtime(active_work_dir)
        return status_md, dev_json, output_md, step_md, str(active_work_dir)

    result = _ORIGINAL_REBUILD_AFTER_PROOFREAD_CLICK(
        work_dir_output,
        work_dir,
        speaker_name,
        duration_tolerance_sec,
        rebuild_timeout_seconds,
    )
    status_md, dev_json, output_md, step_md, resolved_output = result
    diagnostics = dev_json if isinstance(dev_json, dict) else {"base_result": dev_json}
    diagnostics["proofreader_stop"] = stop_result
    return status_md, diagnostics, output_md, step_md, resolved_output


def skip_proofread_user_click(work_dir_output: str, work_dir: str, speaker_name: str):
    active_work_dir = _resolve_active_work_dir(work_dir_output, work_dir, speaker_name)
    try:
        stop_result = _stop_proofreader_for_transition(active_work_dir)
    except Exception as exc:  # noqa: BLE001
        status_md, dev_json, output_md, step_md = base._format_exception(exc, title="人工校对器尚未关闭")
        if isinstance(dev_json, dict):
            dev_json["proofreader_runtime"] = inspect_proofreader_runtime(active_work_dir)
        return status_md, dev_json, output_md, step_md, str(active_work_dir)

    result = _ORIGINAL_SKIP_PROOFREAD_CLICK(work_dir_output, work_dir, speaker_name)
    status_md, dev_json, output_md, step_md, resolved_output = result
    diagnostics = dev_json if isinstance(dev_json, dict) else {"base_result": dev_json}
    diagnostics["proofreader_stop"] = stop_result
    return status_md, diagnostics, output_md, step_md, resolved_output


def _component_values(demo: gr.Blocks) -> list[Any]:
    blocks = getattr(demo, "blocks", {})
    return list(blocks.values()) if isinstance(blocks, dict) else []


def _find_component(demo: gr.Blocks, *, label: str | None = None, value: str | None = None, elem_class: str | None = None):
    for component in _component_values(demo):
        if label is not None and str(getattr(component, "label", "") or "") != label:
            continue
        if value is not None and str(getattr(component, "value", "") or "") != value:
            continue
        if elem_class is not None:
            classes = getattr(component, "elem_classes", None) or []
            if elem_class not in classes:
                continue
        return component
    return None


def _clean_development_defaults(demo: gr.Blocks) -> None:
    for component in _component_values(demo):
        label = str(getattr(component, "label", "") or "")
        value = getattr(component, "value", None)
        if label == "说话人 / 角色名" and not str(value or "").strip():
            try:
                component.placeholder = "请输入声音角色名称，例如 Character_A"
            except Exception:
                pass
        if label == "人工校对启动信息":
            try:
                component.label = "人工校对器诊断"
                component.visible = True
            except Exception:
                pass
        if label == "ASR 后端":
            # VoiceLab User Edition v1.0 exposes only Chinese (zh).  The
            # accepted offline release path is therefore auto -> local FunASR.
            # Hide unsupported/unvalidated branches from clean-machine users.
            try:
                component.choices = ["auto"]
                component.value = "auto"
                component.interactive = False
                component.info = "VoiceLab v1.0 中文数据准备固定使用本地离线 FunASR。"
            except Exception:
                pass
        if label == "ASR 模型大小":
            try:
                component.value = "large"
                component.interactive = False
            except Exception:
                pass
        if label == "ASR 精度":
            try:
                component.value = "float32"
                component.interactive = False
            except Exception:
                pass
        if str(value or "").strip() == "启动人工校对器":
            try:
                component.value = "启动 / 打开人工校对器"
            except Exception:
                pass
        if str(value or "").strip() == "我已完成校对":
            try:
                component.value = "已保存并完成校对"
            except Exception:
                pass


def _build_button_factory(original_button_type):
    def button_factory(*args: Any, **kwargs: Any):
        global _INJECTED_CLOSE_PROOFREAD_BUTTON
        component = original_button_type(*args, **kwargs)
        requested_value = args[0] if args else kwargs.get("value")
        if str(requested_value or "") == "启动人工校对器":
            _INJECTED_CLOSE_PROOFREAD_BUTTON = original_button_type("关闭人工校对器", variant="stop")
        return component
    return button_factory


def _attach_proofreader_refresh_events(demo: gr.Blocks) -> None:
    close_button = _INJECTED_CLOSE_PROOFREAD_BUTTON
    load_button = _find_component(demo, value="载入/扫描工作目录")
    manual_refresh_button = _find_component(demo, value="立即刷新")
    work_dir_output = _find_component(demo, label="当前工作目录")
    work_dir = _find_component(demo, label="工作目录（可选）")
    speaker_name = _find_component(demo, label="说话人 / 角色名")
    proofread_status = _find_component(demo, elem_class="vl-proofread-status")
    proofread_json = _find_component(demo, label="人工校对器诊断")

    required = [work_dir_output, work_dir, speaker_name, proofread_status, proofread_json]
    if any(component is None for component in required):
        raise RuntimeError("RH-1A proofreader UI components could not be resolved.")
    if close_button is None:
        raise RuntimeError("RH-1A close proofreader button was not injected.")

    proofreader_outputs = [proofread_status, proofread_json]
    proofreader_inputs = [work_dir_output, work_dir, speaker_name]

    # Gradio event registration must occur while a Blocks context is active,
    # even when all referenced components were already constructed by v4.
    with demo:
        close_button.click(
            fn=stop_proofreader_user_click,
            inputs=proofreader_inputs,
            outputs=proofreader_outputs,
            queue=False,
        )
        if load_button is not None:
            load_button.click(
                fn=refresh_current_selection_proofreader_click,
                inputs=[work_dir, speaker_name],
                outputs=proofreader_outputs,
                queue=False,
            )
        if manual_refresh_button is not None:
            manual_refresh_button.click(
                fn=refresh_proofreader_user_click,
                inputs=proofreader_inputs,
                outputs=proofreader_outputs,
                queue=False,
            )

        load_method = getattr(demo, "load", None)
        if callable(load_method):
            load_method(
                fn=refresh_proofreader_user_click,
                inputs=proofreader_inputs,
                outputs=proofreader_outputs,
                queue=False,
            )

        timer_type = getattr(gr, "Timer", None)
        if timer_type is not None:
            proofreader_timer = timer_type(
                value=PROOFREADER_REFRESH_INTERVAL_SECONDS,
                active=True,
            )
            proofreader_timer.tick(
                fn=refresh_proofreader_user_click,
                inputs=proofreader_inputs,
                outputs=proofreader_outputs,
                queue=False,
            )


def build_demo() -> gr.Blocks:
    global _INJECTED_CLOSE_PROOFREAD_BUTTON
    _INJECTED_CLOSE_PROOFREAD_BUTTON = None

    original_button_type = gr.Button
    legacy.inspect_work_dir_unified_click = inspect_current_selection_unified_click
    base.launch_proofread_click = launch_proofread_user_click
    base.rebuild_after_proofread_click = rebuild_after_proofread_user_click
    base.skip_proofread_click = skip_proofread_user_click
    gr.Button = _build_button_factory(original_button_type)
    try:
        demo = v4.create_demo()
    finally:
        gr.Button = original_button_type
        legacy.inspect_work_dir_unified_click = _ORIGINAL_INSPECT_WORK_DIR_UNIFIED_CLICK
        base.launch_proofread_click = _ORIGINAL_LAUNCH_PROOFREAD_CLICK
        base.rebuild_after_proofread_click = _ORIGINAL_REBUILD_AFTER_PROOFREAD_CLICK
        base.skip_proofread_click = _ORIGINAL_SKIP_PROOFREAD_CLICK

    _clean_development_defaults(demo)
    _attach_proofreader_refresh_events(demo)
    return demo


def parse_args():
    return v4.parse_args()


def main() -> int:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        selected_port = base._find_available_port(args.host, port, max_tries=int(args.auto_port_tries))
        if selected_port != port:
            print(f"[INFO] Requested port {port} is occupied. Using available port {selected_port} instead.")
        port = selected_port

    print("====================================================")
    print("VoiceLab DataFactory User WebUI v5 (release hardening)")
    print("====================================================")
    print(f"project_root : {PROJECT_ROOT}")
    print(f"host         : {args.host}")
    print(f"port         : {port}")
    print("====================================================")

    demo = build_demo()
    try:
        demo.queue(default_concurrency_limit=8)
    except TypeError:
        demo.queue()
    demo.launch(server_name=args.host, server_port=port, inbrowser=bool(args.inbrowser))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())