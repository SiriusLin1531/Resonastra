from __future__ import annotations

"""Resonastra DataFactory user UI v6 compatibility shell.

This shell preserves the accepted v5 business/event chain while closing
release-only UI/runtime compatibility findings discovered during REL-1-8
clean-machine testing:

- CM-R2-001: keep Gradio 5.x Dropdown internal choices in normalized pair form;
- CM-R2-004: make proofreader status polling read-only and monitor only the
  committed current work directory during automatic refreshes.

Historical v5 remains unchanged for RC2 traceability.
"""

from pathlib import Path
from typing import Any

import gradio as gr

from scripts import webui_data_factory_user_stage1_v5 as v5

base = v5.base
legacy = v5.legacy
PROJECT_ROOT = v5.PROJECT_ROOT


def _normalize_single_choice_dropdown(component: Any, *, value: str) -> None:
    """Keep Gradio 5.x constructed Dropdown internals in normalized pair form."""

    component.choices = [(value, value)]
    component.value = value
    component.interactive = False


def _repair_release_dropdowns(demo: gr.Blocks) -> None:
    for component in v5._component_values(demo):
        label = str(getattr(component, "label", "") or "")
        if label == "ASR 后端":
            _normalize_single_choice_dropdown(component, value="auto")
            try:
                component.info = "Resonastra v1.0.0 中文数据准备固定使用本地离线 FunASR。"
            except Exception:
                pass


def _not_loaded_proofreader_payload(work_dir: str | Path | None = None) -> tuple[str, dict[str, Any]]:
    """Return a stable no-project-loaded proofreader status without filesystem writes."""

    payload: dict[str, Any] = {"status": "not_loaded"}
    if work_dir is not None and str(work_dir).strip():
        payload["work_dir"] = str(work_dir)
    return v5._proofreader_user_markdown(payload), payload


def _failed_proofreader_payload(exc: Exception) -> tuple[str, dict[str, Any]]:
    payload = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    return v5._proofreader_user_markdown(payload), payload


def _resolve_existing_candidate_work_dir(work_dir: Any, speaker_name: Any) -> Path | None:
    """Resolve an explicit user selection, but never create it.

    This helper is used only for explicit Load/Refresh actions. Automatic page
    load/timer polling does not consume these editable fields at all.
    """

    work_dir_text = str(work_dir or "").strip()
    speaker_text = str(speaker_name or "").strip()
    if not work_dir_text and not speaker_text:
        return None

    try:
        candidate = v5._resolve_active_work_dir("", work_dir_text, speaker_text)
    except Exception:
        return None
    return Path(candidate).expanduser().resolve(strict=False)


def _inspect_existing_work_dir_read_only(work_dir: Any) -> tuple[str, dict[str, Any]]:
    """Inspect proofreader runtime only when a real work directory already exists.

    Critical release invariant:
        a status refresh must never materialize a candidate work directory.
    """

    text = str(work_dir or "").strip()
    if not text:
        return _not_loaded_proofreader_payload()

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=False)

    if not path.is_dir():
        return _not_loaded_proofreader_payload(path)

    try:
        payload = v5.inspect_proofreader_runtime(path, write_state=False)
        return v5._proofreader_user_markdown(payload), payload
    except Exception as exc:  # noqa: BLE001
        return _failed_proofreader_payload(exc)


def refresh_committed_proofreader_user_click(work_dir_output: str):
    """Automatic refresh: inspect only the committed Current Work Directory."""

    return _inspect_existing_work_dir_read_only(work_dir_output)


def refresh_selected_existing_proofreader_user_click(work_dir: str, speaker_name: str):
    """Explicit Load refresh: inspect the selected directory only if it already exists."""

    candidate = _resolve_existing_candidate_work_dir(work_dir, speaker_name)
    if candidate is None:
        return _not_loaded_proofreader_payload()
    return _inspect_existing_work_dir_read_only(candidate)


def refresh_existing_proofreader_user_click(
    work_dir_output: str,
    work_dir: str,
    speaker_name: str,
):
    """Manual refresh: prefer committed work dir, otherwise inspect an existing selection."""

    committed = str(work_dir_output or "").strip()
    if committed:
        return _inspect_existing_work_dir_read_only(committed)
    return refresh_selected_existing_proofreader_user_click(work_dir, speaker_name)


def _attach_proofreader_refresh_events_read_only(demo: gr.Blocks) -> None:
    """Attach v6 proofreader events with side-effect-free automatic polling."""

    close_button = v5._INJECTED_CLOSE_PROOFREAD_BUTTON
    load_button = v5._find_component(demo, value="载入/扫描工作目录")
    manual_refresh_button = v5._find_component(demo, value="立即刷新")
    work_dir_output = v5._find_component(demo, label="当前工作目录")
    work_dir = v5._find_component(demo, label="工作目录（可选）")
    speaker_name = v5._find_component(demo, label="说话人 / 角色名")
    proofread_status = v5._find_component(demo, elem_class="vl-proofread-status")
    proofread_json = v5._find_component(demo, label="人工校对器诊断")

    required = [work_dir_output, work_dir, speaker_name, proofread_status, proofread_json]
    if any(component is None for component in required):
        raise RuntimeError("REL-1-9 v6 proofreader UI components could not be resolved.")
    if close_button is None:
        raise RuntimeError("REL-1-9 v6 close proofreader button was not injected.")

    proofreader_outputs = [proofread_status, proofread_json]
    lifecycle_inputs = [work_dir_output, work_dir, speaker_name]

    # Registration must occur inside the existing Blocks context.
    with demo:
        # Stop is a real lifecycle transition and intentionally remains writable.
        close_button.click(
            fn=v5.stop_proofreader_user_click,
            inputs=lifecycle_inputs,
            outputs=proofreader_outputs,
            queue=False,
        )

        # Explicit Load may inspect the user's current selection, but only when
        # the resolved directory already exists; inspection itself is read-only.
        if load_button is not None:
            load_button.click(
                fn=refresh_selected_existing_proofreader_user_click,
                inputs=[work_dir, speaker_name],
                outputs=proofreader_outputs,
                queue=False,
            )

        # Manual refresh preserves convenience while remaining side-effect-free.
        if manual_refresh_button is not None:
            manual_refresh_button.click(
                fn=refresh_existing_proofreader_user_click,
                inputs=lifecycle_inputs,
                outputs=proofreader_outputs,
                queue=False,
            )

        # Automatic events must never consume editable speaker/work-dir fields.
        load_method = getattr(demo, "load", None)
        if callable(load_method):
            load_method(
                fn=refresh_committed_proofreader_user_click,
                inputs=[work_dir_output],
                outputs=proofreader_outputs,
                queue=False,
            )

        timer_type = getattr(gr, "Timer", None)
        if timer_type is not None:
            proofreader_timer = timer_type(
                value=v5.PROOFREADER_REFRESH_INTERVAL_SECONDS,
                active=True,
            )
            proofreader_timer.tick(
                fn=refresh_committed_proofreader_user_click,
                inputs=[work_dir_output],
                outputs=proofreader_outputs,
                queue=False,
            )


def build_demo() -> gr.Blocks:
    # v5 owns construction of the established release UI/event chain. Override
    # only the proofreader refresh attachment while v5 builds the demo, then
    # restore the historical module global immediately afterwards.
    original_attach = v5._attach_proofreader_refresh_events
    v5._attach_proofreader_refresh_events = _attach_proofreader_refresh_events_read_only
    try:
        demo = v5.build_demo()
    finally:
        v5._attach_proofreader_refresh_events = original_attach

    _repair_release_dropdowns(demo)
    return demo


def parse_args():
    return v5.parse_args()


def main() -> int:
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

    print("====================================================")
    print("Resonastra DataFactory User WebUI v6 (release compatibility)")
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
    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=bool(args.inbrowser),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
