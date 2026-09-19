from __future__ import annotations

"""VoiceLab Training user UI v3 — UX-FIX-1 compatibility shell.

Post-freeze cleanup:
- remove development-only sample defaults/placeholders;
- reject blank work_dir + blank speaker instead of silently deriving
  an unnamed user_data work directory;
- keep the accepted Training v2 business callbacks and runtime behavior.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from scripts import webui_training_user_v2 as v2

legacy = v2.legacy


def _resolve_work_dir_no_dev_fallback(work_dir: str, speaker_name: str) -> Path:
    text = legacy._blank_to_none(work_dir)
    if text is not None:
        return Path(text).expanduser().resolve(strict=False)

    speaker = str(speaker_name or "").strip()
    if not speaker:
        raise ValueError("请先填写说话人 / 角色名，或填写工作目录。")
    return (PROJECT_ROOT / "user_data" / f"{speaker}_factory").resolve(strict=False)


def _clean_development_defaults(demo: gr.Blocks) -> None:
    blocks = getattr(demo, "blocks", {})
    values = blocks.values() if isinstance(blocks, dict) else []
    for component in values:
        label = str(getattr(component, "label", "") or "")
        value = getattr(component, "value", None)

        if label in {"说话人 / 角色名", "Profile 名称"} and not str(value or "").strip():
            try:
                component.placeholder = (
                    "请输入声音角色名称，例如 Character_A"
                    if label == "说话人 / 角色名"
                    else "留空则使用当前声音角色名称"
                )
            except Exception:
                pass


def build_demo() -> gr.Blocks:
    legacy._resolve_work_dir = _resolve_work_dir_no_dev_fallback
    demo = v2.create_demo()
    _clean_development_defaults(demo)
    return demo


def parse_args():
    return v2.parse_args()


def main() -> int:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        port = legacy._find_available_port(
            args.host,
            port,
            max_tries=int(args.auto_port_tries),
        )

    print("====================================================")
    print("VoiceLab Training User WebUI v3 (UX-FIX-1 candidate)")
    print("====================================================")
    print(f"project_root : {PROJECT_ROOT}")
    print(f"host         : {args.host}")
    print(f"port         : {port}")
    print("====================================================")

    demo = build_demo()
    try:
        demo.queue(default_concurrency_limit=4)
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
