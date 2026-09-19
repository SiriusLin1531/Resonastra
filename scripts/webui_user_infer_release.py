from __future__ import annotations

"""Stable user-facing entry for the Resonastra Inference UI."""

from scripts import webui_user_infer_v5 as implementation


def build_demo():
    demo = implementation.build_demo()
    demo.title = "Resonastra 语音生成"
    return demo


def parse_args():
    return implementation.parse_args()


def main() -> int:
    args = parse_args()

    print("====================================================")
    print("Resonastra Inference")
    print("====================================================")
    print(f"project_root : {implementation.PROJECT_ROOT}")
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
