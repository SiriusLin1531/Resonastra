from __future__ import annotations

"""Stable user-facing entry for the Resonastra Training UI."""

from scripts import webui_training_user_v3 as implementation


def build_demo():
    demo = implementation.build_demo()
    demo.title = "Resonastra 训练"
    return demo


def parse_args():
    return implementation.parse_args()


def main() -> int:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        port = implementation.legacy._find_available_port(
            args.host,
            port,
            max_tries=int(args.auto_port_tries),
        )

    print("====================================================")
    print("Resonastra Training")
    print("====================================================")
    print(f"project_root : {implementation.PROJECT_ROOT}")
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
