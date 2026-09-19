from __future__ import annotations

"""Stable user-facing entry for the Resonastra DataFactory UI."""

from scripts import webui_data_factory_user_stage1_v6 as implementation


def build_demo():
    demo = implementation.build_demo()
    demo.title = "Resonastra 数据工厂"
    return demo


def parse_args():
    return implementation.parse_args()


def main() -> int:
    args = parse_args()
    port = int(args.port)
    if bool(args.auto_port):
        selected_port = implementation.base._find_available_port(
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
    print("Resonastra DataFactory")
    print("====================================================")
    print(f"project_root : {implementation.PROJECT_ROOT}")
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
