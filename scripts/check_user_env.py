"""
Command-line environment checker for Resonastra.

This script calls src.voicelab_user.runtime.env_check and writes a JSON report
to logs/check_user_env_report.json by default.

Usage:
    python scripts/check_user_env.py
    python scripts/check_user_env.py --strict-runtime
    python scripts/check_user_env.py --output logs/my_env_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_project_path() -> Path:
    """Add the repository root to sys.path and return it.

    The script lives in scripts/check_user_env.py, so the project root is the
    parent directory of scripts/.
    """

    project_root = Path(__file__).resolve(strict=False).parents[1]
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Resonastra offline runtime environment.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Optional Resonastra project root. Defaults to the repository root inferred from this script.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/check_user_env_report.json",
        help="Path to write the JSON environment-check report.",
    )
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Treat missing runtime/env as a failure. Use this for final offline user packages.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return a non-zero exit code when warnings are present.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Do not print the text report. The JSON report is still written to --output.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_path()
    args = parse_args()

    try:
        from src.voicelab_user.runtime.env_check import (
            STATUS_FAIL,
            STATUS_WARN,
            format_env_check_report,
            run_user_env_check,
            write_env_check_report,
        )
    except Exception as exc:  # noqa: BLE001 - keep CLI failure readable.
        print("[FAIL] Could not import Resonastra user env_check module.", file=sys.stderr)
        print(f"Project root: {project_root}", file=sys.stderr)
        print(f"Error: {exc!r}", file=sys.stderr)
        return 2

    try:
        report = run_user_env_check(
            project_root=args.project_root or project_root,
            strict_runtime=args.strict_runtime,
        )
        output_path = write_env_check_report(report, args.output)
    except Exception as exc:  # noqa: BLE001 - user-facing CLI guard.
        print("[FAIL] Resonastra environment check crashed.", file=sys.stderr)
        print(f"Error: {exc!r}", file=sys.stderr)
        return 2

    if not args.json_only:
        print(format_env_check_report(report))
        print()
        print(f"JSON report written to: {output_path}")

    if report.overall_status == STATUS_FAIL:
        return 1
    if args.fail_on_warn and report.overall_status == STATUS_WARN:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
