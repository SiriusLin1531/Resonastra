from __future__ import annotations

"""Release-hardening wrapper for the canonical Stage1 few-shot builder.

The canonical ``build_stage1_fewshot_dataset.py`` remains the low-level builder
and keeps its existing ``--overwrite`` meaning.  This wrapper adds the release
invariant required by RH-1B without duplicating the model-heavy build logic:

1. run the canonical builder with the original arguments;
2. only after a successful build has written the new train/val manifests,
   optionally prune cache files not referenced by those manifests;
3. validate the strict Stage1 manifest/cache contract;
4. persist ``cache_cleanup`` and ``cache_contract`` into build_report.json;
5. return failure if the final contract is not exact.

The wrapper deliberately never prunes after a failed canonical build.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.voicelab_user.data_factory.stage1_cache_contract import (
    analyze_stage1_cache_contract,
    prune_stage1_orphan_cache,
    stage1_paths_from_output_root,
)

LEGACY_BUILDER = PROJECT_ROOT / "scripts" / "build_stage1_fewshot_dataset.py"
PRUNE_FLAG = "--prune_orphans"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)


def parse_release_arguments(argv: Sequence[str]) -> tuple[list[str], Path, bool]:
    """Return legacy argv, output_root and RH-1B prune selection.

    Only ``--prune_orphans`` is consumed by this wrapper. Every canonical
    builder argument is forwarded byte-for-byte as a string, including
    ``--overwrite``.
    """

    raw = [str(value) for value in argv]
    forwarded: list[str] = []
    prune_orphans = False
    output_root: str | None = None

    index = 0
    while index < len(raw):
        token = raw[index]
        if token == PRUNE_FLAG:
            prune_orphans = True
            index += 1
            continue

        forwarded.append(token)
        if token == "--output_root":
            if index + 1 >= len(raw):
                raise ValueError("--output_root requires a value")
            output_root = raw[index + 1]
        elif token.startswith("--output_root="):
            output_root = token.split("=", 1)[1]
        index += 1

    if not str(output_root or "").strip():
        raise ValueError("RH-1B hardened Stage1 builder requires --output_root")

    return forwarded, Path(str(output_root)).expanduser().resolve(strict=False), prune_orphans


def build_legacy_command(argv: Sequence[str]) -> tuple[list[str], Path, bool]:
    forwarded, output_root, prune_orphans = parse_release_arguments(argv)
    command = [sys.executable, str(LEGACY_BUILDER), *forwarded]
    return command, output_root, prune_orphans


def _load_build_report(output_root: Path) -> tuple[Path, dict[str, Any]]:
    report_path = stage1_paths_from_output_root(output_root)["build_report"]
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Stage1 canonical builder did not produce build_report.json: {report_path}"
        )
    value = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Stage1 build report is not a JSON object: {report_path}")
    return report_path, dict(value)


def finalize_stage1_cache_contract(
    output_root: str | Path,
    *,
    prune_orphans: bool,
) -> dict[str, Any]:
    """Apply RH-1B cleanup/validation after a successful canonical build."""

    root = Path(output_root).expanduser().resolve(strict=False)
    report_path, report = _load_build_report(root)

    cleanup_payload: dict[str, Any]
    try:
        if prune_orphans:
            cleanup_payload = prune_stage1_orphan_cache(root, dry_run=False)
        else:
            before = analyze_stage1_cache_contract(root)
            cleanup_payload = {
                "status": "skipped",
                "dry_run": False,
                "output_root": str(root),
                "reason": "--prune_orphans not requested",
                "removed_frontend_count": 0,
                "removed_semantic_count": 0,
                "removed_frontend": [],
                "removed_semantic": [],
                "contract_before": before.to_dict(),
                "contract_after": before.to_dict(),
            }
    except Exception as exc:  # noqa: BLE001
        contract = analyze_stage1_cache_contract(root)
        cleanup_payload = {
            "status": "failed",
            "dry_run": False,
            "output_root": str(root),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        report["cache_cleanup"] = cleanup_payload
        report["cache_contract"] = contract.to_dict()
        report["status"] = "FAILED"
        report["release_hardening"] = {
            "rh_1b": "FAILED",
            "reason": "cache_cleanup_failed",
        }
        _atomic_write_json(report_path, report)
        return {
            "status": "FAILED",
            "output_root": str(root),
            "build_report": str(report_path),
            "cache_cleanup": cleanup_payload,
            "cache_contract": contract.to_dict(),
        }

    contract = analyze_stage1_cache_contract(root)
    contract_payload = contract.to_dict()

    report["cache_cleanup"] = cleanup_payload
    report["cache_contract"] = contract_payload
    report["release_hardening"] = {
        "rh_1b": "OK" if contract.valid else "FAILED",
        "strict_cache_contract": True,
        "prune_orphans": bool(prune_orphans),
    }
    if not contract.valid:
        report["status"] = "FAILED"

    _atomic_write_json(report_path, report)
    return {
        "status": "OK" if contract.valid else "FAILED",
        "output_root": str(root),
        "build_report": str(report_path),
        "cache_cleanup": cleanup_payload,
        "cache_contract": contract_payload,
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        command, output_root, prune_orphans = build_legacy_command(raw_argv)
    except Exception as exc:  # noqa: BLE001
        print(f"[RH-1B] argument error: {type(exc).__name__}: {exc}")
        return 2

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if int(completed.returncode) != 0:
        print(
            "[RH-1B] canonical Stage1 builder failed; "
            "orphan cleanup was intentionally skipped."
        )
        return int(completed.returncode)

    result = finalize_stage1_cache_contract(
        output_root,
        prune_orphans=prune_orphans,
    )
    contract = result["cache_contract"]
    cleanup = result["cache_cleanup"]

    print("=" * 60)
    print("RH-1B Stage1 cache contract")
    print("=" * 60)
    print("status                    :", result["status"])
    print("prune_orphans             :", prune_orphans)
    print("removed_frontend          :", cleanup.get("removed_frontend_count", 0))
    print("removed_semantic          :", cleanup.get("removed_semantic_count", 0))
    print("expected_frontend_count   :", contract.get("expected_frontend_count"))
    print("actual_frontend_count     :", contract.get("actual_frontend_count"))
    print("expected_semantic_count   :", contract.get("expected_semantic_count"))
    print("actual_semantic_count     :", contract.get("actual_semantic_count"))
    print("contract_reasons          :", contract.get("reasons"))
    print("build_report              :", result["build_report"])
    print("=" * 60)

    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
