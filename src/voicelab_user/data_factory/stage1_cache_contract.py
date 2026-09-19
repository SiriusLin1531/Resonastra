from __future__ import annotations

"""Strict Stage1 manifest/cache contract for release hardening.

The Stage1 train/val manifests are the authoritative set of cache references.
A healthy Stage1 output contains exactly the frontend and semantic ``.pt`` files
referenced by those manifests: no missing files and no unreferenced/orphan files.

This module intentionally does not depend on torch, Gradio, or the Stage1
builder so it can be used by status scans, cleanup gates, and transactional
rebuild validation without importing heavy model code.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


TRAIN_MANIFEST_FILENAME = "train_manifest.jsonl"
VAL_MANIFEST_FILENAME = "val_manifest.jsonl"
FRONTEND_ROOT_NAME = "frontend_cache"
SEMANTIC_ROOT_NAME = "semantic_cache"


@dataclass
class Stage1CacheContract:
    valid: bool
    output_root: str
    train_rows: int = 0
    val_rows: int = 0
    manifest_rows: int = 0
    expected_frontend_count: int = 0
    actual_frontend_count: int = 0
    expected_semantic_count: int = 0
    actual_semantic_count: int = 0
    missing_frontend: list[str] = field(default_factory=list)
    missing_semantic: list[str] = field(default_factory=list)
    orphan_frontend: list[str] = field(default_factory=list)
    orphan_semantic: list[str] = field(default_factory=list)
    duplicate_item_ids: list[str] = field(default_factory=list)
    duplicate_frontend_paths: list[str] = field(default_factory=list)
    duplicate_semantic_paths: list[str] = field(default_factory=list)
    split_conflicts: list[str] = field(default_factory=list)
    invalid_paths: list[str] = field(default_factory=list)
    malformed_rows: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage1_paths_from_output_root(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root).expanduser().resolve(strict=False)
    return {
        "output_root": root,
        "train_manifest": root / TRAIN_MANIFEST_FILENAME,
        "val_manifest": root / VAL_MANIFEST_FILENAME,
        "frontend_root": root / FRONTEND_ROOT_NAME,
        "semantic_root": root / SEMANTIC_ROOT_NAME,
        "build_report": root / "metadata" / "build_report.json",
        "split_report": root / "metadata" / "split.json",
    }


def stage1_paths_from_work_dir(work_dir: str | Path) -> dict[str, Path]:
    work = Path(work_dir).expanduser().resolve(strict=False)
    return stage1_paths_from_output_root(work / "07_stage1_ft")


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"missing_manifest:{path.name}")
        return rows, errors

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{path.name}:{line_number}:invalid_json:{type(exc).__name__}"
                )
                continue
            if not isinstance(value, dict):
                errors.append(f"{path.name}:{line_number}:row_not_object")
                continue
            row = dict(value)
            row["__manifest_name__"] = path.name
            row["__line_number__"] = line_number
            rows.append(row)
    return rows, errors


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_manifest_cache_path(
    value: Any,
    *,
    output_root: Path,
    expected_root: Path,
    field_name: str,
    row_label: str,
) -> tuple[Path | None, str | None]:
    text = str(value or "").strip()
    if not text:
        return None, f"{row_label}:{field_name}:missing"

    raw = Path(text)
    path = raw if raw.is_absolute() else output_root / raw
    resolved = path.expanduser().resolve(strict=False)

    if not _is_within(resolved, output_root):
        return None, f"{row_label}:{field_name}:escapes_output_root:{text}"
    if not _is_within(resolved, expected_root):
        return None, f"{row_label}:{field_name}:outside_expected_cache_root:{text}"
    if resolved.suffix.lower() != ".pt":
        return None, f"{row_label}:{field_name}:not_pt:{text}"
    return resolved, None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _all_pt_files(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {path.resolve(strict=False) for path in root.rglob("*.pt") if path.is_file()}


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return sorted(duplicates)


def analyze_stage1_cache_contract(output_root: str | Path) -> Stage1CacheContract:
    paths = stage1_paths_from_output_root(output_root)
    root = paths["output_root"]
    frontend_root = paths["frontend_root"].resolve(strict=False)
    semantic_root = paths["semantic_root"].resolve(strict=False)

    train_rows, train_errors = _read_jsonl(paths["train_manifest"])
    val_rows, val_errors = _read_jsonl(paths["val_manifest"])
    malformed_rows = [*train_errors, *val_errors]

    expected_frontend: list[Path] = []
    expected_semantic: list[Path] = []
    item_ids: list[str] = []
    split_conflicts: list[str] = []
    invalid_paths: list[str] = []

    for expected_split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            manifest_name = str(row.get("__manifest_name__"))
            line_number = int(row.get("__line_number__") or 0)
            row_label = f"{manifest_name}:{line_number}"

            item_id = str(row.get("item_id") or "").strip()
            if not item_id:
                malformed_rows.append(f"{row_label}:missing_item_id")
            else:
                item_ids.append(item_id)

            split = str(row.get("split") or "").strip().lower()
            if split and split != expected_split:
                split_conflicts.append(
                    f"{row_label}:manifest={expected_split}:row_split={split}"
                )

            frontend_path, frontend_error = _resolve_manifest_cache_path(
                row.get("frontend_path"),
                output_root=root,
                expected_root=frontend_root,
                field_name="frontend_path",
                row_label=row_label,
            )
            if frontend_error:
                invalid_paths.append(frontend_error)
            elif frontend_path is not None:
                expected_frontend.append(frontend_path)

            semantic_path, semantic_error = _resolve_manifest_cache_path(
                row.get("target_semantic_path"),
                output_root=root,
                expected_root=semantic_root,
                field_name="target_semantic_path",
                row_label=row_label,
            )
            if semantic_error:
                invalid_paths.append(semantic_error)
            elif semantic_path is not None:
                expected_semantic.append(semantic_path)

    expected_frontend_set = set(expected_frontend)
    expected_semantic_set = set(expected_semantic)
    actual_frontend = _all_pt_files(frontend_root)
    actual_semantic = _all_pt_files(semantic_root)

    missing_frontend = sorted(
        _relative(path, root) for path in expected_frontend_set if not path.is_file()
    )
    missing_semantic = sorted(
        _relative(path, root) for path in expected_semantic_set if not path.is_file()
    )
    orphan_frontend = sorted(
        _relative(path, root) for path in actual_frontend - expected_frontend_set
    )
    orphan_semantic = sorted(
        _relative(path, root) for path in actual_semantic - expected_semantic_set
    )

    duplicate_item_ids = _duplicates(item_ids)
    duplicate_frontend_paths = _duplicates(str(path) for path in expected_frontend)
    duplicate_semantic_paths = _duplicates(str(path) for path in expected_semantic)

    reasons: list[str] = []
    checks = [
        (malformed_rows, "malformed_manifest_rows"),
        (invalid_paths, "invalid_cache_paths"),
        (duplicate_item_ids, "duplicate_item_ids"),
        (duplicate_frontend_paths, "duplicate_frontend_paths"),
        (duplicate_semantic_paths, "duplicate_semantic_paths"),
        (split_conflicts, "split_conflicts"),
        (missing_frontend, "missing_frontend_cache"),
        (missing_semantic, "missing_semantic_cache"),
        (orphan_frontend, "orphan_frontend_cache"),
        (orphan_semantic, "orphan_semantic_cache"),
    ]
    for values, reason in checks:
        if values:
            reasons.append(reason)

    manifest_rows = len(train_rows) + len(val_rows)
    if manifest_rows <= 0:
        reasons.append("empty_stage1_manifests")

    return Stage1CacheContract(
        valid=not reasons,
        output_root=str(root),
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        manifest_rows=manifest_rows,
        expected_frontend_count=len(expected_frontend_set),
        actual_frontend_count=len(actual_frontend),
        expected_semantic_count=len(expected_semantic_set),
        actual_semantic_count=len(actual_semantic),
        missing_frontend=missing_frontend,
        missing_semantic=missing_semantic,
        orphan_frontend=orphan_frontend,
        orphan_semantic=orphan_semantic,
        duplicate_item_ids=duplicate_item_ids,
        duplicate_frontend_paths=duplicate_frontend_paths,
        duplicate_semantic_paths=duplicate_semantic_paths,
        split_conflicts=sorted(split_conflicts),
        invalid_paths=sorted(invalid_paths),
        malformed_rows=sorted(malformed_rows),
        reasons=reasons,
    )


def _remove_paths(paths: Iterable[Path], *, dry_run: bool) -> list[str]:
    removed: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        removed.append(str(path))
        if not dry_run:
            path.unlink()
    return removed


def _cleanup_empty_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def prune_stage1_orphan_cache(
    output_root: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve(strict=False)
    before = analyze_stage1_cache_contract(root)

    # Never delete cache when manifest parsing/path validation is untrustworthy.
    unsafe_reasons = {
        "malformed_manifest_rows",
        "invalid_cache_paths",
        "duplicate_item_ids",
        "duplicate_frontend_paths",
        "duplicate_semantic_paths",
        "split_conflicts",
        "empty_stage1_manifests",
    }
    blocking = sorted(reason for reason in before.reasons if reason in unsafe_reasons)
    if blocking:
        raise ValueError(
            "Stage1 orphan cleanup refused because manifest contract is unsafe: "
            + ", ".join(blocking)
        )

    frontend_paths = [root / value for value in before.orphan_frontend]
    semantic_paths = [root / value for value in before.orphan_semantic]
    removed_frontend = _remove_paths(frontend_paths, dry_run=dry_run)
    removed_semantic = _remove_paths(semantic_paths, dry_run=dry_run)

    if not dry_run:
        _cleanup_empty_dirs(root / FRONTEND_ROOT_NAME)
        _cleanup_empty_dirs(root / SEMANTIC_ROOT_NAME)

    after = before if dry_run else analyze_stage1_cache_contract(root)
    return {
        "status": "dry_run" if dry_run else "completed",
        "dry_run": bool(dry_run),
        "output_root": str(root),
        "removed_frontend_count": len(removed_frontend),
        "removed_semantic_count": len(removed_semantic),
        "removed_frontend": removed_frontend,
        "removed_semantic": removed_semantic,
        "contract_before": before.to_dict(),
        "contract_after": after.to_dict(),
    }


__all__ = [
    "Stage1CacheContract",
    "stage1_paths_from_output_root",
    "stage1_paths_from_work_dir",
    "analyze_stage1_cache_contract",
    "prune_stage1_orphan_cache",
]
