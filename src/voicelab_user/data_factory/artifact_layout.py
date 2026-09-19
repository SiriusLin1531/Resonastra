from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Local imports
# 本地导入
# ============================================================
from .adapter import DataFactoryAdapter
from .result import DataFactoryArtifactPaths


COMPACT_ARTIFACT_LAYOUT_VERSION = "voicelab_data_factory_compact_layout_v1"
LEGACY_MIGRATION_REPORT_FILENAME = "legacy_artifact_migration_report.json"


# ============================================================
# Generic helpers
# 通用工具
# ============================================================
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path).resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _count_files(path: Optional[Path], pattern: str = "*") -> Optional[int]:
    if path is None or not Path(path).is_dir():
        return None
    return len([x for x in Path(path).glob(pattern) if x.is_file()])


def _file_size(path: Optional[Path]) -> Optional[int]:
    if path is None or not Path(path).is_file():
        return None
    return int(Path(path).stat().st_size)


def _path_has_content(path: Optional[Path]) -> bool:
    if path is None or not Path(path).exists():
        return False
    if Path(path).is_file():
        return Path(path).stat().st_size > 0
    if Path(path).is_dir():
        return any(Path(path).iterdir())
    return False


# ============================================================
# Compact layout helpers
# 紧凑型产物目录布局工具
# ============================================================
def compact_artifacts_from_work_dir(work_dir: str | Path) -> DataFactoryArtifactPaths:
    """Return compact, work-dir-contained artifact paths.

    中文说明：
        P10-11-9 目录收拢布局。

        原先较重的 Stage2 / cache / split / filter 产物会散落在：

            user_data/{speaker_name}_stage2_pt_all
            user_data/{speaker_name}_stage2_pt_all_continuous
            user_data/{speaker_name}_stage2_pt_all_v662_style_f0_spk
            ...

        现在统一收拢到：

            user_data/{speaker_name}_factory/
                07_stage2_pt_all/
                08_continuous_semantic/
                09_style_f0_spk/
                10_train_val/
                11_filter/

        这让停止、扫描、续跑、打包、删除和迁移都只围绕一个 work_dir。
    """

    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    export_dir = work_dir / "06_export"

    stage2_pt_root = work_dir / "07_stage2_pt_all"
    continuous_semantic_root = work_dir / "08_continuous_semantic"
    style_cache_root = work_dir / "09_style_f0_spk"

    split_root = work_dir / "10_train_val"
    train_pt_root = split_root / "train"
    val_pt_root = split_root / "val"
    split_report_path = split_root / "split_report.json"

    filter_root = work_dir / "11_filter"
    rejected_pt_root = filter_root / "rejected"
    filter_report_json_path = filter_root / "filter_bad_style_samples_report.json"
    filter_report_csv_path = filter_root / "filter_bad_style_samples_bad.csv"

    return DataFactoryArtifactPaths(
        work_dir=work_dir,
        request_json_path=work_dir / "data_factory_request.json",
        result_json_path=work_dir / "data_factory_result.json",

        raw_audio_index_path=work_dir / "00_raw_index" / "raw_audio_index.jsonl",
        prepare_report_path=work_dir / "fewshot_prepare_report.json",

        manifest_jsonl_path=export_dir / "manifest.jsonl",
        dataset_list_path=export_dir / "dataset.list",
        stage2_manifest_jsonl_path=export_dir / "stage2_manifest.jsonl",

        export_clips_dir=export_dir / "clips",

        manifest_before_proofread_path=export_dir / "manifest.before_proofread.jsonl",
        dataset_before_proofread_path=export_dir / "dataset.before_proofread.list",
        stage2_manifest_before_proofread_path=export_dir / "stage2_manifest.before_proofread.jsonl",

        manifest_corrected_path=export_dir / "manifest.corrected.jsonl",
        stage2_manifest_corrected_path=export_dir / "stage2_manifest.corrected.jsonl",
        correction_report_path=export_dir / "correction_report.json",

        stage2_manifest_fewshot_path=export_dir / "stage2_manifest.fewshot.jsonl",
        prompt_selection_report_path=export_dir / "prompt_selection_report.json",
        fewshot_manifest_conversion_report_path=export_dir / "fewshot_manifest_conversion_report.json",

        manifest_health_report_path=export_dir / "fewshot_health_manifest_report.json",
        stage2_manifest_health_report_path=export_dir / "fewshot_health_stage2_manifest_report.json",

        stage2_pt_root=stage2_pt_root,
        stage2_pt_health_report_path=stage2_pt_root / "fewshot_health_pt_report.json",

        continuous_semantic_root=continuous_semantic_root,
        style_cache_root=style_cache_root,

        train_pt_root=train_pt_root,
        val_pt_root=val_pt_root,
        split_report_path=split_report_path,

        rejected_pt_root=rejected_pt_root,
        filter_report_json_path=filter_report_json_path,
        filter_report_csv_path=filter_report_csv_path,
    )


# ============================================================
# Legacy layout helpers
# 旧 sibling 目录布局工具
# ============================================================
def legacy_sibling_artifacts_from_work_dir(work_dir: str | Path) -> DataFactoryArtifactPaths:
    """Return legacy sibling-directory artifact paths.

    中文说明：
        旧版本布局中，work_dir 仍然保存 prepare / manifest / proofread 产物，
        但 Stage2 .pt、continuous、style、train/val、filter 产物会散落在
        work_dir 同级目录下。

        示例：
            user_data/S005_factory
        ->  user_data/S005_stage2_pt_all
            user_data/S005_stage2_pt_all_continuous
            user_data/S005_stage2_pt_all_v662_style_f0_spk
            user_data/S005_stage2_pt_v662_style_f0_spk
            user_data/S005_stage2_pt_v662_style_f0_spk_rejected
    """

    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    export_dir = work_dir / "06_export"

    name = work_dir.name
    if name.endswith("_factory"):
        speaker_prefix = name[: -len("_factory")]
        stage2_pt_root = work_dir.parent / f"{speaker_prefix}_stage2_pt_all"
    else:
        stage2_pt_root = work_dir / "07_stage2_pt"

    continuous_semantic_root = stage2_pt_root.parent / f"{stage2_pt_root.name}_continuous"
    style_cache_root = stage2_pt_root.parent / f"{stage2_pt_root.name}_v662_style_f0_spk"

    style_name = style_cache_root.name
    split_name = style_name.replace("_stage2_pt_all_", "_stage2_pt_")
    if split_name == style_name and style_name.endswith("_all"):
        split_name = style_name[: -len("_all")]

    split_root = style_cache_root.parent / split_name
    train_pt_root = split_root / "train"
    val_pt_root = split_root / "val"
    split_report_path = split_root / "split_report.json"

    rejected_pt_root = split_root.parent / f"{split_root.name}_rejected"
    filter_report_json_path = split_root.parent / f"filter_{split_root.name}_bad_style_samples_report.json"
    filter_report_csv_path = split_root.parent / f"filter_{split_root.name}_bad_style_samples_bad.csv"

    return DataFactoryArtifactPaths(
        work_dir=work_dir,
        request_json_path=work_dir / "data_factory_request.json",
        result_json_path=work_dir / "data_factory_result.json",

        raw_audio_index_path=work_dir / "00_raw_index" / "raw_audio_index.jsonl",
        prepare_report_path=work_dir / "fewshot_prepare_report.json",

        manifest_jsonl_path=export_dir / "manifest.jsonl",
        dataset_list_path=export_dir / "dataset.list",
        stage2_manifest_jsonl_path=export_dir / "stage2_manifest.jsonl",

        export_clips_dir=export_dir / "clips",

        manifest_before_proofread_path=export_dir / "manifest.before_proofread.jsonl",
        dataset_before_proofread_path=export_dir / "dataset.before_proofread.list",
        stage2_manifest_before_proofread_path=export_dir / "stage2_manifest.before_proofread.jsonl",

        manifest_corrected_path=export_dir / "manifest.corrected.jsonl",
        stage2_manifest_corrected_path=export_dir / "stage2_manifest.corrected.jsonl",
        correction_report_path=export_dir / "correction_report.json",

        stage2_manifest_fewshot_path=export_dir / "stage2_manifest.fewshot.jsonl",
        prompt_selection_report_path=export_dir / "prompt_selection_report.json",
        fewshot_manifest_conversion_report_path=export_dir / "fewshot_manifest_conversion_report.json",

        manifest_health_report_path=export_dir / "fewshot_health_manifest_report.json",
        stage2_manifest_health_report_path=export_dir / "fewshot_health_stage2_manifest_report.json",

        stage2_pt_root=stage2_pt_root,
        stage2_pt_health_report_path=stage2_pt_root / "fewshot_health_pt_report.json",

        continuous_semantic_root=continuous_semantic_root,
        style_cache_root=style_cache_root,

        train_pt_root=train_pt_root,
        val_pt_root=val_pt_root,
        split_report_path=split_report_path,

        rejected_pt_root=rejected_pt_root,
        filter_report_json_path=filter_report_json_path,
        filter_report_csv_path=filter_report_csv_path,
    )


def legacy_artifacts_from_work_dir(adapter: DataFactoryAdapter, work_dir: str | Path) -> DataFactoryArtifactPaths:
    """Return legacy sibling-directory artifact paths without compact patching.

    中文说明：
        保持旧函数名用于兼容早期调用；内部现在直接使用确定性的旧 sibling
        布局推断，不依赖 monkey patch 前后的 adapter 状态。
    """

    return legacy_sibling_artifacts_from_work_dir(work_dir)


# ============================================================
# Legacy scanner and migration
# 旧目录扫描与迁移
# ============================================================
def _legacy_migration_items(work_dir: str | Path) -> list[dict[str, Any]]:
    legacy = legacy_sibling_artifacts_from_work_dir(work_dir)
    compact = compact_artifacts_from_work_dir(work_dir)

    split_root_legacy = None if legacy.train_pt_root is None else legacy.train_pt_root.parent
    split_root_compact = None if compact.train_pt_root is None else compact.train_pt_root.parent

    mapping: list[tuple[str, str, Optional[Path], Optional[Path], str]] = [
        ("stage2_pt_root", "dir", legacy.stage2_pt_root, compact.stage2_pt_root, "Stage2 .pt 样本目录"),
        ("continuous_semantic_root", "dir", legacy.continuous_semantic_root, compact.continuous_semantic_root, "continuous semantic cache 目录"),
        ("style_cache_root", "dir", legacy.style_cache_root, compact.style_cache_root, "Style/F0/Speaker cache 目录"),
        ("train_val_root", "dir", split_root_legacy, split_root_compact, "train/val 切分目录"),
        ("rejected_pt_root", "dir", legacy.rejected_pt_root, compact.rejected_pt_root, "异常样本 rejected 目录"),
        ("filter_report_json", "file", legacy.filter_report_json_path, compact.filter_report_json_path, "异常样本过滤 JSON 报告"),
        ("filter_report_csv", "file", legacy.filter_report_csv_path, compact.filter_report_csv_path, "异常样本过滤 CSV 报告"),
    ]

    items: list[dict[str, Any]] = []
    for key, kind, source, destination, description in mapping:
        if source is None or destination is None:
            continue

        source = Path(source).resolve(strict=False)
        destination = Path(destination).resolve(strict=False)
        items.append(
            {
                "key": key,
                "kind": kind,
                "description": description,
                "source": str(source),
                "destination": str(destination),
                "source_exists": source.exists(),
                "source_has_content": _path_has_content(source),
                "destination_exists": destination.exists(),
                "destination_has_content": _path_has_content(destination),
                "source_file_count": _count_files(source) if kind == "dir" else None,
                "destination_file_count": _count_files(destination) if kind == "dir" else None,
                "source_pt_count": _count_files(source, "*.pt") if kind == "dir" else None,
                "destination_pt_count": _count_files(destination, "*.pt") if kind == "dir" else None,
                "source_size_bytes": _file_size(source) if kind == "file" else None,
                "destination_size_bytes": _file_size(destination) if kind == "file" else None,
                "same_path": source == destination,
            }
        )

    return items


def scan_legacy_artifacts(work_dir: str | Path) -> dict[str, Any]:
    """Scan legacy sibling artifacts for a work_dir.

    中文说明：
        只扫描，不复制、不移动、不删除任何文件。
    """

    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    legacy = legacy_sibling_artifacts_from_work_dir(work_dir)
    compact = compact_artifacts_from_work_dir(work_dir)
    items = _legacy_migration_items(work_dir)

    detected_items = [item for item in items if item.get("source_has_content") and not item.get("same_path")]
    destination_conflicts = [
        item
        for item in detected_items
        if item.get("destination_has_content")
    ]

    return {
        "schema_version": "voicelab_data_factory_legacy_artifact_scan_v1",
        "layout_version": COMPACT_ARTIFACT_LAYOUT_VERSION,
        "work_dir": str(work_dir),
        "created_at": _now_iso(),
        "legacy_roots": {
            "stage2_pt_root": str(legacy.stage2_pt_root),
            "continuous_semantic_root": str(legacy.continuous_semantic_root),
            "style_cache_root": str(legacy.style_cache_root),
            "train_pt_root": str(legacy.train_pt_root),
            "val_pt_root": str(legacy.val_pt_root),
            "rejected_pt_root": str(legacy.rejected_pt_root),
        },
        "compact_roots": {
            "stage2_pt_root": str(compact.stage2_pt_root),
            "continuous_semantic_root": str(compact.continuous_semantic_root),
            "style_cache_root": str(compact.style_cache_root),
            "train_pt_root": str(compact.train_pt_root),
            "val_pt_root": str(compact.val_pt_root),
            "rejected_pt_root": str(compact.rejected_pt_root),
        },
        "items": items,
        "num_items": len(items),
        "num_detected_items": len(detected_items),
        "num_destination_conflicts": len(destination_conflicts),
        "has_legacy_artifacts": bool(detected_items),
        "has_destination_conflicts": bool(destination_conflicts),
    }


def _copy_or_move_path(
    *,
    source: Path,
    destination: Path,
    kind: str,
    mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    source = Path(source).resolve(strict=False)
    destination = Path(destination).resolve(strict=False)

    if not source.exists():
        return {
            "status": "skipped",
            "reason": "source_missing",
        }

    if destination.exists() and not overwrite:
        return {
            "status": "skipped",
            "reason": "destination_exists",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and overwrite:
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if mode == "copy":
        if kind == "dir":
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    elif mode == "move":
        shutil.move(str(source), str(destination))
    else:
        raise ValueError("mode must be copy or move.")

    return {
        "status": "migrated",
        "reason": None,
    }


def migrate_legacy_artifacts(
    work_dir: str | Path,
    *,
    mode: str = "copy",
    overwrite: bool = False,
    dry_run: bool = True,
    report_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Migrate legacy sibling artifacts into compact work_dir layout.

    中文说明：
        将旧 sibling 目录中的重产物迁移到当前 work_dir 内部。

        默认 dry_run=True，只生成计划，不真正复制/移动。
        默认 mode="copy"，即使 execute 也不删除旧目录。
    """

    mode = str(mode).strip().lower()
    if mode not in {"copy", "move"}:
        raise ValueError("mode must be copy or move.")

    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    scan = scan_legacy_artifacts(work_dir)

    actions: list[dict[str, Any]] = []
    for item in scan["items"]:
        action: dict[str, Any] = {
            "key": item["key"],
            "kind": item["kind"],
            "description": item["description"],
            "source": item["source"],
            "destination": item["destination"],
            "dry_run": bool(dry_run),
            "mode": mode,
            "overwrite": bool(overwrite),
        }

        if item.get("same_path"):
            action.update({"status": "skipped", "reason": "same_path"})
        elif not item.get("source_has_content"):
            action.update({"status": "skipped", "reason": "source_missing_or_empty"})
        elif item.get("destination_has_content") and not overwrite:
            action.update({"status": "skipped", "reason": "destination_exists"})
        elif dry_run:
            action.update({"status": "planned", "reason": None})
        else:
            action.update(
                _copy_or_move_path(
                    source=Path(item["source"]),
                    destination=Path(item["destination"]),
                    kind=str(item["kind"]),
                    mode=mode,
                    overwrite=overwrite,
                )
            )

        actions.append(action)

    num_planned = sum(1 for action in actions if action.get("status") == "planned")
    num_migrated = sum(1 for action in actions if action.get("status") == "migrated")
    num_skipped = sum(1 for action in actions if action.get("status") == "skipped")

    report = {
        "schema_version": "voicelab_data_factory_legacy_artifact_migration_v1",
        "layout_version": COMPACT_ARTIFACT_LAYOUT_VERSION,
        "work_dir": str(work_dir),
        "created_at": _now_iso(),
        "mode": mode,
        "overwrite": bool(overwrite),
        "dry_run": bool(dry_run),
        "scan": scan,
        "actions": actions,
        "num_planned": num_planned,
        "num_migrated": num_migrated,
        "num_skipped": num_skipped,
    }

    resolved_report_path = (
        Path(report_path).expanduser().resolve(strict=False)
        if report_path is not None and str(report_path).strip()
        else work_dir / LEGACY_MIGRATION_REPORT_FILENAME
    )
    _write_json(resolved_report_path, report)
    report["report_path"] = str(resolved_report_path)
    return report


# ============================================================
# Compact layout installer
# 紧凑型布局安装器
# ============================================================
def install_compact_artifact_layout() -> None:
    """Install compact artifact layout onto DataFactoryAdapter.

    中文说明：
        通过替换 DataFactoryAdapter.expected_artifacts()，让现有 DataFactoryService、
        UserPipeline、ResumePipeline 和 WebUI 状态扫描全部自动使用 work_dir 内聚目录。
    """

    if getattr(DataFactoryAdapter, "_voicelab_compact_layout_installed", False):
        return

    original_expected_artifacts = DataFactoryAdapter.expected_artifacts
    setattr(
        DataFactoryAdapter,
        "_voicelab_original_expected_artifacts",
        original_expected_artifacts,
    )

    def _compact_expected_artifacts(self: DataFactoryAdapter, work_dir: str | Path) -> DataFactoryArtifactPaths:
        return compact_artifacts_from_work_dir(work_dir)

    DataFactoryAdapter.expected_artifacts = _compact_expected_artifacts  # type: ignore[method-assign]
    setattr(DataFactoryAdapter, "_voicelab_compact_layout_installed", True)


def artifact_layout_debug_summary(work_dir: str | Path) -> dict[str, Any]:
    """Return a JSON-friendly summary of compact layout paths."""

    artifacts = compact_artifacts_from_work_dir(work_dir)
    legacy_scan = scan_legacy_artifacts(work_dir)
    return {
        "layout_version": COMPACT_ARTIFACT_LAYOUT_VERSION,
        "work_dir": str(artifacts.work_dir),
        "stage2_pt_root": str(artifacts.stage2_pt_root),
        "continuous_semantic_root": str(artifacts.continuous_semantic_root),
        "style_cache_root": str(artifacts.style_cache_root),
        "train_pt_root": str(artifacts.train_pt_root),
        "val_pt_root": str(artifacts.val_pt_root),
        "split_report_path": str(artifacts.split_report_path),
        "rejected_pt_root": str(artifacts.rejected_pt_root),
        "filter_report_json_path": str(artifacts.filter_report_json_path),
        "filter_report_csv_path": str(artifacts.filter_report_csv_path),
        "has_legacy_artifacts": bool(legacy_scan.get("has_legacy_artifacts")),
        "num_legacy_detected_items": int(legacy_scan.get("num_detected_items") or 0),
    }


__all__ = [
    "COMPACT_ARTIFACT_LAYOUT_VERSION",
    "LEGACY_MIGRATION_REPORT_FILENAME",
    "compact_artifacts_from_work_dir",
    "legacy_sibling_artifacts_from_work_dir",
    "legacy_artifacts_from_work_dir",
    "scan_legacy_artifacts",
    "migrate_legacy_artifacts",
    "install_compact_artifact_layout",
    "artifact_layout_debug_summary",
]
