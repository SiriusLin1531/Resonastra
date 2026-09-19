from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ============================================================
# Local imports
# 本地导入
# ============================================================
from . import create_data_factory_adapter


UNIFIED_STATUS_SCHEMA_VERSION = "voicelab_data_factory_unified_status_v1"
UNIFIED_STATUS_FILENAME = "data_factory_unified_status.json"


# ============================================================
# Generic helpers
# 通用工具
# ============================================================
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json_dict(path: Optional[str | Path]) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def _file_exists(path: Optional[str | Path]) -> bool:
    return path is not None and Path(path).is_file()


def _file_has_content(path: Optional[str | Path]) -> bool:
    return path is not None and Path(path).is_file() and Path(path).stat().st_size > 0


def _dir_exists(path: Optional[str | Path]) -> bool:
    return path is not None and Path(path).is_dir()


def _count_jsonl_rows(path: Optional[str | Path]) -> Optional[int]:
    if path is None or not Path(path).is_file():
        return None
    count = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_pt_files(path: Optional[str | Path], *, recursive: bool = False) -> Optional[int]:
    if path is None or not Path(path).is_dir():
        return None
    iterator = Path(path).rglob("*.pt") if recursive else Path(path).glob("*.pt")
    return len(list(iterator))


def _status_icon(status: str) -> str:
    if status == "done":
        return "✅"
    if status == "partial":
        return "⚠️"
    if status == "blocked":
        return "⛔"
    return "⬜"


def _completion_ratio(found: Optional[int], expected: Optional[int]) -> Optional[float]:
    if found is None or expected is None or expected <= 0:
        return None
    return min(float(found) / float(expected), 1.0)


def _stage_status(
    *,
    done: bool,
    partial: bool = False,
    blocked: bool = False,
) -> str:
    if done:
        return "done"
    if blocked:
        return "blocked"
    if partial:
        return "partial"
    return "missing"


# ============================================================
# Stage1 paths and scan
# Stage1 路径与扫描
# ============================================================
def stage1_paths_from_work_dir(work_dir: str | Path) -> dict[str, Path]:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    output_root = work_dir / "07_stage1_ft"
    return {
        "output_root": output_root,
        "metadata_dir": output_root / "metadata",
        "frontend_root": output_root / "frontend_cache",
        "semantic_root": output_root / "semantic_cache",
        "train_manifest": output_root / "train_manifest.jsonl",
        "val_manifest": output_root / "val_manifest.jsonl",
        "build_report": output_root / "metadata" / "build_report.json",
    }


def scan_stage1_status(work_dir: str | Path) -> dict[str, Any]:
    paths = stage1_paths_from_work_dir(work_dir)
    build_report = _read_json_dict(paths["build_report"])
    report_status = str(build_report.get("status") or "")
    stats = build_report.get("stats") if isinstance(build_report.get("stats"), dict) else {}

    train_rows = _count_jsonl_rows(paths["train_manifest"])
    val_rows = _count_jsonl_rows(paths["val_manifest"])
    frontend_pt_count = _count_pt_files(paths["frontend_root"], recursive=True)
    semantic_pt_count = _count_pt_files(paths["semantic_root"], recursive=True)

    manifest_done = bool((train_rows or 0) > 0 and val_rows is not None)
    cache_done = bool((frontend_pt_count or 0) > 0 and (semantic_pt_count or 0) > 0)
    done = bool(report_status == "OK" and manifest_done and cache_done)
    partial = bool(
        not done
        and (
            _dir_exists(paths["output_root"])
            or manifest_done
            or cache_done
            or _file_exists(paths["build_report"])
        )
    )

    return {
        "name": "stage1_dataset",
        "display_name": "Stage1 训练数据",
        "status": _stage_status(done=done, partial=partial),
        "done": done,
        "partial": partial,
        "paths": {key: str(value) for key, value in paths.items()},
        "counts": {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "frontend_pt_count": frontend_pt_count,
            "semantic_pt_count": semantic_pt_count,
            "num_input": stats.get("num_input"),
            "num_kept": stats.get("num_kept"),
            "num_skipped_total": build_report.get("num_skipped_total"),
        },
        "build_report_status": report_status or None,
        "build_report": build_report,
    }


# ============================================================
# Unified scanner
# 统一扫描器
# ============================================================
def scan_unified_data_factory_status(work_dir: str | Path, *, write_status_json: bool = True) -> dict[str, Any]:
    """Scan prepare/proofread/Stage1/Stage2 artifacts under one work_dir.

    中文说明：
        统一扫描 DataFactory 用户工作目录下的所有关键产物：
        - 数据工厂 prepare 输出；
        - 人工校对 / 跳过校对输出；
        - Stage1 训练数据；
        - Stage2 few-shot manifest；
        - Stage2 .pt；
        - continuous cache；
        - Style/F0/Speaker cache；
        - train/val；
        - filter report。
    """

    adapter = create_data_factory_adapter()
    resolved_work_dir = adapter.resolve_path(work_dir)
    artifacts = adapter.expected_artifacts(resolved_work_dir)
    stage1 = scan_stage1_status(resolved_work_dir)

    manifest_rows = _count_jsonl_rows(artifacts.manifest_jsonl_path)
    corrected_rows = _count_jsonl_rows(artifacts.manifest_corrected_path)
    fewshot_rows = _count_jsonl_rows(artifacts.stage2_manifest_fewshot_path)

    stage2_pt_count = _count_pt_files(artifacts.stage2_pt_root)
    continuous_count = _count_pt_files(artifacts.continuous_semantic_root)
    style_count = _count_pt_files(artifacts.style_cache_root)
    train_count = _count_pt_files(artifacts.train_pt_root)
    val_count = _count_pt_files(artifacts.val_pt_root)

    prepare_done = bool(_file_has_content(artifacts.manifest_jsonl_path) and _file_has_content(artifacts.dataset_list_path))
    proofread_done = bool(_file_has_content(artifacts.manifest_corrected_path))
    fewshot_done = bool(_file_has_content(artifacts.stage2_manifest_fewshot_path) and (fewshot_rows or 0) > 0)
    stage2_done = bool((fewshot_rows or 0) > 0 and (stage2_pt_count or 0) >= (fewshot_rows or 0))
    stage2_partial = bool(not stage2_done and (stage2_pt_count or 0) > 0)

    continuous_done = bool((stage2_pt_count or 0) > 0 and (continuous_count or 0) >= (stage2_pt_count or 0))
    continuous_partial = bool(not continuous_done and (continuous_count or 0) > 0)

    style_done = bool((continuous_count or 0) > 0 and (style_count or 0) >= (continuous_count or 0))
    style_partial = bool(not style_done and (style_count or 0) > 0)

    split_expected = style_count if style_count is not None and style_count > 0 else None
    split_found = None if train_count is None and val_count is None else (train_count or 0) + (val_count or 0)
    split_done = bool(split_expected and split_found is not None and split_found >= split_expected and _file_has_content(artifacts.split_report_path))
    split_partial = bool(not split_done and split_found is not None and split_found > 0)

    filter_done = bool(_file_has_content(artifacts.filter_report_json_path))

    stages = [
        {
            "name": "prepare",
            "display_name": "执行数据工厂",
            "status": _stage_status(done=prepare_done, partial=bool(manifest_rows)),
            "done": prepare_done,
            "counts": {"manifest_rows": manifest_rows},
            "paths": {
                "manifest_jsonl": str(artifacts.manifest_jsonl_path),
                "dataset_list": str(artifacts.dataset_list_path),
            },
            "message": "检测到 manifest.jsonl 和 dataset.list。" if prepare_done else "尚未检测到完整 prepare 产物。",
        },
        {
            "name": "proofread",
            "display_name": "人工校对 / 跳过校对",
            "status": _stage_status(done=proofread_done, blocked=not prepare_done),
            "done": proofread_done,
            "counts": {"corrected_rows": corrected_rows},
            "paths": {"manifest_corrected": str(artifacts.manifest_corrected_path)},
            "message": "检测到 manifest.corrected.jsonl。" if proofread_done else "尚未检测到 corrected manifest。",
        },
        stage1,
        {
            "name": "stage2_manifest",
            "display_name": "Stage2 few-shot manifest",
            "status": _stage_status(done=fewshot_done, blocked=not proofread_done),
            "done": fewshot_done,
            "counts": {"stage2_manifest_fewshot_rows": fewshot_rows},
            "paths": {"stage2_manifest_fewshot": str(artifacts.stage2_manifest_fewshot_path)},
            "message": "检测到 few-shot Stage2 manifest。" if fewshot_done else "尚未检测到 few-shot Stage2 manifest。",
        },
        {
            "name": "stage2_pt",
            "display_name": "Stage2 .pt",
            "status": _stage_status(done=stage2_done, partial=stage2_partial, blocked=not fewshot_done),
            "done": stage2_done,
            "partial": stage2_partial,
            "ratio": _completion_ratio(stage2_pt_count, fewshot_rows),
            "counts": {"expected": fewshot_rows, "found": stage2_pt_count},
            "paths": {"stage2_pt_root": str(artifacts.stage2_pt_root)},
            "message": f"Stage2 .pt {stage2_pt_count}/{fewshot_rows}" if fewshot_rows else "尚未检测到 Stage2 .pt。",
        },
        {
            "name": "continuous_semantic",
            "display_name": "continuous semantic cache",
            "status": _stage_status(done=continuous_done, partial=continuous_partial, blocked=not (stage2_pt_count or 0)),
            "done": continuous_done,
            "partial": continuous_partial,
            "ratio": _completion_ratio(continuous_count, stage2_pt_count),
            "counts": {"expected": stage2_pt_count, "found": continuous_count},
            "paths": {"continuous_semantic_root": str(artifacts.continuous_semantic_root)},
            "message": f"continuous {continuous_count}/{stage2_pt_count}" if stage2_pt_count else "尚未检测到 continuous cache。",
        },
        {
            "name": "style_cache",
            "display_name": "Style/F0/Speaker cache",
            "status": _stage_status(done=style_done, partial=style_partial, blocked=not (continuous_count or 0)),
            "done": style_done,
            "partial": style_partial,
            "ratio": _completion_ratio(style_count, continuous_count),
            "counts": {"expected": continuous_count, "found": style_count},
            "paths": {"style_cache_root": str(artifacts.style_cache_root)},
            "message": f"style cache {style_count}/{continuous_count}" if continuous_count else "尚未检测到 style cache。",
        },
        {
            "name": "train_val_split",
            "display_name": "train / val split",
            "status": _stage_status(done=split_done, partial=split_partial, blocked=not (style_count or 0)),
            "done": split_done,
            "partial": split_partial,
            "ratio": _completion_ratio(split_found, split_expected),
            "counts": {"expected": split_expected, "train": train_count, "val": val_count, "found": split_found},
            "paths": {
                "train_pt_root": str(artifacts.train_pt_root),
                "val_pt_root": str(artifacts.val_pt_root),
                "split_report": str(artifacts.split_report_path),
            },
            "message": f"train {train_count}, val {val_count}" if split_found is not None else "尚未检测到 train/val。",
        },
        {
            "name": "filter",
            "display_name": "异常样本报告",
            "status": _stage_status(done=filter_done, blocked=not split_done),
            "done": filter_done,
            "counts": {},
            "paths": {
                "filter_report_json": str(artifacts.filter_report_json_path),
                "filter_report_csv": str(artifacts.filter_report_csv_path),
                "rejected_pt_root": str(artifacts.rejected_pt_root),
            },
            "message": "检测到异常样本过滤报告。" if filter_done else "尚未检测到异常样本过滤报告。",
        },
    ]

    next_actions: list[str] = []
    if not prepare_done:
        next_actions.append("先点击“执行数据工厂”。")
    elif not proofread_done:
        next_actions.append("启动人工校对器完成校对，或点击“跳过人工校对”。")
    if prepare_done and proofread_done and not stage1.get("done"):
        next_actions.append("点击“生成 Stage1 训练数据”。")
    if proofread_done and not filter_done:
        next_actions.append("点击“继续生成 Stage2 训练数据”补齐 Stage2 产物。")
    if not next_actions and filter_done and stage1.get("done"):
        next_actions.append("Stage1 / Stage2 数据产物均已就绪，可进入训练阶段。")

    status_json_path = resolved_work_dir / UNIFIED_STATUS_FILENAME
    payload = {
        "schema_version": UNIFIED_STATUS_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "work_dir": str(resolved_work_dir),
        "status_json_path": str(status_json_path),
        "stages": stages,
        "summary": {
            "prepare_done": prepare_done,
            "proofread_done": proofread_done,
            "stage1_done": bool(stage1.get("done")),
            "stage2_manifest_done": fewshot_done,
            "stage2_pt_done": stage2_done,
            "continuous_done": continuous_done,
            "style_done": style_done,
            "split_done": split_done,
            "filter_done": filter_done,
            "all_stage2_done": bool(fewshot_done and stage2_done and continuous_done and style_done and split_done and filter_done),
            "all_training_data_done": bool(stage1.get("done") and fewshot_done and stage2_done and continuous_done and style_done and split_done and filter_done),
        },
        "next_actions": next_actions,
    }

    if write_status_json:
        _write_json(status_json_path, payload)

    return payload


# ============================================================
# Formatting
# 格式化输出
# ============================================================
def format_unified_status_markdown(status_payload: dict[str, Any]) -> str:
    if not status_payload:
        return "暂无统一状态。"

    lines = [
        "### Stage1 / Stage2 数据产物统一状态",
        "",
        f"**工作目录：** `{status_payload.get('work_dir')}`",
        f"**状态文件：** `{status_payload.get('status_json_path')}`",
        "",
        "| 阶段 | 状态 | 数量/说明 |",
        "|---|---:|---|",
    ]

    for stage in status_payload.get("stages") or []:
        status = str(stage.get("status") or "missing")
        icon = _status_icon(status)
        message = stage.get("message") or ""
        counts = stage.get("counts") if isinstance(stage.get("counts"), dict) else {}
        ratio = stage.get("ratio")
        ratio_text = f"，完成度 `{ratio * 100:.1f}%`" if isinstance(ratio, (int, float)) else ""

        count_bits = []
        for key in ["manifest_rows", "corrected_rows", "stage2_manifest_fewshot_rows", "train_rows", "val_rows", "frontend_pt_count", "semantic_pt_count", "expected", "found", "train", "val"]:
            if key in counts and counts.get(key) is not None:
                count_bits.append(f"{key}=`{counts.get(key)}`")
        count_text = "，".join(count_bits)
        detail = count_text or message
        if ratio_text:
            detail = f"{detail}{ratio_text}"
        lines.append(f"| {stage.get('display_name')} | {icon} | {detail} |")

    next_actions = status_payload.get("next_actions") or []
    if next_actions:
        lines.extend(["", "### 下一步建议"])
        for item in next_actions:
            lines.append(f"- {item}")

    return "\n".join(lines)


def format_unified_output_summary(status_payload: dict[str, Any]) -> str:
    if not status_payload:
        return "暂无输出。"

    lines = [
        "### 统一状态输出",
        f"- **work_dir**：`{status_payload.get('work_dir')}`",
        f"- **status_json**：`{status_payload.get('status_json_path')}`",
    ]

    for stage in status_payload.get("stages") or []:
        paths = stage.get("paths") if isinstance(stage.get("paths"), dict) else {}
        for key, value in paths.items():
            if value:
                lines.append(f"- **{stage.get('name')}.{key}**：`{value}`")

    return "\n".join(lines)


def format_unified_step_summary(status_payload: dict[str, Any]) -> str:
    if not status_payload:
        return "暂无步骤记录。"

    status_zh = {
        "done": "已完成",
        "partial": "部分完成",
        "blocked": "等待前置步骤",
        "missing": "缺失",
    }
    lines = []
    for index, stage in enumerate(status_payload.get("stages") or [], start=1):
        status = str(stage.get("status") or "missing")
        lines.append(
            f"{index}. **{stage.get('display_name')}**：{status_zh.get(status, status)}，{stage.get('message') or ''}"
        )
    return "\n".join(lines) if lines else "暂无步骤记录。"


__all__ = [
    "UNIFIED_STATUS_SCHEMA_VERSION",
    "UNIFIED_STATUS_FILENAME",
    "stage1_paths_from_work_dir",
    "scan_stage1_status",
    "scan_unified_data_factory_status",
    "format_unified_status_markdown",
    "format_unified_output_summary",
    "format_unified_step_summary",
]
