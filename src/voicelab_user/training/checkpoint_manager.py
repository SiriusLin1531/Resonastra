from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.voicelab_user.runtime.default_checkpoints import DEFAULT_PROFILE_NAME, default_checkpoint_summary

from .service import RUNS_DIRNAME, TRAINING_RUN_DIRNAME, TRAINING_RUN_STATE_FILENAME


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt")
BEST_CHECKPOINT_FILENAME = "best_model.pt"
BEST_CHECKPOINT_META_FILENAME = "best_checkpoint.json"
ACTIVE_BEST_FILENAME = "active_best.json"
CHECKPOINT_DELETION_HISTORY_FILENAME = "checkpoint_deletion_history.jsonl"
PROFILE_MANIFEST_FILENAME = "profile_manifest.json"
PROFILE_CONFIG_FILENAME = "profile_config.json"
USER_PROFILES_DIR_RELATIVE = Path("user_profiles")
USER_PROFILE_SCHEMA_VERSION = "voicelab_user_profile_manifest_v2"
VOICE_PROFILE_SCHEMA_VERSION = "voice_profile_v2"
CHECKPOINT_SCAN_SCHEMA_VERSION = "voicelab_training_checkpoint_scan_v2"
CHECKPOINT_DELETION_SCHEMA_VERSION = "voicelab_checkpoint_deletion_v1"
ACTIVE_RUN_STATUSES = frozenset({"pending", "running", "stopping"})

_EPOCH_CHECKPOINT_RE = re.compile(r"^epoch[_-]?(\d+)", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json_value(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_json_dict(path: str | Path) -> dict[str, Any]:
    obj = _read_json_value(path)
    return obj if isinstance(obj, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")
        handle.flush()
    return p


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except Exception:
        return False


def _same_path(a: Path, b: Path) -> bool:
    return a.resolve(strict=False) == b.resolve(strict=False)


def _is_checkpoint_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS


def _stage_paths(work_dir: str | Path, stage: str) -> dict[str, Path]:
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported stage: {stage}")
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    training_root = work_dir / TRAINING_RUN_DIRNAME
    stage_root = training_root / stage
    return {
        "training_root": training_root,
        "stage_root": stage_root,
        "runs_root": stage_root / RUNS_DIRNAME,
        "run_state_path": stage_root / TRAINING_RUN_STATE_FILENAME,
        "best_checkpoint_path": stage_root / BEST_CHECKPOINT_FILENAME,
        "best_checkpoint_meta_path": stage_root / BEST_CHECKPOINT_META_FILENAME,
        "active_best_path": stage_root / ACTIVE_BEST_FILENAME,
        "deletion_history_path": training_root / CHECKPOINT_DELETION_HISTORY_FILENAME,
    }


def _checkpoint_run_context(path: Path) -> tuple[Optional[str], Optional[Path]]:
    resolved = path.resolve(strict=False)
    parts = resolved.parts
    try:
        runs_index = parts.index(RUNS_DIRNAME)
        run_id = parts[runs_index + 1]
        run_root = Path(*parts[: runs_index + 2])
        return run_id, run_root
    except (ValueError, IndexError):
        return None, None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _history_rows(run_root: Path | None) -> list[dict[str, Any]]:
    if run_root is None:
        return []
    history_value = _read_json_value(run_root / "checkpoints" / "history.json")
    if isinstance(history_value, list):
        return [dict(row) for row in history_value if isinstance(row, dict)]
    if isinstance(history_value, dict):
        rows = history_value.get("history")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _summary_dict(run_root: Path | None) -> dict[str, Any]:
    if run_root is None:
        return {}
    return _read_json_dict(run_root / "checkpoints" / "summary.json")


def _checkpoint_epoch_from_name(name: str) -> Optional[int]:
    match = _EPOCH_CHECKPOINT_RE.match(str(name or ""))
    return int(match.group(1)) if match else None


def _row_for_epoch(rows: list[dict[str, Any]], epoch: Optional[int]) -> dict[str, Any]:
    if epoch is None:
        return {}
    for row in rows:
        if _int_or_none(row.get("epoch")) == epoch:
            return row
    return {}


def _checkpoint_metric_metadata(path: Path, *, stage: str) -> dict[str, Any]:
    """Resolve validation metadata from light-weight run sidecars only."""

    run_id, run_root = _checkpoint_run_context(path)
    summary = _summary_dict(run_root)
    rows = _history_rows(run_root)
    name = path.name
    lowered = name.lower()

    checkpoint_epoch: Optional[int] = None
    metric_name = "val_loss_per_token" if stage == "stage1" else "val_loss"
    metric_value: Optional[float] = None
    metric_source = "unavailable"

    if lowered.startswith("best_model"):
        if stage == "stage1":
            summary_metric_name = str(summary.get("best_metric_name") or "")
            if summary_metric_name == "val_loss_per_token":
                checkpoint_epoch = _int_or_none(summary.get("best_epoch"))
                metric_value = _float_or_none(summary.get("best_metric"))
                metric_source = "summary.best_metric"
            else:
                checkpoint_epoch = _int_or_none(summary.get("best_epoch"))
                row = _row_for_epoch(rows, checkpoint_epoch)
                metric_value = _float_or_none(row.get("val_loss_per_token"))
                if metric_value is not None:
                    metric_source = "history.val_loss_per_token"
        else:
            checkpoint_epoch = _int_or_none(summary.get("best_epoch"))
            metric_value = _float_or_none(summary.get("best_val_loss"))
            if metric_value is not None:
                metric_source = "summary.best_val_loss"
            else:
                row = _row_for_epoch(rows, checkpoint_epoch)
                metric_value = _float_or_none(row.get("val_loss"))
                if metric_value is not None:
                    metric_source = "history.val_loss"

    elif lowered.startswith("last_model"):
        row = rows[-1] if rows else {}
        checkpoint_epoch = _int_or_none(row.get("epoch"))
        if checkpoint_epoch is None:
            checkpoint_epoch = _int_or_none(summary.get("last_epoch"))
        metric_value = _float_or_none(row.get(metric_name))
        if metric_value is not None:
            metric_source = f"history.{metric_name}"
        elif stage == "stage2":
            metric_value = _float_or_none(summary.get("val_loss"))
            if metric_value is not None:
                metric_source = "summary.val_loss"

    else:
        checkpoint_epoch = _checkpoint_epoch_from_name(name)
        row = _row_for_epoch(rows, checkpoint_epoch)
        metric_value = _float_or_none(row.get(metric_name))
        if metric_value is not None:
            metric_source = f"history.{metric_name}"

    return {
        "checkpoint_epoch": checkpoint_epoch,
        "validation_metric_name": metric_name,
        "validation_metric_value": metric_value,
        "validation_metric_source": metric_source,
        "validation_metric_available": metric_value is not None,
        "run_id": run_id,
    }


def _checkpoint_info(
    path: Path,
    *,
    work_dir: Path,
    stage: str,
    active_source: Optional[Path],
    active_run_id: Optional[str],
) -> dict[str, Any]:
    stat = path.stat()
    run_id, run_root = _checkpoint_run_context(path)
    resolved = path.resolve(strict=False)
    is_active_source = bool(active_source and _same_path(resolved, active_source))
    is_history_checkpoint = bool(run_id and run_root and _path_within(resolved, _stage_paths(work_dir, stage)["runs_root"]))
    is_active_run_checkpoint = bool(run_id and active_run_id and run_id == active_run_id)
    is_trainer_best = str(path.name).lower().startswith("best_model")
    metric = _checkpoint_metric_metadata(resolved, stage=stage)

    deletion_block_reason = ""
    if not is_history_checkpoint:
        deletion_block_reason = "仅允许删除 runs/<run_id>/ 下的历史 checkpoint。"
    elif is_active_source:
        deletion_block_reason = "当前 checkpoint 是 Active Best source；请先切换 Active Best。"
    elif is_active_run_checkpoint:
        deletion_block_reason = "当前 checkpoint 属于正在运行的训练 run，训练结束后才能删除。"

    return {
        "name": path.name,
        "path": str(resolved),
        "relative_path": _safe_relative(resolved, work_dir),
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "stage": stage,
        "run_id": run_id,
        "is_best": is_active_source,
        "is_active_source": is_active_source,
        "is_history_checkpoint": is_history_checkpoint,
        "is_active_run_checkpoint": is_active_run_checkpoint,
        "is_trainer_best": is_trainer_best,
        "deletion_allowed": not bool(deletion_block_reason),
        "deletion_block_reason": deletion_block_reason,
        "deletion_warning": (
            "这是该 run 的 Trainer Best；删除后该 run 将失去其最佳模型文件。"
            if is_trainer_best and not is_active_source
            else ""
        ),
        "checkpoint_epoch": metric.get("checkpoint_epoch"),
        "validation_metric_name": metric.get("validation_metric_name"),
        "validation_metric_value": metric.get("validation_metric_value"),
        "validation_metric_source": metric.get("validation_metric_source"),
        "validation_metric_available": bool(metric.get("validation_metric_available")),
    }


def _active_source_path(active_meta: dict[str, Any]) -> Optional[Path]:
    text = str(active_meta.get("source_checkpoint") or "").strip()
    return Path(text).expanduser().resolve(strict=False) if text else None


def _metric_ui_name(stage: str) -> str:
    return "Val Loss / Token" if stage == "stage1" else "Val Loss"


def _metric_text(item: dict[str, Any], *, stage: str) -> str:
    value = _float_or_none(item.get("validation_metric_value"))
    return "N/A" if value is None else f"{value:.6f}"


def _scan_stage_checkpoints(work_dir: Path, stage: str) -> list[dict[str, Any]]:
    paths = _stage_paths(work_dir, stage)
    active_meta = _read_json_dict(paths["active_best_path"]) or _read_json_dict(paths["best_checkpoint_meta_path"])
    active_source = _active_source_path(active_meta)
    current_state = _read_json_dict(paths["run_state_path"])
    current_status = str(current_state.get("status") or "").strip().lower()
    active_run_id = str(current_state.get("run_id") or "").strip() if current_status in ACTIVE_RUN_STATUSES else ""

    candidates: list[Path] = []
    if paths["runs_root"].is_dir():
        candidates.extend(path for path in paths["runs_root"].rglob("*") if _is_checkpoint_file(path))
    legacy_dir = paths["stage_root"] / "checkpoints"
    if legacy_dir.is_dir():
        candidates.extend(path for path in legacy_dir.rglob("*") if _is_checkpoint_file(path))

    unique = {str(path.resolve(strict=False)): path.resolve(strict=False) for path in candidates}
    items = [
        _checkpoint_info(
            path,
            work_dir=work_dir,
            stage=stage,
            active_source=active_source,
            active_run_id=active_run_id or None,
        )
        for path in unique.values()
    ]
    items.sort(key=lambda item: str(item.get("modified_at") or ""), reverse=True)

    for item in items:
        size_mb = float(item.get("size_bytes") or 0) / 1024.0 / 1024.0
        tags = []
        if item.get("is_active_source"):
            tags.append("ACTIVE BEST")
        if item.get("is_trainer_best"):
            tags.append("TRAINER BEST")
        if str(item.get("name") or "").startswith("last_model"):
            tags.append("LAST")
        if item.get("is_active_run_checkpoint"):
            tags.append("RUNNING RUN")
        if not item.get("deletion_allowed"):
            tags.append("DELETE PROTECTED")
        suffix = f" [{' | '.join(tags)}]" if tags else ""
        run_label = f"run={item.get('run_id')} | " if item.get("run_id") else ""
        epoch_text = item.get("checkpoint_epoch")
        epoch_label = f"epoch={epoch_text} | " if epoch_text is not None else ""
        metric_label = f"{_metric_ui_name(stage)}={_metric_text(item, stage=stage)}"
        item["label"] = (
            f"{run_label}{item.get('name')} | {epoch_label}{metric_label} | "
            f"{size_mb:.1f} MB{suffix}"
        )
    return items


def _find_checkpoint_by_path_or_label(stage_payload: dict[str, Any], checkpoint_value: str) -> Path:
    text = str(checkpoint_value or "").strip()
    if not text:
        raise ValueError("checkpoint value is empty.")
    direct = Path(text).expanduser().resolve(strict=False)
    if direct.is_file():
        return direct
    for item in stage_payload.get("checkpoints") or []:
        candidates = {str(item.get(key) or "") for key in ("path", "relative_path", "label", "name")}
        if text in candidates:
            path = Path(str(item.get("path"))).expanduser().resolve(strict=False)
            if path.is_file():
                return path
    raise FileNotFoundError(f"checkpoint not found: {checkpoint_value}")


def _find_scanned_checkpoint_item(stage_payload: dict[str, Any], checkpoint_value: str) -> dict[str, Any]:
    text = str(checkpoint_value or "").strip()
    if not text:
        raise ValueError("checkpoint value is empty.")
    for item in stage_payload.get("checkpoints") or []:
        if not isinstance(item, dict):
            continue
        candidates = {str(item.get(key) or "") for key in ("path", "relative_path", "label", "name")}
        if text in candidates:
            return dict(item)
    raise FileNotFoundError("所选 checkpoint 不在当前可管理的历史 checkpoint 列表中。")


def _copy_active_best(work_dir: Path, stage: str, selected: Path, meta: dict[str, Any]) -> dict[str, Any]:
    paths = _stage_paths(work_dir, stage)
    stable_path = paths["best_checkpoint_path"]
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    if selected.resolve(strict=False) != stable_path.resolve(strict=False):
        shutil.copy2(selected, stable_path)

    metric = _checkpoint_metric_metadata(selected, stage=stage)
    source_run_id = meta.get("source_run_id") or metric.get("run_id")
    payload = {
        "schema_version": "voicelab_active_best_v2",
        "created_at": meta.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
        "stage": stage,
        "work_dir": str(work_dir),
        "selection_mode": str(meta.get("selection_mode") or "automatic"),
        "selected_by": str(meta.get("selected_by") or "training_worker"),
        "source_run_id": source_run_id,
        "source_checkpoint": str(selected),
        "active_checkpoint": str(stable_path),
        "copy_mode": True,
        "checkpoint_epoch": metric.get("checkpoint_epoch"),
        "validation_metric_name": metric.get("validation_metric_name"),
        "validation_metric_value": metric.get("validation_metric_value"),
        "validation_metric_source": metric.get("validation_metric_source"),
        "validation_metric_available": bool(metric.get("validation_metric_available")),
    }
    _write_json(paths["active_best_path"], payload)
    _write_json(paths["best_checkpoint_meta_path"], payload)
    return payload


def register_trainer_best(work_dir: str | Path, *, stage: str, run_id: str, checkpoint_path: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    selected = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not _is_checkpoint_file(selected):
        raise FileNotFoundError(f"Trainer best checkpoint not found: {selected}")
    existing = _read_json_dict(_stage_paths(work_dir, stage)["active_best_path"])
    if existing.get("selection_mode") == "manual":
        return {
            "registered": False,
            "reason": "manual_active_best_locked",
            "active_best": existing,
            "trainer_best": str(selected),
            "run_id": run_id,
        }
    payload = _copy_active_best(
        work_dir,
        stage,
        selected,
        {"selection_mode": "automatic", "selected_by": "training_worker", "source_run_id": run_id},
    )
    return {"registered": True, "reason": "automatic_trainer_best", "active_best": payload}


def mark_best_checkpoint(
    work_dir: str | Path,
    *,
    stage: str,
    checkpoint_value: str,
    copy_mode: bool = True,
) -> dict[str, Any]:
    del copy_mode
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    scan = scan_training_outputs(work_dir)
    selected = _find_checkpoint_by_path_or_label(scan["stages"][stage], checkpoint_value)
    info = next((item for item in scan["stages"][stage]["checkpoints"] if item.get("path") == str(selected)), {})
    return _copy_active_best(
        work_dir,
        stage,
        selected,
        {"selection_mode": "manual", "selected_by": "user", "source_run_id": info.get("run_id")},
    )


def restore_automatic_best(work_dir: str | Path, *, stage: str) -> dict[str, Any]:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    items = _scan_stage_checkpoints(work_dir, stage)
    trainer_best = next((item for item in items if str(item.get("name") or "").startswith("best_model")), None)
    if trainer_best is None:
        raise FileNotFoundError(f"No trainer-produced best_model found for {stage}")
    return _copy_active_best(
        work_dir,
        stage,
        Path(str(trainer_best["path"])),
        {"selection_mode": "automatic", "selected_by": "restore_automatic", "source_run_id": trainer_best.get("run_id")},
    )


def _enrich_active_meta(active_meta: dict[str, Any], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    if not active_meta:
        return {}
    enriched = dict(active_meta)
    source = str(enriched.get("source_checkpoint") or "").strip()
    source_path = str(Path(source).expanduser().resolve(strict=False)) if source else ""
    match = next(
        (
            item
            for item in checkpoints
            if source_path and str(item.get("path") or "") == source_path
        ),
        None,
    )
    if isinstance(match, dict):
        for key in (
            "checkpoint_epoch",
            "validation_metric_name",
            "validation_metric_value",
            "validation_metric_source",
            "validation_metric_available",
        ):
            if enriched.get(key) is None or key not in enriched:
                enriched[key] = match.get(key)
    return enriched


def scan_training_outputs(work_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCAN_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "work_dir": str(work_dir),
        "deletion_history_path": str(work_dir / TRAINING_RUN_DIRNAME / CHECKPOINT_DELETION_HISTORY_FILENAME),
        "stages": {},
    }
    for stage in ("stage1", "stage2"):
        paths = _stage_paths(work_dir, stage)
        checkpoints = _scan_stage_checkpoints(work_dir, stage)
        active_meta = _enrich_active_meta(
            _read_json_dict(paths["active_best_path"]),
            checkpoints,
        )
        runs = []
        if paths["runs_root"].is_dir():
            for run_dir in sorted(paths["runs_root"].iterdir(), key=lambda p: p.name, reverse=True):
                if run_dir.is_dir():
                    runs.append({"run_id": run_dir.name, "run_root": str(run_dir), "run_state": _read_json_dict(run_dir / TRAINING_RUN_STATE_FILENAME)})
        payload["stages"][stage] = {
            "stage": stage,
            "stage_root": str(paths["stage_root"]),
            "runs_root": str(paths["runs_root"]),
            "run_state_path": str(paths["run_state_path"]),
            "run_state": _read_json_dict(paths["run_state_path"]),
            "best_checkpoint_path": str(paths["best_checkpoint_path"]),
            "best_checkpoint_exists": paths["best_checkpoint_path"].is_file(),
            "active_best_path": str(paths["active_best_path"]),
            "active_best": active_meta,
            "selection_mode": active_meta.get("selection_mode"),
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
            "runs": runs,
            "num_runs": len(runs),
        }
    return payload


def checkpoint_choices(scan_payload: dict[str, Any], stage: str) -> list[str]:
    stage_payload = (scan_payload.get("stages") or {}).get(stage) or {}
    return [str(item.get("label") or item.get("path")) for item in stage_payload.get("checkpoints") or []]


def delete_checkpoint(
    work_dir: str | Path,
    *,
    stage: str,
    checkpoint_value: str,
    confirmed: bool,
) -> dict[str, Any]:
    """Permanently delete one historical checkpoint under runs/<run_id>/.

    Safety rules are re-validated immediately before unlinking. Run metadata,
    history, summary and logs are intentionally preserved.
    """

    if not bool(confirmed):
        raise PermissionError("请先勾选永久删除确认框。")

    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    paths = _stage_paths(work_dir, stage)
    scan = scan_training_outputs(work_dir)
    stage_payload = (scan.get("stages") or {}).get(stage) or {}
    item = _find_scanned_checkpoint_item(stage_payload, checkpoint_value)
    selected = Path(str(item.get("path") or "")).expanduser().resolve(strict=False)

    if not selected.is_file():
        raise FileNotFoundError("所选 checkpoint 已不存在，请刷新列表后重试。")
    if not _path_within(selected, paths["runs_root"]):
        raise PermissionError("安全策略仅允许删除 runs/<run_id>/ 下的历史 checkpoint。")
    if _same_path(selected, paths["best_checkpoint_path"]):
        raise PermissionError("Stage 根目录的稳定 Active Best 工作副本禁止删除。")

    active_meta = _read_json_dict(paths["active_best_path"])
    active_source = _active_source_path(active_meta)
    if active_source and _same_path(selected, active_source):
        raise PermissionError("所选 checkpoint 是当前 Active Best source；请先切换到其它 Active Best。")

    current_state = _read_json_dict(paths["run_state_path"])
    current_status = str(current_state.get("status") or "").strip().lower()
    current_run_id = str(current_state.get("run_id") or "").strip()
    selected_run_id = str(item.get("run_id") or "").strip()
    if current_status in ACTIVE_RUN_STATUSES and selected_run_id and selected_run_id == current_run_id:
        raise PermissionError("所选 checkpoint 属于正在运行的训练 run，训练结束后才能删除。")

    if not bool(item.get("deletion_allowed")):
        raise PermissionError(str(item.get("deletion_block_reason") or "该 checkpoint 当前受保护，不能删除。"))

    size_bytes = int(selected.stat().st_size)
    deletion_record = {
        "schema_version": CHECKPOINT_DELETION_SCHEMA_VERSION,
        "deleted_at": _now_iso(),
        "stage": stage,
        "run_id": selected_run_id or None,
        "checkpoint_name": selected.name,
        "checkpoint_relative_path": _safe_relative(selected, work_dir),
        "size_bytes": size_bytes,
        "checkpoint_epoch": item.get("checkpoint_epoch"),
        "validation_metric_name": item.get("validation_metric_name"),
        "validation_metric_value": item.get("validation_metric_value"),
        "validation_metric_source": item.get("validation_metric_source"),
        "was_trainer_best": bool(item.get("is_trainer_best")),
        "was_active_best_source": False,
        "confirmation_required": True,
        "confirmation_received": True,
        "run_metadata_preserved": True,
    }

    selected.unlink()
    audit_path = _append_jsonl(paths["deletion_history_path"], deletion_record)
    deletion_record["audit_log_path"] = str(audit_path)
    deletion_record["deleted"] = True
    deletion_record["released_bytes"] = size_bytes
    return deletion_record


def format_checkpoint_status_markdown(scan_payload: dict[str, Any]) -> str:
    if not scan_payload:
        return "暂无 checkpoint 状态。"

    stages = scan_payload.get("stages") if isinstance(scan_payload.get("stages"), dict) else {}
    lines = [
        "### Checkpoint 状态",
        "",
        "| Stage | Runs | Checkpoints | Active Best | 模式 | Active Validation Loss |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for stage in ("stage1", "stage2"):
        item = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        meta = item.get("active_best") if isinstance(item.get("active_best"), dict) else {}
        metric = _metric_text(meta, stage=stage) if meta else "N/A"
        lines.append(
            f"| {stage} | `{item.get('num_runs') or 0}` | `{item.get('num_checkpoints') or 0}` | "
            f"{'✅' if item.get('best_checkpoint_exists') else '⬜'} | `{item.get('selection_mode') or 'zero-shot'}` | "
            f"`{_metric_ui_name(stage)}={metric}` |"
        )

    lines.extend(["", "### 当前 Active Best"])
    for stage in ("stage1", "stage2"):
        item = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        meta = item.get("active_best") if isinstance(item.get("active_best"), dict) else {}
        source_name = Path(str(meta.get("source_checkpoint") or "")).name if meta.get("source_checkpoint") else "zero-shot default"
        epoch = meta.get("checkpoint_epoch")
        epoch_text = str(epoch) if epoch is not None else "N/A"
        metric = _metric_text(meta, stage=stage) if meta else "N/A"
        lines.append(
            f"- **{stage}**：run=`{meta.get('source_run_id') or '-'}`，checkpoint=`{source_name}`，"
            f"epoch=`{epoch_text}`，{_metric_ui_name(stage)}=`{metric}`"
        )

    protected_counts = {}
    for stage in ("stage1", "stage2"):
        item = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        checkpoints = item.get("checkpoints") if isinstance(item.get("checkpoints"), list) else []
        protected_counts[stage] = sum(1 for checkpoint in checkpoints if isinstance(checkpoint, dict) and not checkpoint.get("deletion_allowed"))
    lines.extend(
        [
            "",
            "### 删除保护",
            f"- Stage1：`{protected_counts.get('stage1', 0)}` 个 checkpoint 当前受保护",
            f"- Stage2：`{protected_counts.get('stage2', 0)}` 个 checkpoint 当前受保护",
            "- Active Best source 与正在训练 run 的 checkpoint 不允许删除；删除仅作用于历史模型文件，run metadata 与日志会保留。",
        ]
    )
    return "\n".join(lines)


def _copy_profile_checkpoint(profile_root: Path, stage: str, source: Optional[Path]) -> Optional[dict[str, Any]]:
    if source is None or not source.is_file():
        return None
    dst = profile_root / stage / BEST_CHECKPOINT_FILENAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    return {
        "source": str(source),
        "destination": str(dst),
        "relative_destination": _safe_relative(dst, PROJECT_ROOT),
        "profile_relative_destination": _safe_relative(dst, profile_root),
    }


def _profile_type(stage1_custom: bool, stage2_custom: bool) -> str:
    if stage1_custom and stage2_custom:
        return "few_shot_dual"
    if stage1_custom:
        return "few_shot_stage1_only"
    if stage2_custom:
        return "few_shot_stage2_only"
    return "base_zeroshot"


def generate_user_profile(
    work_dir: str | Path,
    *,
    profile_name: str,
    include_stage1: bool = True,
    include_stage2: bool = True,
    force_default_stage1: bool = False,
    force_default_stage2: bool = False,
    overwrite: bool = True,
) -> dict[str, Any]:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(profile_name).strip())
    if not safe_name:
        raise ValueError("profile_name is empty.")
    profile_root = PROJECT_ROOT / USER_PROFILES_DIR_RELATIVE / safe_name
    if profile_root.exists() and not overwrite:
        raise FileExistsError(f"Profile already exists: {profile_root}")
    profile_root.mkdir(parents=True, exist_ok=True)

    scan = scan_training_outputs(work_dir)
    stage1_active = Path(str(scan["stages"]["stage1"]["best_checkpoint_path"]))
    stage2_active = Path(str(scan["stages"]["stage2"]["best_checkpoint_path"]))
    use_stage1 = bool(include_stage1 and not force_default_stage1 and stage1_active.is_file())
    use_stage2 = bool(include_stage2 and not force_default_stage2 and stage2_active.is_file())
    copied: dict[str, Any] = {}
    stage1_copy = _copy_profile_checkpoint(profile_root, "stage1", stage1_active if use_stage1 else None)
    stage2_copy = _copy_profile_checkpoint(profile_root, "stage2", stage2_active if use_stage2 else None)
    if stage1_copy:
        copied["stage1"] = stage1_copy
    if stage2_copy:
        copied["stage2"] = stage2_copy

    defaults = default_checkpoint_summary(PROJECT_ROOT)
    profile_type = _profile_type(bool(stage1_copy), bool(stage2_copy))
    profile_config = {
        "schema_version": VOICE_PROFILE_SCHEMA_VERSION,
        "profile_name": safe_name,
        "display_name": f"VoiceLab 声音角色 - {safe_name}",
        "description": "由 VoiceLab Training UI 生成；未微调的 Stage 自动回退 zero-shot 默认 checkpoint。",
        "language": "zh",
        "profile_type": profile_type,
        "fallback_profile": DEFAULT_PROFILE_NAME,
        "stage1": {
            "mode": "few_shot_checkpoint" if stage1_copy else "default_checkpoint",
            "checkpoint_path": "stage1/best_model.pt" if stage1_copy else None,
            "fallback_profile": DEFAULT_PROFILE_NAME,
            "display_name": "Few-shot 文本转语义模型" if stage1_copy else "默认文本转语义模型",
        },
        "stage2": {
            "mode": "few_shot_checkpoint" if stage2_copy else "default_checkpoint",
            "checkpoint_path": "stage2/best_model.pt" if stage2_copy else None,
            "fallback_profile": DEFAULT_PROFILE_NAME,
            "model_family": "voicelab_stage2_fewshot_zh" if stage2_copy else "voicelab_stage2_v65_zh_base",
            "acoustic_generation_mode": "standard",
            "internal_sampling_mode": "coarse_only",
            "display_name": "Few-shot 声学生成模型" if stage2_copy else "默认声学生成模型",
        },
        "vocoder": {"type": "hifigan", "profile": "universal_v1", "display_name": "HiFi-GAN 默认声码器"},
        "capabilities": {
            "supports_zero_shot": True,
            "supports_partial_few_shot": True,
            "supports_few_shot_checkpoint_override": True,
            "supports_training": True,
            "language": ["zh"],
            "offline": True,
            "ready_for_inference": bool(defaults.get("ready")),
        },
        "ui": {
            "show_stage1_checkpoint_picker": False,
            "show_stage2_checkpoint_picker": False,
            "show_advanced_sampling": True,
            "show_force_default_stage_controls": True,
        },
        "training_origin": {
            "source_work_dir": str(work_dir),
            "source_work_dir_relative": _safe_relative(work_dir, PROJECT_ROOT),
            "profile_root_relative": _safe_relative(profile_root, PROJECT_ROOT),
            "copied_checkpoints": copied,
            "created_at": _now_iso(),
        },
    }
    profile_config_path = _write_json(profile_root / PROFILE_CONFIG_FILENAME, profile_config)
    manifest = {
        "schema_version": USER_PROFILE_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "profile_name": safe_name,
        "profile_type": profile_type,
        "profile_root": str(profile_root),
        "profile_root_relative": _safe_relative(profile_root, PROJECT_ROOT),
        "source_work_dir": str(work_dir),
        "copied_checkpoints": copied,
        "default_checkpoints": defaults,
        "ready_for_inference": bool(defaults.get("ready")),
        "voice_profile_schema_version": VOICE_PROFILE_SCHEMA_VERSION,
        "profile_config_path": str(profile_config_path),
        "profile_config_relative": _safe_relative(profile_config_path, PROJECT_ROOT),
        "inference_contract": {
            "profile_dir": _safe_relative(profile_root, PROJECT_ROOT),
            "profile_config": _safe_relative(profile_config_path, PROJECT_ROOT),
            "stage1_ckpt": copied.get("stage1", {}).get("relative_destination"),
            "stage2_ckpt": copied.get("stage2", {}).get("relative_destination"),
            "stage1_fallback": None if stage1_copy else "default_zh",
            "stage2_fallback": None if stage2_copy else "default_zh",
        },
    }
    manifest_path = _write_json(profile_root / PROFILE_MANIFEST_FILENAME, manifest)
    manifest["profile_manifest_path"] = str(manifest_path)
    manifest["profile_manifest_relative"] = _safe_relative(manifest_path, PROJECT_ROOT)
    return manifest


__all__ = [
    "PROJECT_ROOT", "CHECKPOINT_EXTENSIONS", "BEST_CHECKPOINT_FILENAME",
    "BEST_CHECKPOINT_META_FILENAME", "ACTIVE_BEST_FILENAME", "CHECKPOINT_DELETION_HISTORY_FILENAME",
    "PROFILE_MANIFEST_FILENAME", "PROFILE_CONFIG_FILENAME", "USER_PROFILE_SCHEMA_VERSION", "VOICE_PROFILE_SCHEMA_VERSION",
    "scan_training_outputs", "format_checkpoint_status_markdown", "checkpoint_choices", "delete_checkpoint",
    "register_trainer_best", "mark_best_checkpoint", "restore_automatic_best", "generate_user_profile",
]
