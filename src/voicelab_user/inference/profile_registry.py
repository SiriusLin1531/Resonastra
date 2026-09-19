from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.voicelab_user.profiles.profile_loader import load_voice_profile, resolve_model_paths
from src.voicelab_user.runtime.default_checkpoints import default_checkpoint_summary


PROJECT_ROOT = Path(__file__).resolve().parents[3]
USER_PROFILES_DIR_RELATIVE = Path("user_profiles")
PROFILE_MANIFEST_FILENAME = "profile_manifest.json"
PROFILE_CONFIG_FILENAME = "profile_config.json"
ACTIVE_PROFILE_FILENAME = "active_profile.json"
PROFILE_REGISTRY_SCHEMA_VERSION = "voicelab_inference_profile_registry_v3"
ACTIVE_PROFILE_SCHEMA_VERSION = "voicelab_active_inference_profile_v2"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json_dict(path: str | Path) -> dict[str, Any]:
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
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _safe_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _profile_root() -> Path:
    return (PROJECT_ROOT / USER_PROFILES_DIR_RELATIVE).resolve(strict=False)


def _active_profile_path() -> Path:
    return _profile_root() / ACTIVE_PROFILE_FILENAME


def _resolve_profile_info(profile_dir: Path) -> dict[str, Any]:
    manifest_path = profile_dir / PROFILE_MANIFEST_FILENAME
    config_path = profile_dir / PROFILE_CONFIG_FILENAME
    manifest = _read_json_dict(manifest_path)
    config = _read_json_dict(config_path)
    if not manifest and not config:
        return {}

    profile_name = str(manifest.get("profile_name") or config.get("profile_name") or profile_dir.name)
    stage1_declared = ((config.get("stage1") or {}).get("checkpoint_path") if isinstance(config.get("stage1"), dict) else None)
    stage2_declared = ((config.get("stage2") or {}).get("checkpoint_path") if isinstance(config.get("stage2"), dict) else None)
    stage1_path = (profile_dir / str(stage1_declared)).resolve(strict=False) if stage1_declared else None
    stage2_path = (profile_dir / str(stage2_declared)).resolve(strict=False) if stage2_declared else None

    resolved_stage1 = None
    resolved_stage2 = None
    stage1_source = None
    stage2_source = None
    resolve_error = None
    try:
        profile = load_voice_profile(profile_dir)
        resolved = resolve_model_paths(profile)
        resolved_stage1 = resolved.stage1_ckpt
        resolved_stage2 = resolved.stage2_ckpt
        stage1_source = resolved.stage1_source
        stage2_source = resolved.stage2_source
    except Exception as exc:
        resolve_error = f"{type(exc).__name__}: {exc}"

    defaults = default_checkpoint_summary(PROJECT_ROOT)
    stage1_ready = bool(stage1_source == "default_gpt_sovits_v2" and defaults.get("stage1_exists")) or bool(resolved_stage1 and resolved_stage1.is_file())
    stage2_ready = bool(resolved_stage2 and resolved_stage2.is_file())
    ready = bool(stage1_ready and stage2_ready and not resolve_error)
    try:
        modified_at = datetime.fromtimestamp(profile_dir.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        modified_at = ""
    status = "READY" if ready else "NOT_READY"
    return {
        "profile_name": profile_name,
        "profile_type": config.get("profile_type") or manifest.get("profile_type"),
        "schema_version": config.get("schema_version"),
        "profile_root": str(profile_dir),
        "profile_root_relative": _safe_relative(profile_dir),
        "profile_manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "profile_manifest_relative": _safe_relative(manifest_path) if manifest_path.is_file() else None,
        "profile_config_path": str(config_path) if config_path.is_file() else None,
        "profile_config_relative": _safe_relative(config_path) if config_path.is_file() else None,
        "modified_at": modified_at,
        "stage1_ckpt_relative": stage1_declared,
        "stage1_ckpt_path": str(stage1_path) if stage1_path else None,
        "stage1_ckpt_exists": bool(stage1_path and stage1_path.is_file()),
        "stage2_ckpt_relative": stage2_declared,
        "stage2_ckpt_path": str(stage2_path) if stage2_path else None,
        "stage2_ckpt_exists": bool(stage2_path and stage2_path.is_file()),
        "resolved_stage1_ckpt_path": str(resolved_stage1) if resolved_stage1 else None,
        "resolved_stage2_ckpt_path": str(resolved_stage2) if resolved_stage2 else None,
        "stage1_source": stage1_source,
        "stage2_source": stage2_source,
        "stage1_ready": stage1_ready,
        "stage2_ready": stage2_ready,
        "ready_for_inference": ready,
        "resolve_error": resolve_error,
        "profile_manifest": manifest,
        "profile_config": config,
        "label": f"{profile_name} | {status} | {config.get('profile_type') or ''} | {modified_at}",
    }


def _find_profile_info(value: str, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("profile value is empty")
    direct = Path(text).expanduser().resolve(strict=False)
    if direct.is_file():
        info = _resolve_profile_info(direct.parent)
        if info:
            return info
    if direct.is_dir():
        info = _resolve_profile_info(direct)
        if info:
            return info
    for info in profiles:
        candidates = {
            str(info.get("profile_name") or ""), str(info.get("label") or ""),
            str(info.get("profile_manifest_path") or ""), str(info.get("profile_config_path") or ""),
        }
        if text in candidates:
            return info
    raise FileNotFoundError(f"profile not found: {value}")


def scan_user_profiles(*, require_ready: bool = False) -> dict[str, Any]:
    root = _profile_root()
    profiles: list[dict[str, Any]] = []
    if root.is_dir():
        for profile_dir in root.iterdir():
            if not profile_dir.is_dir():
                continue
            info = _resolve_profile_info(profile_dir)
            if not info or (require_ready and not info.get("ready_for_inference")):
                continue
            profiles.append(info)
    profiles.sort(key=lambda x: str(x.get("modified_at") or ""), reverse=True)
    active = _read_json_dict(_active_profile_path())
    active_name = str(active.get("active_profile_name") or "")
    for item in profiles:
        item["is_active"] = bool(active_name and item.get("profile_name") == active_name)
        if item["is_active"]:
            item["label"] = f"{item.get('label')} [ACTIVE]"
    return {
        "schema_version": PROFILE_REGISTRY_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "profiles_root": str(root),
        "active_profile_path": str(_active_profile_path()),
        "active_profile_payload": active,
        "profiles": profiles,
        "num_profiles": len(profiles),
    }


def profile_choices(scan_payload: dict[str, Any] | None = None, *, require_ready: bool = False) -> list[str]:
    payload = scan_payload or scan_user_profiles(require_ready=require_ready)
    return [str(item.get("label") or item.get("profile_name")) for item in payload.get("profiles") or []]


def load_profile_manifest(profile_value: str) -> dict[str, Any]:
    return _find_profile_info(profile_value, scan_user_profiles().get("profiles") or [])


def set_active_profile(profile_value: str | dict[str, Any]) -> dict[str, Any]:
    info = profile_value if isinstance(profile_value, dict) else load_profile_manifest(str(profile_value))
    if not info.get("ready_for_inference"):
        raise ValueError(f"Profile is not ready for inference: {info.get('profile_name')}; {info.get('resolve_error') or ''}")
    payload = {
        "schema_version": ACTIVE_PROFILE_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "active_profile_name": info.get("profile_name"),
        "profile_manifest_path": info.get("profile_manifest_path"),
        "profile_config_path": info.get("profile_config_path"),
        "stage1_ckpt_path": info.get("resolved_stage1_ckpt_path"),
        "stage2_ckpt_path": info.get("resolved_stage2_ckpt_path"),
        "stage1_source": info.get("stage1_source"),
        "stage2_source": info.get("stage2_source"),
        "ready_for_inference": True,
    }
    _write_json(_active_profile_path(), payload)
    return payload


def get_active_profile(*, fallback_to_latest: bool = True, require_ready: bool = True) -> dict[str, Any]:
    active = _read_json_dict(_active_profile_path())
    profiles = scan_user_profiles(require_ready=False).get("profiles") or []
    active_name = active.get("active_profile_name")
    if active_name:
        for item in profiles:
            if item.get("profile_name") == active_name and (not require_ready or item.get("ready_for_inference")):
                return item
    if fallback_to_latest:
        candidates = [p for p in profiles if not require_ready or p.get("ready_for_inference")]
        if candidates:
            return candidates[0]
    return {}


def resolve_inference_profile(profile_value: str | None = None, *, require_stage2: bool = True) -> dict[str, Any]:
    info = load_profile_manifest(profile_value) if profile_value else get_active_profile(require_ready=True)
    if not info:
        raise FileNotFoundError("No inference profile found")
    if not info.get("ready_for_inference"):
        raise FileNotFoundError(f"Profile is not ready: {info.get('profile_name')}; {info.get('resolve_error') or ''}")
    if require_stage2 and not info.get("stage2_ready"):
        raise FileNotFoundError(f"Resolved Stage2 checkpoint missing: {info.get('profile_name')}")
    return {
        "schema_version": "voicelab_resolved_inference_profile_v3",
        "profile_name": info.get("profile_name"),
        "profile_root": info.get("profile_root"),
        "profile_config_path": info.get("profile_config_path"),
        "profile_manifest_path": info.get("profile_manifest_path"),
        "stage1_ckpt": info.get("resolved_stage1_ckpt_path"),
        "stage2_ckpt": info.get("resolved_stage2_ckpt_path"),
        "stage1_source": info.get("stage1_source"),
        "stage2_source": info.get("stage2_source"),
        "stage1_ckpt_exists": info.get("stage1_ready"),
        "stage2_ckpt_exists": info.get("stage2_ready"),
        "ready_for_inference": info.get("ready_for_inference"),
        "profile_config": info.get("profile_config") or {},
    }


def format_profile_registry_markdown(scan_payload: dict[str, Any] | None = None) -> str:
    payload = scan_payload or scan_user_profiles()
    lines = [
        "### Inference Profile Registry v3",
        "",
        f"profiles_root: `{payload.get('profiles_root')}`",
        "",
        "| Profile | Type | Ready | Stage1 source | Stage2 source | Config | Manifest |",
        "|-|-|-|-|-|-|-|",
    ]
    for item in payload.get("profiles") or []:
        lines.append(
            f"| {item.get('profile_name')} | `{item.get('profile_type') or ''}` | "
            f"{'✅' if item.get('ready_for_inference') else '⬜'} | `{item.get('stage1_source') or ''}` | "
            f"`{item.get('stage2_source') or ''}` | {'✅' if item.get('profile_config_path') else '⬜'} | "
            f"{'✅' if item.get('profile_manifest_path') else '⬜'} |"
        )
        if item.get("resolve_error"):
            lines.append(f"| ↳ error |  |  |  | `{item.get('resolve_error')}` |  |  |")
    return "\n".join(lines)


__all__ = [
    "PROJECT_ROOT", "USER_PROFILES_DIR_RELATIVE", "PROFILE_MANIFEST_FILENAME",
    "PROFILE_CONFIG_FILENAME", "ACTIVE_PROFILE_FILENAME", "scan_user_profiles",
    "profile_choices", "load_profile_manifest", "set_active_profile", "get_active_profile",
    "resolve_inference_profile", "format_profile_registry_markdown",
]
