from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from src.voicelab_user.runtime.default_checkpoints import resolve_default_checkpoints

from ..runtime.path_resolver import (
    PathLike,
    ensure_dir_exists,
    ensure_file_exists,
    normalize_path_text,
    resolve_profile_relative_path,
    resolve_project_path,
    safe_relpath,
)


class VoiceProfileError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceProfile:
    profile_dir: Path
    profile_name: str
    display_name: str
    description: Optional[str]
    language: str
    profile_type: str
    stage1: dict[str, Any]
    stage2: dict[str, Any]
    vocoder: dict[str, Any]
    capabilities: dict[str, Any]
    ui: dict[str, Any]
    raw_config: dict[str, Any]
    schema_version: str = "voice_profile_v1"
    fallback_profile: str = "default_zh"


@dataclass(frozen=True)
class ResolvedModelPaths:
    stage1_ckpt: Optional[Path]
    stage1_source: str
    stage2_ckpt: Path
    stage2_source: str
    sovits_ckpt: Optional[Path]
    vocoder_type: str
    vocoder_profile: str
    hifigan_root: Optional[Path]


_VALID_STAGE1_SOURCES = {"default_gpt_sovits_v2", "profile", "user_override"}
_VALID_STAGE2_SOURCES = {"profile", "default_config", "default_zeroshot", "user_override"}
_SUPPORTED_SCHEMAS = {"voice_profile_v1", "voice_profile_v2"}


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到 JSON 配置文件：\n  {path}")
    except json.JSONDecodeError as exc:
        raise VoiceProfileError(
            f"JSON 配置文件格式错误：\n  {path}\n\n"
            f"错误位置：line {exc.lineno}, column {exc.colno}\n错误信息：{exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise VoiceProfileError(f"JSON 配置文件顶层必须是对象 dict：\n  {path}")
    return data


def _require_dict(data: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise VoiceProfileError(f"{context} 缺少必需对象字段：{key}")
    return value


def _require_non_empty_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VoiceProfileError(f"{context} 缺少必需字符串字段：{key}")
    return value.strip()


def load_voice_profile(profile_dir: PathLike) -> VoiceProfile:
    resolved_profile_dir = resolve_project_path(profile_dir)
    ensure_dir_exists(resolved_profile_dir, label="声音模型 profile")
    assert resolved_profile_dir is not None
    config_path = resolved_profile_dir / "profile_config.json"
    ensure_file_exists(config_path, label="声音模型 profile_config.json")
    raw = load_json(config_path)
    schema_version = str(raw.get("schema_version") or "")
    if schema_version not in _SUPPORTED_SCHEMAS:
        raise VoiceProfileError(
            f"不支持的 voice profile schema_version：{schema_version!r}\n"
            f"当前支持：{', '.join(sorted(_SUPPORTED_SCHEMAS))}。"
        )
    profile = VoiceProfile(
        profile_dir=resolved_profile_dir,
        profile_name=_require_non_empty_str(raw, "profile_name", context="voice profile"),
        display_name=_require_non_empty_str(raw, "display_name", context="voice profile"),
        description=raw.get("description") if isinstance(raw.get("description"), str) else None,
        language=_require_non_empty_str(raw, "language", context="voice profile"),
        profile_type=_require_non_empty_str(raw, "profile_type", context="voice profile"),
        stage1=_require_dict(raw, "stage1", context="voice profile"),
        stage2=_require_dict(raw, "stage2", context="voice profile"),
        vocoder=_require_dict(raw, "vocoder", context="voice profile"),
        capabilities=_require_dict(raw, "capabilities", context="voice profile"),
        ui=raw.get("ui") if isinstance(raw.get("ui"), dict) else {},
        raw_config=raw,
        schema_version=schema_version,
        fallback_profile=str(raw.get("fallback_profile") or "default_zh"),
    )
    validate_voice_profile(profile)
    return profile


def validate_voice_profile(profile: VoiceProfile) -> None:
    if profile.language != "zh":
        raise VoiceProfileError("VoiceLab 用户版当前仅支持中文 profile：language 必须为 zh。")
    stage1_mode = profile.stage1.get("mode")
    if not isinstance(stage1_mode, str) or not stage1_mode.strip():
        raise VoiceProfileError("voice profile 的 stage1.mode 不能为空。")
    if profile.schema_version == "voice_profile_v1":
        stage2_checkpoint = normalize_path_text(profile.stage2.get("checkpoint_path"))
        if stage2_checkpoint is None:
            raise VoiceProfileError("voice_profile_v1 的 stage2.checkpoint_path 不能为空。")
    else:
        stage2_mode = str(profile.stage2.get("mode") or "").strip()
        if stage2_mode not in {"few_shot_checkpoint", "default_checkpoint"}:
            raise VoiceProfileError("voice_profile_v2 的 stage2.mode 必须为 few_shot_checkpoint 或 default_checkpoint。")
        if stage2_mode == "few_shot_checkpoint" and normalize_path_text(profile.stage2.get("checkpoint_path")) is None:
            raise VoiceProfileError("stage2.mode=few_shot_checkpoint 时 checkpoint_path 不能为空。")
    if profile.vocoder.get("type", "hifigan") != "hifigan":
        raise VoiceProfileError("VoiceLab 用户版当前仅支持 hifigan vocoder。")
    if not str(profile.vocoder.get("profile") or "").strip():
        raise VoiceProfileError("voice profile 的 vocoder.profile 不能为空。")


def resolve_stage1_checkpoint(profile: VoiceProfile, user_override: Optional[PathLike]) -> Tuple[Optional[Path], str]:
    override_text = normalize_path_text(user_override)
    if override_text is not None:
        override_path = resolve_project_path(override_text)
        ensure_file_exists(override_path, label="自定义 Stage1 文本转语义模型")
        return override_path, "user_override"
    profile_text = normalize_path_text(profile.stage1.get("checkpoint_path"))
    if profile_text is not None:
        profile_path = resolve_profile_relative_path(profile_text, profile_dir=profile.profile_dir)
        ensure_file_exists(profile_path, label="profile Stage1 文本转语义模型")
        return profile_path, "profile"
    return None, "default_gpt_sovits_v2"


def resolve_stage2_checkpoint(
    profile: VoiceProfile,
    user_override: Optional[PathLike],
    *,
    default_config: Optional[dict[str, Any]] = None,
) -> Tuple[Path, str]:
    override_text = normalize_path_text(user_override)
    if override_text is not None:
        override_path = resolve_project_path(override_text)
        ensure_file_exists(override_path, label="自定义 Stage2 声学生成模型")
        assert override_path is not None
        return override_path, "user_override"
    profile_text = normalize_path_text(profile.stage2.get("checkpoint_path"))
    if profile_text is not None:
        profile_path = resolve_profile_relative_path(profile_text, profile_dir=profile.profile_dir)
        ensure_file_exists(profile_path, label="profile Stage2 声学生成模型")
        assert profile_path is not None
        return profile_path, "profile"
    if default_config is not None and isinstance(default_config.get("stage2"), dict):
        default_text = normalize_path_text(default_config["stage2"].get("checkpoint_path"))
        if default_text is not None:
            default_path = resolve_project_path(default_text)
            ensure_file_exists(default_path, label="默认 Stage2 声学生成模型")
            assert default_path is not None
            return default_path, "default_config"
    defaults = resolve_default_checkpoints(validate=False)
    ensure_file_exists(defaults.stage2_checkpoint, label="默认 Stage2 声学生成模型")
    return defaults.stage2_checkpoint, "default_zeroshot"


def resolve_vocoder_settings(profile: VoiceProfile, default_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = default_config.get("vocoder", {}) if default_config and isinstance(default_config.get("vocoder"), dict) else {}
    vocoder_type = profile.vocoder.get("type") or defaults.get("type") or "hifigan"
    vocoder_profile = profile.vocoder.get("profile") or defaults.get("profile") or "universal_v1"
    hifigan_text = normalize_path_text(defaults.get("hifigan_root"))
    return {
        "type": vocoder_type,
        "profile": vocoder_profile,
        "hifigan_root": resolve_project_path(hifigan_text) if hifigan_text else None,
        "checkpoint_path": resolve_project_path(defaults.get("checkpoint_path")),
        "config_path": resolve_project_path(defaults.get("config_path")),
    }


def resolve_model_paths(
    profile: VoiceProfile,
    *,
    user_stage1_ckpt: Optional[PathLike] = None,
    user_stage2_ckpt: Optional[PathLike] = None,
    default_config: Optional[dict[str, Any]] = None,
) -> ResolvedModelPaths:
    stage1_ckpt, stage1_source = resolve_stage1_checkpoint(profile, user_stage1_ckpt)
    stage2_ckpt, stage2_source = resolve_stage2_checkpoint(profile, user_stage2_ckpt, default_config=default_config)
    if stage1_source not in _VALID_STAGE1_SOURCES:
        raise VoiceProfileError(f"未知 Stage1 来源：{stage1_source}")
    if stage2_source not in _VALID_STAGE2_SOURCES:
        raise VoiceProfileError(f"未知 Stage2 来源：{stage2_source}")
    vocoder = resolve_vocoder_settings(profile, default_config=default_config)
    return ResolvedModelPaths(
        stage1_ckpt=stage1_ckpt,
        stage1_source=stage1_source,
        stage2_ckpt=stage2_ckpt,
        stage2_source=stage2_source,
        sovits_ckpt=None,
        vocoder_type=str(vocoder["type"]),
        vocoder_profile=str(vocoder["profile"]),
        hifigan_root=vocoder["hifigan_root"],
    )


def detect_inference_mode(paths: ResolvedModelPaths) -> str:
    stage1_custom = paths.stage1_source in {"profile", "user_override"}
    stage2_custom = paths.stage2_source in {"profile", "user_override"}
    if stage1_custom and stage2_custom:
        return "few_shot_dual"
    if stage1_custom:
        return "few_shot_stage1_only"
    if stage2_custom:
        return "few_shot_stage2_only"
    return "zero_shot_default"


def profile_to_summary_dict(profile: VoiceProfile, paths: ResolvedModelPaths) -> dict[str, Any]:
    return {
        "profile_name": profile.profile_name,
        "display_name": profile.display_name,
        "language": profile.language,
        "profile_type": profile.profile_type,
        "schema_version": profile.schema_version,
        "stage1_source": paths.stage1_source,
        "stage1_ckpt": None if paths.stage1_ckpt is None else safe_relpath(paths.stage1_ckpt),
        "stage2_source": paths.stage2_source,
        "stage2_ckpt": safe_relpath(paths.stage2_ckpt),
        "vocoder_type": paths.vocoder_type,
        "vocoder_profile": paths.vocoder_profile,
        "hifigan_root": None if paths.hifigan_root is None else safe_relpath(paths.hifigan_root),
        "inference_mode": detect_inference_mode(paths),
    }


__all__ = [
    "VoiceProfileError", "VoiceProfile", "ResolvedModelPaths", "load_json",
    "load_voice_profile", "validate_voice_profile", "resolve_stage1_checkpoint",
    "resolve_stage2_checkpoint", "resolve_vocoder_settings", "resolve_model_paths",
    "detect_inference_mode", "profile_to_summary_dict",
]
