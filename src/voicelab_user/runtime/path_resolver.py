"""
Path utilities for the VoiceLab user edition.

This module is intentionally lightweight. It should not import torch,
GPT-SoVITS, vocoder modules, metrics backends, or any heavy runtime code.

The goal of this module is to make user-edition paths predictable and safe:
    - support absolute paths;
    - support project-root-relative paths;
    - support profile-relative paths;
    - avoid developer-machine absolute paths in user-facing configs;
    - produce clear errors for missing offline resources.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


class VoiceLabUserPathError(RuntimeError):
    """Raised when a user-edition path cannot be resolved or validated."""


def normalize_path_text(path: Optional[PathLike]) -> Optional[str]:
    """Normalize a user-provided path string.

    Rules:
        - None stays None.
        - Leading/trailing whitespace is stripped.
        - Empty strings become None.
        - One pair of surrounding quotes is removed.
        - Windows backslashes are preserved.

    Examples:
        "" -> None
        "   " -> None
        '"D:\\a\\b.pt"' -> 'D:\\a\\b.pt'
    """

    if path is None:
        return None

    text = str(path).strip()
    if not text:
        return None

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    return text or None


def is_abs_path(path: Optional[PathLike]) -> bool:
    """Return whether *path* is absolute.

    This wrapper exists mainly to keep Windows drive paths handled in one place.
    """

    text = normalize_path_text(path)
    if text is None:
        return False
    return Path(text).is_absolute()


def get_user_project_root() -> Path:
    """Return the VoiceLab user-edition project root.

    Resolution order:
        1. VOICELAB_USER_ROOT environment variable, if set.
        2. The repository root inferred from this file location:
           src/voicelab_user/runtime/path_resolver.py -> parents[3].

    Returns:
        Absolute project root path.
    """

    env_root = normalize_path_text(os.environ.get("VOICELAB_USER_ROOT"))
    if env_root is not None:
        return Path(env_root).expanduser().resolve(strict=False)

    try:
        return Path(__file__).resolve(strict=False).parents[3]
    except IndexError as exc:
        raise VoiceLabUserPathError("无法解析 VoiceLab 用户版项目根目录。") from exc


def resolve_project_path(
    path: Optional[PathLike],
    *,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve a path against the user-edition project root.

    Args:
        path: Absolute path, project-root-relative path, or None.
        project_root: Optional explicit project root. If omitted, it is inferred.

    Returns:
        Absolute Path, or None if path is empty.
    """

    text = normalize_path_text(path)
    if text is None:
        return None

    raw = Path(text).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)

    root = project_root or get_user_project_root()
    return (root / raw).resolve(strict=False)


def resolve_profile_relative_path(
    path: Optional[PathLike],
    *,
    profile_dir: Path,
) -> Optional[Path]:
    """Resolve a path that may be relative to a voice profile directory.

    This is used for entries inside profile_config.json, for example:
        "stage2/best_model.pt"
    """

    text = normalize_path_text(path)
    if text is None:
        return None

    raw = Path(text).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)

    return (profile_dir / raw).resolve(strict=False)


def ensure_file_exists(
    path: Optional[Path],
    *,
    label: str,
    required: bool = True,
) -> Optional[Path]:
    """Validate that a required file exists.

    Args:
        path: Path to check.
        label: User-facing resource label, such as "默认 Stage2 声学生成模型".
        required: If False, None is allowed.

    Returns:
        The original path if valid, otherwise None for optional missing paths.
    """

    if path is None:
        if required:
            raise FileNotFoundError(
                f"缺少{label}：\n"
                f"  <未提供路径>\n\n"
                f"请确认配置文件填写正确，或重新解压完整 VoiceLab_User 离线包。"
            )
        return None

    if not path.exists():
        raise FileNotFoundError(
            f"缺少{label}：\n"
            f"  {path}\n\n"
            f"请确认你已经解压完整 VoiceLab_User 离线包，"
            f"或将对应文件放到上述路径。"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"{label}路径不是文件：\n"
            f"  {path}\n\n"
            f"请确认该路径指向具体模型文件。"
        )

    return path


def ensure_dir_exists(
    path: Optional[Path],
    *,
    label: str,
    required: bool = True,
) -> Optional[Path]:
    """Validate that a required directory exists."""

    if path is None:
        if required:
            raise FileNotFoundError(
                f"缺少{label}目录：\n"
                f"  <未提供路径>\n\n"
                f"请确认配置文件填写正确，或重新解压完整 VoiceLab_User 离线包。"
            )
        return None

    if not path.exists():
        raise FileNotFoundError(
            f"缺少{label}目录：\n"
            f"  {path}\n\n"
            f"请确认你已经解压完整 VoiceLab_User 离线包。"
        )

    if not path.is_dir():
        raise FileNotFoundError(
            f"{label}路径不是目录：\n"
            f"  {path}\n\n"
            f"请确认该路径指向目录，而不是文件。"
        )

    return path


def _sanitize_run_component(text: Optional[str], *, fallback: str = "run") -> str:
    """Return a filesystem-friendly run-id component."""

    if text is None:
        return fallback

    text = text.strip()
    if not text:
        return fallback

    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text)
    text = text.strip("_")
    return text or fallback


def make_run_id(profile_name: Optional[str] = None) -> str:
    """Create a timestamped inference run id.

    Format:
        YYYYMMDD_HHMMSS_<profile>
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_run_component(profile_name, fallback="profile")
    return f"{timestamp}_{suffix}"


def resolve_output_dir(
    output_dir: Optional[PathLike],
    *,
    output_root: PathLike,
    profile_name: str,
    project_root: Optional[Path] = None,
    create: bool = True,
) -> Path:
    """Resolve the output directory for a user inference run.

    If output_dir is provided, it is resolved against project root.
    Otherwise a timestamped directory is created under output_root.
    """

    root = project_root or get_user_project_root()

    resolved_output_dir = resolve_project_path(output_dir, project_root=root)
    if resolved_output_dir is None:
        resolved_root = resolve_project_path(output_root, project_root=root)
        if resolved_root is None:
            raise VoiceLabUserPathError("无法解析用户版推理输出根目录。")
        resolved_output_dir = resolved_root / make_run_id(profile_name)

    if create:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    return resolved_output_dir.resolve(strict=False)


def safe_relpath(path: Path, *, start: Optional[Path] = None) -> str:
    """Return a readable relative path when possible.

    If relative conversion fails, return the absolute path string.
    """

    base = start or get_user_project_root()
    try:
        return str(path.resolve(strict=False).relative_to(base.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


__all__ = [
    "PathLike",
    "VoiceLabUserPathError",
    "normalize_path_text",
    "is_abs_path",
    "get_user_project_root",
    "resolve_project_path",
    "resolve_profile_relative_path",
    "ensure_file_exists",
    "ensure_dir_exists",
    "make_run_id",
    "resolve_output_dir",
    "safe_relpath",
]
