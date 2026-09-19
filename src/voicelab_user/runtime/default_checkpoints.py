from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.adapters.gsv_env import DEFAULT_V2_STAGE1_CKPT_RELPATH


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_NAME = "default_zh"
DEFAULT_PROFILE_DIR_RELATIVE = Path("user_profiles") / DEFAULT_PROFILE_NAME
DEFAULT_STAGE1_CKPT_RELATIVE = Path(DEFAULT_V2_STAGE1_CKPT_RELPATH)
DEFAULT_STAGE2_CKPT_RELATIVE = DEFAULT_PROFILE_DIR_RELATIVE / "stage2" / "best_model.pt"


@dataclass(frozen=True, slots=True)
class DefaultCheckpointSet:
    project_root: Path
    profile_name: str
    profile_dir: Path
    stage1_checkpoint: Path
    stage2_checkpoint: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def resolve_default_checkpoints(
    project_root: str | Path | None = None,
    *,
    validate: bool = False,
) -> DefaultCheckpointSet:
    root = Path(project_root).expanduser().resolve(strict=False) if project_root is not None else PROJECT_ROOT
    resolved = DefaultCheckpointSet(
        project_root=root,
        profile_name=DEFAULT_PROFILE_NAME,
        profile_dir=(root / DEFAULT_PROFILE_DIR_RELATIVE).resolve(strict=False),
        stage1_checkpoint=(root / DEFAULT_STAGE1_CKPT_RELATIVE).resolve(strict=False),
        stage2_checkpoint=(root / DEFAULT_STAGE2_CKPT_RELATIVE).resolve(strict=False),
    )
    if validate:
        missing: list[str] = []
        if not resolved.stage1_checkpoint.is_file():
            missing.append(f"Stage1 default checkpoint not found: {resolved.stage1_checkpoint}")
        if not resolved.stage2_checkpoint.is_file():
            missing.append(f"Stage2 default checkpoint not found: {resolved.stage2_checkpoint}")
        if missing:
            raise FileNotFoundError("\n".join(missing))
    return resolved


def resolve_default_checkpoint(
    stage: str,
    project_root: str | Path | None = None,
    *,
    validate: bool = False,
) -> Path:
    defaults = resolve_default_checkpoints(project_root, validate=validate)
    normalized = str(stage).strip().lower()
    if normalized == "stage1":
        return defaults.stage1_checkpoint
    if normalized == "stage2":
        return defaults.stage2_checkpoint
    raise ValueError(f"Unsupported stage: {stage!r}")


def default_checkpoint_summary(project_root: str | Path | None = None) -> dict[str, Any]:
    defaults = resolve_default_checkpoints(project_root, validate=False)
    return {
        **defaults.to_dict(),
        "stage1_exists": defaults.stage1_checkpoint.is_file(),
        "stage2_exists": defaults.stage2_checkpoint.is_file(),
        "ready": defaults.stage1_checkpoint.is_file() and defaults.stage2_checkpoint.is_file(),
    }


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_PROFILE_NAME",
    "DEFAULT_PROFILE_DIR_RELATIVE",
    "DEFAULT_STAGE1_CKPT_RELATIVE",
    "DEFAULT_STAGE2_CKPT_RELATIVE",
    "DefaultCheckpointSet",
    "resolve_default_checkpoints",
    "resolve_default_checkpoint",
    "default_checkpoint_summary",
]
