from __future__ import annotations

"""Behavior helpers for the VoiceLab UX-FIX-1 regression pass.

This module intentionally contains no Gradio code and no heavy model imports.
It centralizes the user-facing semantics agreed for the post-freeze fixes:

1. A selected inference Profile keeps its identity. Profile selection is not
   converted into checkpoint overrides; only Advanced Options are overrides.
2. DataFactory/Training editable project inputs win over stale read-only output
   state. A previously loaded work directory is used only when both editable
   inputs are blank.
3. Quality metrics use three direct checkboxes. The hidden backend master switch
   is derived from those checkboxes.
4. Empty speaker names are invalid when VoiceLab needs to derive a work directory;
   there is no {speaker_name} fallback.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class InferenceSelection:
    profile_dir: Optional[str]
    stage1_override: Optional[str]
    stage2_override: Optional[str]
    resolved_profile: dict[str, Any]


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def derive_enable_metrics(
    compute_dnsmos: bool,
    compute_speaker_sim: bool,
    compute_wer: bool,
) -> bool:
    """Return the backend master metrics flag from the three visible controls."""

    return bool(compute_dnsmos or compute_speaker_sim or compute_wer)


def resolve_inference_selection(
    profile_choice_value: Any,
    *,
    stage1_override: Any = None,
    stage2_override: Any = None,
    is_default_choice: Callable[[Any], bool],
    resolve_profile: Callable[..., dict[str, Any]],
) -> InferenceSelection:
    """Keep selected Profile identity while preserving explicit Advanced overrides.

    A non-default Profile contributes ``profile_dir`` only.  Stage1/Stage2
    checkpoint fields remain reserved for explicit Advanced Options so the
    service/config layer can apply its normal precedence:

        manual override > selected profile > user-edition default.
    """

    manual_stage1 = _optional_text(stage1_override)
    manual_stage2 = _optional_text(stage2_override)

    if is_default_choice(profile_choice_value):
        return InferenceSelection(
            profile_dir=None,
            stage1_override=manual_stage1,
            stage2_override=manual_stage2,
            resolved_profile={},
        )

    # If Stage2 is explicitly overridden, the selected Profile itself does not
    # need to provide a usable Stage2 checkpoint for this request.
    resolved = resolve_profile(
        str(profile_choice_value),
        require_stage2=manual_stage2 is None,
    )
    profile_root = _optional_text(resolved.get("profile_root"))
    if profile_root is None:
        raise ValueError("Resolved inference profile does not provide profile_root.")

    return InferenceSelection(
        profile_dir=profile_root,
        stage1_override=manual_stage1,
        stage2_override=manual_stage2,
        resolved_profile=dict(resolved),
    )


def resolve_user_work_dir(
    *,
    work_dir_output: Any,
    work_dir: Any,
    speaker_name: Any,
    resolve_path: Callable[[str], Path],
    suggest_work_dir: Callable[[str], Path],
) -> Path:
    """Resolve the project currently described by editable user inputs.

    Precedence is intentionally different from the old implementation:

        editable work_dir > editable speaker_name > previous work_dir_output

    ``work_dir_output`` is read-only UI state from the last successful load. It
    must never pin the page to an old character after the user edits the left
    side of the UI.
    """

    configured_work_dir = _optional_text(work_dir)
    if configured_work_dir is not None:
        return resolve_path(configured_work_dir)

    speaker = _optional_text(speaker_name)
    if speaker is not None:
        return suggest_work_dir(speaker)

    previous = _optional_text(work_dir_output)
    if previous is not None:
        return resolve_path(previous)

    raise ValueError("请先填写说话人 / 角色名，或填写工作目录。")


def require_speaker_name(value: Any) -> str:
    """Normalize a speaker name without falling back to a development placeholder."""

    speaker = _optional_text(value)
    if speaker is None:
        raise ValueError("说话人 / 角色名不能为空。")
    return speaker


__all__ = [
    "InferenceSelection",
    "derive_enable_metrics",
    "resolve_inference_selection",
    "resolve_user_work_dir",
    "require_speaker_name",
]
