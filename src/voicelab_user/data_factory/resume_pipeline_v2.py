from __future__ import annotations

from pathlib import Path
from typing import Any

from .stage2_report_contract import analyze_stage2_split_filter_state


def decide_split_resume(
    artifacts: Any,
    *,
    split_overwrite: bool = False,
    requested_filter_mode: str | None = None,
) -> dict[str, Any]:
    """Return split/filter resume decision using quarantine-aware counts."""

    state = analyze_stage2_split_filter_state(
        artifacts,
        requested_filter_mode=requested_filter_mode,
    )

    if split_overwrite:
        return {
            "action": "run",
            "reason": "user_requested_split_overwrite",
            "state": state.to_dict(),
        }

    if state.completed:
        return {
            "action": "skip",
            "reason": "validated_split_and_filter_outputs_exist",
            "state": state.to_dict(),
        }

    if state.split_done and not state.filter_done:
        return {
            "action": "run_filter_only",
            "reason": "split_exists_but_filter_invalid",
            "state": state.to_dict(),
        }

    return {
        "action": "run_split",
        "reason": "split_outputs_missing_or_invalid",
        "state": state.to_dict(),
    }


__all__ = ["decide_split_resume"]
