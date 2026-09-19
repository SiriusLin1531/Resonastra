from __future__ import annotations

"""
User-edition inference package.

This package provides profile discovery and resolution APIs used by future
Inference UI components.
"""

from .profile_registry import (
    ACTIVE_PROFILE_FILENAME,
    PROFILE_MANIFEST_FILENAME,
    format_profile_registry_markdown,
    get_active_profile,
    load_profile_manifest,
    profile_choices,
    resolve_inference_profile,
    scan_user_profiles,
    set_active_profile,
)

__all__ = [
    "ACTIVE_PROFILE_FILENAME",
    "PROFILE_MANIFEST_FILENAME",
    "scan_user_profiles",
    "profile_choices",
    "load_profile_manifest",
    "set_active_profile",
    "get_active_profile",
    "resolve_inference_profile",
    "format_profile_registry_markdown",
]
