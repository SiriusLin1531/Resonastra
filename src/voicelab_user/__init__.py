"""
VoiceLab user-facing package.

This package contains the user-edition layers for VoiceLab, including
configuration, profile loading, runtime checks, inference services, metrics
wrappers, and UI components.

Design rule:
    Keep this package lightweight at import time.
    Do not import heavy model dependencies such as torch, transformers,
    GPT-SoVITS modules, vocoders, or metrics backends here.

The user edition should wrap the stable developer pipeline without exposing
developer-only options such as residual refiner, historical sampling modes,
A/B comparison, oracle semantic debugging, or raw tensor dumps.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
