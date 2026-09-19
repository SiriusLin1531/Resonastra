from __future__ import annotations

"""Resonastra v1.0.0 canonical offline ASR router.

The public/release DataFactory boundary validated for v1.0.0 is intentionally
restricted to Chinese local FunASR:

    language=zh + backend=auto/funasr -> local FunASR

Faster Whisper and multilingual routing remain outside the v1.0.0 accepted
DataFactory release boundary. This source is canonical for both the public
repository and the packaged v1.0.0 release.
"""

import subprocess
import sys
from pathlib import Path

from src.data_factory.utils.path_utils import clean_path, ensure_dir
from src.data_factory.utils.validation import (
    assert_asr_output_exists,
    assert_directory_exists,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _normalize_backend(backend: str, language: str) -> str:
    b = str(backend).strip().lower()
    lang = str(language).strip().lower()

    if lang != "zh":
        raise ValueError(
            "Resonastra v1.0.0 DataFactory release path supports Chinese (zh) ASR only."
        )

    if b in {"auto", "", "funasr", "damo", "达摩 asr (中文)", "达摩 asr", "达摩"}:
        return "funasr"

    raise ValueError(
        "Resonastra v1.0.0 release package supports backend=auto/funasr only; "
        f"received: {backend!r}."
    )


def _get_asr_script_path(backend: str) -> Path:
    if backend != "funasr":
        raise ValueError(f"Unsupported release ASR backend: {backend}")

    script = PROJECT_ROOT / "tools" / "asr" / "funasr_asr.py"
    if not script.is_file():
        raise FileNotFoundError(f"Required offline FunASR script not found: {script}")
    return script.resolve()


def _expected_asr_list_path(input_dir: str | Path, output_dir: str | Path) -> Path:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    return output_dir / f"{input_dir.name}.list"


def run_asr(
    input_dir: str,
    output_dir: str,
    backend: str = "auto",
    language: str = "zh",
    model_size: str = "large-v3",
    precision: str = "float32",
) -> Path:
    del model_size, precision

    input_root = assert_directory_exists(clean_path(input_dir), name="ASR input_dir")
    output_root = ensure_dir(clean_path(output_dir))

    normalized_backend = _normalize_backend(backend=backend, language=language)
    script_path = _get_asr_script_path(normalized_backend)

    cmd = [
        sys.executable,
        str(script_path),
        "-i",
        str(input_root),
        "-o",
        str(output_root),
        "-s",
        "large",
        "-l",
        "zh",
        "-p",
        "float32",
    ]

    print("====================================================")
    print("Running ASR (Resonastra v1.0.0 offline release path)")
    print("backend    : funasr")
    print("language   : zh")
    print(f"input_dir  : {input_root}")
    print(f"output_dir : {output_root}")
    print(f"script     : {script_path}")
    print("====================================================")

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    asr_list_path = _expected_asr_list_path(input_root, output_root)
    return assert_asr_output_exists(asr_list_path)
