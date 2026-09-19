"""
Environment check utilities for VoiceLab user edition.

This first P4 file provides a lightweight reusable check engine. It does not
load VoiceLab models or run inference. Later files will add the CLI wrapper and
launch scripts.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from .path_resolver import (
    PathLike,
    get_user_project_root,
    normalize_path_text,
    resolve_project_path,
)

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"


@dataclass(frozen=True)
class EnvCheckItem:
    """Single environment check item."""

    name: str
    status: str
    message: str
    path: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserEnvCheckReport:
    """Full user environment check report."""

    schema_version: str
    created_at: str
    overall_status: str
    project_root: str
    python_executable: str
    platform: str
    items: list[EnvCheckItem]


def _item(
    name: str,
    status: str,
    message: str,
    *,
    path: Optional[PathLike] = None,
    detail: Optional[dict[str, Any]] = None,
) -> EnvCheckItem:
    return EnvCheckItem(
        name=name,
        status=status,
        message=message,
        path=None if path is None else str(path),
        detail=detail or {},
    )


def _check_path(
    *,
    name: str,
    path: Path,
    label: str,
    kind: str,
    required: bool = True,
) -> EnvCheckItem:
    exists = path.exists()
    valid = path.is_file() if kind == "file" else path.is_dir()

    if exists and valid:
        return _item(name, STATUS_OK, f"{label} exists.", path=path)

    status = STATUS_FAIL if required else STATUS_WARN
    if exists and not valid:
        return _item(
            name,
            status,
            f"{label} exists but is not a {kind}.",
            path=path,
        )

    return _item(name, status, f"Missing {label}.", path=path)


def _package_version(
    module_name: str,
    distribution_name: Optional[str] = None,
) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution_name or module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python_runtime(
    *,
    project_root: Optional[Path] = None,
    strict_runtime: bool = False,
) -> list[EnvCheckItem]:
    """Check Python executable and optional bundled runtime/env layout."""

    root = project_root or get_user_project_root()
    runtime_env = root / "runtime" / "env"
    expected_python = runtime_env / (
        "python.exe" if platform.system() == "Windows" else "bin/python"
    )
    current_python = Path(sys.executable).resolve(strict=False)

    items = [
        _item(
            "python.executable",
            STATUS_OK,
            "Current Python executable detected.",
            path=current_python,
            detail={"version": sys.version.replace("\n", " ")},
        )
    ]

    if runtime_env.exists() and runtime_env.is_dir():
        items.append(
            _item(
                "runtime.env_dir",
                STATUS_OK,
                "Bundled runtime/env directory exists.",
                path=runtime_env,
            )
        )
    else:
        items.append(
            _item(
                "runtime.env_dir",
                STATUS_FAIL if strict_runtime else STATUS_WARN,
                (
                    "Bundled runtime/env directory is missing. "
                    "This is acceptable in a developer checkout but not in a final user package."
                ),
                path=runtime_env,
            )
        )

    if expected_python.exists():
        items.append(
            _item(
                "runtime.python",
                STATUS_OK,
                "Bundled Python executable exists.",
                path=expected_python,
            )
        )
    else:
        items.append(
            _item(
                "runtime.python",
                STATUS_FAIL if strict_runtime else STATUS_WARN,
                "Bundled Python executable is missing.",
                path=expected_python,
            )
        )

    return items


def check_python_packages() -> list[EnvCheckItem]:
    """Check required Python package import availability without importing them."""

    specs = [
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("soundfile", "soundfile"),
        ("librosa", "librosa"),
        ("gradio", "gradio"),
        ("transformers", "transformers"),
        ("fast_langdetect", "fast-langdetect"),
        ("split_lang", "split-lang"),
    ]

    items: list[EnvCheckItem] = []
    for module_name, dist_name in specs:
        if importlib.util.find_spec(module_name) is None:
            items.append(
                _item(
                    f"package.{module_name}",
                    STATUS_FAIL,
                    f"Missing Python package: {module_name}.",
                    detail={"distribution": dist_name},
                )
            )
        else:
            items.append(
                _item(
                    f"package.{module_name}",
                    STATUS_OK,
                    f"Python package available: {module_name}.",
                    detail={
                        "distribution": dist_name,
                        "version": _package_version(module_name, dist_name),
                    },
                )
            )

    return items


def check_user_config_files(*, project_root: Optional[Path] = None) -> list[EnvCheckItem]:
    """Check user-edition default config/profile files."""

    root = project_root or get_user_project_root()
    return [
        _check_path(
            name="config.user_inference_default",
            path=root / "configs" / "user_inference_default.yaml",
            label="configs/user_inference_default.yaml",
            kind="file",
            required=True,
        ),
        _check_path(
            name="profile.default_zh.config",
            path=root / "user_profiles" / "default_zh" / "profile_config.json",
            label="user_profiles/default_zh/profile_config.json",
            kind="file",
            required=True,
        ),
    ]


def check_model_assets(*, project_root: Optional[Path] = None) -> list[EnvCheckItem]:
    """Check offline assets required by the default Chinese user profile."""

    root = project_root or get_user_project_root()

    checks = [
        (
            "asset.gsv_compat",
            root / "third_party" / "gsv_compat",
            "third_party/gsv_compat",
            "dir",
        ),
        (
            "asset.bert",
            root
            / "GPT_SoVITS"
            / "pretrained_models"
            / "chinese-roberta-wwm-ext-large",
            "Chinese BERT directory",
            "dir",
        ),
        (
            "asset.cnhubert",
            root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base",
            "CNHuBERT directory",
            "dir",
        ),
        (
            "asset.g2pw",
            root / "GPT_SoVITS" / "text" / "G2PWModel",
            "G2PW directory",
            "dir",
        ),
        (
            "asset.default_stage1",
            root
            / "GPT_SoVITS"
            / "pretrained_models"
            / "gsv-v2final-pretrained"
            / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "default GPT-SoVITS v2 Stage1 checkpoint",
            "file",
        ),
        (
            "asset.default_sovits",
            root
            / "GPT_SoVITS"
            / "pretrained_models"
            / "gsv-v2final-pretrained"
            / "s2G2333k.pth",
            "default GPT-SoVITS v2 SoVITS checkpoint",
            "file",
        ),
        (
            "asset.default_stage2",
            root / "user_profiles" / "default_zh" / "stage2" / "best_model.pt",
            "default Resonastra Stage2 checkpoint",
            "file",
        ),
        (
            "asset.hifigan",
            root / "third_party" / "hifi-gan",
            "third_party/hifi-gan",
            "dir",
        ),
    ]

    return [
        _check_path(
            name=name,
            path=path,
            label=label,
            kind=kind,
            required=True,
        )
        for name, path, label, kind in checks
    ]


def compute_overall_status(items: Iterable[EnvCheckItem]) -> str:
    statuses = [item.status for item in items]
    if STATUS_FAIL in statuses:
        return STATUS_FAIL
    if STATUS_WARN in statuses:
        return STATUS_WARN
    return STATUS_OK


def run_user_env_check(
    *,
    project_root: Optional[PathLike] = None,
    strict_runtime: bool = False,
) -> UserEnvCheckReport:
    """Run user-edition environment checks and return a structured report."""

    root = (
        resolve_project_path(project_root)
        if normalize_path_text(project_root)
        else get_user_project_root()
    )
    assert root is not None

    items: list[EnvCheckItem] = []
    items.extend(check_python_runtime(project_root=root, strict_runtime=strict_runtime))
    items.extend(check_python_packages())
    items.extend(check_user_config_files(project_root=root))
    items.extend(check_model_assets(project_root=root))

    return UserEnvCheckReport(
        schema_version="voicelab_user_env_check_v1",
        created_at=datetime.now().isoformat(timespec="seconds"),
        overall_status=compute_overall_status(items),
        project_root=str(root),
        python_executable=str(Path(sys.executable).resolve(strict=False)),
        platform=platform.platform(),
        items=items,
    )


def report_to_dict(report: UserEnvCheckReport) -> dict[str, Any]:
    """Convert report to JSON-serializable dictionary."""

    return asdict(report)


def write_env_check_report(report: UserEnvCheckReport, output_path: PathLike) -> Path:
    """Write environment check report JSON."""

    output = resolve_project_path(output_path)
    if output is None:
        raise ValueError("output_path must not be empty.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def format_env_check_report(report: UserEnvCheckReport) -> str:
    """Format report as readable plain text."""

    title = {
        STATUS_OK: "Resonastra environment check passed.",
        STATUS_WARN: "Resonastra environment check completed with warnings.",
        STATUS_FAIL: "Resonastra environment check failed.",
    }.get(report.overall_status, "Resonastra environment check completed.")

    lines = [
        title,
        f"overall_status: {report.overall_status}",
        f"project_root: {report.project_root}",
        f"python: {report.python_executable}",
        "",
    ]

    for item in report.items:
        prefix = {
            STATUS_OK: "[OK]",
            STATUS_WARN: "[WARN]",
            STATUS_FAIL: "[FAIL]",
        }.get(item.status, "[INFO]")

        lines.append(f"{prefix} {item.name}: {item.message}")
        if item.path:
            lines.append(f"      path: {item.path}")

    return "\n".join(lines)


__all__ = [
    "STATUS_OK",
    "STATUS_WARN",
    "STATUS_FAIL",
    "EnvCheckItem",
    "UserEnvCheckReport",
    "check_python_runtime",
    "check_python_packages",
    "check_user_config_files",
    "check_model_assets",
    "compute_overall_status",
    "run_user_env_check",
    "report_to_dict",
    "write_env_check_report",
    "format_env_check_report",
]