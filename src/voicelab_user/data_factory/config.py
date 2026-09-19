from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


DATA_FACTORY_CONFIG_SCHEMA_VERSION = "voicelab_data_factory_config_v1"
DEFAULT_DATA_FACTORY_CONFIG_PATH = "configs/data_factory_default.yaml"


class DataFactoryConfigError(ValueError):
    """Raised when data factory config is invalid."""


@dataclass(frozen=True)
class DataFactoryPathConfig:
    default_user_data_root: Path
    work_dir_pattern: str = "{speaker_name}_factory"


@dataclass(frozen=True)
class DataFactoryDatasetConfig:
    language: str = "zh"


@dataclass(frozen=True)
class DataFactoryASRConfig:
    backend: str = "auto"
    model_size: str = "large-v3"
    precision: str = "float32"


@dataclass(frozen=True)
class DataFactoryPreprocessConfig:
    enable_uvr: bool = False
    enable_denoise: bool = False


@dataclass(frozen=True)
class DataFactorySliceConfig:
    threshold: int = -34
    min_length: int = 4000
    min_interval: int = 300
    hop_size: int = 10
    max_sil_kept: int = 500
    normalize_max: float = 0.9
    alpha_mix: float = 0.25


@dataclass(frozen=True)
class DataFactoryExportConfig:
    export_list: bool = True
    export_stage2_manifest: bool = True

    prompt_mode: str = "speaker_pool"
    min_prompt_sec: float = 3.0
    max_prompt_sec: float = 10.0
    prefer_prompt_sec: float = 6.0
    allow_self_prompt: bool = True


@dataclass(frozen=True)
class DataFactoryHealthConfig:
    min_samples_pass: int = 20
    min_samples_warn: int = 10
    min_total_duration_sec_pass: float = 60.0
    min_total_duration_sec_warn: float = 30.0
    min_clip_sec_fail: float = 0.8
    min_clip_sec_warn: float = 2.0
    max_clip_sec_warn: float = 12.0
    max_clip_sec_fail: float = 30.0
    min_prompt_sec: float = 3.0
    max_prompt_sec: float = 10.0


@dataclass(frozen=True)
class DataFactoryRuntimeConfig:
    overwrite_work_dir: bool = False
    dry_run: bool = False
    timeout_seconds: Optional[float] = 7200


@dataclass(frozen=True)
class DataFactoryConfig:
    schema_version: str
    project_root: Path

    paths: DataFactoryPathConfig
    dataset: DataFactoryDatasetConfig
    asr: DataFactoryASRConfig
    preprocess: DataFactoryPreprocessConfig
    slice: DataFactorySliceConfig
    export: DataFactoryExportConfig
    health: DataFactoryHealthConfig
    runtime: DataFactoryRuntimeConfig


def _project_root() -> Path:
    """Infer repository root from this file location.

    当前文件路径：
        src/voicelab_user/data_factory/config.py

    parents[3] 应为项目根目录。
    """

    return Path(__file__).resolve().parents[3]


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _bool_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    raise DataFactoryConfigError(f"{name} must be a boolean-like value, got: {value!r}")


def _float_or_none(value: Any, *, name: str) -> Optional[float]:
    if value is None:
        return None

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DataFactoryConfigError(f"{name} must be a number or null.") from exc

    if out <= 0:
        return None

    return out


def _resolve_project_path(value: Any, *, project_root: Path) -> Path:
    text = "" if value is None else str(value).strip()
    if not text:
        raise DataFactoryConfigError("Path value must not be empty.")

    p = Path(text).expanduser()
    if p.is_absolute():
        return p.resolve(strict=False)

    return (project_root / p).resolve(strict=False)


def load_data_factory_config(
    config_path: str | Path = DEFAULT_DATA_FACTORY_CONFIG_PATH,
) -> dict[str, Any]:
    """Load raw YAML config as dict."""

    project_root = _project_root()
    path = _resolve_project_path(config_path, project_root=project_root)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Data factory config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise DataFactoryConfigError(f"Data factory config must be a dict: {path}")

    return data


def build_data_factory_config(
    defaults: Optional[dict[str, Any]] = None,
    *,
    config_path: str | Path = DEFAULT_DATA_FACTORY_CONFIG_PATH,
) -> DataFactoryConfig:
    """Build typed DataFactoryConfig from YAML defaults.

    中文说明：
        从 configs/data_factory_default.yaml 构建强类型配置。

        P10-1 只负责配置读取和基础校验。
        不调用数据工厂脚本。
    """

    project_root = _project_root()
    raw = defaults if defaults is not None else load_data_factory_config(config_path)

    paths = _section(raw, "paths")
    dataset = _section(raw, "dataset")
    asr = _section(raw, "asr")
    preprocess = _section(raw, "preprocess")
    slice_cfg = _section(raw, "slice")
    export = _section(raw, "export")
    health = _section(raw, "health")
    runtime = _section(raw, "runtime")

    cfg = DataFactoryConfig(
        schema_version=str(raw.get("schema_version", DATA_FACTORY_CONFIG_SCHEMA_VERSION)),
        project_root=project_root,
        paths=DataFactoryPathConfig(
            default_user_data_root=_resolve_project_path(
                paths.get("default_user_data_root", "user_data"),
                project_root=project_root,
            ),
            work_dir_pattern=str(paths.get("work_dir_pattern", "{speaker_name}_factory")),
        ),
        dataset=DataFactoryDatasetConfig(
            language=str(dataset.get("language", "zh")),
        ),
        asr=DataFactoryASRConfig(
            backend=str(asr.get("backend", "auto")),
            model_size=str(asr.get("model_size", "large-v3")),
            precision=str(asr.get("precision", "float32")),
        ),
        preprocess=DataFactoryPreprocessConfig(
            enable_uvr=_bool_value(preprocess.get("enable_uvr", False), name="preprocess.enable_uvr"),
            enable_denoise=_bool_value(
                preprocess.get("enable_denoise", False),
                name="preprocess.enable_denoise",
            ),
        ),
        slice=DataFactorySliceConfig(
            threshold=int(slice_cfg.get("threshold", -34)),
            min_length=int(slice_cfg.get("min_length", 4000)),
            min_interval=int(slice_cfg.get("min_interval", 300)),
            hop_size=int(slice_cfg.get("hop_size", 10)),
            max_sil_kept=int(slice_cfg.get("max_sil_kept", 500)),
            normalize_max=float(slice_cfg.get("normalize_max", 0.9)),
            alpha_mix=float(slice_cfg.get("alpha_mix", 0.25)),
        ),
        export=DataFactoryExportConfig(
            export_list=_bool_value(export.get("export_list", True), name="export.export_list"),
            export_stage2_manifest=_bool_value(
                export.get("export_stage2_manifest", True),
                name="export.export_stage2_manifest",
            ),
            prompt_mode=str(export.get("prompt_mode", "speaker_pool")),
            min_prompt_sec=float(export.get("min_prompt_sec", 3.0)),
            max_prompt_sec=float(export.get("max_prompt_sec", 10.0)),
            prefer_prompt_sec=float(export.get("prefer_prompt_sec", 6.0)),
            allow_self_prompt=_bool_value(
                export.get("allow_self_prompt", True),
                name="export.allow_self_prompt",
            ),
        ),
        health=DataFactoryHealthConfig(
            min_samples_pass=int(health.get("min_samples_pass", 20)),
            min_samples_warn=int(health.get("min_samples_warn", 10)),
            min_total_duration_sec_pass=float(health.get("min_total_duration_sec_pass", 60.0)),
            min_total_duration_sec_warn=float(health.get("min_total_duration_sec_warn", 30.0)),
            min_clip_sec_fail=float(health.get("min_clip_sec_fail", 0.8)),
            min_clip_sec_warn=float(health.get("min_clip_sec_warn", 2.0)),
            max_clip_sec_warn=float(health.get("max_clip_sec_warn", 12.0)),
            max_clip_sec_fail=float(health.get("max_clip_sec_fail", 30.0)),
            min_prompt_sec=float(health.get("min_prompt_sec", 3.0)),
            max_prompt_sec=float(health.get("max_prompt_sec", 10.0)),
        ),
        runtime=DataFactoryRuntimeConfig(
            overwrite_work_dir=_bool_value(
                runtime.get("overwrite_work_dir", False),
                name="runtime.overwrite_work_dir",
            ),
            dry_run=_bool_value(runtime.get("dry_run", False), name="runtime.dry_run"),
            timeout_seconds=_float_or_none(
                runtime.get("timeout_seconds", 7200),
                name="runtime.timeout_seconds",
            ),
        ),
    )

    validate_data_factory_config(cfg)
    return cfg


def validate_data_factory_config(cfg: DataFactoryConfig) -> None:
    """Validate data factory config."""

    if cfg.schema_version != DATA_FACTORY_CONFIG_SCHEMA_VERSION:
        raise DataFactoryConfigError(
            f"Unsupported schema_version: {cfg.schema_version}. "
            f"Expected: {DATA_FACTORY_CONFIG_SCHEMA_VERSION}"
        )

    if not cfg.dataset.language:
        raise DataFactoryConfigError("dataset.language must not be empty.")

    if cfg.asr.backend not in {"auto", "faster_whisper", "whisper", "none"}:
        raise DataFactoryConfigError(
            "asr.backend must be one of: auto, faster_whisper, whisper, none."
        )

    if not cfg.asr.model_size:
        raise DataFactoryConfigError("asr.model_size must not be empty.")

    if not cfg.asr.precision:
        raise DataFactoryConfigError("asr.precision must not be empty.")

    if cfg.preprocess.enable_uvr:
        raise DataFactoryConfigError(
            "preprocess.enable_uvr=true is not supported yet. "
            "The underlying build_dataset currently raises NotImplementedError."
        )

    if cfg.preprocess.enable_denoise:
        raise DataFactoryConfigError(
            "preprocess.enable_denoise=true is not supported yet. "
            "The underlying build_dataset currently raises NotImplementedError."
        )

    if cfg.slice.min_length <= 0:
        raise DataFactoryConfigError("slice.min_length must be > 0.")

    if cfg.slice.min_interval < 0:
        raise DataFactoryConfigError("slice.min_interval must be >= 0.")

    if cfg.slice.hop_size <= 0:
        raise DataFactoryConfigError("slice.hop_size must be > 0.")

    if cfg.slice.max_sil_kept < 0:
        raise DataFactoryConfigError("slice.max_sil_kept must be >= 0.")

    if cfg.slice.normalize_max <= 0:
        raise DataFactoryConfigError("slice.normalize_max must be > 0.")

    if cfg.export.prompt_mode not in {"self", "speaker_pool", "fixed_reference"}:
        raise DataFactoryConfigError(
            "export.prompt_mode must be self, speaker_pool, or fixed_reference."
        )

    if cfg.export.min_prompt_sec <= 0:
        raise DataFactoryConfigError("export.min_prompt_sec must be > 0.")

    if cfg.export.max_prompt_sec < cfg.export.min_prompt_sec:
        raise DataFactoryConfigError(
            "export.max_prompt_sec must be >= export.min_prompt_sec."
        )