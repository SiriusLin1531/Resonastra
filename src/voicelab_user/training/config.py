from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_CONFIG_PATH = Path("configs/training_default.yaml")
TRAINING_CONFIG_SCHEMA_VERSION = "voicelab_training_default_config_v1"

T = TypeVar("T")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except Exception:
        pass
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _fallback_yaml_load(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if ":" not in raw_line:
            raise ValueError(f"Invalid YAML-like line {line_no}: {raw_line!r}")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = raw_line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation at line {line_no}")
        current = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def resolve_training_config_path(config_path: str | Path | None = None) -> Path:
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / DEFAULT_TRAINING_CONFIG_PATH
    return path.expanduser().resolve(strict=False)


def load_training_default_config(
    config_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> dict[str, Any]:
    path = resolve_training_config_path(config_path)
    if not path.is_file():
        if require_exists:
            raise FileNotFoundError(f"Training config not found: {path}")
        return {}
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except ModuleNotFoundError:
        data = _fallback_yaml_load(path)
    except Exception as exc:
        raise ValueError(f"Failed to read training config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Training config root must be a mapping: {path}")
    schema_version = data.get("schema_version")
    if schema_version not in {None, TRAINING_CONFIG_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported training config schema_version={schema_version!r}; "
            f"expected {TRAINING_CONFIG_SCHEMA_VERSION!r}."
        )
    return data


def _coerce_value(value: Any, fallback: Any, *, field_name: str) -> Any:
    expected = type(fallback)
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "on"}:
                return True
            if text in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"{field_name} must be boolean, got {value!r}")
    if expected is int and not isinstance(fallback, bool):
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be integer, got boolean")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be integer, got {value!r}") from exc
    if expected is float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be number, got boolean")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be number, got {value!r}") from exc
    if expected is str:
        if value is None:
            raise ValueError(f"{field_name} must be string, got null")
        return str(value)
    return value


def merge_training_defaults(
    stage_name: str,
    fallback: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(fallback)
    config = config if config is not None else load_training_default_config()
    section = config.get(stage_name, {})
    if section is None:
        return merged
    if not isinstance(section, dict):
        raise ValueError(f"Training config section {stage_name!r} must be a mapping.")
    unknown = sorted(set(section) - set(fallback))
    if unknown:
        raise ValueError(f"Unknown {stage_name} training config fields: {', '.join(unknown)}")
    for key, value in section.items():
        merged[key] = _coerce_value(value, fallback[key], field_name=f"{stage_name}.{key}")
    return merged


def build_dataclass_defaults(
    defaults_type: type[T],
    stage_name: str,
    *,
    config: dict[str, Any] | None = None,
) -> T:
    if not is_dataclass(defaults_type):
        raise TypeError(f"defaults_type must be a dataclass type: {defaults_type!r}")
    fallback_obj = defaults_type()  # type: ignore[call-arg]
    fallback = {field.name: getattr(fallback_obj, field.name) for field in fields(fallback_obj)}
    merged = merge_training_defaults(stage_name, fallback, config)
    result = defaults_type(**merged)  # type: ignore[call-arg]
    validate_training_defaults(stage_name, merged)
    return result


def validate_training_defaults(stage_name: str, values: dict[str, Any]) -> None:
    positive_ints = ["epochs", "batch_size"]
    nonnegative_ints = ["num_workers", "last_n_layers", "log_every_steps"]
    positive_numbers = ["lr"]
    nonnegative_numbers = [
        "weight_decay", "grad_clip", "bootstrapper_lora_dropout",
        "bootstrapper_attention_lora_dropout", "v66_energy_loss_weight",
        "v66_f0_loss_weight", "v66_speaker_loss_weight",
    ]
    for key in positive_ints:
        if key in values and int(values[key]) <= 0:
            raise ValueError(f"{stage_name}.{key} must be > 0")
    for key in nonnegative_ints:
        if key in values and int(values[key]) < 0:
            raise ValueError(f"{stage_name}.{key} must be >= 0")
    for key in positive_numbers:
        if key in values and float(values[key]) <= 0:
            raise ValueError(f"{stage_name}.{key} must be > 0")
    for key in nonnegative_numbers:
        if key in values and float(values[key]) < 0:
            raise ValueError(f"{stage_name}.{key} must be >= 0")
    if not str(values.get("device", "")).strip():
        raise ValueError(f"{stage_name}.device must not be empty")
    if not str(values.get("trainable_scope", "")).strip():
        raise ValueError(f"{stage_name}.trainable_scope must not be empty")


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_TRAINING_CONFIG_PATH",
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "resolve_training_config_path",
    "load_training_default_config",
    "merge_training_defaults",
    "build_dataclass_defaults",
    "validate_training_defaults",
]
