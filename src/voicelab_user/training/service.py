from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from src.voicelab_user.runtime.default_checkpoints import DEFAULT_STAGE2_CKPT_RELATIVE
from src.voicelab_user.runtime.terminal_progress import launch_log_tail_terminal

from .config import build_dataclass_defaults, load_training_default_config, resolve_training_config_path
from .data_contract import (
    build_training_input_contract,
    validate_stage1_ready,
    validate_stage2_ready,
    write_training_interface_json,
)
from .progress import snapshot_from_run_state


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_RUN_DIRNAME = "12_training"
TRAINING_RUN_STATE_FILENAME = "training_run_state.json"
TRAINING_COMMAND_FILENAME = "training_command.json"
TRAINING_STDOUT_FILENAME = "training_stdout.log"
TRAINING_STDERR_FILENAME = "training_stderr.log"
LATEST_RUN_FILENAME = "latest_run.json"
RUNS_DIRNAME = "runs"
DEFAULT_STAGE2_BASE_CKPT_RELATIVE = DEFAULT_STAGE2_CKPT_RELATIVE
TRAINING_UI_OVERRIDE_FIELDS = frozenset({"epochs", "batch_size"})


@dataclass(slots=True)
class Stage1TrainDefaults:
    device: str = "cuda"
    use_amp: bool = True
    epochs: int = 10
    batch_size: int = 1
    lr: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    trainable_scope: str = "head_and_last_n"
    last_n_layers: int = 4
    log_every_steps: int = 20


@dataclass(slots=True)
class Stage2TrainDefaults:
    device: str = "cuda"
    epochs: int = 20
    batch_size: int = 1
    lr: float = 5e-6
    num_workers: int = 0
    use_amp: bool = True
    trainable_scope: str = "reference_style_plus_lora_plus_attention_lora_plus_aux"
    include_continuous_semantic: bool = True
    require_continuous_semantic: bool = True
    include_v66_style_fields: bool = True
    require_v66_style_fields: bool = True
    use_reference_acoustic_style: bool = True
    reference_style_fusion_mode: str = "gated_add"
    reference_style_gate_init: float = -3.0
    use_v66_energy_loss: bool = True
    v66_energy_loss_weight: float = 0.15
    use_v66_f0_loss: bool = True
    v66_f0_loss_weight: float = 0.05
    use_v66_speaker_loss: bool = True
    v66_speaker_loss_weight: float = 0.03
    v66_speaker_embedding_dim: int = 192
    v66_require_aux_targets: bool = True
    use_bootstrapper_lora: bool = True
    bootstrapper_lora_rank: int = 4
    bootstrapper_lora_alpha: float = 8.0
    bootstrapper_lora_dropout: float = 0.05
    bootstrapper_lora_target: str = "core"
    bootstrapper_lora_init_scale: float = 0.01
    use_bootstrapper_attention_lora: bool = True
    bootstrapper_attention_lora_rank: int = 4
    bootstrapper_attention_lora_alpha: float = 8.0
    bootstrapper_attention_lora_dropout: float = 0.05
    bootstrapper_attention_lora_target: str = "cross_only"
    bootstrapper_attention_lora_gate_init: float = -3.0
    bootstrapper_attention_lora_enable_q: bool = True
    bootstrapper_attention_lora_enable_k: bool = True
    bootstrapper_attention_lora_enable_v: bool = True
    bootstrapper_attention_lora_enable_o: bool = True
    load_strict: bool = False
    force_v65_coarse_only_defaults: bool = True


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _pid_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid_int}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = completed.stdout.strip().lower()
        return completed.returncode == 0 and bool(out) and "no tasks" not in out and str(pid_int) in out
    try:
        os.kill(pid_int, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _as_path(value: Any, *, label: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing path for {label}.")
    return Path(str(value)).expanduser().resolve(strict=False)


def _require_file(path: str | Path, *, label: str, nonempty: bool = False) -> Path:
    p = _as_path(path, label=label)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    if nonempty and p.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {p}")
    return p


def _require_dir(path: str | Path, *, label: str, pattern: str | None = None) -> Path:
    p = _as_path(path, label=label)
    if not p.is_dir():
        raise NotADirectoryError(f"{label} not found: {p}")
    if pattern and not any(p.glob(pattern)):
        raise ValueError(f"{label} contains no {pattern} files: {p}")
    return p


def _stage_path(contract: dict[str, Any], stage: str, group: str, key: str) -> str:
    stage_obj = contract.get(stage) if isinstance(contract.get(stage), dict) else {}
    group_obj = stage_obj.get(group) if isinstance(stage_obj.get(group), dict) else {}
    value = group_obj.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing contract path: {stage}.{group}.{key}")
    return str(value)


def _stage1_path(contract: dict[str, Any], key: str) -> str:
    stage1 = contract.get("stage1") if isinstance(contract.get("stage1"), dict) else {}
    paths = stage1.get("paths") if isinstance(stage1.get("paths"), dict) else {}
    value = paths.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing contract path: stage1.paths.{key}")
    return str(value)


def _stage_root(work_dir: str | Path, stage: str) -> Path:
    return Path(work_dir).expanduser().resolve(strict=False) / TRAINING_RUN_DIRNAME / stage


def _make_run_dirs(work_dir: str | Path, stage: str, run_id: str | None = None) -> dict[str, Any]:
    work_dir = Path(work_dir).expanduser().resolve(strict=False)
    stage_root = _stage_root(work_dir, stage)
    resolved_run_id = run_id or _new_run_id()
    run_root = stage_root / RUNS_DIRNAME / resolved_run_id
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "stage_root": stage_root,
        "runs_root": stage_root / RUNS_DIRNAME,
        "run_id": resolved_run_id,
        "run_root": run_root,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "run_state_path": run_root / TRAINING_RUN_STATE_FILENAME,
        "stage_state_path": stage_root / TRAINING_RUN_STATE_FILENAME,
        "latest_run_path": stage_root / LATEST_RUN_FILENAME,
        "command_json_path": run_root / TRAINING_COMMAND_FILENAME,
        "stdout_log_path": logs_dir / TRAINING_STDOUT_FILENAME,
        "stderr_log_path": logs_dir / TRAINING_STDERR_FILENAME,
    }


def _existing_active_run(work_dir: str | Path, stage: str) -> dict[str, Any]:
    state = _read_json(_stage_root(work_dir, stage) / TRAINING_RUN_STATE_FILENAME)
    if state.get("status") in {"pending", "running", "stopping"} and _pid_running(state.get("pid")):
        return state
    return {}


def _training_terminal_title(
    *,
    stage: str,
    run_id: str,
    effective_parameters: Mapping[str, Any],
) -> str:
    stage_key = str(stage).strip().lower()
    stage_label = "Stage1" if stage_key == "stage1" else "Stage2" if stage_key == "stage2" else str(stage)
    epochs = effective_parameters.get("epochs")
    batch_size = effective_parameters.get("batch_size")
    return (
        f"Resonastra {stage_label} Training | "
        f"run={run_id} | epochs={epochs} | batch={batch_size}"
    )


def _coerce_positive_override_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, got boolean")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a whole number, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def _normalize_training_ui_overrides(
    stage: str,
    ui_overrides: Mapping[str, Any] | None,
) -> dict[str, int]:
    if ui_overrides is None:
        return {}
    if not isinstance(ui_overrides, Mapping):
        raise TypeError(
            f"{stage} ui_overrides must be a mapping, got {type(ui_overrides).__name__}"
        )

    unknown = sorted(set(ui_overrides) - TRAINING_UI_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(
            f"Unsupported {stage} UI override fields: {', '.join(unknown)}. "
            f"Allowed fields: {', '.join(sorted(TRAINING_UI_OVERRIDE_FIELDS))}."
        )

    normalized: dict[str, int] = {}
    for key in sorted(TRAINING_UI_OVERRIDE_FIELDS):
        if key not in ui_overrides:
            continue
        value = ui_overrides.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        normalized[key] = _coerce_positive_override_int(
            value,
            field_name=f"{stage}.{key}",
        )
    return normalized


class TrainingService:
    def __init__(
        self,
        *,
        project_root: str | Path = PROJECT_ROOT,
        python_executable: Optional[str] = None,
        stage1_defaults: Optional[Stage1TrainDefaults] = None,
        stage2_defaults: Optional[Stage2TrainDefaults] = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve(strict=False)
        self.python_executable = python_executable or sys.executable
        self.config_path = resolve_training_config_path(config_path)
        config = load_training_default_config(self.config_path, require_exists=False)
        self.stage1_defaults = stage1_defaults or build_dataclass_defaults(Stage1TrainDefaults, "stage1", config=config)
        self.stage2_defaults = stage2_defaults or build_dataclass_defaults(Stage2TrainDefaults, "stage2", config=config)

    def build_contract(self, work_dir: str | Path, *, refresh_status: bool = True) -> dict[str, Any]:
        write_training_interface_json(work_dir, refresh_status=refresh_status)
        return build_training_input_contract(work_dir, refresh_status=False)

    def resolve_training_parameters(
        self,
        stage: str,
        *,
        ui_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve YAML/builtin defaults plus the allowed per-run UI overrides."""

        if stage == "stage1":
            defaults = asdict(self.stage1_defaults)
        elif stage == "stage2":
            defaults = asdict(self.stage2_defaults)
        else:
            raise ValueError(f"Unsupported training stage: {stage!r}")

        normalized = _normalize_training_ui_overrides(stage, ui_overrides)
        effective = dict(defaults)
        effective.update(normalized)
        parameter_sources = {
            key: "ui_override" if key in normalized else "default"
            for key in effective
        }
        return {
            "schema_version": "voicelab_training_parameter_resolution_v1",
            "stage": stage,
            "config_path": str(self.config_path),
            "config_loaded": self.config_path.is_file(),
            "default_source": "yaml" if self.config_path.is_file() else "builtin_fallback",
            "editable_ui_fields": sorted(TRAINING_UI_OVERRIDE_FIELDS),
            "defaults": defaults,
            "ui_override": normalized,
            "effective_parameters": effective,
            "parameter_sources": parameter_sources,
        }

    def get_training_progress(
        self,
        work_dir: str | Path,
        stage: str,
    ) -> dict[str, Any]:
        """Read the latest run state and derive a live progress snapshot.

        This method is intentionally read-only. The Worker owns run_state writes;
        UI progress refreshes must not race with Worker status/PID updates.
        """

        stage_key = str(stage).strip().lower()
        if stage_key not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported training stage: {stage!r}")

        stage_state_path = _stage_root(work_dir, stage_key) / TRAINING_RUN_STATE_FILENAME
        state = _read_json(stage_state_path)
        if not state:
            return {
                "schema_version": "voicelab_training_progress_view_v1",
                "stage": stage_key,
                "state_found": False,
                "run_state_path": str(stage_state_path),
                "progress_snapshot": None,
            }

        snapshot = snapshot_from_run_state(state, stage=stage_key)
        return {
            "schema_version": "voicelab_training_progress_view_v1",
            "stage": stage_key,
            "state_found": True,
            "run_id": state.get("run_id"),
            "run_state_path": str(stage_state_path),
            "status": state.get("status"),
            "effective_parameters": state.get("effective_parameters") or {},
            "terminal_title": state.get("terminal_title"),
            "progress_snapshot": snapshot.to_dict(),
        }

    def _validate_stage1_inputs(self, contract: dict[str, Any]) -> None:
        _require_file(_stage1_path(contract, "train_manifest"), label="Stage1 train_manifest", nonempty=True)
        _require_file(_stage1_path(contract, "val_manifest"), label="Stage1 val_manifest")
        _require_dir(_stage1_path(contract, "output_root"), label="Stage1 output_root")
        _require_dir(_stage1_path(contract, "frontend_root"), label="Stage1 frontend_root")
        _require_dir(_stage1_path(contract, "semantic_root"), label="Stage1 semantic_root")
        _require_file(self.project_root / "scripts/train_stage1_fewshot.py", label="Stage1 training script")

    def _validate_stage2_inputs(self, contract: dict[str, Any]) -> None:
        _require_dir(_stage_path(contract, "stage2", "split_paths", "train_pt_root"), label="Stage2 train_pt_root", pattern="*.pt")
        _require_dir(_stage_path(contract, "stage2", "split_paths", "val_pt_root"), label="Stage2 val_pt_root", pattern="*.pt")
        _require_file(self.project_root / DEFAULT_STAGE2_BASE_CKPT_RELATIVE, label="Default Stage2 base checkpoint")
        _require_file(self.project_root / "scripts/train_fewshot_stage2.py", label="Stage2 training script")

    def build_stage1_train_command(
        self,
        contract: dict[str, Any],
        *,
        run_id: str | None = None,
        ui_overrides: Mapping[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        ok, problems = validate_stage1_ready(contract)
        if not ok:
            raise ValueError("Stage1 training data is not ready: " + "; ".join(problems))
        self._validate_stage1_inputs(contract)
        parameters = self.resolve_training_parameters("stage1", ui_overrides=ui_overrides)
        d = parameters["effective_parameters"]
        work_dir = _as_path(contract.get("work_dir"), label="contract.work_dir")
        run_paths = _make_run_dirs(work_dir, "stage1", run_id)
        cmd = [
            self.python_executable, "scripts/train_stage1_fewshot.py",
            "--train_manifest", _stage1_path(contract, "train_manifest"),
            "--val_manifest", _stage1_path(contract, "val_manifest"),
            "--root_dir", _stage1_path(contract, "output_root"),
            "--output_dir", str(run_paths["checkpoints_dir"]),
            "--device", str(d["device"]), "--epochs", str(int(d["epochs"])),
            "--batch_size", str(int(d["batch_size"])), "--lr", str(d["lr"]),
            "--weight_decay", str(d["weight_decay"]), "--grad_clip", str(d["grad_clip"]),
            "--trainable_scope", str(d["trainable_scope"]), "--last_n_layers", str(int(d["last_n_layers"])),
            "--log_every_steps", str(int(d["log_every_steps"])),
        ]
        if bool(d["use_amp"]):
            cmd.append("--use_amp")
        return cmd, run_paths

    def build_stage2_train_command(
        self,
        contract: dict[str, Any],
        *,
        run_id: str | None = None,
        ui_overrides: Mapping[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        ok, problems = validate_stage2_ready(contract)
        if not ok:
            raise ValueError("Stage2 training data is not ready: " + "; ".join(problems))
        self._validate_stage2_inputs(contract)
        parameters = self.resolve_training_parameters("stage2", ui_overrides=ui_overrides)
        d = parameters["effective_parameters"]
        work_dir = _as_path(contract.get("work_dir"), label="contract.work_dir")
        run_paths = _make_run_dirs(work_dir, "stage2", run_id)
        cmd = [
            self.python_executable, "scripts/train_fewshot_stage2.py",
            "--base_ckpt", DEFAULT_STAGE2_BASE_CKPT_RELATIVE.as_posix(),
            "--fewshot_train_root", _stage_path(contract, "stage2", "split_paths", "train_pt_root"),
            "--fewshot_val_root", _stage_path(contract, "stage2", "split_paths", "val_pt_root"),
            "--output_dir", str(run_paths["checkpoints_dir"]),
            "--device", str(d["device"]), "--epochs", str(int(d["epochs"])),
            "--batch_size", str(int(d["batch_size"])), "--lr", str(d["lr"]),
            "--num_workers", str(int(d["num_workers"])), "--use_amp", _bool_text(bool(d["use_amp"])),
            "--trainable_scope", str(d["trainable_scope"]),
            "--include_continuous_semantic", _bool_text(bool(d["include_continuous_semantic"])),
            "--require_continuous_semantic", _bool_text(bool(d["require_continuous_semantic"])),
            "--include_v66_style_fields", _bool_text(bool(d["include_v66_style_fields"])),
            "--require_v66_style_fields", _bool_text(bool(d["require_v66_style_fields"])),
            "--use_reference_acoustic_style", _bool_text(bool(d["use_reference_acoustic_style"])),
            "--reference_style_fusion_mode", str(d["reference_style_fusion_mode"]),
            "--reference_style_gate_init", str(d["reference_style_gate_init"]),
            "--use_v66_energy_loss", _bool_text(bool(d["use_v66_energy_loss"])),
            "--v66_energy_loss_weight", str(d["v66_energy_loss_weight"]),
            "--use_v66_f0_loss", _bool_text(bool(d["use_v66_f0_loss"])),
            "--v66_f0_loss_weight", str(d["v66_f0_loss_weight"]),
            "--use_v66_speaker_loss", _bool_text(bool(d["use_v66_speaker_loss"])),
            "--v66_speaker_loss_weight", str(d["v66_speaker_loss_weight"]),
            "--v66_speaker_embedding_dim", str(int(d["v66_speaker_embedding_dim"])),
            "--v66_require_aux_targets", _bool_text(bool(d["v66_require_aux_targets"])),
            "--use_bootstrapper_lora", _bool_text(bool(d["use_bootstrapper_lora"])),
            "--bootstrapper_lora_rank", str(int(d["bootstrapper_lora_rank"])),
            "--bootstrapper_lora_alpha", str(d["bootstrapper_lora_alpha"]),
            "--bootstrapper_lora_dropout", str(d["bootstrapper_lora_dropout"]),
            "--bootstrapper_lora_target", str(d["bootstrapper_lora_target"]),
            "--bootstrapper_lora_init_scale", str(d["bootstrapper_lora_init_scale"]),
            "--use_bootstrapper_attention_lora", _bool_text(bool(d["use_bootstrapper_attention_lora"])),
            "--bootstrapper_attention_lora_rank", str(int(d["bootstrapper_attention_lora_rank"])),
            "--bootstrapper_attention_lora_alpha", str(d["bootstrapper_attention_lora_alpha"]),
            "--bootstrapper_attention_lora_dropout", str(d["bootstrapper_attention_lora_dropout"]),
            "--bootstrapper_attention_lora_target", str(d["bootstrapper_attention_lora_target"]),
            "--bootstrapper_attention_lora_gate_init", str(d["bootstrapper_attention_lora_gate_init"]),
            "--bootstrapper_attention_lora_enable_q", _bool_text(bool(d["bootstrapper_attention_lora_enable_q"])),
            "--bootstrapper_attention_lora_enable_k", _bool_text(bool(d["bootstrapper_attention_lora_enable_k"])),
            "--bootstrapper_attention_lora_enable_v", _bool_text(bool(d["bootstrapper_attention_lora_enable_v"])),
            "--bootstrapper_attention_lora_enable_o", _bool_text(bool(d["bootstrapper_attention_lora_enable_o"])),
            "--load_strict", _bool_text(bool(d["load_strict"])),
            "--force_v65_coarse_only_defaults", _bool_text(bool(d["force_v65_coarse_only_defaults"])),
        ]
        return cmd, run_paths

    def _launch_command(
        self,
        *,
        stage: str,
        work_dir: str | Path,
        command: list[str],
        run_paths: dict[str, Any],
        contract: dict[str, Any],
        parameter_resolution: dict[str, Any] | None = None,
        show_terminal_progress: bool = False,
    ) -> dict[str, Any]:
        active = _existing_active_run(work_dir, stage)
        if active:
            raise RuntimeError(
                f"{stage} training is already running: run_id={active.get('run_id')}, pid={active.get('pid')}. "
                "Stop the active run before launching another one."
            )
        run_id = str(run_paths["run_id"])
        resolved_parameters = parameter_resolution or self.resolve_training_parameters(stage)
        effective_parameters = resolved_parameters["effective_parameters"]
        terminal_requested = bool(show_terminal_progress)
        terminal_title = _training_terminal_title(
            stage=stage,
            run_id=run_id,
            effective_parameters=effective_parameters,
        )
        command_payload = {
            "schema_version": "voicelab_training_command_v5",
            "stage": stage, "run_id": run_id, "created_at": _now_iso(),
            "cwd": str(self.project_root), "command": command,
            "defaults": resolved_parameters["defaults"],
            "ui_override": resolved_parameters["ui_override"],
            "effective_parameters": effective_parameters,
            "parameter_sources": resolved_parameters["parameter_sources"],
            "override_policy": {
                "editable_ui_fields": resolved_parameters["editable_ui_fields"],
                "scope": "this_run_only",
                "yaml_defaults_are_not_modified": True,
            },
            "terminal_policy": {
                "show_terminal_progress": terminal_requested,
                "viewer_only": True,
                "title": terminal_title,
                "primary_log": str(run_paths["stdout_log_path"]),
                "extra_logs": [str(run_paths["stderr_log_path"])],
            },
            "progress_policy": {
                "mode": "read_only_log_snapshot",
                "run_state_writer": "worker",
                "service_live_reader": "get_training_progress",
            },
            "config_path": str(self.config_path), "config_loaded": self.config_path.is_file(),
            "stage2_base_ckpt_relative": DEFAULT_STAGE2_BASE_CKPT_RELATIVE.as_posix() if stage == "stage2" else None,
        }
        _write_json(run_paths["command_json_path"], command_payload)
        state = {
            "schema_version": "voicelab_training_run_state_v5",
            "stage": stage, "run_id": run_id, "status": "pending",
            "message": f"{stage} training worker is starting.",
            "work_dir": str(Path(work_dir).expanduser().resolve(strict=False)),
            "stage_root": str(run_paths["stage_root"]), "run_root": str(run_paths["run_root"]),
            "output_dir": str(run_paths["checkpoints_dir"]),
            "training_interface_path": contract.get("training_interface_path"),
            "command_json_path": str(run_paths["command_json_path"]),
            "stdout_log_path": str(run_paths["stdout_log_path"]),
            "stderr_log_path": str(run_paths["stderr_log_path"]),
            "run_state_path": str(run_paths["run_state_path"]),
            "stage_state_path": str(run_paths["stage_state_path"]),
            "ui_override": resolved_parameters["ui_override"],
            "effective_parameters": effective_parameters,
            "terminal_requested": terminal_requested,
            "terminal_started": False,
            "terminal_error": None,
            "terminal_title": terminal_title,
            "pid": None, "worker_pid": None, "child_pid": None,
            "command": command, "problems": [], "stop_requested": False,
            "created_at": _now_iso(), "started_at": None, "ended_at": None,
            "updated_at": _now_iso(), "returncode": None,
        }
        state["progress_snapshot"] = snapshot_from_run_state(
            state,
            stage=stage,
        ).to_dict()
        _write_json(run_paths["run_state_path"], state)
        _write_json(run_paths["stage_state_path"], state)
        _write_json(run_paths["latest_run_path"], {
            "schema_version": "voicelab_latest_training_run_v1",
            "stage": stage, "run_id": run_id, "run_root": str(run_paths["run_root"]),
            "run_state_path": str(run_paths["run_state_path"]), "updated_at": _now_iso(),
        })
        worker_cmd = [
            self.python_executable, "-m", "src.voicelab_user.training.worker",
            "--command_json", str(run_paths["command_json_path"]),
            "--run_state", str(run_paths["run_state_path"]),
            "--stage_state", str(run_paths["stage_state_path"]),
        ]
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags = int(subprocess.CREATE_NEW_PROCESS_GROUP)
        process = subprocess.Popen(
            worker_cmd, cwd=str(self.project_root),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        terminal_started = False
        terminal_error: str | None = None
        if terminal_requested:
            try:
                terminal_process = launch_log_tail_terminal(
                    run_paths["stdout_log_path"],
                    extra_log_paths=[run_paths["stderr_log_path"]],
                    title=terminal_title,
                    tail_lines=80,
                    keep_open=True,
                )
                terminal_started = terminal_process is not None
                if terminal_process is None and os.name != "nt":
                    terminal_error = "Training log terminal is only available on Windows."
            except Exception as exc:
                terminal_error = f"{type(exc).__name__}: {exc}"

        state.update(
            {
                "status": "running",
                "message": f"{stage} training worker launched.",
                "pid": int(process.pid),
                "worker_pid": int(process.pid),
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "terminal_started": terminal_started,
                "terminal_error": terminal_error,
            }
        )
        state["progress_snapshot"] = snapshot_from_run_state(
            state,
            stage=stage,
        ).to_dict()
        _write_json(run_paths["run_state_path"], state)
        _write_json(run_paths["stage_state_path"], state)
        return state

    def launch_stage1_training(
        self,
        work_dir: str | Path,
        *,
        refresh_status: bool = True,
        ui_overrides: Mapping[str, Any] | None = None,
        show_terminal_progress: bool = False,
    ) -> dict[str, Any]:
        parameter_resolution = self.resolve_training_parameters(
            "stage1",
            ui_overrides=ui_overrides,
        )
        contract = self.build_contract(work_dir, refresh_status=refresh_status)
        active = _existing_active_run(contract["work_dir"], "stage1")
        if active:
            raise RuntimeError(f"stage1 training is already running: pid={active.get('pid')}")
        command, run_paths = self.build_stage1_train_command(
            contract,
            ui_overrides=parameter_resolution["ui_override"],
        )
        return self._launch_command(
            stage="stage1",
            work_dir=contract["work_dir"],
            command=command,
            run_paths=run_paths,
            contract=contract,
            parameter_resolution=parameter_resolution,
            show_terminal_progress=show_terminal_progress,
        )

    def launch_stage2_training(
        self,
        work_dir: str | Path,
        *,
        refresh_status: bool = True,
        ui_overrides: Mapping[str, Any] | None = None,
        show_terminal_progress: bool = False,
    ) -> dict[str, Any]:
        parameter_resolution = self.resolve_training_parameters(
            "stage2",
            ui_overrides=ui_overrides,
        )
        contract = self.build_contract(work_dir, refresh_status=refresh_status)
        active = _existing_active_run(contract["work_dir"], "stage2")
        if active:
            raise RuntimeError(f"stage2 training is already running: pid={active.get('pid')}")
        command, run_paths = self.build_stage2_train_command(
            contract,
            ui_overrides=parameter_resolution["ui_override"],
        )
        return self._launch_command(
            stage="stage2",
            work_dir=contract["work_dir"],
            command=command,
            run_paths=run_paths,
            contract=contract,
            parameter_resolution=parameter_resolution,
            show_terminal_progress=show_terminal_progress,
        )


def create_training_service(**kwargs: Any) -> TrainingService:
    return TrainingService(**kwargs)


__all__ = [
    "PROJECT_ROOT", "TRAINING_RUN_DIRNAME", "TRAINING_RUN_STATE_FILENAME",
    "TRAINING_COMMAND_FILENAME", "TRAINING_STDOUT_FILENAME", "TRAINING_STDERR_FILENAME",
    "LATEST_RUN_FILENAME", "RUNS_DIRNAME", "DEFAULT_STAGE2_BASE_CKPT_RELATIVE",
    "TRAINING_UI_OVERRIDE_FIELDS",
    "Stage1TrainDefaults", "Stage2TrainDefaults", "TrainingService", "create_training_service",
]
