from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return data


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _update_states(run_state_path: Path, stage_state_path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    state = _read_json(run_state_path)
    state.update(updates)
    state["updated_at"] = _now_iso()
    _write_json(run_state_path, state)
    _write_json(stage_state_path, state)
    return state


def _best_candidate(stage: str, output_dir: Path) -> Path | None:
    names = ["best_model.ckpt", "best_model.pt", "best_model.pth"] if stage == "stage1" else ["best_model.pt", "best_model.pth", "best_model.ckpt"]
    for name in names:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    return None


def _register_best(state: dict[str, Any]) -> dict[str, Any] | None:
    stage = str(state.get("stage") or "")
    output_dir = Path(str(state.get("output_dir") or "")).expanduser().resolve(strict=False)
    candidate = _best_candidate(stage, output_dir)
    if candidate is None:
        return None
    from .checkpoint_manager import register_trainer_best
    return register_trainer_best(
        state["work_dir"],
        stage=stage,
        run_id=str(state.get("run_id") or ""),
        checkpoint_path=candidate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceLab persistent training worker")
    parser.add_argument("--command_json", required=True)
    parser.add_argument("--run_state", required=True)
    parser.add_argument("--stage_state", required=True)
    args = parser.parse_args()

    command_json_path = Path(args.command_json).expanduser().resolve(strict=False)
    run_state_path = Path(args.run_state).expanduser().resolve(strict=False)
    stage_state_path = Path(args.stage_state).expanduser().resolve(strict=False)
    command_payload = _read_json(command_json_path)
    state = _read_json(run_state_path)
    command = command_payload.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"Invalid training command in {command_json_path}")

    stdout_path = Path(str(state["stdout_log_path"])).expanduser().resolve(strict=False)
    stderr_path = Path(str(state["stderr_log_path"])).expanduser().resolve(strict=False)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    start_monotonic = time.monotonic()

    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_f, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_f:
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=str(command_payload.get("cwd") or Path.cwd()),
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
            )
            state = _update_states(
                run_state_path,
                stage_state_path,
                {
                    "status": "running",
                    "message": f"{state.get('stage')} training process is running.",
                    "pid": int(os.getpid()),
                    "worker_pid": int(os.getpid()),
                    "child_pid": int(process.pid),
                    "started_at": state.get("started_at") or _now_iso(),
                },
            )
            returncode = int(process.wait())

        latest = _read_json(run_state_path)
        stopped = bool(latest.get("stop_requested"))
        if stopped:
            status, message = "stopped", "Training was stopped by user request."
        elif returncode == 0:
            status, message = "succeeded", "Training completed successfully."
        else:
            status, message = "failed", f"Training exited with returncode={returncode}."

        best_meta = None
        best_error = None
        if status == "succeeded":
            try:
                best_meta = _register_best(latest)
            except Exception as exc:
                best_error = f"{type(exc).__name__}: {exc}"

        _update_states(
            run_state_path,
            stage_state_path,
            {
                "status": status,
                "message": message,
                "returncode": returncode,
                "ended_at": _now_iso(),
                "duration_seconds": round(time.monotonic() - start_monotonic, 4),
                "child_pid": None,
                "auto_best": best_meta,
                "auto_best_error": best_error,
            },
        )
        return returncode
    except BaseException as exc:
        try:
            _update_states(
                run_state_path,
                stage_state_path,
                {
                    "status": "failed",
                    "message": f"Training worker failed: {type(exc).__name__}: {exc}",
                    "returncode": -1,
                    "ended_at": _now_iso(),
                    "duration_seconds": round(time.monotonic() - start_monotonic, 4),
                    "worker_error_type": type(exc).__name__,
                    "worker_error": str(exc),
                },
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    sys.exit(main())
