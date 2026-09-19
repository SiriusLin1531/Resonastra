"""
User-edition inference service for VoiceLab.

中文说明：
    本文件是用户版推理入口的核心 service 层。
    它负责：
        1. 接收 UserInferenceRequest；
        2. 调用 P3 config.py 构建 UserInferenceConfig；
        3. 准备本次推理输出目录；
        4. 保存 user_request.json 和 internal_summary.json；
        5. 在 dry_run=True 时只做准备，不执行重推理；
        6. 在 dry_run=False 时通过 adapter.py 调用现有稳定开发者推理脚本。

English notes:
    This module is the core service boundary for VoiceLab user-edition inference.
    It prepares validated configs, creates a run directory, writes user-facing
    and internal summaries, and delegates real inference to adapter.py.

Important design rule:
重要设计规则：

    service.py must not directly import GPT-SoVITS, Stage2 acoustic models,
    vocoders, torch, DNSMOS, ASR, or speaker-sim backends.

    service.py 不直接 import GPT-SoVITS、Stage2、vocoder、torch 或指标后端。
    真正推理通过 DeveloperInferenceAdapter 的 subprocess 边界完成。
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from .adapter import (
    DEFAULT_DEVELOPER_SCRIPT,
    create_developer_inference_adapter,
)
from .config import (
    UserInferenceConfig,
    build_user_inference_config,
    config_to_internal_summary_dict,
    config_to_user_dict,
)
from .result import (
    RESULT_STATUS_FAILED,
    RESULT_STATUS_NOT_IMPLEMENTED,
    RESULT_STATUS_PREPARED,
    UserInferenceResult,
    make_failed_result,
    make_not_implemented_result,
    make_prepared_result,
    write_result_json,
)
from ..runtime.path_resolver import PathLike, normalize_path_text, resolve_project_path


# -----------------------------------------------------------------------------
# Service status aliases
# service 状态别名
# -----------------------------------------------------------------------------

SERVICE_STATUS_PREPARED = RESULT_STATUS_PREPARED
SERVICE_STATUS_NOT_IMPLEMENTED = RESULT_STATUS_NOT_IMPLEMENTED
SERVICE_STATUS_FAILED = RESULT_STATUS_FAILED


# -----------------------------------------------------------------------------
# Request / prepared run dataclasses
# 请求与准备阶段数据结构
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class UserInferenceRequest:
    """User-facing inference request.

    中文说明：
        面向 CLI / WebUI 的用户输入请求。
        后续 Gradio 控件值应被转换成这个 dataclass，再交给 UserInferenceService。

    English:
        User-facing request object shared by future CLI and WebUI layers.
    """

    # Required user inputs.
    # 必填用户输入。
    prompt_wav_path: PathLike
    prompt_text: str
    target_text: str

    # Optional profile/checkpoint/output overrides.
    # 可选 profile / checkpoint / 输出目录覆盖。
    profile_dir: Optional[PathLike] = None
    stage1_ckpt: Optional[PathLike] = None
    stage2_ckpt: Optional[PathLike] = None
    output_dir: Optional[PathLike] = None

    # Runtime and sampling overrides.
    # 运行设备与采样参数覆盖。
    device: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    length_scale: Optional[float] = None

    # Optional quality metrics.
    # 可选质量指标开关。
    enable_metrics: Optional[bool] = None
    compute_dnsmos: Optional[bool] = None
    compute_speaker_sim: Optional[bool] = None
    compute_wer: Optional[bool] = None


@dataclass(frozen=True)
class PreparedInferenceRun:
    """Prepared run context before heavy model inference starts.

    中文说明：
        这是推理真正开始前的准备结果。
        包含已解析配置、输出目录、用户请求 JSON、内部摘要 JSON、prompt copy 路径等。
    """

    config: UserInferenceConfig
    run_dir: Path
    request_json_path: Path
    internal_summary_json_path: Path
    prompt_copy_path: Optional[Path] = None


# -----------------------------------------------------------------------------
# Environment/path helpers
# 环境变量与路径工具
# -----------------------------------------------------------------------------


@contextmanager
def _temporary_user_root(project_root: Optional[Path]) -> Iterator[None]:
    """Temporarily set VOICELAB_USER_ROOT for path_resolver-based config loading.

    English:
        P3 config builders resolve relative paths through path_resolver.py.
        The service-level project_root must therefore be reflected into the
        environment while building configs.

    中文：
        P3 的配置构建依赖 path_resolver.py 解析相对路径。
        如果 service 接收了 project_root，就必须在构建配置时临时写入
        VOICELAB_USER_ROOT，否则 project_root 会“看似传入但实际不生效”。
    """

    if project_root is None:
        yield
        return

    old_value = os.environ.get("VOICELAB_USER_ROOT")
    os.environ["VOICELAB_USER_ROOT"] = str(project_root)

    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("VOICELAB_USER_ROOT", None)
        else:
            os.environ["VOICELAB_USER_ROOT"] = old_value


def _now_stamp() -> str:
    """Return timestamp used in run directory names.

    使用微秒，避免同一秒内多次点击推理导致输出目录冲突。
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe_name(text: str, *, fallback: str = "run") -> str:
    """Convert arbitrary text into a safe filename component.

    将任意文本转换成适合作为目录名组成部分的字符串。
    """

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _to_jsonable(value: Any) -> Any:
    """Convert service objects containing Path/dataclass values to JSON-safe values.

    将 Path / dataclass 等对象转换为 JSON 可序列化对象。
    """

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write UTF-8 JSON file.

    写入 UTF-8 JSON 文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _make_run_dir(cfg: UserInferenceConfig) -> Path:
    """Create the run directory for one inference request.

    为一次推理创建输出目录。
    """

    if cfg.output.output_dir is not None:
        run_dir = cfg.output.output_dir
    else:
        profile_name = _safe_name(cfg.profile_name, fallback="profile")
        run_name = f"{_now_stamp()}_{profile_name}"
        run_dir = cfg.output.output_root / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _copy_prompt_if_needed(cfg: UserInferenceConfig, run_dir: Path) -> Optional[Path]:
    """Copy reference prompt wav into the run directory if enabled.

    如果配置允许，则把参考音频复制到本次 run 目录。
    """

    if not cfg.output.copy_prompt_wav:
        return None

    src = cfg.prompt_wav_path
    suffix = src.suffix or ".wav"
    dst = run_dir / f"prompt_reference{suffix}"

    if src.resolve(strict=False) != dst.resolve(strict=False):
        shutil.copy2(src, dst)

    return dst


def _command_to_log_dict(command: Any) -> dict[str, Any]:
    """Return a compact JSON-safe command summary.

    将 adapter 生成的 DeveloperInferenceCommand 转换成适合保存到 JSON 的摘要。

    注意：
        不保存完整 env，因为 PATH / PYTHONPATH 可能很长。
        这里只保存关键环境变量，方便排查路径问题。
    """

    env = getattr(command, "env", {}) or {}

    return {
        "schema_version": "voicelab_developer_command_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": list(getattr(command, "argv", [])),
        "display_command": (
            command.display_command()
            if hasattr(command, "display_command")
            else " ".join(str(x) for x in getattr(command, "argv", []))
        ),
        "cwd": getattr(command, "cwd", None),
        "script_path": getattr(command, "script_path", None),
        "artifacts": getattr(command, "artifacts", None),
        "env_preview": {
            "VOICELAB_USER_ROOT": env.get("VOICELAB_USER_ROOT"),
            "PYTHONPATH": env.get("PYTHONPATH"),
        },
    }


# -----------------------------------------------------------------------------
# Main service
# 主服务
# -----------------------------------------------------------------------------


class UserInferenceService:
    """Stable service boundary for CLI and Gradio user-edition inference.

    中文说明：
        这是未来 CLI 和 Gradio WebUI 的共同推理入口。
        UI 不应该直接调用 scripts/infer_zeroshot_v641.py。
        UI 应该构造 UserInferenceRequest，然后调用本 service。
    """

    def __init__(
        self,
        *,
        default_config_path: PathLike = "configs/user_inference_default.yaml",
        project_root: Optional[PathLike] = None,
        developer_script_path: PathLike = DEFAULT_DEVELOPER_SCRIPT,
        python_executable: Optional[PathLike] = None,
        inference_timeout_seconds: Optional[float] = None,
    ) -> None:
        """Create a user inference service.

        Args:
            default_config_path:
                用户版默认 YAML 配置路径。

            project_root:
                VoiceLab 项目根目录。若传入，将临时写入 VOICELAB_USER_ROOT，
                让 config/path 解析稳定。

            developer_script_path:
                现有稳定开发者推理脚本路径，默认 scripts/infer_zeroshot_v641.py。

            python_executable:
                子进程使用的 Python。
                可以是：
                    - None：使用当前 sys.executable；
                    - runtime/env/python.exe；
                    - python；
                    - python.exe。

            inference_timeout_seconds:
                subprocess 推理超时时间。None 表示不限制。
        """

        self.default_config_path = default_config_path
        self.project_root = (
            resolve_project_path(project_root)
            if normalize_path_text(project_root)
            else None
        )
        self.developer_script_path = developer_script_path
        self.python_executable = python_executable
        self.inference_timeout_seconds = inference_timeout_seconds

    def build_config(self, request: UserInferenceRequest) -> UserInferenceConfig:
        """Build and validate UserInferenceConfig from a service request.

        从用户请求构建并校验 UserInferenceConfig。
        """

        with _temporary_user_root(self.project_root):
            return build_user_inference_config(
                prompt_wav_path=request.prompt_wav_path,
                prompt_text=request.prompt_text,
                target_text=request.target_text,
                profile_dir=request.profile_dir,
                stage1_ckpt=request.stage1_ckpt,
                stage2_ckpt=request.stage2_ckpt,
                output_dir=request.output_dir,
                device=request.device,
                seed=request.seed,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                length_scale=request.length_scale,
                enable_metrics=request.enable_metrics,
                compute_dnsmos=request.compute_dnsmos,
                compute_speaker_sim=request.compute_speaker_sim,
                compute_wer=request.compute_wer,
                default_config_path=self.default_config_path,
            )

    def prepare_run(self, request: UserInferenceRequest) -> PreparedInferenceRun:
        """Prepare output directory and write request/config summaries.

        准备输出目录，并写入：
            - user_request.json
            - internal_summary.json
        """

        cfg = self.build_config(request)
        run_dir = _make_run_dir(cfg)
        prompt_copy_path = _copy_prompt_if_needed(cfg, run_dir)

        request_json_path = _write_json(
            run_dir / "user_request.json",
            {
                "schema_version": "voicelab_user_service_request_v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "request": asdict(request),
                "resolved_config": config_to_user_dict(cfg),
            },
        )

        internal_summary_json_path = _write_json(
            run_dir / "internal_summary.json",
            {
                "schema_version": "voicelab_user_service_internal_summary_v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "summary": config_to_internal_summary_dict(cfg),
                "run_dir": run_dir,
                "prompt_copy_path": prompt_copy_path,
            },
        )

        return PreparedInferenceRun(
            config=cfg,
            run_dir=run_dir,
            request_json_path=request_json_path,
            internal_summary_json_path=internal_summary_json_path,
            prompt_copy_path=prompt_copy_path,
        )

    def run(
        self,
        request: UserInferenceRequest,
        *,
        dry_run: bool = True,
    ) -> UserInferenceResult:
        """Prepare a user inference run and optionally execute real inference.

        English:
            dry_run=True is still the default safe behavior.
            dry_run=False triggers DeveloperInferenceAdapter and runs the
            existing stable developer inference script through subprocess.

        中文：
            dry_run=True 仍然是默认安全行为，只准备配置和输出目录。
            dry_run=False 才会通过 DeveloperInferenceAdapter 调用现有稳定开发者推理脚本。
        """

        prepared = self.prepare_run(request)

        if dry_run:
            result = make_prepared_result(
                run_dir=prepared.run_dir,
                message=(
                    "Inference run prepared successfully. "
                    "Heavy inference is not executed in dry-run mode."
                ),
                request_json_path=prepared.request_json_path,
                internal_summary_json_path=prepared.internal_summary_json_path,
                prompt_copy_path=prepared.prompt_copy_path,
                profile_name=prepared.config.profile_name,
                inference_mode=prepared.config.inference_mode,
            )
            write_result_json(result)
            return result

        try:
            result = self._run_inference_impl(prepared)
            write_result_json(result)
            return result

        except NotImplementedError as exc:
            result = make_not_implemented_result(
                run_dir=prepared.run_dir,
                message=str(exc),
                request_json_path=prepared.request_json_path,
                internal_summary_json_path=prepared.internal_summary_json_path,
                prompt_copy_path=prepared.prompt_copy_path,
                profile_name=prepared.config.profile_name,
                inference_mode=prepared.config.inference_mode,
            )
            write_result_json(result)
            return result

        except Exception as exc:  # noqa: BLE001 - user-facing service boundary.
            result = make_failed_result(
                run_dir=prepared.run_dir,
                message=f"User inference service failed: {exc}",
                error_type=exc.__class__.__name__,
                stage="user_inference_service",
                traceback_text=traceback.format_exc(),
                hint=(
                    "Check user_request.json, internal_summary.json, "
                    "developer_command.json if it exists, and console logs."
                ),
                request_json_path=prepared.request_json_path,
                internal_summary_json_path=prepared.internal_summary_json_path,
                prompt_copy_path=prepared.prompt_copy_path,
                profile_name=prepared.config.profile_name,
                inference_mode=prepared.config.inference_mode,
            )
            write_result_json(result)
            return result

    def _run_inference_impl(self, prepared: PreparedInferenceRun) -> UserInferenceResult:
        """Run real user-edition inference through DeveloperInferenceAdapter.

        English:
            This method is the P5-4 connection point:
                UserInferenceService
                    -> DeveloperInferenceAdapter
                        -> scripts/infer_zeroshot_v641.py
                            -> existing stable inference chain

        中文：
            这是 P5-4 的真正接入点：
                用户版 service
                    -> adapter.py
                        -> 开发者推理脚本
                            -> 当前稳定推理链
        """

        adapter = create_developer_inference_adapter(
            project_root=self.project_root,
            script_path=self.developer_script_path,
            python_executable=self.python_executable,
        )

        command = adapter.build_command(
            prepared.config,
            output_dir=prepared.run_dir,
        )

        developer_command_json_path = _write_json(
            prepared.run_dir / "developer_command.json",
            _command_to_log_dict(command),
        )

        execution = adapter.run_subprocess(
            command,
            timeout_seconds=self.inference_timeout_seconds,
        )

        result = adapter.execution_to_result(
            execution,
            request_json_path=prepared.request_json_path,
            internal_summary_json_path=prepared.internal_summary_json_path,
        )

        # Add service-side command record path into result.extra.
        # 把 service 层保存的 developer_command.json 路径补充到 result.extra。
        result = replace(
            result,
            profile_name=prepared.config.profile_name,
            inference_mode=prepared.config.inference_mode,
            language=prepared.config.language,
            extra={
                **(result.extra or {}),
                "developer_command_json_path": str(developer_command_json_path),
            },
        )

        return result


def create_user_inference_service(
    *,
    default_config_path: PathLike = "configs/user_inference_default.yaml",
    project_root: Optional[PathLike] = None,
    developer_script_path: PathLike = DEFAULT_DEVELOPER_SCRIPT,
    python_executable: Optional[PathLike] = None,
    inference_timeout_seconds: Optional[float] = None,
) -> UserInferenceService:
    """Factory used by future CLI and WebUI layers.

    给未来 CLI / WebUI 使用的 service 工厂函数。
    """

    return UserInferenceService(
        default_config_path=default_config_path,
        project_root=project_root,
        developer_script_path=developer_script_path,
        python_executable=python_executable,
        inference_timeout_seconds=inference_timeout_seconds,
    )


__all__ = [
    "SERVICE_STATUS_PREPARED",
    "SERVICE_STATUS_NOT_IMPLEMENTED",
    "SERVICE_STATUS_FAILED",
    "UserInferenceRequest",
    "PreparedInferenceRun",
    "UserInferenceService",
    "create_user_inference_service",
]