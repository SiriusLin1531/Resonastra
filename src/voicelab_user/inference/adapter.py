"""
Adapter between VoiceLab user-edition service and the stable developer inference script.

中文说明：
    本文件是“用户版推理服务层”和“现有开发者推理脚本”之间的隔离层。
    它不会直接 import scripts/infer_zeroshot_v641.py，而是通过 subprocess 调用该脚本。

English notes:
    This module is the isolation layer between the user-edition inference service
    and the existing developer inference script.  It does not import
    scripts/infer_zeroshot_v641.py directly.  Instead, it builds a subprocess
    command and optionally executes it.

Why subprocess instead of direct import?
为什么使用 subprocess 而不是直接 import？

    scripts/infer_zeroshot_v641.py imports heavy runtime dependencies at module
    import time, including torch, GPT-SoVITS adapters, Stage2 pipeline, vocoder
    code, and optional metrics backends.

    也就是说，如果用户版 service.py 直接 import 开发者脚本，会让用户版服务层变重，
    并且可能污染当前 WebUI 进程的 sys.path、CUDA 状态、模型全局状态等。

    Subprocess keeps this boundary clean:
        UserInferenceService
            -> DeveloperInferenceAdapter
                -> subprocess: python scripts/infer_zeroshot_v641.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import UserInferenceConfig
from .result import (
    InferenceArtifactPaths,
    InferenceErrorInfo,
    InferenceMetricSummary,
    InferenceTiming,
    UserInferenceResult,
    RESULT_STATUS_FAILED,
    RESULT_STATUS_SUCCEEDED,
)
from ..runtime.path_resolver import PathLike, normalize_path_text, resolve_project_path


# -----------------------------------------------------------------------------
# Developer script constants
# 开发者推理脚本相关常量
# -----------------------------------------------------------------------------

DEFAULT_DEVELOPER_SCRIPT = "scripts/infer_zeroshot_v641.py"

# Current fixed output names produced by scripts/infer_zeroshot_v641.py.
# 这些文件名来自当前稳定开发者推理脚本。
DEFAULT_OUTPUT_WAV_NAME = "zeroshot_v641.wav"
DEFAULT_PEAKNORM_WAV_NAME = "zeroshot_v641_peaknorm.wav"
DEFAULT_SUMMARY_JSON_NAME = "zeroshot_v641_summary.json"
DEFAULT_METRICS_JSON_NAME = "zeroshot_v641_metrics.json"
DEFAULT_MEL_NPY_NAME = "zeroshot_v641_mel.npy"
DEFAULT_MEL_PT_NAME = "zeroshot_v641_mel.pt"


# User-edition standardized output names.
# 用户版标准化输出文件名。
USER_OUTPUT_WAV_NAME = "inference_output.wav"
USER_PEAKNORM_WAV_NAME = "inference_output_peaknorm.wav"
USER_SUMMARY_JSON_NAME = "inference_summary.json"
USER_METRICS_JSON_NAME = "inference_metrics.json"

# -----------------------------------------------------------------------------
# Small helpers
# 小工具函数
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    """Return local ISO timestamp.

    返回本地 ISO 时间戳。
    """

    return datetime.now().isoformat(timespec="seconds")


def _bool_text(value: bool) -> str:
    """Convert bool to the str2bool-friendly form expected by developer script.

    将 bool 转成开发者脚本 str2bool 能识别的字符串。
    """

    return "true" if bool(value) else "false"


def _append_arg(argv: list[str], name: str, value: Any) -> None:
    """Append a CLI argument only when value is not None.

    仅当 value 非 None 时添加命令行参数。
    """

    if value is None:
        return
    argv.extend([name, str(value)])


def _prepend_path_list(existing: str, values: Sequence[Path]) -> str:
    """Prepend paths to an environment PATH-like variable.

    将若干路径加入 PATH / PYTHONPATH 类环境变量的开头。
    """

    parts = [str(p) for p in values if p is not None]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _looks_like_path(text: str) -> bool:
    """Return whether text should be treated as a filesystem path.

    判断字符串是否更像文件路径，而不是 PATH 中的命令名。

    Examples:
        "D:/VoiceLab/runtime/env/python.exe" -> path
        "runtime/env/python.exe"             -> path
        "python"                             -> command
        "python.exe"                         -> command on Windows PATH
    """

    if Path(text).is_absolute():
        return True
    return ("/" in text) or ("\\" in text)


def _resolve_against_project_root(path: PathLike, project_root: Path) -> Path:
    """Resolve absolute or project-root-relative path.

    解析绝对路径或项目根目录相对路径。
    """

    text = normalize_path_text(path)
    if text is None:
        raise ValueError("path must not be empty.")

    raw = Path(text).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (project_root / raw).resolve(strict=False)


def _resolve_python_executable(
    python_executable: Optional[PathLike],
    *,
    project_root: Path,
) -> tuple[str, bool]:
    """Resolve python executable.

    解析 Python 可执行入口。

    Returns:
        (executable_text, is_filesystem_path)

    支持三种形式：
        1. None
           -> 使用当前 sys.executable，通常是 runtime/env/python.exe。
        2. 绝对路径或项目相对路径
           -> D:/VoiceLab/runtime/env/python.exe
           -> runtime/env/python.exe
        3. PATH 命令
           -> python
           -> python.exe
    """

    text = normalize_path_text(python_executable)

    if text is None:
        # sys.executable is normally an absolute path.
        # sys.executable 通常是绝对路径。
        return str(Path(sys.executable).resolve(strict=False)), True

    if _looks_like_path(text):
        return str(_resolve_against_project_root(text, project_root)), True

    # Treat as PATH command, e.g. "python" or "python.exe".
    # 作为 PATH 命令处理，例如 python 或 python.exe。
    return text, False


def _python_runtime_path_entries(executable_text: str, *, is_path: bool) -> list[Path]:
    """Return useful PATH entries for a filesystem python executable.

    如果 Python 是具体文件路径，则尝试把 runtime/env、Scripts、Library/bin
    加入子进程 PATH，帮助 Windows 下找到 DLL / ffmpeg / 依赖库。
    """

    if not is_path:
        return []

    exe_path = Path(executable_text)
    env_dir = exe_path.parent

    entries = [
        env_dir,
        env_dir / "Scripts",
        env_dir / "Library" / "bin",
    ]

    return [p for p in entries if p.exists()]


def _float_or_none(value: Any) -> Optional[float]:
    """Best-effort float conversion.

    尽力把值转成 float，失败则返回 None。
    """

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    """Return value if it is a dict, otherwise empty dict.

    若 value 是 dict 则返回，否则返回空 dict。
    """

    return value if isinstance(value, dict) else {}


def _copy_file_if_exists(
    src: Optional[Path],
    dst: Path,
) -> Optional[Path]:
    """Copy src to dst if src exists and return dst.

    中文说明：
        如果 src 文件存在，就复制到 dst，并返回 dst。
        如果 src 不存在，则返回 None。

        P9-4 用于保留开发者原始输出，同时额外生成用户版标准命名文件。
    """

    if src is None:
        return None

    if not src.exists() or not src.is_file():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)

    src_resolved = src.resolve(strict=False)
    dst_resolved = dst.resolve(strict=False)

    if src_resolved == dst_resolved:
        return dst_resolved

    shutil.copy2(src_resolved, dst_resolved)
    return dst_resolved


def _standardized_user_artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return standardized user-edition artifact paths for one run dir.

    返回用户版标准化输出路径。
    """

    return {
        "output_wav": output_dir / USER_OUTPUT_WAV_NAME,
        "peaknorm_wav": output_dir / USER_PEAKNORM_WAV_NAME,
        "summary_json": output_dir / USER_SUMMARY_JSON_NAME,
        "metrics_json": output_dir / USER_METRICS_JSON_NAME,
    }


def _read_json_optional(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Read JSON dict if path exists.

    如果 JSON 文件存在，则读取；不存在或顶层不是 dict 则返回 None。
    """

    if path is None or not path.exists() or not path.is_file():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _stream_to_text(value: Any) -> str:
    """Convert subprocess stdout/stderr payload to text.

    中文说明：
        subprocess.TimeoutExpired 中的 stdout / stderr 可能是：
            - None
            - str
            - bytes

        这里统一转换成 str，方便后续写入 DeveloperInferenceExecution。
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _timeout_stderr_message(timeout_seconds: Optional[float]) -> str:
    """Build a stable timeout stderr message.

    中文说明：
        构造稳定的 timeout 错误文本。
        注意这里保留关键短语：
            subprocess timed out after X seconds
        方便后续日志检索和 WebUI 展示。
    """

    if timeout_seconds is None:
        return "subprocess timed out."

    return f"subprocess timed out after {timeout_seconds:g} seconds."


def _path_from_summary_outputs(
    summary: Optional[dict[str, Any]],
    key: str,
    *,
    fallback: Optional[Path] = None,
) -> Optional[Path]:
    """Extract an output path from developer summary outputs section.

    从开发者 summary 的 outputs 字段中提取路径。
    如果没有，则使用 fallback。
    """

    if isinstance(summary, dict):
        outputs = _dict_or_empty(summary.get("outputs"))
        value = normalize_path_text(outputs.get(key))
        if value is not None:
            return Path(value).resolve(strict=False)

    return fallback


def _extract_timing_from_summary(summary: Optional[dict[str, Any]]) -> InferenceTiming:
    """Extract timing fields from developer summary.

    从 zeroshot_v641_summary.json 中提取耗时信息，映射到正式 InferenceTiming。

    Developer runtime fields may include:
        pipeline_load_seconds
        prompt_semantic_seconds
        frontend_seconds
        stage1_seconds
        continuous_semantic_seconds
        stage2_seconds
        vocoder_seconds
        generation_seconds
        generation_rtf
        metrics_seconds
        metrics_rtf
        total_seconds
        total_rtf
    """

    runtime = _dict_or_empty(summary.get("runtime") if isinstance(summary, dict) else None)

    # For now, use pipeline_load_seconds as prepare_seconds.
    # 当前先把 pipeline_load_seconds 作为 prepare_seconds。
    # More detailed prepare timing can be refined later.
    prepare_seconds = _float_or_none(runtime.get("pipeline_load_seconds"))

    return InferenceTiming(
        finished_at=_now_iso(),
        prepare_seconds=prepare_seconds,
        stage1_seconds=_float_or_none(runtime.get("stage1_seconds")),
        stage2_seconds=_float_or_none(runtime.get("stage2_seconds")),
        vocoder_seconds=_float_or_none(runtime.get("vocoder_seconds")),
        metrics_seconds=_float_or_none(runtime.get("metrics_seconds")),
        total_seconds=_float_or_none(runtime.get("total_seconds")),
        rtf=(
            _float_or_none(runtime.get("total_rtf"))
            or _float_or_none(runtime.get("generation_rtf"))
            or _float_or_none(runtime.get("rtf"))
        ),
    )


def _extract_metrics_from_summary(summary: Optional[dict[str, Any]]) -> Optional[InferenceMetricSummary]:
    """Extract user-visible metrics from developer summary.

    从开发者 summary 中提取用户可见质量指标。

    Expected structure from current developer script:
        summary["metrics"]["dnsmos"]["ovrl"]
        summary["metrics"]["dnsmos"]["sig"]
        summary["metrics"]["dnsmos"]["bak"]
        summary["metrics"]["wer"]["wer"]
        summary["metrics"]["wer"]["cer"]
        summary["metrics"]["speaker_sim"]["cosine"]
    """

    if not isinstance(summary, dict):
        return None

    metrics = _dict_or_empty(summary.get("metrics"))
    if not metrics:
        return None

    dnsmos = _dict_or_empty(metrics.get("dnsmos"))
    wer_info = _dict_or_empty(metrics.get("wer"))
    speaker_info = _dict_or_empty(metrics.get("speaker_sim"))

    runtime = _dict_or_empty(summary.get("runtime"))

    return InferenceMetricSummary(
        dnsmos_ovrl=_float_or_none(dnsmos.get("ovrl")),
        dnsmos_sig=_float_or_none(dnsmos.get("sig")),
        dnsmos_bak=_float_or_none(dnsmos.get("bak")),
        speaker_sim=_float_or_none(speaker_info.get("cosine")),
        wer=_float_or_none(wer_info.get("wer")),
        cer=_float_or_none(wer_info.get("cer")),
        rtf=(
            _float_or_none(metrics.get("total_rtf"))
            or _float_or_none(metrics.get("generation_rtf"))
            or _float_or_none(metrics.get("rtf"))
            or _float_or_none(runtime.get("total_rtf"))
            or _float_or_none(runtime.get("generation_rtf"))
        ),
        extra=metrics,
    )


# -----------------------------------------------------------------------------
# Dataclasses
# 数据结构
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeveloperInferenceArtifacts:
    """Expected files produced by scripts/infer_zeroshot_v641.py.

    开发者推理脚本预期会生成的文件路径。
    """

    output_dir: Path
    output_wav_path: Path
    peaknorm_wav_path: Optional[Path]
    summary_json_path: Path
    metrics_json_path: Optional[Path]
    mel_npy_path: Optional[Path]
    mel_pt_path: Optional[Path]
    prompt_copy_path: Optional[Path]


@dataclass(frozen=True)
class DeveloperInferenceCommand:
    """Subprocess command plan for the stable developer inference script.

    对开发者推理脚本的一次 subprocess 调用计划。
    """

    argv: list[str]
    cwd: Path
    env: dict[str, str]
    script_path: Path
    artifacts: DeveloperInferenceArtifacts

    def display_command(self) -> str:
        """Return a readable command string for logs/debugging.

        返回便于日志展示的命令字符串。
        """

        return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in self.argv)


@dataclass(frozen=True)
class DeveloperInferenceExecution:
    """Raw subprocess execution result.

    subprocess 执行后的原始结果。
    """

    command: DeveloperInferenceCommand
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


# -----------------------------------------------------------------------------
# Main adapter
# 主适配器
# -----------------------------------------------------------------------------


class DeveloperInferenceAdapter:
    """Isolated adapter for invoking scripts/infer_zeroshot_v641.py.

    用于隔离调用开发者推理脚本的适配器。
    """

    def __init__(
        self,
        *,
        project_root: Optional[PathLike] = None,
        script_path: PathLike = DEFAULT_DEVELOPER_SCRIPT,
        python_executable: Optional[PathLike] = None,
    ) -> None:
        # Resolve project root first, because script_path and relative python path
        # may depend on it.
        #
        # 先解析项目根目录，因为 script_path 和相对 python 路径都可能依赖它。
        self.project_root = (
            resolve_project_path(project_root)
            if normalize_path_text(project_root)
            else resolve_project_path(".")
        )

        if self.project_root is None:
            raise ValueError("project_root could not be resolved.")

        self.script_path = _resolve_against_project_root(script_path, self.project_root)

        self.python_executable, self.python_executable_is_path = _resolve_python_executable(
            python_executable,
            project_root=self.project_root,
        )

    def validate(self) -> None:
        """Validate adapter-level filesystem assumptions.

        校验 adapter 层所依赖的文件系统假设。
        """

        if not self.project_root.exists() or not self.project_root.is_dir():
            raise FileNotFoundError(f"VoiceLab project root not found: {self.project_root}")

        if not self.script_path.exists() or not self.script_path.is_file():
            raise FileNotFoundError(f"Developer inference script not found: {self.script_path}")

        if self.python_executable_is_path:
            python_path = Path(self.python_executable)
            if not python_path.exists() or not python_path.is_file():
                raise FileNotFoundError(f"Python executable not found: {python_path}")
        else:
            # PATH command, for example "python" or "python.exe".
            # PATH 命令，例如 python 或 python.exe。
            if shutil.which(self.python_executable) is None:
                raise FileNotFoundError(
                    f"Python command not found on PATH: {self.python_executable}"
                )

    def build_environment(self) -> dict[str, str]:
        """Build subprocess environment without mutating the current process.

        构建子进程环境变量，不污染当前 Python/WebUI 进程。
        """

        env = dict(os.environ)

        # Make user-edition path resolver stable inside subprocess.
        # 让子进程中的 path_resolver 能稳定找到用户版项目根目录。
        env["VOICELAB_USER_ROOT"] = str(self.project_root)

        # Make repository imports stable:
        #   import src.xxx
        #   import voicelab_user.xxx
        #
        # 保证子进程可以稳定 import 项目代码。
        env["PYTHONPATH"] = _prepend_path_list(
            env.get("PYTHONPATH", ""),
            [self.project_root, self.project_root / "src"],
        )

        # If using bundled runtime/env/python.exe, make its DLL/search paths visible.
        # 如果使用 runtime/env/python.exe，则补充 runtime 相关 PATH。
        runtime_path_entries = _python_runtime_path_entries(
            self.python_executable,
            is_path=self.python_executable_is_path,
        )
        if runtime_path_entries:
            env["PATH"] = _prepend_path_list(env.get("PATH", ""), runtime_path_entries)

        return env

    def expected_artifacts(
        self,
        cfg: UserInferenceConfig,
        output_dir: Path,
    ) -> DeveloperInferenceArtifacts:
        """Return expected developer-script output artifact paths.

        返回开发者脚本预期输出的文件路径。
        """

        return DeveloperInferenceArtifacts(
            output_dir=output_dir,
            output_wav_path=output_dir / DEFAULT_OUTPUT_WAV_NAME,
            peaknorm_wav_path=(output_dir / DEFAULT_PEAKNORM_WAV_NAME)
            if cfg.output.save_peaknorm
            else None,
            summary_json_path=output_dir / DEFAULT_SUMMARY_JSON_NAME,
            metrics_json_path=(output_dir / DEFAULT_METRICS_JSON_NAME)
            if cfg.metrics.enable_metrics
            else None,
            mel_npy_path=(output_dir / DEFAULT_MEL_NPY_NAME)
            if cfg.output.save_debug_tensors
            else None,
            mel_pt_path=(output_dir / DEFAULT_MEL_PT_NAME)
            if cfg.output.save_debug_tensors
            else None,
            prompt_copy_path=(output_dir / "prompt_reference.wav")
            if cfg.output.copy_prompt_wav
            else None,
        )

    def build_command(
        self,
        cfg: UserInferenceConfig,
        *,
        output_dir: PathLike,
    ) -> DeveloperInferenceCommand:
        """Translate UserInferenceConfig into scripts/infer_zeroshot_v641.py CLI args.

        将用户版 UserInferenceConfig 转换为开发者脚本 CLI 参数。
        """

        self.validate()

        resolved_output_dir = _resolve_against_project_root(output_dir, self.project_root)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = self.expected_artifacts(cfg, resolved_output_dir)

        argv: list[str] = [self.python_executable, str(self.script_path)]

        # ------------------------------------------------------------------
        # Required user inputs
        # 必填用户输入
        # ------------------------------------------------------------------
        _append_arg(argv, "--target_text", cfg.target_text)
        _append_arg(argv, "--prompt_wav_path", cfg.prompt_wav_path)
        _append_arg(argv, "--prompt_text", cfg.prompt_text)
        _append_arg(argv, "--target_language", cfg.language)
        _append_arg(argv, "--prompt_language", cfg.language)
        _append_arg(argv, "--output_dir", resolved_output_dir)
        _append_arg(argv, "--stage2_ckpt", cfg.stage2.checkpoint_path)

        # ------------------------------------------------------------------
        # Optional GPT-SoVITS Stage1 checkpoint override
        # 可选 Stage1 checkpoint 覆盖
        # ------------------------------------------------------------------
        _append_arg(argv, "--stage1_ckpt", cfg.stage1.checkpoint_path)

        # TODO:
        #   The developer script also supports --sovits_checkpoint_path.
        #   Current user config/profile does not yet expose a resolved SoVITS path.
        #   For now we intentionally do not pass it here, and keep the behavior
        #   aligned with the existing developer script defaults.
        #
        # TODO 中文：
        #   开发者脚本还支持 --sovits_checkpoint_path。
        #   但当前用户版 config/profile 还没有正式暴露 resolved SoVITS 路径。
        #   因此本版本暂时不传该参数，保持与现有开发者脚本默认行为一致。
        #
        # Future recommended fix:
        #   Add default_sovits_checkpoint_path to profile/config resolution.
        #
        # 后续推荐修复：
        #   在 profile/config 解析层增加 default_sovits_checkpoint_path。

        # ------------------------------------------------------------------
        # Stage1 text construction and sampling
        # Stage1 文本构造与采样参数
        # ------------------------------------------------------------------
        _append_arg(argv, "--stage1_text_mode", cfg.stage1.text_mode)
        _append_arg(argv, "--prompt_target_separator", cfg.stage1.prompt_target_separator)
        _append_arg(argv, "--stage1_top_k", cfg.stage1.top_k)
        _append_arg(argv, "--stage1_top_p", cfg.stage1.top_p)
        _append_arg(argv, "--stage1_temperature", cfg.stage1.temperature)
        _append_arg(argv, "--stage1_repetition_penalty", cfg.stage1.repetition_penalty)
        _append_arg(argv, "--stage1_early_stop_num", cfg.stage1.early_stop_num)
        _append_arg(argv, "--crop_stage1_prompt_prefix", _bool_text(cfg.stage1.crop_prompt_prefix))
        _append_arg(argv, "--stage1_crop_mode", cfg.stage1.crop_mode)
        _append_arg(argv, "--stage1_append_tail_mute_tokens", cfg.stage1.append_tail_mute_tokens)
        _append_arg(argv, "--stage1_mute_token_id", cfg.stage1.mute_token_id)

        # ------------------------------------------------------------------
        # Continuous semantic and Stage2 acoustic generation
        # continuous semantic 与 Stage2 声学生成
        # ------------------------------------------------------------------
        _append_arg(argv, "--attach_continuous_semantic", _bool_text(cfg.stage2.attach_continuous_semantic))
        _append_arg(argv, "--continuous_dtype", cfg.stage2.continuous_dtype)
        _append_arg(argv, "--require_continuous_semantic", _bool_text(cfg.stage2.require_continuous_semantic))
        _append_arg(argv, "--target_length_mode", cfg.stage2.target_length_mode)
        _append_arg(argv, "--length_scale", cfg.stage2.length_scale)

        # User edition maps "standard acoustic generation" to the stable internal
        # coarse-only path.
        #
        # 用户版把 “standard acoustic generation” 映射到稳定的 coarse_only 内部路径。
        _append_arg(argv, "--sampling_start_mode", cfg.stage2.internal_sampling_mode)

        # Current stable user-edition path does not expose residual/refiner sampling.
        # 当前用户版稳定路径不开放 residual/refiner 采样。
        _append_arg(argv, "--num_steps", 1)
        _append_arg(argv, "--temperature", 0.3)
        _append_arg(argv, "--guidance_scale", 1.0)
        _append_arg(argv, "--return_bootstrap_mel", "true")

        # ------------------------------------------------------------------
        # Vocoder settings
        # 声码器设置
        # ------------------------------------------------------------------
        _append_arg(argv, "--vocoder_type", cfg.vocoder.vocoder_type)
        _append_arg(argv, "--vocoder_profile", cfg.vocoder.vocoder_profile)
        _append_arg(argv, "--hifigan_root", cfg.vocoder.hifigan_root)
        _append_arg(argv, "--vocoder_checkpoint_path", cfg.vocoder.checkpoint_path)
        _append_arg(argv, "--vocoder_config_path", cfg.vocoder.config_path)

        # ------------------------------------------------------------------
        # Runtime and output behavior
        # 运行设备与输出控制
        # ------------------------------------------------------------------
        _append_arg(argv, "--device", cfg.runtime.device)
        _append_arg(argv, "--seed", cfg.runtime.seed)
        _append_arg(argv, "--save_numpy", _bool_text(cfg.output.save_debug_tensors))
        _append_arg(argv, "--save_pt", _bool_text(cfg.output.save_debug_tensors))
        _append_arg(argv, "--save_peaknorm", _bool_text(cfg.output.save_peaknorm))
        _append_arg(argv, "--copy_prompt_wav", _bool_text(cfg.output.copy_prompt_wav))
        _append_arg(argv, "--suppress_stage1_progress", "true")

        # ------------------------------------------------------------------
        # Optional metrics
        # 可选质量指标
        # ------------------------------------------------------------------
        _append_arg(argv, "--enable_metrics", _bool_text(cfg.metrics.enable_metrics))
        if artifacts.metrics_json_path is not None:
            _append_arg(argv, "--metrics_output_json", artifacts.metrics_json_path)

        _append_arg(argv, "--compute_dnsmos", _bool_text(cfg.metrics.compute_dnsmos))
        _append_arg(argv, "--compute_wer", _bool_text(cfg.metrics.compute_wer))
        _append_arg(argv, "--compute_speaker_sim", _bool_text(cfg.metrics.compute_speaker_sim))
        _append_arg(argv, "--dnsmos_onnx_path", cfg.metrics.dnsmos_onnx_path)

        # ASR / WER settings.
        # 中文说明：
        #   WER/CER 依赖 ASR。用户版默认传入：
        #       --asr_backend faster_whisper
        #       --asr_device cpu
        #       --asr_compute_type int8
        #
        #   这样可以避免 faster-whisper / ctranslate2 在 Windows runtime 中
        #   误走 CUDA 12 依赖并报 cublas64_12.dll 缺失。
        _append_arg(argv, "--asr_backend", cfg.metrics.asr_backend)
        _append_arg(argv, "--asr_model", cfg.metrics.asr_model)
        _append_arg(argv, "--asr_language", cfg.language)
        _append_arg(argv, "--asr_device", cfg.metrics.asr_device)
        _append_arg(argv, "--asr_compute_type", cfg.metrics.asr_compute_type)

        _append_arg(argv, "--speaker_device", cfg.metrics.speaker_device)

        return DeveloperInferenceCommand(
            argv=argv,
            cwd=self.project_root,
            env=self.build_environment(),
            script_path=self.script_path,
            artifacts=artifacts,
        )

    def run_subprocess(
        self,
        command: DeveloperInferenceCommand,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> DeveloperInferenceExecution:
        """Execute a prepared developer inference command.

        English:
            Execute the prepared developer inference command.

            Timeout handling is intentionally done here instead of letting
            subprocess.TimeoutExpired bubble up to service.py.  This keeps the
            adapter contract stable:

                adapter.run_subprocess(...)
                    -> DeveloperInferenceExecution

            even when the subprocess times out.

        中文说明：
            执行已经构造好的开发者推理命令。

            timeout 需要在 adapter 层处理，而不是继续抛给 service.py。
            这样无论正常结束、脚本失败还是超时，adapter 都稳定返回：

                DeveloperInferenceExecution

            后续 execution_to_result() 再统一转成 UserInferenceResult。
        """

        try:
            proc = subprocess.run(
                command.argv,
                cwd=str(command.cwd),
                env=command.env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )

            return DeveloperInferenceExecution(
                command=command,
                returncode=int(proc.returncode),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
            )

        except subprocess.TimeoutExpired as exc:
            # Keep any captured stdout/stderr if Python managed to collect it
            # before killing the subprocess.
            #
            # 中文说明：
            # 如果超时前 subprocess 已经输出了一部分 stdout/stderr，
            # 尽量保留下来，方便判断卡在模型加载、Stage1、Stage2 还是 vocoder。
            captured_stdout = _stream_to_text(getattr(exc, "stdout", None))
            captured_stderr = _stream_to_text(getattr(exc, "stderr", None))

            timeout_message = _timeout_stderr_message(timeout_seconds)

            stderr_parts = [timeout_message]
            if captured_stderr:
                stderr_parts.append("")
                stderr_parts.append("Captured stderr before timeout:")
                stderr_parts.append(captured_stderr)

            return DeveloperInferenceExecution(
                command=command,
                returncode=-124,
                stdout=captured_stdout,
                stderr="\n".join(stderr_parts),
            )


    def load_summary(
        self,
        artifacts: DeveloperInferenceArtifacts,
    ) -> Optional[dict[str, Any]]:
        """Load developer summary JSON if it exists.

        如果开发者 summary JSON 存在，则读取它。
        """

        return _read_json_optional(artifacts.summary_json_path)

    def execution_to_result(
        self,
        execution: DeveloperInferenceExecution,
        *,
        request_json_path: Optional[Path] = None,
        internal_summary_json_path: Optional[Path] = None,
    ) -> UserInferenceResult:
        """Convert raw subprocess execution into formal UserInferenceResult.

        将 subprocess 原始执行结果转换为用户版正式 UserInferenceResult。

        Args:
            execution:
                Raw subprocess execution result.
                原始 subprocess 结果。

            request_json_path:
                user_request.json written by UserInferenceService.prepare_run().
                service.py 生成的 user_request.json。

            internal_summary_json_path:
                internal_summary.json written by UserInferenceService.prepare_run().
                service.py 生成的 internal_summary.json。

        Important:
            Developer summary is a different file:
                zeroshot_v641_summary.json

            It is stored separately in:
                extra["developer_summary_json_path"]

        注意：
            service.py 的 internal_summary.json
            和开发者脚本的 zeroshot_v641_summary.json
            不是同一个文件，不能混用。
        """

        dev_artifacts = execution.command.artifacts
        developer_summary = self.load_summary(dev_artifacts)

        output_wav_path = _path_from_summary_outputs(
            developer_summary,
            "wav_path",
            fallback=dev_artifacts.output_wav_path,
        )

        peaknorm_wav_path = _path_from_summary_outputs(
            developer_summary,
            "peaknorm_wav_path",
            fallback=dev_artifacts.peaknorm_wav_path,
        )

        mel_npy_path = _path_from_summary_outputs(
            developer_summary,
            "mel_npy_path",
            fallback=dev_artifacts.mel_npy_path,
        )

        mel_pt_path = _path_from_summary_outputs(
            developer_summary,
            "mel_pt_path",
            fallback=dev_artifacts.mel_pt_path,
        )

        prompt_copy_path = _path_from_summary_outputs(
            developer_summary,
            "prompt_reference_copy",
            fallback=dev_artifacts.prompt_copy_path,
        )

        metrics_json_path = (
            dev_artifacts.metrics_json_path
            if dev_artifacts.metrics_json_path is not None and dev_artifacts.metrics_json_path.exists()
            else None
        )

        # ------------------------------------------------------------------
        # P9-4: Standardized user-edition output names
        # P9-4：用户版标准化输出命名
        # ------------------------------------------------------------------
        standardized_paths = _standardized_user_artifact_paths(dev_artifacts.output_dir)

        standardized_output_wav_path = _copy_file_if_exists(
            output_wav_path,
            standardized_paths["output_wav"],
        )
        standardized_peaknorm_wav_path = _copy_file_if_exists(
            peaknorm_wav_path,
            standardized_paths["peaknorm_wav"],
        )
        standardized_summary_json_path = _copy_file_if_exists(
            dev_artifacts.summary_json_path,
            standardized_paths["summary_json"],
        )
        standardized_metrics_json_path = _copy_file_if_exists(
            metrics_json_path,
            standardized_paths["metrics_json"],
        )

        user_artifacts = InferenceArtifactPaths(
            run_dir=dev_artifacts.output_dir,
            request_json_path=request_json_path,
            internal_summary_json_path=internal_summary_json_path,
            result_json_path=dev_artifacts.output_dir / "result.json",
            output_wav_path=standardized_output_wav_path,
            output_peaknorm_wav_path=standardized_peaknorm_wav_path,
            stage2_acoustic_path=(
                mel_pt_path
                if mel_pt_path is not None and mel_pt_path.exists()
                else None
            ),
            mel_preview_path=(
                mel_npy_path
                if mel_npy_path is not None and mel_npy_path.exists()
                else None
            ),
            metrics_json_path=standardized_metrics_json_path,
            prompt_copy_path=(
                prompt_copy_path
                if prompt_copy_path is not None and prompt_copy_path.exists()
                else None
            ),
        )

        timing = _extract_timing_from_summary(developer_summary)
        metric_summary = _extract_metrics_from_summary(developer_summary)

        common_extra: dict[str, Any] = {
            "adapter": "DeveloperInferenceAdapter",
            "developer_script": str(execution.command.script_path),
            "developer_summary_json_path": (
                str(standardized_summary_json_path)
                if standardized_summary_json_path is not None and standardized_summary_json_path.exists()
                else None
            ),
            "developer_raw_summary_json_path": (
                str(dev_artifacts.summary_json_path)
                if dev_artifacts.summary_json_path.exists()
                else None
            ),
            "developer_raw_metrics_json_path": (
                str(metrics_json_path)
                if metrics_json_path is not None and metrics_json_path.exists()
                else None
            ),
            "developer_summary_loaded": developer_summary is not None,
            "developer_summary_keys": (
                sorted(str(k) for k in developer_summary.keys())
                if isinstance(developer_summary, dict)
                else []
            ),
            "standardized_output_names": {
                "output_wav": str(standardized_paths["output_wav"]),
                "peaknorm_wav": str(standardized_paths["peaknorm_wav"]),
                "summary_json": str(standardized_paths["summary_json"]),
                "metrics_json": str(standardized_paths["metrics_json"]),
            },
            "returncode": execution.returncode,
            "stdout_tail": execution.stdout[-4000:],
            "stderr_tail": execution.stderr[-4000:],
        }

        # Treat returncode=0 but missing wav as a failure, because user-facing
        # inference is not actually complete without audio output.
        #
        # 如果 returncode=0 但没有 wav，也视为失败，因为用户最终需要音频文件。
        has_output_wav = user_artifacts.output_wav_path is not None

        if execution.succeeded and has_output_wav:
            return UserInferenceResult(
                schema_version="voicelab_user_inference_result_v1",
                status=RESULT_STATUS_SUCCEEDED,
                message="Developer inference subprocess completed successfully.",
                artifacts=user_artifacts,
                timing=timing,
                metrics=metric_summary,
                extra=common_extra,
            )

        if execution.succeeded and not has_output_wav:
            error_message = (
                "Developer inference subprocess returned code 0, "
                "but the expected output wav was not found."
            )
            return UserInferenceResult(
                schema_version="voicelab_user_inference_result_v1",
                status=RESULT_STATUS_FAILED,
                message=error_message,
                artifacts=user_artifacts,
                timing=timing,
                metrics=metric_summary,
                error=InferenceErrorInfo(
                    error_type="MissingOutputWavError",
                    message=error_message,
                    stage="developer_inference_outputs",
                    hint=(
                        "Check zeroshot_v641_summary.json and stdout/stderr. "
                        "The script may have completed without saving the expected wav."
                    ),
                ),
                extra=common_extra,
            )

        if execution.returncode == -124:
            error_message = "Developer inference subprocess timed out."

            return UserInferenceResult(
                schema_version="voicelab_user_inference_result_v1",
                status=RESULT_STATUS_FAILED,
                message=error_message,
                artifacts=user_artifacts,
                timing=timing,
                metrics=metric_summary,
                error=InferenceErrorInfo(
                    error_type="DeveloperInferenceTimeoutError",
                    message=(
                        execution.stderr.strip()
                        or "subprocess timed out after the configured timeout."
                    ),
                    stage="developer_inference_subprocess_timeout",
                    traceback_text=execution.stderr[-8000:] if execution.stderr else None,
                    hint=(
                        "The developer inference subprocess exceeded the configured timeout. "
                        "Increase Timeout Seconds in the WebUI, for example to 600 or 1800. "
                        "This is common when models are cold-started or the target text is long."
                    ),
                ),
                extra={
                    **common_extra,
                    "timeout": True,
                    "timeout_returncode": -124,
                },
            )


        return UserInferenceResult(
            schema_version="voicelab_user_inference_result_v1",
            status=RESULT_STATUS_FAILED,
            message="Developer inference subprocess failed.",
            artifacts=user_artifacts,
            timing=timing,
            metrics=metric_summary,
            error=InferenceErrorInfo(
                error_type="DeveloperInferenceSubprocessError",
                message=f"Subprocess exited with return code {execution.returncode}.",
                stage="developer_inference_subprocess",
                traceback_text=execution.stderr[-8000:] if execution.stderr else None,
                hint=(
                    "Check stdout/stderr, user_request.json, internal_summary.json, "
                    "and zeroshot_v641_summary.json if it was created."
                ),
            ),
            extra=common_extra,
        )


def create_developer_inference_adapter(
    *,
    project_root: Optional[PathLike] = None,
    script_path: PathLike = DEFAULT_DEVELOPER_SCRIPT,
    python_executable: Optional[PathLike] = None,
) -> DeveloperInferenceAdapter:
    """Factory used by UserInferenceService in a later P5 step.

    给 service.py 后续接入使用的工厂函数。
    """

    return DeveloperInferenceAdapter(
        project_root=project_root,
        script_path=script_path,
        python_executable=python_executable,
    )


__all__ = [
    "DEFAULT_DEVELOPER_SCRIPT",
    "DEFAULT_OUTPUT_WAV_NAME",
    "DEFAULT_PEAKNORM_WAV_NAME",
    "DEFAULT_SUMMARY_JSON_NAME",
    "DEFAULT_METRICS_JSON_NAME",
    "DEFAULT_MEL_NPY_NAME",
    "DEFAULT_MEL_PT_NAME",
    "USER_OUTPUT_WAV_NAME",
    "USER_PEAKNORM_WAV_NAME",
    "USER_SUMMARY_JSON_NAME",
    "USER_METRICS_JSON_NAME",
    "DeveloperInferenceArtifacts",
    "DeveloperInferenceCommand",
    "DeveloperInferenceExecution",
    "DeveloperInferenceAdapter",
    "create_developer_inference_adapter",
]