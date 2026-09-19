"""
Dataclass-based configuration layer for VoiceLab user-edition inference.

P3 connects three sources of configuration:
    1. configs/user_inference_default.yaml
    2. user_profiles/<profile_name>/profile_config.json
    3. CLI/UI user overrides such as stage1_ckpt, stage2_ckpt,
       temperature, top_p, top_k, seed, and metrics flags.

This module is intentionally lightweight. It must not import torch,
GPT-SoVITS, Stage2 models, vocoders, ASR backends, or metrics backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled at runtime with a clear message.
    yaml = None  # type: ignore[assignment]

from ..profiles.profile_loader import (
    ResolvedModelPaths,
    VoiceProfile,
    detect_inference_mode,
    load_voice_profile,
    profile_to_summary_dict,
    resolve_model_paths,
)
from ..runtime.path_resolver import (
    PathLike,
    ensure_file_exists,
    normalize_path_text,
    resolve_project_path,
    safe_relpath,
)


class UserInferenceConfigError(ValueError):
    """Raised when user-edition inference configuration is invalid."""


@dataclass(frozen=True)
class UserStage1Config:
    """User-edition Stage1 text-to-semantic configuration."""

    checkpoint_path: Optional[Path] = None

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 15

    repetition_penalty: float = 1.35
    early_stop_num: int = -1

    text_mode: str = "prompt_plus_target"
    prompt_target_separator: str = ""

    crop_prompt_prefix: bool = True
    crop_mode: str = "auto_idx"

    append_tail_mute_tokens: int = 0
    mute_token_id: int = 486


@dataclass(frozen=True)
class UserStage2Config:
    """User-edition Stage2 acoustic generation configuration."""

    checkpoint_path: Path

    acoustic_generation_mode: str = "standard"
    internal_sampling_mode: str = "coarse_only"

    target_length_mode: str = "semantic_ratio"
    length_scale: float = 1.0

    attach_continuous_semantic: bool = True
    require_continuous_semantic: bool = True
    continuous_dtype: str = "float32"


@dataclass(frozen=True)
class UserVocoderConfig:
    """User-edition vocoder configuration."""

    vocoder_type: str = "hifigan"
    vocoder_profile: str = "universal_v1"
    hifigan_root: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    config_path: Optional[Path] = None


@dataclass(frozen=True)
class UserMetricsConfig:
    """Optional user-edition metrics configuration."""

    enable_metrics: bool = False
    compute_dnsmos: bool = True
    compute_speaker_sim: bool = True
    compute_wer: bool = False

    dnsmos_onnx_path: Optional[Path] = None

    speaker_device: str = "cuda"

    # ASR defaults for WER/CER.
    # 中文说明：
    #   WER/CER 依赖 ASR。用户版默认让 ASR 跑 CPU int8，
    #   避免 Windows runtime 中 faster-whisper / ctranslate2 寻找
    #   CUDA 12 cublas64_12.dll 的问题。
    asr_backend: str = "faster_whisper"
    asr_model: str = "medium"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"


@dataclass(frozen=True)
class UserOutputConfig:
    """User-edition output configuration."""

    output_root: Path = field(default_factory=lambda: Path("outputs/inference_runs"))
    output_dir: Optional[Path] = None

    save_peaknorm: bool = True
    copy_prompt_wav: bool = True
    save_debug_tensors: bool = False


@dataclass(frozen=True)
class UserRuntimeConfig:
    """User-edition runtime configuration."""

    device: str = "cuda"
    fallback_to_cpu: bool = True
    seed: Optional[int] = None


@dataclass(frozen=True)
class UserInferenceConfig:
    """Top-level user-edition inference configuration."""

    profile_dir: Path
    profile_name: str
    profile_display_name: str
    inference_mode: str

    prompt_wav_path: Path
    prompt_text: str
    target_text: str

    stage1: UserStage1Config
    stage2: UserStage2Config
    vocoder: UserVocoderConfig
    metrics: UserMetricsConfig
    output: UserOutputConfig
    runtime: UserRuntimeConfig

    language: str = "zh"
    stage1_source: str = "default_gpt_sovits_v2"
    stage2_source: str = "profile"


# -----------------------------------------------------------------------------
# YAML loading
# -----------------------------------------------------------------------------


def load_yaml(path: PathLike) -> dict[str, Any]:
    """Load a UTF-8 YAML file and require a dict at the top level."""

    if yaml is None:
        raise RuntimeError(
            "缺少 PyYAML 依赖，无法读取 YAML 配置文件。\n"
            "请在 VoiceLab 环境中安装 pyyaml，或确认 requirements 已完整安装。"
        )

    resolved_path = resolve_project_path(path)
    ensure_file_exists(resolved_path, label="用户版默认 YAML 配置")
    assert resolved_path is not None

    try:
        data = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - keep user-facing error concise.
        raise UserInferenceConfigError(
            f"YAML 配置文件读取失败：\n  {resolved_path}\n\n错误信息：{exc}"
        ) from exc

    if not isinstance(data, dict):
        raise UserInferenceConfigError(
            f"YAML 配置文件顶层必须是对象 dict：\n  {resolved_path}"
        )

    return data


def load_default_user_config(
    path: PathLike = "configs/user_inference_default.yaml",
) -> dict[str, Any]:
    """Load the user-edition default inference configuration."""

    defaults = load_yaml(path)
    schema_version = defaults.get("schema_version")
    if schema_version != "user_inference_default_v1":
        raise UserInferenceConfigError(
            f"不支持的用户版默认配置 schema_version：{schema_version!r}\n"
            "当前仅支持 user_inference_default_v1。"
        )
    return defaults


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _section(defaults: dict[str, Any], name: str) -> dict[str, Any]:
    value = defaults.get(name, {})
    if not isinstance(value, dict):
        raise UserInferenceConfigError(f"默认配置中的 {name} 必须是对象 dict。")
    return value


def _resolve_optional_project_path(path: Optional[PathLike]) -> Optional[Path]:
    if normalize_path_text(path) is None:
        return None
    return resolve_project_path(path)


def _float_value(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise UserInferenceConfigError(f"配置字段 {name} 必须是数字。") from exc


def _int_value(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise UserInferenceConfigError(f"配置字段 {name} 必须是整数。") from exc


def _bool_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    raise UserInferenceConfigError(f"配置字段 {name} 必须是布尔值。")


def _path_to_str(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return safe_relpath(path)


# -----------------------------------------------------------------------------
# Builders for nested dataclasses
# -----------------------------------------------------------------------------


def build_stage1_config(
    defaults: dict[str, Any],
    *,
    checkpoint_override: Optional[PathLike] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> UserStage1Config:
    """Build UserStage1Config from defaults and user overrides."""

    stage1 = _section(defaults, "stage1")

    checkpoint_path = _resolve_optional_project_path(checkpoint_override)
    if checkpoint_path is None:
        checkpoint_path = _resolve_optional_project_path(stage1.get("checkpoint_path"))

    cfg = UserStage1Config(
        checkpoint_path=checkpoint_path,
        temperature=_float_value(
            temperature if temperature is not None else stage1.get("temperature", 1.0),
            name="stage1.temperature",
        ),
        top_p=_float_value(
            top_p if top_p is not None else stage1.get("top_p", 1.0),
            name="stage1.top_p",
        ),
        top_k=_int_value(
            top_k if top_k is not None else stage1.get("top_k", 15),
            name="stage1.top_k",
        ),
        repetition_penalty=_float_value(
            stage1.get("repetition_penalty", 1.35),
            name="stage1.repetition_penalty",
        ),
        early_stop_num=_int_value(
            stage1.get("early_stop_num", -1),
            name="stage1.early_stop_num",
        ),
        text_mode=str(stage1.get("text_mode", "prompt_plus_target")),
        prompt_target_separator=str(stage1.get("prompt_target_separator", "")),
        crop_prompt_prefix=_bool_value(
            stage1.get("crop_prompt_prefix", True),
            name="stage1.crop_prompt_prefix",
        ),
        crop_mode=str(stage1.get("crop_mode", "auto_idx")),
        append_tail_mute_tokens=_int_value(
            stage1.get("append_tail_mute_tokens", 0),
            name="stage1.append_tail_mute_tokens",
        ),
        mute_token_id=_int_value(
            stage1.get("mute_token_id", 486),
            name="stage1.mute_token_id",
        ),
    )
    validate_stage1_config(cfg)
    return cfg


def build_stage2_config(
    defaults: dict[str, Any],
    *,
    checkpoint_override: Optional[PathLike] = None,
    length_scale: Optional[float] = None,
) -> UserStage2Config:
    """Build UserStage2Config from defaults and user overrides."""

    stage2 = _section(defaults, "stage2")

    checkpoint_path = _resolve_optional_project_path(checkpoint_override)
    if checkpoint_path is None:
        checkpoint_path = _resolve_optional_project_path(stage2.get("checkpoint_path"))
    if checkpoint_path is None:
        raise UserInferenceConfigError("缺少 Stage2 checkpoint_path。")

    cfg = UserStage2Config(
        checkpoint_path=checkpoint_path,
        acoustic_generation_mode=str(stage2.get("acoustic_generation_mode", "standard")),
        internal_sampling_mode=str(stage2.get("internal_sampling_mode", "coarse_only")),
        target_length_mode=str(stage2.get("target_length_mode", "semantic_ratio")),
        length_scale=_float_value(
            length_scale if length_scale is not None else stage2.get("length_scale", 1.0),
            name="stage2.length_scale",
        ),
        attach_continuous_semantic=_bool_value(
            stage2.get("attach_continuous_semantic", True),
            name="stage2.attach_continuous_semantic",
        ),
        require_continuous_semantic=_bool_value(
            stage2.get("require_continuous_semantic", True),
            name="stage2.require_continuous_semantic",
        ),
        continuous_dtype=str(stage2.get("continuous_dtype", "float32")),
    )
    validate_stage2_config(cfg)
    return cfg


def build_vocoder_config(
    defaults: dict[str, Any],
    *,
    model_paths: Optional[ResolvedModelPaths] = None,
) -> UserVocoderConfig:
    """Build UserVocoderConfig from defaults and resolved profile paths."""

    vocoder = _section(defaults, "vocoder")

    cfg = UserVocoderConfig(
        vocoder_type=(model_paths.vocoder_type if model_paths else str(vocoder.get("type", "hifigan"))),
        vocoder_profile=(
            model_paths.vocoder_profile if model_paths else str(vocoder.get("profile", "universal_v1"))
        ),
        hifigan_root=(
            model_paths.hifigan_root
            if model_paths and model_paths.hifigan_root is not None
            else _resolve_optional_project_path(vocoder.get("hifigan_root"))
        ),
        checkpoint_path=_resolve_optional_project_path(vocoder.get("checkpoint_path")),
        config_path=_resolve_optional_project_path(vocoder.get("config_path")),
    )
    validate_vocoder_config(cfg)
    return cfg


def build_metrics_config(
    defaults: dict[str, Any],
    *,
    enable_metrics: Optional[bool] = None,
    compute_dnsmos: Optional[bool] = None,
    compute_speaker_sim: Optional[bool] = None,
    compute_wer: Optional[bool] = None,
) -> UserMetricsConfig:
    """Build UserMetricsConfig from defaults and user overrides.

    中文说明：
        从默认 YAML 和用户覆盖项中构建用户版 metrics 配置。

        P9-3-fix:
            读取 metrics.dnsmos_onnx_path。

        P9-3-wer-fix:
            读取 ASR/WER 相关配置：
                - asr_backend
                - asr_model
                - asr_device
                - asr_compute_type

            用户版默认让 WER/CER 使用 CPU int8 ASR，避免 Windows runtime
            中因 CUDA 12 DLL 缺失导致：

                cublas64_12.dll is not found or cannot be loaded
    """

    metrics = _section(defaults, "metrics")

    cfg = UserMetricsConfig(
        enable_metrics=(
            bool(enable_metrics)
            if enable_metrics is not None
            else _bool_value(metrics.get("enable_metrics", False), name="metrics.enable_metrics")
        ),
        compute_dnsmos=(
            bool(compute_dnsmos)
            if compute_dnsmos is not None
            else _bool_value(metrics.get("compute_dnsmos", True), name="metrics.compute_dnsmos")
        ),
        compute_speaker_sim=(
            bool(compute_speaker_sim)
            if compute_speaker_sim is not None
            else _bool_value(metrics.get("compute_speaker_sim", True), name="metrics.compute_speaker_sim")
        ),
        compute_wer=(
            bool(compute_wer)
            if compute_wer is not None
            else _bool_value(metrics.get("compute_wer", False), name="metrics.compute_wer")
        ),

        dnsmos_onnx_path=_resolve_optional_project_path(metrics.get("dnsmos_onnx_path")),

        speaker_device=str(metrics.get("speaker_device", "cuda")),

        asr_backend=str(metrics.get("asr_backend", "faster_whisper")),
        asr_model=str(metrics.get("asr_model", "medium")),
        asr_device=str(metrics.get("asr_device", "cpu")),
        asr_compute_type=str(metrics.get("asr_compute_type", "int8")),
    )

    validate_metrics_config(cfg)
    return cfg


def build_output_config(
    defaults: dict[str, Any],
    *,
    output_dir: Optional[PathLike] = None,
) -> UserOutputConfig:
    """Build UserOutputConfig from defaults and user overrides."""

    output = _section(defaults, "output")

    cfg = UserOutputConfig(
        output_root=resolve_project_path(output.get("root", "outputs/inference_runs"))
        or Path("outputs/inference_runs"),
        output_dir=_resolve_optional_project_path(output_dir),
        save_peaknorm=_bool_value(output.get("save_peaknorm", True), name="output.save_peaknorm"),
        copy_prompt_wav=_bool_value(output.get("copy_prompt_wav", True), name="output.copy_prompt_wav"),
        save_debug_tensors=_bool_value(
            output.get("save_debug_tensors", False),
            name="output.save_debug_tensors",
        ),
    )
    validate_output_config(cfg)
    return cfg


def build_runtime_config(
    defaults: dict[str, Any],
    *,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> UserRuntimeConfig:
    """Build UserRuntimeConfig from defaults and user overrides."""

    runtime = _section(defaults, "runtime")

    raw_seed = seed if seed is not None else runtime.get("seed")
    resolved_seed = None if raw_seed is None else _int_value(raw_seed, name="runtime.seed")

    cfg = UserRuntimeConfig(
        device=str(device if device is not None else runtime.get("device", "cuda")),
        fallback_to_cpu=_bool_value(
            runtime.get("fallback_to_cpu", True),
            name="runtime.fallback_to_cpu",
        ),
        seed=resolved_seed,
    )
    validate_runtime_config(cfg)
    return cfg


# -----------------------------------------------------------------------------
# Top-level builder
# -----------------------------------------------------------------------------


def build_user_inference_config(
    *,
    prompt_wav_path: PathLike,
    prompt_text: str,
    target_text: str,
    profile_dir: Optional[PathLike] = None,
    stage1_ckpt: Optional[PathLike] = None,
    stage2_ckpt: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    length_scale: Optional[float] = None,
    enable_metrics: Optional[bool] = None,
    compute_dnsmos: Optional[bool] = None,
    compute_speaker_sim: Optional[bool] = None,
    compute_wer: Optional[bool] = None,
    default_config_path: PathLike = "configs/user_inference_default.yaml",
) -> UserInferenceConfig:
    """Build the full user-edition inference config.

    This is the main entry point for both future CLI and Gradio UI layers.
    """

    defaults = load_default_user_config(default_config_path)

    profile_defaults = _section(defaults, "profile")
    resolved_profile_dir = profile_dir or profile_defaults.get("default_profile_dir")
    if normalize_path_text(resolved_profile_dir) is None:
        raise UserInferenceConfigError("缺少 profile_dir，且默认配置中未设置 default_profile_dir。")

    profile: VoiceProfile = load_voice_profile(resolved_profile_dir)
    model_paths = resolve_model_paths(
        profile,
        user_stage1_ckpt=stage1_ckpt,
        user_stage2_ckpt=stage2_ckpt,
        default_config=defaults,
    )

    resolved_prompt_wav = resolve_project_path(prompt_wav_path)
    if resolved_prompt_wav is None:
        raise UserInferenceConfigError("缺少参考音频 prompt_wav_path。")

    cfg = UserInferenceConfig(
        profile_dir=profile.profile_dir,
        profile_name=profile.profile_name,
        profile_display_name=profile.display_name,
        inference_mode=detect_inference_mode(model_paths),
        prompt_wav_path=resolved_prompt_wav,
        prompt_text=prompt_text,
        target_text=target_text,
        stage1=build_stage1_config(
            defaults,
            checkpoint_override=model_paths.stage1_ckpt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
        stage2=build_stage2_config(
            defaults,
            checkpoint_override=model_paths.stage2_ckpt,
            length_scale=length_scale,
        ),
        vocoder=build_vocoder_config(defaults, model_paths=model_paths),
        metrics=build_metrics_config(
            defaults,
            enable_metrics=enable_metrics,
            compute_dnsmos=compute_dnsmos,
            compute_speaker_sim=compute_speaker_sim,
            compute_wer=compute_wer,
        ),
        output=build_output_config(defaults, output_dir=output_dir),
        runtime=build_runtime_config(defaults, device=device, seed=seed),
        language="zh",
        stage1_source=model_paths.stage1_source,
        stage2_source=model_paths.stage2_source,
    )

    validate_user_inference_config(cfg)
    return cfg


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


def validate_stage1_config(cfg: UserStage1Config) -> None:
    if cfg.checkpoint_path is not None:
        ensure_file_exists(cfg.checkpoint_path, label="Stage1 文本转语义模型", required=True)
    if cfg.temperature <= 0:
        raise UserInferenceConfigError("temperature 必须大于 0。")
    if not (0 < cfg.top_p <= 1):
        raise UserInferenceConfigError("top_p 必须满足 0 < top_p <= 1。")
    if cfg.top_k < 1:
        raise UserInferenceConfigError("top_k 必须大于等于 1。")
    if cfg.repetition_penalty <= 0:
        raise UserInferenceConfigError("repetition_penalty 必须大于 0。")
    if cfg.early_stop_num != -1 and cfg.early_stop_num <= 0:
        raise UserInferenceConfigError("early_stop_num 必须为 -1 或正整数。")
    if cfg.text_mode != "prompt_plus_target":
        raise UserInferenceConfigError("用户版 v1.0 固定 text_mode=prompt_plus_target。")
    if cfg.crop_mode not in {"auto_idx", "prompt_len", "none"}:
        raise UserInferenceConfigError("crop_mode 必须是 auto_idx、prompt_len 或 none。")
    if cfg.append_tail_mute_tokens < 0:
        raise UserInferenceConfigError("append_tail_mute_tokens 不能为负数。")


def validate_stage2_config(cfg: UserStage2Config) -> None:
    ensure_file_exists(cfg.checkpoint_path, label="Stage2 声学生成模型", required=True)
    if cfg.acoustic_generation_mode != "standard":
        raise UserInferenceConfigError("用户版 v1.0 固定 acoustic_generation_mode=standard。")
    if cfg.internal_sampling_mode != "coarse_only":
        raise UserInferenceConfigError("用户版 v1.0 固定 internal_sampling_mode=coarse_only。")
    if cfg.target_length_mode != "semantic_ratio":
        raise UserInferenceConfigError("用户版 v1.0 固定 target_length_mode=semantic_ratio。")
    if cfg.length_scale <= 0:
        raise UserInferenceConfigError("length_scale 必须大于 0。")
    if not cfg.attach_continuous_semantic:
        raise UserInferenceConfigError("用户版 v1.0 要求 attach_continuous_semantic=true。")
    if not cfg.require_continuous_semantic:
        raise UserInferenceConfigError("用户版 v1.0 要求 require_continuous_semantic=true。")
    if cfg.continuous_dtype not in {"float16", "float32"}:
        raise UserInferenceConfigError("continuous_dtype 必须是 float16 或 float32。")


def validate_vocoder_config(cfg: UserVocoderConfig) -> None:
    if cfg.vocoder_type != "hifigan":
        raise UserInferenceConfigError("用户版 v1.0 仅支持 hifigan vocoder。")
    if not cfg.vocoder_profile:
        raise UserInferenceConfigError("vocoder_profile 不能为空。")
    # Existence of hifigan_root is checked in P4 env_check to keep config loading lightweight.


def validate_metrics_config(cfg: UserMetricsConfig) -> None:
    if not cfg.enable_metrics:
        return

    if not (cfg.compute_dnsmos or cfg.compute_speaker_sim or cfg.compute_wer):
        raise UserInferenceConfigError("enable_metrics=true 时至少需要启用一个指标。")

    if cfg.compute_wer:
        if not cfg.asr_backend:
            raise UserInferenceConfigError("compute_wer=true 时 asr_backend 不能为空。")
        if cfg.asr_backend not in {"faster_whisper", "whisper", "none"}:
            raise UserInferenceConfigError(
                "asr_backend 必须是 faster_whisper、whisper 或 none。"
            )
        if not cfg.asr_model:
            raise UserInferenceConfigError("compute_wer=true 时 asr_model 不能为空。")
        if cfg.asr_device not in {"cuda", "cpu"}:
            raise UserInferenceConfigError("asr_device 必须是 cuda 或 cpu。")
        if not cfg.asr_compute_type:
            raise UserInferenceConfigError("compute_wer=true 时 asr_compute_type 不能为空。")


def validate_output_config(cfg: UserOutputConfig) -> None:
    if cfg.save_debug_tensors:
        raise UserInferenceConfigError("用户版 v1.0 默认不允许 save_debug_tensors=true。")


def validate_runtime_config(cfg: UserRuntimeConfig) -> None:
    if cfg.device not in {"cuda", "cpu"}:
        raise UserInferenceConfigError("device 必须是 cuda 或 cpu。")
    if cfg.seed is not None and cfg.seed < 0:
        raise UserInferenceConfigError("seed 必须为空或非负整数。")


def validate_user_inference_config(cfg: UserInferenceConfig) -> None:
    if cfg.language != "zh":
        raise UserInferenceConfigError("VoiceLab 用户版 v1.0 仅支持中文 language=zh。")
    ensure_file_exists(cfg.prompt_wav_path, label="参考音频 prompt_wav_path", required=True)
    if not cfg.prompt_text or not cfg.prompt_text.strip():
        raise UserInferenceConfigError("prompt_text 不能为空。")
    if not cfg.target_text or not cfg.target_text.strip():
        raise UserInferenceConfigError("target_text 不能为空。")
    validate_stage1_config(cfg.stage1)
    validate_stage2_config(cfg.stage2)
    validate_vocoder_config(cfg.vocoder)
    validate_metrics_config(cfg.metrics)
    validate_output_config(cfg.output)
    validate_runtime_config(cfg.runtime)


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


def config_to_user_dict(cfg: UserInferenceConfig) -> dict[str, Any]:
    """Convert config into a JSON-serializable request-style dictionary."""

    return {
        "schema_version": "user_inference_request_v1",
        "profile": {
            "profile_dir": _path_to_str(cfg.profile_dir),
            "profile_name": cfg.profile_name,
            "display_name": cfg.profile_display_name,
            "inference_mode": cfg.inference_mode,
        },
        "input": {
            "prompt_wav_path": _path_to_str(cfg.prompt_wav_path),
            "prompt_text": cfg.prompt_text,
            "target_text": cfg.target_text,
            "language": cfg.language,
        },
        "model": {
            "stage1_source": cfg.stage1_source,
            "stage1_ckpt": _path_to_str(cfg.stage1.checkpoint_path),
            "stage2_source": cfg.stage2_source,
            "stage2_ckpt": _path_to_str(cfg.stage2.checkpoint_path),
            "vocoder_type": cfg.vocoder.vocoder_type,
            "vocoder_profile": cfg.vocoder.vocoder_profile,
        },
        "sampling": {
            "temperature": cfg.stage1.temperature,
            "top_p": cfg.stage1.top_p,
            "top_k": cfg.stage1.top_k,
            "length_scale": cfg.stage2.length_scale,
            "seed": cfg.runtime.seed,
        },
        "metrics": {
            "enable_metrics": cfg.metrics.enable_metrics,
            "compute_dnsmos": cfg.metrics.compute_dnsmos,
            "compute_speaker_sim": cfg.metrics.compute_speaker_sim,
            "compute_wer": cfg.metrics.compute_wer,
        },
        "output": {
            "output_root": _path_to_str(cfg.output.output_root),
            "output_dir": _path_to_str(cfg.output.output_dir),
            "save_peaknorm": cfg.output.save_peaknorm,
            "copy_prompt_wav": cfg.output.copy_prompt_wav,
            "save_debug_tensors": cfg.output.save_debug_tensors,
        },
        "runtime": {
            "device": cfg.runtime.device,
            "fallback_to_cpu": cfg.runtime.fallback_to_cpu,
        },
    }


def config_to_internal_summary_dict(cfg: UserInferenceConfig) -> dict[str, Any]:
    """Convert config into an internal summary dictionary for debugging logs."""

    return {
        "schema_version": "user_inference_internal_summary_v1",
        "profile_name": cfg.profile_name,
        "inference_mode": cfg.inference_mode,
        "language": cfg.language,
        "stage1": {
            "source": cfg.stage1_source,
            "checkpoint_path": _path_to_str(cfg.stage1.checkpoint_path),
            "text_mode": cfg.stage1.text_mode,
            "crop_prompt_prefix": cfg.stage1.crop_prompt_prefix,
            "crop_mode": cfg.stage1.crop_mode,
            "append_tail_mute_tokens": cfg.stage1.append_tail_mute_tokens,
            "mute_token_id": cfg.stage1.mute_token_id,
            "repetition_penalty": cfg.stage1.repetition_penalty,
            "early_stop_num": cfg.stage1.early_stop_num,
        },
        "stage2": {
            "source": cfg.stage2_source,
            "checkpoint_path": _path_to_str(cfg.stage2.checkpoint_path),
            "acoustic_generation_mode": cfg.stage2.acoustic_generation_mode,
            "internal_sampling_mode": cfg.stage2.internal_sampling_mode,
            "target_length_mode": cfg.stage2.target_length_mode,
            "attach_continuous_semantic": cfg.stage2.attach_continuous_semantic,
            "require_continuous_semantic": cfg.stage2.require_continuous_semantic,
            "continuous_dtype": cfg.stage2.continuous_dtype,
        },
        "vocoder": {
            "type": cfg.vocoder.vocoder_type,
            "profile": cfg.vocoder.vocoder_profile,
            "hifigan_root": _path_to_str(cfg.vocoder.hifigan_root),
            "checkpoint_path": _path_to_str(cfg.vocoder.checkpoint_path),
            "config_path": _path_to_str(cfg.vocoder.config_path),
        },
    }


def profile_and_paths_to_summary(
    profile: VoiceProfile,
    model_paths: ResolvedModelPaths,
) -> dict[str, Any]:
    """Expose profile_loader summary from the config layer for convenience."""

    return profile_to_summary_dict(profile, model_paths)


__all__ = [
    "UserInferenceConfigError",
    "UserStage1Config",
    "UserStage2Config",
    "UserVocoderConfig",
    "UserMetricsConfig",
    "UserOutputConfig",
    "UserRuntimeConfig",
    "UserInferenceConfig",
    "load_yaml",
    "load_default_user_config",
    "build_stage1_config",
    "build_stage2_config",
    "build_vocoder_config",
    "build_metrics_config",
    "build_output_config",
    "build_runtime_config",
    "build_user_inference_config",
    "validate_stage1_config",
    "validate_stage2_config",
    "validate_vocoder_config",
    "validate_metrics_config",
    "validate_output_config",
    "validate_runtime_config",
    "validate_user_inference_config",
    "config_to_user_dict",
    "config_to_internal_summary_dict",
    "profile_and_paths_to_summary",
]
