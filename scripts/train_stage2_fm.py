from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
from torch.utils.data import DataLoader

# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.datasets.stage2_fm_dataset import (
    Stage2FMDataset,
    create_stage2_fm_collate_fn,
    move_stage2_inputs_to_device,
    stage2_fm_collate_fn,
)
from src.models.stage2_acoustic_model import (
    Stage2FlowMatchingAcousticModel,
    Stage2FlowMatchingAcousticModelV2,
    Stage2FlowMatchingAcousticModelV5,
    Stage2FlowMatchingAcousticModelV6,
    Stage2FlowMatchingAcousticModelV61,
    Stage2FlowMatchingAcousticModelV62,
    Stage2FlowMatchingAcousticModelV63,
)


DEFAULT_ACOUSTIC_RATE_HZ = 22050.0 / 256.0  # 86.1328125


# ============================================================
# Helper functions
# 辅助函数
# ============================================================
def str2bool(x: str | bool) -> bool:
    """
    EN:
    Convert common string forms to bool.

    ZH:
    将常见字符串形式转换成布尔值。
    """
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {x}")


def parse_optional_bool(x: str | bool | None) -> bool | None:
    """
    EN:
    Parse optional bool string. None stays None.

    ZH:
    解析可选布尔字符串；如果为 None，则保持 None。
    """
    if x is None:
        return None
    return str2bool(x)


def parse_int_tuple(x: str | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    """
    EN:
    Parse comma-separated int tuple.

    Examples:
        "1,2" -> (1, 2)
        [1, 2] -> (1, 2)

    ZH:
    解析命令行或 checkpoint 中的整数 tuple。
    """
    if x is None:
        return (1, 2)

    if isinstance(x, tuple):
        values = tuple(int(v) for v in x)
    elif isinstance(x, list):
        values = tuple(int(v) for v in x)
    else:
        parts = str(x).split(",")
        values = tuple(int(p.strip()) for p in parts if p.strip())

    if len(values) <= 0:
        raise ValueError(f"Cannot parse int tuple from: {x}")

    return values


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """
    EN:
    Safe wrapper around torch.load with explicit weights_only=False.

    ZH:
    对 torch.load 的包装，显式指定 weights_only=False。
    """
    return torch.load(str(path), map_location=map_location, weights_only=False)


def save_json(obj: dict[str, Any], path: Path) -> None:
    """
    EN:
    Save JSON file with UTF-8 encoding.

    ZH:
    使用 UTF-8 编码保存 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """
    EN:
    Load JSON file if it exists, otherwise return None.

    ZH:
    如果 JSON 文件存在则读取，否则返回 None。
    """
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_checkpoint_args(x: Any) -> dict[str, Any]:
    """
    EN:
    Normalize checkpoint 'args' to a plain dict.

    ZH:
    将 checkpoint 中的 'args' 统一规范成普通 dict。
    """
    if x is None:
        return {}

    if isinstance(x, dict):
        return dict(x)

    try:
        return vars(x)
    except Exception:
        return {}


def _get_bool_from_dict(d: dict[str, Any], key: str, default: bool) -> bool:
    """
    EN:
    Robust bool extraction from dict.

    ZH:
    从 dict 中稳健读取 bool。
    """
    if key not in d:
        return bool(default)
    return str2bool(d[key])


def extract_model_config_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    """
    EN:
    Extract stage-2 model structural config from a dict.

    This function is intentionally defensive so older checkpoints can still
    be inspected or partially resumed.

    ZH:
    从 dict 中提取第二阶段模型结构配置。

    该函数故意写得比较防御性，以便旧 checkpoint 也能被检查或部分恢复。
    """
    model_version = str(d.get("model_version", "v1")).lower()

    return {
        # ----------------------------------------------------
        # Basic model args / 基础模型参数
        # ----------------------------------------------------
        "model_version": model_version,
        "acoustic_dim": int(d.get("acoustic_dim", 80)),
        "hidden_dim": int(d.get("hidden_dim", 512)),
        "num_layers": int(d.get("num_layers", 6)),
        "num_heads": int(d.get("num_heads", 8)),
        "semantic_vocab_size": int(d.get("semantic_vocab_size", 1024)),
        "semantic_rate_hz": float(d.get("semantic_rate_hz", 25.0)),
        "acoustic_rate_hz": float(d.get("acoustic_rate_hz", DEFAULT_ACOUSTIC_RATE_HZ)),
        "dropout": float(d.get("dropout", 0.1)),

        # ----------------------------------------------------
        # Self condition / 自条件
        # ----------------------------------------------------
        "use_self_condition": bool(d.get("use_self_condition", True)),
        "self_condition_prob": float(d.get("self_condition_prob", 0.5)),

        # ----------------------------------------------------
        # v4 old length predictor compatibility
        # v4 旧长度预测器兼容字段
        # ----------------------------------------------------
        "enable_length_predictor": bool(d.get("enable_length_predictor", False)),
        "length_predictor_hidden_dim": int(d.get("length_predictor_hidden_dim", 256)),
        "length_loss_weight": float(d.get("length_loss_weight", 0.10)),
        "length_teacher_forcing_prob": float(d.get("length_teacher_forcing_prob", 0.70)),

        # ----------------------------------------------------
        # v5 / v6 / v6.1 condition args
        # ----------------------------------------------------
        "phoneme_vocab_size": int(d.get("phoneme_vocab_size", 4096)),
        "phoneme_embed_dim": int(d.get("phoneme_embed_dim", 256)),
        "bert_dim": int(d.get("bert_dim", 1024)),
        "content_dim": int(d.get("content_dim", 384)),
        "content_refiner_layers": int(d.get("content_refiner_layers", 4)),
        "content_frame_dim": int(d.get("content_frame_dim", 384)),
        "content_expander_layers": int(d.get("content_expander_layers", 2)),
        "style_dim": int(d.get("style_dim", 256)),

        # ----------------------------------------------------
        # v6.1 condition args
        # ----------------------------------------------------
        "content_main_dim": int(d.get("content_main_dim", d.get("hidden_dim", 512))),
        "content_injection_gate_init": float(d.get("content_injection_gate_init", 0.5)),
        "content_refiner_dilations": parse_int_tuple(d.get("content_refiner_dilations", "1,2")),
        "content_boundary_enhance": bool(d.get("content_boundary_enhance", True)),
        "content_boundary_kernel_size": int(d.get("content_boundary_kernel_size", 3)),
        "content_boundary_residual_scale": float(d.get("content_boundary_residual_scale", 0.3)),
        "content_refiner_residual_scale": float(d.get("content_refiner_residual_scale", 0.5)),

        # ----------------------------------------------------
        # v6.1 block args
        # ----------------------------------------------------
        "ff_mult": int(d.get("ff_mult", 4)),
        "local_detail_kernel_size": int(d.get("local_detail_kernel_size", 5)),
        "local_detail_dilation_cycle": parse_int_tuple(d.get("local_detail_dilation_cycle", "1,2")),
        "local_detail_residual_scale": float(d.get("local_detail_residual_scale", 0.5)),
        "semantic_guide_gate_init": float(d.get("semantic_guide_gate_init", 0.3)),
        "use_semantic_guide_cross_attn": bool(d.get("use_semantic_guide_cross_attn", True)),

        # ----------------------------------------------------
        # Loss weights / 损失权重
        # ----------------------------------------------------
        "fm_loss_weight": float(d.get("fm_loss_weight", 1.0)),
        "recon_loss_weight": float(d.get("recon_loss_weight", 0.20)),
        "refined_recon_loss_weight": float(d.get("refined_recon_loss_weight", 1.00)),
        "delta_loss_weight": float(d.get("delta_loss_weight", 0.15)),
        "delta2_loss_weight": float(d.get("delta2_loss_weight", 0.05)),
        "mid_content_aux_weight": float(d.get("mid_content_aux_weight", 0.10)),
        "final_content_aux_weight": float(d.get("final_content_aux_weight", 0.20)),
        "span_focus_aux_weight": float(d.get("span_focus_aux_weight", 0.20)),
        "span_focus_ratio_min": float(d.get("span_focus_ratio_min", 0.10)),
        "span_focus_ratio_max": float(d.get("span_focus_ratio_max", 0.35)),
        "span_recon_weight": float(d.get("span_recon_weight", 0.50)),
        "span_content_weight": float(d.get("span_content_weight", 0.50)),
        "recon_loss_type": str(d.get("recon_loss_type", "l1")),

        # ----------------------------------------------------
        # Conditional dropout / 条件 dropout
        # ----------------------------------------------------
        "cond_drop_prob_content": float(d.get("cond_drop_prob_content", 0.05)),
        "cond_drop_prob_semantic": float(d.get("cond_drop_prob_semantic", 0.10)),
        "cond_drop_prob_style": float(d.get("cond_drop_prob_style", 0.10)),
        "cond_drop_all_prob": float(d.get("cond_drop_all_prob", 0.05)),

        # ----------------------------------------------------
        # Postnet args / postnet 参数
        # ----------------------------------------------------
        "postnet_num_layers": int(d.get("postnet_num_layers", 3)),
        "postnet_dropout": float(d.get("postnet_dropout", 0.1)),
        "postnet_conv_kernel_size": int(d.get("postnet_conv_kernel_size", 5)),
        "detach_coarse_for_refinement": bool(d.get("detach_coarse_for_refinement", True)),

        # ----------------------------------------------------
        # Training dynamics / 训练动力学
        # ----------------------------------------------------
        "t_sampling_mode": str(d.get("t_sampling_mode", "near_clean")),
        "t_bias_power": float(d.get("t_bias_power", 2.0)),
        "t_min": float(d.get("t_min", 0.05)),
        "t_max": float(d.get("t_max", 0.98)),

        # ----------------------------------------------------
        # Frequency weighting / 频带加权
        # ----------------------------------------------------
        "freq_weight_min": float(d.get("freq_weight_min", 1.0)),
        "freq_weight_max": float(d.get("freq_weight_max", 2.0)),
        "freq_weight_power": float(d.get("freq_weight_power", 1.5)),

        # ----------------------------------------------------
        # Content aux / 内容辅助
        # ----------------------------------------------------
        "content_head_dropout": float(d.get("content_head_dropout", 0.1)),

        # ----------------------------------------------------
        # Length correction / 长度修正
        # ----------------------------------------------------
        "enable_length_clamp": bool(d.get("enable_length_clamp", True)),
        "length_correction_mode": str(d.get("length_correction_mode", "gentle_ratio")),
        "length_ratio_min": float(d.get("length_ratio_min", 2.6)),
        "length_ratio_max": float(d.get("length_ratio_max", 3.8)),
        "length_bias_scale": float(d.get("length_bias_scale", 0.95)),

        # ----------------------------------------------------
        # v6.2 cold-start bootstrapper args
        # v6.2 冷启动 bootstrapper 参数
        # ----------------------------------------------------
        "use_coarse_bootstrapper": bool(d.get("use_coarse_bootstrapper", True)),
        "bootstrap_hidden_dim": (
            None
            if d.get("bootstrap_hidden_dim", None) is None
            else int(d.get("bootstrap_hidden_dim"))
        ),
        "bootstrap_num_blocks": int(d.get("bootstrap_num_blocks", 4)),
        "bootstrap_kernel_size": int(d.get("bootstrap_kernel_size", 5)),
        "bootstrap_dropout": float(d.get("bootstrap_dropout", 0.1)),
        "bootstrap_expansion_factor": int(d.get("bootstrap_expansion_factor", 4)),

        "bootstrap_mel_bias_init": float(d.get("bootstrap_mel_bias_init", -5.0)),
        "bootstrap_residual_scale_init": float(d.get("bootstrap_residual_scale_init", 0.1)),
        "bootstrap_semantic_gate_init": float(d.get("bootstrap_semantic_gate_init", 0.5)),
        "bootstrap_style_residual_scale_init": float(d.get("bootstrap_style_residual_scale_init", 0.1)),
        "bootstrap_clamp_output": bool(d.get("bootstrap_clamp_output", False)),
        "bootstrap_output_min": float(d.get("bootstrap_output_min", -12.0)),
        "bootstrap_output_max": float(d.get("bootstrap_output_max", 4.0)),

        "coarse_mel_loss_weight": float(d.get("coarse_mel_loss_weight", 1.0)),
        "coarse_mel_mse_loss_weight": float(d.get("coarse_mel_mse_loss_weight", 0.0)),
        "coarse_delta_loss_weight": float(d.get("coarse_delta_loss_weight", 0.2)),

        "use_bridge_loss": bool(d.get("use_bridge_loss", True)),
        "bridge_t_start": float(d.get("bridge_t_start", 0.075)),
        "bridge_noise_temperature": float(d.get("bridge_noise_temperature", 0.3)),
        "bridge_recon_loss_weight": float(d.get("bridge_recon_loss_weight", 1.0)),
        "bridge_delta_loss_weight": float(d.get("bridge_delta_loss_weight", 0.2)),
        "detach_coarse_for_bridge": bool(d.get("detach_coarse_for_bridge", True)),

        # ----------------------------------------------------
        # v6.3 Conformer bootstrapper args
        # ----------------------------------------------------
        "use_conformer_bootstrapper": bool(d.get("use_conformer_bootstrapper", True)),
        "v63_bootstrap_hidden_dim": (
            None
            if d.get("v63_bootstrap_hidden_dim", None) is None
            else int(d.get("v63_bootstrap_hidden_dim"))
        ),
        "v63_bootstrap_num_layers": int(d.get("v63_bootstrap_num_layers", 6)),
        "v63_bootstrap_num_heads": int(d.get("v63_bootstrap_num_heads", 8)),
        "v63_bootstrap_ff_mult": int(d.get("v63_bootstrap_ff_mult", 4)),
        "v63_bootstrap_conv_kernel_size": int(
            d.get("v63_bootstrap_conv_kernel_size", 15)
        ),
        "v63_bootstrap_dropout": float(d.get("v63_bootstrap_dropout", 0.1)),
        "v63_bootstrap_mel_bias_init": float(
            d.get("v63_bootstrap_mel_bias_init", -5.0)
        ),
        "v63_bootstrap_zero_init_output": bool(
            d.get("v63_bootstrap_zero_init_output", False)
        ),
        "v63_bootstrap_clamp_output": bool(
            d.get("v63_bootstrap_clamp_output", False)
        ),

        # ----------------------------------------------------
        # v6.3.3 native text memory args
        # ----------------------------------------------------
        "v63_use_native_text_memory": bool(
            d.get("v63_use_native_text_memory", False)
        ),
        "v63_text_memory_dim": (
            None
            if d.get("v63_text_memory_dim", None) is None
            else int(d.get("v63_text_memory_dim"))
        ),

        # ----------------------------------------------------
        # v6.3 residual refiner args
        # ----------------------------------------------------
        "use_residual_refiner": bool(d.get("use_residual_refiner", True)),
        "residual_hidden_dim": (
            None
            if d.get("residual_hidden_dim", None) is None
            else int(d.get("residual_hidden_dim"))
        ),
        "residual_num_layers": int(d.get("residual_num_layers", 6)),
        "residual_num_heads": int(d.get("residual_num_heads", 8)),
        "residual_ff_mult": int(d.get("residual_ff_mult", 4)),
        "residual_conv_kernel_size": int(d.get("residual_conv_kernel_size", 15)),
        "residual_dropout": float(d.get("residual_dropout", 0.1)),
        "residual_noise_scale": float(d.get("residual_noise_scale", 0.5)),
        "residual_sample_temperature": float(
            d.get("residual_sample_temperature", 0.3)
        ),
        "residual_t_min": float(d.get("residual_t_min", 0.0)),
        "residual_t_max": float(d.get("residual_t_max", 1.0)),
        "detach_coarse_for_residual_refiner": bool(
            d.get("detach_coarse_for_residual_refiner", False)
        ),

        # ----------------------------------------------------
        # v6.4.1 continuous semantic
        # ----------------------------------------------------
        "use_continuous_semantic": _get_bool_from_dict(
            d,
            "use_continuous_semantic",
            False,
        ),
        "continuous_semantic_dim": int(d.get("continuous_semantic_dim", 768)),
        "continuous_semantic_fusion_mode": str(
            d.get("continuous_semantic_fusion_mode", "gated_add")
        ),
        "continuous_semantic_gate_init": float(
            d.get("continuous_semantic_gate_init", -2.0)
        ),
        "continuous_semantic_dropout": float(
            d.get("continuous_semantic_dropout", 0.0)
        ),

        # ----------------------------------------------------
        # v6.3 loss args
        # ----------------------------------------------------
        # ----------------------------------------------------
        # v6.6.4 Speaker Identity Adapter
        # ----------------------------------------------------
        "use_v664_speaker_identity_adapter": bool(d.get("use_v664_speaker_identity_adapter", False)),
        "v664_speaker_identity_source": str(d.get("v664_speaker_identity_source", "prompt_or_target")),
        "v664_speaker_embedding_dim": int(d.get("v664_speaker_embedding_dim", d.get("v66_speaker_embedding_dim", 192))),
        "v664_speaker_adapter_hidden_dim": int(d.get("v664_speaker_adapter_hidden_dim", 256)),
        "v664_speaker_adapter_num_layers": int(d.get("v664_speaker_adapter_num_layers", 2)),
        "v664_speaker_adapter_dropout": float(d.get("v664_speaker_adapter_dropout", 0.05)),
        "v664_speaker_adapter_fusion_mode": str(d.get("v664_speaker_adapter_fusion_mode", "gated_add")),
        "v664_speaker_adapter_gate_init": float(d.get("v664_speaker_adapter_gate_init", -3.0)),
        "v664_speaker_adapter_normalize": bool(d.get("v664_speaker_adapter_normalize", True)),

        # ----------------------------------------------------
        # v6.6.3 Bootstrapper LoRA
        # ----------------------------------------------------
        "use_bootstrapper_lora": bool(d.get("use_bootstrapper_lora", False)),
        "bootstrapper_lora_rank": int(d.get("bootstrapper_lora_rank", 4)),
        "bootstrapper_lora_alpha": float(d.get("bootstrapper_lora_alpha", 8.0)),
        "bootstrapper_lora_dropout": float(d.get("bootstrapper_lora_dropout", 0.05)),
        "bootstrapper_lora_target": str(d.get("bootstrapper_lora_target", "core")),
        "bootstrapper_lora_init_scale": float(d.get("bootstrapper_lora_init_scale", 0.01)),

        # ----------------------------------------------------
        # v6.6.3-A Bootstrapper Attention LoRA
        # ----------------------------------------------------
        "use_bootstrapper_attention_lora": bool(d.get("use_bootstrapper_attention_lora", False)),
        "bootstrapper_attention_lora_rank": int(d.get("bootstrapper_attention_lora_rank", 4)),
        "bootstrapper_attention_lora_alpha": float(d.get("bootstrapper_attention_lora_alpha", 8.0)),
        "bootstrapper_attention_lora_dropout": float(d.get("bootstrapper_attention_lora_dropout", 0.05)),
        "bootstrapper_attention_lora_target": str(d.get("bootstrapper_attention_lora_target", "cross_only")),
        "bootstrapper_attention_lora_gate_init": float(d.get("bootstrapper_attention_lora_gate_init", -3.0)),
        "bootstrapper_attention_lora_enable_q": bool(d.get("bootstrapper_attention_lora_enable_q", True)),
        "bootstrapper_attention_lora_enable_k": bool(d.get("bootstrapper_attention_lora_enable_k", True)),
        "bootstrapper_attention_lora_enable_v": bool(d.get("bootstrapper_attention_lora_enable_v", True)),
        "bootstrapper_attention_lora_enable_o": bool(d.get("bootstrapper_attention_lora_enable_o", True)),

        "include_v61_base_loss": bool(d.get("include_v61_base_loss", False)),
        "v61_base_loss_weight": float(d.get("v61_base_loss_weight", 1.0)),

        "v63_coarse_l1_weight": float(d.get("v63_coarse_l1_weight", 2.0)),
        "v63_coarse_mse_weight": float(d.get("v63_coarse_mse_weight", 0.2)),
        "v63_coarse_delta_weight": float(d.get("v63_coarse_delta_weight", 0.5)),
        "v63_coarse_delta2_weight": float(d.get("v63_coarse_delta2_weight", 0.2)),

        "residual_refiner_loss_weight": float(
            d.get("residual_refiner_loss_weight", 1.0)
        ),
        "residual_flow_mse_weight": float(d.get("residual_flow_mse_weight", 1.0)),
        "residual_flow_l1_weight": float(d.get("residual_flow_l1_weight", 0.0)),
        "residual_recon_l1_weight": float(
            d.get("residual_recon_l1_weight", 1.0)
        ),
        "final_recon_l1_weight": float(d.get("final_recon_l1_weight", 1.0)),
        "final_delta_l1_weight": float(d.get("final_delta_l1_weight", 0.3)),
        "final_delta2_l1_weight": float(d.get("final_delta2_l1_weight", 0.1)),
    }






def build_model_config_from_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    """
    EN:
    Build model config from current CLI args.

    ZH:
    从当前命令行参数构建模型配置。
    """
    return {
        "model_version": str(args.model_version),

        "acoustic_dim": int(args.acoustic_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "semantic_vocab_size": int(args.semantic_vocab_size),
        "semantic_rate_hz": float(args.semantic_rate_hz),
        "acoustic_rate_hz": float(args.acoustic_rate_hz),
        "dropout": float(args.dropout),

        "use_self_condition": bool(args.use_self_condition),
        "self_condition_prob": float(args.self_condition_prob),

        "enable_length_predictor": bool(args.enable_length_predictor),
        "length_predictor_hidden_dim": int(args.length_predictor_hidden_dim),
        "length_loss_weight": float(args.length_loss_weight),
        "length_teacher_forcing_prob": float(args.length_teacher_forcing_prob),

        "phoneme_vocab_size": int(args.phoneme_vocab_size),
        "phoneme_embed_dim": int(args.phoneme_embed_dim),
        "bert_dim": int(args.bert_dim),
        "content_dim": int(args.content_dim),
        "content_refiner_layers": int(args.content_refiner_layers),
        "content_frame_dim": int(args.content_frame_dim),
        "content_expander_layers": int(args.content_expander_layers),
        "style_dim": int(args.style_dim),

        "content_main_dim": int(args.content_main_dim),
        "content_injection_gate_init": float(args.content_injection_gate_init),
        "content_refiner_dilations": parse_int_tuple(args.content_refiner_dilations),
        "content_boundary_enhance": bool(args.content_boundary_enhance),
        "content_boundary_kernel_size": int(args.content_boundary_kernel_size),
        "content_boundary_residual_scale": float(args.content_boundary_residual_scale),
        "content_refiner_residual_scale": float(args.content_refiner_residual_scale),

        "ff_mult": int(args.ff_mult),
        "local_detail_kernel_size": int(args.local_detail_kernel_size),
        "local_detail_dilation_cycle": parse_int_tuple(args.local_detail_dilation_cycle),
        "local_detail_residual_scale": float(args.local_detail_residual_scale),
        "semantic_guide_gate_init": float(args.semantic_guide_gate_init),
        "use_semantic_guide_cross_attn": bool(args.use_semantic_guide_cross_attn),

        "fm_loss_weight": float(args.fm_loss_weight),
        "recon_loss_weight": float(args.recon_loss_weight),
        "refined_recon_loss_weight": float(args.refined_recon_loss_weight),
        "delta_loss_weight": float(args.delta_loss_weight),
        "delta2_loss_weight": float(args.delta2_loss_weight),
        "mid_content_aux_weight": float(args.mid_content_aux_weight),
        "final_content_aux_weight": float(args.final_content_aux_weight),
        "span_focus_aux_weight": float(args.span_focus_aux_weight),
        "span_focus_ratio_min": float(args.span_focus_ratio_min),
        "span_focus_ratio_max": float(args.span_focus_ratio_max),
        "span_recon_weight": float(args.span_recon_weight),
        "span_content_weight": float(args.span_content_weight),
        "recon_loss_type": str(args.recon_loss_type),

        "cond_drop_prob_content": float(args.cond_drop_prob_content),
        "cond_drop_prob_semantic": float(args.cond_drop_prob_semantic),
        "cond_drop_prob_style": float(args.cond_drop_prob_style),
        "cond_drop_all_prob": float(args.cond_drop_all_prob),

        "postnet_num_layers": int(args.postnet_num_layers),
        "postnet_dropout": float(args.postnet_dropout),
        "postnet_conv_kernel_size": int(args.postnet_conv_kernel_size),
        "detach_coarse_for_refinement": bool(args.detach_coarse_for_refinement),

        "t_sampling_mode": str(args.t_sampling_mode),
        "t_bias_power": float(args.t_bias_power),
        "t_min": float(args.t_min),
        "t_max": float(args.t_max),

        "freq_weight_min": float(args.freq_weight_min),
        "freq_weight_max": float(args.freq_weight_max),
        "freq_weight_power": float(args.freq_weight_power),

        "content_head_dropout": float(args.content_head_dropout),

        "enable_length_clamp": bool(args.enable_length_clamp),
        "length_correction_mode": str(args.length_correction_mode),
        "length_ratio_min": float(args.length_ratio_min),
        "length_ratio_max": float(args.length_ratio_max),
        "length_bias_scale": float(args.length_bias_scale),

        # ----------------------------------------------------
        # v6.2 cold-start bootstrapper args
        # v6.2 冷启动 bootstrapper 参数
        # ----------------------------------------------------
        "use_coarse_bootstrapper": bool(args.use_coarse_bootstrapper),
        "bootstrap_hidden_dim": (
            None
            if args.bootstrap_hidden_dim is None
            else int(args.bootstrap_hidden_dim)
        ),
        "bootstrap_num_blocks": int(args.bootstrap_num_blocks),
        "bootstrap_kernel_size": int(args.bootstrap_kernel_size),
        "bootstrap_dropout": float(args.bootstrap_dropout),
        "bootstrap_expansion_factor": int(args.bootstrap_expansion_factor),

        "bootstrap_mel_bias_init": float(args.bootstrap_mel_bias_init),
        "bootstrap_residual_scale_init": float(args.bootstrap_residual_scale_init),
        "bootstrap_semantic_gate_init": float(args.bootstrap_semantic_gate_init),
        "bootstrap_style_residual_scale_init": float(args.bootstrap_style_residual_scale_init),
        "bootstrap_clamp_output": bool(args.bootstrap_clamp_output),
        "bootstrap_output_min": float(args.bootstrap_output_min),
        "bootstrap_output_max": float(args.bootstrap_output_max),

        "coarse_mel_loss_weight": float(args.coarse_mel_loss_weight),
        "coarse_mel_mse_loss_weight": float(args.coarse_mel_mse_loss_weight),
        "coarse_delta_loss_weight": float(args.coarse_delta_loss_weight),

        "use_bridge_loss": bool(args.use_bridge_loss),
        "bridge_t_start": float(args.bridge_t_start),
        "bridge_noise_temperature": float(args.bridge_noise_temperature),
        "bridge_recon_loss_weight": float(args.bridge_recon_loss_weight),
        "bridge_delta_loss_weight": float(args.bridge_delta_loss_weight),
        "detach_coarse_for_bridge": bool(args.detach_coarse_for_bridge),

        # ----------------------------------------------------
        # v6.3 Conformer bootstrapper args
        # ----------------------------------------------------
        "use_conformer_bootstrapper": bool(args.use_conformer_bootstrapper),
        "v63_bootstrap_hidden_dim": (
            None
            if args.v63_bootstrap_hidden_dim is None
            else int(args.v63_bootstrap_hidden_dim)
        ),
        "v63_bootstrap_num_layers": int(args.v63_bootstrap_num_layers),
        "v63_bootstrap_num_heads": int(args.v63_bootstrap_num_heads),
        "v63_bootstrap_ff_mult": int(args.v63_bootstrap_ff_mult),
        "v63_bootstrap_conv_kernel_size": int(args.v63_bootstrap_conv_kernel_size),
        "v63_bootstrap_dropout": float(args.v63_bootstrap_dropout),
        "v63_bootstrap_mel_bias_init": float(args.v63_bootstrap_mel_bias_init),
        "v63_bootstrap_zero_init_output": bool(args.v63_bootstrap_zero_init_output),
        "v63_bootstrap_clamp_output": bool(args.v63_bootstrap_clamp_output),

        # ----------------------------------------------------
        # v6.3.3 native text memory args
        # ----------------------------------------------------
        "v63_use_native_text_memory": bool(args.v63_use_native_text_memory),
        "v63_text_memory_dim": (
            None
            if args.v63_text_memory_dim is None
            else int(args.v63_text_memory_dim)
        ),

        # ----------------------------------------------------
        # v6.3 residual refiner args
        # ----------------------------------------------------
        "use_residual_refiner": bool(args.use_residual_refiner),
        "residual_hidden_dim": (
            None
            if args.residual_hidden_dim is None
            else int(args.residual_hidden_dim)
        ),
        "residual_num_layers": int(args.residual_num_layers),
        "residual_num_heads": int(args.residual_num_heads),
        "residual_ff_mult": int(args.residual_ff_mult),
        "residual_conv_kernel_size": int(args.residual_conv_kernel_size),
        "residual_dropout": float(args.residual_dropout),
        "residual_noise_scale": float(args.residual_noise_scale),
        "residual_sample_temperature": float(args.residual_sample_temperature),
        "residual_t_min": float(args.residual_t_min),
        "residual_t_max": float(args.residual_t_max),
        "detach_coarse_for_residual_refiner": bool(
            args.detach_coarse_for_residual_refiner
        ),

        # ----------------------------------------------------
        # v6.4.1 continuous semantic
        # ----------------------------------------------------
        "use_continuous_semantic": bool(args.use_continuous_semantic),
        "continuous_semantic_dim": int(args.continuous_semantic_dim),
        "continuous_semantic_fusion_mode": str(args.continuous_semantic_fusion_mode),
        "continuous_semantic_gate_init": float(args.continuous_semantic_gate_init),
        "continuous_semantic_dropout": float(args.continuous_semantic_dropout),

        # ----------------------------------------------------
        # v6.6 reference acoustic style
        # ----------------------------------------------------
        "use_reference_acoustic_style": bool(args.use_reference_acoustic_style),
        "reference_acoustic_dim": int(args.reference_acoustic_dim),
        "reference_style_dim": int(args.reference_style_dim),
        "reference_style_hidden_dim": int(args.reference_style_hidden_dim),
        "reference_style_num_layers": int(args.reference_style_num_layers),
        "reference_style_kernel_size": int(args.reference_style_kernel_size),
        "reference_style_dropout": float(args.reference_style_dropout),
        "reference_style_fusion_mode": str(args.reference_style_fusion_mode),
        "reference_style_gate_init": float(args.reference_style_gate_init),

        # ----------------------------------------------------
        # v6.6.2 prosody / speaker auxiliary losses
        # ----------------------------------------------------
        "use_v66_energy_loss": bool(args.use_v66_energy_loss),
        "v66_energy_loss_weight": float(args.v66_energy_loss_weight),
        "v66_energy_loss_type": str(args.v66_energy_loss_type),
        "v66_energy_normalize": bool(args.v66_energy_normalize),
        "use_v66_f0_loss": bool(args.use_v66_f0_loss),
        "v66_f0_loss_weight": float(args.v66_f0_loss_weight),
        "v66_f0_loss_type": str(args.v66_f0_loss_type),
        "v66_f0_normalize": bool(args.v66_f0_normalize),
        "v66_f0_hidden_dim": int(args.v66_f0_hidden_dim),
        "use_v66_speaker_loss": bool(args.use_v66_speaker_loss),
        "v66_speaker_loss_weight": float(args.v66_speaker_loss_weight),
        "v66_speaker_embedding_dim": int(args.v66_speaker_embedding_dim),
        "v66_require_aux_targets": bool(args.v66_require_aux_targets),

        # ----------------------------------------------------
        # v6.3 loss args
        # ----------------------------------------------------
        # ----------------------------------------------------
        # v6.6.4 Speaker Identity Adapter
        # ----------------------------------------------------
        "use_v664_speaker_identity_adapter": bool(args.use_v664_speaker_identity_adapter),
        "v664_speaker_identity_source": str(args.v664_speaker_identity_source),
        "v664_speaker_embedding_dim": int(args.v664_speaker_embedding_dim),
        "v664_speaker_adapter_hidden_dim": int(args.v664_speaker_adapter_hidden_dim),
        "v664_speaker_adapter_num_layers": int(args.v664_speaker_adapter_num_layers),
        "v664_speaker_adapter_dropout": float(args.v664_speaker_adapter_dropout),
        "v664_speaker_adapter_fusion_mode": str(args.v664_speaker_adapter_fusion_mode),
        "v664_speaker_adapter_gate_init": float(args.v664_speaker_adapter_gate_init),
        "v664_speaker_adapter_normalize": bool(args.v664_speaker_adapter_normalize),

        # ----------------------------------------------------
        # v6.6.3 Bootstrapper LoRA
        # ----------------------------------------------------
        "use_bootstrapper_lora": bool(args.use_bootstrapper_lora),
        "bootstrapper_lora_rank": int(args.bootstrapper_lora_rank),
        "bootstrapper_lora_alpha": float(args.bootstrapper_lora_alpha),
        "bootstrapper_lora_dropout": float(args.bootstrapper_lora_dropout),
        "bootstrapper_lora_target": str(args.bootstrapper_lora_target),
        "bootstrapper_lora_init_scale": float(args.bootstrapper_lora_init_scale),

        # ----------------------------------------------------
        # v6.6.3-A Bootstrapper Attention LoRA
        # ----------------------------------------------------
        "use_bootstrapper_attention_lora": bool(args.use_bootstrapper_attention_lora),
        "bootstrapper_attention_lora_rank": int(args.bootstrapper_attention_lora_rank),
        "bootstrapper_attention_lora_alpha": float(args.bootstrapper_attention_lora_alpha),
        "bootstrapper_attention_lora_dropout": float(args.bootstrapper_attention_lora_dropout),
        "bootstrapper_attention_lora_target": str(args.bootstrapper_attention_lora_target),
        "bootstrapper_attention_lora_gate_init": float(args.bootstrapper_attention_lora_gate_init),
        "bootstrapper_attention_lora_enable_q": bool(args.bootstrapper_attention_lora_enable_q),
        "bootstrapper_attention_lora_enable_k": bool(args.bootstrapper_attention_lora_enable_k),
        "bootstrapper_attention_lora_enable_v": bool(args.bootstrapper_attention_lora_enable_v),
        "bootstrapper_attention_lora_enable_o": bool(args.bootstrapper_attention_lora_enable_o),

        "include_v61_base_loss": bool(args.include_v61_base_loss),
        "v61_base_loss_weight": float(args.v61_base_loss_weight),

        "v63_coarse_l1_weight": float(args.v63_coarse_l1_weight),
        "v63_coarse_mse_weight": float(args.v63_coarse_mse_weight),
        "v63_coarse_delta_weight": float(args.v63_coarse_delta_weight),
        "v63_coarse_delta2_weight": float(args.v63_coarse_delta2_weight),

        "residual_refiner_loss_weight": float(args.residual_refiner_loss_weight),
        "residual_flow_mse_weight": float(args.residual_flow_mse_weight),
        "residual_flow_l1_weight": float(args.residual_flow_l1_weight),
        "residual_recon_l1_weight": float(args.residual_recon_l1_weight),
        "final_recon_l1_weight": float(args.final_recon_l1_weight),
        "final_delta_l1_weight": float(args.final_delta_l1_weight),
        "final_delta2_l1_weight": float(args.final_delta2_l1_weight),
    }


def build_model_from_config(model_config: dict[str, Any]):
    """
    EN:
    Instantiate stage-2 model from model_config.

    ZH:
    根据模型配置实例化第二阶段模型。
    """
    model_version = str(model_config.get("model_version", "v1")).lower()

    common_kwargs = dict(
        acoustic_dim=model_config["acoustic_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        semantic_vocab_size=model_config["semantic_vocab_size"],
        semantic_rate_hz=model_config["semantic_rate_hz"],
        acoustic_rate_hz=model_config["acoustic_rate_hz"],
        dropout=model_config["dropout"],

        use_self_condition=model_config.get("use_self_condition", True),
        self_condition_prob=model_config.get("self_condition_prob", 0.5),

        phoneme_vocab_size=model_config.get("phoneme_vocab_size", 4096),
        phoneme_embed_dim=model_config.get("phoneme_embed_dim", 256),
        bert_dim=model_config.get("bert_dim", 1024),
        content_dim=model_config.get("content_dim", 384),
        content_refiner_layers=model_config.get("content_refiner_layers", 4),
        content_frame_dim=model_config.get("content_frame_dim", 384),
        content_expander_layers=model_config.get("content_expander_layers", 2),
        style_dim=model_config.get("style_dim", 256),

        content_main_dim=model_config.get("content_main_dim", 512),
        content_injection_gate_init=model_config.get("content_injection_gate_init", 0.5),
        content_refiner_dilations=model_config.get("content_refiner_dilations", (1, 2)),
        content_boundary_enhance=model_config.get("content_boundary_enhance", True),
        content_boundary_kernel_size=model_config.get("content_boundary_kernel_size", 3),
        content_boundary_residual_scale=model_config.get("content_boundary_residual_scale", 0.3),
        content_refiner_residual_scale=model_config.get("content_refiner_residual_scale", 0.5),

        ff_mult=model_config.get("ff_mult", 4),
        local_detail_kernel_size=model_config.get("local_detail_kernel_size", 5),
        local_detail_dilation_cycle=model_config.get("local_detail_dilation_cycle", (1, 2)),
        local_detail_residual_scale=model_config.get("local_detail_residual_scale", 0.5),
        semantic_guide_gate_init=model_config.get("semantic_guide_gate_init", 0.3),
        use_semantic_guide_cross_attn=model_config.get("use_semantic_guide_cross_attn", True),

        fm_loss_weight=model_config.get("fm_loss_weight", 1.0),
        recon_loss_weight=model_config.get("recon_loss_weight", 0.20),
        refined_recon_loss_weight=model_config.get("refined_recon_loss_weight", 1.00),
        delta_loss_weight=model_config.get("delta_loss_weight", 0.15),
        delta2_loss_weight=model_config.get("delta2_loss_weight", 0.05),
        mid_content_aux_weight=model_config.get("mid_content_aux_weight", 0.10),
        final_content_aux_weight=model_config.get("final_content_aux_weight", 0.20),
        span_focus_aux_weight=model_config.get("span_focus_aux_weight", 0.20),
        span_focus_ratio_min=model_config.get("span_focus_ratio_min", 0.10),
        span_focus_ratio_max=model_config.get("span_focus_ratio_max", 0.35),
        span_recon_weight=model_config.get("span_recon_weight", 0.50),
        span_content_weight=model_config.get("span_content_weight", 0.50),
        recon_loss_type=model_config.get("recon_loss_type", "l1"),

        cond_drop_prob_content=model_config.get("cond_drop_prob_content", 0.05),
        cond_drop_prob_semantic=model_config.get("cond_drop_prob_semantic", 0.10),
        cond_drop_prob_style=model_config.get("cond_drop_prob_style", 0.10),
        cond_drop_all_prob=model_config.get("cond_drop_all_prob", 0.05),

        postnet_num_layers=model_config.get("postnet_num_layers", 3),
        postnet_dropout=model_config.get("postnet_dropout", 0.1),
        postnet_conv_kernel_size=model_config.get("postnet_conv_kernel_size", 5),
        detach_coarse_for_refinement=model_config.get("detach_coarse_for_refinement", True),

        t_sampling_mode=model_config.get("t_sampling_mode", "near_clean"),
        t_bias_power=model_config.get("t_bias_power", 2.0),
        t_min=model_config.get("t_min", 0.05),
        t_max=model_config.get("t_max", 0.98),

        freq_weight_min=model_config.get("freq_weight_min", 1.0),
        freq_weight_max=model_config.get("freq_weight_max", 2.0),
        freq_weight_power=model_config.get("freq_weight_power", 1.5),

        content_head_dropout=model_config.get("content_head_dropout", 0.1),

        enable_length_clamp=model_config.get("enable_length_clamp", True),
        length_correction_mode=model_config.get("length_correction_mode", "gentle_ratio"),
        length_ratio_min=model_config.get("length_ratio_min", 2.6),
        length_ratio_max=model_config.get("length_ratio_max", 3.8),
        length_bias_scale=model_config.get("length_bias_scale", 0.95),

        # ----------------------------------------------------
        # v6.2 cold-start bootstrapper args
        # ----------------------------------------------------
        use_coarse_bootstrapper=model_config.get("use_coarse_bootstrapper", True),
        bootstrap_hidden_dim=model_config.get("bootstrap_hidden_dim", None),
        bootstrap_num_blocks=model_config.get("bootstrap_num_blocks", 4),
        bootstrap_kernel_size=model_config.get("bootstrap_kernel_size", 5),
        bootstrap_dropout=model_config.get("bootstrap_dropout", 0.1),
        bootstrap_expansion_factor=model_config.get("bootstrap_expansion_factor", 4),

        bootstrap_mel_bias_init=model_config.get("bootstrap_mel_bias_init", -5.0),
        bootstrap_residual_scale_init=model_config.get("bootstrap_residual_scale_init", 0.1),
        bootstrap_semantic_gate_init=model_config.get("bootstrap_semantic_gate_init", 0.5),
        bootstrap_style_residual_scale_init=model_config.get("bootstrap_style_residual_scale_init", 0.1),
        bootstrap_clamp_output=model_config.get("bootstrap_clamp_output", False),
        bootstrap_output_min=model_config.get("bootstrap_output_min", -12.0),
        bootstrap_output_max=model_config.get("bootstrap_output_max", 4.0),

        coarse_mel_loss_weight=model_config.get("coarse_mel_loss_weight", 1.0),
        coarse_mel_mse_loss_weight=model_config.get("coarse_mel_mse_loss_weight", 0.0),
        coarse_delta_loss_weight=model_config.get("coarse_delta_loss_weight", 0.2),

        use_bridge_loss=model_config.get("use_bridge_loss", True),
        bridge_t_start=model_config.get("bridge_t_start", 0.075),
        bridge_noise_temperature=model_config.get("bridge_noise_temperature", 0.3),
        bridge_recon_loss_weight=model_config.get("bridge_recon_loss_weight", 1.0),
        bridge_delta_loss_weight=model_config.get("bridge_delta_loss_weight", 0.2),
        detach_coarse_for_bridge=model_config.get("detach_coarse_for_bridge", True),

        # ----------------------------------------------------
        # v6.3 Conformer bootstrapper args
        # ----------------------------------------------------
        use_conformer_bootstrapper=model_config.get(
            "use_conformer_bootstrapper",
            True,
        ),
        v63_bootstrap_hidden_dim=model_config.get(
            "v63_bootstrap_hidden_dim",
            None,
        ),
        v63_bootstrap_num_layers=model_config.get(
            "v63_bootstrap_num_layers",
            6,
        ),
        v63_bootstrap_num_heads=model_config.get(
            "v63_bootstrap_num_heads",
            8,
        ),
        v63_bootstrap_ff_mult=model_config.get(
            "v63_bootstrap_ff_mult",
            4,
        ),
        v63_bootstrap_conv_kernel_size=model_config.get(
            "v63_bootstrap_conv_kernel_size",
            15,
        ),
        v63_bootstrap_dropout=model_config.get(
            "v63_bootstrap_dropout",
            0.1,
        ),
        v63_bootstrap_mel_bias_init=model_config.get(
            "v63_bootstrap_mel_bias_init",
            -5.0,
        ),
        v63_bootstrap_zero_init_output=model_config.get(
            "v63_bootstrap_zero_init_output",
            False,
        ),
        v63_bootstrap_clamp_output=model_config.get(
            "v63_bootstrap_clamp_output",
            False,
        ),

        # ----------------------------------------------------
        # v6.3.3 native text memory args
        # ----------------------------------------------------
        v63_use_native_text_memory=model_config.get(
            "v63_use_native_text_memory",
            False,
        ),
        v63_text_memory_dim=model_config.get(
            "v63_text_memory_dim",
            None,
        ),

        # ----------------------------------------------------
        # v6.3 residual refiner args
        # ----------------------------------------------------
        use_residual_refiner=model_config.get(
            "use_residual_refiner",
            True,
        ),
        residual_hidden_dim=model_config.get(
            "residual_hidden_dim",
            None,
        ),
        residual_num_layers=model_config.get(
            "residual_num_layers",
            6,
        ),
        residual_num_heads=model_config.get(
            "residual_num_heads",
            8,
        ),
        residual_ff_mult=model_config.get(
            "residual_ff_mult",
            4,
        ),
        residual_conv_kernel_size=model_config.get(
            "residual_conv_kernel_size",
            15,
        ),
        residual_dropout=model_config.get(
            "residual_dropout",
            0.1,
        ),
        residual_noise_scale=model_config.get(
            "residual_noise_scale",
            0.5,
        ),
        residual_sample_temperature=model_config.get(
            "residual_sample_temperature",
            0.3,
        ),
        residual_t_min=model_config.get(
            "residual_t_min",
            0.0,
        ),
        residual_t_max=model_config.get(
            "residual_t_max",
            1.0,
        ),
        detach_coarse_for_residual_refiner=model_config.get(
            "detach_coarse_for_residual_refiner",
            False,
        ),

        # ----------------------------------------------------
        # v6.4.1 continuous semantic
        # ----------------------------------------------------
        use_continuous_semantic=model_config.get("use_continuous_semantic", False),
        continuous_semantic_dim=model_config.get("continuous_semantic_dim", 768),
        continuous_semantic_fusion_mode=model_config.get(
            "continuous_semantic_fusion_mode",
            "gated_add",
        ),
        continuous_semantic_gate_init=model_config.get(
            "continuous_semantic_gate_init",
            -2.0,
        ),
        continuous_semantic_dropout=model_config.get(
            "continuous_semantic_dropout",
            0.0,
        ),

        # ----------------------------------------------------
        # v6.6 reference acoustic style args
        # ----------------------------------------------------
        use_reference_acoustic_style=model_config.get("use_reference_acoustic_style", False),
        reference_acoustic_dim=model_config.get("reference_acoustic_dim", 80),
        reference_style_dim=model_config.get("reference_style_dim", 256),
        reference_style_hidden_dim=model_config.get("reference_style_hidden_dim", 256),
        reference_style_num_layers=model_config.get("reference_style_num_layers", 4),
        reference_style_kernel_size=model_config.get("reference_style_kernel_size", 5),
        reference_style_dropout=model_config.get("reference_style_dropout", 0.1),
        reference_style_fusion_mode=model_config.get("reference_style_fusion_mode", "gated_add"),
        reference_style_gate_init=model_config.get("reference_style_gate_init", -3.0),

        # ----------------------------------------------------
        # v6.6.2 prosody / speaker auxiliary losses
        # ----------------------------------------------------
        use_v66_energy_loss=model_config.get("use_v66_energy_loss", False),
        v66_energy_loss_weight=model_config.get("v66_energy_loss_weight", 0.15),
        v66_energy_loss_type=model_config.get("v66_energy_loss_type", "l1"),
        v66_energy_normalize=model_config.get("v66_energy_normalize", True),
        use_v66_f0_loss=model_config.get("use_v66_f0_loss", False),
        v66_f0_loss_weight=model_config.get("v66_f0_loss_weight", 0.05),
        v66_f0_loss_type=model_config.get("v66_f0_loss_type", "l1"),
        v66_f0_normalize=model_config.get("v66_f0_normalize", True),
        v66_f0_hidden_dim=model_config.get("v66_f0_hidden_dim", 128),
        use_v66_speaker_loss=model_config.get("use_v66_speaker_loss", False),
        v66_speaker_loss_weight=model_config.get("v66_speaker_loss_weight", 0.05),
        v66_speaker_embedding_dim=model_config.get("v66_speaker_embedding_dim", 192),
        v66_require_aux_targets=model_config.get("v66_require_aux_targets", False),

        # ----------------------------------------------------
        # v6.3 loss args
        # ----------------------------------------------------
        # ----------------------------------------------------
        # v6.6.4 Speaker Identity Adapter
        # ----------------------------------------------------
        use_v664_speaker_identity_adapter=model_config.get("use_v664_speaker_identity_adapter", False),
        v664_speaker_identity_source=model_config.get("v664_speaker_identity_source", "prompt_or_target"),
        v664_speaker_embedding_dim=model_config.get("v664_speaker_embedding_dim", model_config.get("v66_speaker_embedding_dim", 192)),
        v664_speaker_adapter_hidden_dim=model_config.get("v664_speaker_adapter_hidden_dim", 256),
        v664_speaker_adapter_num_layers=model_config.get("v664_speaker_adapter_num_layers", 2),
        v664_speaker_adapter_dropout=model_config.get("v664_speaker_adapter_dropout", 0.05),
        v664_speaker_adapter_fusion_mode=model_config.get("v664_speaker_adapter_fusion_mode", "gated_add"),
        v664_speaker_adapter_gate_init=model_config.get("v664_speaker_adapter_gate_init", -3.0),
        v664_speaker_adapter_normalize=model_config.get("v664_speaker_adapter_normalize", True),

        # ----------------------------------------------------
        # v6.6.3 Bootstrapper LoRA
        # ----------------------------------------------------
        use_bootstrapper_lora=model_config.get("use_bootstrapper_lora", False),
        bootstrapper_lora_rank=model_config.get("bootstrapper_lora_rank", 4),
        bootstrapper_lora_alpha=model_config.get("bootstrapper_lora_alpha", 8.0),
        bootstrapper_lora_dropout=model_config.get("bootstrapper_lora_dropout", 0.05),
        bootstrapper_lora_target=model_config.get("bootstrapper_lora_target", "core"),
        bootstrapper_lora_init_scale=model_config.get("bootstrapper_lora_init_scale", 0.01),

        # ----------------------------------------------------
        # v6.6.3-A Bootstrapper Attention LoRA
        # ----------------------------------------------------
        use_bootstrapper_attention_lora=model_config.get("use_bootstrapper_attention_lora", False),
        bootstrapper_attention_lora_rank=model_config.get("bootstrapper_attention_lora_rank", 4),
        bootstrapper_attention_lora_alpha=model_config.get("bootstrapper_attention_lora_alpha", 8.0),
        bootstrapper_attention_lora_dropout=model_config.get("bootstrapper_attention_lora_dropout", 0.05),
        bootstrapper_attention_lora_target=model_config.get("bootstrapper_attention_lora_target", "cross_only"),
        bootstrapper_attention_lora_gate_init=model_config.get("bootstrapper_attention_lora_gate_init", -3.0),
        bootstrapper_attention_lora_enable_q=model_config.get("bootstrapper_attention_lora_enable_q", True),
        bootstrapper_attention_lora_enable_k=model_config.get("bootstrapper_attention_lora_enable_k", True),
        bootstrapper_attention_lora_enable_v=model_config.get("bootstrapper_attention_lora_enable_v", True),
        bootstrapper_attention_lora_enable_o=model_config.get("bootstrapper_attention_lora_enable_o", True),

        include_v61_base_loss=model_config.get(
            "include_v61_base_loss",
            False,
        ),
        v61_base_loss_weight=model_config.get(
            "v61_base_loss_weight",
            1.0,
        ),

        v63_coarse_l1_weight=model_config.get(
            "v63_coarse_l1_weight",
            2.0,
        ),
        v63_coarse_mse_weight=model_config.get(
            "v63_coarse_mse_weight",
            0.2,
        ),
        v63_coarse_delta_weight=model_config.get(
            "v63_coarse_delta_weight",
            0.5,
        ),
        v63_coarse_delta2_weight=model_config.get(
            "v63_coarse_delta2_weight",
            0.2,
        ),

        residual_refiner_loss_weight=model_config.get(
            "residual_refiner_loss_weight",
            1.0,
        ),
        residual_flow_mse_weight=model_config.get(
            "residual_flow_mse_weight",
            1.0,
        ),
        residual_flow_l1_weight=model_config.get(
            "residual_flow_l1_weight",
            0.0,
        ),
        residual_recon_l1_weight=model_config.get(
            "residual_recon_l1_weight",
            1.0,
        ),
        final_recon_l1_weight=model_config.get(
            "final_recon_l1_weight",
            1.0,
        ),
        final_delta_l1_weight=model_config.get(
            "final_delta_l1_weight",
            0.3,
        ),
        final_delta2_l1_weight=model_config.get(
            "final_delta2_l1_weight",
            0.1,
        ),
    )

    if model_version in {"v6_3", "v6.3"}:
        return Stage2FlowMatchingAcousticModelV63(**common_kwargs)

    if model_version in {"v6_2", "v6.2"}:
        return Stage2FlowMatchingAcousticModelV62(**common_kwargs)

    if model_version in {"v6_1", "v6.1"}:
        return Stage2FlowMatchingAcousticModelV61(**common_kwargs)

    if model_version == "v6":
        return Stage2FlowMatchingAcousticModelV6(**common_kwargs)

    if model_version == "v5":
        return Stage2FlowMatchingAcousticModelV5(**common_kwargs)

    if model_version == "v2":
        return Stage2FlowMatchingAcousticModelV2(**common_kwargs)

    return Stage2FlowMatchingAcousticModel(**common_kwargs)


def make_amp_context(device: torch.device, use_amp: bool):
    """
    EN:
    Return a context manager for autocast.

    ZH:
    返回自动混合精度上下文管理器。
    """
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def build_grad_scaler(device: torch.device, use_amp: bool):
    """
    EN:
    Build AMP GradScaler only when CUDA AMP is enabled.

    ZH:
    仅在 CUDA AMP 开启时构建 GradScaler。
    """
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return None

def count_trainable_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0

    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
    }


def set_requires_grad(module: torch.nn.Module | None, flag: bool) -> None:
    if module is None:
        return

    for p in module.parameters():
        p.requires_grad = bool(flag)


def set_named_parameters_trainable(
    model: torch.nn.Module,
    *,
    include_keywords: list[str],
) -> list[str]:
    """
    EN:
    Freeze all parameters, then unfreeze parameters whose names contain
    one of include_keywords.

    ZH:
    先冻结全部参数，再解冻名称命中 include_keywords 的参数。
    """
    for p in model.parameters():
        p.requires_grad = False

    unfrozen: list[str] = []

    for name, p in model.named_parameters():
        if any(keyword in name for keyword in include_keywords):
            p.requires_grad = True
            unfrozen.append(name)

    return unfrozen


def _build_partial_scope_report(
    model: torch.nn.Module,
    *,
    scope: str,
    include_keywords: list[str],
    unfrozen: list[str],
) -> dict[str, Any]:
    """
    EN:
    Build a report for partial fine-tuning scopes.

    ZH:
    构建部分微调范围报告。
    """
    if len(unfrozen) <= 0:
        raise RuntimeError(
            f"trainable_scope={scope} matched no parameters. "
            f"include_keywords={include_keywords}"
        )

    return {
        "trainable_scope": scope,
        "unfrozen_param_keywords": include_keywords,
        "num_unfrozen_param_names": int(len(unfrozen)),
        "unfrozen_param_names_preview": unfrozen[:80],
        "frozen_modules_policy": "all_other_parameters_frozen",
        **count_trainable_parameters(model),
    }


def apply_trainable_scope(
    model: torch.nn.Module,
    *,
    trainable_scope: str,
) -> dict[str, Any]:
    """
    Apply trainable scope.

    Supported scopes:
        all:
            Train all parameters. This is useful for from-scratch training,
            but is too aggressive for small few-shot adaptation.

        residual_refiner_only:
            Freeze everything, then unfreeze model.residual_refiner.
            This is kept for older residual-refiner experiments.

        style_condition_only:
            Freeze the acoustic generation backbone and train only the
            prompt/style condition path inside condition_encoder.

        style_adapter_only:
            Freeze the content/semantic/text generation path and train only
            the style injection adapters inside conformer_bootstrapper.

        style_condition_plus_adapter:
            Train both the condition-side style mapping and the bootstrapper
            style injection adapters. This is stronger than style_adapter_only.
    """
    scope = str(trainable_scope).strip().lower()

    if scope == "all":
        for p in model.parameters():
            p.requires_grad = True

        return {
            "trainable_scope": "all",
            **count_trainable_parameters(model),
        }

    if scope == "residual_refiner_only":
        for p in model.parameters():
            p.requires_grad = False

        residual_refiner = getattr(model, "residual_refiner", None)

        if residual_refiner is None:
            raise RuntimeError(
                "trainable_scope=residual_refiner_only requires model.residual_refiner, "
                "but residual_refiner is None."
            )

        set_requires_grad(residual_refiner, True)

        return {
            "trainable_scope": "residual_refiner_only",
            "unfrozen_modules": ["residual_refiner"],
            "frozen_modules_policy": "all_other_modules_frozen",
            **count_trainable_parameters(model),
        }

    if scope == "style_condition_only":
        include_keywords = [
            "condition_encoder.prompt_seq_proj",
            "condition_encoder.prompt_global_proj",
            "condition_encoder.global_fuse",
            "condition_encoder.style_global_out_proj",
        ]
        unfrozen = set_named_parameters_trainable(
            model,
            include_keywords=include_keywords,
        )
        return _build_partial_scope_report(
            model,
            scope=scope,
            include_keywords=include_keywords,
            unfrozen=unfrozen,
        )

    if scope == "style_adapter_only":
        include_keywords = [
            "conformer_bootstrapper.query_builder.style_to_query",
            "conformer_bootstrapper.query_builder.style_residual_scale",
            "conformer_bootstrapper.style_film",
        ]
        unfrozen = set_named_parameters_trainable(
            model,
            include_keywords=include_keywords,
        )
        return _build_partial_scope_report(
            model,
            scope=scope,
            include_keywords=include_keywords,
            unfrozen=unfrozen,
        )

    if scope == "style_condition_plus_adapter":
        include_keywords = [
            "condition_encoder.prompt_seq_proj",
            "condition_encoder.prompt_global_proj",
            "condition_encoder.global_fuse",
            "condition_encoder.style_global_out_proj",
            "conformer_bootstrapper.query_builder.style_to_query",
            "conformer_bootstrapper.query_builder.style_residual_scale",
            "conformer_bootstrapper.style_film",
        ]
        unfrozen = set_named_parameters_trainable(
            model,
            include_keywords=include_keywords,
        )
        return _build_partial_scope_report(
            model,
            scope=scope,
            include_keywords=include_keywords,
            unfrozen=unfrozen,
        )


    if scope == "reference_style_encoder_only":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_adapter":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "conformer_bootstrapper.query_builder.style_to_query",
            "conformer_bootstrapper.query_builder.style_residual_scale",
            "conformer_bootstrapper.style_film",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_condition_adapter":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "condition_encoder.prompt_seq_proj",
            "condition_encoder.prompt_global_proj",
            "condition_encoder.global_fuse",
            "condition_encoder.style_global_out_proj",
            "conformer_bootstrapper.query_builder.style_to_query",
            "conformer_bootstrapper.query_builder.style_residual_scale",
            "conformer_bootstrapper.style_film",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)


    if scope == "reference_style_plus_prosody":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "conformer_bootstrapper.query_builder.style_to_query",
            "conformer_bootstrapper.query_builder.style_residual_scale",
            "conformer_bootstrapper.style_film",
            "v66_f0_head",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_all_aux":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "conformer_bootstrapper.query_builder.style_to_query",
            "conformer_bootstrapper.query_builder.style_residual_scale",
            "conformer_bootstrapper.style_film",
            "v66_f0_head",
            "v66_speaker_head",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)


    if scope == "bootstrapper_lora_only":
        include_keywords = [".lora_A.", ".lora_B."]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_lora":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            ".lora_A.",
            ".lora_B.",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_lora_plus_aux":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "v66_f0_head",
            "v66_speaker_head",
            ".lora_A.",
            ".lora_B.",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)


    if scope == "speaker_identity_adapter_only":
        include_keywords = ["speaker_identity_adapter"]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "speaker_identity_plus_lora":
        include_keywords = [
            "speaker_identity_adapter",
            ".lora_A.",
            ".lora_B.",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "speaker_identity_plus_lora_plus_aux":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "speaker_identity_adapter",
            "v66_f0_head",
            "v66_speaker_head",
            ".lora_A.",
            ".lora_B.",
        ]
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)


    attention_lora_keywords = [
        "lora_q_A", "lora_q_B",
        "lora_k_A", "lora_k_B",
        "lora_v_A", "lora_v_B",
        "lora_o_A", "lora_o_B",
        "attention_lora_gate",
    ]

    if scope == "attention_lora_only":
        include_keywords = list(attention_lora_keywords)
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_attention_lora":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
        ] + list(attention_lora_keywords)
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    if scope == "reference_style_plus_lora_plus_attention_lora_plus_aux":
        include_keywords = [
            "reference_style_encoder",
            "reference_style_to_model",
            "reference_style_gate",
            "v66_f0_head",
            "v66_speaker_head",
            ".lora_A.",
            ".lora_B.",
        ] + list(attention_lora_keywords)
        unfrozen = set_named_parameters_trainable(model, include_keywords=include_keywords)
        return _build_partial_scope_report(model, scope=scope, include_keywords=include_keywords, unfrozen=unfrozen)

    raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")


def set_frozen_modules_eval_for_partial_scope(model: torch.nn.Module) -> None:
    """
    Keep frozen paths deterministic during partial fine-tuning.

    Important:
        The training loop calls model.train() every epoch.
        This helper should be called after model.train().

    Policy:
        - modules containing trainable parameters stay in train mode;
        - modules with no trainable parameters are set to eval mode.

    This avoids dropout noise in frozen content/semantic/text/coarse paths,
    which is important when adapting with very small few-shot datasets.
    """
    for module in model.modules():
        has_trainable = any(
            p.requires_grad
            for p in module.parameters(recurse=True)
        )

        if has_trainable:
            module.train()
        else:
            module.eval()


def set_frozen_modules_eval_for_residual_only(model: torch.nn.Module) -> None:
    """
    Backward-compatible wrapper for older residual-only experiments.
    """
    set_frozen_modules_eval_for_partial_scope(model)

def build_dataloader(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    semantic_source_mode: str = "oracle",
    semantic_oracle_prob: float = 0.5,
    include_continuous_semantic: bool = False,
    require_continuous_semantic: bool = False,
    continuous_semantic_dim: int = 768,
    include_v66_style_fields: bool = True,
    require_v66_style_fields: bool = False,
) -> tuple[Stage2FMDataset, DataLoader]:
    """
    EN:
    Build one dataset + dataloader pair.

    ZH:
    构建一个 dataset + dataloader 对。
    """
    require_predicted = str(semantic_source_mode).lower() in {"predicted", "mixed"}

    dataset = Stage2FMDataset(
        data_root,
        require_stage1_pred_semantic=require_predicted,
        require_v66_style_fields=bool(require_v66_style_fields),
    )

    collate_fn = create_stage2_fm_collate_fn(
        semantic_source_mode=semantic_source_mode,
        semantic_oracle_prob=semantic_oracle_prob,
        include_continuous_semantic=bool(include_continuous_semantic),
        require_continuous_semantic=bool(require_continuous_semantic),
        continuous_semantic_dim=int(continuous_semantic_dim),
        include_v66_style_fields=bool(include_v66_style_fields),
        require_v66_style_fields=bool(require_v66_style_fields),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return dataset, loader


def build_optional_val_loaders(
    *,
    args: argparse.Namespace,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_semantic_source_mode: str,
) -> tuple[dict[str, Stage2FMDataset], dict[str, DataLoader]]:
    """
    Build optional extra validation loaders.

    EN:
    These validation sets are for monitoring only.
    best_model.pt is still selected using the primary --val_data_root.

    ZH:
    构建额外验证集 dataloader。
    这些验证集只用于监控，不参与 best_model.pt 选择。
    """
    extra_val_roots: dict[str, str | None] = {
        "val_seen": getattr(args, "val_seen_data_root", None),
        "val_unseen": getattr(args, "val_unseen_data_root", None),
    }

    extra_val_datasets: dict[str, Stage2FMDataset] = {}
    extra_val_loaders: dict[str, DataLoader] = {}

    for name, root in extra_val_roots.items():
        if root is None:
            continue

        root_str = str(root).strip()
        if not root_str:
            continue

        dataset, loader = build_dataloader(
            data_root=root_str,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            semantic_source_mode=val_semantic_source_mode,
            semantic_oracle_prob=float(args.val_semantic_oracle_prob),
            include_continuous_semantic=bool(args.include_continuous_semantic),
            require_continuous_semantic=bool(args.require_continuous_semantic),
            continuous_semantic_dim=int(args.continuous_semantic_dim),
        )

        extra_val_datasets[name] = dataset
        extra_val_loaders[name] = loader

    return extra_val_datasets, extra_val_loaders


def run_validation(
    model,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    *,
    collect_semantic_source_stats: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """
    EN:
    Run validation.

    v6.4:
    Optionally collect semantic source stats over validation batches.

    ZH:
    执行验证。

    v6.4:
    可选地统计验证集 batch 的 semantic source 信息。
    """
    model.eval()

    total_loss = 0.0
    num_steps = 0

    source_running: dict[str, Any] = {}

    with torch.inference_mode():
        for batch in val_loader:
            batch = move_stage2_inputs_to_device(batch, device)

            if collect_semantic_source_stats:
                batch_source_stats = summarize_semantic_source_batch(batch)
                update_semantic_source_running_stats(source_running, batch_source_stats)

            with make_amp_context(device=device, use_amp=use_amp):
                loss, _ = model.compute_flow_matching_loss(batch)

            total_loss += float(loss.item())
            num_steps += 1

    if num_steps == 0:
        val_loss = math.nan
    else:
        val_loss = total_loss / num_steps

    if collect_semantic_source_stats:
        return val_loss, finalize_semantic_source_running_stats(source_running)

    return val_loss


def run_extra_validations(
    *,
    model,
    extra_val_loaders: dict[str, DataLoader],
    device: torch.device,
    use_amp: bool,
    collect_semantic_source_stats: bool,
) -> dict[str, dict[str, Any]]:
    """
    Run extra validation loaders.

    Return:
        {
          "val_seen": {
              "loss": ...,
              "semantic_source_stats": {...},
          },
          "val_unseen": {
              "loss": ...,
              "semantic_source_stats": {...},
          },
        }
    """
    results: dict[str, dict[str, Any]] = {}

    for name, loader in extra_val_loaders.items():
        if collect_semantic_source_stats:
            loss, source_stats = run_validation(
                model=model,
                val_loader=loader,
                device=device,
                use_amp=use_amp,
                collect_semantic_source_stats=True,
            )
        else:
            loss = run_validation(
                model=model,
                val_loader=loader,
                device=device,
                use_amp=use_amp,
                collect_semantic_source_stats=False,
            )
            source_stats = {}

        results[name] = {
            "loss": float(loss),
            "semantic_source_stats": source_stats,
            "num_batches": int(len(loader)),
        }

    return results


def load_history_for_resume(save_dir: Path, resume_ckpt_path: Path) -> list[dict[str, Any]]:
    """
    EN:
    Try to load existing history for resume.

    Priority:
    1. save_dir/history.json
    2. resume_ckpt_path.parent/history.json

    ZH:
    尝试为 resume 模式加载已有训练历史。

    优先级：
    1. save_dir/history.json
    2. resume_ckpt_path.parent/history.json
    """
    candidates = [
        save_dir / "history.json",
        resume_ckpt_path.parent / "history.json",
    ]

    for p in candidates:
        data = load_json_if_exists(p)
        if data and isinstance(data, dict) and isinstance(data.get("history"), list):
            return list(data["history"])

    return []


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_epoch: int,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    training_mode: str,
    source_checkpoint: str | None = None,
) -> None:
    """
    EN:
    Save one training checkpoint.

    ZH:
    保存一个训练 checkpoint。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "model_config": model_config,
        "training_mode": training_mode,
        "source_checkpoint": source_checkpoint,
        "training_meta": {
            "experiment_name": args.experiment_name,
            "train_data_root": str(Path(args.train_data_root).resolve()),
            "val_data_root": str(Path(args.val_data_root).resolve()),
            "save_dir": str(Path(args.save_dir).resolve()),
        },
    }

    if scaler is not None:
        try:
            payload["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            pass

    torch.save(payload, path)


def resolve_training_mode(args: argparse.Namespace) -> str:
    """
    EN:
    Resolve training mode from CLI arguments.

    ZH:
    根据命令行参数解析训练模式。
    """
    if args.resume_from and args.init_from:
        raise ValueError("--resume_from and --init_from are mutually exclusive; only one may be used.")

    if args.resume_from:
        return "resume_training"

    if args.init_from:
        return "initialize_from_checkpoint"

    return "train_from_scratch"


def print_mode_banner(
    training_mode: str,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    start_epoch: int,
    global_step: int,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    """
    EN:
    Print mode-specific startup summary.

    ZH:
    打印与训练模式对应的启动信息。
    """
    print("==== Stage2 FM training with validation started ====")
    print(f"mode            : {training_mode}")
    print(f"experiment_name : {args.experiment_name}")
    print(f"model_version   : {model_config.get('model_version')}")
    print(f"train_data_root : {Path(args.train_data_root).resolve()}")
    print(f"val_data_root   : {Path(args.val_data_root).resolve()}")
    print(f"device          : {args.device}")
    print(f"epochs          : {args.epochs}")
    print(f"batch_size      : {args.batch_size}")
    print(f"use_amp         : {args.use_amp}")
    print(f"save_dir        : {Path(args.save_dir).resolve()}")

    if args.resume_from:
        print(f"resume_from     : {Path(args.resume_from).resolve()}")
    if args.init_from:
        print(f"init_from       : {Path(args.init_from).resolve()}")

    print(f"start_epoch     : {start_epoch}")
    print(f"global_step     : {global_step}")
    print(f"best_val_loss   : {best_val_loss}")
    print(f"best_epoch      : {best_epoch}")
    print("model_config    :", json.dumps(model_config, ensure_ascii=False))

    print(
        "continuous_semantic:",
        {
            "include_continuous_semantic": bool(
                getattr(args, "include_continuous_semantic", False)
            ),
            "require_continuous_semantic": bool(
                getattr(args, "require_continuous_semantic", False)
            ),
            "use_continuous_semantic": bool(
                model_config.get("use_continuous_semantic", False)
            ),
            "continuous_semantic_dim": int(
                model_config.get("continuous_semantic_dim", 768)
            ),
            "fusion_mode": str(
                model_config.get("continuous_semantic_fusion_mode", "gated_add")
            ),
            "gate_init": float(
                model_config.get("continuous_semantic_gate_init", -2.0)
            ),
        },
    )

    try:
        ratio = float(model_config["acoustic_rate_hz"]) / float(model_config["semantic_rate_hz"])
        print(f"base_length_ratio: {ratio:.6f}")
    except Exception:
        pass

    print("====================================================")

def _jsonl_append(obj: dict[str, Any], path: Path) -> None:
    """
    EN:
    Append one JSON object as one line.

    ZH:
    以 JSONL 格式追加写入一条记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    EN:
    Convert scalar-like values to float safely.

    ZH:
    将标量形式安全转换为 float。
    """
    try:
        if torch.is_tensor(x):
            if x.numel() <= 0:
                return float(default)
            return float(x.detach().float().mean().item())
        return float(x)
    except Exception:
        return float(default)


def _safe_tensor_to_list(x: Any, max_items: int = 16) -> list[Any]:
    """
    EN:
    Convert a small tensor/list into a Python list for logging.

    ZH:
    将小 tensor/list 转成 Python list，便于日志记录。
    """
    if x is None:
        return []

    if torch.is_tensor(x):
        y = x.detach().cpu().view(-1)
        if y.numel() > max_items:
            y = y[:max_items]
        return y.tolist()

    if isinstance(x, (list, tuple)):
        return list(x[:max_items])

    return [x]


def _masked_positive_values(x: torch.Tensor) -> torch.Tensor:
    """
    EN:
    Return values greater than 0 from a tensor.

    ZH:
    返回 tensor 中大于 0 的有效值。
    """
    if not torch.is_tensor(x):
        return torch.empty(0)

    y = x.detach().float().view(-1).cpu()
    return y[y > 0]


def summarize_semantic_source_batch(batch) -> dict[str, Any]:
    """
    EN:
    Summarize v6.4 semantic source metadata for one batch.

    This function is intentionally read-only and does not affect training.
    It supports both old batches and v6.4 source-aware batches.

    ZH:
    汇总一个 batch 的 v6.4 semantic source 信息。

    该函数只读，不影响训练。
    同时兼容旧 batch 和 v6.4 source-aware batch。
    """
    result: dict[str, Any] = {}

    # ------------------------------------------------------------
    # Batch size and selected semantic length
    # ------------------------------------------------------------
    if hasattr(batch, "semantic_tokens") and torch.is_tensor(batch.semantic_tokens):
        result["batch_size"] = int(batch.semantic_tokens.shape[0])
        result["selected_semantic_max_len"] = int(batch.semantic_tokens.shape[1])
    else:
        result["batch_size"] = 0
        result["selected_semantic_max_len"] = 0

    if getattr(batch, "semantic_lengths", None) is not None:
        sem_lens = batch.semantic_lengths.detach().cpu().long().view(-1)
        result["selected_semantic_len_mean"] = float(sem_lens.float().mean().item())
        result["selected_semantic_len_min"] = int(sem_lens.min().item())
        result["selected_semantic_len_max"] = int(sem_lens.max().item())
        result["selected_semantic_lengths_preview"] = sem_lens[:16].tolist()
    else:
        result["selected_semantic_len_mean"] = -1.0
        result["selected_semantic_len_min"] = -1
        result["selected_semantic_len_max"] = -1
        result["selected_semantic_lengths_preview"] = []

    # ------------------------------------------------------------
    # Source ids
    # 0 = oracle, 1 = predicted, 2 = noised_oracle, 3 = aligned_predicted
    # ------------------------------------------------------------
    source_ids = getattr(batch, "semantic_source_ids", None)

    if torch.is_tensor(source_ids):
        ids = source_ids.detach().cpu().long().view(-1)
        result["semantic_source_ids_preview"] = ids[:16].tolist()
        result["semantic_source_oracle_count"] = int((ids == 0).sum().item())
        result["semantic_source_predicted_count"] = int((ids == 1).sum().item())
        result["semantic_source_noised_oracle_count"] = int((ids == 2).sum().item())
        result["semantic_source_aligned_predicted_count"] = int((ids == 3).sum().item())

        batch_size = max(int(ids.numel()), 1)
        result["semantic_source_oracle_frac"] = (
            float(result["semantic_source_oracle_count"]) / float(batch_size)
        )
        result["semantic_source_predicted_frac"] = (
            float(result["semantic_source_predicted_count"]) / float(batch_size)
        )
    else:
        result["semantic_source_ids_preview"] = []
        result["semantic_source_oracle_count"] = -1
        result["semantic_source_predicted_count"] = -1
        result["semantic_source_noised_oracle_count"] = -1
        result["semantic_source_aligned_predicted_count"] = -1
        result["semantic_source_oracle_frac"] = -1.0
        result["semantic_source_predicted_frac"] = -1.0

    # ------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------
    reliability = getattr(batch, "semantic_reliability", None)

    if torch.is_tensor(reliability):
        r = reliability.detach().cpu().float().view(-1)
        result["semantic_reliability_mean"] = float(r.mean().item())
        result["semantic_reliability_min"] = float(r.min().item())
        result["semantic_reliability_max"] = float(r.max().item())
        result["semantic_reliability_preview"] = r[:16].tolist()
    else:
        result["semantic_reliability_mean"] = -1.0
        result["semantic_reliability_min"] = -1.0
        result["semantic_reliability_max"] = -1.0
        result["semantic_reliability_preview"] = []

    # ------------------------------------------------------------
    # Oracle / predicted semantic lengths and pred/oracle ratio
    # ------------------------------------------------------------
    oracle_lengths = getattr(batch, "oracle_semantic_lengths", None)
    pred_lengths = getattr(batch, "pred_semantic_lengths", None)

    if torch.is_tensor(oracle_lengths):
        o = oracle_lengths.detach().cpu().long().view(-1)
        result["oracle_semantic_len_mean"] = float(o.float().mean().item())
        result["oracle_semantic_len_min"] = int(o.min().item())
        result["oracle_semantic_len_max"] = int(o.max().item())
        result["oracle_semantic_lengths_preview"] = o[:16].tolist()
    else:
        o = None
        result["oracle_semantic_len_mean"] = -1.0
        result["oracle_semantic_len_min"] = -1
        result["oracle_semantic_len_max"] = -1
        result["oracle_semantic_lengths_preview"] = []

    if torch.is_tensor(pred_lengths):
        p = pred_lengths.detach().cpu().long().view(-1)
        result["pred_semantic_len_mean"] = float(p.float().mean().item())
        result["pred_semantic_len_min"] = int(p.min().item())
        result["pred_semantic_len_max"] = int(p.max().item())
        result["pred_semantic_lengths_preview"] = p[:16].tolist()
    else:
        p = None
        result["pred_semantic_len_mean"] = -1.0
        result["pred_semantic_len_min"] = -1
        result["pred_semantic_len_max"] = -1
        result["pred_semantic_lengths_preview"] = []

    if o is not None and p is not None:
        valid = (o > 0) & (p > 0)
        if bool(valid.any().item()):
            ratio = p[valid].float() / o[valid].float().clamp_min(1.0)
            result["pred_over_oracle_len_ratio_mean"] = float(ratio.mean().item())
            result["pred_over_oracle_len_ratio_min"] = float(ratio.min().item())
            result["pred_over_oracle_len_ratio_max"] = float(ratio.max().item())
            result["pred_over_oracle_len_ratio_preview"] = ratio[:16].tolist()
        else:
            result["pred_over_oracle_len_ratio_mean"] = -1.0
            result["pred_over_oracle_len_ratio_min"] = -1.0
            result["pred_over_oracle_len_ratio_max"] = -1.0
            result["pred_over_oracle_len_ratio_preview"] = []
    else:
        result["pred_over_oracle_len_ratio_mean"] = -1.0
        result["pred_over_oracle_len_ratio_min"] = -1.0
        result["pred_over_oracle_len_ratio_max"] = -1.0
        result["pred_over_oracle_len_ratio_preview"] = []

    # ------------------------------------------------------------
    # Also preserve collate extras if available.
    # ------------------------------------------------------------
    extras = getattr(batch, "extras", None)
    if isinstance(extras, dict):
        result["extras_semantic_source_counts"] = extras.get("semantic_source_counts", None)
        result["extras_pred_over_oracle_length_ratio_mean"] = extras.get(
            "pred_over_oracle_length_ratio_mean",
            None,
        )
        result["extras_semantic_reliability_mean"] = extras.get(
            "semantic_reliability_mean",
            None,
        )

    return result

def _masked_sequence_norm_stats(
    x: torch.Tensor,
    lengths: torch.Tensor | None,
) -> dict[str, float]:
    """
    EN:
    Compute norm stats for valid sequence positions only.

    Args:
        x:
            (B, T, C)
        lengths:
            (B,)

    ZH:
    只统计有效序列位置上的向量 norm。
    """
    if not torch.is_tensor(x) or x.ndim != 3:
        return {
            "norm_mean": -1.0,
            "norm_std": -1.0,
            "norm_min": -1.0,
            "norm_max": -1.0,
        }

    y = x.detach().float().cpu()
    B, T, _ = y.shape

    if lengths is None or not torch.is_tensor(lengths):
        lengths_cpu = torch.full(
            size=(B,),
            fill_value=T,
            dtype=torch.long,
        )
    else:
        lengths_cpu = lengths.detach().cpu().long().view(-1).clamp(
            min=0,
            max=T,
        )

    idx = torch.arange(T).unsqueeze(0)
    mask = idx < lengths_cpu.unsqueeze(1)

    norms = torch.linalg.norm(y, dim=-1)

    if bool(mask.any().item()):
        valid_norms = norms[mask]
    else:
        valid_norms = norms.reshape(-1)

    if valid_norms.numel() <= 0:
        return {
            "norm_mean": -1.0,
            "norm_std": -1.0,
            "norm_min": -1.0,
            "norm_max": -1.0,
        }

    return {
        "norm_mean": float(valid_norms.mean().item()),
        "norm_std": float(valid_norms.std().item()) if valid_norms.numel() > 1 else 0.0,
        "norm_min": float(valid_norms.min().item()),
        "norm_max": float(valid_norms.max().item()),
    }


def summarize_continuous_semantic_batch(batch) -> dict[str, Any]:
    """
    EN:
    Summarize v6.4.1 continuous semantic cache carried by Stage2Inputs.

    ZH:
    汇总 Stage2Inputs 中携带的 v6.4.1 continuous semantic cache 信息。
    """
    result: dict[str, Any] = {}

    selected = getattr(batch, "semantic_continuous", None)
    selected_lengths = getattr(batch, "semantic_continuous_lengths", None)

    oracle = getattr(batch, "oracle_semantic_continuous", None)
    oracle_lengths = getattr(batch, "oracle_semantic_continuous_lengths", None)

    pred = getattr(batch, "pred_semantic_continuous", None)
    pred_lengths = getattr(batch, "pred_semantic_continuous_lengths", None)

    # ------------------------------------------------------------
    # Selected continuous semantic
    # ------------------------------------------------------------
    if torch.is_tensor(selected):
        result["continuous_semantic_has_selected"] = 1
        result["continuous_semantic_selected_shape_b"] = int(selected.shape[0])
        result["continuous_semantic_selected_shape_t"] = int(selected.shape[1])
        result["continuous_semantic_selected_shape_c"] = int(selected.shape[2])

        if torch.is_tensor(selected_lengths):
            lens = selected_lengths.detach().cpu().long().view(-1)
            result["continuous_semantic_len_mean"] = float(lens.float().mean().item())
            result["continuous_semantic_len_min"] = int(lens.min().item())
            result["continuous_semantic_len_max"] = int(lens.max().item())
            result["continuous_semantic_lengths_preview"] = lens[:16].tolist()
        else:
            result["continuous_semantic_len_mean"] = -1.0
            result["continuous_semantic_len_min"] = -1
            result["continuous_semantic_len_max"] = -1
            result["continuous_semantic_lengths_preview"] = []

        stats = _masked_sequence_norm_stats(selected, selected_lengths)
        result["continuous_semantic_norm_mean"] = stats["norm_mean"]
        result["continuous_semantic_norm_std"] = stats["norm_std"]
        result["continuous_semantic_norm_min"] = stats["norm_min"]
        result["continuous_semantic_norm_max"] = stats["norm_max"]

    else:
        result["continuous_semantic_has_selected"] = 0
        result["continuous_semantic_selected_shape_b"] = 0
        result["continuous_semantic_selected_shape_t"] = 0
        result["continuous_semantic_selected_shape_c"] = 0
        result["continuous_semantic_len_mean"] = -1.0
        result["continuous_semantic_len_min"] = -1
        result["continuous_semantic_len_max"] = -1
        result["continuous_semantic_lengths_preview"] = []
        result["continuous_semantic_norm_mean"] = -1.0
        result["continuous_semantic_norm_std"] = -1.0
        result["continuous_semantic_norm_min"] = -1.0
        result["continuous_semantic_norm_max"] = -1.0

    # ------------------------------------------------------------
    # Oracle continuous semantic
    # ------------------------------------------------------------
    if torch.is_tensor(oracle):
        result["continuous_semantic_has_oracle"] = 1
        result["oracle_continuous_shape_t"] = int(oracle.shape[1])
        result["oracle_continuous_shape_c"] = int(oracle.shape[2])

        if torch.is_tensor(oracle_lengths):
            lens = oracle_lengths.detach().cpu().long().view(-1)
            result["oracle_continuous_len_mean"] = float(lens.float().mean().item())
        else:
            result["oracle_continuous_len_mean"] = -1.0
    else:
        result["continuous_semantic_has_oracle"] = 0
        result["oracle_continuous_shape_t"] = 0
        result["oracle_continuous_shape_c"] = 0
        result["oracle_continuous_len_mean"] = -1.0

    # ------------------------------------------------------------
    # Predicted continuous semantic
    # ------------------------------------------------------------
    if torch.is_tensor(pred):
        result["continuous_semantic_has_predicted"] = 1
        result["pred_continuous_shape_t"] = int(pred.shape[1])
        result["pred_continuous_shape_c"] = int(pred.shape[2])

        if torch.is_tensor(pred_lengths):
            lens = pred_lengths.detach().cpu().long().view(-1)
            result["pred_continuous_len_mean"] = float(lens.float().mean().item())
        else:
            result["pred_continuous_len_mean"] = -1.0
    else:
        result["continuous_semantic_has_predicted"] = 0
        result["pred_continuous_shape_t"] = 0
        result["pred_continuous_shape_c"] = 0
        result["pred_continuous_len_mean"] = -1.0

    return result

def extract_continuous_semantic_encoder_stats(model) -> dict[str, Any]:
    """
    EN:
    Read continuous semantic fusion stats from model.condition_encoder.

    ZH:
    从 model.condition_encoder 中读取 continuous semantic fusion 统计。
    """
    encoder = getattr(model, "condition_encoder", None)

    # Compatibility with DataParallel-like wrappers.
    # 兼容可能的 DataParallel-like wrapper。
    if encoder is None and hasattr(model, "module"):
        encoder = getattr(model.module, "condition_encoder", None)

    if encoder is None:
        return {
            "continuous_semantic_encoder_has_stats": 0,
            "continuous_semantic_encoder_used": -1,
            "continuous_semantic_encoder_gate": -1.0,
            "continuous_semantic_encoder_norm_mean": -1.0,
        }

    stats = getattr(encoder, "_last_continuous_semantic_stats", None)

    if not isinstance(stats, dict):
        return {
            "continuous_semantic_encoder_has_stats": 0,
            "continuous_semantic_encoder_used": -1,
            "continuous_semantic_encoder_gate": -1.0,
            "continuous_semantic_encoder_norm_mean": -1.0,
        }

    return {
        "continuous_semantic_encoder_has_stats": 1,
        "continuous_semantic_encoder_used": int(stats.get("used", 0)),
        "continuous_semantic_encoder_gate": float(stats.get("gate", -1.0)),
        "continuous_semantic_encoder_norm_mean": float(
            stats.get("continuous_norm_mean", -1.0)
        ),
        "continuous_semantic_encoder_len_mean": float(
            stats.get("continuous_len_mean", -1.0)
        ),
    }



# ============================================================
# v6.6.3 auxiliary loss logging helpers
# v6.6.3 辅助损失日志辅助函数
# ============================================================
V663_AUX_LOG_KEYS: list[str] = [
    # v6.6.2 total auxiliary loss
    "v66_aux_total_loss",

    # v6.6.2 energy loss
    "v66_energy_available",
    "v66_energy_loss",
    "v66_energy_weighted_loss",
    "v66_energy_pred_mean",
    "v66_energy_target_mean",

    # v6.6.2 F0 loss
    "v66_f0_available",
    "v66_f0_loss",
    "v66_f0_weighted_loss",
    "v66_f0_voiced_ratio",
    "v66_f0_pred_mean",
    "v66_f0_target_mean",

    # v6.6.2 speaker loss
    "v66_speaker_available",
    "v66_speaker_loss",
    "v66_speaker_weighted_loss",
    "v66_speaker_cosine_mean",
    "v66_speaker_pred_norm_mean",
    "v66_speaker_target_norm_mean",

    # v6.6 reference style stats
    "reference_style_enabled",
    "reference_style_used",
    "reference_style_gate",
    "reference_style_ref_style_norm_mean",
    "reference_style_fused_style_norm_mean",
    "reference_style_prompt_acoustic_len_mean",


    # v6.6.4 Speaker Identity Adapter
    "v664_speaker_identity_enabled",
    "v664_speaker_identity_available",
    "v664_speaker_identity_used",
    "v664_speaker_identity_source_id",
    "v664_speaker_identity_gate",
    "v664_speaker_identity_speaker_embedding_norm_mean",
    "v664_speaker_identity_speaker_style_norm_mean",
    "v664_speaker_identity_input_style_norm_mean",
    "v664_speaker_identity_fused_style_norm_mean",

    # v6.6.3-A Attention LoRA static stats
    "bootstrapper_attention_lora_enabled",
    "bootstrapper_attention_lora_target_id",
    "bootstrapper_attention_lora_q_enabled",
    "bootstrapper_attention_lora_k_enabled",
    "bootstrapper_attention_lora_v_enabled",
    "bootstrapper_attention_lora_o_enabled",
    "bootstrapper_attention_lora_gate_raw_mean",
    "bootstrapper_attention_lora_gate_sigmoid_mean",
    "bootstrapper_attention_lora_gate_sigmoid_min",
    "bootstrapper_attention_lora_gate_sigmoid_max",
    "bootstrapper_attention_lora_trainable_params",
    "bootstrapper_attention_lora_num_replaced",
    # v6.6.3 LoRA-related coarse path stats, if emitted by model/bootstrapper.
    "bootstrapper_lora_trainable_params",
    "bootstrapper_lora_num_replaced",
]


def update_v663_aux_running_stats(
    running: dict[str, Any],
    aux: dict[str, Any],
) -> None:
    """
    EN:
    Accumulate per-step v6.6.2/v6.6.3 auxiliary metrics for epoch-level history.

    ZH:
    累积每个 step 的 v6.6.2/v6.6.3 辅助指标，用于写入 epoch 级 history。
    """
    if not isinstance(aux, dict):
        return

    running["num_logged_steps"] = int(running.get("num_logged_steps", 0)) + 1

    for key in V663_AUX_LOG_KEYS:
        if key not in aux:
            continue

        value = _safe_float(aux.get(key), default=float("nan"))
        if not math.isfinite(value):
            continue

        sum_key = f"{key}_sum"
        count_key = f"{key}_count"
        last_key = f"{key}_last"

        running[sum_key] = float(running.get(sum_key, 0.0)) + float(value)
        running[count_key] = int(running.get(count_key, 0)) + 1
        running[last_key] = float(value)


def finalize_v663_aux_running_stats(running: dict[str, Any]) -> dict[str, Any]:
    """
    EN:
    Finalize epoch-level v6.6.2/v6.6.3 auxiliary metric means and last values.

    ZH:
    汇总 epoch 级 v6.6.2/v6.6.3 辅助指标均值与最后一次记录值。
    """
    out: dict[str, Any] = {
        "num_logged_steps": int(running.get("num_logged_steps", 0)),
    }

    for key in V663_AUX_LOG_KEYS:
        count = int(running.get(f"{key}_count", 0))
        if count <= 0:
            continue

        out[f"{key}_mean"] = float(running.get(f"{key}_sum", 0.0)) / float(count)
        out[f"{key}_last"] = float(running.get(f"{key}_last", 0.0))
        out[f"{key}_count"] = int(count)

    return out
def format_semantic_source_summary(stats: dict[str, Any]) -> str:
    """
    EN:
    Convert semantic source stats into a compact one-line string.

    ZH:
    将 semantic source 统计转换成紧凑的一行日志。
    """
    if not stats:
        return ""

    parts = [
        f"src_o={stats.get('semantic_source_oracle_count', -1)}",
        f"src_p={stats.get('semantic_source_predicted_count', -1)}",
        f"src_p_frac={stats.get('semantic_source_predicted_frac', -1.0):.3f}",
        f"rel_mean={stats.get('semantic_reliability_mean', -1.0):.4f}",
        f"rel_min={stats.get('semantic_reliability_min', -1.0):.4f}",
        f"rel_max={stats.get('semantic_reliability_max', -1.0):.4f}",
        f"sem_len_mean={stats.get('selected_semantic_len_mean', -1.0):.2f}",
        f"pred_oracle_ratio={stats.get('pred_over_oracle_len_ratio_mean', -1.0):.4f}",
    ]

    return " ".join(parts)


def update_semantic_source_running_stats(
    running: dict[str, Any],
    batch_stats: dict[str, Any],
) -> None:
    """
    EN:
    Accumulate epoch-level semantic source statistics.

    ZH:
    累积 epoch 级别的 semantic source 统计。
    """
    if not batch_stats:
        return

    running["num_batches"] = int(running.get("num_batches", 0)) + 1
    running["num_samples"] = int(running.get("num_samples", 0)) + int(
        batch_stats.get("batch_size", 0)
    )

    for key in [
        "semantic_source_oracle_count",
        "semantic_source_predicted_count",
        "semantic_source_noised_oracle_count",
        "semantic_source_aligned_predicted_count",
    ]:
        running[key] = int(running.get(key, 0)) + int(batch_stats.get(key, 0))

    weighted_keys = [
        "semantic_reliability_mean",
        "selected_semantic_len_mean",
        "oracle_semantic_len_mean",
        "pred_semantic_len_mean",
        "pred_over_oracle_len_ratio_mean",
    ]

    bs = max(int(batch_stats.get("batch_size", 0)), 0)

    for key in weighted_keys:
        value = float(batch_stats.get(key, -1.0))
        if value < 0 or bs <= 0:
            continue

        sum_key = f"{key}_weighted_sum"
        weight_key = f"{key}_weight"

        running[sum_key] = float(running.get(sum_key, 0.0)) + value * float(bs)
        running[weight_key] = int(running.get(weight_key, 0)) + int(bs)


def finalize_semantic_source_running_stats(running: dict[str, Any]) -> dict[str, Any]:
    """
    EN:
    Finalize accumulated epoch-level semantic source statistics.

    ZH:
    汇总 epoch 级别的 semantic source 统计。
    """
    out: dict[str, Any] = {
        "num_batches": int(running.get("num_batches", 0)),
        "num_samples": int(running.get("num_samples", 0)),
        "semantic_source_oracle_count": int(
            running.get("semantic_source_oracle_count", 0)
        ),
        "semantic_source_predicted_count": int(
            running.get("semantic_source_predicted_count", 0)
        ),
        "semantic_source_noised_oracle_count": int(
            running.get("semantic_source_noised_oracle_count", 0)
        ),
        "semantic_source_aligned_predicted_count": int(
            running.get("semantic_source_aligned_predicted_count", 0)
        ),
    }

    num_samples = max(int(out["num_samples"]), 1)
    out["semantic_source_oracle_frac"] = (
        float(out["semantic_source_oracle_count"]) / float(num_samples)
    )
    out["semantic_source_predicted_frac"] = (
        float(out["semantic_source_predicted_count"]) / float(num_samples)
    )

    for key in [
        "semantic_reliability_mean",
        "selected_semantic_len_mean",
        "oracle_semantic_len_mean",
        "pred_semantic_len_mean",
        "pred_over_oracle_len_ratio_mean",
    ]:
        sum_key = f"{key}_weighted_sum"
        weight_key = f"{key}_weight"

        weight = int(running.get(weight_key, 0))
        if weight > 0:
            out[key] = float(running.get(sum_key, 0.0)) / float(weight)
        else:
            out[key] = -1.0

    return out

def _aux_to_log_string(aux: dict[str, Any]) -> str:
    """
    EN:
    Convert selected aux metrics to a compact log string.

    ZH:
    将部分 aux 指标转为紧凑日志字符串。
    """
    ordered_keys = [
        "t_mean",

        "fm_loss",
        "recon_coarse_loss",
        "recon_refined_loss",
        "delta_loss",
        "delta2_loss",

        "mid_content_aux_loss",
        "mid_content_acc",
        "final_content_aux_loss",
        "final_content_acc",

        "span_flow_loss",
        "span_recon_loss",
        "span_content_loss",
        "span_content_acc",

        # v6.2 bootstrapper losses
        "v62_base_loss",
        "coarse_mel_l1_loss",
        "coarse_mel_mse_loss",
        "coarse_mel_delta_l1_loss",
        "coarse_mel_total_loss",
        "bridge_recon_loss",
        "bridge_delta_loss",
        "bridge_total_loss",
        "bridge_t_start",
        "bridge_noise_temperature",

        # v6.3 losses
        "v63_base_loss",
        "v63_coarse_total_loss",
        "v63_residual_total_loss",
        "coarse_mel_delta2_l1_loss",

        "residual_refiner_total_loss",
        "residual_flow_mse_loss",
        "residual_flow_l1_loss",
        "residual_recon_l1_loss",
        "final_recon_l1_loss",
        "final_delta_l1_loss",
        "final_delta2_l1_loss",

        "coarse_vs_target_l1_loss",
        "coarse_vs_target_delta_l1_loss",
        "coarse_vs_target_delta2_l1_loss",
        "final_minus_coarse_l1_improvement",
        "final_minus_coarse_delta_l1_improvement",

        "text_lengths_mean",
        "text_memory_is_native",
        "v63_use_native_text_memory",

        "t_min",
        "t_max",

        "length_ratio_used",
        "inferred_length_mean",
        "target_length_mean",
        "content_main_norm",
        "semantic_guide_norm",


        # v6.6.2 prosody / speaker auxiliary losses
        "v66_aux_total_loss",
        "v66_energy_available",
        "v66_energy_loss",
        "v66_energy_weighted_loss",
        "v66_energy_pred_mean",
        "v66_energy_target_mean",
        "v66_f0_available",
        "v66_f0_loss",
        "v66_f0_weighted_loss",
        "v66_f0_voiced_ratio",
        "v66_f0_pred_mean",
        "v66_f0_target_mean",
        "v66_speaker_available",
        "v66_speaker_loss",
        "v66_speaker_weighted_loss",
        "v66_speaker_cosine_mean",
        "v66_speaker_pred_norm_mean",
        "v66_speaker_target_norm_mean",

        # v6.6 reference style stats
        "reference_style_enabled",
        "reference_style_used",
        "reference_style_gate",
        "reference_style_ref_style_norm_mean",
        "reference_style_fused_style_norm_mean",
        "reference_style_prompt_acoustic_len_mean",


        # v6.6.4 Speaker Identity Adapter
        "v664_speaker_identity_enabled",
        "v664_speaker_identity_available",
        "v664_speaker_identity_used",
        "v664_speaker_identity_source_id",
        "v664_speaker_identity_gate",
        "v664_speaker_identity_speaker_embedding_norm_mean",
        "v664_speaker_identity_speaker_style_norm_mean",
        "v664_speaker_identity_input_style_norm_mean",
        "v664_speaker_identity_fused_style_norm_mean",


        # v6.6.3-A Attention LoRA diagnostics
        "bootstrapper_attention_lora_enabled",
        "bootstrapper_attention_lora_num_replaced",
        "bootstrapper_attention_lora_trainable_params",
        "bootstrapper_attention_lora_target_id",
        "bootstrapper_attention_lora_q_enabled",
        "bootstrapper_attention_lora_k_enabled",
        "bootstrapper_attention_lora_v_enabled",
        "bootstrapper_attention_lora_o_enabled",
        "bootstrapper_attention_lora_gate_raw_mean",
        "bootstrapper_attention_lora_gate_sigmoid_mean",
        "bootstrapper_attention_lora_gate_sigmoid_min",
        "bootstrapper_attention_lora_gate_sigmoid_max",

        # v6.4 semantic source logging
        "semantic_source_oracle_count",
        "semantic_source_predicted_count",
        "semantic_source_predicted_frac",
        "semantic_reliability_mean",
        "semantic_reliability_min",
        "semantic_reliability_max",
        "selected_semantic_len_mean",
        "oracle_semantic_len_mean",
        "pred_semantic_len_mean",
        "pred_over_oracle_len_ratio_mean",

        # v6.4.1 continuous semantic logging
        "continuous_semantic_has_selected",
        "continuous_semantic_len_mean",
        "continuous_semantic_norm_mean",
        "continuous_semantic_norm_std",
        "continuous_semantic_encoder_used",
        "continuous_semantic_encoder_gate",
        "continuous_semantic_encoder_norm_mean",
        "continuous_semantic_encoder_len_mean",
    ]

    parts: list[str] = []

    for k in ordered_keys:
        if k not in aux:
            continue

        v = aux[k]

        try:
            if torch.is_tensor(v):
                if v.numel() == 1:
                    v = float(v.detach().item())
                else:
                    continue
            else:
                v = float(v)

            if "acc" in k:
                parts.append(f"{k}={v:.4f}")
            elif k == "t_mean":
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v:.6f}")
        except Exception:
            continue

    return " ".join(parts)


# ============================================================
# Argument parsing
# 参数解析
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse command-line arguments for stage-2 Flow Matching training
    with validation, plus support for:
    - resume_from
    - init_from
    - v6.1 model version

    ZH:
    解析带验证集的第二阶段 Flow Matching 训练脚本参数，
    并支持：
    - resume_from
    - init_from
    - v6.1 模型版本
    """
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Dataset arguments
    # 数据集参数
    # --------------------------------------------------------
    parser.add_argument("--train_data_root", type=str, required=True)
    parser.add_argument(
        "--val_data_root",
        type=str,
        required=True,
        help=(
            "Primary validation dataset root. "
            "best_model.pt is selected according to this validation loss. "
            "For v6.5 100h, recommended: val_all."
        ),
    )
    parser.add_argument(
        "--val_seen_data_root",
        type=str,
        default=None,
        help=(
            "Optional extra validation root for seen speakers. "
            "Only used for logging/evaluation, not best checkpoint selection."
        ),
    )
    parser.add_argument(
        "--val_unseen_data_root",
        type=str,
        default=None,
        help=(
            "Optional extra validation root for unseen speakers. "
            "Only used for logging/evaluation, not best checkpoint selection."
        ),
    )
    parser.add_argument(
        "--semantic_source_mode",
        type=str,
        default="oracle",
        choices=["oracle", "predicted", "mixed"],
        help=(
            "Semantic source mode for training dataloader. "
            "oracle uses semantic_tokens; predicted uses stage1_pred_semantic_tokens; "
            "mixed samples oracle/predicted per sample."
        ),
    )
    parser.add_argument(
        "--semantic_oracle_prob",
        type=float,
        default=0.5,
        help=(
            "Probability of using oracle semantic when semantic_source_mode=mixed. "
            "Ignored by oracle/predicted modes."
        ),
    )
    parser.add_argument(
        "--val_semantic_source_mode",
        type=str,
        default="oracle",
        choices=["oracle", "predicted", "mixed", "same"],
        help=(
            "Semantic source mode for validation dataloader. "
            "Use 'same' to reuse --semantic_source_mode. "
            "Default oracle keeps validation deterministic."
        ),
    )
    parser.add_argument(
        "--val_semantic_oracle_prob",
        type=float,
        default=1.0,
        help=(
            "Probability of using oracle semantic for validation when "
            "val_semantic_source_mode=mixed."
        ),
    )

    # --------------------------------------------------------
    # v6.4.1 continuous semantic dataloader args
    # v6.4.1 continuous semantic 数据读取参数
    # --------------------------------------------------------
    parser.add_argument(
        "--include_continuous_semantic",
        type=str2bool,
        default=False,
        help=(
            "Whether dataloader should read continuous semantic cache fields "
            "from Stage2 .pt samples."
        ),
    )
    parser.add_argument(
        "--require_continuous_semantic",
        type=str2bool,
        default=False,
        help=(
            "If true, dataloader raises an error when continuous semantic cache "
            "is missing."
        ),
    )
    parser.add_argument(
        "--continuous_semantic_dim",
        type=int,
        default=768,
        help="Continuous semantic feature dimension. GPT-SoVITS v2 expects 768.",
    )

    # --------------------------------------------------------
    # v6.6 reference acoustic style dataloader/model args
    # v6.6 参考音频声学风格数据与模型参数
    # --------------------------------------------------------
    parser.add_argument("--include_v66_style_fields", type=str2bool, default=True)
    parser.add_argument("--require_v66_style_fields", type=str2bool, default=False)
    parser.add_argument("--use_reference_acoustic_style", type=str2bool, default=False)
    parser.add_argument("--reference_acoustic_dim", type=int, default=80)
    parser.add_argument("--reference_style_dim", type=int, default=256)
    parser.add_argument("--reference_style_hidden_dim", type=int, default=256)
    parser.add_argument("--reference_style_num_layers", type=int, default=4)
    parser.add_argument("--reference_style_kernel_size", type=int, default=5)
    parser.add_argument("--reference_style_dropout", type=float, default=0.1)
    parser.add_argument("--reference_style_fusion_mode", type=str, default="gated_add", choices=["gated_add", "none"])
    parser.add_argument("--reference_style_gate_init", type=float, default=-3.0)

    # --------------------------------------------------------
    # v6.6.2 prosody / speaker auxiliary loss args
    # v6.6.2 韵律 / 说话人辅助损失参数
    # --------------------------------------------------------
    parser.add_argument("--use_v66_energy_loss", type=str2bool, default=False)
    parser.add_argument("--v66_energy_loss_weight", type=float, default=0.15)
    parser.add_argument("--v66_energy_loss_type", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--v66_energy_normalize", type=str2bool, default=True)
    parser.add_argument("--use_v66_f0_loss", type=str2bool, default=False)
    parser.add_argument("--v66_f0_loss_weight", type=float, default=0.05)
    parser.add_argument("--v66_f0_loss_type", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--v66_f0_normalize", type=str2bool, default=True)
    parser.add_argument("--v66_f0_hidden_dim", type=int, default=128)
    parser.add_argument("--use_v66_speaker_loss", type=str2bool, default=False)
    parser.add_argument("--v66_speaker_loss_weight", type=float, default=0.05)
    parser.add_argument("--v66_speaker_embedding_dim", type=int, default=192)
    parser.add_argument("--v66_require_aux_targets", type=str2bool, default=False)


    # --------------------------------------------------------
    # v6.6.4 Speaker Identity Adapter args
    # v6.6.4 说话人身份适配器参数
    # --------------------------------------------------------
    parser.add_argument("--use_v664_speaker_identity_adapter", type=str2bool, default=False)
    parser.add_argument("--v664_speaker_identity_source", type=str, default="prompt_or_target", choices=["prompt", "target", "prompt_or_target", "target_or_prompt"])
    parser.add_argument("--v664_speaker_embedding_dim", type=int, default=192)
    parser.add_argument("--v664_speaker_adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--v664_speaker_adapter_num_layers", type=int, default=2)
    parser.add_argument("--v664_speaker_adapter_dropout", type=float, default=0.05)
    parser.add_argument("--v664_speaker_adapter_fusion_mode", type=str, default="gated_add", choices=["gated_add", "add", "film"])
    parser.add_argument("--v664_speaker_adapter_gate_init", type=float, default=-3.0)
    parser.add_argument("--v664_speaker_adapter_normalize", type=str2bool, default=True)
    # --------------------------------------------------------
    # v6.6.3 Bootstrapper LoRA args
    # v6.6.3 Bootstrapper LoRA 参数
    # --------------------------------------------------------
    parser.add_argument("--use_bootstrapper_lora", type=str2bool, default=False)
    parser.add_argument("--bootstrapper_lora_rank", type=int, default=4)
    parser.add_argument("--bootstrapper_lora_alpha", type=float, default=8.0)
    parser.add_argument("--bootstrapper_lora_dropout", type=float, default=0.05)
    parser.add_argument("--bootstrapper_lora_target", type=str, default="core", choices=["core", "all_linear"])
    parser.add_argument("--bootstrapper_lora_init_scale", type=float, default=0.01)

    # --------------------------------------------------------
    # v6.6.3-A Bootstrapper Attention LoRA args
    # v6.6.3-A Bootstrapper 注意力 LoRA 参数
    # --------------------------------------------------------
    parser.add_argument("--use_bootstrapper_attention_lora", type=str2bool, default=False)
    parser.add_argument("--bootstrapper_attention_lora_rank", type=int, default=4)
    parser.add_argument("--bootstrapper_attention_lora_alpha", type=float, default=8.0)
    parser.add_argument("--bootstrapper_attention_lora_dropout", type=float, default=0.05)
    parser.add_argument("--bootstrapper_attention_lora_target", type=str, default="cross_only", choices=["cross_only", "all_attention", "self_only", "semantic_cross", "text_cross"])
    parser.add_argument("--bootstrapper_attention_lora_gate_init", type=float, default=-3.0)
    parser.add_argument("--bootstrapper_attention_lora_enable_q", type=str2bool, default=True)
    parser.add_argument("--bootstrapper_attention_lora_enable_k", type=str2bool, default=True)
    parser.add_argument("--bootstrapper_attention_lora_enable_v", type=str2bool, default=True)
    parser.add_argument("--bootstrapper_attention_lora_enable_o", type=str2bool, default=True)

    # --------------------------------------------------------
    # Runtime arguments
    # 运行参数
    # --------------------------------------------------------
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")

    # --------------------------------------------------------
    # Optimizer arguments
    # 优化器参数
    # --------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # --------------------------------------------------------
    # Trainable scope / freezing
    # 训练范围 / 冻结策略
    # --------------------------------------------------------
    parser.add_argument(
        "--trainable_scope",
        type=str,
        default="all",
        choices=[
            "all",
            "residual_refiner_only",
            "style_condition_only",
            "style_adapter_only",
            "style_condition_plus_adapter",
            "reference_style_encoder_only",
            "reference_style_plus_adapter",
            "reference_style_plus_condition_adapter",
            "reference_style_plus_prosody",
            "reference_style_plus_all_aux",
            "bootstrapper_lora_only",
            "reference_style_plus_lora",
            "reference_style_plus_lora_plus_aux",
            "attention_lora_only",
            "reference_style_plus_attention_lora",
            "reference_style_plus_lora_plus_attention_lora_plus_aux",
            "speaker_identity_adapter_only",
            "speaker_identity_plus_lora",
            "speaker_identity_plus_lora_plus_aux",
        ],
        help=(
            "Which module scope is trainable. "
            "all: train all parameters. "
            "residual_refiner_only: freeze all modules except residual_refiner. "
            "style_condition_only: train prompt/style condition mapping only. "
            "style_adapter_only: train bootstrapper style injection adapters only. "
            "style_condition_plus_adapter: train both style condition mapping and style adapters."
        ),
    )
    parser.add_argument(
        "--keep_frozen_modules_eval",
        type=str2bool,
        default=True,
        help=(
            "When trainable_scope is not all, keep frozen modules in eval mode "
            "after model.train() to disable dropout in frozen condition/coarse paths."
        ),
    )

    # --------------------------------------------------------
    # Basic model arguments
    # 基础模型参数
    # --------------------------------------------------------
    parser.add_argument("--acoustic_dim", type=int, default=80)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--semantic_vocab_size", type=int, default=1024)
    parser.add_argument("--semantic_rate_hz", type=float, default=25.0)
    parser.add_argument("--acoustic_rate_hz", type=float, default=DEFAULT_ACOUSTIC_RATE_HZ)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model_version",
        type=str,
        default="v6_1",
        choices=["v1", "v2", "v5", "v6", "v6_1", "v6_2", "v6_3"],
    )


    # --------------------------------------------------------
    # Self condition
    # 自条件
    # --------------------------------------------------------
    parser.add_argument("--use_self_condition", action="store_true")
    parser.add_argument("--self_condition_prob", type=float, default=0.5)

    # --------------------------------------------------------
    # v4 old length predictor compatibility
    # v4 旧长度预测器兼容参数
    # --------------------------------------------------------
    parser.add_argument("--enable_length_predictor", action="store_true")
    parser.add_argument("--length_predictor_hidden_dim", type=int, default=256)
    parser.add_argument("--length_loss_weight", type=float, default=0.10)
    parser.add_argument("--length_teacher_forcing_prob", type=float, default=0.70)

    # --------------------------------------------------------
    # v5 / v6 condition args
    # v5 / v6 条件参数
    # --------------------------------------------------------
    parser.add_argument("--phoneme_vocab_size", type=int, default=4096)
    parser.add_argument("--phoneme_embed_dim", type=int, default=256)
    parser.add_argument("--bert_dim", type=int, default=1024)
    parser.add_argument("--content_dim", type=int, default=384)
    parser.add_argument("--content_refiner_layers", type=int, default=4)
    parser.add_argument("--content_frame_dim", type=int, default=384)
    parser.add_argument("--content_expander_layers", type=int, default=2)
    parser.add_argument("--style_dim", type=int, default=256)

    # --------------------------------------------------------
    # v6.1 content-main condition args
    # v6.1 content 主条件参数
    # --------------------------------------------------------
    parser.add_argument("--content_main_dim", type=int, default=512)
    parser.add_argument("--content_injection_gate_init", type=float, default=0.5)
    parser.add_argument("--content_refiner_dilations", type=str, default="1,2")
    parser.add_argument("--content_boundary_enhance", type=str2bool, default=True)
    parser.add_argument("--content_boundary_kernel_size", type=int, default=3)
    parser.add_argument("--content_boundary_residual_scale", type=float, default=0.3)
    parser.add_argument("--content_refiner_residual_scale", type=float, default=0.5)

    # --------------------------------------------------------
    # v6.1 local detail args
    # v6.1 局部细节参数
    # --------------------------------------------------------
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--local_detail_kernel_size", type=int, default=5)
    parser.add_argument("--local_detail_dilation_cycle", type=str, default="1,2")
    parser.add_argument("--local_detail_residual_scale", type=float, default=0.5)

    # --------------------------------------------------------
    # v6.1 semantic guide args
    # v6.1 semantic guide 参数
    # --------------------------------------------------------
    parser.add_argument("--semantic_guide_gate_init", type=float, default=0.3)
    parser.add_argument("--use_semantic_guide_cross_attn", type=str2bool, default=True)

    # --------------------------------------------------------
    # Loss weights
    # 损失权重
    # --------------------------------------------------------
    parser.add_argument("--fm_loss_weight", type=float, default=1.0)
    parser.add_argument("--recon_loss_weight", type=float, default=0.20)
    parser.add_argument("--refined_recon_loss_weight", type=float, default=1.00)
    parser.add_argument("--delta_loss_weight", type=float, default=0.15)
    parser.add_argument("--delta2_loss_weight", type=float, default=0.05)

    parser.add_argument("--mid_content_aux_weight", type=float, default=0.10)
    parser.add_argument("--final_content_aux_weight", type=float, default=0.20)

    parser.add_argument("--span_focus_aux_weight", type=float, default=0.20)
    parser.add_argument("--span_focus_ratio_min", type=float, default=0.10)
    parser.add_argument("--span_focus_ratio_max", type=float, default=0.35)
    parser.add_argument("--span_recon_weight", type=float, default=0.50)
    parser.add_argument("--span_content_weight", type=float, default=0.50)

    parser.add_argument(
        "--recon_loss_type",
        type=str,
        default="l1",
        choices=["l1", "mse", "smooth_l1"],
    )

    # --------------------------------------------------------
    # Conditional dropout
    # 条件 dropout
    # --------------------------------------------------------
    parser.add_argument("--cond_drop_prob_content", type=float, default=0.05)
    parser.add_argument("--cond_drop_prob_semantic", type=float, default=0.10)
    parser.add_argument("--cond_drop_prob_style", type=float, default=0.10)
    parser.add_argument("--cond_drop_all_prob", type=float, default=0.05)

    # --------------------------------------------------------
    # Postnet args
    # postnet 参数
    # --------------------------------------------------------
    parser.add_argument("--postnet_num_layers", type=int, default=3)
    parser.add_argument("--postnet_dropout", type=float, default=0.1)
    parser.add_argument("--postnet_conv_kernel_size", type=int, default=5)
    parser.add_argument("--detach_coarse_for_refinement", type=str2bool, default=True)

    # --------------------------------------------------------
    # Training dynamics
    # 训练动力学
    # --------------------------------------------------------
    parser.add_argument(
        "--t_sampling_mode",
        type=str,
        default="near_clean",
        choices=["near_clean", "uniform"],
    )
    parser.add_argument("--t_bias_power", type=float, default=2.0)
    parser.add_argument("--t_min", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=0.98)

    # --------------------------------------------------------
    # Frequency weighting
    # 频带加权
    # --------------------------------------------------------
    parser.add_argument("--freq_weight_min", type=float, default=1.0)
    parser.add_argument("--freq_weight_max", type=float, default=2.0)
    parser.add_argument("--freq_weight_power", type=float, default=1.5)

    # --------------------------------------------------------
    # Content auxiliary supervision
    # 内容辅助监督
    # --------------------------------------------------------
    parser.add_argument("--content_head_dropout", type=float, default=0.1)

    # --------------------------------------------------------
    # v6.1 length correction args
    # v6.1 长度修正参数
    # --------------------------------------------------------
    parser.add_argument("--enable_length_clamp", type=str2bool, default=True)
    parser.add_argument("--length_correction_mode", type=str, default="gentle_ratio")
    parser.add_argument("--length_ratio_min", type=float, default=2.6)
    parser.add_argument("--length_ratio_max", type=float, default=3.8)
    parser.add_argument("--length_bias_scale", type=float, default=0.95)

    # --------------------------------------------------------
    # v6.2 cold-start bootstrapper args
    # v6.2 冷启动 bootstrapper 参数
    # --------------------------------------------------------
    parser.add_argument("--use_coarse_bootstrapper", type=str2bool, default=True)
    parser.add_argument("--bootstrap_hidden_dim", type=int, default=None)
    parser.add_argument("--bootstrap_num_blocks", type=int, default=4)
    parser.add_argument("--bootstrap_kernel_size", type=int, default=5)
    parser.add_argument("--bootstrap_dropout", type=float, default=0.1)
    parser.add_argument("--bootstrap_expansion_factor", type=int, default=4)

    parser.add_argument("--bootstrap_mel_bias_init", type=float, default=-5.0)
    parser.add_argument("--bootstrap_residual_scale_init", type=float, default=0.1)
    parser.add_argument("--bootstrap_semantic_gate_init", type=float, default=0.5)
    parser.add_argument("--bootstrap_style_residual_scale_init", type=float, default=0.1)
    parser.add_argument("--bootstrap_clamp_output", type=str2bool, default=False)
    parser.add_argument("--bootstrap_output_min", type=float, default=-12.0)
    parser.add_argument("--bootstrap_output_max", type=float, default=4.0)

    parser.add_argument("--coarse_mel_loss_weight", type=float, default=1.0)
    parser.add_argument("--coarse_mel_mse_loss_weight", type=float, default=0.0)
    parser.add_argument("--coarse_delta_loss_weight", type=float, default=0.2)

    parser.add_argument("--use_bridge_loss", type=str2bool, default=True)
    parser.add_argument("--bridge_t_start", type=float, default=0.075)
    parser.add_argument("--bridge_noise_temperature", type=float, default=0.3)
    parser.add_argument("--bridge_recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--bridge_delta_loss_weight", type=float, default=0.2)
    parser.add_argument("--detach_coarse_for_bridge", type=str2bool, default=True)

    # --------------------------------------------------------
    # v6.3 Conformer bootstrapper args
    # v6.3 Conformer bootstrapper 参数
    # --------------------------------------------------------
    parser.add_argument("--use_conformer_bootstrapper", type=str2bool, default=True)
    parser.add_argument("--v63_bootstrap_hidden_dim", type=int, default=None)
    parser.add_argument("--v63_bootstrap_num_layers", type=int, default=6)
    parser.add_argument("--v63_bootstrap_num_heads", type=int, default=8)
    parser.add_argument("--v63_bootstrap_ff_mult", type=int, default=4)
    parser.add_argument("--v63_bootstrap_conv_kernel_size", type=int, default=15)
    parser.add_argument("--v63_bootstrap_dropout", type=float, default=0.1)
    parser.add_argument("--v63_bootstrap_mel_bias_init", type=float, default=-5.0)
    parser.add_argument("--v63_bootstrap_zero_init_output", type=str2bool, default=False)
    parser.add_argument("--v63_bootstrap_clamp_output", type=str2bool, default=False)

    # --------------------------------------------------------
    # v6.3.3 native text memory args
    # v6.3.3 原生文本记忆参数
    # --------------------------------------------------------
    parser.add_argument("--v63_use_native_text_memory", type=str2bool, default=False)
    parser.add_argument("--v63_text_memory_dim", type=int, default=None)

    # --------------------------------------------------------
    # v6.3 residual refiner args
    # v6.3 residual refiner 参数
    # --------------------------------------------------------
    parser.add_argument("--use_residual_refiner", type=str2bool, default=True)
    parser.add_argument("--residual_hidden_dim", type=int, default=None)
    parser.add_argument("--residual_num_layers", type=int, default=6)
    parser.add_argument("--residual_num_heads", type=int, default=8)
    parser.add_argument("--residual_ff_mult", type=int, default=4)
    parser.add_argument("--residual_conv_kernel_size", type=int, default=15)
    parser.add_argument("--residual_dropout", type=float, default=0.1)
    parser.add_argument("--residual_noise_scale", type=float, default=0.5)
    parser.add_argument("--residual_sample_temperature", type=float, default=0.3)
    parser.add_argument("--residual_t_min", type=float, default=0.0)
    parser.add_argument("--residual_t_max", type=float, default=1.0)
    parser.add_argument(
        "--detach_coarse_for_residual_refiner",
        type=str2bool,
        default=False,
    )

    # --------------------------------------------------------
    # v6.4.1 continuous semantic model args
    # v6.4.1 continuous semantic 模型参数
    # --------------------------------------------------------
    parser.add_argument(
        "--use_continuous_semantic",
        type=str2bool,
        default=False,
        help=(
            "Whether Stage2ConditionEncoder should fuse continuous semantic "
            "with token semantic embeddings."
        ),
    )
    parser.add_argument(
        "--continuous_semantic_fusion_mode",
        type=str,
        default="gated_add",
        choices=["gated_add", "add", "replace"],
    )
    parser.add_argument(
        "--continuous_semantic_gate_init",
        type=float,
        default=-2.0,
        help=(
            "Initial scalar gate logit for gated_add fusion. "
            "sigmoid(-2.0) is about 0.119, so the path starts conservative."
        ),
    )
    parser.add_argument(
        "--continuous_semantic_dropout",
        type=float,
        default=0.0,
    )

    # --------------------------------------------------------
    # v6.3 loss args
    # v6.3 损失参数
    # --------------------------------------------------------
    parser.add_argument("--include_v61_base_loss", type=str2bool, default=False)
    parser.add_argument("--v61_base_loss_weight", type=float, default=1.0)

    parser.add_argument("--v63_coarse_l1_weight", type=float, default=2.0)
    parser.add_argument("--v63_coarse_mse_weight", type=float, default=0.2)
    parser.add_argument("--v63_coarse_delta_weight", type=float, default=0.5)
    parser.add_argument("--v63_coarse_delta2_weight", type=float, default=0.2)

    parser.add_argument("--residual_refiner_loss_weight", type=float, default=1.0)
    parser.add_argument("--residual_flow_mse_weight", type=float, default=1.0)
    parser.add_argument("--residual_flow_l1_weight", type=float, default=0.0)
    parser.add_argument("--residual_recon_l1_weight", type=float, default=1.0)
    parser.add_argument("--final_recon_l1_weight", type=float, default=1.0)
    parser.add_argument("--final_delta_l1_weight", type=float, default=0.3)
    parser.add_argument("--final_delta2_l1_weight", type=float, default=0.1)

    # --------------------------------------------------------
    # Logging / saving arguments
    # 日志与保存参数
    # --------------------------------------------------------
    parser.add_argument("--save_dir", type=str, default="outputs/stage2_checkpoints_val")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="stage2_fm_val")

    # --------------------------------------------------------
    # v6.4 semantic source logging
    # v6.4 semantic source 日志
    # --------------------------------------------------------
    parser.add_argument(
        "--log_semantic_source_stats",
        type=str2bool,
        default=True,
        help=(
            "Whether to collect and print v6.4 semantic source stats "
            "such as source ids, reliability and pred/oracle length ratio."
        ),
    )
    parser.add_argument(
        "--semantic_source_log_jsonl",
        type=str2bool,
        default=True,
        help=(
            "Whether to write per-batch semantic source stats to "
            "save_dir/semantic_source_batch_log.jsonl."
        ),
    )
    parser.add_argument(
        "--semantic_source_log_every",
        type=int,
        default=1,
        help=(
            "Write semantic source JSONL every N training steps. "
            "Default 1 means every batch is recorded."
        ),
    )

    # --------------------------------------------------------
    # Checkpoint loading arguments
    # checkpoint 加载参数
    # --------------------------------------------------------
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--load_strict", type=str2bool, default=True)
    parser.add_argument("--reset_optimizer", type=parse_optional_bool, default=None)
    parser.add_argument("--reset_best_metric", type=parse_optional_bool, default=None)

    return parser.parse_args()


# ============================================================
# Main training entry
# 主训练入口
# ============================================================
def main() -> None:
    args = parse_args()
    training_mode = resolve_training_mode(args)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA was requested but is not available. Falling back to CPU.")
        device = torch.device("cpu")
        args.device = "cpu"
    else:
        device = requested_device

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Build dataloaders
    # 构建 dataloader
    # --------------------------------------------------------
    pin_memory = device.type == "cuda"

    val_semantic_source_mode = (
        args.semantic_source_mode
        if str(args.val_semantic_source_mode).lower() == "same"
        else args.val_semantic_source_mode
    )

    train_dataset, train_loader = build_dataloader(
        data_root=args.train_data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        semantic_source_mode=args.semantic_source_mode,
        semantic_oracle_prob=float(args.semantic_oracle_prob),
        include_continuous_semantic=bool(args.include_continuous_semantic),
        require_continuous_semantic=bool(args.require_continuous_semantic),
        continuous_semantic_dim=int(args.continuous_semantic_dim),
        include_v66_style_fields=bool(args.include_v66_style_fields),
        require_v66_style_fields=bool(args.require_v66_style_fields),
    )

    val_dataset, val_loader = build_dataloader(
        data_root=args.val_data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        semantic_source_mode=val_semantic_source_mode,
        semantic_oracle_prob=float(args.val_semantic_oracle_prob),
        include_continuous_semantic=bool(args.include_continuous_semantic),
        require_continuous_semantic=bool(args.require_continuous_semantic),
        continuous_semantic_dim=int(args.continuous_semantic_dim),
        include_v66_style_fields=bool(args.include_v66_style_fields),
        require_v66_style_fields=bool(args.require_v66_style_fields),
    )

    extra_val_datasets, extra_val_loaders = build_optional_val_loaders(
        args=args,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        val_semantic_source_mode=val_semantic_source_mode,
    )


    # --------------------------------------------------------
    # Resolve model config
    # 解析模型配置
    # --------------------------------------------------------
    source_checkpoint: str | None = None
    resume_payload: dict[str, Any] | None = None

    if training_mode == "resume_training":
        resume_ckpt_path = Path(args.resume_from).resolve()
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_ckpt_path}")

        resume_payload = safe_torch_load(resume_ckpt_path, map_location="cpu")
        source_checkpoint = str(resume_ckpt_path)

        ckpt_config = resume_payload.get("model_config", None)
        if not isinstance(ckpt_config, dict):
            ckpt_config = parse_checkpoint_args(resume_payload.get("args", {}))

        model_config = extract_model_config_from_dict(ckpt_config)

    else:
        model_config = build_model_config_from_cli_args(args)

    # --------------------------------------------------------
    # Build model
    # 构建模型
    # --------------------------------------------------------
    model = build_model_from_config(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = build_grad_scaler(device=device, use_amp=args.use_amp)

    # --------------------------------------------------------
    # Load checkpoint if needed
    # 如有需要，加载 checkpoint
    # --------------------------------------------------------
    start_epoch = 1
    global_step = 0
    best_val_loss = math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Deferred optimizer/scaler loading
    # 延迟加载 optimizer/scaler 状态
    #
    # Reason:
    #   We must apply trainable_scope first, then rebuild the final
    #   optimizer from trainable_params, and only then load optimizer state.
    #
    # 原因：
    #   必须先应用 trainable_scope，再用 trainable_params 重建最终 optimizer，
    #   最后才能加载 optimizer state。否则 optimizer 会被后续重建覆盖。
    # --------------------------------------------------------
    optimizer_state_to_load: dict[str, Any] | None = None
    scaler_state_to_load: dict[str, Any] | None = None

    reset_optimizer_effective = True
    reset_best_metric_effective = True

    if training_mode == "resume_training":
        assert resume_payload is not None
        assert args.resume_from is not None

        missing_unexpected = model.load_state_dict(
            resume_payload["model_state_dict"],
            strict=args.load_strict,
        )
        if not args.load_strict:
            print(f"[INFO] load_state_dict result: {missing_unexpected}")

        default_reset_optimizer = False
        reset_optimizer = (
            bool(args.reset_optimizer)
            if args.reset_optimizer is not None
            else default_reset_optimizer
        )

        # If trainable_scope is not all, parameter groups may differ from
        # checkpoint optimizer groups, so optimizer state should be reset.
        #
        # 如果 trainable_scope 不是 all，参数组可能和 checkpoint 中不一致，
        # 因此必须重置 optimizer。
        if str(args.trainable_scope).strip().lower() != "all":
            reset_optimizer = True

        reset_optimizer_effective = bool(reset_optimizer)

        # Defer optimizer/scaler loading until after the final optimizer
        # has been rebuilt from trainable_params.
        #
        # 延迟到最终 optimizer 重建后再加载。
        if (
            not reset_optimizer_effective
            and "optimizer_state_dict" in resume_payload
        ):
            optimizer_state_to_load = resume_payload["optimizer_state_dict"]

        if scaler is not None and "scaler_state_dict" in resume_payload:
            scaler_state_to_load = resume_payload["scaler_state_dict"]

        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        global_step = int(resume_payload.get("global_step", 0))

        default_reset_best_metric = False
        reset_best_metric = (
            bool(args.reset_best_metric)
            if args.reset_best_metric is not None
            else default_reset_best_metric
        )

        reset_best_metric_effective = bool(reset_best_metric)

        if not reset_best_metric_effective:
            best_val_loss = float(resume_payload.get("best_val_loss", math.inf))
            best_epoch = int(resume_payload.get("best_epoch", -1))

        history = load_history_for_resume(
            save_dir=save_dir,
            resume_ckpt_path=Path(args.resume_from).resolve(),
        )

    elif training_mode == "initialize_from_checkpoint":
        assert args.init_from is not None

        init_ckpt_path = Path(args.init_from).resolve()
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"init_from checkpoint not found: {init_ckpt_path}")

        init_payload = safe_torch_load(init_ckpt_path, map_location="cpu")
        source_checkpoint = str(init_ckpt_path)

        missing_unexpected = model.load_state_dict(
            init_payload["model_state_dict"],
            strict=args.load_strict,
        )
        if not args.load_strict:
            print(f"[INFO] load_state_dict result: {missing_unexpected}")

        # initialize_from_checkpoint normally starts a new run.
        default_reset_optimizer = True
        reset_optimizer = (
            bool(args.reset_optimizer)
            if args.reset_optimizer is not None
            else default_reset_optimizer
        )

        if str(args.trainable_scope).strip().lower() != "all":
            reset_optimizer = True

        reset_optimizer_effective = bool(reset_optimizer)

        # Defer optimizer loading until after final optimizer is rebuilt.
        #
        # 延迟加载 optimizer state，直到最终 optimizer 重建完成后。
        if (
            not reset_optimizer_effective
            and "optimizer_state_dict" in init_payload
        ):
            optimizer_state_to_load = init_payload["optimizer_state_dict"]

        default_reset_best_metric = True
        reset_best_metric = (
            bool(args.reset_best_metric)
            if args.reset_best_metric is not None
            else default_reset_best_metric
        )

        reset_best_metric_effective = bool(reset_best_metric)

        if not reset_best_metric_effective:
            best_val_loss = float(init_payload.get("best_val_loss", math.inf))
            best_epoch = int(init_payload.get("best_epoch", -1))

    # --------------------------------------------------------
    # Apply trainable scope and rebuild optimizer
    # 应用训练范围并重建 optimizer
    # --------------------------------------------------------
    trainable_info = apply_trainable_scope(
        model,
        trainable_scope=str(args.trainable_scope),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) <= 0:
        raise RuntimeError(
            f"No trainable parameters found for trainable_scope={args.trainable_scope}"
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Load optimizer/scaler state after final optimizer is built
    # 在最终 optimizer 构建后加载 optimizer/scaler 状态
    # --------------------------------------------------------
    if optimizer_state_to_load is not None:
        try:
            optimizer.load_state_dict(optimizer_state_to_load)
            print("[INFO] Optimizer state loaded into final optimizer.")
        except Exception as e:
            print(f"[WARN] Failed to load optimizer state into final optimizer: {e}")

    if scaler is not None and scaler_state_to_load is not None:
        try:
            scaler.load_state_dict(scaler_state_to_load)
            print("[INFO] AMP scaler state loaded.")
        except Exception as e:
            print(f"[WARN] Failed to load AMP scaler state: {e}")

    print("[trainable_scope]", json.dumps(trainable_info, ensure_ascii=False))
    print("[resume_state]", json.dumps(
        {
            "reset_optimizer_effective": bool(reset_optimizer_effective),
            "optimizer_state_requested": optimizer_state_to_load is not None,
            "scaler_state_requested": scaler_state_to_load is not None,
            "reset_best_metric_effective": bool(reset_best_metric_effective),
        },
        ensure_ascii=False,
    ))

    # --------------------------------------------------------
    # Save startup config
    # 保存启动配置
    # --------------------------------------------------------
    save_json(
        {
            "args": vars(args),
            "model_config": model_config,
            "training_mode": training_mode,
            "source_checkpoint": source_checkpoint,
            "train_num_samples": len(train_dataset),
            "val_num_samples": len(val_dataset),
            "extra_val_num_samples": {
                name: len(ds)
                for name, ds in extra_val_datasets.items()
            },
            "extra_val_roots": {
                "val_seen": args.val_seen_data_root,
                "val_unseen": args.val_unseen_data_root,
            },
            "trainable_info": trainable_info,
            "resume_state": {
                "reset_optimizer_effective": bool(reset_optimizer_effective),
                "optimizer_state_requested": optimizer_state_to_load is not None,
                "scaler_state_requested": scaler_state_to_load is not None,
                "reset_best_metric_effective": bool(reset_best_metric_effective),
            },
            "semantic_source_config": {
                "train_semantic_source_mode": args.semantic_source_mode,
                "train_semantic_oracle_prob": float(args.semantic_oracle_prob),
                "val_semantic_source_mode": val_semantic_source_mode,
                "val_semantic_oracle_prob": float(args.val_semantic_oracle_prob),
            },
        },
        save_dir / "run_config.json",
    )

    print_mode_banner(
        training_mode=training_mode,
        args=args,
        model_config=model_config,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )
    print(f"train_num_samples: {len(train_dataset)}")
    print(f"val_num_samples  : {len(val_dataset)}")

    if extra_val_datasets:
        for name, ds in extra_val_datasets.items():
            print(f"{name}_num_samples: {len(ds)}")

    if start_epoch > args.epochs:
        print(
            f"[WARN] start_epoch={start_epoch} is greater than args.epochs={args.epochs}. "
            "No training will be run."
        )

    # --------------------------------------------------------
    # Training loop
    # 训练循环
    # --------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        if (
            str(args.trainable_scope).strip().lower() != "all"
            and bool(args.keep_frozen_modules_eval)
        ):
            set_frozen_modules_eval_for_partial_scope(model)

        epoch_total_loss = 0.0
        epoch_num_steps = 0

        # v6.4 semantic source epoch-level running stats
        train_source_running: dict[str, Any] = {}
        train_v663_aux_running: dict[str, Any] = {}

        semantic_source_jsonl_path = save_dir / "semantic_source_batch_log.jsonl"

        print("----------------------------------------------------")
        print(f"[Epoch {epoch}/{args.epochs}] training started")

        for step, batch in enumerate(train_loader, start=1):
            batch = move_stage2_inputs_to_device(batch, device)

            batch_source_stats: dict[str, Any] = {}

            if bool(args.log_semantic_source_stats):
                batch_source_stats = summarize_semantic_source_batch(batch)
                update_semantic_source_running_stats(
                    train_source_running,
                    batch_source_stats,
                )

            batch_continuous_stats: dict[str, Any] = {}

            if bool(args.include_continuous_semantic):
                batch_continuous_stats = summarize_continuous_semantic_batch(batch)

            optimizer.zero_grad(set_to_none=True)

            with make_amp_context(device=device, use_amp=args.use_amp):
                loss, aux = model.compute_flow_matching_loss(batch)

            if bool(args.include_continuous_semantic):
                batch_continuous_stats.update(
                    extract_continuous_semantic_encoder_stats(model)
                )

            if bool(args.log_semantic_source_stats) and isinstance(aux, dict):
                # Add source stats into aux so _aux_to_log_string can print them.
                # 将 source stats 合入 aux，方便统一打印。
                aux.update(
                    {
                        "semantic_source_oracle_count": batch_source_stats.get(
                            "semantic_source_oracle_count",
                            -1,
                        ),
                        "semantic_source_predicted_count": batch_source_stats.get(
                            "semantic_source_predicted_count",
                            -1,
                        ),
                        "semantic_source_predicted_frac": batch_source_stats.get(
                            "semantic_source_predicted_frac",
                            -1.0,
                        ),
                        "semantic_reliability_mean": batch_source_stats.get(
                            "semantic_reliability_mean",
                            -1.0,
                        ),
                        "semantic_reliability_min": batch_source_stats.get(
                            "semantic_reliability_min",
                            -1.0,
                        ),
                        "semantic_reliability_max": batch_source_stats.get(
                            "semantic_reliability_max",
                            -1.0,
                        ),
                        "selected_semantic_len_mean": batch_source_stats.get(
                            "selected_semantic_len_mean",
                            -1.0,
                        ),
                        "oracle_semantic_len_mean": batch_source_stats.get(
                            "oracle_semantic_len_mean",
                            -1.0,
                        ),
                        "pred_semantic_len_mean": batch_source_stats.get(
                            "pred_semantic_len_mean",
                            -1.0,
                        ),
                        "pred_over_oracle_len_ratio_mean": batch_source_stats.get(
                            "pred_over_oracle_len_ratio_mean",
                            -1.0,
                        ),
                    }
                )

            if bool(args.include_continuous_semantic) and isinstance(aux, dict):
                aux.update(
                    {
                        "continuous_semantic_has_selected": batch_continuous_stats.get(
                            "continuous_semantic_has_selected",
                            0,
                        ),
                        "continuous_semantic_len_mean": batch_continuous_stats.get(
                            "continuous_semantic_len_mean",
                            -1.0,
                        ),
                        "continuous_semantic_norm_mean": batch_continuous_stats.get(
                            "continuous_semantic_norm_mean",
                            -1.0,
                        ),
                        "continuous_semantic_norm_std": batch_continuous_stats.get(
                            "continuous_semantic_norm_std",
                            -1.0,
                        ),
                        "continuous_semantic_encoder_used": batch_continuous_stats.get(
                            "continuous_semantic_encoder_used",
                            -1,
                        ),
                        "continuous_semantic_encoder_gate": batch_continuous_stats.get(
                            "continuous_semantic_encoder_gate",
                            -1.0,
                        ),
                        "continuous_semantic_encoder_norm_mean": batch_continuous_stats.get(
                            "continuous_semantic_encoder_norm_mean",
                            -1.0,
                        ),
                        "continuous_semantic_encoder_len_mean": batch_continuous_stats.get(
                            "continuous_semantic_encoder_len_mean",
                            -1.0,
                        ),
                    }
                )

            update_v663_aux_running_stats(train_v663_aux_running, aux)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch={epoch}, step={step}: {loss.item()}")

            if scaler is not None:
                scaler.scale(loss).backward()

                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

            global_step += 1
            epoch_total_loss += float(loss.item())
            epoch_num_steps += 1

            if (
                bool(args.log_semantic_source_stats)
                and bool(args.semantic_source_log_jsonl)
                and int(args.semantic_source_log_every) > 0
                and (global_step % int(args.semantic_source_log_every) == 0)
            ):
                _jsonl_append(
                    {
                        "phase": "train",
                        "epoch": int(epoch),
                        "step": int(step),
                        "num_steps_in_epoch": int(len(train_loader)),
                        "global_step": int(global_step),
                        "loss": float(loss.item()),
                        **batch_source_stats,
                        **batch_continuous_stats,
                    },
                    semantic_source_jsonl_path,
                )


            if args.log_every > 0 and (step % args.log_every == 0 or step == 1):
                aux_str = _aux_to_log_string(aux)

                source_str = ""
                if bool(args.log_semantic_source_stats):
                    source_str = format_semantic_source_summary(batch_source_stats)

                print(
                    f"[epoch={epoch} step={step}/{len(train_loader)} "
                    f"global_step={global_step}] "
                    f"loss={float(loss.item()):.6f} {aux_str} {source_str}"
                )

        train_loss = epoch_total_loss / max(epoch_num_steps, 1)

        print(f"[Epoch {epoch}/{args.epochs}] training finished. train_loss={train_loss:.6f}")
        print(f"[Epoch {epoch}/{args.epochs}] validation started")

        if bool(args.log_semantic_source_stats):
            val_loss, val_source_stats = run_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                use_amp=args.use_amp,
                collect_semantic_source_stats=True,
            )
        else:
            val_loss = run_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                use_amp=args.use_amp,
            )
            val_source_stats = {}

        print(f"[Epoch {epoch}/{args.epochs}] validation finished. val_loss={val_loss:.6f}")

        extra_val_results: dict[str, dict[str, Any]] = {}

        if extra_val_loaders:
            print(f"[Epoch {epoch}/{args.epochs}] extra validation started")

            extra_val_results = run_extra_validations(
                model=model,
                extra_val_loaders=extra_val_loaders,
                device=device,
                use_amp=args.use_amp,
                collect_semantic_source_stats=bool(args.log_semantic_source_stats),
            )

            for extra_name, extra_result in extra_val_results.items():
                extra_loss = float(extra_result["loss"])
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"{extra_name} validation finished. "
                    f"{extra_name}_loss={extra_loss:.6f}"
                )

                if bool(args.log_semantic_source_stats):
                    extra_source_stats = extra_result.get("semantic_source_stats", {})
                    print(
                        f"[Epoch {epoch}/{args.epochs}] "
                        f"{extra_name} semantic source stats: "
                        f"{format_semantic_source_summary(extra_source_stats)}"
                    )

        train_source_stats = finalize_semantic_source_running_stats(
            train_source_running
        )

        train_v663_aux_stats = finalize_v663_aux_running_stats(train_v663_aux_running)

        if bool(args.log_semantic_source_stats):
            print(
                f"[Epoch {epoch}/{args.epochs}] train semantic source stats: "
                f"{format_semantic_source_summary(train_source_stats)}"
            )
            print(
                f"[Epoch {epoch}/{args.epochs}] val semantic source stats: "
                f"{format_semantic_source_summary(val_source_stats)}"
            )

        improved = False
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            improved = True

        epoch_record = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),

            # Primary validation loss.
            # 主验证集 loss。best_model.pt 根据它选择。
            "val_loss": float(val_loss),
            "primary_val_name": "val",
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "improved": bool(improved),

            # Extra validation losses.
            # 额外验证集 loss，仅用于监控。
            "extra_val_results": extra_val_results,
        }

        if train_v663_aux_stats.get("num_logged_steps", 0) > 0:
            epoch_record["train_v663_aux_stats"] = train_v663_aux_stats

        if bool(args.log_semantic_source_stats):
            epoch_record["train_semantic_source_stats"] = train_source_stats
            epoch_record["val_semantic_source_stats"] = val_source_stats

        history.append(epoch_record)

        save_json(
            {
                "history": history,
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "global_step": int(global_step),
                "model_config": model_config,
            },
            save_dir / "history.json",
        )

        # Always save last checkpoint.
        save_checkpoint(
            path=save_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            args=args,
            model_config=model_config,
            training_mode=training_mode,
            source_checkpoint=source_checkpoint,
        )

        # Optional epoch checkpoint.
        if args.save_every_epoch:
            save_checkpoint(
                path=save_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                args=args,
                model_config=model_config,
                training_mode=training_mode,
                source_checkpoint=source_checkpoint,
            )

        # Save best checkpoint.
        if improved:
            save_checkpoint(
                path=save_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                args=args,
                model_config=model_config,
                training_mode=training_mode,
                source_checkpoint=source_checkpoint,
            )
            print(f"[BEST] Updated best_model.pt at epoch={epoch}, val_loss={val_loss:.6f}")

        summary_payload = {
            "experiment_name": args.experiment_name,
            "training_mode": training_mode,
            "source_checkpoint": source_checkpoint,
            "model_version": model_config.get("model_version"),
            "last_epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),

            # Primary validation.
            "val_loss": float(val_loss),
            "primary_val_name": "val",
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),

            # Extra validations.
            "extra_val_results": extra_val_results,

            "save_dir": str(save_dir),
        }

        if train_v663_aux_stats.get("num_logged_steps", 0) > 0:
            summary_payload["train_v663_aux_stats"] = train_v663_aux_stats

        if bool(args.log_semantic_source_stats):
            summary_payload["train_semantic_source_stats"] = train_source_stats
            summary_payload["val_semantic_source_stats"] = val_source_stats
            summary_payload["semantic_source_batch_log_jsonl"] = str(
                semantic_source_jsonl_path
            )

        save_json(
            summary_payload,
            save_dir / "summary.json",
        )

    print("====================================================")
    print("[DONE] Stage2 FM training finished.")
    print(f"save_dir      : {save_dir}")
    print(f"best_val_loss : {best_val_loss}")
    print(f"best_epoch    : {best_epoch}")
    print(f"global_step   : {global_step}")
    print("====================================================")


if __name__ == "__main__":
    main()