from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch


# ============================================================
# Make project root importable
# 把项目根目录加入导入路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Constants
# 常量
# ============================================================
# Keys that can be forwarded from checkpoint model_config / args
# to scripts/train_stage2_fm.py.
#
# 这些 key 会尽量从 base checkpoint 的 model_config / args 中继承，
# 以避免 init_from 时模型结构与 checkpoint 不匹配。
FORWARDABLE_MODEL_ARG_NAMES: set[str] = {
    # Basic model
    "acoustic_dim",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "semantic_vocab_size",
    "semantic_rate_hz",
    "acoustic_rate_hz",
    "dropout",
    "model_version",

    # Self condition
    "use_self_condition",
    "self_condition_prob",

    # v4 old length predictor
    "enable_length_predictor",
    "length_predictor_hidden_dim",
    "length_loss_weight",
    "length_teacher_forcing_prob",

    # v5 / v6 condition
    "phoneme_vocab_size",
    "phoneme_embed_dim",
    "bert_dim",
    "content_dim",
    "content_refiner_layers",
    "content_frame_dim",
    "content_expander_layers",
    "style_dim",

    # v6.1 content-main condition
    "content_main_dim",
    "content_injection_gate_init",
    "content_refiner_dilations",
    "content_boundary_enhance",
    "content_boundary_kernel_size",
    "content_boundary_residual_scale",
    "content_refiner_residual_scale",

    # v6.1 local detail
    "ff_mult",
    "local_detail_kernel_size",
    "local_detail_dilation_cycle",
    "local_detail_residual_scale",

    # v6.1 semantic guide
    "semantic_guide_gate_init",
    "use_semantic_guide_cross_attn",

    # Loss weights
    "fm_loss_weight",
    "recon_loss_weight",
    "refined_recon_loss_weight",
    "delta_loss_weight",
    "delta2_loss_weight",
    "mid_content_aux_weight",
    "final_content_aux_weight",
    "span_focus_aux_weight",
    "span_focus_ratio_min",
    "span_focus_ratio_max",
    "span_recon_weight",
    "span_content_weight",
    "recon_loss_type",

    # Conditional dropout
    "cond_drop_prob_content",
    "cond_drop_prob_semantic",
    "cond_drop_prob_style",
    "cond_drop_all_prob",

    # Postnet
    "postnet_num_layers",
    "postnet_dropout",
    "postnet_conv_kernel_size",
    "detach_coarse_for_refinement",

    # Training dynamics
    "t_sampling_mode",
    "t_bias_power",
    "t_min",
    "t_max",

    # Frequency weighting
    "freq_weight_min",
    "freq_weight_max",
    "freq_weight_power",

    # Content auxiliary supervision
    "content_head_dropout",

    # v6.1 length correction
    "enable_length_clamp",
    "length_correction_mode",
    "length_ratio_min",
    "length_ratio_max",
    "length_bias_scale",

    # v6.2 bootstrapper
    "use_coarse_bootstrapper",
    "bootstrap_hidden_dim",
    "bootstrap_num_blocks",
    "bootstrap_kernel_size",
    "bootstrap_dropout",
    "bootstrap_expansion_factor",
    "bootstrap_mel_bias_init",
    "bootstrap_residual_scale_init",
    "bootstrap_semantic_gate_init",
    "bootstrap_style_residual_scale_init",
    "bootstrap_clamp_output",
    "bootstrap_output_min",
    "bootstrap_output_max",
    "coarse_mel_loss_weight",
    "coarse_mel_mse_loss_weight",
    "coarse_delta_loss_weight",
    "use_bridge_loss",
    "bridge_t_start",
    "bridge_noise_temperature",
    "bridge_recon_loss_weight",
    "bridge_delta_loss_weight",
    "detach_coarse_for_bridge",

    # v6.3 Conformer bootstrapper
    "use_conformer_bootstrapper",
    "v63_bootstrap_hidden_dim",
    "v63_bootstrap_num_layers",
    "v63_bootstrap_num_heads",
    "v63_bootstrap_ff_mult",
    "v63_bootstrap_conv_kernel_size",
    "v63_bootstrap_dropout",
    "v63_bootstrap_mel_bias_init",
    "v63_bootstrap_zero_init_output",
    "v63_bootstrap_clamp_output",

    # v6.3.3 native text memory
    "v63_use_native_text_memory",
    "v63_text_memory_dim",

    # v6.3 residual refiner
    "use_residual_refiner",
    "residual_hidden_dim",
    "residual_num_layers",
    "residual_num_heads",
    "residual_ff_mult",
    "residual_conv_kernel_size",
    "residual_dropout",
    "residual_noise_scale",
    "residual_sample_temperature",
    "residual_t_min",
    "residual_t_max",
    "detach_coarse_for_residual_refiner",

    # v6.4.1 continuous semantic
    "use_continuous_semantic",
    "continuous_semantic_dim",
    "continuous_semantic_fusion_mode",
    "continuous_semantic_gate_init",
    "continuous_semantic_dropout",

    # v6.6 reference acoustic style
    "use_reference_acoustic_style",
    "reference_acoustic_dim",
    "reference_style_dim",
    "reference_style_hidden_dim",
    "reference_style_num_layers",
    "reference_style_kernel_size",
    "reference_style_dropout",
    "reference_style_fusion_mode",
    "reference_style_gate_init",

    # v6.6.2 prosody / speaker auxiliary losses
    "use_v66_energy_loss",
    "v66_energy_loss_weight",
    "v66_energy_loss_type",
    "v66_energy_normalize",
    "use_v66_f0_loss",
    "v66_f0_loss_weight",
    "v66_f0_loss_type",
    "v66_f0_normalize",
    "v66_f0_hidden_dim",
    "use_v66_speaker_loss",
    "v66_speaker_loss_weight",
    "v66_speaker_embedding_dim",
    "v66_require_aux_targets",


    # v6.6.4 Speaker Identity Adapter
    "use_v664_speaker_identity_adapter",
    "v664_speaker_identity_source",
    "v664_speaker_embedding_dim",
    "v664_speaker_adapter_hidden_dim",
    "v664_speaker_adapter_num_layers",
    "v664_speaker_adapter_dropout",
    "v664_speaker_adapter_fusion_mode",
    "v664_speaker_adapter_gate_init",
    "v664_speaker_adapter_normalize",
    # v6.6.3 Bootstrapper LoRA
    "use_bootstrapper_lora",
    "bootstrapper_lora_rank",
    "bootstrapper_lora_alpha",
    "bootstrapper_lora_dropout",
    "bootstrapper_lora_target",
    "bootstrapper_lora_init_scale",


    # v6.6.3-A Bootstrapper Attention LoRA
    "use_bootstrapper_attention_lora",
    "bootstrapper_attention_lora_rank",
    "bootstrapper_attention_lora_alpha",
    "bootstrapper_attention_lora_dropout",
    "bootstrapper_attention_lora_target",
    "bootstrapper_attention_lora_gate_init",
    "bootstrapper_attention_lora_enable_q",
    "bootstrapper_attention_lora_enable_k",
    "bootstrapper_attention_lora_enable_v",
    "bootstrapper_attention_lora_enable_o",
    # v6.3 loss args
    "include_v61_base_loss",
    "v61_base_loss_weight",
    "v63_coarse_l1_weight",
    "v63_coarse_mse_weight",
    "v63_coarse_delta_weight",
    "v63_coarse_delta2_weight",
    "residual_refiner_loss_weight",
    "residual_flow_mse_weight",
    "residual_flow_l1_weight",
    "residual_recon_l1_weight",
    "final_recon_l1_weight",
    "final_delta_l1_weight",
    "final_delta2_l1_weight",
}

# Flags in train_stage2_fm.py that are action="store_true".
# 这些参数不是 "--key true" 形式，而是只要出现就表示 true。
STORE_TRUE_ARG_NAMES: set[str] = {
    "use_self_condition",
    "enable_length_predictor",
}


# ============================================================
# Basic helpers
# 基础工具函数
# ============================================================
def str2bool(x: str | bool) -> bool:
    """
    EN:
    Convert common string values to bool.

    ZH:
    将常见字符串形式转换为 bool。
    """
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()

    if s in {"1", "true", "yes", "y", "on"}:
        return True

    if s in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {x}")


def bool_to_cli_str(x: bool) -> str:
    """
    EN:
    Convert bool to string accepted by train_stage2_fm.py str2bool args.

    ZH:
    将 bool 转成 train_stage2_fm.py 可接受的字符串。
    """
    return "true" if bool(x) else "false"


def parse_args() -> argparse.Namespace:
    """
    EN:
    Parse few-shot training wrapper arguments.

    ZH:
    解析 few-shot 训练包装器参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot Stage2 training wrapper. "
            "It calls scripts/train_stage2_fm.py with --init_from base checkpoint."
        )
    )

    # --------------------------------------------------------
    # Required paths
    # 必需路径
    # --------------------------------------------------------
    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help=(
            "Base zero-shot Stage2 checkpoint, e.g. 100h 35e best_model.pt."
        ),
    )
    parser.add_argument(
        "--fewshot_train_root",
        type=str,
        required=True,
        help="Few-shot Stage2 .pt train directory.",
    )
    parser.add_argument(
        "--fewshot_val_root",
        type=str,
        required=True,
        help="Few-shot Stage2 .pt validation directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for user-specific few-shot checkpoint.",
    )

    # --------------------------------------------------------
    # Runtime
    # 运行参数
    # --------------------------------------------------------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", type=str2bool, default=True)

    # --------------------------------------------------------
    # Optimizer
    # 优化器参数
    # --------------------------------------------------------
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # --------------------------------------------------------
    # Few-shot semantic/data policy
    # few-shot 语义与数据策略
    # --------------------------------------------------------
    parser.add_argument(
        "--semantic_source_mode",
        type=str,
        default="oracle",
        choices=["oracle", "predicted", "mixed"],
        help=(
            "For few-shot v1, oracle is recommended first. "
            "mixed/predicted can be tested later for robustness."
        ),
    )
    parser.add_argument("--semantic_oracle_prob", type=float, default=0.8)
    parser.add_argument(
        "--val_semantic_source_mode",
        type=str,
        default="oracle",
        choices=["oracle", "predicted", "mixed", "same"],
    )
    parser.add_argument("--val_semantic_oracle_prob", type=float, default=1.0)

    parser.add_argument(
        "--include_continuous_semantic",
        type=str2bool,
        default=True,
        help="Read oracle_semantic_continuous from Stage2 .pt samples.",
    )
    parser.add_argument(
        "--require_continuous_semantic",
        type=str2bool,
        default=True,
        help="Require continuous semantic fields in few-shot .pt samples.",
    )
    parser.add_argument(
        "--continuous_semantic_dim",
        type=int,
        default=768,
    )

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
    # --------------------------------------------------------
    parser.add_argument("--use_bootstrapper_lora", type=str2bool, default=False)
    parser.add_argument("--bootstrapper_lora_rank", type=int, default=4)
    parser.add_argument("--bootstrapper_lora_alpha", type=float, default=8.0)
    parser.add_argument("--bootstrapper_lora_dropout", type=float, default=0.05)
    parser.add_argument("--bootstrapper_lora_target", type=str, default="core", choices=["core", "all_linear"])
    parser.add_argument("--bootstrapper_lora_init_scale", type=float, default=0.01)

    # --------------------------------------------------------
    # v6.6.3-A Bootstrapper Attention LoRA args
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
    # Trainable scope
    # 训练范围
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
            "Few-shot trainable scope. "
            "For small few-shot datasets, prefer style_adapter_only first, "
            "then style_condition_plus_adapter if similarity is still insufficient."
        ),
    )
    parser.add_argument(
        "--keep_frozen_modules_eval",
        type=str2bool,
        default=True,
    )

    # --------------------------------------------------------
    # Checkpoint loading
    # checkpoint 加载策略
    # --------------------------------------------------------
    parser.add_argument("--load_strict", type=str2bool, default=True)
    parser.add_argument("--reset_optimizer", type=str2bool, default=True)
    parser.add_argument("--reset_best_metric", type=str2bool, default=True)

    # --------------------------------------------------------
    # Base checkpoint config inheritance
    # base checkpoint 配置继承
    # --------------------------------------------------------
    parser.add_argument(
        "--inherit_model_config_from_base",
        type=str2bool,
        default=True,
        help=(
            "If true, read model_config / args from base_ckpt and forward "
            "compatible architecture args to train_stage2_fm.py. "
            "This is strongly recommended for init_from."
        ),
    )
    parser.add_argument(
        "--force_v65_coarse_only_defaults",
        type=str2bool,
        default=True,
        help=(
            "If true, force the known v6.5 coarse-only defaults: "
            "model_version=v6_3, use_continuous_semantic=true, "
            "v63_use_native_text_memory=true, use_residual_refiner=false, "
            "residual_refiner_loss_weight=0.0. "
            "These can still be overridden by --extra_train_args."
        ),
    )

    # --------------------------------------------------------
    # Logging / saving
    # 日志与保存
    # --------------------------------------------------------
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="fewshot_stage2_v1",
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_epoch", type=str2bool, default=False)
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help=(
            "Few-shot wrapper report path. "
            "Default: <output_dir>/fewshot_train_report.json"
        ),
    )

    # --------------------------------------------------------
    # Optional extra args
    # 额外透传参数
    # --------------------------------------------------------
    parser.add_argument(
        "--extra_train_args",
        type=str,
        default="",
        help=(
            "Extra raw args appended to train_stage2_fm.py. "
            "Example: \"--dropout 0.05 --log_every 5\". "
            "These are appended last and can override earlier choices if "
            "train_stage2_fm.py uses last-value behavior."
        ),
    )

    # --------------------------------------------------------
    # Safety / dry-run
    # 安全与预览
    # --------------------------------------------------------
    parser.add_argument(
        "--dry_run",
        type=str2bool,
        default=False,
        help="Only print and save command report; do not launch training.",
    )
    parser.add_argument(
        "--print_command",
        type=str2bool,
        default=True,
        help="Print the generated train_stage2_fm.py command.",
    )
    parser.add_argument(
        "--fail_if_no_cuda",
        type=str2bool,
        default=False,
        help=(
            "If true and --device cuda but CUDA is unavailable, fail before "
            "calling train_stage2_fm.py. If false, train_stage2_fm.py may "
            "fall back to CPU."
        ),
    )

    return parser.parse_args()


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """
    EN:
    Save JSON object.

    ZH:
    保存 JSON 对象。
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_torch_load(path: str | Path) -> Any:
    """
    EN:
    Load checkpoint with weights_only=False.

    ZH:
    加载 checkpoint，显式 weights_only=False。
    """
    return torch.load(str(path), map_location="cpu", weights_only=False)


def normalize_path(path: str | Path) -> str:
    """
    EN:
    Resolve path to absolute string.

    ZH:
    将路径转为绝对路径字符串。
    """
    return str(Path(path).resolve())


def command_to_printable(cmd: list[str]) -> str:
    """
    EN:
    Convert command list into readable shell string.

    ZH:
    将命令列表转换为便于复制的字符串。
    """
    parts: list[str] = []

    for x in cmd:
        s = str(x)
        if " " in s or "\\" in s or ":" in s:
            parts.append(f'"{s}"')
        else:
            parts.append(s)

    return " ".join(parts)


def scan_pt_files(root: str | Path) -> list[Path]:
    """
    EN:
    Scan .pt files directly under a directory.

    ZH:
    扫描目录下一层 .pt 文件。
    """
    p = Path(root).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Stage2 .pt root not found: {p}")

    if not p.is_dir():
        raise NotADirectoryError(f"Stage2 .pt root is not a directory: {p}")

    files = sorted([x.resolve() for x in p.glob("*.pt") if x.is_file()])

    if not files:
        raise ValueError(f"No .pt samples found under: {p}")

    return files


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """
    EN:
    Validate paths and basic runtime assumptions.

    ZH:
    校验路径和基本运行条件。
    """
    base_ckpt = Path(args.base_ckpt).resolve()
    train_root = Path(args.fewshot_train_root).resolve()
    val_root = Path(args.fewshot_val_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not base_ckpt.exists() or not base_ckpt.is_file():
        raise FileNotFoundError(f"base_ckpt not found: {base_ckpt}")

    train_pt_files = scan_pt_files(train_root)
    val_pt_files = scan_pt_files(val_root)

    if (
        str(args.device).strip().lower().startswith("cuda")
        and bool(args.fail_if_no_cuda)
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is false."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_ckpt": str(base_ckpt),
        "fewshot_train_root": str(train_root),
        "fewshot_val_root": str(val_root),
        "output_dir": str(output_dir),
        "num_train_pt": int(len(train_pt_files)),
        "num_val_pt": int(len(val_pt_files)),
        "train_pt_preview": [str(p) for p in train_pt_files[:8]],
        "val_pt_preview": [str(p) for p in val_pt_files[:8]],
        "cuda_available": bool(torch.cuda.is_available()),
    }


def parse_checkpoint_args(x: Any) -> dict[str, Any]:
    """
    EN:
    Normalize checkpoint args object into dict.

    ZH:
    将 checkpoint 中的 args 规范为 dict。
    """
    if x is None:
        return {}

    if isinstance(x, dict):
        return dict(x)

    try:
        return vars(x)
    except Exception:
        return {}


def extract_base_config(base_ckpt_path: str | Path) -> dict[str, Any]:
    """
    EN:
    Extract model config from base checkpoint.

    Priority:
    1. checkpoint["model_config"]
    2. checkpoint["args"]

    ZH:
    从 base checkpoint 中提取模型配置。

    优先级：
    1. checkpoint["model_config"]
    2. checkpoint["args"]
    """
    payload = safe_torch_load(base_ckpt_path)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint payload dict, got: {type(payload)}")

    model_config = payload.get("model_config", None)

    if isinstance(model_config, dict):
        config = dict(model_config)
        source = "model_config"
    else:
        config = parse_checkpoint_args(payload.get("args", {}))
        source = "args"

    if not config:
        raise ValueError(
            f"Could not extract model_config or args from checkpoint: {base_ckpt_path}"
        )

    return {
        "source": source,
        "config": config,
        "checkpoint_keys": sorted(list(payload.keys())),
        "epoch": payload.get("epoch", None),
        "global_step": payload.get("global_step", None),
        "best_val_loss": payload.get("best_val_loss", None),
        "best_epoch": payload.get("best_epoch", None),
    }


def format_cli_value(v: Any) -> str:
    """
    EN:
    Convert Python value to CLI string.

    ZH:
    将 Python 值转换为命令行字符串。
    """
    if isinstance(v, bool):
        return bool_to_cli_str(v)

    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)

    return str(v)


def add_model_config_args_from_base(
    cmd: list[str],
    base_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """
    EN:
    Forward model architecture args from base checkpoint into train_stage2_fm.py.

    This is important because train_stage2_fm.py uses CLI args to build the model
    before loading --init_from model_state_dict.

    ZH:
    将 base checkpoint 中的模型结构参数透传给 train_stage2_fm.py。

    这很重要，因为 train_stage2_fm.py 会先根据 CLI 参数建模，
    再加载 --init_from 的 model_state_dict。
    """
    forwarded: dict[str, Any] = {}

    for key in sorted(FORWARDABLE_MODEL_ARG_NAMES):
        if key not in base_config:
            continue

        value = base_config[key]

        # Skip None. Optional args with None should rely on train_stage2_fm.py defaults.
        # 跳过 None，避免 argparse int/float 解析失败。
        if value is None:
            continue

        if key in STORE_TRUE_ARG_NAMES:
            if bool(value):
                cmd.append(f"--{key}")
                forwarded[key] = True
            continue

        cmd.extend([f"--{key}", format_cli_value(value)])
        forwarded[key] = value

    return cmd, forwarded


def apply_v65_coarse_only_defaults(cmd: list[str]) -> tuple[list[str], dict[str, Any]]:
    """
    EN:
    Append known v6.5 coarse-only defaults.

    These are appended after inherited base config so they can ensure the expected
    VoiceLab v6.5 path. User can still append --extra_train_args after this.

    ZH:
    追加 v6.5 coarse-only 主线默认参数。

    它们会放在继承配置之后，确保进入当前推荐的 v6.5 路径。
    用户仍可通过 --extra_train_args 在最后覆盖。
    """
    defaults: dict[str, Any] = {
        "model_version": "v6_3",

        # v6.3.3 native text memory
        "v63_use_native_text_memory": True,
        "v63_text_memory_dim": 512,

        # v6.4.1 continuous semantic
        "use_continuous_semantic": True,
        "continuous_semantic_dim": 768,
        "continuous_semantic_fusion_mode": "gated_add",

        # Coarse-only mainline
        "use_residual_refiner": False,
        "residual_refiner_loss_weight": 0.0,
    }

    for key, value in defaults.items():
        if key in STORE_TRUE_ARG_NAMES:
            if bool(value):
                cmd.append(f"--{key}")
            continue

        cmd.extend([f"--{key}", format_cli_value(value)])

    return cmd, defaults



def append_explicit_v66_override_args(
    cmd: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    """
    EN:
    Re-append current-run v6.6 / v6.6.2 args after inherited base config.

    The few-shot wrapper forwards architecture args from the base checkpoint after
    the main command list is built. If the base checkpoint was trained with, for
    example, use_v66_speaker_loss=false, it can override a current run that passes
    --use_v66_speaker_loss true. Re-appending explicit current-run args here makes
    the current command win while still preserving inherited architecture defaults.

    ZH:
    在继承 base checkpoint 配置之后，重新追加当前运行的 v6.6 / v6.6.2 参数。

    few-shot wrapper 会在主命令之后透传 base checkpoint 的结构参数。如果 base
    checkpoint 中 use_v66_speaker_loss=false，就可能覆盖当前命令传入的 true。
    这里再次追加当前参数，保证最终以本次命令为准。
    """
    override_items: list[tuple[str, Any]] = [
        # v6.4.1 continuous semantic
        ("include_continuous_semantic", bool(args.include_continuous_semantic)),
        ("require_continuous_semantic", bool(args.require_continuous_semantic)),
        ("continuous_semantic_dim", int(args.continuous_semantic_dim)),

        # v6.6 reference acoustic style
        ("include_v66_style_fields", bool(args.include_v66_style_fields)),
        ("require_v66_style_fields", bool(args.require_v66_style_fields)),
        ("use_reference_acoustic_style", bool(args.use_reference_acoustic_style)),
        ("reference_acoustic_dim", int(args.reference_acoustic_dim)),
        ("reference_style_dim", int(args.reference_style_dim)),
        ("reference_style_hidden_dim", int(args.reference_style_hidden_dim)),
        ("reference_style_num_layers", int(args.reference_style_num_layers)),
        ("reference_style_kernel_size", int(args.reference_style_kernel_size)),
        ("reference_style_dropout", float(args.reference_style_dropout)),
        ("reference_style_fusion_mode", str(args.reference_style_fusion_mode)),
        ("reference_style_gate_init", float(args.reference_style_gate_init)),

        # v6.6.2 prosody / speaker auxiliary losses
        ("use_v66_energy_loss", bool(args.use_v66_energy_loss)),
        ("v66_energy_loss_weight", float(args.v66_energy_loss_weight)),
        ("v66_energy_loss_type", str(args.v66_energy_loss_type)),
        ("v66_energy_normalize", bool(args.v66_energy_normalize)),
        ("use_v66_f0_loss", bool(args.use_v66_f0_loss)),
        ("v66_f0_loss_weight", float(args.v66_f0_loss_weight)),
        ("v66_f0_loss_type", str(args.v66_f0_loss_type)),
        ("v66_f0_normalize", bool(args.v66_f0_normalize)),
        ("v66_f0_hidden_dim", int(args.v66_f0_hidden_dim)),
        ("use_v66_speaker_loss", bool(args.use_v66_speaker_loss)),
        ("v66_speaker_loss_weight", float(args.v66_speaker_loss_weight)),
        ("v66_speaker_embedding_dim", int(args.v66_speaker_embedding_dim)),
        ("v66_require_aux_targets", bool(args.v66_require_aux_targets)),
        ("use_v664_speaker_identity_adapter", bool(args.use_v664_speaker_identity_adapter)),
        ("v664_speaker_identity_source", str(args.v664_speaker_identity_source)),
        ("v664_speaker_embedding_dim", int(args.v664_speaker_embedding_dim)),
        ("v664_speaker_adapter_hidden_dim", int(args.v664_speaker_adapter_hidden_dim)),
        ("v664_speaker_adapter_num_layers", int(args.v664_speaker_adapter_num_layers)),
        ("v664_speaker_adapter_dropout", float(args.v664_speaker_adapter_dropout)),
        ("v664_speaker_adapter_fusion_mode", str(args.v664_speaker_adapter_fusion_mode)),
        ("v664_speaker_adapter_gate_init", float(args.v664_speaker_adapter_gate_init)),
        ("v664_speaker_adapter_normalize", bool(args.v664_speaker_adapter_normalize)),
        ("use_bootstrapper_attention_lora", bool(args.use_bootstrapper_attention_lora)),
        ("bootstrapper_attention_lora_rank", int(args.bootstrapper_attention_lora_rank)),
        ("bootstrapper_attention_lora_alpha", float(args.bootstrapper_attention_lora_alpha)),
        ("bootstrapper_attention_lora_dropout", float(args.bootstrapper_attention_lora_dropout)),
        ("bootstrapper_attention_lora_target", str(args.bootstrapper_attention_lora_target)),
        ("bootstrapper_attention_lora_gate_init", float(args.bootstrapper_attention_lora_gate_init)),
        ("bootstrapper_attention_lora_enable_q", bool(args.bootstrapper_attention_lora_enable_q)),
        ("bootstrapper_attention_lora_enable_k", bool(args.bootstrapper_attention_lora_enable_k)),
        ("bootstrapper_attention_lora_enable_v", bool(args.bootstrapper_attention_lora_enable_v)),
        ("bootstrapper_attention_lora_enable_o", bool(args.bootstrapper_attention_lora_enable_o)),
        ("use_bootstrapper_lora", bool(args.use_bootstrapper_lora)),
        ("bootstrapper_lora_rank", int(args.bootstrapper_lora_rank)),
        ("bootstrapper_lora_alpha", float(args.bootstrapper_lora_alpha)),
        ("bootstrapper_lora_dropout", float(args.bootstrapper_lora_dropout)),
        ("bootstrapper_lora_target", str(args.bootstrapper_lora_target)),
        ("bootstrapper_lora_init_scale", float(args.bootstrapper_lora_init_scale)),
    ]

    applied: dict[str, Any] = {}
    for key, value in override_items:
        cmd.extend([f"--{key}", format_cli_value(value)])
        applied[key] = value

    return cmd, applied


def build_train_command(
    args: argparse.Namespace,
    *,
    inherited_base_config: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    """
    EN:
    Build train_stage2_fm.py command.

    ZH:
    构造 train_stage2_fm.py 训练命令。
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_stage2_fm.py"),

        "--train_data_root",
        normalize_path(args.fewshot_train_root),
        "--val_data_root",
        normalize_path(args.fewshot_val_root),

        "--device",
        str(args.device),
        "--epochs",
        str(int(args.epochs)),
        "--batch_size",
        str(int(args.batch_size)),
        "--num_workers",
        str(int(args.num_workers)),

        "--lr",
        str(float(args.lr)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--grad_clip",
        str(float(args.grad_clip)),

        "--semantic_source_mode",
        str(args.semantic_source_mode),
        "--semantic_oracle_prob",
        str(float(args.semantic_oracle_prob)),
        "--val_semantic_source_mode",
        str(args.val_semantic_source_mode),
        "--val_semantic_oracle_prob",
        str(float(args.val_semantic_oracle_prob)),

        "--include_continuous_semantic",
        bool_to_cli_str(bool(args.include_continuous_semantic)),
        "--require_continuous_semantic",
        bool_to_cli_str(bool(args.require_continuous_semantic)),
        "--continuous_semantic_dim",
        str(int(args.continuous_semantic_dim)),

        "--include_v66_style_fields",
        bool_to_cli_str(bool(args.include_v66_style_fields)),
        "--require_v66_style_fields",
        bool_to_cli_str(bool(args.require_v66_style_fields)),
        "--use_reference_acoustic_style",
        bool_to_cli_str(bool(args.use_reference_acoustic_style)),
        "--reference_acoustic_dim",
        str(int(args.reference_acoustic_dim)),
        "--reference_style_dim",
        str(int(args.reference_style_dim)),
        "--reference_style_hidden_dim",
        str(int(args.reference_style_hidden_dim)),
        "--reference_style_num_layers",
        str(int(args.reference_style_num_layers)),
        "--reference_style_kernel_size",
        str(int(args.reference_style_kernel_size)),
        "--reference_style_dropout",
        str(float(args.reference_style_dropout)),
        "--reference_style_fusion_mode",
        str(args.reference_style_fusion_mode),
        "--reference_style_gate_init",
        str(float(args.reference_style_gate_init)),

        "--use_v66_energy_loss",
        bool_to_cli_str(bool(args.use_v66_energy_loss)),
        "--v66_energy_loss_weight",
        str(float(args.v66_energy_loss_weight)),
        "--v66_energy_loss_type",
        str(args.v66_energy_loss_type),
        "--v66_energy_normalize",
        bool_to_cli_str(bool(args.v66_energy_normalize)),

        "--use_v66_f0_loss",
        bool_to_cli_str(bool(args.use_v66_f0_loss)),
        "--v66_f0_loss_weight",
        str(float(args.v66_f0_loss_weight)),
        "--v66_f0_loss_type",
        str(args.v66_f0_loss_type),
        "--v66_f0_normalize",
        bool_to_cli_str(bool(args.v66_f0_normalize)),
        "--v66_f0_hidden_dim",
        str(int(args.v66_f0_hidden_dim)),

        "--use_v66_speaker_loss",
        bool_to_cli_str(bool(args.use_v66_speaker_loss)),
        "--v66_speaker_loss_weight",
        str(float(args.v66_speaker_loss_weight)),
        "--v66_speaker_embedding_dim",
        str(int(args.v66_speaker_embedding_dim)),
        "--v66_require_aux_targets",
        bool_to_cli_str(bool(args.v66_require_aux_targets)),

        "--trainable_scope",
        str(args.trainable_scope),

        "--keep_frozen_modules_eval",
        bool_to_cli_str(bool(args.keep_frozen_modules_eval)),

        "--save_dir",
        normalize_path(args.output_dir),
        "--experiment_name",
        str(args.experiment_name),
        "--log_every",
        str(int(args.log_every)),

        "--init_from",
        normalize_path(args.base_ckpt),
        "--load_strict",
        bool_to_cli_str(bool(args.load_strict)),
        "--reset_optimizer",
        bool_to_cli_str(bool(args.reset_optimizer)),
        "--reset_best_metric",
        bool_to_cli_str(bool(args.reset_best_metric)),
    ]

    if bool(args.use_amp):
        cmd.append("--use_amp")

    if bool(args.save_every_epoch):
        cmd.append("--save_every_epoch")

    forwarded_config: dict[str, Any] = {}
    forced_defaults: dict[str, Any] = {}
    explicit_v66_overrides: dict[str, Any] = {}

    if bool(args.inherit_model_config_from_base):
        if inherited_base_config is None:
            raise ValueError("inherited_base_config is None but inheritance was requested.")

        cmd, forwarded_config = add_model_config_args_from_base(
            cmd,
            inherited_base_config,
        )

    if bool(args.force_v65_coarse_only_defaults):
        cmd, forced_defaults = apply_v65_coarse_only_defaults(cmd)

    cmd, explicit_v66_overrides = append_explicit_v66_override_args(
        cmd,
        args,
    )

    if str(args.extra_train_args).strip():
        extra = shlex.split(str(args.extra_train_args), posix=False)
        cmd.extend(extra)

    debug_info = {
        "forwarded_model_config": forwarded_config,
        "forced_v65_defaults": forced_defaults,
        "explicit_v66_overrides": explicit_v66_overrides,
        "extra_train_args": str(args.extra_train_args),
    }

    return cmd, debug_info


def print_command_block(title: str, cmd: list[str]) -> None:
    """
    EN:
    Print command block.

    ZH:
    打印命令块。
    """
    print("====================================================")
    print(title)
    print("====================================================")
    print(command_to_printable(cmd))
    print("====================================================")


def print_summary(report: dict[str, Any]) -> None:
    """
    EN:
    Print few-shot wrapper summary.

    ZH:
    打印 few-shot wrapper 摘要。
    """
    print("====================================================")
    print("Few-shot Stage2 training wrapper")
    print("====================================================")
    print(f"status              : {report['status']}")
    print(f"base_ckpt           : {report['inputs']['base_ckpt']}")
    print(f"fewshot_train_root  : {report['inputs']['fewshot_train_root']}")
    print(f"fewshot_val_root    : {report['inputs']['fewshot_val_root']}")
    print(f"output_dir          : {report['inputs']['output_dir']}")
    print(f"num_train_pt        : {report['input_check']['num_train_pt']}")
    print(f"num_val_pt          : {report['input_check']['num_val_pt']}")
    print(f"device              : {report['config']['device']}")
    print(f"epochs              : {report['config']['epochs']}")
    print(f"batch_size          : {report['config']['batch_size']}")
    print(f"lr                  : {report['config']['lr']}")
    print(f"trainable_scope     : {report['config']['trainable_scope']}")
    print(f"semantic_source_mode: {report['config']['semantic_source_mode']}")
    print(f"report_path         : {report['report_path']}")
    print("====================================================")


# ============================================================
# Main
# 主入口
# ============================================================
def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else output_dir / "fewshot_train_report.json"
    )

    input_check = validate_inputs(args)

    base_config_info: dict[str, Any] | None = None
    inherited_config: dict[str, Any] | None = None

    if bool(args.inherit_model_config_from_base):
        base_config_info = extract_base_config(args.base_ckpt)
        inherited_config = dict(base_config_info["config"])

    train_cmd, command_debug = build_train_command(
        args,
        inherited_base_config=inherited_config,
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "PENDING",
        "project_root": str(PROJECT_ROOT),
        "report_path": str(report_path),
        "inputs": {
            "base_ckpt": normalize_path(args.base_ckpt),
            "fewshot_train_root": normalize_path(args.fewshot_train_root),
            "fewshot_val_root": normalize_path(args.fewshot_val_root),
            "output_dir": normalize_path(args.output_dir),
        },
        "input_check": input_check,
        "base_checkpoint": base_config_info,
        "config": {
            "device": str(args.device),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "use_amp": bool(args.use_amp),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "semantic_source_mode": str(args.semantic_source_mode),
            "semantic_oracle_prob": float(args.semantic_oracle_prob),
            "val_semantic_source_mode": str(args.val_semantic_source_mode),
            "val_semantic_oracle_prob": float(args.val_semantic_oracle_prob),
            "include_continuous_semantic": bool(args.include_continuous_semantic),
            "require_continuous_semantic": bool(args.require_continuous_semantic),
            "continuous_semantic_dim": int(args.continuous_semantic_dim),
            "trainable_scope": str(args.trainable_scope),
            "keep_frozen_modules_eval": bool(args.keep_frozen_modules_eval),
            "load_strict": bool(args.load_strict),
            "reset_optimizer": bool(args.reset_optimizer),
            "reset_best_metric": bool(args.reset_best_metric),
            "inherit_model_config_from_base": bool(args.inherit_model_config_from_base),
            "force_v65_coarse_only_defaults": bool(args.force_v65_coarse_only_defaults),
            "experiment_name": str(args.experiment_name),
            "log_every": int(args.log_every),
            "save_every_epoch": bool(args.save_every_epoch),
            "dry_run": bool(args.dry_run),
        },
        "command": {
            "argv": train_cmd,
            "printable": command_to_printable(train_cmd),
            "debug": command_debug,
        },
        "return_codes": {},
    }

    if bool(args.print_command):
        print_command_block(
            "Generated train_stage2_fm.py command",
            train_cmd,
        )

    if bool(args.dry_run):
        report["status"] = "DRY_RUN"
        save_json(report, report_path)
        print_summary(report)
        return

    try:
        completed = subprocess.run(
            train_cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        report["return_codes"]["train_stage2_fm"] = int(completed.returncode)
        report["status"] = "OK"

    except subprocess.CalledProcessError as e:
        report["return_codes"]["train_stage2_fm"] = int(e.returncode)
        report["status"] = "FAILED_TRAIN_STAGE2_FM"
        report["error"] = {
            "type": "CalledProcessError",
            "message": str(e),
        }
        save_json(report, report_path)
        raise

    save_json(report, report_path)
    print_summary(report)


if __name__ == "__main__":
    main()