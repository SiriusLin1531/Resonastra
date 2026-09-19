from __future__ import annotations

from dataclasses import fields

import inspect
from pathlib import Path
from typing import Any

import torch

from src.interfaces.stage2_io import Stage2Inputs, Stage2Outputs
from src.models.stage2_acoustic_model import (
    Stage2FlowMatchingAcousticModel,
    Stage2FlowMatchingAcousticModelV2,
    Stage2FlowMatchingAcousticModelV5,
    Stage2FlowMatchingAcousticModelV6,
    Stage2FlowMatchingAcousticModelV61,
    Stage2FlowMatchingAcousticModelV62,
    Stage2FlowMatchingAcousticModelV63,
)
from src.vocoder.stage2_vocoder_adapter import Stage2VocoderAdapter
from src.features.fewshot_style_features import extract_log_mel_from_path

DEFAULT_SEMANTIC_RATE_HZ = 25.0
DEFAULT_ACOUSTIC_RATE_HZ = 22050.0 / 256.0  # 86.1328125


def _as_bool(x: Any, default: bool = False) -> bool:
    """
    EN:
    Robust bool conversion for checkpoint config values.

    ZH:
    用于 checkpoint config 的稳健 bool 转换。
    """
    if x is None:
        return bool(default)

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        return bool(x)

    s = str(x).strip().lower()

    if s in {"1", "true", "yes", "y", "on"}:
        return True

    if s in {"0", "false", "no", "n", "off"}:
        return False

    return bool(default)


def _as_int_tuple(
    x: Any,
    default: tuple[int, ...] = (1, 2),
) -> tuple[int, ...]:
    """
    EN:
    Robust int tuple parser for checkpoint config values.

    Examples:
        "1,2" -> (1, 2)
        [1, 2] -> (1, 2)

    ZH:
    用于 checkpoint config 的整数 tuple 解析器。
    """
    if x is None:
        return default

    if isinstance(x, tuple):
        values = tuple(int(v) for v in x)
    elif isinstance(x, list):
        values = tuple(int(v) for v in x)
    else:
        values = tuple(int(v.strip()) for v in str(x).split(",") if v.strip())

    if len(values) <= 0:
        return default

    return values


def _normalize_checkpoint_args(args_obj: Any) -> dict[str, Any]:
    """
    EN:
    Normalize checkpoint args object into a plain dict.

    ZH:
    将 checkpoint 中的 args 对象统一转换为普通 dict。
    """
    if args_obj is None:
        return {}

    if isinstance(args_obj, dict):
        return dict(args_obj)

    try:
        return vars(args_obj)
    except Exception:
        return {}


class Stage2InferencePipeline:
    """
    EN:
    Pipeline:
    frontend -> prompt tokenizer -> stage-1 -> stage-2 FM acoustic model -> vocoder

    v6.1 additions:
    - Load Stage2FlowMatchingAcousticModelV61 from checkpoint config.
    - Return stage2_aux in mel/audio outputs.
    - Expose semantic/acoustic frame rates in final result.
    - Keep target_lengths_override for oracle/manual length tests.

    ZH:
    流水线：
    frontend -> prompt tokenizer -> stage-1 -> stage-2 Flow Matching 声学模型 -> vocoder

    v6.1 新增：
    - 支持从 checkpoint config 加载 Stage2FlowMatchingAcousticModelV61。
    - 在 mel/audio 输出中返回 stage2_aux。
    - 在最终结果中返回 semantic/acoustic 帧率。
    - 保留 target_lengths_override，方便 oracle/manual length 测试。
    """

    def __init__(
        self,
        frontend,
        prompt_tokenizer,
        stage1,
        stage2_model: (
            Stage2FlowMatchingAcousticModel
            | Stage2FlowMatchingAcousticModelV2
            | Stage2FlowMatchingAcousticModelV5
            | Stage2FlowMatchingAcousticModelV6
            | Stage2FlowMatchingAcousticModelV61
            | Stage2FlowMatchingAcousticModelV62
            | Stage2FlowMatchingAcousticModelV63
            | None
        ) = None,
        vocoder_adapter: Stage2VocoderAdapter | None = None,
        device: str = "cpu",
    ) -> None:
        self.frontend = frontend
        self.prompt_tokenizer = prompt_tokenizer
        self.stage1 = stage1
        self.stage2_model = stage2_model
        self.vocoder_adapter = vocoder_adapter
        self.device = torch.device(device)
        self.stage2_checkpoint_meta: dict[str, Any] = {}

        if self.stage2_model is not None:
            self.stage2_model = self.stage2_model.to(self.device)
            self.stage2_model.eval()

    # --------------------------------------------------------
    # Helpers / 辅助函数
    # --------------------------------------------------------
    def _filter_supported_kwargs(self, fn, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        EN:
        Keep only kwargs supported by a callable signature.

        ZH:
        只保留目标函数签名支持的 kwargs。
        """
        sig = inspect.signature(fn)
        supported = {}

        for k, v in kwargs.items():
            if k in sig.parameters and v is not None:
                supported[k] = v

        return supported

    def _to_device_if_tensor(self, x: Any) -> Any:
        """
        EN:
        Move tensor to current pipeline device.

        ZH:
        如果输入是 tensor，则移动到当前 pipeline device。
        """
        if torch.is_tensor(x):
            return x.to(self.device)
        return x

    def _move_stage2_inputs_to_device(self, batch: Stage2Inputs) -> Stage2Inputs:
        """
        EN:
        Move all tensor fields inside Stage2Inputs to current device.

        v6.4:
        This method is now dataclass-field based, so newly added tensor fields
        such as semantic_source_ids, semantic_reliability, oracle_semantic_tokens,
        pred_semantic_tokens and future continuous semantic fields are moved
        automatically.

        ZH:
        将 Stage2Inputs 中的所有 tensor 字段移动到当前 device。

        v6.4:
        改为基于 dataclass fields 自动移动，避免后续新增字段后遗漏。
        """
        kwargs: dict[str, Any] = {}

        for f in fields(Stage2Inputs):
            value = getattr(batch, f.name)

            if torch.is_tensor(value):
                kwargs[f.name] = value.to(self.device)
                continue

            if isinstance(value, dict):
                moved_dict: dict[str, Any] = {}
                for k, v in value.items():
                    if torch.is_tensor(v):
                        moved_dict[k] = v.to(self.device)
                    else:
                        moved_dict[k] = v
                kwargs[f.name] = moved_dict
                continue

            kwargs[f.name] = value

        return Stage2Inputs(**kwargs)

    def _normalize_target_lengths_override(
        self,
        target_lengths_override: Any,
        batch_size: int,
    ) -> torch.Tensor | None:
        """
        EN:
        Normalize target_lengths_override into a LongTensor on current device.

        Accepted forms:
        - None
        - int
        - list/tuple of ints
        - torch.Tensor

        Return:
        - shape (B,) long tensor on self.device
        - or None

        ZH:
        将 target_lengths_override 统一整理成当前 device 上的 LongTensor。

        支持输入形式：
        - None
        - 单个 int
        - int 列表 / tuple
        - torch.Tensor

        返回：
        - 形状 (B,) 的 long tensor
        - 或 None
        """
        if target_lengths_override is None:
            return None

        if isinstance(target_lengths_override, int):
            return torch.full(
                size=(batch_size,),
                fill_value=int(target_lengths_override),
                dtype=torch.long,
                device=self.device,
            )

        if isinstance(target_lengths_override, (list, tuple)):
            t = torch.tensor(target_lengths_override, dtype=torch.long, device=self.device)
        elif torch.is_tensor(target_lengths_override):
            t = target_lengths_override.to(device=self.device, dtype=torch.long)
        else:
            raise TypeError(
                "target_lengths_override must be one of: "
                "None / int / list / tuple / torch.Tensor, "
                f"but got {type(target_lengths_override)}"
            )

        if t.ndim == 0:
            t = t.view(1)

        t = t.view(-1)

        if t.numel() == 1 and batch_size > 1:
            t = t.repeat(batch_size)

        if t.numel() != batch_size:
            raise ValueError(
                f"target_lengths_override has {t.numel()} elements, "
                f"but batch_size is {batch_size}."
            )

        return t.clamp_min(1)


    def _build_v66_prompt_acoustic_extras(self, prompt_wav_path: str | Path) -> dict[str, torch.Tensor]:
        """
        Build v6.6 prompt_acoustic fields for inference.

        These defaults match scripts/export_fewshot_style_cache.py and the
        historical Stage2 preprocessing mel setup.
        """
        model_config = self.stage2_checkpoint_meta.get("model_config", {})
        n_mels = int(model_config.get("reference_acoustic_dim", model_config.get("acoustic_dim", 80)))

        prompt_acoustic = extract_log_mel_from_path(
            prompt_wav_path,
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
        ).float()

        prompt_acoustic = prompt_acoustic.unsqueeze(0).to(self.device)
        prompt_acoustic_lengths = torch.tensor(
            [prompt_acoustic.shape[1]],
            dtype=torch.long,
            device=self.device,
        )
        return {
            "prompt_acoustic": prompt_acoustic,
            "prompt_acoustic_lengths": prompt_acoustic_lengths,
        }

    # --------------------------------------------------------
    # Stage-2 model construction / 第二阶段模型构建
    # --------------------------------------------------------
    def _build_stage2_model_from_config(
        self,
        model_config: dict[str, Any],
    ):
        """
        EN:
        Build stage-2 model from checkpoint model_config.

        ZH:
        根据 checkpoint 中的 model_config 构建 stage-2 模型。
        """
        model_version = str(model_config.get("model_version", "v1")).lower()

        common_kwargs = dict(
            acoustic_dim=int(model_config.get("acoustic_dim", 80)),
            hidden_dim=int(model_config.get("hidden_dim", 512)),
            num_layers=int(model_config.get("num_layers", 6)),
            num_heads=int(model_config.get("num_heads", 8)),
            semantic_vocab_size=int(model_config.get("semantic_vocab_size", 1024)),
            semantic_rate_hz=float(model_config.get("semantic_rate_hz", DEFAULT_SEMANTIC_RATE_HZ)),
            acoustic_rate_hz=float(model_config.get("acoustic_rate_hz", DEFAULT_ACOUSTIC_RATE_HZ)),
            dropout=float(model_config.get("dropout", 0.1)),

            use_self_condition=_as_bool(model_config.get("use_self_condition", True), True),
            self_condition_prob=float(model_config.get("self_condition_prob", 0.5)),

            phoneme_vocab_size=int(model_config.get("phoneme_vocab_size", 4096)),
            phoneme_embed_dim=int(model_config.get("phoneme_embed_dim", 256)),
            bert_dim=int(model_config.get("bert_dim", 1024)),
            content_dim=int(model_config.get("content_dim", 384)),
            content_refiner_layers=int(model_config.get("content_refiner_layers", 4)),
            content_frame_dim=int(model_config.get("content_frame_dim", 384)),
            content_expander_layers=int(model_config.get("content_expander_layers", 2)),
            style_dim=int(model_config.get("style_dim", 256)),

            content_main_dim=int(model_config.get("content_main_dim", model_config.get("hidden_dim", 512))),
            content_injection_gate_init=float(model_config.get("content_injection_gate_init", 0.5)),
            content_refiner_dilations=_as_int_tuple(
                model_config.get("content_refiner_dilations", "1,2"),
                default=(1, 2),
            ),
            content_boundary_enhance=_as_bool(model_config.get("content_boundary_enhance", True), True),
            content_boundary_kernel_size=int(model_config.get("content_boundary_kernel_size", 3)),
            content_boundary_residual_scale=float(model_config.get("content_boundary_residual_scale", 0.3)),
            content_refiner_residual_scale=float(model_config.get("content_refiner_residual_scale", 0.5)),

            ff_mult=int(model_config.get("ff_mult", 4)),
            local_detail_kernel_size=int(model_config.get("local_detail_kernel_size", 5)),
            local_detail_dilation_cycle=_as_int_tuple(
                model_config.get("local_detail_dilation_cycle", "1,2"),
                default=(1, 2),
            ),
            local_detail_residual_scale=float(model_config.get("local_detail_residual_scale", 0.5)),
            semantic_guide_gate_init=float(model_config.get("semantic_guide_gate_init", 0.3)),
            use_semantic_guide_cross_attn=_as_bool(
                model_config.get("use_semantic_guide_cross_attn", True),
                True,
            ),

            fm_loss_weight=float(model_config.get("fm_loss_weight", 1.0)),
            recon_loss_weight=float(model_config.get("recon_loss_weight", 0.20)),
            refined_recon_loss_weight=float(model_config.get("refined_recon_loss_weight", 1.00)),
            delta_loss_weight=float(model_config.get("delta_loss_weight", 0.15)),
            delta2_loss_weight=float(model_config.get("delta2_loss_weight", 0.05)),
            mid_content_aux_weight=float(model_config.get("mid_content_aux_weight", 0.10)),
            final_content_aux_weight=float(model_config.get("final_content_aux_weight", 0.20)),
            span_focus_aux_weight=float(model_config.get("span_focus_aux_weight", 0.20)),
            span_focus_ratio_min=float(model_config.get("span_focus_ratio_min", 0.10)),
            span_focus_ratio_max=float(model_config.get("span_focus_ratio_max", 0.35)),
            span_recon_weight=float(model_config.get("span_recon_weight", 0.50)),
            span_content_weight=float(model_config.get("span_content_weight", 0.50)),
            recon_loss_type=str(model_config.get("recon_loss_type", "l1")),

            cond_drop_prob_content=float(model_config.get("cond_drop_prob_content", 0.05)),
            cond_drop_prob_semantic=float(model_config.get("cond_drop_prob_semantic", 0.10)),
            cond_drop_prob_style=float(model_config.get("cond_drop_prob_style", 0.10)),
            cond_drop_all_prob=float(model_config.get("cond_drop_all_prob", 0.05)),

            postnet_num_layers=int(model_config.get("postnet_num_layers", 3)),
            postnet_dropout=float(model_config.get("postnet_dropout", 0.1)),
            postnet_conv_kernel_size=int(model_config.get("postnet_conv_kernel_size", 5)),
            detach_coarse_for_refinement=_as_bool(
                model_config.get("detach_coarse_for_refinement", True),
                True,
            ),

            t_sampling_mode=str(model_config.get("t_sampling_mode", "near_clean")),
            t_bias_power=float(model_config.get("t_bias_power", 2.0)),
            t_min=float(model_config.get("t_min", 0.05)),
            t_max=float(model_config.get("t_max", 0.98)),

            freq_weight_min=float(model_config.get("freq_weight_min", 1.0)),
            freq_weight_max=float(model_config.get("freq_weight_max", 2.0)),
            freq_weight_power=float(model_config.get("freq_weight_power", 1.5)),

            content_head_dropout=float(model_config.get("content_head_dropout", 0.1)),

            enable_length_clamp=_as_bool(model_config.get("enable_length_clamp", True), True),
            length_correction_mode=str(model_config.get("length_correction_mode", "gentle_ratio")),
            length_ratio_min=float(model_config.get("length_ratio_min", 2.6)),
            length_ratio_max=float(model_config.get("length_ratio_max", 3.8)),
            length_bias_scale=float(model_config.get("length_bias_scale", 0.95)),

            # ------------------------------------------------
            # v6.2 cold-start bootstrapper args
            # v6.2 冷启动 bootstrapper 参数
            # ------------------------------------------------
            use_coarse_bootstrapper=_as_bool(
                model_config.get("use_coarse_bootstrapper", True),
                True,
            ),
            bootstrap_hidden_dim=model_config.get("bootstrap_hidden_dim", None),
            bootstrap_num_blocks=int(model_config.get("bootstrap_num_blocks", 4)),
            bootstrap_kernel_size=int(model_config.get("bootstrap_kernel_size", 5)),
            bootstrap_dropout=float(model_config.get("bootstrap_dropout", 0.1)),
            bootstrap_expansion_factor=int(model_config.get("bootstrap_expansion_factor", 4)),

            bootstrap_mel_bias_init=float(model_config.get("bootstrap_mel_bias_init", -5.0)),
            bootstrap_residual_scale_init=float(model_config.get("bootstrap_residual_scale_init", 0.1)),
            bootstrap_semantic_gate_init=float(model_config.get("bootstrap_semantic_gate_init", 0.5)),
            bootstrap_style_residual_scale_init=float(
                model_config.get("bootstrap_style_residual_scale_init", 0.1)
            ),
            bootstrap_clamp_output=_as_bool(
                model_config.get("bootstrap_clamp_output", False),
                False,
            ),
            bootstrap_output_min=float(model_config.get("bootstrap_output_min", -12.0)),
            bootstrap_output_max=float(model_config.get("bootstrap_output_max", 4.0)),

            coarse_mel_loss_weight=float(model_config.get("coarse_mel_loss_weight", 1.0)),
            coarse_mel_mse_loss_weight=float(
                model_config.get("coarse_mel_mse_loss_weight", 0.0)
            ),
            coarse_delta_loss_weight=float(model_config.get("coarse_delta_loss_weight", 0.2)),

            use_bridge_loss=_as_bool(model_config.get("use_bridge_loss", True), True),
            bridge_t_start=float(model_config.get("bridge_t_start", 0.075)),
            bridge_noise_temperature=float(model_config.get("bridge_noise_temperature", 0.3)),
            bridge_recon_loss_weight=float(model_config.get("bridge_recon_loss_weight", 1.0)),
            bridge_delta_loss_weight=float(model_config.get("bridge_delta_loss_weight", 0.2)),
            detach_coarse_for_bridge=_as_bool(
                model_config.get("detach_coarse_for_bridge", True),
                True,
            ),

            # ------------------------------------------------
            # v6.3 Conformer bootstrapper args
            # v6.3 Conformer bootstrapper 参数
            # ------------------------------------------------
            use_conformer_bootstrapper=_as_bool(
                model_config.get("use_conformer_bootstrapper", True),
                True,
            ),
            v63_bootstrap_hidden_dim=model_config.get("v63_bootstrap_hidden_dim", None),
            v63_bootstrap_num_layers=int(model_config.get("v63_bootstrap_num_layers", 6)),
            v63_bootstrap_num_heads=int(model_config.get("v63_bootstrap_num_heads", 8)),
            v63_bootstrap_ff_mult=int(model_config.get("v63_bootstrap_ff_mult", 4)),
            v63_bootstrap_conv_kernel_size=int(
                model_config.get("v63_bootstrap_conv_kernel_size", 15)
            ),
            v63_bootstrap_dropout=float(model_config.get("v63_bootstrap_dropout", 0.1)),
            v63_bootstrap_mel_bias_init=float(
                model_config.get("v63_bootstrap_mel_bias_init", -5.0)
            ),
            v63_bootstrap_zero_init_output=_as_bool(
                model_config.get("v63_bootstrap_zero_init_output", False),
                False,
            ),
            v63_bootstrap_clamp_output=_as_bool(
                model_config.get("v63_bootstrap_clamp_output", False),
                False,
            ),

            # ------------------------------------------------
            # v6.3.3 native text memory args
            # v6.3.3 原生文本记忆参数
            # ------------------------------------------------
            v63_use_native_text_memory=_as_bool(
                model_config.get("v63_use_native_text_memory", False),
                False,
            ),
            v63_text_memory_dim=model_config.get("v63_text_memory_dim", None),
            
            # ------------------------------------------------
            # v6.3 residual refiner args
            # v6.3 residual refiner 参数
            # ------------------------------------------------
            use_residual_refiner=_as_bool(
                model_config.get("use_residual_refiner", True),
                True,
            ),
            residual_hidden_dim=model_config.get("residual_hidden_dim", None),
            residual_num_layers=int(model_config.get("residual_num_layers", 6)),
            residual_num_heads=int(model_config.get("residual_num_heads", 8)),
            residual_ff_mult=int(model_config.get("residual_ff_mult", 4)),
            residual_conv_kernel_size=int(model_config.get("residual_conv_kernel_size", 15)),
            residual_dropout=float(model_config.get("residual_dropout", 0.1)),
            residual_noise_scale=float(model_config.get("residual_noise_scale", 0.5)),
            residual_sample_temperature=float(
                model_config.get("residual_sample_temperature", 0.3)
            ),
            residual_t_min=float(model_config.get("residual_t_min", 0.0)),
            residual_t_max=float(model_config.get("residual_t_max", 1.0)),
            detach_coarse_for_residual_refiner=_as_bool(
                model_config.get("detach_coarse_for_residual_refiner", False),
                False,
            ),

            # ------------------------------------------------
            # v6.4.1 continuous semantic args
            # v6.4.1 continuous semantic 参数
            # ------------------------------------------------
            use_continuous_semantic=_as_bool(
                model_config.get("use_continuous_semantic", False),
                False,
            ),
            continuous_semantic_dim=int(
                model_config.get("continuous_semantic_dim", 768)
            ),
            continuous_semantic_fusion_mode=str(
                model_config.get("continuous_semantic_fusion_mode", "gated_add")
            ),
            continuous_semantic_gate_init=float(
                model_config.get("continuous_semantic_gate_init", -2.0)
            ),
            continuous_semantic_dropout=float(
                model_config.get("continuous_semantic_dropout", 0.0)
            ),
            

            # ------------------------------------------------
            # v6.6 reference acoustic style args
            # v6.6 参考音频声学风格参数
            # ------------------------------------------------
            use_reference_acoustic_style=_as_bool(
                model_config.get("use_reference_acoustic_style", False),
                False,
            ),
            reference_acoustic_dim=int(model_config.get("reference_acoustic_dim", 80)),
            reference_style_dim=int(model_config.get("reference_style_dim", 256)),
            reference_style_hidden_dim=int(
                model_config.get("reference_style_hidden_dim", 256)
            ),
            reference_style_num_layers=int(
                model_config.get("reference_style_num_layers", 4)
            ),
            reference_style_kernel_size=int(
                model_config.get("reference_style_kernel_size", 5)
            ),
            reference_style_dropout=float(
                model_config.get("reference_style_dropout", 0.1)
            ),
            reference_style_fusion_mode=str(
                model_config.get("reference_style_fusion_mode", "gated_add")
            ),
            reference_style_gate_init=float(
                model_config.get("reference_style_gate_init", -3.0)
            ),

            # ------------------------------------------------
            # v6.6.2 prosody / speaker auxiliary loss args
            # v6.6.2 韵律 / 说话人辅助损失参数
            # ------------------------------------------------
            use_v66_energy_loss=_as_bool(model_config.get("use_v66_energy_loss", False), False),
            v66_energy_loss_weight=float(model_config.get("v66_energy_loss_weight", 0.15)),
            v66_energy_loss_type=str(model_config.get("v66_energy_loss_type", "l1")),
            v66_energy_normalize=_as_bool(model_config.get("v66_energy_normalize", True), True),
            use_v66_f0_loss=_as_bool(model_config.get("use_v66_f0_loss", False), False),
            v66_f0_loss_weight=float(model_config.get("v66_f0_loss_weight", 0.05)),
            v66_f0_loss_type=str(model_config.get("v66_f0_loss_type", "l1")),
            v66_f0_normalize=_as_bool(model_config.get("v66_f0_normalize", True), True),
            v66_f0_hidden_dim=int(model_config.get("v66_f0_hidden_dim", 128)),
            use_v66_speaker_loss=_as_bool(model_config.get("use_v66_speaker_loss", False), False),
            v66_speaker_loss_weight=float(model_config.get("v66_speaker_loss_weight", 0.05)),
            v66_speaker_embedding_dim=int(model_config.get("v66_speaker_embedding_dim", 192)),
            v66_require_aux_targets=_as_bool(model_config.get("v66_require_aux_targets", False), False),



            # ------------------------------------------------
            # v6.6.4 Speaker Identity Adapter
            # ------------------------------------------------
            use_v664_speaker_identity_adapter=model_config.get("use_v664_speaker_identity_adapter", False),
            v664_speaker_identity_source=model_config.get("v664_speaker_identity_source", "prompt_or_target"),
            v664_speaker_embedding_dim=model_config.get("v664_speaker_embedding_dim", model_config.get("v66_speaker_embedding_dim", 192)),
            v664_speaker_adapter_hidden_dim=model_config.get("v664_speaker_adapter_hidden_dim", 256),
            v664_speaker_adapter_num_layers=model_config.get("v664_speaker_adapter_num_layers", 2),
            v664_speaker_adapter_dropout=model_config.get("v664_speaker_adapter_dropout", 0.05),
            v664_speaker_adapter_fusion_mode=model_config.get("v664_speaker_adapter_fusion_mode", "gated_add"),
            v664_speaker_adapter_gate_init=model_config.get("v664_speaker_adapter_gate_init", -3.0),
            v664_speaker_adapter_normalize=model_config.get("v664_speaker_adapter_normalize", True),

            # ------------------------------------------------
            # v6.6.3-A Bootstrapper Attention LoRA
            # ------------------------------------------------
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
            # ------------------------------------------------
            # v6.6.3 Bootstrapper LoRA
            # v6.6.3 Bootstrapper LoRA 参数
            # ------------------------------------------------
            use_bootstrapper_lora=model_config.get("use_bootstrapper_lora", False),
            bootstrapper_lora_rank=model_config.get("bootstrapper_lora_rank", 4),
            bootstrapper_lora_alpha=model_config.get("bootstrapper_lora_alpha", 8.0),
            bootstrapper_lora_dropout=model_config.get("bootstrapper_lora_dropout", 0.05),
            bootstrapper_lora_target=model_config.get("bootstrapper_lora_target", "core"),
            bootstrapper_lora_init_scale=model_config.get("bootstrapper_lora_init_scale", 0.01),
            # ------------------------------------------------
            # v6.3 loss args
            # v6.3 loss 参数
            # ------------------------------------------------
            include_v61_base_loss=_as_bool(
                model_config.get("include_v61_base_loss", False),
                False,
            ),
            v61_base_loss_weight=float(model_config.get("v61_base_loss_weight", 1.0)),

            v63_coarse_l1_weight=float(model_config.get("v63_coarse_l1_weight", 2.0)),
            v63_coarse_mse_weight=float(model_config.get("v63_coarse_mse_weight", 0.2)),
            v63_coarse_delta_weight=float(model_config.get("v63_coarse_delta_weight", 0.5)),
            v63_coarse_delta2_weight=float(
                model_config.get("v63_coarse_delta2_weight", 0.2)
            ),

            residual_refiner_loss_weight=float(
                model_config.get("residual_refiner_loss_weight", 1.0)
            ),
            residual_flow_mse_weight=float(
                model_config.get("residual_flow_mse_weight", 1.0)
            ),
            residual_flow_l1_weight=float(
                model_config.get("residual_flow_l1_weight", 0.0)
            ),
            residual_recon_l1_weight=float(
                model_config.get("residual_recon_l1_weight", 1.0)
            ),
            final_recon_l1_weight=float(model_config.get("final_recon_l1_weight", 1.0)),
            final_delta_l1_weight=float(model_config.get("final_delta_l1_weight", 0.3)),
            final_delta2_l1_weight=float(model_config.get("final_delta2_l1_weight", 0.1)),
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

    # --------------------------------------------------------
    # Stage-2 checkpoint loading / 第二阶段 checkpoint 加载
    # --------------------------------------------------------
    def load_stage2_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        EN:
        Load stage-2 checkpoint and build corresponding model.

        ZH:
        加载 stage-2 checkpoint，并构建对应模型。
        """
        ckpt_path = Path(checkpoint_path).resolve()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        if not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint does not contain 'model_state_dict': {ckpt_path}")

        args = _normalize_checkpoint_args(ckpt.get("args", {}))

        model_config = ckpt.get("model_config", None)
        if not isinstance(model_config, dict):
            model_config = dict(args)

        if "model_version" not in model_config and "model_version" in args:
            model_config["model_version"] = args["model_version"]


        # v6.6 compatibility: checkpoints trained with ReferenceAcousticStyleEncoder
        # contain reference_style_* state_dict keys. If an older report/config did
        # not persist the new config fields, recover them from the state_dict.
        state_dict = ckpt["model_state_dict"]
        has_reference_style_state = any(
            str(k).startswith("reference_style_encoder.")
            or str(k).startswith("reference_style_to_model.")
            or str(k) == "reference_style_gate"
            for k in state_dict.keys()
        )
        has_v664_speaker_identity_state = any(
            str(k).startswith("speaker_identity_adapter.")
            for k in state_dict.keys()
        )
        if has_v664_speaker_identity_state and not _as_bool(
            model_config.get("use_v664_speaker_identity_adapter", False),
            False,
        ):
            print(
                "[INFO] Detected v6.6.4 speaker identity adapter weights; "
                "enabling use_v664_speaker_identity_adapter for inference."
            )
            model_config = dict(model_config)
            model_config["use_v664_speaker_identity_adapter"] = True
            model_config.setdefault("v664_speaker_identity_source", "prompt_or_target")
            model_config.setdefault("v664_speaker_embedding_dim", model_config.get("v66_speaker_embedding_dim", 192))
            model_config.setdefault("v664_speaker_adapter_hidden_dim", 256)
            model_config.setdefault("v664_speaker_adapter_num_layers", 2)
            model_config.setdefault("v664_speaker_adapter_dropout", 0.05)
            model_config.setdefault("v664_speaker_adapter_fusion_mode", "gated_add")
            model_config.setdefault("v664_speaker_adapter_gate_init", -3.0)
            model_config.setdefault("v664_speaker_adapter_normalize", True)

        has_bootstrapper_attention_lora_state = any(
            any(token in str(k) for token in [
                "lora_q_A", "lora_q_B",
                "lora_k_A", "lora_k_B",
                "lora_v_A", "lora_v_B",
                "lora_o_A", "lora_o_B",
                "attention_lora_gate",
            ])
            for k in state_dict.keys()
        )
        if has_bootstrapper_attention_lora_state and not _as_bool(
            model_config.get("use_bootstrapper_attention_lora", False),
            False,
        ):
            print(
                "[INFO] Detected bootstrapper attention LoRA weights; "
                "enabling use_bootstrapper_attention_lora for inference."
            )
            model_config = dict(model_config)
            model_config["use_bootstrapper_attention_lora"] = True
            model_config.setdefault("bootstrapper_attention_lora_rank", 4)
            model_config.setdefault("bootstrapper_attention_lora_alpha", 8.0)
            model_config.setdefault("bootstrapper_attention_lora_dropout", 0.05)
            model_config.setdefault("bootstrapper_attention_lora_target", "cross_only")
            model_config.setdefault("bootstrapper_attention_lora_gate_init", -3.0)
            model_config.setdefault("bootstrapper_attention_lora_enable_q", True)
            model_config.setdefault("bootstrapper_attention_lora_enable_k", True)
            model_config.setdefault("bootstrapper_attention_lora_enable_v", True)
            model_config.setdefault("bootstrapper_attention_lora_enable_o", True)

        has_bootstrapper_lora_state = any(
            ".lora_A." in str(k) or ".lora_B." in str(k)
            for k in state_dict.keys()
        )
        if has_bootstrapper_lora_state and not _as_bool(
            model_config.get("use_bootstrapper_lora", False),
            False,
        ):
            print(
                "[INFO] Detected bootstrapper LoRA weights in checkpoint; "
                "enabling use_bootstrapper_lora for inference model construction."
            )
            model_config = dict(model_config)
            model_config["use_bootstrapper_lora"] = True
            model_config.setdefault("bootstrapper_lora_rank", 4)
            model_config.setdefault("bootstrapper_lora_alpha", 8.0)
            model_config.setdefault("bootstrapper_lora_dropout", 0.05)
            model_config.setdefault("bootstrapper_lora_target", "core")
            model_config.setdefault("bootstrapper_lora_init_scale", 0.01)

        if has_reference_style_state and not _as_bool(
            model_config.get("use_reference_acoustic_style", False),
            False,
        ):
            print(
                "[INFO] Detected reference_style_* weights in checkpoint; "
                "enabling use_reference_acoustic_style for inference model construction."
            )
            model_config["use_reference_acoustic_style"] = True
            model_config.setdefault("reference_acoustic_dim", 80)
            model_config.setdefault("reference_style_dim", 256)
            model_config.setdefault("reference_style_hidden_dim", 256)
            model_config.setdefault("reference_style_num_layers", 4)
            model_config.setdefault("reference_style_kernel_size", 5)
            model_config.setdefault("reference_style_dropout", 0.1)
            model_config.setdefault("reference_style_fusion_mode", "gated_add")
            model_config.setdefault("reference_style_gate_init", -3.0)

            # v6.6.2 optional auxiliary heads can add state_dict keys.
            # Enable corresponding heads if checkpoint contains their weights but config is incomplete.
            has_v66_f0_head_state = any(str(k).startswith("v66_f0_head.") for k in state_dict.keys())
            has_v66_speaker_head_state = any(str(k).startswith("v66_speaker_head.") for k in state_dict.keys())
            if has_v66_f0_head_state and not _as_bool(model_config.get("use_v66_f0_loss", False), False):
                print("[INFO] Detected v66_f0_head.* weights; enabling use_v66_f0_loss for model construction.")
                model_config["use_v66_f0_loss"] = True
                model_config.setdefault("v66_f0_hidden_dim", 128)
            if has_v66_speaker_head_state and not _as_bool(model_config.get("use_v66_speaker_loss", False), False):
                print("[INFO] Detected v66_speaker_head.* weights; enabling use_v66_speaker_loss for model construction.")
                model_config["use_v66_speaker_loss"] = True
                model_config.setdefault("v66_speaker_embedding_dim", 192)

        model = self._build_stage2_model_from_config(model_config).to(self.device)

        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=strict)

        if not strict:
            print(f"[INFO] stage2 load_state_dict result: {load_result}")

        model.eval()

        self.stage2_model = model
        self.stage2_checkpoint_meta = {
            "checkpoint_path": str(ckpt_path),
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "best_epoch": ckpt.get("best_epoch"),
            "args": args,
            "model_config": model_config,
            "model_version": str(model_config.get("model_version", "unknown")),
            "model_variant": getattr(model, "model_variant", None),
        }

        return self.stage2_checkpoint_meta

    def ensure_stage2_loaded(
        self,
        checkpoint_path: str | Path | None = None,
        strict: bool = True,
    ) -> None:
        """
        EN:
        Ensure stage-2 model is already loaded.

        ZH:
        确保 stage-2 模型已经加载。
        """
        if self.stage2_model is None:
            if checkpoint_path is None:
                raise ValueError("stage2_model is not loaded. Please provide stage2 checkpoint_path.")

            self.load_stage2_checkpoint(
                checkpoint_path=checkpoint_path,
                strict=strict,
            )

    # --------------------------------------------------------
    # Vocoder setup / Vocoder 加载
    # --------------------------------------------------------
    def setup_vocoder(
        self,
        vocoder_type: str = "hifigan",
        profile: str = "universal_v1",
        hifigan_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        EN:
        Create and load vocoder adapter if needed.

        ZH:
        如有需要，创建并加载 vocoder adapter。
        """
        if self.vocoder_adapter is None:
            self.vocoder_adapter = Stage2VocoderAdapter(
                vocoder_type=vocoder_type,
                profile=profile,
                hifigan_root=hifigan_root,
                device=str(self.device),
            )

        self.vocoder_adapter.ensure_loaded(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )

    # --------------------------------------------------------
    # Prepare inputs / 准备输入
    # --------------------------------------------------------
    @torch.inference_mode()
    def prepare_stage2_inputs(
        self,
        text: str,
        language: str,
        prompt_wav_path: str,
        stage1_checkpoint_path: str | None = None,
        sovits_checkpoint_path: str | None = None,
    ) -> Stage2Inputs:
        """
        EN:
        Run frontend, prompt tokenizer and stage-1 model to build Stage2Inputs.

        ZH:
        运行 frontend、prompt tokenizer 和 stage-1 模型，构建 Stage2Inputs。
        """
        phoneme_ids, phoneme_lens, bert_feature, norm_text = self.frontend.prepare_ids_and_bert(
            text=text,
            language=language,
        )

        prompt_kwargs = self._filter_supported_kwargs(
            self.prompt_tokenizer.extract_prompt_semantic_from_wav,
            {
                "wav_path": prompt_wav_path,
                "sovits_checkpoint_path": sovits_checkpoint_path,
            },
        )
        prompt_tokens = self.prompt_tokenizer.extract_prompt_semantic_from_wav(**prompt_kwargs)

        stage1_kwargs = self._filter_supported_kwargs(
            self.stage1.generate_semantic,
            {
                "phoneme_ids": phoneme_ids,
                "phoneme_lens": phoneme_lens,
                "bert_feature": bert_feature,
                "prompt_tokens": prompt_tokens,
                "checkpoint_path": stage1_checkpoint_path,
            },
        )
        pred_semantic, aux = self.stage1.generate_semantic(**stage1_kwargs)

        phoneme_ids = self._to_device_if_tensor(phoneme_ids)
        phoneme_lens = self._to_device_if_tensor(phoneme_lens)
        bert_feature = self._to_device_if_tensor(bert_feature)
        prompt_tokens = self._to_device_if_tensor(prompt_tokens)
        pred_semantic = self._to_device_if_tensor(pred_semantic)

        semantic_lengths = torch.full(
            size=(pred_semantic.shape[0],),
            fill_value=pred_semantic.shape[1],
            dtype=torch.long,
            device=pred_semantic.device,
        )

        prompt_lengths = None
        if torch.is_tensor(prompt_tokens):
            prompt_lengths = torch.full(
                size=(prompt_tokens.shape[0],),
                fill_value=prompt_tokens.shape[1],
                dtype=torch.long,
                device=prompt_tokens.device,
            )

        v66_extras: dict[str, Any] = {}
        stage2_model = self.stage2_model
        if stage2_model is not None and bool(getattr(stage2_model, "use_reference_acoustic_style", False)):
            v66_extras = self._build_v66_prompt_acoustic_extras(prompt_wav_path)
            v66_extras["v66_reference_style_attached"] = True

        return Stage2Inputs(
            semantic_tokens=pred_semantic.long(),
            semantic_lengths=semantic_lengths,
            phoneme_ids=phoneme_ids,
            phoneme_lens=phoneme_lens,
            bert_feature=bert_feature,
            prompt_tokens=prompt_tokens.long() if torch.is_tensor(prompt_tokens) else None,
            prompt_lengths=prompt_lengths,
            norm_text=norm_text,
            raw_text=text,
            language=language,
            extras={
                "stage1_aux": aux,
                "prompt_wav_path": str(Path(prompt_wav_path).resolve()),
                **v66_extras,
            },
        )

    # --------------------------------------------------------
    # Stage-2 mel inference / 第二阶段 mel 推理
    # --------------------------------------------------------
    @torch.inference_mode()
    def run_to_mel(
        self,
        text: str,
        language: str,
        prompt_wav_path: str,
        stage2_checkpoint_path: str | Path | None = None,
        stage1_checkpoint_path: str | None = None,
        sovits_checkpoint_path: str | None = None,
        num_steps: int = 32,
        temperature: float = 1.0,
        strict: bool = True,
        guidance_scale: float = 1.0,
        use_heun: bool = False,
        target_lengths_override: torch.Tensor | list[int] | tuple[int, ...] | int | None = None,

        # v6.2 bootstrap sampling
        sampling_start_mode: str = "pure_noise",
        bootstrap_t_start: float = 0.075,
        bootstrap_noise_temperature: float | None = None,
        return_bootstrap_mel: bool = False,
    ) -> dict[str, Any]:
        """
        EN:
        Run full frontend + stage1 + stage2 path and return predicted mel.

        ZH:
        运行 frontend + stage1 + stage2，返回预测 mel。
        """
        self.ensure_stage2_loaded(
            checkpoint_path=stage2_checkpoint_path,
            strict=strict,
        )
        assert self.stage2_model is not None

        batch = self.prepare_stage2_inputs(
            text=text,
            language=language,
            prompt_wav_path=prompt_wav_path,
            stage1_checkpoint_path=stage1_checkpoint_path,
            sovits_checkpoint_path=sovits_checkpoint_path,
        )
        batch = self._move_stage2_inputs_to_device(batch)

        normalized_target_lengths = self._normalize_target_lengths_override(
            target_lengths_override=target_lengths_override,
            batch_size=batch.semantic_tokens.shape[0],
        )

        sampling_start_mode = str(sampling_start_mode).strip().lower()

        if sampling_start_mode in {
            "coarse_only",
            "coarse",
            "v6_3_coarse",
            "v6.3_coarse",
            "v63_coarse",
            "bootstrap_only",
        }:
            if not hasattr(self.stage2_model, "sample_coarse_only"):
                raise RuntimeError(
                    "sampling_start_mode='coarse_only' requires a v6.3 model with "
                    "sample_coarse_only(...), but the loaded model does not support it."
                )

            stage2_outputs: Stage2Outputs = self.stage2_model.sample_coarse_only(
                batch=batch,
                target_lengths=normalized_target_lengths,
                return_intermediates=bool(return_bootstrap_mel),
            )

        elif sampling_start_mode in {
            "residual_refine",
            "residual",
            "v6_3",
            "v6.3",
            "v63",
            "v6_3_residual",
        }:
            if not hasattr(self.stage2_model, "sample_with_residual_refiner"):
                raise RuntimeError(
                    "sampling_start_mode='residual_refine' requires a v6.3 model with "
                    "sample_with_residual_refiner(...), but the loaded model does not support it."
                )

            stage2_outputs: Stage2Outputs = self.stage2_model.sample_with_residual_refiner(
                batch=batch,
                num_steps=num_steps,
                target_lengths=normalized_target_lengths,
                temperature=(
                    float(bootstrap_noise_temperature)
                    if bootstrap_noise_temperature is not None
                    else float(temperature)
                ),
                use_heun=use_heun,
                guidance_scale=guidance_scale,
                return_intermediates=bool(return_bootstrap_mel),
            )

        elif sampling_start_mode in {
            "bootstrap",
            "v6_2_bootstrap",
            "coarse_bootstrap",
            "v6_3_bootstrap",
        }:
            if not hasattr(self.stage2_model, "sample_with_bootstrap"):
                raise RuntimeError(
                    "sampling_start_mode='bootstrap' requires a model with "
                    "sample_with_bootstrap(...), but the loaded model does not support it."
                )

            stage2_outputs: Stage2Outputs = self.stage2_model.sample_with_bootstrap(
                batch=batch,
                num_steps=num_steps,
                target_lengths=normalized_target_lengths,
                temperature=temperature,
                use_heun=use_heun,
                guidance_scale=guidance_scale,
                bootstrap_t_start=float(bootstrap_t_start),
                bootstrap_noise_temperature=bootstrap_noise_temperature,
                return_bootstrap_mel=bool(return_bootstrap_mel),
            )

        elif sampling_start_mode in {"pure_noise", "noise", "default", "v6_1"}:
            stage2_outputs: Stage2Outputs = self.stage2_model.sample(
                batch=batch,
                num_steps=num_steps,
                target_lengths=normalized_target_lengths,
                temperature=temperature,
                use_heun=use_heun,
                guidance_scale=guidance_scale,
            )

        else:
            raise ValueError(
                "Unsupported sampling_start_mode: "
                f"{sampling_start_mode}. Expected one of: "
                "pure_noise / bootstrap / residual_refine / coarse_only."
            )


        return {
            "batch": batch,
            "stage2_outputs": stage2_outputs,
            "predicted_mel": stage2_outputs.acoustic,
            "lengths": stage2_outputs.lengths,
            "norm_text": batch.norm_text,

            # v6.1
            "stage2_aux": stage2_outputs.aux,
        }

    # Backward-compatible alias
    run = run_to_mel

    # --------------------------------------------------------
    # Vocoder decode / Vocoder 解码
    # --------------------------------------------------------
    @torch.inference_mode()
    def decode_mel_to_wav(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
        vocoder_type: str = "hifigan",
        vocoder_profile: str = "universal_v1",
        hifigan_root: str | Path | None = None,
        vocoder_checkpoint_path: str | Path | None = None,
        vocoder_config_path: str | Path | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        EN:
        Decode mel to waveform using vocoder adapter.

        ZH:
        使用 vocoder adapter 将 mel 解码成 waveform。
        """
        self.setup_vocoder(
            vocoder_type=vocoder_type,
            profile=vocoder_profile,
            hifigan_root=hifigan_root,
            checkpoint_path=vocoder_checkpoint_path,
            config_path=vocoder_config_path,
        )

        assert self.vocoder_adapter is not None

        mel = mel.to(self.device)

        if lengths is not None:
            lengths = lengths.to(self.device)

        waveform = self.vocoder_adapter.decode(
            mel=mel,
            lengths=lengths,
        )
        sample_rate = int(self.vocoder_adapter.sample_rate or 22050)

        return waveform, sample_rate

    @torch.inference_mode()
    def run_to_audio(
        self,
        text: str,
        language: str,
        prompt_wav_path: str,
        stage2_checkpoint_path: str | Path,
        stage1_checkpoint_path: str | None = None,
        sovits_checkpoint_path: str | None = None,
        num_steps: int = 32,
        temperature: float = 1.0,
        vocoder_type: str = "hifigan",
        vocoder_profile: str = "universal_v1",
        hifigan_root: str | Path | None = None,
        vocoder_checkpoint_path: str | Path | None = None,
        vocoder_config_path: str | Path | None = None,
        strict: bool = True,
        guidance_scale: float = 1.0,
        use_heun: bool = False,
        target_lengths_override: torch.Tensor | list[int] | tuple[int, ...] | int | None = None,

        # v6.2 bootstrap sampling
        sampling_start_mode: str = "pure_noise",
        bootstrap_t_start: float = 0.075,
        bootstrap_noise_temperature: float | None = None,
        return_bootstrap_mel: bool = False,
    ) -> dict[str, Any]:
        """
        EN:
        Run full text/prompt -> mel -> waveform inference.

        ZH:
        运行完整 text/prompt -> mel -> waveform 推理流程。
        """
        mel_result = self.run_to_mel(
            text=text,
            language=language,
            prompt_wav_path=prompt_wav_path,
            stage2_checkpoint_path=stage2_checkpoint_path,
            stage1_checkpoint_path=stage1_checkpoint_path,
            sovits_checkpoint_path=sovits_checkpoint_path,
            num_steps=num_steps,
            temperature=temperature,
            strict=strict,
            guidance_scale=guidance_scale,
            use_heun=use_heun,
            target_lengths_override=target_lengths_override,

            # v6.2 bootstrap sampling
            sampling_start_mode=sampling_start_mode,
            bootstrap_t_start=bootstrap_t_start,
            bootstrap_noise_temperature=bootstrap_noise_temperature,
            return_bootstrap_mel=return_bootstrap_mel,
        )

        waveform, sample_rate = self.decode_mel_to_wav(
            mel=mel_result["predicted_mel"],
            lengths=mel_result["lengths"],
            vocoder_type=vocoder_type,
            vocoder_profile=vocoder_profile,
            hifigan_root=hifigan_root,
            vocoder_checkpoint_path=vocoder_checkpoint_path,
            vocoder_config_path=vocoder_config_path,
        )

        warnings: list[str] = []

        vocoder_meta = (
            self.vocoder_adapter.backend_meta
            if self.vocoder_adapter is not None
            else {}
        )

        stage2_args = self.stage2_checkpoint_meta.get("args", {})
        model_config = self.stage2_checkpoint_meta.get("model_config", {})

        acoustic_rate_hz = float(
            model_config.get(
                "acoustic_rate_hz",
                stage2_args.get("acoustic_rate_hz", DEFAULT_ACOUSTIC_RATE_HZ),
            )
        )

        semantic_rate_hz = float(
            model_config.get(
                "semantic_rate_hz",
                stage2_args.get("semantic_rate_hz", DEFAULT_SEMANTIC_RATE_HZ),
            )
        )

        vocoder_frame_rate_hz = None

        if (
            self.vocoder_adapter is not None
            and self.vocoder_adapter.sample_rate
            and self.vocoder_adapter.hop_size
        ):
            vocoder_frame_rate_hz = float(self.vocoder_adapter.sample_rate) / float(
                self.vocoder_adapter.hop_size
            )

            if abs(vocoder_frame_rate_hz - acoustic_rate_hz) > 1e-3:
                warnings.append(
                    "Vocoder mel frame rate does not match current stage-2 acoustic_rate_hz. "
                    f"stage2={acoustic_rate_hz:.6f} Hz, "
                    f"vocoder={vocoder_frame_rate_hz:.6f} Hz. "
                    "First-pass inference may still work, but audio quality / timing may be affected."
                )

        stage2_aux = mel_result.get("stage2_aux", {})

        return {
            **mel_result,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "stage2_checkpoint_meta": self.stage2_checkpoint_meta,
            "vocoder_meta": vocoder_meta,
            "warnings": warnings,
            "vocoder_frame_rate_hz": vocoder_frame_rate_hz,

            # v6.1 stable summary fields
            "stage2_aux": stage2_aux,
            "stage2_acoustic_rate_hz": acoustic_rate_hz,
            "stage2_semantic_rate_hz": semantic_rate_hz,
            "stage2_model_version": self.stage2_checkpoint_meta.get("model_version"),
            "stage2_model_variant": self.stage2_checkpoint_meta.get("model_variant"),

            # v6.2 sampling summary
            "sampling_start_mode": sampling_start_mode,
            "bootstrap_t_start": bootstrap_t_start,
            "bootstrap_noise_temperature": bootstrap_noise_temperature,
        }
