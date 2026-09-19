from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.interfaces.stage2_io import Stage2Inputs, Stage2Outputs
from src.encoders.stage2_condition_encoder import Stage2ConditionEncoder, Stage2ConditionBundleV6
from src.encoders.reference_acoustic_style_encoder import (
    ReferenceAcousticStyleEncoder,
    ReferenceAcousticStyleEncoderConfig,
)
from src.flow_matching.stage2_time_embedding import Stage2TimeEmbedding
from src.flow_matching.stage2_fm_blocks import (
    Stage2MelRefinementPostnet,
    Stage2DiTStyleFMBlockV61,
)
from src.losses.v66_prosody_losses import (
    compute_v66_energy_loss,
    compute_v66_f0_loss,
    compute_v66_speaker_loss,
)
from src.adapters.lora_layers import (
    apply_lora_to_named_linears,
    count_lora_parameters,
)
from src.adapters.attention_lora_layers import (
    apply_attention_lora_to_named_mha,
    count_attention_lora_parameters,
)
from src.adapters.speaker_identity_adapter import (
    SpeakerIdentityAdapter,
    SpeakerIdentityAdapterConfig,
    select_v664_speaker_identity_embedding,
)
from src.models.stage2_bootstrapper import (
    CoarseMelBootstrapper,
    CoarseMelBootstrapperConfig,
    ConformerMelBootstrapper,
    ConformerMelBootstrapperConfig,
    compute_coarse_mel_losses,
    build_bootstrap_start,
)
from src.models.stage2_residual_refiner import (
    ResidualFlowRefiner,
    ResidualFlowRefinerConfig,
    compute_residual_refiner_training_loss,
)


DEFAULT_SEMANTIC_RATE_HZ = 25.0
DEFAULT_ACOUSTIC_RATE_HZ = 22050.0 / 256.0  # 86.1328125


def make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    EN:
    lengths: (B,)
    return: (B, max_len), bool, True for valid positions

    ZH:
    根据长度张量生成有效位置 mask。
    返回形状 (B, max_len)，True 表示有效位置。
    """
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def make_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    EN:
    Return key_padding_mask for PyTorch attention:
    shape (B, max_len), True means PAD / ignore.

    ZH:
    生成给 PyTorch attention 使用的 key_padding_mask：
    形状 (B, max_len)，True 表示 PAD / 忽略。
    """
    return ~make_length_mask(lengths, max_len)


class Stage2SinusoidalPositionEmbedding(nn.Module):
    """
    EN:
    Simple sinusoidal positional embedding for acoustic frames.

    ZH:
    给 acoustic frame 使用的简单正弦位置嵌入。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        length = max(int(length), 1)

        positions = torch.arange(length, device=device).float()
        half_dim = self.hidden_dim // 2

        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device).float()
            / max(half_dim - 1, 1)
        )
        phase = positions[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        if self.hidden_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb.unsqueeze(0)


class Stage2LengthPredictor(nn.Module):
    """
    EN:
    Predict utterance-level log acoustic/semantic length ratio.

    ZH:
    预测整句级别的 log(acoustic_len / semantic_len)。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Stage2FlowMatchingAcousticModelV61(nn.Module):
    """
    EN:
    VoiceLab stage2 Flow Matching acoustic model v6.1.

    Main upgrades:
    - frame-level content as main condition
    - v6.1 DiT-style block with content injection
    - local short-time detail modeling
    - semantic guide as auxiliary condition
    - gentle length correction during inference
    - keeps the useful v5/v6 loss family, but avoids adding too many new losses

    ZH:
    VoiceLab 第二阶段 Flow Matching 声学模型 v6.1。

    主要升级：
    - frame-level content 作为主条件流
    - 使用带 content injection 的 v6.1 DiT-style block
    - 增强局部短时细节建模
    - semantic guide 作为辅助条件
    - 推理阶段使用温和长度修正
    - 保留 v5/v6 中有效的 loss 家族，但不继续无节制堆 loss
    """

    def __init__(
        self,
        acoustic_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        semantic_vocab_size: int = 1024,
        semantic_rate_hz: float = DEFAULT_SEMANTIC_RATE_HZ,
        acoustic_rate_hz: float = DEFAULT_ACOUSTIC_RATE_HZ,
        dropout: float = 0.1,

        # ----------------------------------------------------
        # Self condition / 自条件
        # ----------------------------------------------------
        use_self_condition: bool = True,
        self_condition_prob: float = 0.5,

        # ----------------------------------------------------
        # Condition encoder params / 条件编码器参数
        # ----------------------------------------------------
        phoneme_vocab_size: int = 4096,
        phoneme_embed_dim: int = 256,
        bert_dim: int = 1024,
        content_dim: int = 384,
        content_refiner_layers: int = 4,
        content_frame_dim: int = 384,
        content_expander_layers: int = 2,
        style_dim: int = 256,

        # ----------------------------------------------------
        # v6.1 condition params / v6.1 条件参数
        # ----------------------------------------------------
        content_main_dim: int = 512,
        content_injection_gate_init: float = 0.5,
        content_refiner_dilations: tuple[int, ...] = (1, 2),
        content_boundary_enhance: bool = True,
        content_boundary_kernel_size: int = 3,
        content_boundary_residual_scale: float = 0.3,
        content_refiner_residual_scale: float = 0.5,

        # ----------------------------------------------------
        # v6.1 block params / v6.1 block 参数
        # ----------------------------------------------------
        ff_mult: int = 4,
        local_detail_kernel_size: int = 5,
        local_detail_dilation_cycle: tuple[int, ...] = (1, 2),
        local_detail_residual_scale: float = 0.5,
        semantic_guide_gate_init: float = 0.3,
        use_semantic_guide_cross_attn: bool = True,

        # ----------------------------------------------------
        # v6.4.1 continuous semantic condition
        # v6.4.1 continuous semantic 条件
        # ----------------------------------------------------
        use_continuous_semantic: bool = False,
        continuous_semantic_dim: int = 768,
        continuous_semantic_fusion_mode: str = "gated_add",
        continuous_semantic_gate_init: float = -2.0,
        continuous_semantic_dropout: float = 0.0,

        # ----------------------------------------------------
        # Loss weights / 损失权重
        # ----------------------------------------------------
        fm_loss_weight: float = 1.0,
        recon_loss_weight: float = 0.20,
        refined_recon_loss_weight: float = 1.00,
        delta_loss_weight: float = 0.15,
        delta2_loss_weight: float = 0.05,
        mid_content_aux_weight: float = 0.10,
        final_content_aux_weight: float = 0.20,
        span_focus_aux_weight: float = 0.20,
        span_focus_ratio_min: float = 0.10,
        span_focus_ratio_max: float = 0.35,
        span_recon_weight: float = 0.50,
        span_content_weight: float = 0.50,
        recon_loss_type: str = "l1",

        # ----------------------------------------------------
        # Conditional dropout / 条件 dropout
        # ----------------------------------------------------
        cond_drop_prob_content: float = 0.05,
        cond_drop_prob_semantic: float = 0.10,
        cond_drop_prob_style: float = 0.10,
        cond_drop_all_prob: float = 0.05,

        # ----------------------------------------------------
        # Postnet / 后处理网络
        # ----------------------------------------------------
        postnet_num_layers: int = 3,
        postnet_dropout: float = 0.1,
        postnet_conv_kernel_size: int = 5,
        detach_coarse_for_refinement: bool = True,

        # ----------------------------------------------------
        # Training dynamics / 训练动力学
        # ----------------------------------------------------
        t_sampling_mode: str = "near_clean",
        t_bias_power: float = 2.0,
        t_min: float = 0.05,
        t_max: float = 0.98,

        # ----------------------------------------------------
        # Frequency weighting / 频带加权
        # ----------------------------------------------------
        freq_weight_min: float = 1.0,
        freq_weight_max: float = 2.0,
        freq_weight_power: float = 1.5,

        # ----------------------------------------------------
        # Content auxiliary supervision / 内容辅助监督
        # ----------------------------------------------------
        content_head_dropout: float = 0.1,

        # ----------------------------------------------------
        # Length correction / 长度修正
        # ----------------------------------------------------
        enable_length_clamp: bool = True,
        length_correction_mode: str = "gentle_ratio",
        length_ratio_min: float = 2.6,
        length_ratio_max: float = 3.8,
        length_bias_scale: float = 0.95,

        # ----------------------------------------------------
        # Extra compatibility args / 兼容旧 config 的额外参数
        # ----------------------------------------------------
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()

        self.model_version = "v6_1"
        self.model_variant = "v6.1"

        self.acoustic_dim = int(acoustic_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        self.semantic_vocab_size = int(semantic_vocab_size)
        self.semantic_rate_hz = float(semantic_rate_hz)
        self.acoustic_rate_hz = float(acoustic_rate_hz)

        self.use_self_condition = bool(use_self_condition)
        self.self_condition_prob = float(self_condition_prob)

        self.phoneme_vocab_size = int(phoneme_vocab_size)
        self.content_dim = int(content_dim)
        self.content_frame_dim = int(content_frame_dim)
        self.content_main_dim = int(content_main_dim)
        self.style_dim = int(style_dim)

        self.use_continuous_semantic = bool(use_continuous_semantic)
        self.continuous_semantic_dim = int(continuous_semantic_dim)
        self.continuous_semantic_fusion_mode = str(continuous_semantic_fusion_mode)
        self.continuous_semantic_gate_init = float(continuous_semantic_gate_init)
        self.continuous_semantic_dropout = float(continuous_semantic_dropout)

        # ----------------------------------------------------
        # Loss weights / 损失权重
        # ----------------------------------------------------
        self.fm_loss_weight = float(fm_loss_weight)
        self.recon_loss_weight = float(recon_loss_weight)
        self.refined_recon_loss_weight = float(refined_recon_loss_weight)
        self.delta_loss_weight = float(delta_loss_weight)
        self.delta2_loss_weight = float(delta2_loss_weight)
        self.mid_content_aux_weight = float(mid_content_aux_weight)
        self.final_content_aux_weight = float(final_content_aux_weight)
        self.span_focus_aux_weight = float(span_focus_aux_weight)
        self.span_focus_ratio_min = float(span_focus_ratio_min)
        self.span_focus_ratio_max = float(span_focus_ratio_max)
        self.span_recon_weight = float(span_recon_weight)
        self.span_content_weight = float(span_content_weight)
        self.recon_loss_type = str(recon_loss_type).lower()

        # ----------------------------------------------------
        # Conditional dropout / 条件 dropout
        # ----------------------------------------------------
        self.cond_drop_prob_content = float(cond_drop_prob_content)
        self.cond_drop_prob_semantic = float(cond_drop_prob_semantic)
        self.cond_drop_prob_style = float(cond_drop_prob_style)
        self.cond_drop_all_prob = float(cond_drop_all_prob)

        self.detach_coarse_for_refinement = bool(detach_coarse_for_refinement)

        # ----------------------------------------------------
        # Training dynamics / 训练动力学
        # ----------------------------------------------------
        self.t_sampling_mode = str(t_sampling_mode).lower()
        self.t_bias_power = float(t_bias_power)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        # ----------------------------------------------------
        # Length correction / 长度修正
        # ----------------------------------------------------
        self.enable_length_clamp = bool(enable_length_clamp)
        self.length_correction_mode = str(length_correction_mode)
        self.length_ratio_min = float(length_ratio_min)
        self.length_ratio_max = float(length_ratio_max)
        self.length_bias_scale = float(length_bias_scale)

        # ----------------------------------------------------
        # Condition encoder / 条件编码器
        # ----------------------------------------------------
        self.condition_encoder = Stage2ConditionEncoder(
            semantic_vocab_size=semantic_vocab_size,
            semantic_embed_dim=256,
            hidden_dim=hidden_dim,
            bert_dim=bert_dim,

            phoneme_vocab_size=phoneme_vocab_size,
            phoneme_embed_dim=phoneme_embed_dim,
            content_dim=content_dim,
            content_refiner_layers=content_refiner_layers,
            content_frame_dim=content_frame_dim,
            content_expander_layers=content_expander_layers,
            style_dim=style_dim,
            dropout=dropout,

            content_main_dim=content_main_dim,
            content_refiner_dilations=content_refiner_dilations,
            content_boundary_enhance=content_boundary_enhance,
            content_boundary_kernel_size=content_boundary_kernel_size,
            content_boundary_residual_scale=content_boundary_residual_scale,
            content_refiner_residual_scale=content_refiner_residual_scale,
            semantic_guide_dim=hidden_dim,

            use_continuous_semantic=use_continuous_semantic,
            continuous_semantic_dim=continuous_semantic_dim,
            continuous_semantic_fusion_mode=continuous_semantic_fusion_mode,
            continuous_semantic_gate_init=continuous_semantic_gate_init,
            continuous_semantic_dropout=continuous_semantic_dropout,
        )

        self.time_embedding = Stage2TimeEmbedding(hidden_dim)
        self.position_embedding = Stage2SinusoidalPositionEmbedding(hidden_dim)

        self.input_proj = nn.Linear(acoustic_dim, hidden_dim)

        if self.use_self_condition:
            self.self_cond_proj = nn.Linear(acoustic_dim, hidden_dim)
        else:
            self.self_cond_proj = None

        if len(local_detail_dilation_cycle) <= 0:
            raise ValueError("local_detail_dilation_cycle must contain at least one value.")

        # ----------------------------------------------------
        # v6.1 main blocks / v6.1 主干 blocks
        # ----------------------------------------------------
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = int(local_detail_dilation_cycle[i % len(local_detail_dilation_cycle)])
            self.blocks.append(
                Stage2DiTStyleFMBlockV61(
                    hidden_dim=hidden_dim,
                    content_dim=content_main_dim,
                    style_dim=style_dim,
                    semantic_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    local_kernel_size=local_detail_kernel_size,
                    local_dilation=dilation,
                    local_residual_scale=local_detail_residual_scale,
                    content_gate_init=content_injection_gate_init,
                    semantic_gate_init=semantic_guide_gate_init,
                    use_semantic_guide_cross_attn=use_semantic_guide_cross_attn,
                )
            )

        self.output_proj = nn.Linear(hidden_dim, acoustic_dim)

        # ----------------------------------------------------
        # Refinement postnet / 细化 postnet
        # ----------------------------------------------------
        self.frame_cond_to_hidden = nn.Linear(content_main_dim, hidden_dim)

        self.refinement_postnet = Stage2MelRefinementPostnet(
            acoustic_dim=acoustic_dim,
            hidden_dim=hidden_dim,
            num_layers=postnet_num_layers,
            num_heads=num_heads,
            dropout=postnet_dropout,
            conv_kernel_size=postnet_conv_kernel_size,
        )

        # ----------------------------------------------------
        # Content auxiliary heads / 内容辅助头
        # ----------------------------------------------------
        self.mid_content_aux_head = nn.Sequential(
            nn.Dropout(content_head_dropout),
            nn.Linear(hidden_dim, self.phoneme_vocab_size),
        )
        self.final_content_aux_head = nn.Sequential(
            nn.Dropout(content_head_dropout),
            nn.Linear(hidden_dim, self.phoneme_vocab_size),
        )

        # ----------------------------------------------------
        # Frequency weights / 频带权重
        # ----------------------------------------------------
        weights = self._build_frequency_weights(
            acoustic_dim=acoustic_dim,
            w_min=freq_weight_min,
            w_max=freq_weight_max,
            power=freq_weight_power,
        )
        self.register_buffer("mel_bin_weights", weights, persistent=False)

    # --------------------------------------------------------
    # Basic helpers / 基础辅助函数
    # --------------------------------------------------------
    def _build_frequency_weights(
        self,
        acoustic_dim: int,
        w_min: float,
        w_max: float,
        power: float,
    ) -> torch.Tensor:
        base = torch.linspace(0.0, 1.0, acoustic_dim)
        weights = w_min + (w_max - w_min) * torch.pow(base, power)
        weights = weights / weights.mean().clamp_min(1e-8)
        return weights.float()

    def _resolve_semantic_lengths(self, batch: Stage2Inputs) -> torch.Tensor:
        if batch.semantic_lengths is not None:
            return batch.semantic_lengths.long().to(batch.semantic_tokens.device)

        B, T_sem = batch.semantic_tokens.shape
        return torch.full(
            size=(B,),
            fill_value=T_sem,
            dtype=torch.long,
            device=batch.semantic_tokens.device,
        )

    def infer_acoustic_lengths(self, batch: Stage2Inputs) -> torch.Tensor:
        """
        EN:
        Infer acoustic lengths for inference.

        v6.1 default:
        - base ratio = acoustic_rate_hz / semantic_rate_hz
        - optional gentle ratio bias and clamp

        ZH:
        推理阶段推断 acoustic length。

        v6.1 默认：
        - 基础比例 = acoustic_rate_hz / semantic_rate_hz
        - 可选温和 bias 和 clamp
        """
        sem_lengths = self._resolve_semantic_lengths(batch)

        base_ratio = self.acoustic_rate_hz / max(self.semantic_rate_hz, 1e-8)
        ratio = float(base_ratio)

        if self.enable_length_clamp and self.length_correction_mode == "gentle_ratio":
            ratio = ratio * self.length_bias_scale
            ratio = max(self.length_ratio_min, min(self.length_ratio_max, ratio))

        acoustic_lengths = torch.round(sem_lengths.float() * ratio).long().clamp_min(1)
        return acoustic_lengths

    def _resolve_acoustic_lengths(
        self,
        batch: Stage2Inputs,
        target_len: int | None = None,
        target_lengths_override: torch.Tensor | None = None,
        prefer_target_lengths: bool = True,
    ) -> torch.Tensor:
        device = batch.semantic_tokens.device

        if target_lengths_override is not None:
            lengths = target_lengths_override.long().to(device)
        elif prefer_target_lengths and batch.target_lengths is not None:
            lengths = batch.target_lengths.long().to(device)
        else:
            lengths = self.infer_acoustic_lengths(batch)

        lengths = lengths.clamp_min(1)

        if target_len is not None:
            lengths = lengths.clamp(max=max(int(target_len), 1))

        return lengths

    def _masked_frame_loss(
        self,
        loss_map: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        EN:
        Masked average for frame-level loss map.

        Args:
            loss_map: (B, T)
            lengths:  (B,)

        ZH:
        对 frame-level loss map 做 masked average。
        """
        if loss_map.ndim != 2:
            raise ValueError(f"loss_map must be 2D, got shape {tuple(loss_map.shape)}")

        B, T = loss_map.shape

        if lengths is None:
            return loss_map.mean()

        lengths = lengths.long().to(loss_map.device).clamp(min=1, max=T)
        mask = make_length_mask(lengths, T).float()
        return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)

    def _weighted_recon_map(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Frequency-weighted reconstruction map.

        Return:
            (B, T)

        ZH:
        频带加权 reconstruction map。
        """
        weights = self.mel_bin_weights.to(
            device=pred_mel.device,
            dtype=pred_mel.dtype,
        ).view(1, 1, -1)

        if self.recon_loss_type == "mse":
            elem = (pred_mel - target_mel) ** 2
        elif self.recon_loss_type == "smooth_l1":
            elem = F.smooth_l1_loss(pred_mel, target_mel, reduction="none")
        else:
            elem = (pred_mel - target_mel).abs()

        elem = elem * weights
        return elem.mean(dim=-1)

    def _weighted_delta_detail_losses(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        target_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        EN:
        Frequency-weighted delta and delta-delta losses.

        ZH:
        频带加权的一阶 / 二阶差分细节损失。
        """
        weights = self.mel_bin_weights.to(
            device=pred_mel.device,
            dtype=pred_mel.dtype,
        ).view(1, 1, -1)

        pred_delta = pred_mel[:, 1:, :] - pred_mel[:, :-1, :]
        target_delta = target_mel[:, 1:, :] - target_mel[:, :-1, :]

        delta_map = (pred_delta - target_delta).abs() * weights
        delta_map = delta_map.mean(dim=-1)

        if target_lengths is not None:
            delta_lengths = (target_lengths - 1).clamp_min(1)
        else:
            delta_lengths = None

        loss_delta = self._masked_frame_loss(delta_map, delta_lengths)

        pred_delta2 = pred_delta[:, 1:, :] - pred_delta[:, :-1, :]
        target_delta2 = target_delta[:, 1:, :] - target_delta[:, :-1, :]

        delta2_map = (pred_delta2 - target_delta2).abs() * weights
        delta2_map = delta2_map.mean(dim=-1)

        if target_lengths is not None:
            delta2_lengths = (target_lengths - 2).clamp_min(1)
        else:
            delta2_lengths = None

        loss_delta2 = self._masked_frame_loss(delta2_map, delta2_lengths)
        return loss_delta, loss_delta2

    def _sample_training_t(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        EN:
        Sample training time t.

        ZH:
        采样训练阶段的 Flow Matching 时间 t。
        """
        u = torch.rand(batch_size, device=device)

        if self.t_sampling_mode == "uniform":
            t = u
        else:
            t = 1.0 - torch.pow(u, self.t_bias_power)

        t = t.clamp(min=self.t_min, max=self.t_max)
        return t

    # --------------------------------------------------------
    # Conditional dropout / 条件 dropout
    # --------------------------------------------------------
    def _make_keep_masks(
        self,
        batch: Stage2Inputs,
        force_mode: str = "full",
    ) -> dict[str, torch.Tensor]:
        """
        EN:
        Create keep masks for content / semantic / style conditions.

        force_mode:
        - "full": keep all available conditions
        - "uncond": drop all conditions
        - "dropout" or "train": stochastic conditional dropout

        ZH:
        为 content / semantic / style 条件创建 keep mask。
        """
        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        ones = torch.ones(B, device=device)
        zeros = torch.zeros(B, device=device)

        if force_mode == "full":
            return {
                "content": ones,
                "semantic": ones,
                "style": ones,
            }

        if force_mode == "uncond":
            return {
                "content": zeros,
                "semantic": zeros,
                "style": zeros,
            }

        content_keep = (torch.rand(B, device=device) > self.cond_drop_prob_content).float()
        semantic_keep = (torch.rand(B, device=device) > self.cond_drop_prob_semantic).float()
        style_keep = (torch.rand(B, device=device) > self.cond_drop_prob_style).float()

        all_drop = torch.rand(B, device=device) < self.cond_drop_all_prob
        content_keep[all_drop] = 0.0
        semantic_keep[all_drop] = 0.0
        style_keep[all_drop] = 0.0

        return {
            "content": content_keep,
            "semantic": semantic_keep,
            "style": style_keep,
        }

    # --------------------------------------------------------
    # v6.1 condition building / v6.1 条件构建
    # --------------------------------------------------------
    def _build_v61_conditions(
        self,
        batch: Stage2Inputs,
        target_len: int,
        target_lengths: torch.Tensor,
    ) -> Stage2ConditionBundleV6:
        return self.condition_encoder.build_conditions_v6_1(
            batch=batch,
            target_len=target_len,
            target_lengths=target_lengths,
        )

    def _resolve_condition_tensors(
        self,
        cond: Stage2ConditionBundleV6,
        keep_masks: dict[str, torch.Tensor],
        target_len: int,
    ) -> dict[str, torch.Tensor | None]:
        """
        EN:
        Convert condition bundle to tensors used by v6.1 blocks.

        ZH:
        将条件包转成 v6.1 blocks 使用的张量。
        """
        if cond.content_frame_main is None:
            raise ValueError("cond.content_frame_main is required for v6.1 model.")

        B = cond.content_frame_main.shape[0]
        device = cond.content_frame_main.device
        dtype = cond.content_frame_main.dtype

        content_keep = keep_masks["content"].to(device=device, dtype=dtype).view(B, 1, 1)
        semantic_keep = keep_masks["semantic"].to(device=device, dtype=dtype).view(B, 1)
        style_keep = keep_masks["style"].to(device=device, dtype=dtype).view(B, 1)

        content_frame = cond.content_frame_main * content_keep

        style_global = cond.style_global
        if style_global is not None:
            style_global = style_global * style_keep

        semantic_guide = cond.semantic_guide_cond
        semantic_key_padding_mask = None

        if semantic_guide is not None and cond.semantic_guide_mask is not None:
            semantic_key_padding_mask = ~cond.semantic_guide_mask

        return {
            "content_frame": content_frame,
            "content_mask": cond.content_frame_main_mask,
            "style_global": style_global,
            "semantic_guide": semantic_guide,
            "semantic_key_padding_mask": semantic_key_padding_mask,
            "semantic_keep_mask": semantic_keep.view(B),
        }

    # --------------------------------------------------------
    # Content auxiliary supervision / 内容辅助监督
    # --------------------------------------------------------
    def _expand_phoneme_targets_to_frames(
        self,
        batch: Stage2Inputs,
        target_lengths: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor | None:
        """
        EN:
        Expand phoneme ids to frame-level pseudo targets.

        This is a lightweight auxiliary target. It is not a true duration model.

        Return:
            targets: (B, T), int64, -100 for ignore

        ZH:
        将 phoneme id 扩展到 frame-level 伪标签。

        这是轻量辅助监督，不是真正的 duration model。
        """
        if batch.phoneme_ids is None:
            return None

        device = batch.semantic_tokens.device
        phoneme_ids = batch.phoneme_ids.long().to(device)
        B, T_text = phoneme_ids.shape

        if batch.phoneme_lens is not None:
            phoneme_lens = batch.phoneme_lens.long().to(device).clamp(min=1, max=T_text)
        else:
            phoneme_lens = torch.full(
                size=(B,),
                fill_value=T_text,
                dtype=torch.long,
                device=device,
            )

        target_lengths = target_lengths.long().to(device).clamp(min=1, max=target_len)

        targets = torch.full(
            size=(B, target_len),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )

        for i in range(B):
            src_len = int(phoneme_lens[i].item())
            tgt_len = int(target_lengths[i].item())

            if src_len <= 0 or tgt_len <= 0:
                continue

            idx = torch.linspace(
                0,
                src_len - 1,
                steps=tgt_len,
                device=device,
            ).round().long().clamp(min=0, max=src_len - 1)

            targets[i, :tgt_len] = phoneme_ids[i, idx].clamp(
                min=0,
                max=self.phoneme_vocab_size - 1,
            )

        return targets

    def _compute_content_aux_loss(
        self,
        hidden: torch.Tensor,
        head: nn.Module,
        targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        EN:
        Compute frame-level content auxiliary CE loss and accuracy.

        ZH:
        计算 frame-level content 辅助交叉熵损失与准确率。
        """
        if targets is None:
            zero = hidden.new_tensor(0.0)
            return zero, zero

        logits = head(hidden)  # (B, T, V)
        B, T, V = logits.shape

        if targets.shape[1] != T:
            if targets.shape[1] > T:
                targets = targets[:, :T]
            else:
                pad = torch.full(
                    size=(B, T - targets.shape[1]),
                    fill_value=-100,
                    dtype=targets.dtype,
                    device=targets.device,
                )
                targets = torch.cat([targets, pad], dim=1)

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=-100,
        )

        with torch.no_grad():
            valid = targets != -100
            if valid.any():
                pred = logits.argmax(dim=-1)
                acc = (pred[valid] == targets[valid]).float().mean()
            else:
                acc = hidden.new_tensor(0.0)

        return loss, acc

    # --------------------------------------------------------
    # Span focus helpers / span focus 辅助函数
    # --------------------------------------------------------
    def _make_random_span_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        EN:
        Create one random valid span per sample.

        ZH:
        为每条样本创建一个随机有效 span。
        """
        device = lengths.device
        B = lengths.shape[0]

        mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)

        for i in range(B):
            L = max(int(lengths[i].item()), 1)

            ratio = float(
                torch.empty(1, device=device).uniform_(
                    self.span_focus_ratio_min,
                    self.span_focus_ratio_max,
                ).item()
            )

            span_len = max(1, int(round(L * ratio)))
            span_len = min(span_len, L)

            if L - span_len > 0:
                start = int(torch.randint(0, L - span_len + 1, (1,), device=device).item())
            else:
                start = 0

            mask[i, start : start + span_len] = True

        return mask

    def _masked_mean_by_bool_mask(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        values: (B, T)
        mask:   (B, T), bool

        ZH:
        使用 bool mask 对 (B, T) 数值做平均。
        """
        m = mask.float()
        return (values * m).sum() / m.sum().clamp_min(1.0)

    # --------------------------------------------------------
    # Forward / 前向传播
    # --------------------------------------------------------
    def forward(
        self,
        noisy_acoustic: torch.Tensor,              # (B, T, C)
        t: torch.Tensor,                           # (B,)
        batch: Stage2Inputs,
        self_condition: torch.Tensor | None = None,
        force_mode: str = "full",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        EN:
        Forward pass for v6.1 acoustic model.

        Return:
            pred_flow, aux

        ZH:
        v6.1 声学模型前向传播。

        返回：
            pred_flow, aux
        """
        B, T_ac, _ = noisy_acoustic.shape
        device = noisy_acoustic.device

        target_lengths = self._resolve_acoustic_lengths(
            batch=batch,
            target_len=T_ac,
            target_lengths_override=None,
            prefer_target_lengths=True,
        )

        valid_mask = make_length_mask(target_lengths, T_ac)
        key_padding_mask = ~valid_mask

        cond = self._build_v61_conditions(
            batch=batch,
            target_len=T_ac,
            target_lengths=target_lengths,
        )

        keep_masks = self._make_keep_masks(
            batch=batch,
            force_mode=force_mode,
        )

        cond_tensors = self._resolve_condition_tensors(
            cond=cond,
            keep_masks=keep_masks,
            target_len=T_ac,
        )

        time_emb = self.time_embedding(t)

        x = self.input_proj(noisy_acoustic)
        x = x + self.position_embedding(T_ac, device=device).to(dtype=x.dtype)

        if self_condition is not None and self.self_cond_proj is not None:
            x = x + self.self_cond_proj(self_condition)

        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        mid_hidden = None

        for i, block in enumerate(self.blocks):
            x = block(
                x=x,
                time_emb=time_emb,
                content_frame=cond_tensors["content_frame"],
                style_global=cond_tensors["style_global"],
                semantic_guide=cond_tensors["semantic_guide"],
                self_key_padding_mask=key_padding_mask,
                content_mask=cond_tensors["content_mask"],
                semantic_key_padding_mask=cond_tensors["semantic_key_padding_mask"],
                semantic_keep_mask=cond_tensors["semantic_keep_mask"],
            )

            if i == len(self.blocks) // 2:
                mid_hidden = x

        if mid_hidden is None:
            mid_hidden = x

        pred_flow = self.output_proj(x)
        pred_flow = pred_flow * valid_mask.unsqueeze(-1).to(dtype=pred_flow.dtype)

        frame_cond_hidden = self.frame_cond_to_hidden(cond.content_frame_main)
        frame_cond_hidden = frame_cond_hidden * valid_mask.unsqueeze(-1).to(dtype=frame_cond_hidden.dtype)

        aux: dict[str, torch.Tensor] = {
            "mid_hidden": mid_hidden,
            "final_hidden": x,
            "frame_cond_hidden": frame_cond_hidden,
            "valid_mask": valid_mask,
            "target_lengths": target_lengths,
            "content_main_norm": cond.content_frame_main.detach().norm(dim=-1).mean(),
            "semantic_guide_norm": (
                cond.semantic_guide_cond.detach().norm(dim=-1).mean()
                if cond.semantic_guide_cond is not None
                else noisy_acoustic.new_tensor(0.0)
            ),
        }

        return pred_flow, aux

    # --------------------------------------------------------
    # Training loss / 训练损失
    # --------------------------------------------------------
    def compute_flow_matching_loss(
        self,
        batch: Stage2Inputs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        EN:
        Compute v6.1 training loss.

        ZH:
        计算 v6.1 训练损失。
        """
        if batch.target_acoustic is None:
            raise ValueError("batch.target_acoustic is required for training loss.")

        x1 = batch.target_acoustic.float()
        B, T_ac, _ = x1.shape
        device = x1.device

        target_lengths = self._resolve_acoustic_lengths(
            batch=batch,
            target_len=T_ac,
            target_lengths_override=batch.target_lengths,
            prefer_target_lengths=True,
        )

        valid_mask = make_length_mask(target_lengths, T_ac)

        x0 = torch.randn_like(x1)
        t = self._sample_training_t(batch_size=B, device=device)
        t_view = t.view(B, 1, 1)

        xt = (1.0 - t_view) * x0 + t_view * x1
        target_flow = x1 - x0

        self_condition = None
        if self.use_self_condition and self.self_cond_proj is not None:
            if torch.rand((), device=device).item() < self.self_condition_prob:
                with torch.no_grad():
                    pred_flow_sc, _ = self.forward(
                        noisy_acoustic=xt,
                        t=t,
                        batch=batch,
                        self_condition=None,
                        force_mode="full",
                    )
                    self_condition = xt + (1.0 - t_view) * pred_flow_sc
                    self_condition = self_condition.detach()

        pred_flow, aux = self.forward(
            noisy_acoustic=xt,
            t=t,
            batch=batch,
            self_condition=self_condition,
            force_mode="dropout",
        )

        flow_map = (pred_flow - target_flow) ** 2
        flow_map = flow_map.mean(dim=-1)
        fm_loss = self._masked_frame_loss(flow_map, target_lengths)

        coarse_mel = xt + (1.0 - t_view) * pred_flow
        coarse_mel = coarse_mel * valid_mask.unsqueeze(-1).to(dtype=coarse_mel.dtype)

        recon_coarse_map = self._weighted_recon_map(coarse_mel, x1)
        recon_coarse_loss = self._masked_frame_loss(recon_coarse_map, target_lengths)

        frame_cond_hidden = aux["frame_cond_hidden"]
        key_padding_mask = ~valid_mask

        coarse_for_refine = coarse_mel.detach() if self.detach_coarse_for_refinement else coarse_mel

        refined_residual = self.refinement_postnet(
            coarse_mel=coarse_for_refine,
            frame_cond=frame_cond_hidden,
            global_style=None,
            key_padding_mask=key_padding_mask,
        )

        refined_mel = coarse_mel + refined_residual
        refined_mel = refined_mel * valid_mask.unsqueeze(-1).to(dtype=refined_mel.dtype)

        recon_refined_map = self._weighted_recon_map(refined_mel, x1)
        recon_refined_loss = self._masked_frame_loss(recon_refined_map, target_lengths)

        delta_loss, delta2_loss = self._weighted_delta_detail_losses(
            pred_mel=refined_mel,
            target_mel=x1,
            target_lengths=target_lengths,
        )

        phoneme_frame_targets = self._expand_phoneme_targets_to_frames(
            batch=batch,
            target_lengths=target_lengths,
            target_len=T_ac,
        )

        mid_content_aux_loss, mid_content_acc = self._compute_content_aux_loss(
            hidden=aux["mid_hidden"],
            head=self.mid_content_aux_head,
            targets=phoneme_frame_targets,
        )

        final_content_aux_loss, final_content_acc = self._compute_content_aux_loss(
            hidden=aux["final_hidden"],
            head=self.final_content_aux_head,
            targets=phoneme_frame_targets,
        )

        # ----------------------------------------------------
        # Span focus losses
        # ----------------------------------------------------
        span_mask = self._make_random_span_mask(target_lengths, T_ac)

        span_flow_loss = self._masked_mean_by_bool_mask(flow_map, span_mask)
        span_recon_loss = self._masked_mean_by_bool_mask(recon_refined_map, span_mask)

        if phoneme_frame_targets is not None:
            span_targets = phoneme_frame_targets.clone()
            span_targets[~span_mask] = -100
            span_content_loss, span_content_acc = self._compute_content_aux_loss(
                hidden=aux["final_hidden"],
                head=self.final_content_aux_head,
                targets=span_targets,
            )
        else:
            span_content_loss = x1.new_tensor(0.0)
            span_content_acc = x1.new_tensor(0.0)

        span_total_loss = (
            span_flow_loss
            + self.span_recon_weight * span_recon_loss
            + self.span_content_weight * span_content_loss
        )

        loss = (
            self.fm_loss_weight * fm_loss
            + self.recon_loss_weight * recon_coarse_loss
            + self.refined_recon_loss_weight * recon_refined_loss
            + self.delta_loss_weight * delta_loss
            + self.delta2_loss_weight * delta2_loss
            + self.mid_content_aux_weight * mid_content_aux_loss
            + self.final_content_aux_weight * final_content_aux_loss
            + self.span_focus_aux_weight * span_total_loss
        )

        semantic_lengths = self._resolve_semantic_lengths(batch)
        length_ratio_used = target_lengths.float().mean() / semantic_lengths.float().mean().clamp_min(1.0)

        log_aux: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "t_mean": t.mean().detach(),

            "fm_loss": fm_loss.detach(),
            "recon_coarse_loss": recon_coarse_loss.detach(),
            "recon_refined_loss": recon_refined_loss.detach(),
            "delta_loss": delta_loss.detach(),
            "delta2_loss": delta2_loss.detach(),

            "mid_content_aux_loss": mid_content_aux_loss.detach(),
            "mid_content_acc": mid_content_acc.detach(),
            "final_content_aux_loss": final_content_aux_loss.detach(),
            "final_content_acc": final_content_acc.detach(),

            "span_flow_loss": span_flow_loss.detach(),
            "span_recon_loss": span_recon_loss.detach(),
            "span_content_loss": span_content_loss.detach(),
            "span_content_acc": span_content_acc.detach(),

            "length_ratio_used": length_ratio_used.detach(),
            "target_length_mean": target_lengths.float().mean().detach(),
            "inferred_length_mean": self.infer_acoustic_lengths(batch).float().mean().detach(),

            "content_main_norm": aux["content_main_norm"].detach(),
            "semantic_guide_norm": aux["semantic_guide_norm"].detach(),
        }

        return loss, log_aux

    # --------------------------------------------------------
    # Sampling / 采样
    # --------------------------------------------------------
    @torch.inference_mode()
    def sample(
        self,
        batch: Stage2Inputs,
        num_steps: int = 32,
        target_lengths: torch.Tensor | None = None,
        temperature: float = 1.0,
        use_heun: bool = False,
        guidance_scale: float = 1.0,
    ) -> Stage2Outputs:
        """
        EN:
        v6.1 Euler / Heun ODE sampling.

        ZH:
        v6.1 Euler / Heun ODE 采样。
        """
        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        if target_lengths is None:
            target_lengths = self.infer_acoustic_lengths(batch)
        else:
            target_lengths = target_lengths.long().to(device).clamp_min(1)

        max_len = int(target_lengths.max().item())
        x = torch.randn(B, max_len, self.acoustic_dim, device=device) * float(temperature)

        dt = 1.0 / float(num_steps)

        for step in range(num_steps):
            t_scalar = step / float(num_steps)
            t = torch.full((B,), fill_value=t_scalar, device=device)

            v_full, _ = self.forward(
                noisy_acoustic=x,
                t=t,
                batch=batch,
                self_condition=None,
                force_mode="full",
            )

            if guidance_scale != 1.0:
                v_uncond, _ = self.forward(
                    noisy_acoustic=x,
                    t=t,
                    batch=batch,
                    self_condition=None,
                    force_mode="uncond",
                )
                v = v_uncond + float(guidance_scale) * (v_full - v_uncond)
            else:
                v = v_full

            if use_heun:
                x_euler = x + dt * v

                t_next_scalar = min((step + 1) / float(num_steps), 1.0)
                t_next = torch.full((B,), fill_value=t_next_scalar, device=device)

                v_next_full, _ = self.forward(
                    noisy_acoustic=x_euler,
                    t=t_next,
                    batch=batch,
                    self_condition=None,
                    force_mode="full",
                )

                if guidance_scale != 1.0:
                    v_next_uncond, _ = self.forward(
                        noisy_acoustic=x_euler,
                        t=t_next,
                        batch=batch,
                        self_condition=None,
                        force_mode="uncond",
                    )
                    v_next = v_next_uncond + float(guidance_scale) * (v_next_full - v_next_uncond)
                else:
                    v_next = v_next_full

                x = x + 0.5 * dt * (v + v_next)
            else:
                x = x + dt * v

            valid_mask = make_length_mask(target_lengths, max_len)
            x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        aux = {
            "num_steps": int(num_steps),
            "model_version": self.model_version,
            "model_variant": self.model_variant,
            "use_heun": bool(use_heun),
            "guidance_scale": float(guidance_scale),
            "temperature": float(temperature),
            "length_correction_mode": self.length_correction_mode,
            "length_bias_scale": float(self.length_bias_scale),
            "enable_length_clamp": bool(self.enable_length_clamp),
            "inferred_lengths": target_lengths.detach().cpu().tolist(),
            "semantic_rate_hz": float(self.semantic_rate_hz),
            "acoustic_rate_hz": float(self.acoustic_rate_hz),
        }

        return Stage2Outputs(
            acoustic=x,
            acoustic_type="mel",
            lengths=target_lengths,
            aux=aux,
        )

class Stage2FlowMatchingAcousticModelV62(Stage2FlowMatchingAcousticModelV61):
    """
    EN:
    VoiceLab Stage2 Flow Matching acoustic model v6.2.

    v6.2 is a cold-start-aware extension of v6.1.

    Core idea:
        condition -> coarse_mel_pred
        x_start = (1 - t_start) * noise + t_start * coarse_mel_pred
        sample from t_start to 1.0

    This class intentionally keeps the v6.1 backbone unchanged and adds:
    - a condition-to-coarse-mel bootstrapper
    - coarse mel reconstruction loss
    - bridge-start reconstruction loss
    - bootstrap sampling API

    ZH:
    VoiceLab Stage2 Flow Matching 声学模型 v6.2。

    v6.2 是在 v6.1 基础上的 cold-start aware 扩展。

    核心思想：
        condition -> coarse_mel_pred
        x_start = (1 - t_start) * noise + t_start * coarse_mel_pred
        从 t_start 采样到 1.0

    本类刻意保留 v6.1 主干不变，只新增：
    - condition-to-coarse-mel bootstrapper
    - coarse mel reconstruction loss
    - bridge-start reconstruction loss
    - bootstrap sampling API
    """

    def __init__(
        self,
        *args: Any,

        # ----------------------------------------------------
        # v6.2 bootstrapper switch / v6.2 bootstrapper 开关
        # ----------------------------------------------------
        use_coarse_bootstrapper: bool = True,
        bootstrap_hidden_dim: int | None = None,
        bootstrap_num_blocks: int = 4,
        bootstrap_kernel_size: int = 5,
        bootstrap_dropout: float = 0.1,
        bootstrap_expansion_factor: int = 4,

        # ----------------------------------------------------
        # v6.2 bootstrapper initialization / 初始化
        # ----------------------------------------------------
        bootstrap_mel_bias_init: float = -5.0,
        bootstrap_residual_scale_init: float = 0.1,
        bootstrap_semantic_gate_init: float = 0.5,
        bootstrap_style_residual_scale_init: float = 0.1,
        bootstrap_clamp_output: bool = False,
        bootstrap_output_min: float = -12.0,
        bootstrap_output_max: float = 4.0,

        # ----------------------------------------------------
        # v6.2 coarse mel losses / 粗 mel 损失
        # ----------------------------------------------------
        coarse_mel_loss_weight: float = 1.0,
        coarse_mel_mse_loss_weight: float = 0.0,
        coarse_delta_loss_weight: float = 0.2,

        # ----------------------------------------------------
        # v6.2 bridge losses / bridge 起点损失
        # ----------------------------------------------------
        use_bridge_loss: bool = True,
        bridge_t_start: float = 0.075,
        bridge_noise_temperature: float = 0.3,
        bridge_recon_loss_weight: float = 1.0,
        bridge_delta_loss_weight: float = 0.2,
        detach_coarse_for_bridge: bool = True,

        # ----------------------------------------------------
        # Extra compatibility args / 兼容额外参数
        # ----------------------------------------------------
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model_version = "v6_2"
        self.model_variant = "v6.2_cold_start_bootstrap"

        # ----------------------------------------------------
        # v6.2 config values
        # ----------------------------------------------------
        self.use_coarse_bootstrapper = bool(use_coarse_bootstrapper)

        self.coarse_mel_loss_weight = float(coarse_mel_loss_weight)
        self.coarse_mel_mse_loss_weight = float(coarse_mel_mse_loss_weight)
        self.coarse_delta_loss_weight = float(coarse_delta_loss_weight)

        self.use_bridge_loss = bool(use_bridge_loss)
        self.bridge_t_start = float(bridge_t_start)
        self.bridge_noise_temperature = float(bridge_noise_temperature)
        self.bridge_recon_loss_weight = float(bridge_recon_loss_weight)
        self.bridge_delta_loss_weight = float(bridge_delta_loss_weight)
        self.detach_coarse_for_bridge = bool(detach_coarse_for_bridge)

        if not (0.0 <= self.bridge_t_start < 1.0):
            raise ValueError(
                f"bridge_t_start must be in [0, 1), got {self.bridge_t_start}"
            )

        # ----------------------------------------------------
        # v6.2 coarse mel bootstrapper
        # ----------------------------------------------------
        if self.use_coarse_bootstrapper:
            bootstrap_cfg = CoarseMelBootstrapperConfig(
                input_dim=self.content_main_dim,
                semantic_dim=self.hidden_dim,
                style_dim=self.style_dim,

                hidden_dim=int(bootstrap_hidden_dim or self.content_main_dim),
                n_mels=self.acoustic_dim,

                num_blocks=int(bootstrap_num_blocks),
                kernel_size=int(bootstrap_kernel_size),
                dropout=float(bootstrap_dropout),
                expansion_factor=int(bootstrap_expansion_factor),

                use_semantic_guide=True,
                use_style_film=True,
                use_style_residual=True,

                mel_bias_init=float(bootstrap_mel_bias_init),
                residual_scale_init=float(bootstrap_residual_scale_init),
                semantic_gate_init=float(bootstrap_semantic_gate_init),
                style_residual_scale_init=float(bootstrap_style_residual_scale_init),

                use_final_layer_norm=True,
                clamp_output=bool(bootstrap_clamp_output),
                output_min=float(bootstrap_output_min),
                output_max=float(bootstrap_output_max),
            )

            self.coarse_bootstrapper = CoarseMelBootstrapper(bootstrap_cfg)
        else:
            self.coarse_bootstrapper = None

    def _resize_frame_condition_to_target_len(
        self,
        x: torch.Tensor | None,
        target_len: int,
        name: str = "condition",
    ) -> torch.Tensor | None:
        """
        Resize a frame-like condition tensor to target acoustic length.

        Args:
            x:
                Optional tensor with shape [B, T, D].
            target_len:
                Target acoustic frame length.
            name:
                Name used in error message.

        Returns:
            Tensor with shape [B, target_len, D], or None.

        Why this is needed:
            In v6.1, content_frame is already expanded to acoustic length,
            but semantic_guide may still be semantic-token length. v6.2
            bootstrapper requires both to share [B, T].
        """
        if x is None:
            return None

        if x.ndim != 3:
            raise ValueError(
                f"{name} must be [B, T, D], got shape={tuple(x.shape)}"
            )

        if x.shape[1] == int(target_len):
            return x

        # [B, T, D] -> [B, D, T]
        x_t = x.transpose(1, 2)

        # Linear interpolation along temporal dimension.
        x_t = F.interpolate(
            x_t,
            size=int(target_len),
            mode="linear",
            align_corners=False,
        )

        # [B, D, T] -> [B, T, D]
        x = x_t.transpose(1, 2)

        return x

    # --------------------------------------------------------
    # v6.2 condition-to-coarse-mel path
    # v6.2 条件到 coarse mel 路径
    # --------------------------------------------------------
    def predict_coarse_mel(
        self,
        batch: Stage2Inputs,
        target_len: int | None = None,
        target_lengths: torch.Tensor | None = None,
        force_mode: str = "full",
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        EN:
        Predict a coarse mel scaffold from conditions.

        This is the v6.2 bootstrap path:
            semantic/text/prompt condition -> coarse_mel_pred

        ZH:
        根据条件预测 coarse mel scaffold。

        这是 v6.2 的 bootstrap 路径：
            semantic/text/prompt condition -> coarse_mel_pred
        """
        if self.coarse_bootstrapper is None:
            raise RuntimeError("coarse_bootstrapper is disabled.")

        if target_len is None:
            if target_lengths is not None:
                target_len = int(target_lengths.max().item())
            elif batch.target_acoustic is not None:
                target_len = int(batch.target_acoustic.shape[1])
            else:
                inferred = self.infer_acoustic_lengths(batch)
                target_len = int(inferred.max().item())

        resolved_lengths = self._resolve_acoustic_lengths(
            batch=batch,
            target_len=target_len,
            target_lengths_override=target_lengths,
            prefer_target_lengths=True,
        )

        cond = self._build_v61_conditions(
            batch=batch,
            target_len=target_len,
            target_lengths=resolved_lengths,
        )

        keep_masks = self._make_keep_masks(
            batch=batch,
            force_mode=force_mode,
        )

        cond_tensors = self._resolve_condition_tensors(
            cond=cond,
            keep_masks=keep_masks,
            target_len=target_len,
        )

        content_frame = cond_tensors["content_frame"]
        if content_frame is None:
            raise ValueError("content_frame is required for v6.2 bootstrapper.")

        semantic_guide = cond_tensors["semantic_guide"]

        # ----------------------------------------------------
        # Important:
        # In v6.1, content_frame is acoustic-frame length,
        # but semantic_guide may still be semantic-token length.
        # v6.2 bootstrapper requires both to share [B, T].
        # Therefore we resize semantic_guide to target_len here.
        # ----------------------------------------------------
        semantic_guide = self._resize_frame_condition_to_target_len(
            x=semantic_guide,
            target_len=target_len,
            name="semantic_guide",
        )

        # _resolve_condition_tensors keeps semantic_keep_mask separate because
        # v6.1 blocks consume it explicitly. The bootstrapper consumes
        # semantic_guide directly, so we apply the keep mask here.
        semantic_keep = cond_tensors.get("semantic_keep_mask")
        if semantic_guide is not None and semantic_keep is not None:
            semantic_guide = semantic_guide * semantic_keep.to(
                device=semantic_guide.device,
                dtype=semantic_guide.dtype,
            ).view(-1, 1, 1)

        style_global = cond_tensors["style_global"]

        out = self.coarse_bootstrapper(
            content_frame=content_frame,
            semantic_guide=semantic_guide,
            style_global=style_global,
            target_lengths=resolved_lengths,
            return_aux=return_aux,
        )

        return out

    # --------------------------------------------------------
    # v6.2 training loss / v6.2 训练损失
    # --------------------------------------------------------
    def compute_flow_matching_loss(
        self,
        batch: Stage2Inputs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        EN:
        Compute v6.2 training loss.

        v6.2 keeps the full v6.1 loss and adds:
        - coarse mel reconstruction loss
        - optional bridge-start reconstruction loss

        ZH:
        计算 v6.2 训练损失。

        v6.2 保留完整 v6.1 loss，并新增：
        - coarse mel reconstruction loss
        - 可选 bridge-start reconstruction loss
        """
        base_loss, log_aux = super().compute_flow_matching_loss(batch)

        if not self.use_coarse_bootstrapper or self.coarse_bootstrapper is None:
            return base_loss, log_aux

        if batch.target_acoustic is None:
            raise ValueError("batch.target_acoustic is required for v6.2 loss.")

        x1 = batch.target_acoustic.float()
        B, T_ac, _ = x1.shape
        device = x1.device

        target_lengths = self._resolve_acoustic_lengths(
            batch=batch,
            target_len=T_ac,
            target_lengths_override=batch.target_lengths,
            prefer_target_lengths=True,
        )

        valid_mask = make_length_mask(target_lengths, T_ac)

        # ----------------------------------------------------
        # 1. Coarse mel prediction loss
        # ----------------------------------------------------
        coarse_out = self.predict_coarse_mel(
            batch=batch,
            target_len=T_ac,
            target_lengths=target_lengths,
            force_mode="dropout",
            return_aux=True,
        )

        coarse_mel, coarse_aux = coarse_out
        coarse_mel = coarse_mel * valid_mask.unsqueeze(-1).to(dtype=coarse_mel.dtype)

        coarse_losses = compute_coarse_mel_losses(
            coarse_mel=coarse_mel,
            target_acoustic=x1,
            target_lengths=target_lengths,
            l1_weight=self.coarse_mel_loss_weight,
            mse_weight=self.coarse_mel_mse_loss_weight,
            delta_l1_weight=self.coarse_delta_loss_weight,
        )

        coarse_total_loss = coarse_losses["coarse_mel_total_loss"]

        # ----------------------------------------------------
        # 2. Optional bridge-start reconstruction loss
        # ----------------------------------------------------
        if self.use_bridge_loss:
            bridge_coarse = coarse_mel.detach() if self.detach_coarse_for_bridge else coarse_mel

            x_bridge, bridge_noise = build_bootstrap_start(
                coarse_mel=bridge_coarse,
                t_start=self.bridge_t_start,
                temperature=self.bridge_noise_temperature,
                lengths=target_lengths,
            )

            t_bridge = torch.full(
                size=(B,),
                fill_value=float(self.bridge_t_start),
                dtype=torch.float32,
                device=device,
            )

            pred_bridge_flow, bridge_aux = self.forward(
                noisy_acoustic=x_bridge,
                t=t_bridge,
                batch=batch,
                self_condition=None,
                force_mode="dropout",
            )

            t_view = t_bridge.view(B, 1, 1)
            bridge_recon = x_bridge + (1.0 - t_view) * pred_bridge_flow
            bridge_recon = bridge_recon * valid_mask.unsqueeze(-1).to(dtype=bridge_recon.dtype)

            bridge_recon_map = self._weighted_recon_map(bridge_recon, x1)
            bridge_recon_loss = self._masked_frame_loss(
                bridge_recon_map,
                target_lengths,
            )

            bridge_delta_loss, _bridge_delta2_loss = self._weighted_delta_detail_losses(
                pred_mel=bridge_recon,
                target_mel=x1,
                target_lengths=target_lengths,
            )

            bridge_total_loss = (
                self.bridge_recon_loss_weight * bridge_recon_loss
                + self.bridge_delta_loss_weight * bridge_delta_loss
            )
        else:
            bridge_recon_loss = x1.new_tensor(0.0)
            bridge_delta_loss = x1.new_tensor(0.0)
            bridge_total_loss = x1.new_tensor(0.0)

        # ----------------------------------------------------
        # 3. Final v6.2 total loss
        # ----------------------------------------------------
        loss = base_loss + coarse_total_loss + bridge_total_loss

        log_aux.update(
            {
                "loss": loss.detach(),

                "v62_base_loss": base_loss.detach(),

                "coarse_mel_l1_loss": coarse_losses["coarse_mel_l1_loss"].detach(),
                "coarse_mel_mse_loss": coarse_losses["coarse_mel_mse_loss"].detach(),
                "coarse_mel_delta_l1_loss": coarse_losses["coarse_mel_delta_l1_loss"].detach(),
                "coarse_mel_total_loss": coarse_total_loss.detach(),

                "bridge_recon_loss": bridge_recon_loss.detach(),
                "bridge_delta_loss": bridge_delta_loss.detach(),
                "bridge_total_loss": bridge_total_loss.detach(),

                "bridge_t_start": x1.new_tensor(float(self.bridge_t_start)).detach(),
                "bridge_noise_temperature": x1.new_tensor(
                    float(self.bridge_noise_temperature)
                ).detach(),
            }
        )

        if isinstance(coarse_aux, dict):
            for key, value in coarse_aux.items():
                if torch.is_tensor(value):
                    log_aux[f"coarse_{key}"] = value.detach()

        return loss, log_aux

    # --------------------------------------------------------
    # v6.2 bootstrap sampling / v6.2 bootstrap 采样
    # --------------------------------------------------------
    @torch.inference_mode()
    def sample_with_bootstrap(
        self,
        batch: Stage2Inputs,
        num_steps: int = 128,
        target_lengths: torch.Tensor | None = None,
        temperature: float = 0.3,
        use_heun: bool = True,
        guidance_scale: float = 1.0,
        bootstrap_t_start: float = 0.075,
        bootstrap_noise_temperature: float | None = None,
        return_bootstrap_mel: bool = False,
    ) -> Stage2Outputs:
        """
        EN:
        v6.2 bootstrap sampling.

        Instead of starting from pure noise at t=0:
            x0 = randn

        v6.2 starts from:
            coarse_mel_pred = bootstrapper(condition)
            x_start = (1 - t_start) * noise + t_start * coarse_mel_pred

        Then it integrates the ODE from t_start to 1.0.

        ZH:
        v6.2 bootstrap 采样。

        不再从 t=0 pure noise 开始：
            x0 = randn

        而是从：
            coarse_mel_pred = bootstrapper(condition)
            x_start = (1 - t_start) * noise + t_start * coarse_mel_pred

        然后从 t_start 积分到 1.0。
        """
        if self.coarse_bootstrapper is None:
            raise RuntimeError("sample_with_bootstrap requires coarse_bootstrapper.")

        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        bootstrap_t_start = float(bootstrap_t_start)
        if not (0.0 <= bootstrap_t_start < 1.0):
            raise ValueError(
                f"bootstrap_t_start must be in [0, 1), got {bootstrap_t_start}"
            )

        if target_lengths is None:
            target_lengths = self.infer_acoustic_lengths(batch)
        else:
            target_lengths = target_lengths.long().to(device).clamp_min(1)

        max_len = int(target_lengths.max().item())

        coarse_mel, coarse_aux = self.predict_coarse_mel(
            batch=batch,
            target_len=max_len,
            target_lengths=target_lengths,
            force_mode="full",
            return_aux=True,
        )

        coarse_mel = coarse_mel.to(device=device)
        valid_mask = make_length_mask(target_lengths, max_len)
        coarse_mel = coarse_mel * valid_mask.unsqueeze(-1).to(dtype=coarse_mel.dtype)

        noise_temperature = (
            float(bootstrap_noise_temperature)
            if bootstrap_noise_temperature is not None
            else float(temperature)
        )

        x, noise = build_bootstrap_start(
            coarse_mel=coarse_mel,
            t_start=bootstrap_t_start,
            temperature=noise_temperature,
            lengths=target_lengths,
        )

        interval = 1.0 - bootstrap_t_start
        dt = interval / float(num_steps)

        for step in range(num_steps):
            t_scalar = bootstrap_t_start + interval * (step / float(num_steps))
            t = torch.full((B,), fill_value=float(t_scalar), device=device)

            v_full, _ = self.forward(
                noisy_acoustic=x,
                t=t,
                batch=batch,
                self_condition=None,
                force_mode="full",
            )

            if guidance_scale != 1.0:
                v_uncond, _ = self.forward(
                    noisy_acoustic=x,
                    t=t,
                    batch=batch,
                    self_condition=None,
                    force_mode="uncond",
                )
                v = v_uncond + float(guidance_scale) * (v_full - v_uncond)
            else:
                v = v_full

            if use_heun:
                x_euler = x + dt * v

                t_next_scalar = bootstrap_t_start + interval * (
                    (step + 1) / float(num_steps)
                )
                t_next_scalar = min(float(t_next_scalar), 1.0)

                t_next = torch.full(
                    (B,),
                    fill_value=t_next_scalar,
                    device=device,
                )

                v_next_full, _ = self.forward(
                    noisy_acoustic=x_euler,
                    t=t_next,
                    batch=batch,
                    self_condition=None,
                    force_mode="full",
                )

                if guidance_scale != 1.0:
                    v_next_uncond, _ = self.forward(
                        noisy_acoustic=x_euler,
                        t=t_next,
                        batch=batch,
                        self_condition=None,
                        force_mode="uncond",
                    )
                    v_next = v_next_uncond + float(guidance_scale) * (
                        v_next_full - v_next_uncond
                    )
                else:
                    v_next = v_next_full

                x = x + 0.5 * dt * (v + v_next)
            else:
                x = x + dt * v

            x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        aux: dict[str, Any] = {
            "num_steps": int(num_steps),
            "model_version": self.model_version,
            "model_variant": self.model_variant,

            "sampling_start_mode": "bootstrap",
            "bootstrap_t_start": float(bootstrap_t_start),
            "bootstrap_noise_temperature": float(noise_temperature),

            "use_heun": bool(use_heun),
            "guidance_scale": float(guidance_scale),
            "temperature": float(temperature),

            "length_correction_mode": self.length_correction_mode,
            "length_bias_scale": float(self.length_bias_scale),
            "enable_length_clamp": bool(self.enable_length_clamp),

            "inferred_lengths": target_lengths.detach().cpu().tolist(),
            "semantic_rate_hz": float(self.semantic_rate_hz),
            "acoustic_rate_hz": float(self.acoustic_rate_hz),
        }

        if isinstance(coarse_aux, dict):
            for key, value in coarse_aux.items():
                if torch.is_tensor(value):
                    aux[f"coarse_{key}"] = value.detach().cpu()
                else:
                    aux[f"coarse_{key}"] = value

        if return_bootstrap_mel:
            aux["coarse_mel_pred"] = coarse_mel.detach().cpu()
            aux["bootstrap_noise"] = noise.detach().cpu()

        return Stage2Outputs(
            acoustic=x,
            acoustic_type="mel",
            lengths=target_lengths,
            aux=aux,
        )

class Stage2FlowMatchingAcousticModelV63(Stage2FlowMatchingAcousticModelV62):
    """
    EN:
    VoiceLab Stage2 acoustic model v6.3.

    v6.3 upgrades v6.2 in three directions:
    1. Stronger ConformerMelBootstrapper instead of lightweight Conv1D bootstrapper.
    2. ResidualFlowRefiner that learns target_mel - coarse_mel.
    3. Sampling path where final_mel = coarse_mel + sampled_residual.

    ZH:
    VoiceLab Stage2 声学模型 v6.3。

    v6.3 针对 v6.2 暴露出的三个瓶颈升级：
    1. 使用更强的 ConformerMelBootstrapper 替代轻量 Conv1D bootstrapper；
    2. 使用 ResidualFlowRefiner 学习 target_mel - coarse_mel；
    3. 推理时 final_mel = coarse_mel + sampled_residual。
    """

    def __init__(
        self,
        *args: Any,

        # ----------------------------------------------------
        # v6.3 bootstrapper config
        # ----------------------------------------------------
        use_conformer_bootstrapper: bool = True,
        v63_bootstrap_hidden_dim: int | None = None,
        v63_bootstrap_num_layers: int = 6,
        v63_bootstrap_num_heads: int = 8,
        v63_bootstrap_ff_mult: int = 4,
        v63_bootstrap_conv_kernel_size: int = 15,
        v63_bootstrap_dropout: float = 0.1,
        v63_bootstrap_mel_bias_init: float = -5.0,
        v63_bootstrap_zero_init_output: bool = False,
        v63_bootstrap_clamp_output: bool = False,

        # ----------------------------------------------------
        # v6.3.3 native text memory config
        # ----------------------------------------------------
        v63_use_native_text_memory: bool = False,
        v63_text_memory_dim: int | None = None,

        # ----------------------------------------------------
        # v6.6 reference acoustic style config
        # v6.6 参考音频声学风格配置
        # ----------------------------------------------------
        use_reference_acoustic_style: bool = False,
        reference_acoustic_dim: int = 80,
        reference_style_dim: int = 256,
        reference_style_hidden_dim: int = 256,
        reference_style_num_layers: int = 4,
        reference_style_kernel_size: int = 5,
        reference_style_dropout: float = 0.1,
        reference_style_fusion_mode: str = "gated_add",
        reference_style_gate_init: float = -3.0,

        # ----------------------------------------------------
        # v6.6.2 prosody / speaker auxiliary losses
        # v6.6.2 韵律 / 说话人辅助损失
        # ----------------------------------------------------
        use_v66_energy_loss: bool = False,
        v66_energy_loss_weight: float = 0.15,
        v66_energy_loss_type: str = "l1",
        v66_energy_normalize: bool = True,

        use_v66_f0_loss: bool = False,
        v66_f0_loss_weight: float = 0.05,
        v66_f0_loss_type: str = "l1",
        v66_f0_normalize: bool = True,
        v66_f0_hidden_dim: int = 128,

        use_v66_speaker_loss: bool = False,
        v66_speaker_loss_weight: float = 0.05,
        v66_speaker_embedding_dim: int = 192,
        v66_require_aux_targets: bool = False,

        # ----------------------------------------------------
        # v6.6.3 Bootstrapper LoRA
        # v6.6.3 Bootstrapper 低秩适配
        # ----------------------------------------------------
        use_bootstrapper_lora: bool = False,
        bootstrapper_lora_rank: int = 4,
        bootstrapper_lora_alpha: float = 8.0,
        bootstrapper_lora_dropout: float = 0.05,
        bootstrapper_lora_target: str = "core",
        bootstrapper_lora_init_scale: float = 0.01,

        # ----------------------------------------------------
        # v6.6.3-A Bootstrapper Attention LoRA
        # v6.6.3-A Bootstrapper 注意力 LoRA
        # ----------------------------------------------------
        use_bootstrapper_attention_lora: bool = False,
        bootstrapper_attention_lora_rank: int = 4,
        bootstrapper_attention_lora_alpha: float = 8.0,
        bootstrapper_attention_lora_dropout: float = 0.05,
        bootstrapper_attention_lora_target: str = "cross_only",
        bootstrapper_attention_lora_gate_init: float = -3.0,
        bootstrapper_attention_lora_enable_q: bool = True,
        bootstrapper_attention_lora_enable_k: bool = True,
        bootstrapper_attention_lora_enable_v: bool = True,
        bootstrapper_attention_lora_enable_o: bool = True,

        # ----------------------------------------------------
        # v6.6.4 Speaker Identity Adapter
        # v6.6.4 说话人身份适配器
        # ----------------------------------------------------
        use_v664_speaker_identity_adapter: bool = False,
        v664_speaker_identity_source: str = "prompt_or_target",
        v664_speaker_embedding_dim: int = 192,
        v664_speaker_adapter_hidden_dim: int = 256,
        v664_speaker_adapter_num_layers: int = 2,
        v664_speaker_adapter_dropout: float = 0.05,
        v664_speaker_adapter_fusion_mode: str = "gated_add",
        v664_speaker_adapter_gate_init: float = -3.0,
        v664_speaker_adapter_normalize: bool = True,

        # ----------------------------------------------------
        # v6.3 residual refiner config
        # ----------------------------------------------------
        use_residual_refiner: bool = True,
        residual_hidden_dim: int | None = None,
        residual_num_layers: int = 6,
        residual_num_heads: int = 8,
        residual_ff_mult: int = 4,
        residual_conv_kernel_size: int = 15,
        residual_dropout: float = 0.1,
        residual_noise_scale: float = 0.5,
        residual_sample_temperature: float = 0.3,
        residual_t_min: float = 0.0,
        residual_t_max: float = 1.0,
        detach_coarse_for_residual_refiner: bool = False,

        # ----------------------------------------------------
        # v6.3 loss weights
        # ----------------------------------------------------
        include_v61_base_loss: bool = False,
        v61_base_loss_weight: float = 1.0,

        v63_coarse_l1_weight: float = 2.0,
        v63_coarse_mse_weight: float = 0.2,
        v63_coarse_delta_weight: float = 0.5,
        v63_coarse_delta2_weight: float = 0.2,

        residual_refiner_loss_weight: float = 1.0,
        residual_flow_mse_weight: float = 1.0,
        residual_flow_l1_weight: float = 0.0,
        residual_recon_l1_weight: float = 1.0,
        final_recon_l1_weight: float = 1.0,
        final_delta_l1_weight: float = 0.3,
        final_delta2_l1_weight: float = 0.1,

        # ----------------------------------------------------
        # Extra compatibility args
        # ----------------------------------------------------
        **kwargs: Any,
    ) -> None:
        # ----------------------------------------------------
        # Disable v6.2 legacy bootstrapper inside parent V62.
        # v6.3 uses its own ConformerMelBootstrapper.
        # ----------------------------------------------------
        kwargs["use_coarse_bootstrapper"] = False

        super().__init__(*args, **kwargs)

        self.model_version = "v6_3"
        self.model_variant = "v6.3_conformer_bootstrap_residual_refiner"

        # ----------------------------------------------------
        # v6.3 switches
        # ----------------------------------------------------
        self.use_conformer_bootstrapper = bool(use_conformer_bootstrapper)
        self.use_residual_refiner = bool(use_residual_refiner)

        self.include_v61_base_loss = bool(include_v61_base_loss)
        self.v61_base_loss_weight = float(v61_base_loss_weight)

        self.v63_coarse_l1_weight = float(v63_coarse_l1_weight)
        self.v63_coarse_mse_weight = float(v63_coarse_mse_weight)
        self.v63_coarse_delta_weight = float(v63_coarse_delta_weight)
        self.v63_coarse_delta2_weight = float(v63_coarse_delta2_weight)

        self.residual_refiner_loss_weight = float(residual_refiner_loss_weight)
        self.residual_flow_mse_weight = float(residual_flow_mse_weight)
        self.residual_flow_l1_weight = float(residual_flow_l1_weight)
        self.residual_recon_l1_weight = float(residual_recon_l1_weight)
        self.final_recon_l1_weight = float(final_recon_l1_weight)
        self.final_delta_l1_weight = float(final_delta_l1_weight)
        self.final_delta2_l1_weight = float(final_delta2_l1_weight)

        self.residual_noise_scale = float(residual_noise_scale)
        self.residual_sample_temperature = float(residual_sample_temperature)
        self.residual_t_min = float(residual_t_min)
        self.residual_t_max = float(residual_t_max)
        self.detach_coarse_for_residual_refiner = bool(detach_coarse_for_residual_refiner)

        # ----------------------------------------------------
        # v6.6.2 prosody / speaker auxiliary losses
        # v6.6.2 韵律 / 说话人辅助损失
        # ----------------------------------------------------
        self.use_v66_energy_loss = bool(use_v66_energy_loss)
        self.v66_energy_loss_weight = float(v66_energy_loss_weight)
        self.v66_energy_loss_type = str(v66_energy_loss_type).strip().lower()
        self.v66_energy_normalize = bool(v66_energy_normalize)

        self.use_v66_f0_loss = bool(use_v66_f0_loss)
        self.v66_f0_loss_weight = float(v66_f0_loss_weight)
        self.v66_f0_loss_type = str(v66_f0_loss_type).strip().lower()
        self.v66_f0_normalize = bool(v66_f0_normalize)
        self.v66_f0_hidden_dim = int(v66_f0_hidden_dim)

        self.use_v66_speaker_loss = bool(use_v66_speaker_loss)
        self.v66_speaker_loss_weight = float(v66_speaker_loss_weight)
        self.v66_speaker_embedding_dim = int(v66_speaker_embedding_dim)
        self.v66_require_aux_targets = bool(v66_require_aux_targets)

        # ----------------------------------------------------
        # v6.6.3 Bootstrapper LoRA
        # ----------------------------------------------------
        self.use_bootstrapper_lora = bool(use_bootstrapper_lora)
        self.bootstrapper_lora_rank = int(bootstrapper_lora_rank)
        self.bootstrapper_lora_alpha = float(bootstrapper_lora_alpha)
        self.bootstrapper_lora_dropout = float(bootstrapper_lora_dropout)
        self.bootstrapper_lora_target = str(bootstrapper_lora_target).strip().lower()
        self.bootstrapper_lora_init_scale = float(bootstrapper_lora_init_scale)
        self.bootstrapper_lora_report = None

        # ----------------------------------------------------
        # v6.6.3-A Bootstrapper Attention LoRA
        # ----------------------------------------------------
        self.use_bootstrapper_attention_lora = bool(use_bootstrapper_attention_lora)
        self.bootstrapper_attention_lora_rank = int(bootstrapper_attention_lora_rank)
        self.bootstrapper_attention_lora_alpha = float(bootstrapper_attention_lora_alpha)
        self.bootstrapper_attention_lora_dropout = float(bootstrapper_attention_lora_dropout)
        self.bootstrapper_attention_lora_target = str(bootstrapper_attention_lora_target).strip().lower()
        self.bootstrapper_attention_lora_gate_init = float(bootstrapper_attention_lora_gate_init)
        self.bootstrapper_attention_lora_enable_q = bool(bootstrapper_attention_lora_enable_q)
        self.bootstrapper_attention_lora_enable_k = bool(bootstrapper_attention_lora_enable_k)
        self.bootstrapper_attention_lora_enable_v = bool(bootstrapper_attention_lora_enable_v)
        self.bootstrapper_attention_lora_enable_o = bool(bootstrapper_attention_lora_enable_o)
        self.bootstrapper_attention_lora_report = None

        if self.use_v66_f0_loss:
            self.v66_f0_head = nn.Sequential(
                nn.LayerNorm(self.acoustic_dim),
                nn.Linear(self.acoustic_dim, self.v66_f0_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.v66_f0_hidden_dim, 1),
            )
        else:
            self.v66_f0_head = None

        if self.use_v66_speaker_loss:
            self.v66_speaker_head = nn.Sequential(
                nn.LayerNorm(self.style_dim),
                nn.Linear(self.style_dim, self.v66_speaker_embedding_dim),
            )
        else:
            self.v66_speaker_head = None

        # ----------------------------------------------------
        # v6.6 reference acoustic style encoder
        # v6.6 参考音频声学风格编码器
        # ----------------------------------------------------
        self.use_reference_acoustic_style = bool(use_reference_acoustic_style)
        self.reference_style_fusion_mode = str(reference_style_fusion_mode).strip().lower()
        self.reference_style_gate_init = float(reference_style_gate_init)
        if self.reference_style_fusion_mode not in {"gated_add", "none"}:
            raise ValueError(
                f"reference_style_fusion_mode must be gated_add/none, got {reference_style_fusion_mode!r}"
            )
        if self.use_reference_acoustic_style:
            reference_style_cfg = ReferenceAcousticStyleEncoderConfig(
                acoustic_dim=int(reference_acoustic_dim),
                hidden_dim=int(reference_style_hidden_dim),
                style_dim=int(reference_style_dim),
                num_layers=int(reference_style_num_layers),
                kernel_size=int(reference_style_kernel_size),
                dropout=float(reference_style_dropout),
                pooling="attentive_mean",
            )
            self.reference_style_encoder = ReferenceAcousticStyleEncoder(reference_style_cfg)
            self.reference_style_to_model = nn.Linear(int(reference_style_dim), self.style_dim)
            self.reference_style_gate = nn.Parameter(torch.tensor(float(reference_style_gate_init)))
        else:
            self.reference_style_encoder = None
            self.reference_style_to_model = None
            self.reference_style_gate = None
        self._last_reference_style_stats: dict[str, Any] = {
            "enabled": int(self.use_reference_acoustic_style),
            "used": 0,
            "gate": 0.0,
        }

        # ----------------------------------------------------
        # v6.6.4 Speaker Identity Adapter
        # v6.6.4 说话人身份适配器
        # ----------------------------------------------------
        self.use_v664_speaker_identity_adapter = bool(use_v664_speaker_identity_adapter)
        self.v664_speaker_identity_source = str(v664_speaker_identity_source).strip().lower()
        self.v664_speaker_embedding_dim = int(v664_speaker_embedding_dim)
        self.v664_speaker_adapter_hidden_dim = int(v664_speaker_adapter_hidden_dim)
        self.v664_speaker_adapter_num_layers = int(v664_speaker_adapter_num_layers)
        self.v664_speaker_adapter_dropout = float(v664_speaker_adapter_dropout)
        self.v664_speaker_adapter_fusion_mode = str(v664_speaker_adapter_fusion_mode).strip().lower()
        self.v664_speaker_adapter_gate_init = float(v664_speaker_adapter_gate_init)
        self.v664_speaker_adapter_normalize = bool(v664_speaker_adapter_normalize)

        if self.use_v664_speaker_identity_adapter:
            speaker_identity_cfg = SpeakerIdentityAdapterConfig(
                speaker_embedding_dim=self.v664_speaker_embedding_dim,
                style_dim=self.style_dim,
                hidden_dim=self.v664_speaker_adapter_hidden_dim,
                num_layers=self.v664_speaker_adapter_num_layers,
                dropout=self.v664_speaker_adapter_dropout,
                fusion_mode=self.v664_speaker_adapter_fusion_mode,
                gate_init=self.v664_speaker_adapter_gate_init,
                normalize_input=self.v664_speaker_adapter_normalize,
            )
            self.speaker_identity_adapter = SpeakerIdentityAdapter(speaker_identity_cfg)
        else:
            self.speaker_identity_adapter = None

        self._last_v664_speaker_identity_stats: dict[str, Any] = {
            "enabled": int(self.use_v664_speaker_identity_adapter),
            "used": 0,
            "available": 0,
            "gate": 0.0,
        }

        # ----------------------------------------------------
        # v6.3.3 native text/content memory
        # ----------------------------------------------------
        self.v63_use_native_text_memory = bool(v63_use_native_text_memory)
        self.v63_text_memory_dim = int(v63_text_memory_dim or self.content_main_dim)

        if self.v63_use_native_text_memory:
            self.v63_native_text_memory_proj = nn.Sequential(
                nn.Linear(self.content_dim, self.v63_text_memory_dim),
                nn.LayerNorm(self.v63_text_memory_dim),
            )
        else:
            self.v63_native_text_memory_proj = None

        if not (0.0 <= self.residual_t_min < self.residual_t_max <= 1.0):
            raise ValueError(
                "Invalid residual t range: "
                f"residual_t_min={self.residual_t_min}, "
                f"residual_t_max={self.residual_t_max}"
            )

        # ----------------------------------------------------
        # v6.3 Conformer bootstrapper
        # ----------------------------------------------------
        if self.use_conformer_bootstrapper:
            bootstrap_text_dim = (
                self.v63_text_memory_dim
                if self.v63_use_native_text_memory
                else self.content_main_dim
            )

            bootstrap_cfg = ConformerMelBootstrapperConfig(
                input_dim=self.content_main_dim,
                semantic_dim=self.hidden_dim,
                text_dim=bootstrap_text_dim,
                style_dim=self.style_dim,

                hidden_dim=int(v63_bootstrap_hidden_dim or self.content_main_dim),
                n_mels=self.acoustic_dim,

                num_layers=int(v63_bootstrap_num_layers),
                num_heads=int(v63_bootstrap_num_heads),
                ff_mult=int(v63_bootstrap_ff_mult),
                conv_kernel_size=int(v63_bootstrap_conv_kernel_size),
                dropout=float(v63_bootstrap_dropout),

                use_semantic_cross_attn=True,
                use_text_cross_attn=True,
                use_style_film=True,
                use_style_residual=True,

                mel_bias_init=float(v63_bootstrap_mel_bias_init),
                zero_init_output=bool(v63_bootstrap_zero_init_output),
                clamp_output=bool(v63_bootstrap_clamp_output),
            )

            self.conformer_bootstrapper = ConformerMelBootstrapper(bootstrap_cfg)
            if self.use_bootstrapper_lora:
                self.bootstrapper_lora_report = apply_lora_to_named_linears(
                    self.conformer_bootstrapper,
                    target_mode=self.bootstrapper_lora_target,
                    rank=self.bootstrapper_lora_rank,
                    alpha=self.bootstrapper_lora_alpha,
                    dropout=self.bootstrapper_lora_dropout,
                    init_scale=self.bootstrapper_lora_init_scale,
                    freeze_base=True,
                )
                self.bootstrapper_lora_trainable_params = count_lora_parameters(self.conformer_bootstrapper)
            else:
                self.bootstrapper_lora_trainable_params = 0
            if self.use_bootstrapper_attention_lora:
                self.bootstrapper_attention_lora_report = apply_attention_lora_to_named_mha(
                    self.conformer_bootstrapper,
                    target_mode=self.bootstrapper_attention_lora_target,
                    rank=self.bootstrapper_attention_lora_rank,
                    alpha=self.bootstrapper_attention_lora_alpha,
                    dropout=self.bootstrapper_attention_lora_dropout,
                    gate_init=self.bootstrapper_attention_lora_gate_init,
                    enable_q=self.bootstrapper_attention_lora_enable_q,
                    enable_k=self.bootstrapper_attention_lora_enable_k,
                    enable_v=self.bootstrapper_attention_lora_enable_v,
                    enable_o=self.bootstrapper_attention_lora_enable_o,
                    freeze_base=True,
                )
                self.bootstrapper_attention_lora_trainable_params = count_attention_lora_parameters(self.conformer_bootstrapper)
            else:
                self.bootstrapper_attention_lora_trainable_params = 0
        else:
            self.conformer_bootstrapper = None
            self.bootstrapper_lora_trainable_params = 0

        # ----------------------------------------------------
        # v6.3 Residual flow refiner
        # ----------------------------------------------------
        if self.use_residual_refiner:
            residual_text_dim = (
                self.v63_text_memory_dim
                if self.v63_use_native_text_memory
                else self.content_main_dim
            )

            residual_cfg = ResidualFlowRefinerConfig(
                n_mels=self.acoustic_dim,
                input_dim=self.content_main_dim,
                semantic_dim=self.hidden_dim,
                text_dim=residual_text_dim,
                style_dim=self.style_dim,

                hidden_dim=int(residual_hidden_dim or self.content_main_dim),
                num_layers=int(residual_num_layers),
                num_heads=int(residual_num_heads),
                ff_mult=int(residual_ff_mult),
                conv_kernel_size=int(residual_conv_kernel_size),
                dropout=float(residual_dropout),

                use_content_frame=True,
                use_semantic_cross_attn=True,
                use_text_cross_attn=True,
                use_style_film=True,
                use_self_condition=True,

                zero_init_output=True,
            )

            self.residual_refiner = ResidualFlowRefiner(residual_cfg)
        else:
            self.residual_refiner = None



    # --------------------------------------------------------
    # v6.6.2 prosody / speaker auxiliary loss helper
    # v6.6.2 韵律 / 说话人辅助损失辅助函数
    # --------------------------------------------------------
    def _compute_v66_aux_losses(
        self,
        *,
        coarse_mel: torch.Tensor,
        pack: dict[str, Any],
        batch: Stage2Inputs,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        extras = getattr(batch, "extras", None)
        if not isinstance(extras, dict):
            extras = {}

        aux_loss = coarse_mel.new_tensor(0.0)
        aux: dict[str, torch.Tensor] = {}

        if self.use_v66_energy_loss:
            energy_loss, energy_aux = compute_v66_energy_loss(
                pred_mel=coarse_mel,
                target_energy=extras.get("target_energy"),
                target_lengths=target_lengths,
                normalize=self.v66_energy_normalize,
                loss_type=self.v66_energy_loss_type,
            )
            aux_loss = aux_loss + self.v66_energy_loss_weight * energy_loss
            aux.update(energy_aux)
            aux["v66_energy_weighted_loss"] = (
                self.v66_energy_loss_weight * energy_loss
            ).detach()
            if self.v66_require_aux_targets and float(energy_aux.get("v66_energy_available", coarse_mel.new_tensor(0.0)).item()) < 0.5:
                raise ValueError("use_v66_energy_loss=True but batch.extras['target_energy'] is missing.")

        if self.use_v66_f0_loss:
            if self.v66_f0_head is None:
                raise RuntimeError("use_v66_f0_loss=True but v66_f0_head is None.")
            pred_f0 = self.v66_f0_head(coarse_mel).squeeze(-1)
            f0_loss, f0_aux = compute_v66_f0_loss(
                pred_f0=pred_f0,
                target_f0=extras.get("target_f0"),
                target_f0_voiced_mask=extras.get("target_f0_voiced_mask"),
                target_lengths=target_lengths,
                normalize=self.v66_f0_normalize,
                loss_type=self.v66_f0_loss_type,
            )
            aux_loss = aux_loss + self.v66_f0_loss_weight * f0_loss
            aux.update(f0_aux)
            aux["v66_f0_weighted_loss"] = (self.v66_f0_loss_weight * f0_loss).detach()
            if self.v66_require_aux_targets and float(f0_aux.get("v66_f0_available", coarse_mel.new_tensor(0.0)).item()) < 0.5:
                raise ValueError("use_v66_f0_loss=True but F0 target fields are missing.")

        if self.use_v66_speaker_loss:
            if self.v66_speaker_head is None:
                raise RuntimeError("use_v66_speaker_loss=True but v66_speaker_head is None.")
            style_global = pack.get("style_global")
            pred_speaker = self.v66_speaker_head(style_global) if torch.is_tensor(style_global) else None
            target_speaker = extras.get("target_speaker_embedding")
            if target_speaker is None:
                target_speaker = extras.get("prompt_speaker_embedding")
            speaker_loss, speaker_aux = compute_v66_speaker_loss(
                pred_speaker_embedding=pred_speaker,
                target_speaker_embedding=target_speaker,
            )
            speaker_loss = speaker_loss.to(device=coarse_mel.device, dtype=coarse_mel.dtype)
            aux_loss = aux_loss + self.v66_speaker_loss_weight * speaker_loss
            aux.update(speaker_aux)
            aux["v66_speaker_weighted_loss"] = (
                self.v66_speaker_loss_weight * speaker_loss
            ).detach()
            if self.v66_require_aux_targets and float(speaker_aux.get("v66_speaker_available", coarse_mel.new_tensor(0.0)).item()) < 0.5:
                raise ValueError("use_v66_speaker_loss=True but speaker embedding target is missing.")

        aux["v66_aux_total_loss"] = aux_loss.detach()
        return aux_loss, aux


    # --------------------------------------------------------
    # v6.6.4 speaker identity adapter helper
    # v6.6.4 说话人身份适配器辅助函数
    # --------------------------------------------------------
    def _fuse_v664_speaker_identity(
        self,
        *,
        batch: Stage2Inputs,
        style_global: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self._last_v664_speaker_identity_stats = {
            "enabled": int(bool(getattr(self, "use_v664_speaker_identity_adapter", False))),
            "used": 0,
            "available": 0,
            "gate": 0.0,
        }

        if not bool(getattr(self, "use_v664_speaker_identity_adapter", False)):
            return style_global
        if self.speaker_identity_adapter is None:
            return style_global

        extras = getattr(batch, "extras", None)
        speaker_embedding, source_name = select_v664_speaker_identity_embedding(
            extras if isinstance(extras, dict) else None,
            source=self.v664_speaker_identity_source,
        )
        if not torch.is_tensor(speaker_embedding):
            self._last_v664_speaker_identity_stats.update({
                "available": 0,
                "used": 0,
                "source_id": -1,
            })
            return style_global

        if speaker_embedding.dim() > 2:
            speaker_embedding = speaker_embedding.view(speaker_embedding.shape[0], -1)
        if speaker_embedding.dim() != 2:
            raise ValueError(
                "v6.6.4 speaker identity embedding must be 2D after flattening, "
                f"got {tuple(speaker_embedding.shape)}"
            )

        B = int(speaker_embedding.shape[0])
        device = speaker_embedding.device
        dtype = speaker_embedding.dtype

        if torch.is_tensor(style_global):
            device = style_global.device
            dtype = style_global.dtype
            B = int(style_global.shape[0])
        else:
            style_global = torch.zeros(
                B,
                self.style_dim,
                device=device,
                dtype=dtype,
            )

        fused, stats = self.speaker_identity_adapter(
            style_global=style_global,
            speaker_embedding=speaker_embedding,
        )

        source_id_map = {
            "prompt": 1,
            "target": 2,
        }
        source_id = source_id_map.get(str(source_name), 0)

        self._last_v664_speaker_identity_stats = {
            "enabled": 1,
            "available": 1,
            "used": 1,
            "source_id": int(source_id),
        }
        for key, value in stats.items():
            if torch.is_tensor(value):
                self._last_v664_speaker_identity_stats[key] = float(value.detach().float().mean().cpu().item())
            else:
                self._last_v664_speaker_identity_stats[key] = float(value)

        return fused

    # --------------------------------------------------------
    # v6.6 reference acoustic style helpers
    # v6.6 参考音频声学风格辅助函数
    # --------------------------------------------------------
    def _fuse_v66_reference_style(
        self,
        *,
        batch: Stage2Inputs,
        style_global: torch.Tensor | None,
        style_keep: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self._last_reference_style_stats = {
            "enabled": int(bool(getattr(self, "use_reference_acoustic_style", False))),
            "used": 0,
            "gate": 0.0,
        }
        if not bool(getattr(self, "use_reference_acoustic_style", False)):
            return style_global
        if self.reference_style_encoder is None or self.reference_style_to_model is None:
            return style_global
        if self.reference_style_fusion_mode == "none":
            return style_global
        extras = getattr(batch, "extras", None)
        if not isinstance(extras, dict):
            return style_global
        prompt_acoustic = extras.get("prompt_acoustic", None)
        prompt_acoustic_lengths = extras.get("prompt_acoustic_lengths", None)
        if not torch.is_tensor(prompt_acoustic):
            return style_global

        if style_global is None:
            device = prompt_acoustic.device
            dtype = prompt_acoustic.dtype
            B = int(prompt_acoustic.shape[0])
            style_global = torch.zeros(B, self.style_dim, device=device, dtype=dtype)
        else:
            device = style_global.device
            dtype = style_global.dtype
            B = int(style_global.shape[0])

        prompt_acoustic = prompt_acoustic.to(device=device, dtype=torch.float32)
        if torch.is_tensor(prompt_acoustic_lengths):
            prompt_acoustic_lengths = prompt_acoustic_lengths.to(device=device).long()
        else:
            prompt_acoustic_lengths = torch.full(
                size=(B,),
                fill_value=int(prompt_acoustic.shape[1]),
                dtype=torch.long,
                device=device,
            )

        ref_out = self.reference_style_encoder(
            prompt_acoustic=prompt_acoustic,
            prompt_acoustic_lengths=prompt_acoustic_lengths,
        )
        ref_style = self.reference_style_to_model(ref_out.style_global).to(device=device, dtype=dtype)
        if style_keep is not None:
            ref_style = ref_style * style_keep.to(device=device, dtype=dtype).view(B, 1)
        gate = torch.sigmoid(self.reference_style_gate).to(device=device, dtype=dtype)
        fused = style_global + gate * ref_style
        self._last_reference_style_stats = {
            "enabled": 1,
            "used": 1,
            "gate": float(gate.detach().cpu().item()),
            "ref_style_norm_mean": float(ref_style.detach().float().norm(dim=-1).mean().cpu().item()),
            "fused_style_norm_mean": float(fused.detach().float().norm(dim=-1).mean().cpu().item()),
            "prompt_acoustic_len_mean": float(prompt_acoustic_lengths.detach().float().mean().cpu().item()),
        }
        return fused

    # --------------------------------------------------------
    # v6.3.3 native text memory helper
    # --------------------------------------------------------
    def _build_v63_text_memory(
        self,
        *,
        cond: Stage2ConditionBundleV6,
        keep_masks: dict[str, torch.Tensor],
        content_frame: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Build text_memory for v6.3+ modules.

        Old v6.3 behavior:
            text_memory = content_frame
            text_lengths = target_lengths

        v6.3.3 native text memory behavior:
            text_memory = projection(cond.content_states)
            text_lengths = cond.content_lengths / cond.content_mask

        Important:
            If v63_use_native_text_memory=True, this method does NOT silently
            fallback to content_frame. It raises immediately when content_states
            is unavailable. This avoids hidden regressions where a model appears
            to use native text memory but actually does not.
        """
        if not self.v63_use_native_text_memory:
            return content_frame, target_lengths, "content_frame_fallback"

        if self.v63_native_text_memory_proj is None:
            raise RuntimeError(
                "v63_use_native_text_memory=True but "
                "v63_native_text_memory_proj is None."
            )

        if cond.content_states is None:
            raise ValueError(
                "v63_use_native_text_memory=True but cond.content_states is None. "
                "This should not happen when Stage2ConditionEncoder.build_conditions_v6_1 "
                "returns a valid native content stream."
            )

        text_memory = self.v63_native_text_memory_proj(cond.content_states)

        if cond.content_mask is not None:
            content_mask = cond.content_mask.to(
                device=text_memory.device,
                dtype=text_memory.dtype,
            )
            text_memory = text_memory * content_mask.unsqueeze(-1)

            if cond.content_lengths is not None:
                text_lengths = cond.content_lengths.long().to(text_memory.device)
            else:
                text_lengths = cond.content_mask.long().sum(dim=1).to(text_memory.device)
        elif cond.content_lengths is not None:
            text_lengths = cond.content_lengths.long().to(text_memory.device)
        else:
            text_lengths = torch.full(
                size=(text_memory.shape[0],),
                fill_value=text_memory.shape[1],
                dtype=torch.long,
                device=text_memory.device,
            )

        text_lengths = text_lengths.clamp_min(1).clamp(max=text_memory.shape[1])

        # Apply content keep mask. Under force_mode="uncond", content keep is 0,
        # so native text memory is also dropped.
        content_keep = keep_masks["content"].to(
            device=text_memory.device,
            dtype=text_memory.dtype,
        ).view(text_memory.shape[0], 1, 1)
        text_memory = text_memory * content_keep

        return text_memory, text_lengths, "native_content_states"

    # --------------------------------------------------------
    # v6.3 condition packing
    # --------------------------------------------------------
    def _build_v63_condition_pack(
        self,
        batch: Stage2Inputs,
        target_len: int,
        target_lengths: torch.Tensor,
        force_mode: str = "full",
    ) -> dict[str, Any]:
        """
        Build v6.3 condition pack.

        It reuses the v6.1 condition encoder, but exposes:
        - content_frame: acoustic-frame condition
        - semantic_guide: native semantic-length memory
        - text_memory:
            old v6.3: acoustic-frame content fallback
            v6.3.3: native content/text memory from cond.content_states
        - style_global
        - lengths
        """
        cond = self._build_v61_conditions(
            batch=batch,
            target_len=target_len,
            target_lengths=target_lengths,
        )

        keep_masks = self._make_keep_masks(
            batch=batch,
            force_mode=force_mode,
        )

        cond_tensors = self._resolve_condition_tensors(
            cond=cond,
            keep_masks=keep_masks,
            target_len=target_len,
        )

        content_frame = cond_tensors["content_frame"]
        if content_frame is None:
            raise ValueError("content_frame is required for v6.3.")

        style_global = cond_tensors["style_global"]
        style_global = self._fuse_v66_reference_style(
            batch=batch,
            style_global=style_global,
            style_keep=keep_masks.get("style"),
        )
        style_global = self._fuse_v664_speaker_identity(
            batch=batch,
            style_global=style_global,
        )

        semantic_guide = cond_tensors["semantic_guide"]
        semantic_keep = cond_tensors.get("semantic_keep_mask")

        if semantic_guide is not None and semantic_keep is not None:
            semantic_guide = semantic_guide * semantic_keep.to(
                device=semantic_guide.device,
                dtype=semantic_guide.dtype,
            ).view(-1, 1, 1)

        if cond.semantic_guide_mask is not None:
            semantic_lengths = cond.semantic_guide_mask.long().sum(dim=1).clamp_min(1)
        elif semantic_guide is not None:
            semantic_lengths = torch.full(
                size=(semantic_guide.shape[0],),
                fill_value=semantic_guide.shape[1],
                dtype=torch.long,
                device=semantic_guide.device,
            )
        else:
            semantic_lengths = None

        text_memory, text_lengths, text_memory_source = self._build_v63_text_memory(
            cond=cond,
            keep_masks=keep_masks,
            content_frame=content_frame,
            target_lengths=target_lengths,
        )

        valid_mask = make_length_mask(target_lengths, target_len)

        return {
            "cond": cond,
            "cond_tensors": cond_tensors,
            "content_frame": content_frame,
            "semantic_guide": semantic_guide,
            "semantic_lengths": semantic_lengths,
            "text_memory": text_memory,
            "text_lengths": text_lengths,
            "text_memory_source": text_memory_source,
            "text_memory_shape": tuple(text_memory.shape),
            "text_lengths_mean": text_lengths.float().mean().detach(),
            "style_global": style_global,
            "reference_style_stats": dict(getattr(self, "_last_reference_style_stats", {})),
            "speaker_identity_stats": dict(getattr(self, "_last_v664_speaker_identity_stats", {})),
            "target_lengths": target_lengths,
            "valid_mask": valid_mask,
        }

    # --------------------------------------------------------
    # v6.3 coarse mel prediction
    # --------------------------------------------------------
    def predict_coarse_mel(
        self,
        batch: Stage2Inputs,
        target_len: int | None = None,
        target_lengths: torch.Tensor | None = None,
        force_mode: str = "full",
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        v6.3 condition -> coarse_mel_pred using ConformerMelBootstrapper.
        """
        if self.conformer_bootstrapper is None:
            raise RuntimeError("conformer_bootstrapper is disabled.")

        if target_len is None:
            if target_lengths is not None:
                target_len = int(target_lengths.max().item())
            elif batch.target_acoustic is not None:
                target_len = int(batch.target_acoustic.shape[1])
            else:
                inferred = self.infer_acoustic_lengths(batch)
                target_len = int(inferred.max().item())

        resolved_lengths = self._resolve_acoustic_lengths(
            batch=batch,
            target_len=target_len,
            target_lengths_override=target_lengths,
            prefer_target_lengths=True,
        )

        pack = self._build_v63_condition_pack(
            batch=batch,
            target_len=target_len,
            target_lengths=resolved_lengths,
            force_mode=force_mode,
        )

        out = self.conformer_bootstrapper(
            content_frame=pack["content_frame"],
            semantic_guide=pack["semantic_guide"],
            text_memory=pack["text_memory"],
            style_global=pack["style_global"],
            target_lengths=pack["target_lengths"],
            semantic_lengths=pack["semantic_lengths"],
            text_lengths=pack["text_lengths"],
            return_aux=return_aux,
        )

        return out

    # --------------------------------------------------------
    # v6.3 coarse-only sampling
    # --------------------------------------------------------
    @torch.inference_mode()
    def sample_coarse_only(
        self,
        batch: Stage2Inputs,
        target_lengths: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Stage2Outputs:
        """
        v6.3 coarse-only inference path.

        This path directly returns:
            final_mel = conformer_bootstrapper(condition)

        It intentionally skips ResidualFlowRefiner.

        This is useful after the v6.3 10h diagnosis showed:
            coarse_only <= direct_t1_zero < direct_t0_zero << residual_sampled

        In other words, the residual refiner currently does not improve
        coarse mel and sampled residual clearly damages the output.
        """
        if self.conformer_bootstrapper is None:
            raise RuntimeError("sample_coarse_only requires conformer_bootstrapper.")

        device = batch.semantic_tokens.device

        if target_lengths is None:
            target_lengths = self.infer_acoustic_lengths(batch)
        else:
            target_lengths = target_lengths.long().to(device).clamp_min(1)

        max_len = int(target_lengths.max().item())

        pack = self._build_v63_condition_pack(
            batch=batch,
            target_len=max_len,
            target_lengths=target_lengths,
            force_mode="full",
        )

        coarse_mel, coarse_aux = self.conformer_bootstrapper(
            content_frame=pack["content_frame"],
            semantic_guide=pack["semantic_guide"],
            text_memory=pack["text_memory"],
            style_global=pack["style_global"],
            target_lengths=pack["target_lengths"],
            semantic_lengths=pack["semantic_lengths"],
            text_lengths=pack["text_lengths"],
            return_aux=True,
        )

        valid_mask = make_length_mask(target_lengths, max_len)
        coarse_mel = coarse_mel * valid_mask.unsqueeze(-1).to(dtype=coarse_mel.dtype)

        aux: dict[str, Any] = {
            "model_version": self.model_version,
            "model_variant": self.model_variant,
            "sampling_start_mode": "coarse_only",
            "residual_refiner_skipped": True,

            "v63_use_native_text_memory": bool(self.v63_use_native_text_memory),
            "text_memory_source": pack.get("text_memory_source"),
            "text_memory_shape": pack.get("text_memory_shape"),
            "text_lengths_mean": (
                float(pack["text_lengths_mean"].detach().cpu().item())
                if torch.is_tensor(pack.get("text_lengths_mean"))
                else pack.get("text_lengths_mean")
            ),

            "inferred_lengths": target_lengths.detach().cpu().tolist(),
            "semantic_rate_hz": float(self.semantic_rate_hz),
            "acoustic_rate_hz": float(self.acoustic_rate_hz),
        }

        if isinstance(coarse_aux, dict):
            for key, value in coarse_aux.items():
                if torch.is_tensor(value):
                    aux[f"coarse_{key}"] = value.detach().cpu()
                else:
                    aux[f"coarse_{key}"] = value

        if return_intermediates:
            aux["coarse_mel_pred"] = coarse_mel.detach().cpu()
            aux["final_mel"] = coarse_mel.detach().cpu()

        return Stage2Outputs(
            acoustic=coarse_mel,
            acoustic_type="mel",
            lengths=target_lengths,
            aux=aux,
        )

    # --------------------------------------------------------
    # v6.3 training loss
    # --------------------------------------------------------
    def compute_flow_matching_loss(
        self,
        batch: Stage2Inputs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute v6.3 training loss.

        Main components:
        - optional v6.1 base loss
        - Conformer bootstrapper coarse mel loss
        - ResidualFlowRefiner loss

        This method intentionally does not call V62 loss, because V63 replaces
        the v6.2 bridge/full-mel bootstrap refinement with residual refinement.
        """
        if batch.target_acoustic is None:
            raise ValueError("batch.target_acoustic is required for v6.3 loss.")

        x1 = batch.target_acoustic.float()
        B, T_ac, _ = x1.shape

        target_lengths = self._resolve_acoustic_lengths(
            batch=batch,
            target_len=T_ac,
            target_lengths_override=batch.target_lengths,
            prefer_target_lengths=True,
        )

        valid_mask = make_length_mask(target_lengths, T_ac)
        valid_mask_f = valid_mask.unsqueeze(-1).to(dtype=x1.dtype)

        # ----------------------------------------------------
        # Optional v6.1 base loss
        # ----------------------------------------------------
        if self.include_v61_base_loss:
            base_loss, base_aux = Stage2FlowMatchingAcousticModelV61.compute_flow_matching_loss(
                self,
                batch,
            )
            base_loss = self.v61_base_loss_weight * base_loss
        else:
            base_loss = x1.new_tensor(0.0)
            base_aux = {}

        # ----------------------------------------------------
        # Build conditions once for v6.3 modules
        # ----------------------------------------------------
        pack = self._build_v63_condition_pack(
            batch=batch,
            target_len=T_ac,
            target_lengths=target_lengths,
            force_mode="dropout",
        )

        # ----------------------------------------------------
        # 1. Conformer coarse mel loss
        # ----------------------------------------------------
        coarse_out = self.conformer_bootstrapper(
            content_frame=pack["content_frame"],
            semantic_guide=pack["semantic_guide"],
            text_memory=pack["text_memory"],
            style_global=pack["style_global"],
            target_lengths=pack["target_lengths"],
            semantic_lengths=pack["semantic_lengths"],
            text_lengths=pack["text_lengths"],
            return_aux=True,
        )

        coarse_mel, coarse_aux = coarse_out
        coarse_mel = coarse_mel * valid_mask_f

        coarse_losses = compute_coarse_mel_losses(
            coarse_mel=coarse_mel,
            target_acoustic=x1,
            target_lengths=target_lengths,
            l1_weight=self.v63_coarse_l1_weight,
            mse_weight=self.v63_coarse_mse_weight,
            delta_l1_weight=self.v63_coarse_delta_weight,
            delta2_l1_weight=self.v63_coarse_delta2_weight,
        )

        coarse_total_loss = coarse_losses["coarse_mel_total_loss"]

        # ----------------------------------------------------
        # v6.6.2 auxiliary prosody / speaker losses
        # ----------------------------------------------------
        v66_aux_loss, v66_aux = self._compute_v66_aux_losses(
            coarse_mel=coarse_mel,
            pack=pack,
            batch=batch,
            target_lengths=target_lengths,
        )

        # ----------------------------------------------------
        # 2. Residual refiner loss
        # ----------------------------------------------------
        if self.use_residual_refiner and self.residual_refiner is not None:
            coarse_for_residual = (
                coarse_mel.detach()
                if self.detach_coarse_for_residual_refiner
                else coarse_mel
            )

            residual_loss, residual_aux = compute_residual_refiner_training_loss(
                refiner=self.residual_refiner,
                coarse_mel=coarse_for_residual,
                target_acoustic=x1,

                content_frame=pack["content_frame"],
                semantic_guide=pack["semantic_guide"],
                text_memory=pack["text_memory"],
                style_global=pack["style_global"],

                target_lengths=target_lengths,
                semantic_lengths=pack["semantic_lengths"],
                text_lengths=pack["text_lengths"],

                noise_scale=self.residual_noise_scale,
                min_t=self.residual_t_min,
                max_t=self.residual_t_max,

                flow_mse_weight=self.residual_flow_mse_weight,
                flow_l1_weight=self.residual_flow_l1_weight,
                residual_recon_l1_weight=self.residual_recon_l1_weight,
                final_recon_l1_weight=self.final_recon_l1_weight,
                final_delta_l1_weight=self.final_delta_l1_weight,
                final_delta2_l1_weight=self.final_delta2_l1_weight,
            )

            residual_total_loss = self.residual_refiner_loss_weight * residual_loss
        else:
            residual_total_loss = x1.new_tensor(0.0)
            residual_aux = {}

        # ----------------------------------------------------
        # 3. Total loss
        # ----------------------------------------------------
        loss = base_loss + coarse_total_loss + residual_total_loss + v66_aux_loss

        log_aux: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "v63_base_loss": base_loss.detach(),
            "v63_coarse_total_loss": coarse_total_loss.detach(),
            "v63_residual_total_loss": residual_total_loss.detach(),
            "v66_aux_total_loss": v66_aux_loss.detach(),

            "coarse_mel_l1_loss": coarse_losses["coarse_mel_l1_loss"].detach(),
            "coarse_mel_mse_loss": coarse_losses["coarse_mel_mse_loss"].detach(),
            "coarse_mel_delta_l1_loss": coarse_losses["coarse_mel_delta_l1_loss"].detach(),
            "coarse_mel_delta2_l1_loss": coarse_losses["coarse_mel_delta2_l1_loss"].detach(),
            "coarse_mel_total_loss": coarse_total_loss.detach(),

            "target_length_mean": target_lengths.float().mean().detach(),
            "inferred_length_mean": self.infer_acoustic_lengths(batch).float().mean().detach(),

            "v63_use_native_text_memory": x1.new_tensor(
                1.0 if self.v63_use_native_text_memory else 0.0
            ).detach(),
            "text_lengths_mean": pack["text_lengths"].float().mean().detach(),
            "text_memory_is_native": x1.new_tensor(
                1.0 if pack.get("text_memory_source") == "native_content_states" else 0.0
            ).detach(),
        }


        ref_stats = pack.get("reference_style_stats", {})
        if isinstance(ref_stats, dict):
            for key in [
                "enabled",
                "used",
                "gate",
                "ref_style_norm_mean",
                "fused_style_norm_mean",
                "prompt_acoustic_len_mean",
            ]:
                if key in ref_stats:
                    log_aux[f"reference_style_{key}"] = x1.new_tensor(float(ref_stats[key])).detach()

        speaker_identity_stats = pack.get("speaker_identity_stats", {})
        if isinstance(speaker_identity_stats, dict):
            for key in [
                "enabled",
                "available",
                "used",
                "source_id",
                "gate",
                "speaker_embedding_norm_mean",
                "speaker_style_norm_mean",
                "input_style_norm_mean",
                "fused_style_norm_mean",
            ]:
                if key in speaker_identity_stats:
                    log_aux[f"v664_speaker_identity_{key}"] = x1.new_tensor(float(speaker_identity_stats[key])).detach()


        # ----------------------------------------------------
        # v6.6.3-A Attention LoRA logging
        # v6.6.3-A Attention LoRA 日志
        # ----------------------------------------------------
        attn_lora_enabled = bool(getattr(self, "use_bootstrapper_attention_lora", False))
        attn_lora_report = getattr(self, "bootstrapper_attention_lora_report", None)
        attn_lora_target = str(getattr(self, "bootstrapper_attention_lora_target", "none")).strip().lower()
        attn_lora_target_id_map = {
            "cross": 1,
            "cross_only": 1,
            "semantic_text": 1,
            "all": 2,
            "all_attention": 2,
            "self": 3,
            "self_only": 3,
            "semantic": 4,
            "semantic_cross": 4,
            "text": 5,
            "text_cross": 5,
        }

        attn_lora_num_replaced = 0
        if attn_lora_report is not None:
            attn_lora_num_replaced = int(getattr(attn_lora_report, "num_replaced", 0))

        log_aux["bootstrapper_attention_lora_enabled"] = x1.new_tensor(1.0 if attn_lora_enabled else 0.0).detach()
        log_aux["bootstrapper_attention_lora_num_replaced"] = x1.new_tensor(float(attn_lora_num_replaced)).detach()
        log_aux["bootstrapper_attention_lora_trainable_params"] = x1.new_tensor(
            float(getattr(self, "bootstrapper_attention_lora_trainable_params", 0))
        ).detach()
        log_aux["bootstrapper_attention_lora_target_id"] = x1.new_tensor(
            float(attn_lora_target_id_map.get(attn_lora_target, 0))
        ).detach()
        log_aux["bootstrapper_attention_lora_q_enabled"] = x1.new_tensor(
            1.0 if bool(getattr(self, "bootstrapper_attention_lora_enable_q", False)) else 0.0
        ).detach()
        log_aux["bootstrapper_attention_lora_k_enabled"] = x1.new_tensor(
            1.0 if bool(getattr(self, "bootstrapper_attention_lora_enable_k", False)) else 0.0
        ).detach()
        log_aux["bootstrapper_attention_lora_v_enabled"] = x1.new_tensor(
            1.0 if bool(getattr(self, "bootstrapper_attention_lora_enable_v", False)) else 0.0
        ).detach()
        log_aux["bootstrapper_attention_lora_o_enabled"] = x1.new_tensor(
            1.0 if bool(getattr(self, "bootstrapper_attention_lora_enable_o", False)) else 0.0
        ).detach()

        attn_gate_raw_values = []
        attn_gate_sigmoid_values = []
        bootstrapper = getattr(self, "conformer_bootstrapper", None)
        if bootstrapper is not None:
            for module in bootstrapper.modules():
                gate = getattr(module, "attention_lora_gate", None)
                if torch.is_tensor(gate):
                    gate_raw = gate.detach().float().mean()
                    attn_gate_raw_values.append(gate_raw)
                    attn_gate_sigmoid_values.append(torch.sigmoid(gate_raw))

        if attn_gate_raw_values:
            raw_stack = torch.stack(attn_gate_raw_values)
            sigmoid_stack = torch.stack(attn_gate_sigmoid_values)
            log_aux["bootstrapper_attention_lora_gate_raw_mean"] = raw_stack.mean().to(device=x1.device, dtype=x1.dtype).detach()
            log_aux["bootstrapper_attention_lora_gate_sigmoid_mean"] = sigmoid_stack.mean().to(device=x1.device, dtype=x1.dtype).detach()
            log_aux["bootstrapper_attention_lora_gate_sigmoid_min"] = sigmoid_stack.min().to(device=x1.device, dtype=x1.dtype).detach()
            log_aux["bootstrapper_attention_lora_gate_sigmoid_max"] = sigmoid_stack.max().to(device=x1.device, dtype=x1.dtype).detach()
        else:
            log_aux["bootstrapper_attention_lora_gate_raw_mean"] = x1.new_tensor(0.0).detach()
            log_aux["bootstrapper_attention_lora_gate_sigmoid_mean"] = x1.new_tensor(0.0).detach()
            log_aux["bootstrapper_attention_lora_gate_sigmoid_min"] = x1.new_tensor(0.0).detach()
            log_aux["bootstrapper_attention_lora_gate_sigmoid_max"] = x1.new_tensor(0.0).detach()

        for key, value in v66_aux.items():
            if torch.is_tensor(value):
                log_aux[key] = value.detach()

        for key, value in residual_aux.items():
            if torch.is_tensor(value):
                log_aux[key] = value.detach()

        for key, value in base_aux.items():
            if torch.is_tensor(value):
                log_aux[f"v61_{key}"] = value.detach()

        if isinstance(coarse_aux, dict):
            for key, value in coarse_aux.items():
                if torch.is_tensor(value):
                    log_aux[f"coarse_{key}"] = value.detach()

        return loss, log_aux

    # --------------------------------------------------------
    # v6.3 residual-refine sampling
    # --------------------------------------------------------
    @torch.inference_mode()
    def sample_with_residual_refiner(
        self,
        batch: Stage2Inputs,
        num_steps: int = 128,
        target_lengths: torch.Tensor | None = None,
        temperature: float = 0.3,
        use_heun: bool = True,
        guidance_scale: float = 1.0,
        return_intermediates: bool = False,
    ) -> Stage2Outputs:
        """
        v6.3 sampling path.

        Steps:
            1. condition -> conformer coarse_mel
            2. residual_refiner samples residual
            3. final_mel = coarse_mel + residual

        Note:
            guidance_scale is currently recorded for compatibility, but the
            residual refiner path does not yet implement classifier-free guidance.
        """
        if self.conformer_bootstrapper is None:
            raise RuntimeError("sample_with_residual_refiner requires conformer_bootstrapper.")

        device = batch.semantic_tokens.device

        if target_lengths is None:
            target_lengths = self.infer_acoustic_lengths(batch)
        else:
            target_lengths = target_lengths.long().to(device).clamp_min(1)

        max_len = int(target_lengths.max().item())

        pack = self._build_v63_condition_pack(
            batch=batch,
            target_len=max_len,
            target_lengths=target_lengths,
            force_mode="full",
        )

        coarse_mel, coarse_aux = self.conformer_bootstrapper(
            content_frame=pack["content_frame"],
            semantic_guide=pack["semantic_guide"],
            text_memory=pack["text_memory"],
            style_global=pack["style_global"],
            target_lengths=pack["target_lengths"],
            semantic_lengths=pack["semantic_lengths"],
            text_lengths=pack["text_lengths"],
            return_aux=True,
        )

        valid_mask = make_length_mask(target_lengths, max_len)
        coarse_mel = coarse_mel * valid_mask.unsqueeze(-1).to(dtype=coarse_mel.dtype)

        if self.use_residual_refiner and self.residual_refiner is not None:
            residual_out = self.residual_refiner.sample(
                coarse_mel=coarse_mel,
                content_frame=pack["content_frame"],
                semantic_guide=pack["semantic_guide"],
                text_memory=pack["text_memory"],
                style_global=pack["style_global"],
                target_lengths=target_lengths,
                semantic_lengths=pack["semantic_lengths"],
                text_lengths=pack["text_lengths"],
                num_steps=int(num_steps),
                temperature=float(temperature),
                use_heun=bool(use_heun),
                return_aux=True,
            )

            final_mel = residual_out["final_mel"]
            predicted_residual = residual_out["predicted_residual"]
            residual_aux = residual_out.get("aux", {})
        else:
            final_mel = coarse_mel
            predicted_residual = torch.zeros_like(coarse_mel)
            residual_aux = {}

        final_mel = final_mel * valid_mask.unsqueeze(-1).to(dtype=final_mel.dtype)

        aux: dict[str, Any] = {
            "num_steps": int(num_steps),
            "model_version": self.model_version,
            "model_variant": self.model_variant,

            "sampling_start_mode": "residual_refine",
            "use_heun": bool(use_heun),
            "guidance_scale": float(guidance_scale),
            "temperature": float(temperature),

            "v63_use_native_text_memory": bool(self.v63_use_native_text_memory),
            "text_memory_source": pack.get("text_memory_source"),
            "text_memory_shape": pack.get("text_memory_shape"),
            "text_lengths_mean": (
                float(pack["text_lengths_mean"].detach().cpu().item())
                if torch.is_tensor(pack.get("text_lengths_mean"))
                else pack.get("text_lengths_mean")
            ),

            "inferred_lengths": target_lengths.detach().cpu().tolist(),
            "semantic_rate_hz": float(self.semantic_rate_hz),
            "acoustic_rate_hz": float(self.acoustic_rate_hz),
        }

        if isinstance(coarse_aux, dict):
            for key, value in coarse_aux.items():
                if torch.is_tensor(value):
                    aux[f"coarse_{key}"] = value.detach().cpu()
                else:
                    aux[f"coarse_{key}"] = value

        if isinstance(residual_aux, dict):
            for key, value in residual_aux.items():
                if torch.is_tensor(value):
                    aux[f"residual_{key}"] = value.detach().cpu()
                else:
                    aux[f"residual_{key}"] = value

        if return_intermediates:
            aux["coarse_mel_pred"] = coarse_mel.detach().cpu()
            aux["predicted_residual"] = predicted_residual.detach().cpu()
            aux["final_mel"] = final_mel.detach().cpu()

        return Stage2Outputs(
            acoustic=final_mel,
            acoustic_type="mel",
            lengths=target_lengths,
            aux=aux,
        )

    # --------------------------------------------------------
    # Compatibility alias for current pipeline
    # --------------------------------------------------------
    @torch.inference_mode()
    def sample_with_bootstrap(
        self,
        batch: Stage2Inputs,
        num_steps: int = 128,
        target_lengths: torch.Tensor | None = None,
        temperature: float = 0.3,
        use_heun: bool = True,
        guidance_scale: float = 1.0,
        bootstrap_t_start: float = 0.075,
        bootstrap_noise_temperature: float | None = None,
        return_bootstrap_mel: bool = False,
    ) -> Stage2Outputs:
        """
        Compatibility alias.

        Existing v6.2 pipeline calls sample_with_bootstrap(...).
        For v6.3, we route it to residual refinement sampling.

        bootstrap_t_start / bootstrap_noise_temperature are kept in the
        signature for compatibility, but v6.3 does not use bridge-start
        full-mel sampling.
        """
        residual_temperature = (
            float(bootstrap_noise_temperature)
            if bootstrap_noise_temperature is not None
            else float(temperature)
        )

        out = self.sample_with_residual_refiner(
            batch=batch,
            num_steps=num_steps,
            target_lengths=target_lengths,
            temperature=residual_temperature,
            use_heun=use_heun,
            guidance_scale=guidance_scale,
            return_intermediates=return_bootstrap_mel,
        )

        out.aux["sampling_start_mode"] = "residual_refine"
        out.aux["compat_called_as"] = "sample_with_bootstrap"
        out.aux["bootstrap_t_start_ignored"] = float(bootstrap_t_start)
        out.aux["bootstrap_noise_temperature_as_residual_temperature"] = float(
            residual_temperature
        )

        return out

# ============================================================
# Compatibility wrappers / 兼容旧类名的 wrapper
# ============================================================

class Stage2FlowMatchingAcousticModel(Stage2FlowMatchingAcousticModelV61):
    """
    EN:
    Compatibility wrapper for the old v1 class name.

    ZH:
    旧 v1 类名的兼容 wrapper。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("num_layers", 6)
        kwargs.setdefault("hidden_dim", 512)
        super().__init__(*args, **kwargs)
        self.model_version = "v1"
        self.model_variant = "v1_compat_on_v6_1"


class Stage2FlowMatchingAcousticModelV2(Stage2FlowMatchingAcousticModelV61):
    """
    EN:
    Compatibility wrapper for the old v2/v3.1/v4.0 class name.

    Note:
    This is not a bit-exact reproduction of the old v2 implementation.
    It exists to keep imports and basic training entrypoints working.

    ZH:
    旧 v2 / v3.1 / v4.0 类名的兼容 wrapper。

    注意：
    这不是旧 v2 实现的逐行复刻，只用于保持 import 和基础训练入口可用。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("enable_length_predictor", None)
        kwargs.pop("length_predictor_hidden_dim", None)
        kwargs.pop("length_loss_weight", None)
        kwargs.pop("length_teacher_forcing_prob", None)

        super().__init__(*args, **kwargs)
        self.model_version = "v2"
        self.model_variant = "v2_compat_on_v6_1"


class Stage2FlowMatchingAcousticModelV5(Stage2FlowMatchingAcousticModelV61):
    """
    EN:
    Compatibility wrapper for the old v5 class name.

    ZH:
    旧 v5 类名的兼容 wrapper。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("num_layers", kwargs.get("num_layers", 6))
        kwargs.setdefault("hidden_dim", kwargs.get("hidden_dim", 512))
        super().__init__(*args, **kwargs)
        self.model_version = "v5"
        self.model_variant = "v5_compat_on_v6_1"


class Stage2FlowMatchingAcousticModelV6(Stage2FlowMatchingAcousticModelV61):
    """
    EN:
    Compatibility wrapper for the old v6 class name.

    ZH:
    旧 v6 类名的兼容 wrapper。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_version = "v6"
        self.model_variant = "v6_compat_on_v6_1"