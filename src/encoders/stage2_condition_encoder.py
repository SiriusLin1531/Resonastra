from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.interfaces.stage2_io import Stage2Inputs


def _make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    EN:
    lengths: (B,)
    return: (B, max_len), bool, True for valid positions

    ZH:
    根据长度生成 mask。
    返回形状 (B, max_len)，True 表示有效位置。
    """
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


@dataclass
class Stage2ConditionBundleV6:
    """
    EN:
    Unified v6 / v6.1 condition bundle.

    ZH:
    v6 / v6.1 统一条件打包结构。
    """

    # --------------------------------------------------------
    # Semantic stream / semantic 条件流
    # --------------------------------------------------------
    semantic_cond: Optional[torch.Tensor] = None       # (B, T_sem, H_sem)
    semantic_mask: Optional[torch.Tensor] = None       # (B, T_sem), True for valid
    semantic_lengths: Optional[torch.Tensor] = None    # (B,)

    # --------------------------------------------------------
    # Content stream, native length / 原生长度 content 条件流
    # --------------------------------------------------------
    content_states: Optional[torch.Tensor] = None      # (B, T_content, C_content)
    content_mask: Optional[torch.Tensor] = None        # (B, T_content), True for valid
    content_lengths: Optional[torch.Tensor] = None     # (B,)

    # --------------------------------------------------------
    # Content stream expanded to frame-level
    # 扩展到 acoustic frame-level 的 content 条件流
    # --------------------------------------------------------
    content_frame_cond: Optional[torch.Tensor] = None  # (B, T_frame, C_frame)
    content_frame_mask: Optional[torch.Tensor] = None  # (B, T_frame), True for valid

    # --------------------------------------------------------
    # Style stream / 风格条件流
    # --------------------------------------------------------
    style_global: Optional[torch.Tensor] = None        # (B, C_style)

    # --------------------------------------------------------
    # v6.1 additions / v6.1 新增字段
    # --------------------------------------------------------
    content_frame_main: Optional[torch.Tensor] = None       # (B, T_frame, C_main)
    content_frame_main_mask: Optional[torch.Tensor] = None  # (B, T_frame), True for valid

    semantic_guide_cond: Optional[torch.Tensor] = None      # (B, T_sem, C_sem_guide)
    semantic_guide_mask: Optional[torch.Tensor] = None      # (B, T_sem), True for valid

    length_prior: Optional[torch.Tensor] = None             # optional future hook
    length_ratio_hint: Optional[torch.Tensor] = None        # optional future hook


class Stage2LocalContentRefinerBlock(nn.Module):
    """
    EN:
    Lightweight ConvNeXt-style 1D local refinement block for content states.

    v6.1 additions:
    - dilation support
    - residual scaling

    ZH:
    用于内容序列的轻量 ConvNeXt 风格 1D 局部精炼 block。

    v6.1 新增：
    - 支持 dilation
    - 支持 residual scaling
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        padding = (kernel_size // 2) * dilation

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.residual_scale = float(residual_scale)

        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise_in = nn.Linear(hidden_dim, hidden_dim * 4)
        self.pointwise_out = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, C)
        mask: Optional[torch.Tensor] = None,     # (B, T), True for valid
    ) -> torch.Tensor:
        residual = x
        _, T, _ = x.shape

        h = self.norm(x)
        h = h.transpose(1, 2)                    # (B, C, T)
        h = self.depthwise(h)

        # Defensive crop/pad for even kernel edge cases.
        # 正常 odd kernel 下长度应保持不变；这里做防御性处理。
        if h.shape[-1] != T:
            h = h[..., :T]
            if h.shape[-1] < T:
                h = F.pad(h, (0, T - h.shape[-1]))

        h = h.transpose(1, 2)                    # (B, T, C)

        h = self.pointwise_in(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.pointwise_out(h)
        h = self.dropout(h)

        x = residual + self.residual_scale * h

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


class Stage2LocalContentRefiner(nn.Module):
    """
    EN:
    Stack of local refinement blocks for content representations.

    v6.1 additions:
    - dilation cycle
    - residual scaling

    ZH:
    内容表征的局部精炼器。

    v6.1 新增：
    - dilation cycle
    - residual scaling
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 5,
        dilations: tuple[int, ...] | None = None,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        if dilations is None:
            dilations = (1, 2)

        if len(dilations) <= 0:
            raise ValueError("dilations must contain at least one value.")

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = int(dilations[i % len(dilations)])
            self.blocks.append(
                Stage2LocalContentRefinerBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    residual_scale=residual_scale,
                )
            )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T_content, C_in)
        mask: Optional[torch.Tensor] = None,     # (B, T_content), True for valid
    ) -> torch.Tensor:
        x = self.input_proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.out_norm(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


class Stage2FrameSmoothingBlock(nn.Module):
    """
    EN:
    Lightweight frame-level smoothing block after content expansion.

    ZH:
    内容扩展到 frame-level 之后的轻量平滑 block。
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        padding = kernel_size // 2

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)

        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, C)
        mask: Optional[torch.Tensor] = None,     # (B, T), True for valid
    ) -> torch.Tensor:
        residual = x
        _, T, _ = x.shape

        h = self.norm(x)
        h = h.transpose(1, 2)                    # (B, C, T)
        h = self.depthwise(h)

        if h.shape[-1] != T:
            h = h[..., :T]
            if h.shape[-1] < T:
                h = F.pad(h, (0, T - h.shape[-1]))

        h = F.gelu(h)
        h = self.pointwise(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)                    # (B, T, C)

        x = residual + h

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


class Stage2ContentBoundaryEnhancer(nn.Module):
    """
    EN:
    Lightweight boundary/detail enhancer for frame-level content condition.

    This module is intentionally small. Its goal is not to replace the
    acoustic backbone, but to make frame-level content condition less
    over-smoothed after interpolation.

    ZH:
    frame-level content condition 的轻量边界 / 细节增强模块。

    该模块故意保持轻量。它不是为了替代声学主干，
    而是为了减少 content 插值扩展后的过度平滑。
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        residual_scale: float = 0.3,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.residual_scale = float(residual_scale)

        padding = kernel_size // 2

        self.norm = nn.LayerNorm(hidden_dim)

        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, C)
        mask: Optional[torch.Tensor] = None,     # (B, T), True for valid
    ) -> torch.Tensor:
        residual = x
        _, T, _ = x.shape

        h = self.norm(x)

        conv_h = h.transpose(1, 2)               # (B, C, T)
        conv_h = self.depthwise(conv_h)

        if conv_h.shape[-1] != T:
            conv_h = conv_h[..., :T]
            if conv_h.shape[-1] < T:
                conv_h = F.pad(conv_h, (0, T - conv_h.shape[-1]))

        conv_h = conv_h.transpose(1, 2)          # (B, T, C)

        gate = torch.sigmoid(self.gate_proj(h))
        h = self.out_proj(conv_h * gate)
        h = self.dropout(h)

        x = residual + self.residual_scale * h

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


class Stage2FrameContentExpander(nn.Module):
    """
    EN:
    Expand native content states to frame-level acoustic-length condition.

    v6.1 additions:
    - optional boundary/detail enhancer after smoothing

    ZH:
    将原生内容序列扩展到声学帧长度附近。

    v6.1 新增：
    - 在 smoothing 后加入可选的 boundary/detail enhancer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_boundary_enhancer: bool = True,
        boundary_enhancer_kernel_size: int = 3,
        boundary_residual_scale: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_boundary_enhancer = bool(use_boundary_enhancer)

        self.input_proj = nn.Linear(input_dim, output_dim)

        self.blocks = nn.ModuleList(
            [
                Stage2FrameSmoothingBlock(
                    hidden_dim=output_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_boundary_enhancer:
            self.boundary_enhancer = Stage2ContentBoundaryEnhancer(
                hidden_dim=output_dim,
                kernel_size=boundary_enhancer_kernel_size,
                dropout=dropout,
                residual_scale=boundary_residual_scale,
            )
        else:
            self.boundary_enhancer = None

        self.out_norm = nn.LayerNorm(output_dim)

    def _expand_single(
        self,
        x: torch.Tensor,         # (T_content, C)
        target_len: int,
    ) -> torch.Tensor:
        if x.shape[0] == target_len:
            return x

        if x.shape[0] <= 0:
            raise ValueError("Cannot expand empty content sequence.")

        x_ = x.transpose(0, 1).unsqueeze(0)      # (1, C, T)
        y_ = F.interpolate(
            x_,
            size=target_len,
            mode="linear",
            align_corners=False,
        )
        y = y_.squeeze(0).transpose(0, 1)        # (target_len, C)
        return y

    def forward(
        self,
        content_states: torch.Tensor,            # (B, T_content, C)
        content_lengths: torch.Tensor,           # (B,)
        target_lengths: torch.Tensor,            # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, C = content_states.shape
        device = content_states.device
        dtype = content_states.dtype

        target_lengths = target_lengths.long().to(device).clamp_min(1)
        content_lengths = content_lengths.long().to(device).clamp_min(1)

        max_target_len = int(target_lengths.max().item())
        out = torch.zeros(B, max_target_len, C, device=device, dtype=dtype)

        for i in range(B):
            src_len = max(int(content_lengths[i].item()), 1)
            tgt_len = max(int(target_lengths[i].item()), 1)

            src = content_states[i, :src_len, :]          # (T_src, C)
            expanded = self._expand_single(src, tgt_len)  # (T_tgt, C)
            out[i, :tgt_len, :] = expanded

        out = self.input_proj(out)
        frame_mask = _make_length_mask(target_lengths.long(), max_target_len)

        out = out * frame_mask.unsqueeze(-1).to(dtype=out.dtype)

        for block in self.blocks:
            out = block(out, mask=frame_mask)

        if self.boundary_enhancer is not None:
            out = self.boundary_enhancer(out, mask=frame_mask)

        out = self.out_norm(out)
        out = out * frame_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out, frame_mask


class Stage2ConditionEncoder(nn.Module):
    """
    EN:
    Backward-compatible stage-2 condition encoder.

    This file keeps:
    - old build_frame_condition()
    - old build_conditions_v2_2()
    - old build_style_global_v5()
    - old build_content_stream_v5()
    - old build_conditions_v5()
    - build_conditions_v6()

    And adds:
    - v6.1 local content refinement upgrades
    - v6.1 frame-level content boundary enhancement
    - build_conditions_v6_1()

    ZH:
    向后兼容的第二阶段条件编码器。

    这个文件保留：
    - 旧版 build_frame_condition()
    - 旧版 build_conditions_v2_2()
    - 旧版 build_style_global_v5()
    - 旧版 build_content_stream_v5()
    - 旧版 build_conditions_v5()
    - build_conditions_v6()

    同时新增：
    - v6.1 内容局部精炼增强
    - v6.1 frame-level content 边界增强
    - build_conditions_v6_1()
    """

    def __init__(
        self,
        semantic_vocab_size: int = 1024,
        semantic_embed_dim: int = 256,
        hidden_dim: int = 512,
        bert_dim: int = 1024,

        # ----------------------------------------------------
        # v6 additions / v6 扩展参数
        # ----------------------------------------------------
        phoneme_vocab_size: int = 4096,
        phoneme_embed_dim: int = 256,
        content_dim: int | None = None,
        content_refiner_layers: int = 4,
        content_frame_dim: int | None = None,
        content_expander_layers: int = 2,
        style_dim: int | None = None,
        prompt_vocab_size: int | None = None,
        dropout: float = 0.1,

        # ----------------------------------------------------
        # v6.1 additions / v6.1 扩展参数
        # ----------------------------------------------------
        content_main_dim: int | None = None,
        content_refiner_dilations: tuple[int, ...] | None = None,
        content_boundary_enhance: bool = True,
        content_boundary_kernel_size: int = 3,
        content_boundary_residual_scale: float = 0.3,
        content_refiner_residual_scale: float = 0.5,
        semantic_guide_dim: int | None = None,

        # ----------------------------------------------------
        # v6.4.1 continuous semantic additions
        # v6.4.1 continuous semantic 扩展参数
        # ----------------------------------------------------
        use_continuous_semantic: bool = False,
        continuous_semantic_dim: int = 768,
        continuous_semantic_fusion_mode: str = "gated_add",
        continuous_semantic_gate_init: float = -2.0,
        continuous_semantic_dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.bert_dim = int(bert_dim)
        self.semantic_vocab_size = int(semantic_vocab_size)
        self.semantic_embed_dim = int(semantic_embed_dim)

        self.content_dim = int(content_dim or hidden_dim)
        self.content_frame_dim = int(content_frame_dim or self.content_dim)
        self.content_main_dim = int(content_main_dim or hidden_dim)
        self.semantic_guide_dim = int(semantic_guide_dim or hidden_dim)

        self.style_dim = int(style_dim or hidden_dim)
        self.phoneme_vocab_size = int(phoneme_vocab_size)
        self.phoneme_embed_dim = int(phoneme_embed_dim)

        if prompt_vocab_size is None:
            prompt_vocab_size = semantic_vocab_size
        self.prompt_vocab_size = int(prompt_vocab_size)

        # ----------------------------------------------------
        # Semantic branch / semantic 条件分支
        # ----------------------------------------------------
        self.semantic_embedding = nn.Embedding(
            num_embeddings=self.semantic_vocab_size,
            embedding_dim=semantic_embed_dim,
        )
        self.semantic_proj = nn.Linear(semantic_embed_dim, hidden_dim)
        self.semantic_global_proj = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------
        # v6.4.1 continuous semantic branch
        # v6.4.1 连续 semantic 分支
        # ----------------------------------------------------
        self.use_continuous_semantic = bool(use_continuous_semantic)
        self.continuous_semantic_dim = int(continuous_semantic_dim)
        self.continuous_semantic_fusion_mode = str(continuous_semantic_fusion_mode).lower()
        self.continuous_semantic_dropout_p = float(continuous_semantic_dropout)

        allowed_fusion_modes = {"gated_add", "add", "replace"}
        if self.continuous_semantic_fusion_mode not in allowed_fusion_modes:
            raise ValueError(
                f"continuous_semantic_fusion_mode must be one of {sorted(allowed_fusion_modes)}, "
                f"got {continuous_semantic_fusion_mode!r}"
            )

        if self.continuous_semantic_dim <= 0:
            raise ValueError(
                f"continuous_semantic_dim must be positive, got {continuous_semantic_dim}"
            )

        self.continuous_semantic_norm = nn.LayerNorm(self.continuous_semantic_dim)
        self.continuous_semantic_proj = nn.Linear(
            self.continuous_semantic_dim,
            hidden_dim,
        )

        self.continuous_semantic_dropout = nn.Dropout(self.continuous_semantic_dropout_p)

        # Learnable scalar gate. Use sigmoid(gate) during fusion.
        # 可学习标量门控，融合时使用 sigmoid(gate)。
        self.continuous_semantic_gate = nn.Parameter(
            torch.tensor(float(continuous_semantic_gate_init))
        )

        # Debug holder, updated during forward.
        # 调试用，forward 时更新。
        self._last_continuous_semantic_stats: dict[str, float | int | str] = {
            "used": 0,
            "gate": float(torch.sigmoid(self.continuous_semantic_gate.detach()).item()),
            "mode": self.continuous_semantic_fusion_mode,
        }

        # ----------------------------------------------------
        # Text / BERT branch / 文本 / BERT 条件分支
        # ----------------------------------------------------
        self.bert_proj = nn.Conv1d(
            in_channels=bert_dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.text_global_proj = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------
        # Prompt branch / prompt 条件分支
        # ----------------------------------------------------
        self.prompt_embedding = nn.Embedding(
            num_embeddings=self.prompt_vocab_size,
            embedding_dim=hidden_dim,
        )
        self.prompt_seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prompt_global_proj = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------
        # v5 style branch / v5 风格分支
        # ----------------------------------------------------
        self.style_prompt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ----------------------------------------------------
        # Global fusion / 全局风格融合
        # ----------------------------------------------------
        self.global_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.frame_cond_norm = nn.LayerNorm(hidden_dim)

        # ----------------------------------------------------
        # v6 / v6.1 content branch / v6 / v6.1 内容分支
        # ----------------------------------------------------
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=self.phoneme_vocab_size,
            embedding_dim=phoneme_embed_dim,
        )
        self.phoneme_hidden_proj = nn.Linear(phoneme_embed_dim, hidden_dim)
        self.phoneme_content_proj = nn.Linear(phoneme_embed_dim, self.content_dim)

        self.bert_content_proj = nn.Conv1d(
            in_channels=bert_dim,
            out_channels=self.content_dim,
            kernel_size=1,
        )

        self.semantic_hint_proj = nn.Linear(hidden_dim, self.content_dim)
        self.content_merge_norm = nn.LayerNorm(self.content_dim)

        self.local_content_refiner = Stage2LocalContentRefiner(
            input_dim=self.content_dim,
            hidden_dim=self.content_dim,
            num_layers=content_refiner_layers,
            kernel_size=5,
            dilations=content_refiner_dilations,
            dropout=dropout,
            residual_scale=content_refiner_residual_scale,
        )

        self.frame_content_expander = Stage2FrameContentExpander(
            input_dim=self.content_dim,
            output_dim=self.content_frame_dim,
            num_layers=content_expander_layers,
            kernel_size=5,
            dropout=dropout,
            use_boundary_enhancer=content_boundary_enhance,
            boundary_enhancer_kernel_size=content_boundary_kernel_size,
            boundary_residual_scale=content_boundary_residual_scale,
        )

        # ----------------------------------------------------
        # v6 style output / v6 style 输出
        # ----------------------------------------------------
        self.style_global_out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.style_dim),
        )

        # ----------------------------------------------------
        # v6.1 output projections / v6.1 输出投影
        # ----------------------------------------------------
        self.content_main_proj = nn.Sequential(
            nn.LayerNorm(self.content_frame_dim),
            nn.Linear(self.content_frame_dim, self.content_main_dim),
        )

        self.semantic_guide_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.semantic_guide_dim),
        )

    # --------------------------------------------------------
    # Internal helpers / 内部辅助函数
    # --------------------------------------------------------
    def _normalize_lengths(
        self,
        lengths: torch.Tensor | None,
        batch_size: int,
        full_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        EN:
        Normalize optional lengths into a LongTensor of shape (B,).

        ZH:
        将可选长度统一成形状 (B,) 的 LongTensor。
        """
        full_len = max(int(full_len), 1)

        if lengths is None:
            return torch.full(
                size=(batch_size,),
                fill_value=full_len,
                dtype=torch.long,
                device=device,
            )

        return lengths.long().to(device).clamp(min=1, max=full_len)

    def _masked_mean(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        EN:
        x: (B, T, H)
        lengths: (B,)
        return: (B, H)

        ZH:
        对变长序列做 masked mean pooling。
        """
        _, T, _ = x.shape
        lengths = lengths.long().to(x.device).clamp(min=1, max=max(int(T), 1))

        mask = _make_length_mask(lengths, T).float().unsqueeze(-1)  # (B, T, 1)
        x_sum = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return x_sum / denom

    def _masked_mean_from_mask(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        EN:
        Masked mean pooling using bool mask.

        ZH:
        使用 bool mask 的 masked mean pooling。
        """
        if mask is None:
            return x.mean(dim=1)

        m = mask.float().unsqueeze(-1)
        x_sum = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return x_sum / denom

    def _upsample_seq_length_aware(
        self,
        x: torch.Tensor,                 # (B, T_src, H)
        src_lengths: torch.Tensor,       # (B,)
        target_len: int,
        target_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        EN:
        Length-aware per-sample upsampling.

        For each sample:
        - crop to valid source length
        - interpolate only valid part
        - fill only valid target region
        - padded region remains zero

        Return shape: (B, target_len, H)

        ZH:
        逐样本 length-aware 上采样。

        对每条样本：
        - 先裁到真实源长度
        - 只对真实部分插值
        - 只填充真实目标长度
        - pad 区域保持为 0

        返回形状：(B, target_len, H)
        """
        B, _, H = x.shape
        device = x.device
        target_len = max(int(target_len), 1)

        src_lengths = src_lengths.long().to(device).clamp(min=1, max=max(int(x.shape[1]), 1))

        if target_lengths is None:
            target_lengths = torch.full(
                size=(B,),
                fill_value=target_len,
                dtype=torch.long,
                device=device,
            )
        else:
            target_lengths = target_lengths.long().to(device).clamp(min=1, max=target_len)

        out = torch.zeros(B, target_len, H, dtype=x.dtype, device=device)

        for i in range(B):
            src_len = max(int(src_lengths[i].item()), 1)
            tgt_len = max(int(target_lengths[i].item()), 1)

            src_seq = x[i : i + 1, :src_len, :]          # (1, T_src_i, H)
            src_seq = src_seq.transpose(1, 2)            # (1, H, T_src_i)
            up = F.interpolate(
                src_seq,
                size=tgt_len,
                mode="linear",
                align_corners=False,
            )
            up = up.transpose(1, 2)                      # (1, T_tgt_i, H)
            out[i, :tgt_len, :] = up[0]

        return out

    def _resolve_keep_mask(
        self,
        mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        EN:
        Resolve optional keep mask to shape (B,) float tensor.

        ZH:
        将可选 keep mask 统一成形状 (B,) 的 float 张量。
        """
        if mask is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        return mask.to(device=device, dtype=dtype).view(batch_size)

    def _lengths_from_mask(
        self,
        mask: torch.Tensor | None,
        fallback_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        EN:
        Convert a bool mask (B, T) to lengths (B,).

        ZH:
        将 bool mask (B, T) 转为长度 (B,)。
        """
        if mask is None:
            raise ValueError("_lengths_from_mask expects a non-None mask.")
        return mask.long().sum(dim=1).to(device=device).clamp(
            min=1,
            max=max(int(fallback_len), 1),
        )

    def _match_seq_to_target_length(
        self,
        x: torch.Tensor,                 # (B, T_src, C)
        src_lengths: torch.Tensor,       # (B,)
        target_len: int,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Match a native-length sequence to another target sequence length using
        the same length-aware interpolation helper.

        ZH:
        使用同一个 length-aware 插值辅助函数，把原生长度序列对齐到目标长度。
        """
        target_len = max(int(target_len), 1)
        src_lengths = src_lengths.long().to(x.device)
        target_lengths = target_lengths.long().to(x.device)

        if (
            x.shape[1] == target_len
            and src_lengths.shape == target_lengths.shape
            and torch.equal(src_lengths.long(), target_lengths.long())
        ):
            return x

        return self._upsample_seq_length_aware(
            x=x,
            src_lengths=src_lengths,
            target_len=target_len,
            target_lengths=target_lengths,
        )

    def _safe_token_ids(
        self,
        ids: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        EN:
        Clamp token ids into valid embedding range.

        ZH:
        将 token id clamp 到合法 embedding 范围。
        """
        return ids.long().clamp(min=0, max=max(int(vocab_size) - 1, 0))

    def _pad_or_trim_seq(
        self,
        x: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """
        EN:
        Pad or trim a sequence tensor to target_len.

        Args:
            x: (B, T, C)

        ZH:
        将序列张量 pad 或 trim 到 target_len。
        """
        B, T, C = x.shape
        target_len = max(int(target_len), 1)

        if T == target_len:
            return x

        if T > target_len:
            return x[:, :target_len, :]

        pad = torch.zeros(
            B,
            target_len - T,
            C,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=1)

    def _pad_or_trim_mask(
        self,
        mask: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """
        EN:
        Pad or trim a bool mask to target_len.

        Args:
            mask: (B, T)

        ZH:
        将 bool mask pad 或 trim 到 target_len。
        """
        B, T = mask.shape
        target_len = max(int(target_len), 1)

        if T == target_len:
            return mask

        if T > target_len:
            return mask[:, :target_len]

        pad = torch.zeros(
            B,
            target_len - T,
            device=mask.device,
            dtype=torch.bool,
        )
        return torch.cat([mask, pad], dim=1)

    def _resolve_target_lengths(
        self,
        target_lengths: torch.Tensor | None,
        batch_size: int,
        target_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        EN:
        Resolve target lengths for condition building.

        ZH:
        为条件构建解析 target lengths。
        """
        target_len = max(int(target_len), 1)

        if target_lengths is None:
            return torch.full(
                size=(batch_size,),
                fill_value=target_len,
                dtype=torch.long,
                device=device,
            )

        return target_lengths.long().to(device).clamp(
            min=1,
            max=target_len,
        )

    # --------------------------------------------------------
    # Legacy condition branch combiner / 旧版条件分支融合
    # --------------------------------------------------------
    def combine_condition_branches(
        self,
        *,
        semantic_frame_cond: torch.Tensor,
        text_frame_cond: torch.Tensor | None,
        prompt_frame_cond: torch.Tensor | None,
        semantic_global: torch.Tensor,
        text_global: torch.Tensor,
        prompt_global: torch.Tensor,
        semantic_keep_mask: torch.Tensor | None = None,
        text_keep_mask: torch.Tensor | None = None,
        prompt_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        EN:
        Combine branch-level conditions into:
        - frame_cond
        - global_style

        This is where conditional dropout / guidance masks are applied.

        ZH:
        将各分支条件组合成：
        - frame_cond
        - global_style

        条件 dropout / guidance mask 会在这里生效。
        """
        B, _, _ = semantic_frame_cond.shape
        device = semantic_frame_cond.device
        dtype = semantic_frame_cond.dtype

        sem_keep = self._resolve_keep_mask(semantic_keep_mask, B, device, dtype)
        txt_keep = self._resolve_keep_mask(text_keep_mask, B, device, dtype)
        prm_keep = self._resolve_keep_mask(prompt_keep_mask, B, device, dtype)

        frame_cond = semantic_frame_cond * sem_keep[:, None, None]

        if text_frame_cond is not None:
            frame_cond = frame_cond + text_frame_cond * txt_keep[:, None, None]

        if prompt_frame_cond is not None:
            frame_cond = frame_cond + prompt_frame_cond * prm_keep[:, None, None]

        semantic_global = semantic_global * sem_keep[:, None]
        text_global = text_global * txt_keep[:, None]
        prompt_global = prompt_global * prm_keep[:, None]

        global_style = self.global_fuse(
            torch.cat([semantic_global, text_global, prompt_global], dim=-1)
        )  # (B, H)

        frame_cond = frame_cond + global_style.unsqueeze(1)
        frame_cond = self.frame_cond_norm(frame_cond)

        return {
            "frame_cond": frame_cond,
            "global_style": global_style,
        }

    # --------------------------------------------------------
    # Backward-compatible old API / 向后兼容旧接口
    # --------------------------------------------------------
    def build_frame_condition(
        self,
        batch: Stage2Inputs,
        target_len: int,
    ) -> dict[str, torch.Tensor | None]:
        """
        EN:
        Backward-compatible API.
        Internally it calls build_conditions_v2_2() and returns
        the collapsed frame condition plus one global vector.

        ZH:
        向后兼容旧接口。
        内部调用 build_conditions_v2_2()，
        返回压缩后的 frame condition 和一个全局条件向量。
        """
        target_lengths = batch.target_lengths if batch.target_lengths is not None else None

        cond = self.build_conditions_v2_2(
            batch=batch,
            target_len=target_len,
            target_lengths=target_lengths,
        )

        combined = self.combine_condition_branches(
            semantic_frame_cond=cond["semantic_frame_cond"],
            text_frame_cond=cond["text_frame_cond"],
            prompt_frame_cond=cond["prompt_frame_cond"],
            semantic_global=cond["semantic_global"],
            text_global=cond["text_global"],
            prompt_global=cond["prompt_global"],
        )

        return {
            "frame_cond": combined["frame_cond"],
            "prompt_global": combined["global_style"],
        }

    # --------------------------------------------------------
    # v2.2 API / v2.2 条件接口
    # --------------------------------------------------------
    def build_conditions_v2_2(
        self,
        batch: Stage2Inputs,
        target_len: int,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        EN:
        Build richer v2.2 conditions for the acoustic model.

        ZH:
        为声学模型构建更丰富的 v2.2 条件。
        """
        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        target_lengths = self._resolve_target_lengths(
            target_lengths=target_lengths,
            batch_size=B,
            target_len=target_len,
            device=device,
        )

        # ---------------------------
        # Semantic branch
        # semantic 分支
        # ---------------------------
        semantic_lengths = self._normalize_lengths(
            batch.semantic_lengths,
            batch_size=B,
            full_len=batch.semantic_tokens.shape[1],
            device=device,
        )

        semantic_ids = self._safe_token_ids(
            batch.semantic_tokens,
            vocab_size=self.semantic_vocab_size,
        )

        semantic_hidden = self.semantic_embedding(semantic_ids)
        semantic_hidden = self.semantic_proj(semantic_hidden)

        semantic_frame_cond = self._upsample_seq_length_aware(
            semantic_hidden,
            src_lengths=semantic_lengths,
            target_len=target_len,
            target_lengths=target_lengths,
        )

        semantic_global = self.semantic_global_proj(
            self._masked_mean(semantic_hidden, semantic_lengths)
        )

        # ---------------------------
        # Text / BERT branch
        # 文本 / BERT 分支
        # ---------------------------
        text_hidden = None
        text_lengths = None
        text_frame_cond = None
        text_global = torch.zeros(B, self.hidden_dim, device=device)

        if batch.bert_feature is not None:
            text_hidden = self.bert_proj(batch.bert_feature.float().to(device))
            text_hidden = text_hidden.transpose(1, 2)

            text_lengths = self._normalize_lengths(
                batch.phoneme_lens,
                batch_size=B,
                full_len=text_hidden.shape[1],
                device=device,
            )

            text_frame_cond = self._upsample_seq_length_aware(
                text_hidden,
                src_lengths=text_lengths,
                target_len=target_len,
                target_lengths=target_lengths,
            )

            text_global = self.text_global_proj(
                self._masked_mean(text_hidden, text_lengths)
            )

        # ---------------------------
        # Prompt branch
        # prompt 分支
        # ---------------------------
        prompt_hidden = None
        prompt_lengths = None
        prompt_frame_cond = None
        prompt_global = torch.zeros(B, self.hidden_dim, device=device)

        if batch.prompt_tokens is not None:
            prompt_ids = self._safe_token_ids(
                batch.prompt_tokens.to(device),
                vocab_size=self.prompt_vocab_size,
            )

            prompt_hidden = self.prompt_embedding(prompt_ids)
            prompt_hidden = self.prompt_seq_proj(prompt_hidden)

            prompt_lengths = self._normalize_lengths(
                batch.prompt_lengths,
                batch_size=B,
                full_len=prompt_hidden.shape[1],
                device=device,
            )

            prompt_frame_cond = self._upsample_seq_length_aware(
                prompt_hidden,
                src_lengths=prompt_lengths,
                target_len=target_len,
                target_lengths=target_lengths,
            )

            prompt_global = self.prompt_global_proj(
                self._masked_mean(prompt_hidden, prompt_lengths)
            )

        return {
            "semantic_seq_cond_native": semantic_hidden,
            "semantic_seq_lengths": semantic_lengths,
            "text_seq_cond_native": text_hidden,
            "text_seq_lengths": text_lengths,
            "prompt_seq_cond_native": prompt_hidden,
            "prompt_seq_lengths": prompt_lengths,
            "semantic_frame_cond": semantic_frame_cond,
            "text_frame_cond": text_frame_cond,
            "prompt_frame_cond": prompt_frame_cond,
            "semantic_global": semantic_global,
            "text_global": text_global,
            "prompt_global": prompt_global,
        }

    # --------------------------------------------------------
    # v5.0 dual-stream API / v5.0 双流条件接口
    # --------------------------------------------------------
    def build_style_global_v5(
        self,
        prompt_tokens: torch.Tensor | None,
        prompt_lengths: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        EN:
        Build utterance-level style embedding from prompt tokens.

        ZH:
        从 prompt token 构建句级 style embedding。
        """
        if prompt_tokens is None:
            return None

        B = prompt_tokens.shape[0]
        device = prompt_tokens.device

        prompt_lengths = self._normalize_lengths(
            prompt_lengths,
            batch_size=B,
            full_len=prompt_tokens.shape[1],
            device=device,
        )

        prompt_ids = self._safe_token_ids(
            prompt_tokens,
            vocab_size=self.prompt_vocab_size,
        )

        prompt_hidden = self.prompt_embedding(prompt_ids)
        prompt_hidden = self.style_prompt_proj(prompt_hidden)
        style_global = self.style_mlp(self._masked_mean(prompt_hidden, prompt_lengths))
        return style_global

    def build_content_stream_v5(
        self,
        batch: Stage2Inputs,
    ) -> dict[str, torch.Tensor | None]:
        """
        EN:
        Build content-side native-length streams for v5.

        ZH:
        为 v5 构建内容侧原生长度条件序列。
        """
        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        semantic_lengths = self._normalize_lengths(
            batch.semantic_lengths,
            batch_size=B,
            full_len=batch.semantic_tokens.shape[1],
            device=device,
        )

        semantic_ids = self._safe_token_ids(
            batch.semantic_tokens,
            vocab_size=self.semantic_vocab_size,
        )

        semantic_hidden = self.semantic_embedding(semantic_ids)
        semantic_hidden = self.semantic_proj(semantic_hidden)
        semantic_global = self.semantic_global_proj(
            self._masked_mean(semantic_hidden, semantic_lengths)
        )

        text_hidden = None
        text_lengths = None
        text_global = None

        if batch.bert_feature is not None:
            text_hidden = self.bert_proj(batch.bert_feature.float().to(device))
            text_hidden = text_hidden.transpose(1, 2)

            text_lengths = self._normalize_lengths(
                batch.phoneme_lens,
                batch_size=B,
                full_len=text_hidden.shape[1],
                device=device,
            )

            text_global = self.text_global_proj(
                self._masked_mean(text_hidden, text_lengths)
            )

        return {
            "semantic_seq": semantic_hidden,
            "semantic_lengths": semantic_lengths,
            "text_seq": text_hidden,
            "text_lengths": text_lengths,
            "semantic_global": semantic_global,
            "text_global": text_global,
        }

    def build_conditions_v5(
        self,
        batch: Stage2Inputs,
    ) -> dict[str, torch.Tensor | None]:
        """
        EN:
        Unified v5 condition builder.

        Returns:
        - semantic_seq / semantic_lengths
        - text_seq / text_lengths
        - style_global
        - semantic_global / text_global / prompt_global for compatibility / debug

        ZH:
        v5 统一条件构建接口。
        """
        content = self.build_content_stream_v5(batch)

        style_global = self.build_style_global_v5(
            prompt_tokens=batch.prompt_tokens,
            prompt_lengths=batch.prompt_lengths,
        )

        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        if style_global is None:
            prompt_global = torch.zeros(B, self.hidden_dim, device=device)
        else:
            prompt_global = style_global

        return {
            **content,
            "style_global": style_global,
            "prompt_global": prompt_global,
        }

    # --------------------------------------------------------
    # v6 / v6.1 helper methods / v6 / v6.1 辅助函数
    # --------------------------------------------------------
    def _build_semantic_stream_v61(
        self,
        batch: Stage2Inputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EN:
        Build semantic native stream.

        Returns:
            semantic_cond:    (B, T_sem, H)
            semantic_mask:    (B, T_sem)
            semantic_lengths: (B,)
            semantic_global:  (B, H)

        ZH:
        构建 semantic 原生条件流。
        """
        device = batch.semantic_tokens.device
        B, T_sem = batch.semantic_tokens.shape

        semantic_lengths = self._normalize_lengths(
            batch.semantic_lengths,
            batch_size=B,
            full_len=T_sem,
            device=device,
        )

        semantic_ids = self._safe_token_ids(
            batch.semantic_tokens,
            vocab_size=self.semantic_vocab_size,
        )

        semantic_emb = self.semantic_embedding(semantic_ids)
        semantic_cond = self.semantic_proj(semantic_emb)

        # v6.4.1:
        # Optionally fuse GPT-SoVITS quantizer.decode continuous semantic.
        # 可选融合 GPT-SoVITS quantizer.decode 得到的 continuous semantic。
        semantic_cond = self._maybe_fuse_continuous_semantic_v641(
            batch=batch,
            semantic_cond=semantic_cond,
            semantic_lengths=semantic_lengths,
        )

        semantic_mask = _make_length_mask(
            semantic_lengths,
            semantic_cond.shape[1],
        )

        semantic_cond = semantic_cond * semantic_mask.unsqueeze(-1).to(dtype=semantic_cond.dtype)

        semantic_global = self._masked_mean(
            semantic_cond,
            semantic_lengths,
        )
        semantic_global = self.semantic_global_proj(semantic_global)

        return semantic_cond, semantic_mask, semantic_lengths, semantic_global

    def _build_text_global_v61(
        self,
        batch: Stage2Inputs,
        batch_size: int,
        device: torch.device,
        semantic_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Build text global vector from BERT feature if available.

        ZH:
        如果存在 BERT feature，则构建 text global vector。
        """
        if batch.bert_feature is None:
            return torch.zeros_like(semantic_global)

        bert = batch.bert_feature.float().to(device)
        text_seq = self.bert_proj(bert).transpose(1, 2)

        text_lengths = self._normalize_lengths(
            batch.phoneme_lens,
            batch_size=batch_size,
            full_len=text_seq.shape[1],
            device=device,
        )

        text_global = self._masked_mean(text_seq, text_lengths)
        text_global = self.text_global_proj(text_global)
        return text_global

    def _build_prompt_global_v61(
        self,
        batch: Stage2Inputs,
        batch_size: int,
        device: torch.device,
        semantic_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Build prompt global vector if prompt tokens exist.

        ZH:
        如果存在 prompt tokens，则构建 prompt global vector。
        """
        if batch.prompt_tokens is None:
            return torch.zeros_like(semantic_global)

        prompt_ids = self._safe_token_ids(
            batch.prompt_tokens.to(device),
            vocab_size=self.prompt_vocab_size,
        )

        prompt_emb = self.prompt_embedding(prompt_ids)
        prompt_seq = self.prompt_seq_proj(prompt_emb)

        prompt_lengths = self._normalize_lengths(
            batch.prompt_lengths,
            batch_size=batch_size,
            full_len=prompt_seq.shape[1],
            device=device,
        )

        prompt_global = self._masked_mean(prompt_seq, prompt_lengths)
        prompt_global = self.prompt_global_proj(prompt_global)
        return prompt_global

    def _match_sequence_to_content_length_v61(
        self,
        x: torch.Tensor,
        src_lengths: torch.Tensor,
        target_len: int,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Match any native sequence to content length with length-aware interpolation.

        ZH:
        使用 length-aware 插值将任意原生序列对齐到 content length。
        """
        return self._match_seq_to_target_length(
            x=x,
            src_lengths=src_lengths,
            target_len=target_len,
            target_lengths=target_lengths,
        )

    def _maybe_fuse_continuous_semantic_v641(
            self,
            *,
            batch: Stage2Inputs,
            semantic_cond: torch.Tensor,
            semantic_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Optionally fuse GPT-SoVITS quantizer.decode continuous semantic vectors
        into the token-embedding semantic stream.

        Input:
            semantic_cond:
                (B, T_sem, H), from token embedding + semantic_proj.

            batch.semantic_continuous:
                (B, T_sem_cont, 768), selected source continuous semantic.

        Output:
            fused semantic_cond:
                (B, T_sem, H)

        ZH:
        可选地将 GPT-SoVITS quantizer.decode 得到的 continuous semantic
        融入 token embedding semantic stream。

        该函数是 v6.4.1 的核心改造点。
        """
        self._last_continuous_semantic_stats = {
            "used": 0,
            "gate": float(torch.sigmoid(self.continuous_semantic_gate.detach()).item()),
            "mode": self.continuous_semantic_fusion_mode,
        }

        if not self.use_continuous_semantic:
            return semantic_cond

        if batch.semantic_continuous is None:
            self._last_continuous_semantic_stats.update(
                {
                    "reason": "missing_batch_semantic_continuous",
                }
            )
            return semantic_cond

        device = semantic_cond.device
        dtype = semantic_cond.dtype
        B, T_sem, _ = semantic_cond.shape

        continuous = batch.semantic_continuous.to(device=device).float()

        if continuous.ndim != 3:
            raise ValueError(
                f"batch.semantic_continuous must have shape (B,T,C), "
                f"got {tuple(continuous.shape)}"
            )

        if continuous.shape[0] != B:
            raise ValueError(
                f"continuous batch size mismatch: continuous={continuous.shape[0]}, "
                f"semantic_cond={B}"
            )

        if continuous.shape[-1] != self.continuous_semantic_dim:
            raise ValueError(
                f"continuous semantic dim mismatch: expected {self.continuous_semantic_dim}, "
                f"got {continuous.shape[-1]}"
            )

        continuous_lengths = self._normalize_lengths(
            batch.semantic_continuous_lengths,
            batch_size=B,
            full_len=continuous.shape[1],
            device=device,
        )

        # If continuous length differs from token semantic length, align it.
        # 如果 continuous 序列长度和 token semantic 序列长度不一致，则对齐。
        if continuous.shape[1] != T_sem or not torch.equal(
                continuous_lengths.long(),
                semantic_lengths.long(),
        ):
            continuous = self._match_seq_to_target_length(
                x=continuous.to(dtype=dtype),
                src_lengths=continuous_lengths,
                target_len=T_sem,
                target_lengths=semantic_lengths,
            )
            continuous_lengths = semantic_lengths
        else:
            continuous = continuous.to(dtype=dtype)

        continuous_hidden = self.continuous_semantic_norm(continuous.float())
        continuous_hidden = self.continuous_semantic_proj(continuous_hidden)
        continuous_hidden = continuous_hidden.to(dtype=dtype)
        continuous_hidden = self.continuous_semantic_dropout(continuous_hidden)

        if self.continuous_semantic_fusion_mode == "replace":
            fused = continuous_hidden
            gate_value = 1.0

        elif self.continuous_semantic_fusion_mode == "add":
            fused = semantic_cond + continuous_hidden
            gate_value = 1.0

        elif self.continuous_semantic_fusion_mode == "gated_add":
            gate = torch.sigmoid(self.continuous_semantic_gate).to(dtype=dtype)
            fused = semantic_cond + gate * continuous_hidden
            gate_value = float(gate.detach().cpu().item())

        else:
            raise ValueError(
                f"Unsupported continuous_semantic_fusion_mode: "
                f"{self.continuous_semantic_fusion_mode}"
            )

        semantic_mask = _make_length_mask(
            semantic_lengths,
            T_sem,
        ).to(device=device)

        fused = fused * semantic_mask.unsqueeze(-1).to(dtype=fused.dtype)

        self._last_continuous_semantic_stats = {
            "used": 1,
            "gate": float(gate_value),
            "mode": self.continuous_semantic_fusion_mode,
            "continuous_len_mean": float(continuous_lengths.float().mean().detach().cpu().item()),
            "continuous_norm_mean": float(
                torch.linalg.norm(continuous_hidden.detach().float(), dim=-1).mean().cpu().item()
            ),
        }

        return fused

    def _build_content_stream_v61(
        self,
        batch: Stage2Inputs,
        semantic_cond: torch.Tensor,
        semantic_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EN:
        Build v6.1 native content stream from:
        - phoneme ids
        - BERT feature
        - semantic hint

        Returns:
            content_states:  (B, T_content, C_content)
            content_mask:    (B, T_content)
            content_lengths: (B,)

        ZH:
        从 phoneme ids、BERT feature 和 semantic hint 构建 v6.1 原生 content stream。
        """
        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        content_parts: list[torch.Tensor] = []

        # ----------------------------------------------------
        # 1. Base content length from phoneme if available
        # 1. 优先使用 phoneme 作为 content 原生长度
        # ----------------------------------------------------
        if batch.phoneme_ids is not None:
            phoneme_ids = self._safe_token_ids(
                batch.phoneme_ids.to(device),
                vocab_size=self.phoneme_vocab_size,
            )

            phoneme_emb = self.phoneme_embedding(phoneme_ids)
            phoneme_content = self.phoneme_content_proj(phoneme_emb)

            content_lengths = self._normalize_lengths(
                batch.phoneme_lens,
                batch_size=B,
                full_len=phoneme_content.shape[1],
                device=device,
            )

            content_len = phoneme_content.shape[1]
            content_parts.append(phoneme_content)

        # ----------------------------------------------------
        # 2. If no phoneme, use BERT length
        # 2. 如果没有 phoneme，则使用 BERT 长度
        # ----------------------------------------------------
        elif batch.bert_feature is not None:
            bert = batch.bert_feature.float().to(device)
            bert_content = self.bert_content_proj(bert).transpose(1, 2)

            content_lengths = self._normalize_lengths(
                batch.phoneme_lens,
                batch_size=B,
                full_len=bert_content.shape[1],
                device=device,
            )

            content_len = bert_content.shape[1]
            content_parts.append(bert_content)

        # ----------------------------------------------------
        # 3. Final fallback: semantic length
        # 3. 最后兜底：使用 semantic 长度
        # ----------------------------------------------------
        else:
            semantic_hint = self.semantic_hint_proj(semantic_cond)

            content_lengths = semantic_lengths
            content_len = semantic_hint.shape[1]
            content_parts.append(semantic_hint)

        # ----------------------------------------------------
        # 4. Add BERT content if available and align to content length
        # 4. 如果存在 BERT，将其对齐到 content length 后加入
        # ----------------------------------------------------
        if batch.bert_feature is not None:
            bert = batch.bert_feature.float().to(device)
            bert_content = self.bert_content_proj(bert).transpose(1, 2)

            bert_lengths = self._normalize_lengths(
                batch.phoneme_lens,
                batch_size=B,
                full_len=bert_content.shape[1],
                device=device,
            )

            bert_content = self._match_sequence_to_content_length_v61(
                x=bert_content,
                src_lengths=bert_lengths,
                target_len=content_len,
                target_lengths=content_lengths,
            )
            content_parts.append(bert_content)

        # ----------------------------------------------------
        # 5. Add semantic hint aligned to content length
        # 5. 将 semantic hint 对齐到 content length 后加入
        # ----------------------------------------------------
        semantic_hint = self.semantic_hint_proj(semantic_cond)

        semantic_hint = self._match_sequence_to_content_length_v61(
            x=semantic_hint,
            src_lengths=semantic_lengths,
            target_len=content_len,
            target_lengths=content_lengths,
        )
        content_parts.append(semantic_hint)

        # ----------------------------------------------------
        # 6. Merge and refine
        # 6. 融合并局部精炼
        # ----------------------------------------------------
        content_raw = torch.stack(content_parts, dim=0).mean(dim=0)
        content_raw = self.content_merge_norm(content_raw)

        content_mask = _make_length_mask(
            content_lengths,
            content_raw.shape[1],
        )

        content_raw = content_raw * content_mask.unsqueeze(-1).to(dtype=content_raw.dtype)

        content_states = self.local_content_refiner(
            content_raw,
            mask=content_mask,
        )

        return content_states, content_mask, content_lengths

    def _build_style_global_v61(
        self,
        *,
        semantic_global: torch.Tensor,
        text_global: torch.Tensor,
        prompt_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        EN:
        Build v6.1 global style vector.

        ZH:
        构建 v6.1 全局 style vector。
        """
        fused = self.global_fuse(
            torch.cat([semantic_global, text_global, prompt_global], dim=-1)
        )
        style_global = self.style_global_out_proj(fused)
        return style_global

    # --------------------------------------------------------
    # v6.0 condition API / v6.0 条件构建接口
    # --------------------------------------------------------
    def build_conditions_v6(
        self,
        batch: Stage2Inputs,
        target_len: int,
        target_lengths: torch.Tensor | None = None,
    ) -> Stage2ConditionBundleV6:
        """
        EN:
        Backward-compatible v6 condition builder.

        For compatibility, this now reuses the v6.1 builder and returns a
        superset bundle. Existing v6 code can still access:
        - semantic_cond
        - semantic_mask
        - semantic_lengths
        - content_states
        - content_mask
        - content_lengths
        - content_frame_cond
        - content_frame_mask
        - style_global

        ZH:
        向后兼容的 v6 条件构建接口。

        为了兼容，它现在复用 v6.1 builder，并返回一个超集 bundle。
        旧 v6 代码仍然可以访问原有字段。
        """
        return self.build_conditions_v6_1(
            batch=batch,
            target_len=target_len,
            target_lengths=target_lengths,
        )

    # --------------------------------------------------------
    # v6.1 condition API / v6.1 条件构建接口
    # --------------------------------------------------------
    def build_conditions_v6_1(
        self,
        batch: Stage2Inputs,
        target_len: int,
        target_lengths: torch.Tensor | None = None,
    ) -> Stage2ConditionBundleV6:
        """
        EN:
        Build separated v6.1 condition bundle.

        v6.1 condition design:
        - content_frame_main:
            main frame-level content condition for direct content injection
        - semantic_guide_cond:
            auxiliary semantic guide for coarse alignment / sequence guidance
        - style_global:
            utterance-level style condition

        ZH:
        构建 v6.1 分离式条件包。

        v6.1 条件设计：
        - content_frame_main:
            用于直接内容注入的 frame-level 内容主条件
        - semantic_guide_cond:
            用于粗粒度对齐 / 序列引导的 semantic 辅助条件
        - style_global:
            句级风格条件
        """
        device = batch.semantic_tokens.device
        B = batch.semantic_tokens.shape[0]

        target_lengths = self._resolve_target_lengths(
            target_lengths=target_lengths,
            batch_size=B,
            target_len=target_len,
            device=device,
        )

        # ----------------------------------------------------
        # 1. Semantic stream
        # 1. semantic 条件流
        # ----------------------------------------------------
        (
            semantic_cond,
            semantic_mask,
            semantic_lengths,
            semantic_global,
        ) = self._build_semantic_stream_v61(batch)

        # ----------------------------------------------------
        # 2. Text / prompt global
        # 2. text / prompt 全局条件
        # ----------------------------------------------------
        text_global = self._build_text_global_v61(
            batch=batch,
            batch_size=B,
            device=device,
            semantic_global=semantic_global,
        )

        prompt_global = self._build_prompt_global_v61(
            batch=batch,
            batch_size=B,
            device=device,
            semantic_global=semantic_global,
        )

        # ----------------------------------------------------
        # 3. Native content stream
        # 3. 原生长度 content 条件流
        # ----------------------------------------------------
        (
            content_states,
            content_mask,
            content_lengths,
        ) = self._build_content_stream_v61(
            batch=batch,
            semantic_cond=semantic_cond,
            semantic_lengths=semantic_lengths,
        )

        # ----------------------------------------------------
        # 4. Expand content stream to acoustic frame level
        # 4. 将 content 条件扩展到 acoustic frame level
        # ----------------------------------------------------
        content_frame_cond, content_frame_mask = self.frame_content_expander(
            content_states=content_states,
            content_lengths=content_lengths,
            target_lengths=target_lengths,
        )

        # Defensive alignment to target_len.
        content_frame_cond = self._pad_or_trim_seq(
            content_frame_cond,
            target_len=target_len,
        )
        content_frame_mask = self._pad_or_trim_mask(
            content_frame_mask,
            target_len=target_len,
        )

        # ----------------------------------------------------
        # 5. v6.1 main content frame condition
        # 5. v6.1 frame-level 内容主条件
        # ----------------------------------------------------
        content_frame_main = self.content_main_proj(content_frame_cond)
        content_frame_main = content_frame_main * content_frame_mask.unsqueeze(-1).to(
            dtype=content_frame_main.dtype
        )

        # ----------------------------------------------------
        # 6. v6.1 semantic guide
        # 6. v6.1 semantic guide
        # ----------------------------------------------------
        semantic_guide_cond = self.semantic_guide_proj(semantic_cond)
        semantic_guide_cond = semantic_guide_cond * semantic_mask.unsqueeze(-1).to(
            dtype=semantic_guide_cond.dtype
        )

        # ----------------------------------------------------
        # 7. Style global
        # 7. 全局风格条件
        # ----------------------------------------------------
        style_global = self._build_style_global_v61(
            semantic_global=semantic_global,
            text_global=text_global,
            prompt_global=prompt_global,
        )

        # ----------------------------------------------------
        # 8. Optional length hooks from extras
        # 8. 从 extras 中读取可选 length hook
        # ----------------------------------------------------
        length_prior = None
        length_ratio_hint = None

        if isinstance(batch.extras, dict):
            maybe_prior = batch.extras.get("length_prior", None)
            maybe_ratio = batch.extras.get("length_ratio_hint", None)

            if torch.is_tensor(maybe_prior):
                length_prior = maybe_prior.to(device=device)
            if torch.is_tensor(maybe_ratio):
                length_ratio_hint = maybe_ratio.to(device=device)

        return Stage2ConditionBundleV6(
            semantic_cond=semantic_cond,
            semantic_mask=semantic_mask,
            semantic_lengths=semantic_lengths,

            content_states=content_states,
            content_mask=content_mask,
            content_lengths=content_lengths,

            content_frame_cond=content_frame_cond,
            content_frame_mask=content_frame_mask,

            style_global=style_global,

            content_frame_main=content_frame_main,
            content_frame_main_mask=content_frame_mask,

            semantic_guide_cond=semantic_guide_cond,
            semantic_guide_mask=semantic_mask,

            length_prior=length_prior,
            length_ratio_hint=length_ratio_hint,
        )