from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import math
from dataclasses import dataclass
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helper functions
# 辅助函数
# ============================================================
def make_length_mask(
    lengths: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    """
    Build a boolean valid-frame mask from lengths.

    Args:
        lengths:
            Tensor with shape [B].
        max_len:
            Optional max time length. If None, use lengths.max().

    Returns:
        mask:
            Boolean tensor with shape [B, T].
            True means valid frame.
    """
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1D, got shape={tuple(lengths.shape)}")

    if max_len is None:
        max_len = int(lengths.max().item())

    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.long().unsqueeze(1)


def make_key_padding_mask(
    lengths: torch.Tensor | None,
    max_len: int,
) -> torch.Tensor | None:
    """
    Build key_padding_mask for nn.MultiheadAttention.

    PyTorch MultiheadAttention uses:
        True  = ignore / padding
        False = keep / valid

    Args:
        lengths:
            Optional [B].
        max_len:
            Sequence length.

    Returns:
        Optional bool mask with shape [B, T].
    """
    if lengths is None:
        return None

    valid = make_length_mask(lengths, max_len=max_len)
    return ~valid


def apply_frame_mask(
    x: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    """
    Mask padded frames to zero.

    Args:
        x:
            Tensor with shape [B, T, C].
        lengths:
            Optional tensor with shape [B].

    Returns:
        Masked tensor with shape [B, T, C].
    """
    if lengths is None:
        return x

    if x.ndim != 3:
        raise ValueError(f"x must be 3D [B, T, C], got shape={tuple(x.shape)}")

    mask = make_length_mask(lengths, max_len=x.shape[1])
    return x * mask.unsqueeze(-1).to(dtype=x.dtype)


def sinusoidal_position_embedding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build sinusoidal position embedding.

    Returns:
        pos_emb:
            [length, dim]
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)

    half_dim = (dim + 1) // 2
    div_term = torch.exp(
        torch.arange(half_dim, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    )

    angles = position * div_term.unsqueeze(0)

    emb = torch.zeros(length, dim, device=device, dtype=dtype)
    emb[:, 0::2] = torch.sin(angles[:, : emb[:, 0::2].shape[1]])
    emb[:, 1::2] = torch.cos(angles[:, : emb[:, 1::2].shape[1]])

    return emb


def resize_time_to(
    x: torch.Tensor,
    target_len: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Resize a [B, T, C] tensor to target temporal length.

    Args:
        x:
            [B, T, C]
        target_len:
            target T
        mode:
            interpolation mode

    Returns:
        resized:
            [B, target_len, C]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, C], got shape={tuple(x.shape)}")

    if x.shape[1] == int(target_len):
        return x

    y = x.transpose(1, 2)
    y = F.interpolate(
        y,
        size=int(target_len),
        mode=mode,
        align_corners=False if mode in {"linear", "bilinear", "bicubic"} else None,
    )
    return y.transpose(1, 2)


# ============================================================
# v6.2 Legacy Config
# v6.2 旧版配置
# ============================================================
@dataclass
class CoarseMelBootstrapperConfig:
    """
    Configuration for v6.2 condition-to-coarse-mel bootstrapper.

    This module predicts a weak acoustic scaffold from frame-level conditions.
    It is designed for the v6.2 cold-start-aware Flow Matching Stage2.

    Main idea:
        condition -> coarse_mel_pred
        x_start = (1 - t_start) * noise + t_start * coarse_mel_pred
        sample from t_start to 1.0

    The output is still raw log-mel, with shape [B, T_acoustic, n_mels].
    """

    # Input dimensions
    input_dim: int = 512
    semantic_dim: int | None = None
    style_dim: int | None = None

    # Hidden/output dimensions
    hidden_dim: int = 512
    n_mels: int = 80

    # Network structure
    num_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.1
    expansion_factor: int = 4

    # Condition usage
    use_semantic_guide: bool = True
    use_style_film: bool = True
    use_style_residual: bool = True

    # Initialization
    mel_bias_init: float = -5.0
    residual_scale_init: float = 0.1
    semantic_gate_init: float = 0.5
    style_residual_scale_init: float = 0.1

    # Regularization / stability
    use_final_layer_norm: bool = True
    clamp_output: bool = False
    output_min: float = -12.0
    output_max: float = 4.0


# ============================================================
# v6.3 Conformer Bootstrapper Config
# v6.3 Conformer Bootstrapper 配置
# ============================================================
@dataclass
class ConformerMelBootstrapperConfig:
    """
    Configuration for v6.3 Conformer-based mel bootstrapper.

    Purpose:
        Replace the lightweight v6.2 Conv1D bootstrapper with a stronger
        condition-to-mel generator.

    Main idea:
        acoustic frame query
        + content frame condition
        + semantic cross-attention
        + text/BERT cross-attention
        + style FiLM
        + local Conformer convolution
        -> coarse_mel_pred

    This module is intended to solve the v6.2 bottleneck where:
        predicted_mel ≈ coarse_mel_pred
        and coarse_mel_pred is still blurry / mumbling.
    """

    # Input dimensions
    input_dim: int = 512
    semantic_dim: int | None = None
    text_dim: int | None = None
    style_dim: int | None = None

    # Model dimensions
    hidden_dim: int = 512
    n_mels: int = 80
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    conv_kernel_size: int = 15
    dropout: float = 0.1

    # Cross attention switches
    use_semantic_cross_attn: bool = True
    use_text_cross_attn: bool = True

    # Style conditioning
    use_style_film: bool = True
    use_style_residual: bool = True
    style_residual_scale_init: float = 0.1

    # Acoustic query
    use_content_residual: bool = True
    use_time_features: bool = True
    use_sinusoidal_position: bool = True
    position_scale_init: float = 1.0
    content_residual_scale_init: float = 1.0

    # Block residual scales
    ff_residual_scale_init: float = 0.5
    attn_residual_scale_init: float = 1.0
    cross_residual_scale_init: float = 1.0
    conv_residual_scale_init: float = 1.0

    # Output head
    use_final_layer_norm: bool = True
    mel_bias_init: float = -5.0
    zero_init_output: bool = False
    clamp_output: bool = False
    output_min: float = -12.0
    output_max: float = 4.0


# ============================================================
# Style modulation
# 风格调制模块
# ============================================================
class StyleFiLM(nn.Module):
    """
    FiLM modulation from global style embedding.

    Input:
        x:
            [B, T, D]
        style:
            [B, style_dim]

    Output:
        y:
            [B, T, D]

    This is initialized close to identity:
        gamma ≈ 0, beta ≈ 0
        y = x * (1 + gamma) + beta
    """

    def __init__(
        self,
        hidden_dim: int,
        style_dim: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.style_dim = int(style_dim)

        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.hidden_dim * 2),
        )

        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor | None,
    ) -> torch.Tensor:
        if style is None:
            return x

        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape={tuple(x.shape)}")

        if style.ndim != 2:
            raise ValueError(
                f"style must be [B, style_dim], got shape={tuple(style.shape)}"
            )

        gamma_beta = self.to_gamma_beta(style)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        return x * (1.0 + gamma) + beta


# ============================================================
# v6.2 Legacy Temporal Conv Block
# v6.2 旧版时序卷积块
# ============================================================
class TemporalConvBootstrapBlock(nn.Module):
    """
    Lightweight temporal block for v6.2 coarse mel prediction.

    Shape convention:
        input  x: [B, T, D]
        output y: [B, T, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length.")

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.expansion_factor = int(expansion_factor)

        inner_dim = self.hidden_dim * self.expansion_factor
        padding = self.kernel_size // 2

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.in_proj = nn.Linear(self.hidden_dim, inner_dim)

        self.depthwise_conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=inner_dim,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.out_proj = nn.Linear(inner_dim, self.hidden_dim)

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape={tuple(x.shape)}")

        residual = x

        y = self.norm(x)
        y = self.in_proj(y)

        y = y.transpose(1, 2)
        y = self.depthwise_conv(y)
        y = y.transpose(1, 2)

        y = self.activation(y)
        y = self.dropout(y)
        y = self.out_proj(y)

        out = residual + self.residual_scale * y
        out = apply_frame_mask(out, lengths)

        return out


# ============================================================
# v6.3 FeedForward / Attention / Conv Blocks
# v6.3 FFN / Attention / Conv 模块
# ============================================================
class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward block.

    Shape:
        x: [B, T, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        residual_scale_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.ff_mult = int(ff_mult)

        inner_dim = self.hidden_dim * self.ff_mult

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.in_proj = nn.Linear(self.hidden_dim, inner_dim * 2)
        self.dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(inner_dim, self.hidden_dim)

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x

        y = self.norm(x)
        y, gate = self.in_proj(y).chunk(2, dim=-1)
        y = y * F.silu(gate)
        y = self.dropout(y)
        y = self.out_proj(y)
        y = self.dropout(y)

        out = residual + self.residual_scale * y
        out = apply_frame_mask(out, lengths)

        return out


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block over acoustic frames.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = make_key_padding_mask(lengths, max_len=x.shape[1])

        residual = x
        y = self.norm(x)

        y, _ = self.attn(
            query=y,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        y = self.dropout(y)
        out = residual + self.residual_scale * y
        out = apply_frame_mask(out, lengths)

        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from acoustic frame query to condition memory.

    Query:
        x: [B, T_query, D]
    Memory:
        memory: [B, T_mem, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.dropout = nn.Dropout(float(dropout))

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        query_lengths: torch.Tensor | None = None,
        memory_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory is None:
            return x

        if memory.ndim != 3:
            raise ValueError(f"memory must be [B, T, D], got shape={tuple(memory.shape)}")

        if x.shape[0] != memory.shape[0]:
            raise ValueError(
                f"batch mismatch between x and memory: {x.shape[0]} vs {memory.shape[0]}"
            )

        key_padding_mask = make_key_padding_mask(
            memory_lengths,
            max_len=memory.shape[1],
        )

        residual = x

        q = self.query_norm(x)
        kv = self.memory_norm(memory)

        y, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        y = self.dropout(y)
        out = residual + self.residual_scale * y
        out = apply_frame_mask(out, query_lengths)

        return out


class ConformerConvModule(nn.Module):
    """
    Local Conformer convolution module.

    Shape:
        x: [B, T, D]

    Design:
        LayerNorm
        pointwise Conv1d to 2D
        GLU
        depthwise Conv1d
        GroupNorm
        SiLU
        pointwise Conv1d
        Dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Conformer conv kernel_size should be odd.")

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)

        padding = self.kernel_size // 2

        self.norm = nn.LayerNorm(self.hidden_dim)

        self.pointwise_in = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim * 2,
            kernel_size=1,
        )

        self.depthwise = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=self.hidden_dim,
        )

        # GroupNorm(1, C) is stable for variable-length sequences and batch_size=1.
        self.channel_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.hidden_dim,
        )

        self.activation = nn.SiLU()

        self.pointwise_out = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(float(dropout))

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x

        y = self.norm(x)
        y = y.transpose(1, 2)

        y = self.pointwise_in(y)
        y = F.glu(y, dim=1)

        y = self.depthwise(y)
        y = self.channel_norm(y)
        y = self.activation(y)

        y = self.pointwise_out(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)

        out = residual + self.residual_scale * y
        out = apply_frame_mask(out, lengths)

        return out


class ConformerBootstrapBlock(nn.Module):
    """
    v6.3 Conformer bootstrap block.

    Order:
        FFN half-step
        acoustic self-attention
        semantic cross-attention
        text/BERT cross-attention
        convolution module
        FFN half-step
        final norm
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_mult: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        use_semantic_cross_attn: bool = True,
        use_text_cross_attn: bool = True,
        ff_residual_scale_init: float = 0.5,
        attn_residual_scale_init: float = 1.0,
        cross_residual_scale_init: float = 1.0,
        conv_residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()

        self.use_semantic_cross_attn = bool(use_semantic_cross_attn)
        self.use_text_cross_attn = bool(use_text_cross_attn)

        self.ff1 = SwiGLUFeedForward(
            hidden_dim=hidden_dim,
            ff_mult=ff_mult,
            dropout=dropout,
            residual_scale_init=ff_residual_scale_init,
        )

        self.self_attn = SelfAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            residual_scale_init=attn_residual_scale_init,
        )

        if self.use_semantic_cross_attn:
            self.semantic_cross = CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual_scale_init=cross_residual_scale_init,
            )
        else:
            self.semantic_cross = None

        if self.use_text_cross_attn:
            self.text_cross = CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual_scale_init=cross_residual_scale_init,
            )
        else:
            self.text_cross = None

        self.conv = ConformerConvModule(
            hidden_dim=hidden_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            residual_scale_init=conv_residual_scale_init,
        )

        self.ff2 = SwiGLUFeedForward(
            hidden_dim=hidden_dim,
            ff_mult=ff_mult,
            dropout=dropout,
            residual_scale_init=ff_residual_scale_init,
        )

        self.final_norm = nn.LayerNorm(int(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        target_lengths: torch.Tensor | None = None,
        semantic_memory: torch.Tensor | None = None,
        semantic_lengths: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ff1(x, lengths=target_lengths)
        x = self.self_attn(x, lengths=target_lengths)

        if self.semantic_cross is not None and semantic_memory is not None:
            x = self.semantic_cross(
                x=x,
                memory=semantic_memory,
                query_lengths=target_lengths,
                memory_lengths=semantic_lengths,
            )

        if self.text_cross is not None and text_memory is not None:
            x = self.text_cross(
                x=x,
                memory=text_memory,
                query_lengths=target_lengths,
                memory_lengths=text_lengths,
            )

        x = self.conv(x, lengths=target_lengths)
        x = self.ff2(x, lengths=target_lengths)

        x = self.final_norm(x)
        x = apply_frame_mask(x, target_lengths)

        return x


# ============================================================
# Acoustic frame query builder
# Acoustic frame query 构造器
# ============================================================
class AcousticFrameQueryBuilder(nn.Module):
    """
    Build acoustic-frame query for ConformerMelBootstrapper.

    Input:
        content_frame:
            [B, T_acoustic, input_dim]

        style_global:
            Optional [B, style_dim]

    Output:
        query:
            [B, T_acoustic, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        style_dim: int | None = None,
        use_content_residual: bool = True,
        use_time_features: bool = True,
        use_sinusoidal_position: bool = True,
        position_scale_init: float = 1.0,
        content_residual_scale_init: float = 1.0,
        style_residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.style_dim = style_dim

        self.use_content_residual = bool(use_content_residual)
        self.use_time_features = bool(use_time_features)
        self.use_sinusoidal_position = bool(use_sinusoidal_position)

        self.content_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
        )

        self.content_residual_scale = nn.Parameter(
            torch.tensor(float(content_residual_scale_init), dtype=torch.float32)
        )

        if self.use_time_features:
            self.time_proj = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.time_proj = None

        if self.use_sinusoidal_position:
            self.position_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.position_scale = nn.Parameter(
                torch.tensor(float(position_scale_init), dtype=torch.float32)
            )
        else:
            self.position_proj = None
            self.position_scale = None

        if style_dim is not None:
            self.style_to_query = nn.Sequential(
                nn.LayerNorm(int(style_dim)),
                nn.Linear(int(style_dim), self.hidden_dim),
            )
            self.style_residual_scale = nn.Parameter(
                torch.tensor(float(style_residual_scale_init), dtype=torch.float32)
            )
        else:
            self.style_to_query = None
            self.style_residual_scale = None

    def forward(
        self,
        content_frame: torch.Tensor,
        style_global: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if content_frame.ndim != 3:
            raise ValueError(
                f"content_frame must be [B, T, D], got shape={tuple(content_frame.shape)}"
            )

        B, T, _ = content_frame.shape
        device = content_frame.device
        dtype = content_frame.dtype

        query = self.content_proj(content_frame)

        if self.use_content_residual:
            query = self.content_residual_scale * query

        if self.use_sinusoidal_position:
            assert self.position_proj is not None
            assert self.position_scale is not None

            pos = sinusoidal_position_embedding(
                length=T,
                dim=self.hidden_dim,
                device=device,
                dtype=dtype,
            )
            pos = self.position_proj(pos).unsqueeze(0)
            query = query + self.position_scale * pos

        if self.use_time_features:
            assert self.time_proj is not None

            # normalized time position and reverse time position
            t = torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)
            time_feat = torch.stack([t, 1.0 - t], dim=-1)
            time_h = self.time_proj(time_feat).unsqueeze(0)
            query = query + time_h

        if style_global is not None and self.style_to_query is not None:
            if style_global.ndim != 2:
                raise ValueError(
                    f"style_global must be [B, D], got shape={tuple(style_global.shape)}"
                )

            assert self.style_residual_scale is not None

            style_h = self.style_to_query(style_global).unsqueeze(1)
            query = query + self.style_residual_scale * style_h

        query = apply_frame_mask(query, target_lengths)

        return query


# ============================================================
# v6.2 Legacy CoarseMelBootstrapper
# v6.2 旧版 CoarseMelBootstrapper
# ============================================================
class CoarseMelBootstrapper(nn.Module):
    """
    v6.2 condition-to-coarse-mel bootstrapper.

    This class is kept for backward compatibility with v6.2 checkpoints
    and existing Stage2FlowMatchingAcousticModelV62 imports.
    """

    def __init__(
        self,
        config: CoarseMelBootstrapperConfig,
    ) -> None:
        super().__init__()

        self.config = config

        input_dim = int(config.input_dim)
        semantic_dim = int(config.semantic_dim or config.input_dim)
        style_dim = config.style_dim

        hidden_dim = int(config.hidden_dim)
        n_mels = int(config.n_mels)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        self.use_semantic_guide = bool(config.use_semantic_guide)
        if self.use_semantic_guide:
            self.semantic_proj = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, hidden_dim),
            )

            gate_init = float(config.semantic_gate_init)
            gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
            gate_logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            self.semantic_gate_logit = nn.Parameter(gate_logit)
        else:
            self.semantic_proj = None
            self.semantic_gate_logit = None

        self.use_style_film = bool(config.use_style_film and style_dim is not None)
        if self.use_style_film:
            self.style_film = StyleFiLM(
                hidden_dim=hidden_dim,
                style_dim=int(style_dim),
            )
        else:
            self.style_film = None

        self.use_style_residual = bool(config.use_style_residual and style_dim is not None)
        if self.use_style_residual:
            self.style_to_hidden = nn.Sequential(
                nn.LayerNorm(int(style_dim)),
                nn.Linear(int(style_dim), hidden_dim),
            )
            self.style_residual_scale = nn.Parameter(
                torch.tensor(
                    float(config.style_residual_scale_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.style_to_hidden = None
            self.style_residual_scale = None

        self.blocks = nn.ModuleList(
            [
                TemporalConvBootstrapBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=int(config.kernel_size),
                    dropout=float(config.dropout),
                    expansion_factor=int(config.expansion_factor),
                    residual_scale_init=float(config.residual_scale_init),
                )
                for _ in range(int(config.num_blocks))
            ]
        )

        if bool(config.use_final_layer_norm):
            self.final_norm = nn.LayerNorm(hidden_dim)
        else:
            self.final_norm = nn.Identity()

        self.to_mel = nn.Linear(hidden_dim, n_mels)

        nn.init.zeros_(self.to_mel.weight)
        nn.init.constant_(self.to_mel.bias, float(config.mel_bias_init))

    @property
    def n_mels(self) -> int:
        return int(self.config.n_mels)

    @property
    def hidden_dim(self) -> int:
        return int(self.config.hidden_dim)

    def forward(
        self,
        content_frame: torch.Tensor,
        semantic_guide: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if content_frame.ndim != 3:
            raise ValueError(
                f"content_frame must be [B, T, D], got shape={tuple(content_frame.shape)}"
            )

        B, T, _ = content_frame.shape

        h = self.input_proj(content_frame)

        semantic_gate_value: torch.Tensor | None = None

        if self.use_semantic_guide and semantic_guide is not None:
            if semantic_guide.ndim != 3:
                raise ValueError(
                    "semantic_guide must be [B, T, D], "
                    f"got shape={tuple(semantic_guide.shape)}"
                )

            if semantic_guide.shape[0] != B or semantic_guide.shape[1] != T:
                raise ValueError(
                    "semantic_guide must share [B, T] with content_frame, "
                    f"got content={tuple(content_frame.shape)}, "
                    f"semantic={tuple(semantic_guide.shape)}"
                )

            assert self.semantic_proj is not None
            assert self.semantic_gate_logit is not None

            semantic_h = self.semantic_proj(semantic_guide)
            semantic_gate_value = torch.sigmoid(self.semantic_gate_logit)

            h = h + semantic_gate_value * semantic_h

        if self.use_style_residual and style_global is not None:
            if style_global.ndim != 2:
                raise ValueError(
                    f"style_global must be [B, D], got shape={tuple(style_global.shape)}"
                )

            assert self.style_to_hidden is not None
            assert self.style_residual_scale is not None

            style_h = self.style_to_hidden(style_global).unsqueeze(1)
            h = h + self.style_residual_scale * style_h

        if self.use_style_film and style_global is not None:
            assert self.style_film is not None
            h = self.style_film(h, style_global)

        h = apply_frame_mask(h, target_lengths)

        for block in self.blocks:
            h = block(h, lengths=target_lengths)

        h = self.final_norm(h)
        h = apply_frame_mask(h, target_lengths)

        coarse_mel = self.to_mel(h)

        if bool(self.config.clamp_output):
            coarse_mel = torch.clamp(
                coarse_mel,
                min=float(self.config.output_min),
                max=float(self.config.output_max),
            )

        coarse_mel = apply_frame_mask(coarse_mel, target_lengths)

        if not return_aux:
            return coarse_mel

        aux: dict[str, Any] = {
            "bootstrapper_type": "legacy_conv_v62",
            "coarse_mel_shape": tuple(coarse_mel.shape),
            "coarse_mel_mean": coarse_mel.detach().float().mean(),
            "coarse_mel_std": coarse_mel.detach().float().std(),
            "coarse_mel_min": coarse_mel.detach().float().min(),
            "coarse_mel_max": coarse_mel.detach().float().max(),
        }

        if semantic_gate_value is not None:
            aux["semantic_gate"] = semantic_gate_value.detach()

        if target_lengths is not None:
            aux["target_lengths"] = target_lengths.detach()

        return coarse_mel, aux


# ============================================================
# v6.3 ConformerMelBootstrapper
# v6.3 ConformerMelBootstrapper
# ============================================================
class ConformerMelBootstrapper(nn.Module):
    """
    v6.3 Conformer-based condition-to-mel bootstrapper.

    This module is designed to be stronger than the v6.2 Conv1D bootstrapper.

    Expected inputs:
        content_frame:
            [B, T_acoustic, input_dim]
            Usually frame-level condition from Stage2ConditionEncoder.

        semantic_guide:
            Optional [B, T_semantic_or_acoustic, semantic_dim]
            Can be oracle semantic memory, predicted semantic memory, or
            interpolated semantic guide.

        text_memory:
            Optional [B, T_text, text_dim]
            Can be phoneme/BERT/content memory.

        style_global:
            Optional [B, style_dim]

        target_lengths:
            Optional [B]

        semantic_lengths:
            Optional [B]

        text_lengths:
            Optional [B]

    Output:
        coarse_mel:
            [B, T_acoustic, n_mels]
    """

    def __init__(
        self,
        config: ConformerMelBootstrapperConfig,
    ) -> None:
        super().__init__()

        self.config = config

        input_dim = int(config.input_dim)
        semantic_dim = int(config.semantic_dim or config.input_dim)
        text_dim = int(config.text_dim or config.input_dim)
        style_dim = config.style_dim

        hidden_dim = int(config.hidden_dim)
        n_mels = int(config.n_mels)

        self.query_builder = AcousticFrameQueryBuilder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            use_content_residual=bool(config.use_content_residual),
            use_time_features=bool(config.use_time_features),
            use_sinusoidal_position=bool(config.use_sinusoidal_position),
            position_scale_init=float(config.position_scale_init),
            content_residual_scale_init=float(config.content_residual_scale_init),
            style_residual_scale_init=float(config.style_residual_scale_init),
        )

        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )

        self.use_style_film = bool(config.use_style_film and style_dim is not None)
        if self.use_style_film:
            self.style_film = StyleFiLM(
                hidden_dim=hidden_dim,
                style_dim=int(style_dim),
            )
        else:
            self.style_film = None

        self.blocks = nn.ModuleList(
            [
                ConformerBootstrapBlock(
                    hidden_dim=hidden_dim,
                    num_heads=int(config.num_heads),
                    ff_mult=int(config.ff_mult),
                    conv_kernel_size=int(config.conv_kernel_size),
                    dropout=float(config.dropout),
                    use_semantic_cross_attn=bool(config.use_semantic_cross_attn),
                    use_text_cross_attn=bool(config.use_text_cross_attn),
                    ff_residual_scale_init=float(config.ff_residual_scale_init),
                    attn_residual_scale_init=float(config.attn_residual_scale_init),
                    cross_residual_scale_init=float(config.cross_residual_scale_init),
                    conv_residual_scale_init=float(config.conv_residual_scale_init),
                )
                for _ in range(int(config.num_layers))
            ]
        )

        if bool(config.use_final_layer_norm):
            self.final_norm = nn.LayerNorm(hidden_dim)
        else:
            self.final_norm = nn.Identity()

        self.to_mel = nn.Linear(hidden_dim, n_mels)

        if bool(config.zero_init_output):
            nn.init.zeros_(self.to_mel.weight)
        else:
            nn.init.xavier_uniform_(self.to_mel.weight, gain=0.5)

        nn.init.constant_(self.to_mel.bias, float(config.mel_bias_init))

    @property
    def n_mels(self) -> int:
        return int(self.config.n_mels)

    @property
    def hidden_dim(self) -> int:
        return int(self.config.hidden_dim)

    def forward(
        self,
        content_frame: torch.Tensor,
        semantic_guide: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        return_aux: bool = False,

        # v6.3 extra memory inputs
        text_memory: torch.Tensor | None = None,
        semantic_lengths: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if content_frame.ndim != 3:
            raise ValueError(
                f"content_frame must be [B, T, D], got shape={tuple(content_frame.shape)}"
            )

        B, T, _ = content_frame.shape

        # ----------------------------------------------------
        # 1. Build acoustic frame query
        # ----------------------------------------------------
        h = self.query_builder(
            content_frame=content_frame,
            style_global=style_global,
            target_lengths=target_lengths,
        )

        if self.use_style_film and style_global is not None:
            assert self.style_film is not None
            h = self.style_film(h, style_global)

        # ----------------------------------------------------
        # 2. Project semantic memory
        # ----------------------------------------------------
        semantic_memory = None
        if semantic_guide is not None:
            if semantic_guide.ndim != 3:
                raise ValueError(
                    "semantic_guide must be [B, T, D], "
                    f"got shape={tuple(semantic_guide.shape)}"
                )

            if semantic_guide.shape[0] != B:
                raise ValueError(
                    f"semantic_guide batch mismatch: {semantic_guide.shape[0]} vs {B}"
                )

            semantic_memory = self.semantic_proj(semantic_guide)

        # ----------------------------------------------------
        # 3. Project text memory
        # ----------------------------------------------------
        projected_text_memory = None
        if text_memory is not None:
            if text_memory.ndim != 3:
                raise ValueError(
                    f"text_memory must be [B, T, D], got shape={tuple(text_memory.shape)}"
                )

            if text_memory.shape[0] != B:
                raise ValueError(
                    f"text_memory batch mismatch: {text_memory.shape[0]} vs {B}"
                )

            projected_text_memory = self.text_proj(text_memory)

        # ----------------------------------------------------
        # 4. Conformer blocks
        # ----------------------------------------------------
        h = apply_frame_mask(h, target_lengths)

        for block in self.blocks:
            h = block(
                x=h,
                target_lengths=target_lengths,
                semantic_memory=semantic_memory,
                semantic_lengths=semantic_lengths,
                text_memory=projected_text_memory,
                text_lengths=text_lengths,
            )

        # ----------------------------------------------------
        # 5. Mel projection
        # ----------------------------------------------------
        h = self.final_norm(h)
        h = apply_frame_mask(h, target_lengths)

        coarse_mel = self.to_mel(h)

        if bool(self.config.clamp_output):
            coarse_mel = torch.clamp(
                coarse_mel,
                min=float(self.config.output_min),
                max=float(self.config.output_max),
            )

        coarse_mel = apply_frame_mask(coarse_mel, target_lengths)

        if not return_aux:
            return coarse_mel

        aux: dict[str, Any] = {
            "bootstrapper_type": "conformer_v63",
            "coarse_mel_shape": tuple(coarse_mel.shape),
            "coarse_mel_mean": coarse_mel.detach().float().mean(),
            "coarse_mel_std": coarse_mel.detach().float().std(),
            "coarse_mel_min": coarse_mel.detach().float().min(),
            "coarse_mel_max": coarse_mel.detach().float().max(),
            "has_semantic_memory": semantic_memory is not None,
            "has_text_memory": projected_text_memory is not None,
            "num_layers": int(self.config.num_layers),
            "num_heads": int(self.config.num_heads),
        }

        if target_lengths is not None:
            aux["target_lengths"] = target_lengths.detach()

        if semantic_lengths is not None:
            aux["semantic_lengths"] = semantic_lengths.detach()

        if text_lengths is not None:
            aux["text_lengths"] = text_lengths.detach()

        return coarse_mel, aux


# ============================================================
# Loss helpers
# Loss 辅助函数
# ============================================================
def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Masked L1 loss for mel tensors.

    Args:
        pred:
            [B, T, C]
        target:
            [B, T, C]
        lengths:
            Optional [B]

    Returns:
        scalar loss
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if lengths is None:
        return F.l1_loss(pred, target)

    mask = make_length_mask(lengths, max_len=pred.shape[1])
    mask = mask.unsqueeze(-1).to(dtype=pred.dtype)

    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]

    return diff.sum() / denom


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Masked MSE loss for mel tensors.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if lengths is None:
        return F.mse_loss(pred, target)

    mask = make_length_mask(lengths, max_len=pred.shape[1])
    mask = mask.unsqueeze(-1).to(dtype=pred.dtype)

    diff = (pred - target).pow(2) * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]

    return diff.sum() / denom


def compute_delta(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute first-order temporal delta.

    Args:
        x:
            [B, T, C]

    Returns:
        delta:
            [B, T - 1, C]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, C], got shape={tuple(x.shape)}")

    if x.shape[1] < 2:
        return x[:, :0, :]

    return x[:, 1:, :] - x[:, :-1, :]


def compute_delta2(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute second-order temporal delta.

    Args:
        x:
            [B, T, C]

    Returns:
        delta2:
            [B, T - 2, C]
    """
    d = compute_delta(x)

    if d.shape[1] < 2:
        return d[:, :0, :]

    return d[:, 1:, :] - d[:, :-1, :]


def masked_delta_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Masked L1 loss on first-order temporal delta.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    pred_delta = compute_delta(pred)
    target_delta = compute_delta(target)

    if lengths is None:
        delta_lengths = None
    else:
        delta_lengths = (lengths.long() - 1).clamp_min(0)

    return masked_l1_loss(pred_delta, target_delta, delta_lengths)


def masked_delta2_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Masked L1 loss on second-order temporal delta.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)

    pred_delta2 = compute_delta2(pred)
    target_delta2 = compute_delta2(target)

    if lengths is None:
        delta2_lengths = None
    else:
        delta2_lengths = (lengths.long() - 2).clamp_min(0)

    return masked_l1_loss(pred_delta2, target_delta2, delta2_lengths)


def compute_coarse_mel_losses(
    coarse_mel: torch.Tensor,
    target_acoustic: torch.Tensor,
    target_lengths: torch.Tensor | None = None,
    l1_weight: float = 1.0,
    mse_weight: float = 0.0,
    delta_l1_weight: float = 0.2,
    delta2_l1_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Compute bootstrapper losses.

    Backward compatible with v6.2:
        Existing calls can keep using l1_weight / mse_weight / delta_l1_weight.

    v6.3 can additionally use:
        delta2_l1_weight
    """
    losses: dict[str, torch.Tensor] = {}

    l1 = masked_l1_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )
    mse = masked_mse_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )
    delta_l1 = masked_delta_l1_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )
    delta2_l1 = masked_delta2_l1_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )

    total = (
        float(l1_weight) * l1
        + float(mse_weight) * mse
        + float(delta_l1_weight) * delta_l1
        + float(delta2_l1_weight) * delta2_l1
    )

    losses["coarse_mel_l1_loss"] = l1
    losses["coarse_mel_mse_loss"] = mse
    losses["coarse_mel_delta_l1_loss"] = delta_l1
    losses["coarse_mel_delta2_l1_loss"] = delta2_l1
    losses["coarse_mel_total_loss"] = total

    return losses


# ============================================================
# Bootstrap start helper
# Bootstrap 起点构造辅助函数
# ============================================================
def build_bootstrap_start(
    coarse_mel: torch.Tensor,
    t_start: float,
    temperature: float = 0.3,
    lengths: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build v6.2/v6.3 bootstrap sampling start state.

    Formula:
        x_start = (1 - t_start) * noise + t_start * coarse_mel

    Args:
        coarse_mel:
            [B, T, 80]
        t_start:
            Usually 0.05 or 0.075.
        temperature:
            Noise scale.
        lengths:
            Optional [B] valid lengths.
        noise:
            Optional pre-sampled noise. If None, sampled by randn_like.
        seed:
            Optional seed for reproducible noise.

    Returns:
        x_start:
            [B, T, 80]
        noise:
            [B, T, 80]
    """
    if coarse_mel.ndim != 3:
        raise ValueError(
            f"coarse_mel must be [B, T, C], got shape={tuple(coarse_mel.shape)}"
        )

    t_start = float(t_start)

    if not (0.0 <= t_start < 1.0):
        raise ValueError(f"t_start must be in [0, 1), got {t_start}")

    if noise is None:
        if seed is not None:
            torch.manual_seed(int(seed))
            if coarse_mel.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        noise = torch.randn_like(coarse_mel) * float(temperature)
    else:
        if noise.shape != coarse_mel.shape:
            raise ValueError(
                f"noise shape must match coarse_mel, got "
                f"{tuple(noise.shape)} vs {tuple(coarse_mel.shape)}"
            )

    x_start = (1.0 - t_start) * noise + t_start * coarse_mel
    x_start = apply_frame_mask(x_start, lengths)

    return x_start, noise


# ============================================================
# Minimal smoke test helpers
# 最小冒烟测试辅助函数
# ============================================================
def smoke_test_legacy_bootstrapper() -> None:
    """
    Smoke test for v6.2 legacy CoarseMelBootstrapper.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    T = 100
    D = 512
    S = 512
    style_dim = 512

    cfg = CoarseMelBootstrapperConfig(
        input_dim=D,
        semantic_dim=S,
        style_dim=style_dim,
        hidden_dim=512,
        n_mels=80,
        num_blocks=4,
    )

    model = CoarseMelBootstrapper(cfg).to(device)

    content = torch.randn(B, T, D, device=device)
    semantic = torch.randn(B, T, S, device=device)
    style = torch.randn(B, style_dim, device=device)
    lengths = torch.tensor([100, 73], dtype=torch.long, device=device)

    coarse, aux = model(
        content_frame=content,
        semantic_guide=semantic,
        style_global=style,
        target_lengths=lengths,
        return_aux=True,
    )

    assert coarse.shape == (B, T, 80)

    x_start, noise = build_bootstrap_start(
        coarse_mel=coarse,
        t_start=0.075,
        temperature=0.3,
        lengths=lengths,
    )

    assert x_start.shape == coarse.shape
    assert noise.shape == coarse.shape

    print("[legacy] Smoke test passed.")
    print("[legacy] coarse shape:", tuple(coarse.shape))
    print("[legacy] x_start shape:", tuple(x_start.shape))
    print("[legacy] aux keys:", sorted(aux.keys()))


def smoke_test_conformer_bootstrapper() -> None:
    """
    Smoke test for v6.3 ConformerMelBootstrapper.

    This test uses:
        content_frame: [B, T_acoustic, 512]
        semantic_guide: [B, T_semantic, 512]
        text_memory: [B, T_text, 384]
        style_global: [B, 256]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    T_ac = 120
    T_sem = 38
    T_text = 24

    cfg = ConformerMelBootstrapperConfig(
        input_dim=512,
        semantic_dim=512,
        text_dim=384,
        style_dim=256,
        hidden_dim=512,
        n_mels=80,
        num_layers=2,
        num_heads=8,
        ff_mult=4,
        conv_kernel_size=15,
        dropout=0.1,
    )

    model = ConformerMelBootstrapper(cfg).to(device)

    content_frame = torch.randn(B, T_ac, 512, device=device)
    semantic_guide = torch.randn(B, T_sem, 512, device=device)
    text_memory = torch.randn(B, T_text, 384, device=device)
    style_global = torch.randn(B, 256, device=device)

    target_lengths = torch.tensor([120, 93], dtype=torch.long, device=device)
    semantic_lengths = torch.tensor([38, 31], dtype=torch.long, device=device)
    text_lengths = torch.tensor([24, 20], dtype=torch.long, device=device)

    coarse, aux = model(
        content_frame=content_frame,
        semantic_guide=semantic_guide,
        text_memory=text_memory,
        style_global=style_global,
        target_lengths=target_lengths,
        semantic_lengths=semantic_lengths,
        text_lengths=text_lengths,
        return_aux=True,
    )

    assert coarse.shape == (B, T_ac, 80)

    x_start, noise = build_bootstrap_start(
        coarse_mel=coarse,
        t_start=0.075,
        temperature=0.3,
        lengths=target_lengths,
    )

    assert x_start.shape == coarse.shape
    assert noise.shape == coarse.shape

    print("[conformer] Smoke test passed.")
    print("[conformer] coarse shape:", tuple(coarse.shape))
    print("[conformer] x_start shape:", tuple(x_start.shape))
    print("[conformer] aux keys:", sorted(aux.keys()))


def smoke_test_bootstrapper() -> None:
    """
    Run both v6.2 legacy and v6.3 conformer smoke tests.
    """
    smoke_test_legacy_bootstrapper()
    smoke_test_conformer_bootstrapper()


if __name__ == "__main__":
    smoke_test_bootstrapper()