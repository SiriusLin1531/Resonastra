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
        True  = padding / ignore
        False = valid / keep
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
            [B, T, C]
        lengths:
            Optional [B]

    Returns:
        Masked x.
    """
    if lengths is None:
        return x

    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, C], got shape={tuple(x.shape)}")

    mask = make_length_mask(lengths, max_len=x.shape[1])
    return x * mask.unsqueeze(-1).to(dtype=x.dtype)


def sinusoidal_embedding(
    x: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Build sinusoidal embedding for scalar input.

    Args:
        x:
            Tensor with shape [B], usually time t in [0, 1].
        dim:
            Embedding dimension.

    Returns:
        emb:
            [B, dim]
    """
    if x.ndim != 1:
        raise ValueError(f"x must be [B], got shape={tuple(x.shape)}")

    half = dim // 2
    device = x.device
    dtype = x.dtype

    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=device, dtype=dtype)
        / max(half - 1, 1)
    )

    args = x[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


def resize_time_to(
    x: torch.Tensor,
    target_len: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Resize a [B, T, C] tensor to target temporal length.
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
# Config
# 配置
# ============================================================
@dataclass
class ResidualFlowRefinerConfig:
    """
    Configuration for v6.3 residual flow refiner.

    Purpose:
        Learn residual correction from coarse mel to target mel.

    Instead of directly generating full mel:
        final_mel = generated_mel

    v6.3 residual refiner predicts:
        residual_target = target_mel - coarse_mel
        final_mel = coarse_mel + predicted_residual

    Flow Matching is applied in residual space:
        r0 = sigma * noise
        r1 = target_mel - coarse_mel
        r_t = (1 - t) * r0 + t * r1
        target_flow = r1 - r0
    """

    # Input/output dimensions
    n_mels: int = 80
    input_dim: int = 512
    semantic_dim: int | None = None
    text_dim: int | None = None
    style_dim: int | None = None

    # Model dimensions
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    conv_kernel_size: int = 15
    dropout: float = 0.1

    # Conditioning switches
    use_content_frame: bool = True
    use_semantic_cross_attn: bool = True
    use_text_cross_attn: bool = True
    use_style_film: bool = True
    use_self_condition: bool = True

    # Time conditioning
    time_embed_dim: int = 256
    use_time_film: bool = True

    # Residual scales
    ff_residual_scale_init: float = 0.5
    attn_residual_scale_init: float = 1.0
    cross_residual_scale_init: float = 1.0
    conv_residual_scale_init: float = 1.0

    # Output head
    zero_init_output: bool = True
    output_scale_init: float = 1.0

    # Stability
    use_final_layer_norm: bool = True
    clamp_residual: bool = False
    residual_min: float = -8.0
    residual_max: float = 8.0


# ============================================================
# Conditioning modules
# 条件模块
# ============================================================
class TimeEmbeddingMLP(nn.Module):
    """
    Time embedding for residual flow matching.

    Input:
        t: [B], scalar time in [0, 1]

    Output:
        time_h: [B, hidden_dim]
    """

    def __init__(
        self,
        time_embed_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.time_embed_dim = int(time_embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got shape={tuple(t.shape)}")

        emb = sinusoidal_embedding(t.float(), dim=self.time_embed_dim)
        return self.mlp(emb)


class GlobalFiLM(nn.Module):
    """
    FiLM modulation from global condition.

    This module is initialized close to identity:
        y = x * (1 + gamma) + beta
        gamma/beta initially near zero.
    """

    def __init__(
        self,
        hidden_dim: int,
        global_dim: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.global_dim = int(global_dim)

        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(self.global_dim, self.hidden_dim * 2),
        )

        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        if global_cond is None:
            return x

        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape={tuple(x.shape)}")

        if global_cond.ndim != 2:
            raise ValueError(
                f"global_cond must be [B, D], got shape={tuple(global_cond.shape)}"
            )

        gamma_beta = self.to_gamma_beta(global_cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ResidualInputProjection(nn.Module):
    """
    Project residual-space inputs into hidden dimension.

    Inputs:
        residual_t:
            [B, T, n_mels]
        coarse_mel:
            [B, T, n_mels]
        self_condition:
            Optional [B, T, n_mels]

    Output:
        h:
            [B, T, hidden_dim]
    """

    def __init__(
        self,
        n_mels: int,
        hidden_dim: int,
        use_self_condition: bool = True,
    ) -> None:
        super().__init__()

        self.n_mels = int(n_mels)
        self.hidden_dim = int(hidden_dim)
        self.use_self_condition = bool(use_self_condition)

        in_dim = self.n_mels * 2
        if self.use_self_condition:
            in_dim += self.n_mels

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        residual_t: torch.Tensor,
        coarse_mel: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual_t.shape != coarse_mel.shape:
            raise ValueError(
                "residual_t and coarse_mel must have same shape, got "
                f"{tuple(residual_t.shape)} vs {tuple(coarse_mel.shape)}"
            )

        if residual_t.ndim != 3:
            raise ValueError(
                f"residual_t must be [B, T, C], got shape={tuple(residual_t.shape)}"
            )

        inputs = [residual_t, coarse_mel]

        if self.use_self_condition:
            if self_condition is None:
                self_condition = torch.zeros_like(residual_t)
            elif self_condition.shape != residual_t.shape:
                raise ValueError(
                    "self_condition must match residual_t shape, got "
                    f"{tuple(self_condition.shape)} vs {tuple(residual_t.shape)}"
                )

            inputs.append(self_condition)

        x = torch.cat(inputs, dim=-1)
        return self.net(x)


# ============================================================
# Core blocks
# 核心模块
# ============================================================
class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward block.
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
    Multi-head self-attention over acoustic frames.
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
    Cross-attention from acoustic residual query to condition memory.
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

        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape={tuple(x.shape)}")

        if memory.ndim != 3:
            raise ValueError(
                f"memory must be [B, T_mem, D], got shape={tuple(memory.shape)}"
            )

        if x.shape[0] != memory.shape[0]:
            raise ValueError(
                f"batch mismatch: query B={x.shape[0]}, memory B={memory.shape[0]}"
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
    Local convolution module for acoustic residual refinement.

    This is useful for repairing local spectral details and temporal smoothness.
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
            raise ValueError("kernel_size should be odd.")

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

        # GroupNorm is stable for batch_size=1.
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


class ResidualRefinerBlock(nn.Module):
    """
    v6.3 residual refiner block.

    Order:
        global FiLM(time/style)
        FFN
        self-attention
        semantic cross-attention
        text cross-attention
        local convolution
        FFN
        final norm
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        global_cond_dim: int,
        ff_mult: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        use_semantic_cross_attn: bool = True,
        use_text_cross_attn: bool = True,
        use_global_film: bool = True,
        ff_residual_scale_init: float = 0.5,
        attn_residual_scale_init: float = 1.0,
        cross_residual_scale_init: float = 1.0,
        conv_residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()

        self.use_semantic_cross_attn = bool(use_semantic_cross_attn)
        self.use_text_cross_attn = bool(use_text_cross_attn)
        self.use_global_film = bool(use_global_film)

        if self.use_global_film:
            self.global_film = GlobalFiLM(
                hidden_dim=hidden_dim,
                global_dim=global_cond_dim,
            )
        else:
            self.global_film = None

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
        global_cond: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        semantic_memory: torch.Tensor | None = None,
        semantic_lengths: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.global_film is not None:
            x = self.global_film(x, global_cond)

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
# ResidualFlowRefiner
# Residual Flow Refiner
# ============================================================
class ResidualFlowRefiner(nn.Module):
    """
    v6.3 Residual Flow Refiner.

    This module predicts flow in residual mel space.

    Training:
        coarse_mel:
            [B, T, 80]
        target_mel:
            [B, T, 80]
        residual_target = target_mel - coarse_mel

        r0 = sigma * noise
        r1 = residual_target
        r_t = (1 - t) * r0 + t * r1
        target_flow = r1 - r0

        pred_flow = refiner(r_t, t, coarse_mel, condition)

    Sampling:
        r = sigma * noise
        for t in 0 -> 1:
            r = r + dt * refiner(r, t, coarse_mel, condition)

        final_mel = coarse_mel + r
    """

    def __init__(
        self,
        config: ResidualFlowRefinerConfig,
    ) -> None:
        super().__init__()

        self.config = config

        n_mels = int(config.n_mels)
        input_dim = int(config.input_dim)
        semantic_dim = int(config.semantic_dim or config.input_dim)
        text_dim = int(config.text_dim or config.input_dim)
        style_dim = config.style_dim

        hidden_dim = int(config.hidden_dim)

        self.input_proj = ResidualInputProjection(
            n_mels=n_mels,
            hidden_dim=hidden_dim,
            use_self_condition=bool(config.use_self_condition),
        )

        self.use_content_frame = bool(config.use_content_frame)
        if self.use_content_frame:
            self.content_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
            )
        else:
            self.content_proj = None

        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )

        self.time_mlp = TimeEmbeddingMLP(
            time_embed_dim=int(config.time_embed_dim),
            hidden_dim=hidden_dim,
        )

        if style_dim is not None:
            self.style_proj = nn.Sequential(
                nn.LayerNorm(int(style_dim)),
                nn.Linear(int(style_dim), hidden_dim),
            )
            global_cond_dim = hidden_dim * 2
        else:
            self.style_proj = None
            global_cond_dim = hidden_dim

        self.blocks = nn.ModuleList(
            [
                ResidualRefinerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=int(config.num_heads),
                    global_cond_dim=global_cond_dim,
                    ff_mult=int(config.ff_mult),
                    conv_kernel_size=int(config.conv_kernel_size),
                    dropout=float(config.dropout),
                    use_semantic_cross_attn=bool(config.use_semantic_cross_attn),
                    use_text_cross_attn=bool(config.use_text_cross_attn),
                    use_global_film=bool(config.use_time_film),
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

        self.to_flow = nn.Linear(hidden_dim, n_mels)

        if bool(config.zero_init_output):
            nn.init.zeros_(self.to_flow.weight)
            nn.init.zeros_(self.to_flow.bias)
        else:
            nn.init.xavier_uniform_(self.to_flow.weight, gain=0.5)
            nn.init.zeros_(self.to_flow.bias)

        self.output_scale = nn.Parameter(
            torch.tensor(float(config.output_scale_init), dtype=torch.float32)
        )

    @property
    def n_mels(self) -> int:
        return int(self.config.n_mels)

    @property
    def hidden_dim(self) -> int:
        return int(self.config.hidden_dim)

    def build_global_condition(
        self,
        t: torch.Tensor,
        style_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_h = self.time_mlp(t)

        if self.style_proj is not None and style_global is not None:
            if style_global.ndim != 2:
                raise ValueError(
                    f"style_global must be [B, D], got shape={tuple(style_global.shape)}"
                )
            style_h = self.style_proj(style_global)
            return torch.cat([time_h, style_h], dim=-1)

        return time_h

    def forward(
        self,
        residual_t: torch.Tensor,
        t: torch.Tensor,
        coarse_mel: torch.Tensor,

        # Frame-level conditions
        content_frame: torch.Tensor | None = None,
        semantic_guide: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,

        # Lengths
        target_lengths: torch.Tensor | None = None,
        semantic_lengths: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,

        # Optional self-conditioning
        self_condition: torch.Tensor | None = None,

        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Predict residual flow.

        Args:
            residual_t:
                [B, T, 80]
            t:
                [B], flow time in [0, 1]
            coarse_mel:
                [B, T, 80]
            content_frame:
                Optional [B, T, input_dim]
            semantic_guide:
                Optional [B, T_sem, semantic_dim]
            text_memory:
                Optional [B, T_text, text_dim]
            style_global:
                Optional [B, style_dim]
            target_lengths:
                Optional [B]
            semantic_lengths:
                Optional [B]
            text_lengths:
                Optional [B]
            self_condition:
                Optional [B, T, 80]
            return_aux:
                Whether to return aux dict.

        Returns:
            pred_residual_flow:
                [B, T, 80]
        """
        if residual_t.ndim != 3:
            raise ValueError(
                f"residual_t must be [B, T, C], got shape={tuple(residual_t.shape)}"
            )

        if coarse_mel.shape != residual_t.shape:
            raise ValueError(
                f"coarse_mel must match residual_t, got "
                f"{tuple(coarse_mel.shape)} vs {tuple(residual_t.shape)}"
            )

        if t.ndim != 1 or t.shape[0] != residual_t.shape[0]:
            raise ValueError(
                f"t must be [B], got t={tuple(t.shape)}, B={residual_t.shape[0]}"
            )

        B, T, _ = residual_t.shape

        h = self.input_proj(
            residual_t=residual_t,
            coarse_mel=coarse_mel,
            self_condition=self_condition,
        )

        if self.content_proj is not None and content_frame is not None:
            if content_frame.ndim != 3:
                raise ValueError(
                    f"content_frame must be [B, T, D], got {tuple(content_frame.shape)}"
                )

            if content_frame.shape[0] != B:
                raise ValueError(
                    f"content_frame batch mismatch: {content_frame.shape[0]} vs {B}"
                )

            if content_frame.shape[1] != T:
                content_frame = resize_time_to(content_frame, target_len=T)

            h = h + self.content_proj(content_frame)

        semantic_memory = None
        if semantic_guide is not None:
            if semantic_guide.ndim != 3:
                raise ValueError(
                    f"semantic_guide must be [B, T, D], got {tuple(semantic_guide.shape)}"
                )
            if semantic_guide.shape[0] != B:
                raise ValueError(
                    f"semantic_guide batch mismatch: {semantic_guide.shape[0]} vs {B}"
                )
            semantic_memory = self.semantic_proj(semantic_guide)

        projected_text_memory = None
        if text_memory is not None:
            if text_memory.ndim != 3:
                raise ValueError(
                    f"text_memory must be [B, T, D], got {tuple(text_memory.shape)}"
                )
            if text_memory.shape[0] != B:
                raise ValueError(
                    f"text_memory batch mismatch: {text_memory.shape[0]} vs {B}"
                )
            projected_text_memory = self.text_proj(text_memory)

        global_cond = self.build_global_condition(
            t=t,
            style_global=style_global,
        )

        h = apply_frame_mask(h, target_lengths)

        for block in self.blocks:
            h = block(
                x=h,
                global_cond=global_cond,
                target_lengths=target_lengths,
                semantic_memory=semantic_memory,
                semantic_lengths=semantic_lengths,
                text_memory=projected_text_memory,
                text_lengths=text_lengths,
            )

        h = self.final_norm(h)
        h = apply_frame_mask(h, target_lengths)

        pred_flow = self.output_scale * self.to_flow(h)

        if bool(self.config.clamp_residual):
            pred_flow = torch.clamp(
                pred_flow,
                min=float(self.config.residual_min),
                max=float(self.config.residual_max),
            )

        pred_flow = apply_frame_mask(pred_flow, target_lengths)

        if not return_aux:
            return pred_flow

        aux: dict[str, Any] = {
            "refiner_type": "residual_flow_refiner_v63",
            "pred_flow_shape": tuple(pred_flow.shape),
            "pred_flow_mean": pred_flow.detach().float().mean(),
            "pred_flow_std": pred_flow.detach().float().std(),
            "pred_flow_min": pred_flow.detach().float().min(),
            "pred_flow_max": pred_flow.detach().float().max(),
            "has_content_frame": content_frame is not None,
            "has_semantic_memory": semantic_memory is not None,
            "has_text_memory": projected_text_memory is not None,
            "has_style_global": style_global is not None,
            "output_scale": self.output_scale.detach(),
        }

        if target_lengths is not None:
            aux["target_lengths"] = target_lengths.detach()

        if semantic_lengths is not None:
            aux["semantic_lengths"] = semantic_lengths.detach()

        if text_lengths is not None:
            aux["text_lengths"] = text_lengths.detach()

        return pred_flow, aux

    @torch.inference_mode()
    def sample(
        self,
        coarse_mel: torch.Tensor,

        # Frame-level conditions
        content_frame: torch.Tensor | None = None,
        semantic_guide: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,

        # Lengths
        target_lengths: torch.Tensor | None = None,
        semantic_lengths: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,

        # Sampling config
        num_steps: int = 128,
        temperature: float = 0.3,
        use_heun: bool = True,
        return_aux: bool = True,
    ) -> dict[str, Any]:
        """
        Sample residual and return final mel.

        Args:
            coarse_mel:
                [B, T, 80]
            num_steps:
                ODE integration steps.
            temperature:
                Initial residual noise scale.
            use_heun:
                Whether to use Heun correction.

        Returns:
            dict with:
                final_mel
                predicted_residual
                coarse_mel
                aux
        """
        if coarse_mel.ndim != 3:
            raise ValueError(
                f"coarse_mel must be [B, T, C], got shape={tuple(coarse_mel.shape)}"
            )

        B, T, _ = coarse_mel.shape
        device = coarse_mel.device
        dtype = coarse_mel.dtype

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        valid_mask = None
        if target_lengths is not None:
            valid_mask = make_length_mask(target_lengths, max_len=T).unsqueeze(-1)
            valid_mask = valid_mask.to(device=device, dtype=dtype)

        residual = torch.randn_like(coarse_mel) * float(temperature)
        residual = apply_frame_mask(residual, target_lengths)

        dt = 1.0 / float(num_steps)

        for step in range(num_steps):
            t_scalar = step / float(num_steps)
            t = torch.full(
                size=(B,),
                fill_value=float(t_scalar),
                device=device,
                dtype=torch.float32,
            )

            v = self.forward(
                residual_t=residual,
                t=t,
                coarse_mel=coarse_mel,
                content_frame=content_frame,
                semantic_guide=semantic_guide,
                text_memory=text_memory,
                style_global=style_global,
                target_lengths=target_lengths,
                semantic_lengths=semantic_lengths,
                text_lengths=text_lengths,
                self_condition=None,
                return_aux=False,
            )

            if use_heun:
                residual_euler = residual + dt * v
                residual_euler = apply_frame_mask(residual_euler, target_lengths)

                t_next_scalar = min((step + 1) / float(num_steps), 1.0)
                t_next = torch.full(
                    size=(B,),
                    fill_value=float(t_next_scalar),
                    device=device,
                    dtype=torch.float32,
                )

                v_next = self.forward(
                    residual_t=residual_euler,
                    t=t_next,
                    coarse_mel=coarse_mel,
                    content_frame=content_frame,
                    semantic_guide=semantic_guide,
                    text_memory=text_memory,
                    style_global=style_global,
                    target_lengths=target_lengths,
                    semantic_lengths=semantic_lengths,
                    text_lengths=text_lengths,
                    self_condition=None,
                    return_aux=False,
                )

                residual = residual + 0.5 * dt * (v + v_next)
            else:
                residual = residual + dt * v

            residual = apply_frame_mask(residual, target_lengths)

        final_mel = coarse_mel + residual
        final_mel = apply_frame_mask(final_mel, target_lengths)

        aux: dict[str, Any] = {
            "refiner_type": "residual_flow_refiner_v63",
            "num_steps": int(num_steps),
            "temperature": float(temperature),
            "use_heun": bool(use_heun),
            "predicted_residual_shape": tuple(residual.shape),
            "final_mel_shape": tuple(final_mel.shape),
            "predicted_residual_mean": residual.detach().float().mean(),
            "predicted_residual_std": residual.detach().float().std(),
            "final_mel_mean": final_mel.detach().float().mean(),
            "final_mel_std": final_mel.detach().float().std(),
        }

        if target_lengths is not None:
            aux["target_lengths"] = target_lengths.detach()

        if return_aux:
            return {
                "final_mel": final_mel,
                "predicted_residual": residual,
                "coarse_mel": coarse_mel,
                "aux": aux,
            }

        return {
            "final_mel": final_mel,
            "predicted_residual": residual,
            "coarse_mel": coarse_mel,
        }


# ============================================================
# Residual Flow Matching state helpers
# Residual Flow Matching 状态构造
# ============================================================
def sample_flow_times(
    batch_size: int,
    device: torch.device,
    min_t: float = 0.0,
    max_t: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Sample flow matching times in [min_t, max_t].
    """
    min_t = float(min_t)
    max_t = float(max_t)

    if not (0.0 <= min_t < max_t <= 1.0):
        raise ValueError(f"Invalid time range: min_t={min_t}, max_t={max_t}")

    t = torch.rand(batch_size, device=device)
    t = min_t + (max_t - min_t) * t
    t = t.clamp(eps, 1.0 - eps)

    return t


def build_residual_flow_training_state(
    coarse_mel: torch.Tensor,
    target_acoustic: torch.Tensor,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    noise_scale: float = 1.0,
    target_lengths: torch.Tensor | None = None,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Build residual flow matching training state.

    Args:
        coarse_mel:
            [B, T, 80]
        target_acoustic:
            [B, T, 80]
        t:
            Optional [B].
        noise:
            Optional [B, T, 80].
        noise_scale:
            r0 = noise_scale * noise.
        target_lengths:
            Optional [B].

    Returns:
        dict:
            residual_target
            residual_noise
            residual_t
            target_flow
            t
    """
    if coarse_mel.shape != target_acoustic.shape:
        raise ValueError(
            "coarse_mel and target_acoustic must have same shape, got "
            f"{tuple(coarse_mel.shape)} vs {tuple(target_acoustic.shape)}"
        )

    if coarse_mel.ndim != 3:
        raise ValueError(
            f"coarse_mel must be [B, T, C], got shape={tuple(coarse_mel.shape)}"
        )

    B = coarse_mel.shape[0]
    device = coarse_mel.device

    if t is None:
        t = sample_flow_times(
            batch_size=B,
            device=device,
            min_t=min_t,
            max_t=max_t,
        )
    else:
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim != 1 or t.shape[0] != B:
            raise ValueError(f"t must be [B], got shape={tuple(t.shape)}")

    residual_target = target_acoustic - coarse_mel

    if noise is None:
        noise = torch.randn_like(residual_target)
    else:
        if noise.shape != residual_target.shape:
            raise ValueError(
                f"noise must match residual_target, got "
                f"{tuple(noise.shape)} vs {tuple(residual_target.shape)}"
            )

    residual_noise = float(noise_scale) * noise

    t_view = t.view(B, 1, 1)

    residual_t = (1.0 - t_view) * residual_noise + t_view * residual_target
    target_flow = residual_target - residual_noise

    residual_target = apply_frame_mask(residual_target, target_lengths)
    residual_noise = apply_frame_mask(residual_noise, target_lengths)
    residual_t = apply_frame_mask(residual_t, target_lengths)
    target_flow = apply_frame_mask(target_flow, target_lengths)

    return {
        "residual_target": residual_target,
        "residual_noise": residual_noise,
        "residual_t": residual_t,
        "target_flow": target_flow,
        "t": t,
    }


# ============================================================
# Loss helpers
# Loss 辅助函数
# ============================================================
def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if lengths is None:
        return F.l1_loss(pred, target)

    mask = make_length_mask(lengths, max_len=pred.shape[1])
    mask = mask.unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)

    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]

    return diff.sum() / denom


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if lengths is None:
        return F.mse_loss(pred, target)

    mask = make_length_mask(lengths, max_len=pred.shape[1])
    mask = mask.unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)

    diff = (pred - target).pow(2) * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]

    return diff.sum() / denom


def compute_delta(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, C], got shape={tuple(x.shape)}")

    if x.shape[1] < 2:
        return x[:, :0, :]

    return x[:, 1:, :] - x[:, :-1, :]


def compute_delta2(x: torch.Tensor) -> torch.Tensor:
    d = compute_delta(x)

    if d.shape[1] < 2:
        return d[:, :0, :]

    return d[:, 1:, :] - d[:, :-1, :]


def masked_delta_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
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


def compute_residual_flow_losses(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    residual_t: torch.Tensor,
    t: torch.Tensor,
    coarse_mel: torch.Tensor,
    target_acoustic: torch.Tensor,
    target_lengths: torch.Tensor | None = None,

    # Loss weights
    flow_mse_weight: float = 1.0,
    flow_l1_weight: float = 0.0,
    residual_recon_l1_weight: float = 1.0,
    final_recon_l1_weight: float = 1.0,
    final_delta_l1_weight: float = 0.3,
    final_delta2_l1_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    Compute residual flow refiner losses.

    Important:
        This loss is designed to test whether final_mel improves over coarse_mel.

    Reconstruction:
        residual_recon = residual_t + (1 - t) * pred_flow
        final_recon = coarse_mel + residual_recon
    """
    if pred_flow.shape != target_flow.shape:
        raise ValueError(
            f"pred_flow/target_flow mismatch: "
            f"{tuple(pred_flow.shape)} vs {tuple(target_flow.shape)}"
        )

    if residual_t.shape != pred_flow.shape:
        raise ValueError(
            f"residual_t/pred_flow mismatch: "
            f"{tuple(residual_t.shape)} vs {tuple(pred_flow.shape)}"
        )

    if coarse_mel.shape != target_acoustic.shape:
        raise ValueError(
            f"coarse_mel/target_acoustic mismatch: "
            f"{tuple(coarse_mel.shape)} vs {tuple(target_acoustic.shape)}"
        )

    B = pred_flow.shape[0]

    if t.ndim != 1 or t.shape[0] != B:
        raise ValueError(f"t must be [B], got shape={tuple(t.shape)}")

    flow_mse = masked_mse_loss(
        pred=pred_flow,
        target=target_flow,
        lengths=target_lengths,
    )

    flow_l1 = masked_l1_loss(
        pred=pred_flow,
        target=target_flow,
        lengths=target_lengths,
    )

    t_view = t.view(B, 1, 1)

    residual_recon = residual_t + (1.0 - t_view) * pred_flow
    residual_target = target_acoustic - coarse_mel

    residual_recon = apply_frame_mask(residual_recon, target_lengths)
    residual_target = apply_frame_mask(residual_target, target_lengths)

    final_recon = coarse_mel + residual_recon
    final_recon = apply_frame_mask(final_recon, target_lengths)

    residual_recon_l1 = masked_l1_loss(
        pred=residual_recon,
        target=residual_target,
        lengths=target_lengths,
    )

    final_recon_l1 = masked_l1_loss(
        pred=final_recon,
        target=target_acoustic,
        lengths=target_lengths,
    )

    final_delta_l1 = masked_delta_l1_loss(
        pred=final_recon,
        target=target_acoustic,
        lengths=target_lengths,
    )

    final_delta2_l1 = masked_delta2_l1_loss(
        pred=final_recon,
        target=target_acoustic,
        lengths=target_lengths,
    )

    total = (
        float(flow_mse_weight) * flow_mse
        + float(flow_l1_weight) * flow_l1
        + float(residual_recon_l1_weight) * residual_recon_l1
        + float(final_recon_l1_weight) * final_recon_l1
        + float(final_delta_l1_weight) * final_delta_l1
        + float(final_delta2_l1_weight) * final_delta2_l1
    )

    return {
        "residual_flow_mse_loss": flow_mse,
        "residual_flow_l1_loss": flow_l1,
        "residual_recon_l1_loss": residual_recon_l1,
        "final_recon_l1_loss": final_recon_l1,
        "final_delta_l1_loss": final_delta_l1,
        "final_delta2_l1_loss": final_delta2_l1,
        "residual_refiner_total_loss": total,

        # Useful tensors for diagnostics
        "residual_recon": residual_recon,
        "residual_target": residual_target,
        "final_recon": final_recon,
    }


def compute_coarse_vs_target_losses(
    coarse_mel: torch.Tensor,
    target_acoustic: torch.Tensor,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Lightweight diagnostic losses for coarse mel.

    These are useful when training residual refiner to monitor whether
    final_recon is actually better than coarse_mel.
    """
    coarse_l1 = masked_l1_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )

    coarse_delta_l1 = masked_delta_l1_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )

    coarse_delta2_l1 = masked_delta2_l1_loss(
        pred=coarse_mel,
        target=target_acoustic,
        lengths=target_lengths,
    )

    return {
        "coarse_vs_target_l1_loss": coarse_l1,
        "coarse_vs_target_delta_l1_loss": coarse_delta_l1,
        "coarse_vs_target_delta2_l1_loss": coarse_delta2_l1,
    }


# ============================================================
# Full training helper
# 完整训练辅助函数
# ============================================================
def compute_residual_refiner_training_loss(
    refiner: ResidualFlowRefiner,
    coarse_mel: torch.Tensor,
    target_acoustic: torch.Tensor,

    # Frame-level conditions
    content_frame: torch.Tensor | None = None,
    semantic_guide: torch.Tensor | None = None,
    text_memory: torch.Tensor | None = None,
    style_global: torch.Tensor | None = None,

    # Lengths
    target_lengths: torch.Tensor | None = None,
    semantic_lengths: torch.Tensor | None = None,
    text_lengths: torch.Tensor | None = None,

    # Flow config
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    noise_scale: float = 1.0,
    min_t: float = 0.0,
    max_t: float = 1.0,

    # Loss weights
    flow_mse_weight: float = 1.0,
    flow_l1_weight: float = 0.0,
    residual_recon_l1_weight: float = 1.0,
    final_recon_l1_weight: float = 1.0,
    final_delta_l1_weight: float = 0.3,
    final_delta2_l1_weight: float = 0.1,

    # Optional self-conditioning
    self_condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Convenience helper to compute residual refiner loss.

    Returns:
        loss:
            Scalar tensor.
        aux:
            Dict of detached/loggable losses and useful tensors.
    """
    state = build_residual_flow_training_state(
        coarse_mel=coarse_mel,
        target_acoustic=target_acoustic,
        t=t,
        noise=noise,
        noise_scale=noise_scale,
        target_lengths=target_lengths,
        min_t=min_t,
        max_t=max_t,
    )

    pred_flow, forward_aux = refiner(
        residual_t=state["residual_t"],
        t=state["t"],
        coarse_mel=coarse_mel,
        content_frame=content_frame,
        semantic_guide=semantic_guide,
        text_memory=text_memory,
        style_global=style_global,
        target_lengths=target_lengths,
        semantic_lengths=semantic_lengths,
        text_lengths=text_lengths,
        self_condition=self_condition,
        return_aux=True,
    )

    losses = compute_residual_flow_losses(
        pred_flow=pred_flow,
        target_flow=state["target_flow"],
        residual_t=state["residual_t"],
        t=state["t"],
        coarse_mel=coarse_mel,
        target_acoustic=target_acoustic,
        target_lengths=target_lengths,
        flow_mse_weight=flow_mse_weight,
        flow_l1_weight=flow_l1_weight,
        residual_recon_l1_weight=residual_recon_l1_weight,
        final_recon_l1_weight=final_recon_l1_weight,
        final_delta_l1_weight=final_delta_l1_weight,
        final_delta2_l1_weight=final_delta2_l1_weight,
    )

    coarse_losses = compute_coarse_vs_target_losses(
        coarse_mel=coarse_mel,
        target_acoustic=target_acoustic,
        target_lengths=target_lengths,
    )

    loss = losses["residual_refiner_total_loss"]

    aux: dict[str, torch.Tensor] = {
        "residual_refiner_total_loss": loss.detach(),
        "residual_flow_mse_loss": losses["residual_flow_mse_loss"].detach(),
        "residual_flow_l1_loss": losses["residual_flow_l1_loss"].detach(),
        "residual_recon_l1_loss": losses["residual_recon_l1_loss"].detach(),
        "final_recon_l1_loss": losses["final_recon_l1_loss"].detach(),
        "final_delta_l1_loss": losses["final_delta_l1_loss"].detach(),
        "final_delta2_l1_loss": losses["final_delta2_l1_loss"].detach(),

        "coarse_vs_target_l1_loss": coarse_losses["coarse_vs_target_l1_loss"].detach(),
        "coarse_vs_target_delta_l1_loss": coarse_losses[
            "coarse_vs_target_delta_l1_loss"
        ].detach(),
        "coarse_vs_target_delta2_l1_loss": coarse_losses[
            "coarse_vs_target_delta2_l1_loss"
        ].detach(),

        "t_mean": state["t"].detach().float().mean(),
        "t_min": state["t"].detach().float().min(),
        "t_max": state["t"].detach().float().max(),
    }

    # Improvement diagnostics:
    # positive means final_recon is better than coarse.
    aux["final_minus_coarse_l1_improvement"] = (
        coarse_losses["coarse_vs_target_l1_loss"]
        - losses["final_recon_l1_loss"]
    ).detach()

    aux["final_minus_coarse_delta_l1_improvement"] = (
        coarse_losses["coarse_vs_target_delta_l1_loss"]
        - losses["final_delta_l1_loss"]
    ).detach()

    for key, value in forward_aux.items():
        if torch.is_tensor(value):
            aux[f"refiner_{key}"] = value.detach()

    return loss, aux


# ============================================================
# Smoke test
# 冒烟测试
# ============================================================
def smoke_test_residual_refiner() -> None:
    """
    Minimal smoke test for ResidualFlowRefiner.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    T_ac = 120
    T_sem = 38
    T_text = 24

    cfg = ResidualFlowRefinerConfig(
        n_mels=80,
        input_dim=512,
        semantic_dim=512,
        text_dim=384,
        style_dim=256,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        ff_mult=4,
        conv_kernel_size=15,
        dropout=0.1,
    )

    refiner = ResidualFlowRefiner(cfg).to(device)

    coarse_mel = torch.randn(B, T_ac, 80, device=device)
    target_mel = coarse_mel + 0.25 * torch.randn(B, T_ac, 80, device=device)

    content_frame = torch.randn(B, T_ac, 512, device=device)
    semantic_guide = torch.randn(B, T_sem, 512, device=device)
    text_memory = torch.randn(B, T_text, 384, device=device)
    style_global = torch.randn(B, 256, device=device)

    target_lengths = torch.tensor([120, 93], dtype=torch.long, device=device)
    semantic_lengths = torch.tensor([38, 31], dtype=torch.long, device=device)
    text_lengths = torch.tensor([24, 20], dtype=torch.long, device=device)

    loss, aux = compute_residual_refiner_training_loss(
        refiner=refiner,
        coarse_mel=coarse_mel,
        target_acoustic=target_mel,
        content_frame=content_frame,
        semantic_guide=semantic_guide,
        text_memory=text_memory,
        style_global=style_global,
        target_lengths=target_lengths,
        semantic_lengths=semantic_lengths,
        text_lengths=text_lengths,
        noise_scale=0.5,
    )

    assert torch.isfinite(loss), "loss is not finite"

    print("[train] smoke test passed.")
    print("[train] loss:", float(loss.detach().cpu().item()))
    print("[train] aux keys:", sorted(aux.keys())[:20], "...")

    with torch.inference_mode():
        out = refiner.sample(
            coarse_mel=coarse_mel,
            content_frame=content_frame,
            semantic_guide=semantic_guide,
            text_memory=text_memory,
            style_global=style_global,
            target_lengths=target_lengths,
            semantic_lengths=semantic_lengths,
            text_lengths=text_lengths,
            num_steps=8,
            temperature=0.3,
            use_heun=True,
        )

    assert out["final_mel"].shape == coarse_mel.shape
    assert out["predicted_residual"].shape == coarse_mel.shape

    print("[sample] smoke test passed.")
    print("[sample] final_mel shape:", tuple(out["final_mel"].shape))
    print("[sample] residual shape:", tuple(out["predicted_residual"].shape))
    print("[sample] aux keys:", sorted(out["aux"].keys()))


if __name__ == "__main__":
    smoke_test_residual_refiner()