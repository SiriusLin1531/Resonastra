from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn as nn


@dataclass
class ReferenceAcousticStyleEncoderConfig:
    """
    EN:
    Configuration for the v6.6 reference acoustic style encoder.

    ZH:
    v6.6 参考音频声学风格编码器配置。
    """

    acoustic_dim: int = 80
    hidden_dim: int = 256
    style_dim: int = 256
    num_layers: int = 4
    kernel_size: int = 5
    dropout: float = 0.1
    pooling: str = "attentive_mean"


@dataclass
class ReferenceAcousticStyleOutput:
    """
    EN:
    Output bundle for ReferenceAcousticStyleEncoder.

    ZH:
    ReferenceAcousticStyleEncoder 的输出结构。
    """

    style_global: torch.Tensor
    style_seq: torch.Tensor
    style_mask: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


class ReferenceConvBlock(nn.Module):
    """
    EN:
    Lightweight temporal Conv1d residual block for prompt acoustic features.

    ZH:
    用于 prompt acoustic 特征的轻量时序 Conv1d 残差块。
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        kernel_size = int(kernel_size)
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, T, C]
            mask: [B, T], bool
        """
        residual = x
        y = self.net(x)
        y = y.transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        y = self.dropout(y)
        y = residual + y
        return y * mask.unsqueeze(-1).to(dtype=y.dtype)


class AttentiveMeanPooling(nn.Module):
    """
    EN:
    Mask-aware attentive mean pooling.

    ZH:
    支持 mask 的注意力均值池化。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, T, C]
            mask: [B, T], bool

        Returns:
            pooled: [B, C]
            attn:   [B, T]
        """
        score = self.score(x).squeeze(-1)
        score = score.masked_fill(~mask, -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = attn * mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled, attn


class ReferenceAcousticStyleEncoder(nn.Module):
    """
    EN:
    v6.6 reference acoustic style encoder.

    It maps prompt acoustic features to:
    - style_global: utterance-level reference acoustic style vector
    - style_seq:    frame-level reference acoustic style memory
    - style_mask:   valid frame mask

    ZH:
    v6.6 参考音频声学风格编码器。

    它将 prompt acoustic 特征映射为：
    - style_global: 句级参考音频风格向量
    - style_seq:    帧级参考音频风格序列
    - style_mask:   有效帧 mask
    """

    def __init__(self, config: ReferenceAcousticStyleEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReferenceAcousticStyleEncoderConfig()

        acoustic_dim = int(self.config.acoustic_dim)
        hidden_dim = int(self.config.hidden_dim)
        style_dim = int(self.config.style_dim)
        num_layers = int(self.config.num_layers)

        self.acoustic_in_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                ReferenceConvBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=int(self.config.kernel_size),
                    dropout=float(self.config.dropout),
                )
                for _ in range(max(num_layers, 1))
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.pooling = str(self.config.pooling).strip().lower()
        self.attentive_pool = AttentiveMeanPooling(hidden_dim)
        self.style_global_proj = nn.Sequential(
            nn.Linear(hidden_dim, style_dim),
            nn.LayerNorm(style_dim),
        )
        self.style_seq_proj = nn.Sequential(
            nn.Linear(hidden_dim, style_dim),
            nn.LayerNorm(style_dim),
        )

    @staticmethod
    def make_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        lengths = lengths.long().clamp(min=1, max=max(int(max_len), 1))
        idx = torch.arange(int(max_len), device=lengths.device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    def forward(
        self,
        prompt_acoustic: torch.Tensor,
        prompt_acoustic_lengths: torch.Tensor | None = None,
    ) -> ReferenceAcousticStyleOutput:
        """
        Args:
            prompt_acoustic:
                [B, T_ref, acoustic_dim]
            prompt_acoustic_lengths:
                [B]
        """
        if prompt_acoustic.ndim != 3:
            raise ValueError(
                f"prompt_acoustic must be [B,T,C], got {tuple(prompt_acoustic.shape)}"
            )

        B, T, C = prompt_acoustic.shape
        if int(C) != int(self.config.acoustic_dim):
            raise ValueError(
                f"prompt_acoustic dim mismatch: expected {self.config.acoustic_dim}, got {C}"
            )

        if prompt_acoustic_lengths is None:
            prompt_acoustic_lengths = torch.full(
                size=(B,),
                fill_value=T,
                dtype=torch.long,
                device=prompt_acoustic.device,
            )
        else:
            prompt_acoustic_lengths = prompt_acoustic_lengths.long().to(prompt_acoustic.device)

        mask = self.make_mask(prompt_acoustic_lengths, T)

        x = prompt_acoustic.float()
        x = self.acoustic_in_proj(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        if self.pooling == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
            pooled = (x * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom
            attn = mask.to(dtype=x.dtype) / denom
        else:
            pooled, attn = self.attentive_pool(x, mask)

        style_global = self.style_global_proj(pooled)
        style_seq = self.style_seq_proj(x)
        style_seq = style_seq * mask.unsqueeze(-1).to(dtype=style_seq.dtype)

        aux = {
            "prompt_acoustic_lengths": prompt_acoustic_lengths.detach(),
            "style_global_norm_mean": style_global.detach().norm(dim=-1).mean(),
            "style_seq_norm_mean": style_seq.detach().norm(dim=-1).mean(),
            "attn_max_mean": attn.detach().max(dim=-1).values.mean(),
        }

        return ReferenceAcousticStyleOutput(
            style_global=style_global,
            style_seq=style_seq,
            style_mask=mask,
            aux=aux,
        )
