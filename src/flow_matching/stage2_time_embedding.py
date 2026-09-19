from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2TimeEmbedding(nn.Module):
    """
    EN:
    Sinusoidal time embedding for flow matching time t in [0, 1].

    ZH:
    用于 Flow Matching 中时间 t ∈ [0,1] 的正弦时间嵌入。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        EN:
        t: (B,)
        return: (B, H)

        ZH:
        输入时间步 t，输出时间嵌入。
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)

        half_dim = self.hidden_dim // 2
        device = t.device

        freq = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / max(half_dim - 1, 1)
        )
        phase = t[:, None] * freq[None, :] * 1000.0

        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        if self.hidden_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)