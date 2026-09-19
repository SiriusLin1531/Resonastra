from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeakerIdentityAdapterConfig:
    """
    EN:
    Configuration for v6.6.4 Speaker Identity Adapter.

    ZH:
    v6.6.4 Speaker Identity Adapter 配置。
    """

    speaker_embedding_dim: int = 192
    style_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.05
    fusion_mode: str = "gated_add"
    gate_init: float = -3.0
    normalize_input: bool = True


class SpeakerIdentityAdapter(nn.Module):
    """
    EN:
    Project a speaker embedding into the model style space and inject it into
    `style_global`. This is intentionally small and conservative: it gives
    speaker identity a direct path into the coarse mel generator without
    unfreezing the full backbone.

    ZH:
    将 speaker embedding 投影到模型 style 空间，并注入 `style_global`。
    这是一个保守的小模块：在不解冻完整主干的前提下，让说话人身份信息
    直接进入 coarse mel 生成路径。
    """

    def __init__(self, config: SpeakerIdentityAdapterConfig) -> None:
        super().__init__()
        self.config = config

        self.speaker_embedding_dim = int(config.speaker_embedding_dim)
        self.style_dim = int(config.style_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.num_layers = max(int(config.num_layers), 1)
        self.dropout_p = float(config.dropout)
        self.fusion_mode = str(config.fusion_mode).strip().lower()
        self.normalize_input = bool(config.normalize_input)

        if self.fusion_mode not in {"gated_add", "add", "film"}:
            raise ValueError(
                "fusion_mode must be one of gated_add/add/film, "
                f"got {config.fusion_mode!r}"
            )

        layers: list[nn.Module] = []
        in_dim = self.speaker_embedding_dim
        for _ in range(max(self.num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(self.dropout_p))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.style_dim))
        self.projector = nn.Sequential(*layers)

        if self.fusion_mode == "film":
            self.film = nn.Linear(self.style_dim, self.style_dim * 2)
        else:
            self.film = None

        self.gate = nn.Parameter(torch.tensor(float(config.gate_init)))

    def forward(
        self,
        *,
        style_global: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not torch.is_tensor(style_global):
            raise TypeError("style_global must be a tensor.")
        if not torch.is_tensor(speaker_embedding):
            raise TypeError("speaker_embedding must be a tensor.")

        if speaker_embedding.dim() > 2:
            speaker_embedding = speaker_embedding.view(speaker_embedding.shape[0], -1)
        if speaker_embedding.dim() != 2:
            raise ValueError(
                "speaker_embedding must have shape (B, D) after flattening, "
                f"got {tuple(speaker_embedding.shape)}"
            )
        if int(speaker_embedding.shape[-1]) != self.speaker_embedding_dim:
            raise ValueError(
                f"Expected speaker embedding dim {self.speaker_embedding_dim}, "
                f"got {int(speaker_embedding.shape[-1])}."
            )

        speaker_embedding = speaker_embedding.to(
            device=style_global.device,
            dtype=style_global.dtype,
        )

        if self.normalize_input:
            speaker_input = F.normalize(speaker_embedding.float(), dim=-1).to(dtype=style_global.dtype)
        else:
            speaker_input = speaker_embedding

        speaker_style = self.projector(speaker_input)
        gate = torch.sigmoid(self.gate).to(device=style_global.device, dtype=style_global.dtype)

        if self.fusion_mode == "add":
            fused = style_global + speaker_style
        elif self.fusion_mode == "gated_add":
            fused = style_global + gate * speaker_style
        elif self.fusion_mode == "film":
            if self.film is None:
                raise RuntimeError("film layer is missing for fusion_mode='film'.")
            gamma_beta = self.film(speaker_style)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            fused = style_global * (1.0 + gate * torch.tanh(gamma)) + gate * beta
        else:
            raise RuntimeError(f"Unsupported fusion_mode: {self.fusion_mode}")

        stats = {
            "available": style_global.new_tensor(1.0),
            "used": style_global.new_tensor(1.0),
            "gate": gate.detach(),
            "speaker_embedding_norm_mean": speaker_embedding.detach().float().norm(dim=-1).mean(),
            "speaker_style_norm_mean": speaker_style.detach().float().norm(dim=-1).mean(),
            "input_style_norm_mean": style_global.detach().float().norm(dim=-1).mean(),
            "fused_style_norm_mean": fused.detach().float().norm(dim=-1).mean(),
        }
        return fused, stats


def _as_tensor_or_none(x: Any) -> torch.Tensor | None:
    return x if torch.is_tensor(x) else None


def select_v664_speaker_identity_embedding(
    extras: dict[str, Any] | None,
    *,
    source: str = "prompt_or_target",
) -> tuple[torch.Tensor | None, str]:
    """
    EN:
    Select speaker embedding from Stage2Inputs.extras.

    ZH:
    从 Stage2Inputs.extras 中选择用于 speaker identity adapter 的说话人向量。
    """
    if not isinstance(extras, dict):
        return None, "missing_extras"

    mode = str(source).strip().lower()
    if mode == "prompt":
        x = _as_tensor_or_none(extras.get("prompt_speaker_embedding"))
        return x, "prompt" if x is not None else "missing_prompt"
    if mode == "target":
        x = _as_tensor_or_none(extras.get("target_speaker_embedding"))
        return x, "target" if x is not None else "missing_target"
    if mode == "prompt_or_target":
        x = _as_tensor_or_none(extras.get("prompt_speaker_embedding"))
        if x is not None:
            return x, "prompt"
        x = _as_tensor_or_none(extras.get("target_speaker_embedding"))
        return x, "target" if x is not None else "missing_prompt_or_target"
    if mode == "target_or_prompt":
        x = _as_tensor_or_none(extras.get("target_speaker_embedding"))
        if x is not None:
            return x, "target"
        x = _as_tensor_or_none(extras.get("prompt_speaker_embedding"))
        return x, "prompt" if x is not None else "missing_target_or_prompt"

    raise ValueError(
        "source must be one of prompt/target/prompt_or_target/target_or_prompt, "
        f"got {source!r}"
    )
