from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2FMBlock(nn.Module):
    """
    EN:
    Original minimal Transformer-style block for flow matching acoustic modeling.

    ZH:
    原始最小版 Flow Matching 声学生成 Transformer 风格 block。
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.time_proj_1 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj_1 = nn.Linear(hidden_dim, hidden_dim)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.time_proj_2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, H)
        time_emb: torch.Tensor,   # (B, H)
        frame_cond: torch.Tensor, # (B, T, H)
    ) -> torch.Tensor:
        h = self.ln1(x)
        h = h + self.time_proj_1(time_emb).unsqueeze(1)
        h = h + self.cond_proj_1(frame_cond)

        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.ln2(x)
        h = h + self.time_proj_2(time_emb).unsqueeze(1)
        h = h + self.cond_proj_2(frame_cond)

        x = x + self.ffn(h)
        return x


# ============================================================
# v2.x shared components / v2.x 通用组件
# ============================================================

class Stage2AdaptiveModulation(nn.Module):
    """
    EN:
    FiLM/AdaLN-style modulation from time embedding + global style.

    ZH:
    从 time embedding + global style 生成的 FiLM/AdaLN 风格调制。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.time_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.style_proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(
        self,
        x: torch.Tensor,              # (B, T, H)
        time_emb: torch.Tensor,       # (B, H)
        global_style: torch.Tensor | None = None,  # (B, H)
    ) -> torch.Tensor:
        mod = self.time_proj(time_emb)
        if global_style is not None:
            mod = mod + self.style_proj(global_style)

        scale, shift = mod.chunk(2, dim=-1)        # (B, H), (B, H)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class Stage2LocalConvModule(nn.Module):
    """
    EN:
    Lightweight local convolution module for mel continuity modeling.

    ZH:
    用于 mel 局部连续性建模的轻量卷积分支。
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.pointwise_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.pointwise_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        EN:
        x: (B, T, H)
        return: (B, T, H)

        ZH:
        输入输出形状均为 (B, T, H)。
        """
        y = x.transpose(1, 2)                         # (B, H, T)
        y = self.pointwise_in(y)                      # (B, 2H, T)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)                      # GLU-style gate
        y = self.depthwise(y)
        y = F.gelu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)                      # (B, T, H)


# ============================================================
# v2.2 hybrid backbone block / v2.2 主干 block
# ============================================================

class Stage2FMHybridBlock(nn.Module):
    """
    EN:
    v2.2 hybrid block:
    - masked self-attention
    - native-length cross-attention to semantic/text/prompt
    - branch keep masks for conditional dropout / CFG
    - local convolution module
    - FFN
    - time/style adaptive modulation

    ZH:
    v2.2 混合 block：
    - 带 mask 的 self-attention
    - 对 semantic/text/prompt 原生长度序列做 cross-attention
    - 分支 keep mask，用于条件 dropout / CFG
    - 局部卷积分支
    - FFN
    - time/style 自适应调制
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Self-attention branch
        # self-attention 分支
        # ----------------------------------------------------
        self.ln_self = nn.LayerNorm(hidden_dim)
        self.mod_self = Stage2AdaptiveModulation(hidden_dim)
        self.frame_proj_self = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Semantic native cross-attention
        # semantic 原生序列 cross-attention
        # ----------------------------------------------------
        self.ln_sem = nn.LayerNorm(hidden_dim)
        self.mod_sem = Stage2AdaptiveModulation(hidden_dim)
        self.frame_proj_sem = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Text native cross-attention
        # text 原生序列 cross-attention
        # ----------------------------------------------------
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.mod_text = Stage2AdaptiveModulation(hidden_dim)
        self.frame_proj_text = nn.Linear(hidden_dim, hidden_dim)
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Prompt native cross-attention
        # prompt 原生序列 cross-attention
        # ----------------------------------------------------
        self.ln_prompt = nn.LayerNorm(hidden_dim)
        self.mod_prompt = Stage2AdaptiveModulation(hidden_dim)
        self.frame_proj_prompt = nn.Linear(hidden_dim, hidden_dim)
        self.prompt_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Local convolution branch
        # 局部卷积分支
        # ----------------------------------------------------
        self.ln_conv = nn.LayerNorm(hidden_dim)
        self.mod_conv = Stage2AdaptiveModulation(hidden_dim)
        self.frame_proj_conv = nn.Linear(hidden_dim, hidden_dim)
        self.local_conv = Stage2LocalConvModule(
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=conv_kernel_size,
        )

        # ----------------------------------------------------
        # FFN branch
        # FFN 分支
        # ----------------------------------------------------
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.mod_ffn = Stage2AdaptiveModulation(hidden_dim)
        self.frame_proj_ffn = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def _resolve_keep_mask(
        self,
        keep_mask: torch.Tensor | None,
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
        if keep_mask is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        return keep_mask.to(device=device, dtype=dtype).view(batch_size)

    def forward(
        self,
        x: torch.Tensor,                        # (B, T_ac, H)
        time_emb: torch.Tensor,                # (B, H)
        frame_cond: torch.Tensor,              # (B, T_ac, H)
        self_key_padding_mask: torch.Tensor | None = None,   # (B, T_ac), True for PAD

        semantic_seq: torch.Tensor | None = None,            # (B, T_sem, H)
        semantic_key_padding_mask: torch.Tensor | None = None,
        semantic_keep_mask: torch.Tensor | None = None,      # (B,)

        text_seq: torch.Tensor | None = None,                # (B, T_text, H)
        text_key_padding_mask: torch.Tensor | None = None,
        text_keep_mask: torch.Tensor | None = None,          # (B,)

        prompt_seq: torch.Tensor | None = None,              # (B, T_prompt, H)
        prompt_key_padding_mask: torch.Tensor | None = None,
        prompt_keep_mask: torch.Tensor | None = None,        # (B,)

        global_style: torch.Tensor | None = None,            # (B, H)
    ) -> torch.Tensor:
        B, _, H = x.shape
        device = x.device
        dtype = x.dtype

        sem_keep = self._resolve_keep_mask(semantic_keep_mask, B, device, dtype)
        txt_keep = self._resolve_keep_mask(text_keep_mask, B, device, dtype)
        prm_keep = self._resolve_keep_mask(prompt_keep_mask, B, device, dtype)

        # ----------------------------------------------------
        # 1. Masked self-attention
        # 1. 带 mask 的 self-attention
        # ----------------------------------------------------
        h = self.ln_self(x)
        h = self.mod_self(h, time_emb=time_emb, global_style=global_style)
        h = h + self.frame_proj_self(frame_cond)
        self_out, _ = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        x = x + self_out

        # ----------------------------------------------------
        # 2. Semantic cross-attention
        # 2. Semantic cross-attention
        # ----------------------------------------------------
        if semantic_seq is not None:
            h = self.ln_sem(x)
            h = self.mod_sem(h, time_emb=time_emb, global_style=global_style)
            h = h + self.frame_proj_sem(frame_cond)

            sem_out, _ = self.semantic_cross_attn(
                h,
                semantic_seq,
                semantic_seq,
                key_padding_mask=semantic_key_padding_mask,
                need_weights=False,
            )
            sem_out = sem_out * sem_keep[:, None, None]
            x = x + sem_out

        # ----------------------------------------------------
        # 3. Text cross-attention
        # 3. Text cross-attention
        # ----------------------------------------------------
        if text_seq is not None:
            h = self.ln_text(x)
            h = self.mod_text(h, time_emb=time_emb, global_style=global_style)
            h = h + self.frame_proj_text(frame_cond)

            text_out, _ = self.text_cross_attn(
                h,
                text_seq,
                text_seq,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )
            text_out = text_out * txt_keep[:, None, None]
            x = x + text_out

        # ----------------------------------------------------
        # 4. Prompt cross-attention
        # 4. Prompt cross-attention
        # ----------------------------------------------------
        if prompt_seq is not None:
            h = self.ln_prompt(x)
            h = self.mod_prompt(h, time_emb=time_emb, global_style=global_style)
            h = h + self.frame_proj_prompt(frame_cond)

            prompt_out, _ = self.prompt_cross_attn(
                h,
                prompt_seq,
                prompt_seq,
                key_padding_mask=prompt_key_padding_mask,
                need_weights=False,
            )
            prompt_out = prompt_out * prm_keep[:, None, None]
            x = x + prompt_out

        # ----------------------------------------------------
        # 5. Local convolution branch
        # 5. 局部卷积分支
        # ----------------------------------------------------
        h = self.ln_conv(x)
        h = self.mod_conv(h, time_emb=time_emb, global_style=global_style)
        h = h + self.frame_proj_conv(frame_cond)
        x = x + self.local_conv(h)

        # ----------------------------------------------------
        # 6. FFN branch
        # 6. FFN 分支
        # ----------------------------------------------------
        h = self.ln_ffn(x)
        h = self.mod_ffn(h, time_emb=time_emb, global_style=global_style)
        h = h + self.frame_proj_ffn(frame_cond)
        x = x + self.ffn(h)

        return x


# ============================================================
# v2.3 refinement modules / v2.3 细化模块
# ============================================================

class Stage2StyleOnlyModulation(nn.Module):
    """
    EN:
    Style-only FiLM/AdaLN-style modulation for refinement blocks.

    ZH:
    refinement block 使用的 style-only FiLM/AdaLN 风格调制。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.style_proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(
        self,
        x: torch.Tensor,              # (B, T, H)
        global_style: torch.Tensor | None = None,  # (B, H)
    ) -> torch.Tensor:
        if global_style is None:
            return x

        mod = self.style_proj(global_style)
        scale, shift = mod.chunk(2, dim=-1)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class Stage2RefinementBlock(nn.Module):
    """
    EN:
    A lightweight refinement block used inside the postnet.

    Structure:
    - masked self-attention
    - local conv branch
    - FFN

    ZH:
    postnet 内部使用的轻量 refinement block。

    结构：
    - 带 mask 的 self-attention
    - 局部卷积分支
    - FFN
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.mod_attn = Stage2StyleOnlyModulation(hidden_dim)
        self.cond_proj_attn = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_conv = nn.LayerNorm(hidden_dim)
        self.mod_conv = Stage2StyleOnlyModulation(hidden_dim)
        self.cond_proj_conv = nn.Linear(hidden_dim, hidden_dim)
        self.local_conv = Stage2LocalConvModule(
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=conv_kernel_size,
        )

        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.mod_ffn = Stage2StyleOnlyModulation(hidden_dim)
        self.cond_proj_ffn = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, H)
        frame_cond: torch.Tensor,               # (B, T, H)
        global_style: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,  # (B, T), True for PAD
    ) -> torch.Tensor:
        h = self.ln_attn(x)
        h = self.mod_attn(h, global_style=global_style)
        h = h + self.cond_proj_attn(frame_cond)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        h = self.ln_conv(x)
        h = self.mod_conv(h, global_style=global_style)
        h = h + self.cond_proj_conv(frame_cond)
        x = x + self.local_conv(h)

        h = self.ln_ffn(x)
        h = self.mod_ffn(h, global_style=global_style)
        h = h + self.cond_proj_ffn(frame_cond)
        x = x + self.ffn(h)

        return x


class Stage2MelRefinementPostnet(nn.Module):
    """
    EN:
    Coarse-to-fine mel refinement postnet.

    Input:
    - coarse mel
    - frame_cond
    - global_style
    Output:
    - residual mel to be added onto the coarse mel

    ZH:
    coarse-to-fine mel refinement postnet。

    输入：
    - coarse mel
    - frame_cond
    - global_style
    输出：
    - 加到 coarse mel 上的 residual mel
    """

    def __init__(
        self,
        acoustic_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.mel_in_proj = nn.Linear(acoustic_dim, hidden_dim)
        self.frame_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.style_proj = nn.Linear(hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                Stage2RefinementBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, acoustic_dim)

    def forward(
        self,
        coarse_mel: torch.Tensor,                  # (B, T, C)
        frame_cond: torch.Tensor,                  # (B, T, H)
        global_style: torch.Tensor | None = None, # (B, H)
        key_padding_mask: torch.Tensor | None = None,  # (B, T), True for PAD
    ) -> torch.Tensor:
        h = self.mel_in_proj(coarse_mel)
        h = h + self.frame_cond_proj(frame_cond)

        if global_style is not None:
            h = h + self.style_proj(global_style).unsqueeze(1)

        for block in self.blocks:
            h = block(
                x=h,
                frame_cond=frame_cond,
                global_style=global_style,
                key_padding_mask=key_padding_mask,
            )

        h = self.out_ln(h)
        residual = self.out_proj(h)

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).float()
            residual = residual * valid_mask

        return residual


# ============================================================
# v5.0 content-style backbone block / v5.0 内容-风格主干 block
# ============================================================

class Stage2StyleModulation(nn.Module):
    """
    EN:
    Lightweight FiLM-style modulation from utterance-level style embedding.

    ZH:
    基于句级 style embedding 的轻量 FiLM 调制。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(hidden_dim, hidden_dim)
        self.to_shift = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        style_global: torch.Tensor | None,
    ) -> torch.Tensor:
        if style_global is None:
            return x
        scale = self.to_scale(style_global).unsqueeze(1)
        shift = self.to_shift(style_global).unsqueeze(1)
        return x * (1.0 + scale) + shift


class Stage2CrossAttentionBranch(nn.Module):
    """
    EN:
    Shared cross-attention branch used by v5 content streams.

    ZH:
    v5 内容流共用的 cross-attention 分支。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def _resolve_keep_mask(
        self,
        keep_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if keep_mask is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        return keep_mask.to(device=device, dtype=dtype).view(batch_size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory is None:
            return x

        B = x.shape[0]
        keep = self._resolve_keep_mask(keep_mask, B, x.device, x.dtype)

        h = self.ln(x)
        out, _ = self.attn(
            query=h,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        out = out * keep[:, None, None]
        return x + self.dropout(out)


class Stage2FMContentStyleBlock(nn.Module):
    """
    EN:
    v5.0 backbone block:
    - masked self-attention
    - semantic cross-attention
    - text cross-attention
    - style FiLM modulation
    - FFN + local conv mixing

    ZH:
    v5.0 主干 block：
    - 带 mask 的 self-attention
    - semantic cross-attention
    - text cross-attention
    - style FiLM 调制
    - FFN + 局部卷积混合
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.ln_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.semantic_branch = Stage2CrossAttentionBranch(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.text_branch = Stage2CrossAttentionBranch(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.time_proj_1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj_2 = nn.Linear(hidden_dim, hidden_dim)
        self.style_mod = Stage2StyleModulation(hidden_dim)

        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.local_conv = Stage2LocalConvModule(
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=conv_kernel_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        self_key_padding_mask: torch.Tensor | None = None,
        semantic_seq: torch.Tensor | None = None,
        semantic_key_padding_mask: torch.Tensor | None = None,
        semantic_keep_mask: torch.Tensor | None = None,
        text_seq: torch.Tensor | None = None,
        text_key_padding_mask: torch.Tensor | None = None,
        text_keep_mask: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln_self(x)
        self_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(self_out)

        x = x + self.time_proj_1(time_emb).unsqueeze(1)
        x = self.semantic_branch(
            x=x,
            memory=semantic_seq,
            memory_key_padding_mask=semantic_key_padding_mask,
            keep_mask=semantic_keep_mask,
        )
        x = self.text_branch(
            x=x,
            memory=text_seq,
            memory_key_padding_mask=text_key_padding_mask,
            keep_mask=text_keep_mask,
        )
        x = self.style_mod(x, style_global)

        h = self.ln_ffn(x)
        ffn_out = self.ffn(h)
        x = x + self.dropout(ffn_out)
        x = x + self.local_conv(x)
        x = x + self.time_proj_2(time_emb).unsqueeze(1)
        return x


# ============================================================
# v6.0 components / v6.0 组件
# ============================================================

class Stage2AdaLNModulation(nn.Module):
    """
    EN:
    DiT-style AdaLN modulation for v6.0.

    Uses:
    - time embedding
    - optional global style embedding

    Produces:
    - attn branch scale / shift / gate
    - ffn branch scale / shift / gate

    ZH:
    v6.0 使用的 DiT 风格 AdaLN 调制模块。

    输入：
    - time embedding
    - 可选 global style embedding

    输出：
    - attention 分支 scale / shift / gate
    - ffn 分支 scale / shift / gate
    """

    def __init__(
        self,
        hidden_dim: int,
        style_dim: int | None = None,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.style_dim = int(style_dim or 0)
        in_dim = hidden_dim + self.style_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * expansion, hidden_dim * 6),
        )

    def forward(
        self,
        time_emb: torch.Tensor,                   # (B, H)
        style_global: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if style_global is None:
            cond = time_emb
        else:
            cond = torch.cat([time_emb, style_global], dim=-1)

        out = self.net(cond)
        (
            attn_scale,
            attn_shift,
            attn_gate,
            ffn_scale,
            ffn_shift,
            ffn_gate,
        ) = out.chunk(6, dim=-1)

        return {
            "attn_scale": attn_scale,
            "attn_shift": attn_shift,
            "attn_gate": torch.sigmoid(attn_gate),
            "ffn_scale": ffn_scale,
            "ffn_shift": ffn_shift,
            "ffn_gate": torch.sigmoid(ffn_gate),
        }


class Stage2FeedForward(nn.Module):
    """
    EN:
    Feed-forward module used by v6 blocks.

    ZH:
    v6 block 使用的前馈网络。
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        inner_dim = hidden_dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage2FrameContentCrossAttention(nn.Module):
    """
    EN:
    Cross-attention from acoustic hidden states to frame-level content condition.

    ZH:
    acoustic hidden 对 frame-level 内容条件的 cross-attention。
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,                 # (B, T, H)
        content_frame_cond: torch.Tensor,            # (B, T_ctx, C_ctx)
        content_frame_mask: torch.Tensor | None = None,  # (B, T_ctx), True for valid
    ) -> torch.Tensor:
        h = self.ln_q(hidden_states)
        context = self.context_proj(content_frame_cond)

        key_padding_mask = None
        if content_frame_mask is not None:
            key_padding_mask = ~content_frame_mask

        out, _ = self.attn(
            query=h,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(out)


class Stage2SemanticGuideCrossAttention(nn.Module):
    """
    EN:
    Semantic-guide cross-attention used in v6.0.

    ZH:
    v6.0 中使用的 semantic-guide cross-attention。
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,                # (B, T, H)
        semantic_cond: torch.Tensor,                # (B, T_sem, C_sem)
        semantic_mask: torch.Tensor | None = None,  # (B, T_sem), True for valid
    ) -> torch.Tensor:
        h = self.ln_q(hidden_states)
        context = self.context_proj(semantic_cond)

        key_padding_mask = None
        if semantic_mask is not None:
            key_padding_mask = ~semantic_mask

        out, _ = self.attn(
            query=h,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(out)


class Stage2DiTStyleFMBlockV6(nn.Module):
    """
    EN:
    v6.0 main backbone block.

    Structure:
    1. masked self-attention with AdaLN modulation
    2. frame-level content cross-attention
    3. optional semantic-guide cross-attention
    4. local convolution branch
    5. FFN with AdaLN modulation

    ZH:
    v6.0 主干 block。

    结构：
    1. 带 mask 的 self-attention + AdaLN 调制
    2. frame-level 内容 cross-attention
    3. 可选 semantic-guide cross-attention
    4. 局部卷积分支
    5. FFN + AdaLN 调制
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        content_dim: int,
        semantic_dim: int | None = None,
        style_dim: int | None = None,
        ff_mult: int = 4,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
        use_semantic_guide_cross_attn: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_semantic_guide_cross_attn = bool(use_semantic_guide_cross_attn)

        self.modulation = Stage2AdaLNModulation(
            hidden_dim=hidden_dim,
            style_dim=style_dim,
        )

        # self-attention
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(dropout)

        # frame-level content cross-attention
        self.norm_content = nn.LayerNorm(hidden_dim)
        self.content_cross_attn = Stage2FrameContentCrossAttention(
            hidden_dim=hidden_dim,
            context_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # semantic guide cross-attention
        self.norm_semantic = nn.LayerNorm(hidden_dim)
        if self.use_semantic_guide_cross_attn:
            self.semantic_cross_attn = Stage2SemanticGuideCrossAttention(
                hidden_dim=hidden_dim,
                context_dim=int(semantic_dim or hidden_dim),
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.semantic_cross_attn = None

        # local conv
        self.norm_conv = nn.LayerNorm(hidden_dim)
        self.local_conv = Stage2LocalConvModule(
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=conv_kernel_size,
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = Stage2FeedForward(
            hidden_dim=hidden_dim,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def _apply_scale_shift(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,                       # (B, T, H)
        time_emb: torch.Tensor,                            # (B, H)
        content_frame_cond: torch.Tensor,
        content_frame_mask: torch.Tensor | None = None,
        semantic_cond: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
        hidden_mask: torch.Tensor | None = None,          # (B, T), True for valid
    ) -> torch.Tensor:
        mod = self.modulation(
            time_emb=time_emb,
            style_global=style_global,
        )

        key_padding_mask = None
        if hidden_mask is not None:
            key_padding_mask = ~hidden_mask

        # ----------------------------------------------------
        # 1. self-attention
        # ----------------------------------------------------
        h = self.norm_self(hidden_states)
        h = self._apply_scale_shift(h, mod["attn_scale"], mod["attn_shift"])
        self_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + mod["attn_gate"].unsqueeze(1) * self.self_dropout(self_out)

        # ----------------------------------------------------
        # 2. frame-level content cross-attention
        # ----------------------------------------------------
        h = self.norm_content(hidden_states)
        h = self._apply_scale_shift(h, mod["attn_scale"], mod["attn_shift"])
        content_out = self.content_cross_attn(
            hidden_states=h,
            content_frame_cond=content_frame_cond,
            content_frame_mask=content_frame_mask,
        )
        hidden_states = hidden_states + content_out

        # ----------------------------------------------------
        # 3. semantic-guide cross-attention
        # ----------------------------------------------------
        if self.use_semantic_guide_cross_attn and self.semantic_cross_attn is not None and semantic_cond is not None:
            h = self.norm_semantic(hidden_states)
            h = self._apply_scale_shift(h, mod["attn_scale"], mod["attn_shift"])
            semantic_out = self.semantic_cross_attn(
                hidden_states=h,
                semantic_cond=semantic_cond,
                semantic_mask=semantic_mask,
            )
            hidden_states = hidden_states + semantic_out

        # ----------------------------------------------------
        # 4. local conv branch
        # ----------------------------------------------------
        h = self.norm_conv(hidden_states)
        h = self._apply_scale_shift(h, mod["attn_scale"], mod["attn_shift"])
        hidden_states = hidden_states + self.local_conv(h)

        # ----------------------------------------------------
        # 5. FFN
        # ----------------------------------------------------
        h = self.norm_ffn(hidden_states)
        h = self._apply_scale_shift(h, mod["ffn_scale"], mod["ffn_shift"])
        ffn_out = self.ffn(h)
        hidden_states = hidden_states + mod["ffn_gate"].unsqueeze(1) * ffn_out

        if hidden_mask is not None:
            hidden_states = hidden_states * hidden_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        return hidden_states


# ============================================================
# v6.1 components / v6.1 组件
# ============================================================

def _v61_inverse_sigmoid(x: float, eps: float = 1e-4) -> float:
    """
    EN:
    Numerically safe inverse sigmoid for scalar initialization.

    ZH:
    用于标量初始化的安全 inverse sigmoid。
    """
    x = float(max(min(x, 1.0 - eps), eps))
    return float(torch.logit(torch.tensor(x)).item())


def _v61_apply_valid_mask(
    x: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    EN:
    Apply a bool valid mask to a sequence tensor.

    Args:
        x: (B, T, C)
        valid_mask: (B, T), True means valid.

    ZH:
    对序列张量应用有效位置 mask。

    参数：
        x: (B, T, C)
        valid_mask: (B, T)，True 表示有效位置。
    """
    if valid_mask is None:
        return x

    return x * valid_mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)


def _v61_key_padding_to_valid_mask(
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    EN:
    Convert PyTorch key_padding_mask to valid mask.

    PyTorch attention convention:
        key_padding_mask: True means PAD / ignore.

    VoiceLab internal convention:
        valid_mask: True means valid.

    ZH:
    将 PyTorch attention 使用的 key_padding_mask 转成 valid_mask。
    """
    if key_padding_mask is None:
        return None
    return ~key_padding_mask


def _v61_resolve_keep_mask(
    keep_mask: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    EN:
    Resolve optional branch keep mask to shape (B,) float tensor.

    ZH:
    将可选分支 keep mask 统一为形状 (B,) 的 float tensor。
    """
    if keep_mask is None:
        return torch.ones(batch_size, device=device, dtype=dtype)

    return keep_mask.to(device=device, dtype=dtype).view(batch_size)


def _v61_match_time_length(
    x: torch.Tensor,
    target_len: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    EN:
    Safely match sequence time length to target_len.

    This is mainly a defensive utility. In normal v6.1 usage,
    content_frame_main should already have the same length as the
    acoustic sequence.

    Args:
        x: (B, T, C)
        target_len: target time length
        mode: interpolation mode

    ZH:
    安全地将序列长度对齐到 target_len。

    这主要是防御性工具。在正常 v6.1 链路中，
    content_frame_main 应该已经与 acoustic 序列等长。
    """
    if x.shape[1] == target_len:
        return x

    if x.shape[1] <= 0:
        raise ValueError("Cannot match time length for an empty sequence.")

    x_ = x.transpose(1, 2)  # (B, C, T)
    y_ = F.interpolate(
        x_,
        size=target_len,
        mode=mode,
        align_corners=False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
    )
    return y_.transpose(1, 2)  # (B, T_target, C)


def _v61_match_mask_length(
    mask: torch.Tensor | None,
    target_len: int,
) -> torch.Tensor | None:
    """
    EN:
    Match bool mask length to target_len using nearest interpolation.

    Args:
        mask: (B, T), bool
        target_len: target time length

    ZH:
    使用 nearest 插值将 bool mask 的时间长度对齐到 target_len。
    """
    if mask is None:
        return None

    if mask.shape[1] == target_len:
        return mask

    mask_f = mask.float().unsqueeze(1)  # (B, 1, T)
    out = F.interpolate(mask_f, size=target_len, mode="nearest")
    return out.squeeze(1) > 0.5


class Stage2AdaLNModulationV61(nn.Module):
    """
    EN:
    Lightweight AdaLN-style modulation used by v6.1 blocks.

    It takes:
    - time embedding
    - optional global style embedding

    It produces:
    - scale
    - shift
    - gate

    The branch output is usually applied as:
        x = x + gate * branch_output

    ZH:
    v6.1 block 使用的轻量 AdaLN 风格调制模块。

    输入：
    - time embedding
    - 可选 global style embedding

    输出：
    - scale
    - shift
    - gate

    分支输出通常按如下方式注入：
        x = x + gate * branch_output
    """

    def __init__(
        self,
        hidden_dim: int,
        style_dim: int | None = None,
        expansion: int = 4,
        gate_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.style_dim = int(style_dim or 0)
        self.gate_init = float(gate_init)

        in_dim = self.hidden_dim + self.style_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim * expansion),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * expansion, self.hidden_dim * 3),
        )

        self._init_output(gate_init=gate_init)

    def _init_output(self, gate_init: float) -> None:
        """
        EN:
        Initialize the last projection so that:
        - scale starts near 0
        - shift starts near 0
        - gate starts near gate_init

        ZH:
        初始化最后一层，使得：
        - scale 初始接近 0
        - shift 初始接近 0
        - gate 初始接近 gate_init
        """
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            return

        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        gate_logit = _v61_inverse_sigmoid(gate_init)
        with torch.no_grad():
            last.bias[self.hidden_dim * 2 : self.hidden_dim * 3].fill_(gate_logit)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, H)
        time_emb: torch.Tensor,                  # (B, H)
        style_global: torch.Tensor | None = None # (B, C_style)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        EN:
        Return modulated x and residual gate.

        Returns:
            x_mod: (B, T, H)
            gate:  (B, 1, H)

        ZH:
        返回调制后的 x 和 residual gate。
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        time_emb = time_emb.to(device=device, dtype=dtype)

        if self.style_dim > 0:
            if style_global is None:
                style_global = torch.zeros(
                    B,
                    self.style_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                style_global = style_global.to(device=device, dtype=dtype)
                if style_global.shape[-1] != self.style_dim:
                    raise ValueError(
                        "style_global dim mismatch in Stage2AdaLNModulationV61: "
                        f"expected {self.style_dim}, got {style_global.shape[-1]}"
                    )

            cond = torch.cat([time_emb, style_global], dim=-1)
        else:
            cond = time_emb

        mod = self.net(cond)
        scale, shift, gate = mod.chunk(3, dim=-1)

        x_mod = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        gate = torch.sigmoid(gate).unsqueeze(1)

        return x_mod, gate


class Stage2ContentInjectionModuleV61(nn.Module):
    """
    EN:
    v6.1 frame-level content injection module.

    This is the main content-to-acoustic path in v6.1.
    It directly injects frame-level content condition into hidden states.

    ZH:
    v6.1 frame-level 内容主条件注入模块。

    这是 v6.1 中 content → acoustic 的主路径之一。
    它会将 frame-level content condition 直接注入 hidden states。
    """

    def __init__(
        self,
        hidden_dim: int,
        content_dim: int,
        dropout: float = 0.1,
        gate_init: float = 0.5,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.content_dim = int(content_dim)
        self.gate_init = float(gate_init)
        self.use_layer_norm = bool(use_layer_norm)

        self.content_norm = nn.LayerNorm(content_dim) if use_layer_norm else nn.Identity()

        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.gate_proj = nn.Linear(content_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_gate(gate_init=gate_init)

    def _init_gate(self, gate_init: float) -> None:
        """
        EN:
        Initialize gate projection bias to gate_init.

        ZH:
        将 gate projection 的 bias 初始化到 gate_init 附近。
        """
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, _v61_inverse_sigmoid(gate_init))

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, H)
        content_frame: torch.Tensor,           # (B, T, C_content)
        mask: torch.Tensor | None = None,      # (B, T), True for valid
    ) -> torch.Tensor:
        """
        EN:
        Inject frame-level content condition into hidden states.

        ZH:
        将 frame-level content condition 注入 hidden states。
        """
        B, T, _ = x.shape

        if content_frame is None:
            raise ValueError("content_frame is required for Stage2ContentInjectionModuleV61.")

        if content_frame.shape[0] != B:
            raise ValueError(
                "Batch size mismatch in Stage2ContentInjectionModuleV61: "
                f"x has B={B}, content_frame has B={content_frame.shape[0]}"
            )

        content_frame = _v61_match_time_length(content_frame, target_len=T, mode="linear")

        if mask is not None:
            mask = _v61_match_mask_length(mask, target_len=T)

        h = self.content_norm(content_frame)
        content_h = self.content_proj(h)
        gate = torch.sigmoid(self.gate_proj(h))

        x = x + self.dropout(gate * content_h)
        x = _v61_apply_valid_mask(x, mask)

        return x


class Stage2ShortTimeDetailModuleV61(nn.Module):
    """
    EN:
    v6.1 short-time acoustic detail modeling module.

    This module strengthens local temporal transitions:
    - syllable boundaries
    - consonant onsets
    - short local acoustic details

    ZH:
    v6.1 局部短时声学细节建模模块。

    该模块用于增强局部时间结构：
    - 音节边界
    - 辅音起始
    - 短时局部声学细节
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.residual_scale = float(residual_scale)
        self.use_layer_norm = bool(use_layer_norm)

        padding = (kernel_size // 2) * dilation

        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.pointwise_in = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=1,
        )

        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )

        self.pointwise_out = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, H)
        mask: torch.Tensor | None = None,      # (B, T), True for valid
    ) -> torch.Tensor:
        """
        EN:
        Apply short-time local detail modeling.

        ZH:
        应用局部短时细节建模。
        """
        B, T, H = x.shape
        residual = x

        h = self.norm(x)
        h = h.transpose(1, 2)  # (B, H, T)

        h = self.pointwise_in(h)  # (B, 2H, T)
        a, b = h.chunk(2, dim=1)
        h = a * torch.sigmoid(b)

        h = self.depthwise(h)

        # Defensive crop for even kernel / dilation edge cases.
        # 正常 odd kernel 下长度应保持不变；这里做防御性裁剪。
        if h.shape[-1] != T:
            h = h[..., :T]
            if h.shape[-1] < T:
                pad_len = T - h.shape[-1]
                h = F.pad(h, (0, pad_len))

        h = F.gelu(h)
        h = self.pointwise_out(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)  # (B, T, H)

        x = residual + self.residual_scale * h
        x = _v61_apply_valid_mask(x, mask)

        return x


class Stage2SemanticGuideModuleV61(nn.Module):
    """
    EN:
    v6.1 semantic guide cross-attention module.

    In v6.1, semantic guide is treated as an auxiliary guide,
    not the main content realization path. The main path should
    be frame-level content injection.

    ZH:
    v6.1 semantic guide cross-attention 模块。

    在 v6.1 中，semantic guide 被定位为辅助引导，
    不是主要内容落地路径。主要内容落地应由 frame-level
    content injection 承担。
    """

    def __init__(
        self,
        hidden_dim: int,
        semantic_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        gate_init: float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(semantic_dim)
        self.num_heads = int(num_heads)
        self.gate_init = float(gate_init)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.semantic_norm = nn.LayerNorm(semantic_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_dropout = nn.Dropout(dropout)

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self._init_gate(gate_init=gate_init)

    def _init_gate(self, gate_init: float) -> None:
        """
        EN:
        Initialize semantic guide gate to a relatively small value.

        ZH:
        将 semantic guide gate 初始化为较小值。
        """
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, _v61_inverse_sigmoid(gate_init))

    def forward(
        self,
        x: torch.Tensor,                                  # (B, T, H)
        semantic_guide: torch.Tensor | None,              # (B, T_sem, C_sem)
        semantic_key_padding_mask: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,            # (B,)
    ) -> torch.Tensor:
        """
        EN:
        Apply optional semantic guide cross-attention.

        ZH:
        应用可选 semantic guide cross-attention。
        """
        if semantic_guide is None:
            return x

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        if semantic_guide.shape[0] != B:
            raise ValueError(
                "Batch size mismatch in Stage2SemanticGuideModuleV61: "
                f"x has B={B}, semantic_guide has B={semantic_guide.shape[0]}"
            )

        keep = _v61_resolve_keep_mask(
            keep_mask=keep_mask,
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        q = self.query_norm(x)

        semantic_guide = self.semantic_norm(semantic_guide)
        kv = self.semantic_proj(semantic_guide)

        out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=semantic_key_padding_mask,
            need_weights=False,
        )

        gate = torch.sigmoid(self.gate_proj(q))
        out = out * keep[:, None, None]

        x = x + self.out_dropout(gate * out)

        return x


class Stage2DiTStyleFMBlockV61(nn.Module):
    """
    EN:
    v6.1 DiT-style Flow Matching block.

    Main order:
    1. frame-level content injection
    2. self-attention with time/style AdaLN modulation
    3. short-time local detail modeling
    4. optional semantic guide cross-attention
    5. FFN with time/style AdaLN modulation

    ZH:
    v6.1 DiT 风格 Flow Matching 主干 block。

    主流程：
    1. frame-level content 主条件注入
    2. 带 time/style AdaLN 调制的 self-attention
    3. 局部短时细节建模
    4. 可选 semantic guide cross-attention
    5. 带 time/style AdaLN 调制的 FFN
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        content_dim: int = 512,
        style_dim: int | None = None,
        semantic_dim: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        local_kernel_size: int = 5,
        local_dilation: int = 1,
        local_residual_scale: float = 0.5,
        content_gate_init: float = 0.5,
        semantic_gate_init: float = 0.3,
        attn_gate_init: float = 0.5,
        ffn_gate_init: float = 0.5,
        use_semantic_guide_cross_attn: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim or hidden_dim)
        self.semantic_dim = int(semantic_dim or hidden_dim)
        self.num_heads = int(num_heads)
        self.ff_mult = int(ff_mult)
        self.use_semantic_guide_cross_attn = bool(use_semantic_guide_cross_attn)

        # ----------------------------------------------------
        # 1. Main frame-level content injection
        # 1. frame-level content 主条件注入
        # ----------------------------------------------------
        self.content_inject = Stage2ContentInjectionModuleV61(
            hidden_dim=hidden_dim,
            content_dim=content_dim,
            dropout=dropout,
            gate_init=content_gate_init,
        )

        # ----------------------------------------------------
        # 2. Self-attention branch
        # 2. self-attention 分支
        # ----------------------------------------------------
        self.ln_self = nn.LayerNorm(hidden_dim)
        self.mod_self = Stage2AdaLNModulationV61(
            hidden_dim=hidden_dim,
            style_dim=self.style_dim,
            gate_init=attn_gate_init,
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_dropout = nn.Dropout(dropout)

        # ----------------------------------------------------
        # 3. Short-time detail branch
        # 3. 局部短时细节分支
        # ----------------------------------------------------
        self.short_time_detail = Stage2ShortTimeDetailModuleV61(
            hidden_dim=hidden_dim,
            kernel_size=local_kernel_size,
            dilation=local_dilation,
            dropout=dropout,
            residual_scale=local_residual_scale,
        )

        # ----------------------------------------------------
        # 4. Semantic guide branch
        # 4. semantic guide 辅助分支
        # ----------------------------------------------------
        self.semantic_guide = Stage2SemanticGuideModuleV61(
            hidden_dim=hidden_dim,
            semantic_dim=self.semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_init=semantic_gate_init,
        )

        # ----------------------------------------------------
        # 5. FFN branch
        # 5. FFN 分支
        # ----------------------------------------------------
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.mod_ffn = Stage2AdaLNModulationV61(
            hidden_dim=hidden_dim,
            style_dim=self.style_dim,
            gate_init=ffn_gate_init,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def _resolve_valid_mask(
        self,
        content_mask: torch.Tensor | None,
        self_key_padding_mask: torch.Tensor | None,
        target_len: int,
    ) -> torch.Tensor | None:
        """
        EN:
        Resolve valid mask from content_mask or self_key_padding_mask.

        Priority:
        1. content_mask, where True means valid
        2. self_key_padding_mask, where True means PAD

        ZH:
        从 content_mask 或 self_key_padding_mask 解析有效位置 mask。

        优先级：
        1. content_mask，True 表示有效
        2. self_key_padding_mask，True 表示 PAD
        """
        if content_mask is not None:
            return _v61_match_mask_length(content_mask, target_len=target_len)

        valid_mask = _v61_key_padding_to_valid_mask(self_key_padding_mask)
        if valid_mask is not None:
            return _v61_match_mask_length(valid_mask, target_len=target_len)

        return None

    def forward(
        self,
        x: torch.Tensor,                                      # (B, T, H)
        time_emb: torch.Tensor,                               # (B, H)
        content_frame: torch.Tensor,                          # (B, T, C_content)
        style_global: torch.Tensor | None = None,             # (B, C_style)
        semantic_guide: torch.Tensor | None = None,           # (B, T_sem, C_sem)
        self_key_padding_mask: torch.Tensor | None = None,    # (B, T), True for PAD
        content_mask: torch.Tensor | None = None,             # (B, T), True for valid
        semantic_key_padding_mask: torch.Tensor | None = None,
        semantic_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        EN:
        Forward pass of v6.1 block.

        ZH:
        v6.1 block 前向传播。
        """
        B, T, H = x.shape
        if H != self.hidden_dim:
            raise ValueError(
                "Hidden dim mismatch in Stage2DiTStyleFMBlockV61: "
                f"expected {self.hidden_dim}, got {H}"
            )

        valid_mask = self._resolve_valid_mask(
            content_mask=content_mask,
            self_key_padding_mask=self_key_padding_mask,
            target_len=T,
        )

        # ----------------------------------------------------
        # 1. Main frame-level content injection
        # 1. frame-level content 主条件注入
        # ----------------------------------------------------
        x = self.content_inject(
            x=x,
            content_frame=content_frame,
            mask=valid_mask,
        )

        # ----------------------------------------------------
        # 2. Self-attention with time/style modulation
        # 2. 带 time/style 调制的 self-attention
        # ----------------------------------------------------
        h = self.ln_self(x)
        h, attn_gate = self.mod_self(
            h,
            time_emb=time_emb,
            style_global=style_global,
        )

        self_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )

        x = x + attn_gate * self.self_dropout(self_out)
        x = _v61_apply_valid_mask(x, valid_mask)

        # ----------------------------------------------------
        # 3. Short-time local detail modeling
        # 3. 局部短时细节建模
        # ----------------------------------------------------
        x = self.short_time_detail(
            x=x,
            mask=valid_mask,
        )

        # ----------------------------------------------------
        # 4. Optional semantic guide
        # 4. 可选 semantic guide
        # ----------------------------------------------------
        if self.use_semantic_guide_cross_attn:
            x = self.semantic_guide(
                x=x,
                semantic_guide=semantic_guide,
                semantic_key_padding_mask=semantic_key_padding_mask,
                keep_mask=semantic_keep_mask,
            )
            x = _v61_apply_valid_mask(x, valid_mask)

        # ----------------------------------------------------
        # 5. FFN with time/style modulation
        # 5. 带 time/style 调制的 FFN
        # ----------------------------------------------------
        h = self.ln_ffn(x)
        h, ffn_gate = self.mod_ffn(
            h,
            time_emb=time_emb,
            style_global=style_global,
        )

        ffn_out = self.ffn(h)
        x = x + ffn_gate * ffn_out
        x = _v61_apply_valid_mask(x, valid_mask)

        return x