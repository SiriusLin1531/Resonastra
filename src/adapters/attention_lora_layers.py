from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionLoRAApplyReport:
    enabled: bool
    target_mode: str
    rank: int
    alpha: float
    dropout: float
    gate_init: float
    replaced_module_names: list[str]

    @property
    def num_replaced(self) -> int:
        return int(len(self.replaced_module_names))


class LoRAMultiheadAttention(nn.Module):
    """
    EN:
    Drop-in replacement for nn.MultiheadAttention with LoRA updates on Q/K/V/O.

    Important compatibility point:
    This wrapper keeps the original parameter names `in_proj_weight`,
    `in_proj_bias`, and `out_proj.*` directly under the module, so existing
    checkpoints with keys like `blocks.0.self_attn.attn.in_proj_weight` can
    still load into the base attention path.

    ZH:
    带 Q/K/V/O LoRA 的 nn.MultiheadAttention 替换层。

    兼容性重点：
    本模块保留原始 `in_proj_weight`、`in_proj_bias`、`out_proj.*` 参数名，
    因此旧 checkpoint 中的 `blocks.0.self_attn.attn.in_proj_weight` 等 key
    仍可加载到基础 attention 路径。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rank: int = 4,
        alpha: float = 8.0,
        lora_dropout: float = 0.05,
        gate_init: float = -3.0,
        enable_q: bool = True,
        enable_k: bool = True,
        enable_v: bool = True,
        enable_o: bool = True,
        freeze_base: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if not bool(batch_first):
            raise ValueError("LoRAMultiheadAttention currently requires batch_first=True.")
        if int(embed_dim) % int(num_heads) != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.batch_first = bool(batch_first)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(int(rank), 1)
        self.enable_q = bool(enable_q)
        self.enable_k = bool(enable_k)
        self.enable_v = bool(enable_v)
        self.enable_o = bool(enable_o)
        self.freeze_base = bool(freeze_base)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_proj_weight = nn.Parameter(
            torch.empty(self.embed_dim * 3, self.embed_dim, **factory_kwargs)
        )
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(self.embed_dim * 3, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)

        drop = float(lora_dropout)
        self.lora_dropout = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        self.attention_lora_gate = nn.Parameter(torch.tensor(float(gate_init), **factory_kwargs))

        if self.rank > 0 and self.enable_q:
            self.lora_q_A = nn.Linear(self.embed_dim, self.rank, bias=False, **factory_kwargs)
            self.lora_q_B = nn.Linear(self.rank, self.embed_dim, bias=False, **factory_kwargs)
        else:
            self.lora_q_A = None
            self.lora_q_B = None

        if self.rank > 0 and self.enable_k:
            self.lora_k_A = nn.Linear(self.embed_dim, self.rank, bias=False, **factory_kwargs)
            self.lora_k_B = nn.Linear(self.rank, self.embed_dim, bias=False, **factory_kwargs)
        else:
            self.lora_k_A = None
            self.lora_k_B = None

        if self.rank > 0 and self.enable_v:
            self.lora_v_A = nn.Linear(self.embed_dim, self.rank, bias=False, **factory_kwargs)
            self.lora_v_B = nn.Linear(self.rank, self.embed_dim, bias=False, **factory_kwargs)
        else:
            self.lora_v_A = None
            self.lora_v_B = None

        if self.rank > 0 and self.enable_o:
            self.lora_o_A = nn.Linear(self.embed_dim, self.rank, bias=False, **factory_kwargs)
            self.lora_o_B = nn.Linear(self.rank, self.embed_dim, bias=False, **factory_kwargs)
        else:
            self.lora_o_A = None
            self.lora_o_B = None

        self.reset_parameters()

        if self.freeze_base:
            self.in_proj_weight.requires_grad_(False)
            if self.in_proj_bias is not None:
                self.in_proj_bias.requires_grad_(False)
            self.out_proj.weight.requires_grad_(False)
            if self.out_proj.bias is not None:
                self.out_proj.bias.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        for a_name, b_name in [
            ("lora_q_A", "lora_q_B"),
            ("lora_k_A", "lora_k_B"),
            ("lora_v_A", "lora_v_B"),
            ("lora_o_A", "lora_o_B"),
        ]:
            a = getattr(self, a_name, None)
            b = getattr(self, b_name, None)
            if a is not None and b is not None:
                nn.init.normal_(a.weight, mean=0.0, std=0.01)
                nn.init.zeros_(b.weight)

    @classmethod
    def from_multihead_attention(
        cls,
        attn: nn.MultiheadAttention,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        lora_dropout: float = 0.05,
        gate_init: float = -3.0,
        enable_q: bool = True,
        enable_k: bool = True,
        enable_v: bool = True,
        enable_o: bool = True,
        freeze_base: bool = True,
    ) -> "LoRAMultiheadAttention":
        out = cls(
            embed_dim=int(attn.embed_dim),
            num_heads=int(attn.num_heads),
            dropout=float(attn.dropout),
            bias=attn.in_proj_bias is not None,
            batch_first=bool(attn.batch_first),
            rank=int(rank),
            alpha=float(alpha),
            lora_dropout=float(lora_dropout),
            gate_init=float(gate_init),
            enable_q=bool(enable_q),
            enable_k=bool(enable_k),
            enable_v=bool(enable_v),
            enable_o=bool(enable_o),
            freeze_base=bool(freeze_base),
            dtype=attn.in_proj_weight.dtype,
            device=attn.in_proj_weight.device,
        )
        with torch.no_grad():
            out.in_proj_weight.copy_(attn.in_proj_weight.detach())
            if attn.in_proj_bias is not None and out.in_proj_bias is not None:
                out.in_proj_bias.copy_(attn.in_proj_bias.detach())
            out.out_proj.weight.copy_(attn.out_proj.weight.detach())
            if attn.out_proj.bias is not None and out.out_proj.bias is not None:
                out.out_proj.bias.copy_(attn.out_proj.bias.detach())
        return out

    def _lora_update(
        self,
        x: torch.Tensor,
        A: nn.Linear | None,
        B: nn.Linear | None,
    ) -> torch.Tensor:
        if self.rank <= 0 or A is None or B is None:
            return torch.zeros_like(x)
        gate = torch.sigmoid(self.attention_lora_gate).to(device=x.device, dtype=x.dtype)
        return gate * self.scaling * B(A(self.lora_dropout(x)))

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        B, T, _D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Dh] -> [B, T, D]
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.batch_first:
            raise RuntimeError("LoRAMultiheadAttention only supports batch_first=True.")
        if is_causal:
            raise NotImplementedError("is_causal=True is not used in VoiceLab bootstrapper and is unsupported here.")
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query/key/value must be [B, T, D] tensors.")

        q_w, k_w, v_w = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is not None:
            q_b, k_b, v_b = self.in_proj_bias.chunk(3, dim=0)
        else:
            q_b = k_b = v_b = None

        q = F.linear(query, q_w, q_b) + self._lora_update(query, self.lora_q_A, self.lora_q_B)
        k = F.linear(key, k_w, k_b) + self._lora_update(key, self.lora_k_A, self.lora_k_B)
        v = F.linear(value, v_w, v_b) + self._lora_update(value, self.lora_v_A, self.lora_v_B)

        qh = self._shape_heads(q)
        kh = self._shape_heads(k)
        vh = self._shape_heads(v)

        scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype).view(1, 1, *attn_mask.shape)
            elif attn_mask.dim() == 3:
                B = query.shape[0]
                scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype).view(B, self.num_heads, query.shape[1], key.shape[1])
            else:
                raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=scores.device, dtype=torch.bool).view(query.shape[0], 1, 1, key.shape[1])
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1).to(dtype=qh.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        context = torch.matmul(attn_weights, vh)
        context = self._merge_heads(context)

        out = self.out_proj(context)
        out = out + self._lora_update(context, self.lora_o_A, self.lora_o_B)

        if not need_weights:
            return out, None

        weights = attn_weights
        if average_attn_weights:
            weights = weights.mean(dim=1)
        return out, weights


def _set_child_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = str(dotted_name).split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def should_apply_attention_lora(name: str, target_mode: str) -> bool:
    n = str(name)
    mode = str(target_mode).strip().lower()

    is_self = ".self_attn.attn" in n or n.endswith("self_attn.attn")
    is_semantic = ".semantic_cross.attn" in n or n.endswith("semantic_cross.attn")
    is_text = ".text_cross.attn" in n or n.endswith("text_cross.attn")
    is_cross = is_semantic or is_text

    if mode in {"all", "all_attention"}:
        return is_self or is_cross
    if mode in {"cross", "cross_only", "semantic_text"}:
        return is_cross
    if mode in {"self", "self_only"}:
        return is_self
    if mode in {"semantic", "semantic_cross"}:
        return is_semantic
    if mode in {"text", "text_cross"}:
        return is_text

    raise ValueError(f"Unsupported bootstrapper_attention_lora_target: {target_mode}")


def apply_attention_lora_to_named_mha(
    module: nn.Module,
    *,
    target_mode: str = "cross_only",
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05,
    gate_init: float = -3.0,
    enable_q: bool = True,
    enable_k: bool = True,
    enable_v: bool = True,
    enable_o: bool = True,
    freeze_base: bool = True,
) -> AttentionLoRAApplyReport:
    if module is None or int(rank) <= 0:
        return AttentionLoRAApplyReport(
            enabled=False,
            target_mode=str(target_mode),
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            gate_init=float(gate_init),
            replaced_module_names=[],
        )

    replacements: list[tuple[str, nn.MultiheadAttention]] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if isinstance(child, LoRAMultiheadAttention):
            continue
        if isinstance(child, nn.MultiheadAttention) and should_apply_attention_lora(name, target_mode):
            replacements.append((name, child))

    replaced: list[str] = []
    for name, attn in replacements:
        lora_attn = LoRAMultiheadAttention.from_multihead_attention(
            attn,
            rank=int(rank),
            alpha=float(alpha),
            lora_dropout=float(dropout),
            gate_init=float(gate_init),
            enable_q=bool(enable_q),
            enable_k=bool(enable_k),
            enable_v=bool(enable_v),
            enable_o=bool(enable_o),
            freeze_base=bool(freeze_base),
        )
        _set_child_module(module, name, lora_attn)
        replaced.append(name)

    return AttentionLoRAApplyReport(
        enabled=bool(replaced),
        target_mode=str(target_mode),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        gate_init=float(gate_init),
        replaced_module_names=replaced,
    )


def count_attention_lora_parameters(module: nn.Module) -> int:
    total = 0
    for name, p in module.named_parameters():
        if any(token in name for token in [
            "lora_q_A", "lora_q_B",
            "lora_k_A", "lora_k_B",
            "lora_v_A", "lora_v_B",
            "lora_o_A", "lora_o_B",
            "attention_lora_gate",
        ]):
            total += int(p.numel())
    return int(total)
