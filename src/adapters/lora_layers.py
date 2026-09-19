from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
from dataclasses import dataclass
from typing import Iterable

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAApplyReport:
    enabled: bool
    target_mode: str
    rank: int
    alpha: float
    dropout: float
    init_scale: float
    replaced_module_names: list[str]

    @property
    def num_replaced(self) -> int:
        return int(len(self.replaced_module_names))


class LoRALinear(nn.Module):
    """
    EN:
    Drop-in LoRA wrapper for nn.Linear that keeps original state_dict keys
    `weight` and `bias`, so old checkpoints can still load into the base path.

    ZH:
    nn.Linear 的 LoRA 替换层。保留原始 `weight` / `bias` 参数名，
    因此旧 checkpoint 仍可加载到基础线性层。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        freeze_base: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(int(rank), 1)
        self.init_scale = float(init_scale)
        self.freeze_base = bool(freeze_base)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.rank > 0:
            self.lora_A = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
            self.lora_B = nn.Linear(self.rank, self.out_features, bias=False, **factory_kwargs)
        else:
            self.lora_A = None
            self.lora_B = None

        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.reset_parameters()

        if self.freeze_base:
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / max(fan_in, 1) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_A is not None and self.lora_B is not None:
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=float(self.init_scale))
            nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        init_scale: float,
        freeze_base: bool = True,
    ) -> "LoRALinear":
        out = cls(
            in_features=int(linear.in_features),
            out_features=int(linear.out_features),
            bias=linear.bias is not None,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            init_scale=float(init_scale),
            freeze_base=bool(freeze_base),
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        with torch.no_grad():
            out.weight.copy_(linear.weight.detach())
            if linear.bias is not None and out.bias is not None:
                out.bias.copy_(linear.bias.detach())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.rank <= 0 or self.lora_A is None or self.lora_B is None:
            return base
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + update


# ============================================================
# Target selection
# 目标选择
# ============================================================
def _is_core_bootstrapper_lora_target(name: str) -> bool:
    n = str(name)

    # Query builder projections.
    if n.startswith("query_builder.content_proj"):
        return True
    if n.startswith("query_builder.style_to_query"):
        return True
    if n.startswith("query_builder.time_proj"):
        return True
    if n == "query_builder.position_proj":
        return True

    # Memory projections.
    if n.startswith("semantic_proj"):
        return True
    if n.startswith("text_proj"):
        return True

    # FFN blocks inside Conformer bootstrapper.
    if ".ff1.in_proj" in n or ".ff1.out_proj" in n:
        return True
    if ".ff2.in_proj" in n or ".ff2.out_proj" in n:
        return True

    # Output mel head.
    if n == "to_mel":
        return True

    return False


def should_apply_lora_to_linear(name: str, target_mode: str) -> bool:
    mode = str(target_mode).strip().lower()
    if mode in {"core", "bootstrapper_core"}:
        return _is_core_bootstrapper_lora_target(name)
    if mode in {"all_linear", "all"}:
        return True
    raise ValueError(f"Unsupported bootstrapper_lora_target: {target_mode}")


def _set_child_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = str(dotted_name).split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_named_linears(
    module: nn.Module,
    *,
    target_mode: str = "core",
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05,
    init_scale: float = 0.01,
    freeze_base: bool = True,
) -> LoRAApplyReport:
    if module is None:
        return LoRAApplyReport(
            enabled=False,
            target_mode=str(target_mode),
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            init_scale=float(init_scale),
            replaced_module_names=[],
        )

    if int(rank) <= 0:
        return LoRAApplyReport(
            enabled=False,
            target_mode=str(target_mode),
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            init_scale=float(init_scale),
            replaced_module_names=[],
        )

    replacements: list[tuple[str, nn.Linear]] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and should_apply_lora_to_linear(name, target_mode):
            replacements.append((name, child))

    replaced_names: list[str] = []
    for name, linear in replacements:
        lora = LoRALinear.from_linear(
            linear,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            init_scale=float(init_scale),
            freeze_base=bool(freeze_base),
        )
        _set_child_module(module, name, lora)
        replaced_names.append(name)

    return LoRAApplyReport(
        enabled=bool(replaced_names),
        target_mode=str(target_mode),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        init_scale=float(init_scale),
        replaced_module_names=replaced_names,
    )


def iter_lora_parameter_names(module: nn.Module) -> Iterable[str]:
    for name, _param in module.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            yield name


def count_lora_parameters(module: nn.Module) -> int:
    total = 0
    for name, p in module.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            total += int(p.numel())
    return int(total)
