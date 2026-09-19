from __future__ import annotations

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
import torch.nn.functional as F


# ============================================================
# Generic helpers
# 通用工具
# ============================================================
def make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    lengths = lengths.long().clamp(min=1, max=max(int(max_len), 1))
    idx = torch.arange(int(max_len), device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def align_1d_to_length_batch(
    x: torch.Tensor,
    *,
    target_len: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Align [B, T] to [B, target_len] by interpolation.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"x must be torch.Tensor, got {type(x)}")

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"x must be [B,T] or [T], got {tuple(x.shape)}")

    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    if int(x.shape[1]) == target_len:
        return x.contiguous()

    if int(x.shape[1]) <= 1:
        return x[:, :1].repeat(1, target_len).contiguous()

    z = x.float().unsqueeze(1)
    if mode == "nearest":
        y = F.interpolate(z, size=target_len, mode="nearest")
    else:
        y = F.interpolate(z, size=target_len, mode="linear", align_corners=False)
    return y.squeeze(1).contiguous()


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(device=x.device, dtype=x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp_min(float(eps))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).abs(), mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).pow(2), mask)


def masked_z_normalize_1d(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Per-sample z-normalization for [B,T] with mask [B,T].
    """
    if x.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"x/mask must be [B,T], got x={tuple(x.shape)}, mask={tuple(mask.shape)}")

    mask_f = mask.to(device=x.device, dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask_f).sum(dim=1, keepdim=True) / denom
    var = ((x - mean).pow(2) * mask_f).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + float(eps))
    y = (x - mean) / std
    return y * mask_f


# ============================================================
# Energy loss
# 能量损失
# ============================================================
def log_mel_to_log_energy_differentiable(log_mel: torch.Tensor) -> torch.Tensor:
    """
    Convert log power-mel [B,T,M] to differentiable log-energy [B,T].
    """
    if log_mel.ndim != 3:
        raise ValueError(f"log_mel must be [B,T,M], got {tuple(log_mel.shape)}")
    n_mels = max(int(log_mel.shape[-1]), 1)
    return torch.logsumexp(log_mel.float(), dim=-1) - float(torch.log(torch.tensor(n_mels)).item())


def compute_v66_energy_loss(
    *,
    pred_mel: torch.Tensor,
    target_energy: torch.Tensor | None,
    target_lengths: torch.Tensor,
    normalize: bool = True,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute energy-contour loss between predicted mel and cached target_energy.
    """
    zero = pred_mel.new_tensor(0.0)
    aux: dict[str, torch.Tensor] = {
        "v66_energy_available": zero.detach(),
    }

    if target_energy is None or not torch.is_tensor(target_energy):
        return zero, aux

    B, T, _ = pred_mel.shape
    device = pred_mel.device
    lengths = target_lengths.long().to(device).clamp(min=1, max=T)
    mask = make_length_mask(lengths, T)

    pred_energy = log_mel_to_log_energy_differentiable(pred_mel)
    target = target_energy.to(device=device, dtype=pred_energy.dtype)
    target = align_1d_to_length_batch(target, target_len=T, mode="linear")

    if normalize:
        pred_energy = masked_z_normalize_1d(pred_energy, mask)
        target = masked_z_normalize_1d(target, mask)

    if str(loss_type).lower() == "mse":
        loss = masked_mse(pred_energy, target, mask)
    else:
        loss = masked_l1(pred_energy, target, mask)

    aux.update(
        {
            "v66_energy_available": pred_mel.new_tensor(1.0).detach(),
            "v66_energy_loss": loss.detach(),
            "v66_pred_energy_mean": masked_mean(pred_energy.detach(), mask).detach(),
            "v66_target_energy_mean": masked_mean(target.detach(), mask).detach(),
        }
    )
    return loss, aux


# ============================================================
# F0 loss
# F0 损失
# ============================================================
def compute_v66_f0_loss(
    *,
    pred_f0: torch.Tensor | None,
    target_f0: torch.Tensor | None,
    target_f0_voiced_mask: torch.Tensor | None,
    target_lengths: torch.Tensor,
    normalize: bool = True,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute an auxiliary F0 contour loss.

    pred_f0 is expected to be [B,T]. It is usually produced by a small auxiliary
    head from predicted mel frames.
    """
    if pred_f0 is None:
        base = target_lengths.new_tensor(0.0, dtype=torch.float32)
    else:
        base = pred_f0.new_tensor(0.0)

    aux: dict[str, torch.Tensor] = {
        "v66_f0_available": base.detach(),
    }

    if pred_f0 is None or target_f0 is None or not torch.is_tensor(target_f0):
        return base, aux

    if pred_f0.ndim == 3 and pred_f0.shape[-1] == 1:
        pred_f0 = pred_f0.squeeze(-1)
    if pred_f0.ndim != 2:
        raise ValueError(f"pred_f0 must be [B,T], got {tuple(pred_f0.shape)}")

    B, T = pred_f0.shape
    device = pred_f0.device
    lengths = target_lengths.long().to(device).clamp(min=1, max=T)
    len_mask = make_length_mask(lengths, T)

    target = target_f0.to(device=device, dtype=pred_f0.dtype)
    target = align_1d_to_length_batch(target, target_len=T, mode="linear")

    if target_f0_voiced_mask is not None and torch.is_tensor(target_f0_voiced_mask):
        voiced = target_f0_voiced_mask.to(device=device).float()
        voiced = align_1d_to_length_batch(voiced, target_len=T, mode="nearest") > 0.5
    else:
        voiced = target > 1.0

    mask = len_mask & voiced
    if int(mask.long().sum().item()) <= 0:
        return base, aux

    # Target is Hz. Use log-Hz for stability; pred_f0 is unconstrained and trained
    # against normalized log-F0 if normalize=True.
    target_log = torch.log(target.clamp_min(1.0))
    pred = pred_f0.float()

    if normalize:
        target_log = masked_z_normalize_1d(target_log, mask)
        pred = masked_z_normalize_1d(pred, mask)

    if str(loss_type).lower() == "mse":
        loss = masked_mse(pred, target_log, mask)
    else:
        loss = masked_l1(pred, target_log, mask)

    aux.update(
        {
            "v66_f0_available": pred_f0.new_tensor(1.0).detach(),
            "v66_f0_loss": loss.detach(),
            "v66_f0_voiced_ratio": mask.float().mean().detach(),
        }
    )
    return loss, aux


# ============================================================
# Speaker embedding loss
# 说话人嵌入损失
# ============================================================
def compute_v66_speaker_loss(
    *,
    pred_speaker_embedding: torch.Tensor | None,
    target_speaker_embedding: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Cosine distance loss for speaker embeddings.
    """
    if pred_speaker_embedding is None:
        if target_speaker_embedding is not None and torch.is_tensor(target_speaker_embedding):
            zero = target_speaker_embedding.new_tensor(0.0)
        else:
            zero = torch.tensor(0.0)
        return zero, {"v66_speaker_available": zero.detach()}

    zero = pred_speaker_embedding.new_tensor(0.0)
    aux: dict[str, torch.Tensor] = {"v66_speaker_available": zero.detach()}

    if target_speaker_embedding is None or not torch.is_tensor(target_speaker_embedding):
        return zero, aux

    pred = pred_speaker_embedding.float()
    target = target_speaker_embedding.to(device=pred.device, dtype=pred.dtype)

    if target.ndim == 1:
        target = target.unsqueeze(0)
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"speaker embeddings must be [B,D], got pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.shape != target.shape:
        raise ValueError(f"speaker embedding shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    pred_n = F.normalize(pred, dim=-1, eps=1e-6)
    target_n = F.normalize(target, dim=-1, eps=1e-6)
    cosine = (pred_n * target_n).sum(dim=-1)
    loss = (1.0 - cosine).mean()

    aux.update(
        {
            "v66_speaker_available": pred.new_tensor(1.0).detach(),
            "v66_speaker_loss": loss.detach(),
            "v66_speaker_cosine_mean": cosine.detach().mean(),
        }
    )
    return loss, aux
