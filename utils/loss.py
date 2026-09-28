from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def total_variation_loss(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean anisotropic total variation loss for BCHW images/textures.

    If a UV mask is provided, only neighboring texel pairs inside the mask
    contribute to the average.
    """
    if image.ndim != 4:
        raise ValueError(f"`image` must be BCHW, got shape {tuple(image.shape)}")

    dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()

    if mask is None:
        return (dx.sum() + dy.sum()) / (dx.numel() + dy.numel())

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError(f"`mask` must be CHW or BCHW, got shape {tuple(mask.shape)}")
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and image.shape[1] > 1:
        mask = mask.expand(-1, image.shape[1], -1, -1)
    if mask.shape != image.shape:
        raise ValueError(
            f"`mask` shape {tuple(mask.shape)} must broadcast to image shape {tuple(image.shape)}"
        )

    mask = mask.to(device=image.device, dtype=image.dtype)
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    numerator = (dx * mask_x).sum() + (dy * mask_y).sum()
    denominator = (mask_x.sum() + mask_y.sum()).clamp_min(1.0)
    return numerator / denominator


def lora_regularization(unet) -> torch.Tensor:
    reg = None
    for name, parameter in unet.named_parameters():
        if "lora_" not in name or not parameter.requires_grad:
            continue
        current = parameter.float().pow(2).mean()
        reg = current if reg is None else reg + current
    if reg is None:
        raise ValueError("No trainable LoRA parameters found on the UNet.")
    return reg


def margin_detection_loss(
    reward_values: torch.Tensor,
    margin: float,
    loss_type: str = "squared_hinge",
    softplus_beta: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-view margin loss on detection confidence.

    reward_values: (V,) per-view "confidence-like" tensor; lower = better attack.
    Returns (loss, fraction_below_margin) where fraction_below_margin is a logging
    signal for how saturated the margin is.
    """
    excess = reward_values - margin  # positive => still over-confident => need attack

    if loss_type == "linear_hinge":
        per_view = torch.clamp(excess, min=0.0)
    elif loss_type == "squared_hinge":
        per_view = torch.clamp(excess, min=0.0).pow(2)
    elif loss_type == "softplus":
        # smooth hinge: ~excess for excess>>0, ~0 for excess<<0, smooth at boundary
        per_view = torch.nn.functional.softplus(softplus_beta * excess) / softplus_beta
    else:
        raise ValueError(f"Unknown margin loss type: {loss_type}")

    loss = per_view.mean()
    with torch.no_grad():
        frac_below = (reward_values < margin).float().mean()
    return loss, frac_below


@contextmanager
def disable_lora_context(unet):
    """
    Context manager that temporarily disables LoRA adapters on a diffusers
    UNet (i.e. one whose adapters were attached via `unet.add_adapter(...)`,
    so the model uses `PeftAdapterMixin`, NOT `peft.PeftModel`).

    Why this helper exists:
      - `peft.PeftModel.disable_adapter()` IS a context manager.
      - `diffusers.loaders.PeftAdapterMixin.disable_adapters()` (plural) is
        a regular method without `__enter__/__exit__`. Using `with
        unet.disable_adapter():` therefore fails with AttributeError on a
        diffusers UNet that received its LoRA via `add_adapter`.

    This wrapper unifies both code paths and guarantees re-enable on exit.
    """
    # Prefer the diffusers-style API; fall back to peft-style if a real
    # PeftModel is ever passed in.
    if hasattr(unet, "disable_adapters") and hasattr(unet, "enable_adapters"):
        unet.disable_adapters()
        try:
            yield
        finally:
            unet.enable_adapters()
    elif hasattr(unet, "disable_adapter"):
        # peft.PeftModel — its disable_adapter() is itself a context manager.
        with unet.disable_adapter():
            yield
    else:
        raise AttributeError(
            "UNet does not expose disable_adapters / disable_adapter. "
            "Ensure LoRA was attached via `unet.add_adapter(LoraConfig(...))` "
            "(diffusers >= 0.27 with peft installed)."
        )

def diversity_repulsion_loss(
    cur_feat: torch.Tensor,
    bank_feats: Optional[torch.Tensor],
    margin: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    跨 prompt 多样性排斥（memory-bank 版）。

    把当前纹理表征推离 memory bank 里【不同 prompt】的历史表征：相似度超过
    margin（太像）才罚，hinge 有界排斥——不无限推远（无限推会毁攻击/自然），
    只要求"不要太像"。

    ⚠️ 与颜色 anchor 不同：结构/整体表征的排斥和攻击【直接对抗】（因为"卵石阵"
    这类崩塌结构往往就是攻击最优）。margin/weight 要保守，盯着攻击别塌。

    Args:
        cur_feat:   (1, D) 当前纹理的归一化表征（有梯度）。
        bank_feats: (M, D) memory bank 里【不同 prompt】的归一化表征（detached）。
                    None 或 M=0 时返回 0（保持梯度连接，backward 安全）。
        margin:     cosine 相似度阈值，sim>margin 才推。崩塌时 sim≈0.9+。
    Returns:
        (loss, max_sim)：max_sim 是与库中最相似项的相似度（detached，日志用，
        越高=越像别的 prompt=越崩）。
    """
    if bank_feats is None or bank_feats.numel() == 0:
        z = cur_feat.sum() * 0.0
        return z, z.detach()
    sim = (cur_feat * bank_feats).sum(dim=-1)  # (M,) cosine（均已归一化）
    rep = F.relu(sim - margin)
    return rep.mean(), sim.max().detach()
