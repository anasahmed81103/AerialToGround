"""Class-weighted focal + Dice for Phase 2 polar trainer (does not change train_crossnet)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_ce(logits: torch.Tensor, target: torch.Tensor,
             gamma: float = 2.0, ignore_index: int = -1,
             class_weight: torch.Tensor | None = None) -> torch.Tensor:
    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    logp = F.log_softmax(logits, dim=1)
    tgt = target.clamp(0, logits.size(1) - 1)
    logpt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    loss = -(1.0 - pt).pow(gamma) * logpt
    if class_weight is not None:
        w = class_weight.to(logits.device, dtype=loss.dtype)[tgt]
        loss = loss * w
    return loss[valid].mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              num_classes: int, ignore_index: int = -1,
              smooth: float = 1.0,
              class_weight: torch.Tensor | None = None) -> torch.Tensor:
    valid = target != ignore_index
    tgt = target.clamp(0, num_classes - 1)
    pred = F.softmax(logits, dim=1)
    oh = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()
    pred = pred * mask
    oh = oh * mask
    dims = (0, 2, 3)
    inter = (pred * oh).sum(dim=dims)
    den = pred.sum(dim=dims) + oh.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (den + smooth)
    one_minus = 1.0 - dice
    if class_weight is None:
        return one_minus.mean()
    w = class_weight.to(logits.device, dtype=one_minus.dtype)
    w = w / w.sum()
    return (w * one_minus).sum()
