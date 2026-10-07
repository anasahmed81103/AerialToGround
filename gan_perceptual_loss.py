"""
VGG perceptual loss for Stage 2 (SPADE / Pix2Pix), images in [−1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """L1 on VGG19 relu layers (common SPADE / pix2pixHD setting)."""

    def __init__(self, layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg[9:18])) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg[18:27]))  # relu4_1
        self.weights = layer_weights
        for p in self.parameters():
            p.requires_grad = False

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer('_mean', mean)
        self.register_buffer('_std', std)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        return (x - self._mean) / self._std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x_p = self._prep(pred)
        x_t = self._prep(target)
        loss = pred.new_zeros(())
        for slc, w in zip(
            (self.slice1, self.slice2, self.slice3, self.slice4),
            self.weights,
        ):
            x_p = slc(x_p)
            with torch.no_grad():
                x_t = slc(x_t)
            loss = loss + w * torch.nn.functional.l1_loss(x_p, x_t)
        return loss
