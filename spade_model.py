"""
spade_model.py — SPADE generator for semantic layout → RGB (Park et al., CVPR 2019).

Same I/O convention as gan_unet_model.UNetGenerator:
  label : (B, num_classes, H, W)  one-hot in [−1, 1]
  output: (B, 3, H, W)            RGB in [−1, 1]

The semantic map is re-used at every scale via SPADE (spatially-adaptive
normalization), which is stronger than concatenating labels into a U-Net encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """Instance norm without affine params; scale/shift predicted from seg map."""

    def __init__(self, norm_nc: int, label_nc: int, hidden: int = 128):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        seg = F.interpolate(seg, size=x.shape[2:], mode='nearest')
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1.0 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin: int, fout: int, label_nc: int):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.norm_s = SPADE(fin, label_nc)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 0.2, inplace=True)

    def shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg))
        return x

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self._act(self.norm_0(x, seg)))
        dx = self.conv_1(self._act(self.norm_1(dx, seg)))
        return x_s + dx


class SPADEGenerator(nn.Module):
    """
    SPADE generator for fixed (H, W) = (img_h, img_w).

    Five upsampling stages: latent grid is H/32 × W/32 (8×16 for 256×512).
    """

    def __init__(
        self,
        label_nc: int = 4,
        ngf: int = 64,
        img_h: int = 256,
        img_w: int = 512,
        num_up_layers: int = 5,
    ):
        super().__init__()
        self.label_nc = label_nc
        self.sh = img_h // (2 ** num_up_layers)
        self.sw = img_w // (2 ** num_up_layers)
        if self.sh < 1 or self.sw < 1:
            raise ValueError(
                f'img_h={img_h}, img_w={img_w} too small for num_up_layers={num_up_layers}'
            )

        nf = ngf
        nfc = 16 * nf

        self.fc = nn.Conv2d(label_nc, nfc, kernel_size=3, padding=1)
        self.head_0 = SPADEResnetBlock(nfc, nfc, label_nc)
        self.G_middle_0 = SPADEResnetBlock(nfc, nfc, label_nc)
        self.G_middle_1 = SPADEResnetBlock(nfc, nfc, label_nc)

        self.up_0 = SPADEResnetBlock(nfc, 8 * nf, label_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, label_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, label_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, label_nc)

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

    @staticmethod
    def _labels_to_seg(label: torch.Tensor) -> torch.Tensor:
        """One-hot [−1, 1] → soft one-hot [0, 1] for SPADE MLPs."""
        return (label + 1.0) * 0.5

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        seg = self._labels_to_seg(label)

        x = F.interpolate(seg, size=(self.sh, self.sw), mode='nearest')
        x = self.fc(x)
        x = self.head_0(x, seg)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_0(x, seg)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_1(x, seg)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_2(x, seg)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_3(x, seg)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv_img(F.leaky_relu(x, 0.2, inplace=True))
        return torch.tanh(x)
