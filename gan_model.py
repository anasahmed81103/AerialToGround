"""
gan_model.py  —  Pix2Pix-style GAN for semantic-map → ground-RGB synthesis
===========================================================================
Architecture
  Generator    : 8-stage U-Net  (ngf=64, instance norm, dropout on first 3 decoder blocks)
  Discriminator: 4-layer 70×70 PatchGAN  (ndf=64, instance norm)

Design choices for Quadro P1000 (4 GB VRAM)
  - Training resolution 256×512 (fits in < 2 GB with AMP + batch=1)
  - Instance norm instead of batch norm (works with batch_size=1)
  - LSGAN objective (more stable than vanilla BCE, no mode collapse)
  - L1 loss weight λ=100 for sharp reconstructions (Isola et al. 2017)

Input / output convention
  label  : (B, num_classes, H, W)  one-hot, normalised to [−1, 1]
  image  : (B, 3, H, W)            RGB,     normalised to [−1, 1]
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _DownBlock(nn.Module):
    """Encoder block: Conv(k=4, s=2, p=1) → [InstanceNorm] → LeakyReLU(0.2)."""
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                      bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Decoder block: ConvTranspose(k=4, s=2, p=1) → InstanceNorm → ReLU → [Dropout]."""
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ──────────────────────────────────────────────────────────────────────────────
# Generator — U-Net with 8 downsampling stages
# ──────────────────────────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    Pix2Pix U-Net generator.

    Spatial flow for H=256, W=512 (each encoder block halves both dims):
        Input  (B, in_ch, 256, 512)
        e1 →   (B, ngf,    128, 256)    [no norm on first block]
        e2 →   (B, ngf×2,   64, 128)
        e3 →   (B, ngf×4,   32,  64)
        e4 →   (B, ngf×8,   16,  32)
        e5 →   (B, ngf×8,    8,  16)
        e6 →   (B, ngf×8,    4,   8)
        e7 →   (B, ngf×8,    2,   4)
        btl →  (B, ngf×8,    1,   2)   [bottleneck, no norm]
        d1  →  (B, ngf×8,    2,   4)   [dropout]  concat e7 → 1024 ch
        d2  →  (B, ngf×8,    4,   8)   [dropout]  concat e6 → 1024 ch
        d3  →  (B, ngf×8,    8,  16)   [dropout]  concat e5 → 1024 ch
        d4  →  (B, ngf×8,   16,  32)              concat e4 → 1024 ch
        d5  →  (B, ngf×4,   32,  64)              concat e3 →  512 ch
        d6  →  (B, ngf×2,   64, 128)              concat e2 →  256 ch
        d7  →  (B, ngf,    128, 256)              concat e1 →  128 ch
        out →  (B, 3,      256, 512)   Tanh
    """

    def __init__(self, in_ch: int = 4, ngf: int = 64):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.e1 = _DownBlock(in_ch,  ngf,    use_norm=False)
        self.e2 = _DownBlock(ngf,    ngf*2)
        self.e3 = _DownBlock(ngf*2,  ngf*4)
        self.e4 = _DownBlock(ngf*4,  ngf*8)
        self.e5 = _DownBlock(ngf*8,  ngf*8)
        self.e6 = _DownBlock(ngf*8,  ngf*8)
        self.e7 = _DownBlock(ngf*8,  ngf*8)

        # ── Bottleneck (no norm; spatial size 1×2) ────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # ── Decoder (concat channels doubled by skip connections) ─────────────
        self.d1 = _UpBlock(ngf*8,  ngf*8, dropout=True)   # in: 512  → 2×4
        self.d2 = _UpBlock(ngf*16, ngf*8, dropout=True)   # in: 1024 → 4×8
        self.d3 = _UpBlock(ngf*16, ngf*8, dropout=True)   # in: 1024 → 8×16
        self.d4 = _UpBlock(ngf*16, ngf*8)                 # in: 1024 → 16×32
        self.d5 = _UpBlock(ngf*16, ngf*4)                 # in: 1024 → 32×64
        self.d6 = _UpBlock(ngf*8,  ngf*2)                 # in: 512  → 64×128
        self.d7 = _UpBlock(ngf*4,  ngf)                   # in: 256  → 128×256
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )                                                  # in: 128  → 256×512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        b  = self.bottleneck(e7)

        d = self.d1(b)
        d = self.d2(torch.cat([d, e7], dim=1))
        d = self.d3(torch.cat([d, e6], dim=1))
        d = self.d4(torch.cat([d, e5], dim=1))
        d = self.d5(torch.cat([d, e4], dim=1))
        d = self.d6(torch.cat([d, e3], dim=1))
        d = self.d7(torch.cat([d, e2], dim=1))
        return self.out_conv(torch.cat([d, e1], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator — 70×70 PatchGAN
# ──────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    4-layer PatchGAN discriminator with a ~70×70 pixel receptive field.

    The condition (semantic label) and the image (real or generated) are
    concatenated channel-wise before being fed in, so the discriminator
    judges whether the image matches the given label.

    Input : (B, in_ch + 3, H, W)  — [label ‖ image] concatenated
    Output: (B, 1, H', W')        — per-patch real/fake score (no sigmoid;
                                     LSGAN uses raw logits with MSE loss)
    """

    def __init__(self, in_ch: int = 4, ndf: int = 64):
        super().__init__()
        ch = in_ch + 3      # condition + RGB

        self.net = nn.Sequential(
            # Block 1  — no norm on first layer (standard Pix2Pix convention)
            nn.Conv2d(ch,    ndf,    kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.Conv2d(ndf,   ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4 — stride 1 to widen receptive field without shrinking
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output patch scores
            nn.Conv2d(ndf*8, 1,     kernel_size=4, stride=1, padding=1, bias=True),
        )

    def forward(self, label: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([label, img], dim=1))
