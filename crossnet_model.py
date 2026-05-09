"""
crossnet_model.py  —  CrossNet re-implemented in PyTorch
================================================
Architecture (identical to paper arXiv:1612.02709):

  • Network A  : VGG16 (VALID-padding) backbone → hypercolumn → 3×(1×1 conv) → La
  • Network S  : conv4_3 → 2×(1×1 conv) → per-aerial-pixel scalar S(Ia)
  • Network F  : [i, j, y, x, S(Ia)] → weight matrix M  (softmax over source pixels)
  • Transfer T : Lg = M^T @ La + bias

VALID padding ensures the conv4_3 output is 17×17 for a 224×224 input, keeping
the (17×17=289) × (8×40=320) weight matrix at a VRAM-manageable size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helper: build (i, j, y, x) indexing tensor for all (source, target) pairs
# ---------------------------------------------------------------------------

def _build_indexing(source_size, target_size):
    """Return a (n_src, n_tgt, 4) float32 tensor of normalised coordinates.

    For every pair (aerial pixel c, ground pixel r):
        [i, j, y, x]  —  i/j are aerial row/col in [0,1],
                          y/x are ground row/col in [0,1].
    """
    Hs, Ws = source_size
    Hg, Wg = target_size

    i = torch.linspace(0., 1., Hs)
    j = torch.linspace(0., 1., Ws)
    ii, jj = torch.meshgrid(i, j, indexing='ij')           # (Hs, Ws)

    y = torch.linspace(0., 1., Hg)
    x = torch.linspace(0., 1., Wg)
    yy, xx = torch.meshgrid(y, x, indexing='ij')           # (Hg, Wg)

    ii_f = ii.reshape(-1)   # (n_src,)
    jj_f = jj.reshape(-1)
    yy_f = yy.reshape(-1)   # (n_tgt,)
    xx_f = xx.reshape(-1)

    n_src = Hs * Ws
    n_tgt = Hg * Wg

    I = ii_f.unsqueeze(1).expand(n_src, n_tgt)
    J = jj_f.unsqueeze(1).expand(n_src, n_tgt)
    Y = yy_f.unsqueeze(0).expand(n_src, n_tgt)
    X = xx_f.unsqueeze(0).expand(n_src, n_tgt)

    return torch.stack([I, J, Y, X], dim=2)   # (n_src, n_tgt, 4)


# ---------------------------------------------------------------------------
# VGG16 backbone with VALID padding (no zero-padding on conv layers)
# For 224×224 input the conv4_3 output is 17×17, matching the paper.
# ---------------------------------------------------------------------------

class _ConvReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )


class VGG16ValidBackbone(nn.Module):
    """VGG-16 feature extractor with VALID (no) padding on all conv layers.

    Output sizes for 224×224 input:
        block1 → 220×220 (64 ch)
        pool1  → 110×110
        block2 → 106×106 (128 ch)
        pool2  →  53×53
        block3 →  47×47  (256 ch)
        pool3  →  23×23
        block4 →  17×17  (512 ch)   ← last_conv used by condition & weight nets
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.block1 = nn.Sequential(_ConvReLU(3,   64),  _ConvReLU(64,  64))
        self.pool1  = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(_ConvReLU(64,  128), _ConvReLU(128, 128))
        self.pool2  = nn.MaxPool2d(2, 2)
        self.block3 = nn.Sequential(_ConvReLU(128, 256), _ConvReLU(256, 256), _ConvReLU(256, 256))
        self.pool3  = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(_ConvReLU(256, 512), _ConvReLU(512, 512), _ConvReLU(512, 512))

        if pretrained:
            self._load_imagenet_weights()

    # Transfer conv weights from torchvision VGG16 (SAME padding → VALID padding
    # shares identical kernel shapes so weights copy directly).
    def _load_imagenet_weights(self):
        try:
            from torchvision.models import vgg16, VGG16_Weights
            src = list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.children())
            # (src index → our layer)
            mapping = [
                (src[0],  self.block1[0][0]),   # conv1_1
                (src[2],  self.block1[1][0]),   # conv1_2
                (src[5],  self.block2[0][0]),   # conv2_1
                (src[7],  self.block2[1][0]),   # conv2_2
                (src[10], self.block3[0][0]),   # conv3_1
                (src[12], self.block3[1][0]),   # conv3_2
                (src[14], self.block3[2][0]),   # conv3_3
                (src[17], self.block4[0][0]),   # conv4_1
                (src[19], self.block4[1][0]),   # conv4_2
                (src[21], self.block4[2][0]),   # conv4_3
            ]
            with torch.no_grad():
                for s_layer, t_layer in mapping:
                    t_layer.weight.copy_(s_layer.weight)
                    t_layer.bias.copy_(s_layer.bias)
            print("[*] VGG16 backbone: loaded ImageNet pretrained weights.")
        except Exception as e:
            print(f"[!] Could not load pretrained VGG16 weights ({e}). Random init.")

    def forward(self, x):
        f1 = self.block1(x)                   # (B, 64,  220, 220)
        f2 = self.block2(self.pool1(f1))      # (B, 128, 106, 106)
        f3 = self.block3(self.pool2(f2))      # (B, 256,  47,  47)
        f4 = self.block4(self.pool3(f3))      # (B, 512,  17,  17)
        return [f1, f2, f3, f4]              # hypercolumn feature pyramid


# ---------------------------------------------------------------------------
# CrossNet — full model
# ---------------------------------------------------------------------------

class CrossNet(nn.Module):
    """
    CrossNet: aerial-to-ground semantic label transfer.

    Forward input : aerial images  (B, 3, 224, 224),  normalised to [-1, 1]
    Forward output: (La, Lg)
        La — aerial semantic logits   (B, num_classes, 17, 17)
        Lg — ground semantic logits   (B, num_classes,  8, 40)
    """

    HYPERCOLUMN_CH = 64 + 128 + 256 + 512   # 960

    def __init__(self,
                 num_classes: int = 4,
                 conditioned:  bool = True,
                 source_size:  tuple = (17, 17),
                 target_size:  tuple = (8,  40),
                 pretrained:   bool  = True):
        super().__init__()
        self.num_classes = num_classes
        self.conditioned = conditioned
        self.source_size = source_size
        self.target_size = target_size

        Hs, Ws = source_size
        Hg, Wg = target_size
        n_src  = Hs * Ws   # 289
        n_tgt  = Hg * Wg   # 320

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = VGG16ValidBackbone(pretrained=pretrained)

        # ── Network A: hypercolumn MLP → aerial semantic labels ──────────
        self.aerial_mlp = nn.Sequential(
            nn.Conv2d(self.HYPERCOLUMN_CH, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),                 nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1),
        )

        # ── Network S: conditioning (conv4_3 → per-pixel scalar) ─────────
        if conditioned:
            self.cond_net = nn.Sequential(
                nn.Conv2d(512, 64, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64,  1,  1),
            )
            wn_in_ch = 4 + n_src   # 4 + 289 = 293
        else:
            self.cond_net = None
            wn_in_ch = 4

        # ── Network F: transformation weight MLP ─────────────────────────
        self.weight_net = nn.Sequential(
            nn.Conv2d(wn_in_ch, 128, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1),       nn.ReLU(inplace=True),
            nn.Conv2d(64,  1,  1),
        )

        # ── Transfer bias (1 per ground row, broadcast over cols/batch) ──
        self.ground_bias = nn.Parameter(torch.zeros(1, num_classes, Hg, 1))

        # ── Precompute indexing tensor (stored as non-trainable buffer) ──
        self.register_buffer('indexing', _build_indexing(source_size, target_size))
        # indexing: (n_src, n_tgt, 4)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, aerial):
        """
        aerial : (B, 3, 224, 224) float32/16, values in [-1, 1]
        returns: La (B, C, 17, 17),  Lg (B, C, 8, 40)
        """
        B = aerial.shape[0]
        Hs, Ws = self.source_size
        Hg, Wg = self.target_size
        n_src  = Hs * Ws   # 289
        n_tgt  = Hg * Wg   # 320

        # 1. Backbone → feature pyramid
        features  = self.backbone(aerial)   # [f1, f2, f3, f4]
        last_conv = features[-1]            # (B, 512, 17, 17)

        # 2. Hypercolumn: resize all features to source_size, concat
        hyper = torch.cat([
            F.interpolate(f, size=self.source_size,
                          mode='bilinear', align_corners=True)
            for f in features
        ], dim=1)                           # (B, 960, 17, 17)

        # 3. Network A → aerial semantic logits
        La = self.aerial_mlp(hyper)        # (B, C, 17, 17)

        # 4. Build transformation weight matrix M
        #    indexing: (n_src, n_tgt, 4) → (B, n_src, n_tgt, 4)
        idx = self.indexing.unsqueeze(0).expand(B, -1, -1, -1)

        if self.conditioned:
            s = self.cond_net(last_conv)            # (B, 1, 17, 17)
            # Flatten aerial spatial dim → (B, 1, 1, n_src)
            s_flat   = s.reshape(B, 1, 1, n_src)
            # Tile for every target pixel → (B, n_src, n_tgt, n_src)
            s_tiled  = s_flat.expand(B, n_src, n_tgt, n_src)
            # Concat with (i,j,y,x): → (B, n_src, n_tgt, 4+n_src)
            net_in   = torch.cat([idx, s_tiled], dim=3)
        else:
            net_in = idx  # (B, n_src, n_tgt, 4)

        # Permute to (B, C, n_src, n_tgt) for Conv2d
        net_in    = net_in.permute(0, 3, 1, 2).contiguous()
        raw_w     = self.weight_net(net_in)         # (B, 1, n_src, n_tgt)
        raw_w     = raw_w.squeeze(1)                # (B, n_src, n_tgt)
        M         = F.softmax(raw_w, dim=1)         # each ground pixel sums to 1 over aerial

        # 5. Transfer: Lg = La_flat @ M + bias
        La_flat   = La.reshape(B, self.num_classes, n_src)   # (B, C, 289)
        Lg_flat   = torch.bmm(La_flat, M)                    # (B, C, 320)
        Lg        = Lg_flat.reshape(B, self.num_classes, Hg, Wg)
        Lg        = Lg + self.ground_bias                    # broadcast over batch & width

        return La, Lg
