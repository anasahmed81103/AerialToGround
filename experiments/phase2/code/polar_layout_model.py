"""
polar_layout_model.py  —  Phase 2 Stage-1 model: frozen DINOv2 aerial features ->
panorama semantic layout through a geometry-initialised polar mapping.

  aerial feats (B, 384, 24, 24)            cached, frozen DINOv2
    -> 1x1 projection + 2 residual blocks            (B, D, 24, 24)
    -> polar resampling along rays (+ residual offsets)  (B, D, R, W)
    -> column decoder: H row queries attend over the R radial samples of the
       same azimuth (+ one global token); attention bias starts from a
       ground-plane prior                              (B, D, H, W)
    -> circular-padded refinement + 1x1 head           (B, C, H, W)

The decoder is shared across azimuth columns, so the mapping is rotation
equivariant and has no per-pixel (source, target) weights to learn.
mapping='dense' swaps the polar resampling for a learned (g*g -> R*W) matrix
(the CrossNet idea) as an ablation of the geometric prior.

CVUSA convention (measured by geo_calibrate.py): north at the centre column,
azimuth increasing clockwise, bottom row = camera position (tile centre).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _circular_pad_w(x):
    x = F.pad(x, (1, 1, 0, 0), mode='circular')
    return F.pad(x, (0, 0, 1, 1))


class ResBlock(nn.Module):
    def __init__(self, dim, circular_w=False):
        super().__init__()
        self.circular_w = circular_w
        pad = 0 if circular_w else 1
        self.c1 = nn.Conv2d(dim, dim, 3, padding=pad)
        self.n1 = nn.GroupNorm(8, dim)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=pad)
        self.n2 = nn.GroupNorm(8, dim)

    def _conv(self, conv, x):
        return conv(_circular_pad_w(x) if self.circular_w else x)

    def forward(self, x):
        h = F.gelu(self.n1(self._conv(self.c1, x)))
        h = self.n2(self._conv(self.c2, h))
        return F.gelu(x + h)


def polar_base_grid(radial_bins: int, width: int) -> torch.Tensor:
    """(R, W, 2) grid_sample coords: radius 0 = tile centre, 1 = tile edge."""
    r = (torch.arange(radial_bins, dtype=torch.float32) + 0.5) / radial_bins
    theta = 2 * math.pi * (torch.arange(width, dtype=torch.float32) + 0.5) / width - math.pi
    x = r[:, None] * torch.sin(theta)[None, :]
    y = -r[:, None] * torch.cos(theta)[None, :]
    return torch.stack([x, y], dim=-1)


def ground_plane_bias(height: int, radial_bins: int, horizon_frac: float,
                      sigma_bins: float = 1.5) -> torch.Tensor:
    """(H, R) log-prior. Rows below the horizon look at one ground radius
    (bottom row -> centre, horizon -> edge); rows above are left uniform."""
    h0 = height * horizon_frac
    rows = torch.arange(height, dtype=torch.float32) + 0.5
    rho = ((height - rows) / (height - h0)).clamp(max=1.0)
    r = (torch.arange(radial_bins, dtype=torch.float32) + 0.5) / radial_bins
    d = (r[None, :] - rho[:, None]) * radial_bins / sigma_bins
    bias = -0.5 * d ** 2
    bias[rows < h0] = 0.0
    return bias


class ColumnDecoderLayer(nn.Module):
    def __init__(self, dim, heads, prior_bias, prior_mode):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.nf = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

        if prior_mode == 'none':
            bias = torch.zeros_like(prior_bias)
        else:
            bias = prior_bias.clone()
        # extra zero column for the global token
        bias = F.pad(bias, (0, 1)).unsqueeze(0).repeat(heads, 1, 1)
        if prior_mode == 'fixed':
            self.register_buffer('bias', bias)
        else:
            self.bias = nn.Parameter(bias)

    def forward(self, q_tok, kv_tok):
        n, h, d = q_tok.shape
        r = kv_tok.size(1)
        q = self.q(self.nq(q_tok)).view(n, h, self.heads, self.hd).transpose(1, 2)
        kvn = self.nkv(kv_tok)
        k = self.k(kvn).view(n, r, self.heads, self.hd).transpose(1, 2)
        v = self.v(kvn).view(n, r, self.heads, self.hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.bias.to(q.dtype).unsqueeze(0),
        )
        out = out.transpose(1, 2).reshape(n, h, d)
        x = q_tok + self.o(out)
        return x + self.ff(self.nf(x))


class PolarLayoutNet(nn.Module):
    def __init__(self,
                 in_ch: int = 384,
                 dim: int = 128,
                 num_classes: int = 4,
                 radial_bins: int = 32,
                 out_hw: tuple = (64, 320),
                 horizon_frac: float = 0.5,
                 heads: int = 4,
                 layers: int = 2,
                 use_offsets: bool = True,
                 max_offset: float = 0.1,
                 prior_mode: str = 'learned',
                 mapping: str = 'polar',
                 in_grid: int = 24):
        super().__init__()
        assert prior_mode in ('learned', 'fixed', 'none')
        assert mapping in ('polar', 'dense')
        self.out_hw = out_hw
        self.radial_bins = radial_bins
        self.mapping = mapping
        use_offsets = use_offsets and mapping == 'polar'
        self.use_offsets = use_offsets
        self.max_offset = max_offset
        H, W = out_hw

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, dim, 1), nn.GroupNorm(8, dim), nn.GELU(),
            ResBlock(dim), ResBlock(dim),
        )
        self.register_buffer('base_grid', polar_base_grid(radial_bins, W))
        if mapping == 'dense':
            # CrossNet-style learned transform: every (radius, azimuth) cell is a free
            # linear combination of all aerial positions; no geometry is given.
            n_in = in_grid * in_grid
            self.dense_M = nn.Parameter(torch.randn(n_in, radial_bins * W) / math.sqrt(n_in))

        if use_offsets:
            self.offset = nn.Conv2d(dim, 2, 3, padding=0)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)

        self.radial_pos = nn.Parameter(torch.zeros(1, dim, radial_bins, 1))
        nn.init.trunc_normal_(self.radial_pos, std=0.02)
        self.global_tok = nn.Linear(dim, dim)

        self.row_queries = nn.Parameter(torch.zeros(H, dim))
        nn.init.trunc_normal_(self.row_queries, std=0.02)
        prior = ground_plane_bias(H, radial_bins, horizon_frac)
        self.layers = nn.ModuleList(
            ColumnDecoderLayer(dim, heads, prior, prior_mode) for _ in range(layers)
        )

        self.refine = nn.Sequential(ResBlock(dim, circular_w=True),
                                    ResBlock(dim, circular_w=True))
        self.head = nn.Conv2d(dim, num_classes, 1)

    def _sample(self, x, grid):
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='border', align_corners=False)

    def forward(self, feats):
        x = self.proj(feats)
        b, d = x.shape[:2]
        H, W = self.out_hw
        R = self.radial_bins

        if self.mapping == 'dense':
            p = (x.flatten(2) @ self.dense_M).view(b, d, R, W)
        else:
            grid = self.base_grid.unsqueeze(0).expand(b, -1, -1, -1)
            p = self._sample(x, grid)
        if self.use_offsets:
            off = torch.tanh(self.offset(_circular_pad_w(p)))
            p = self._sample(x, grid + off.permute(0, 2, 3, 1) * self.max_offset)
        p = p + self.radial_pos

        kv = p.permute(0, 3, 2, 1).reshape(b * W, R, d)
        g = self.global_tok(x.mean(dim=(2, 3)))
        g = g.unsqueeze(1).expand(b, W, d).reshape(b * W, 1, d)
        kv = torch.cat([kv, g], dim=1)

        q = self.row_queries.unsqueeze(0).expand(b * W, -1, -1)
        for layer in self.layers:
            q = layer(q, kv)

        y = q.reshape(b, W, H, d).permute(0, 3, 2, 1)
        y = self.refine(y)
        return self.head(y)
