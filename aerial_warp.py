"""
Polar warp: full aerial RGB tile -> panorama-shaped tensor (Stage 2 conditioning).

Uses the same geometry as geo_calibrate.py and experiments/phase2/results/geo_calibration.json
(north column, clockwise azimuth, horizon_frac, tile centre -> bottom row).
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from geo_calibrate import polar_grid

DEFAULT_CAL = os.path.join('experiments', 'phase2', 'results', 'geo_calibration.json')


def load_geo_calibration(path: str = DEFAULT_CAL) -> dict:
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    best = data['best']
    return {
        'horizon_frac': float(data['horizon_frac']),
        'clockwise': bool(best['clockwise']),
        'north_at_frac': float(best['north_at_frac']),
        'pano_grid_ref': tuple(data['pano_grid']),
    }


class PanoAerialWarper:
    """Precomputes grid_sample grid for fixed output size (H, W)."""

    def __init__(
        self,
        out_hw: Tuple[int, int] = (256, 512),
        cal_path: str = DEFAULT_CAL,
        device: torch.device | None = None,
    ):
        self.out_h, self.out_w = out_hw
        cal = load_geo_calibration(cal_path)
        self.horizon_frac = cal['horizon_frac']
        self.v0: int
        grid, self.v0 = polar_grid(
            self.out_h, self.out_w, self.horizon_frac, cal['clockwise']
        )
        self.shift = int(round(cal['north_at_frac'] * self.out_w))
        dev = device or torch.device('cpu')
        self.grid = grid.unsqueeze(0).to(dev)  # (1, H_band, W, 2)
        self.device = dev

    def warp_aerial_to_pano(self, aerial_rgb) -> torch.Tensor:
        """
        Parameters
        ----------
        aerial_rgb : path str, or float tensor (3, H, W) in [-1, 1], or (1, 3, H, W)

        Returns
        -------
        (3, out_h, out_w) float32 in [-1, 1]; sky rows (above horizon) are 0.
        """
        tile = _aerial_to_tensor(aerial_rgb).unsqueeze(0).to(self.device)
        band = F.grid_sample(
            tile, self.grid, mode='bilinear', align_corners=False
        )
        band = torch.roll(band, shifts=self.shift, dims=-1)
        out = torch.zeros(1, 3, self.out_h, self.out_w, device=self.device)
        out[:, :, self.v0 :, :] = band
        return out.squeeze(0).clamp(-1.0, 1.0)


def warp_aerial_to_pano(
    aerial_rgb,
    out_hw: Tuple[int, int] = (256, 512),
    cal_path: str = DEFAULT_CAL,
    warper: PanoAerialWarper | None = None,
) -> torch.Tensor:
    """Functional API; reuses *warper* when batching."""
    w = warper or PanoAerialWarper(out_hw, cal_path)
    return w.warp_aerial_to_pano(aerial_rgb)


def _aerial_to_tensor(aerial_rgb) -> torch.Tensor:
    if isinstance(aerial_rgb, torch.Tensor):
        t = aerial_rgb.float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.max() > 1.5:
            t = t / 127.5 - 1.0
        return t
    if isinstance(aerial_rgb, (str, os.PathLike)):
        img = Image.open(aerial_rgb).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1))
    raise TypeError(f'Expected path or tensor, got {type(aerial_rgb)}')
