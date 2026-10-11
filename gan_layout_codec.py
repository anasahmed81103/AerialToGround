"""
Shared semantic layout encoding for Stage-2 GAN / SPADE (matches GANDataset).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def class_indices_to_onehot(lbl_hw: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """(H, W) int labels -> (C, H, W) float32 in [-1, 1]."""
    lbl = lbl_hw.astype(np.int64).clip(0, num_classes - 1)
    one_hot = np.eye(num_classes, dtype=np.float32)[lbl].transpose(2, 0, 1)
    return torch.from_numpy(one_hot * 2.0 - 1.0)


def load_gt_label_tensor(
    label_path: str,
    num_classes: int,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Same rules as GANDataset label branch."""
    lbl_pil = Image.open(label_path)
    if lbl_pil.mode == 'RGB':
        lbl_pil = lbl_pil.convert('L')
    elif lbl_pil.mode == 'RGBA':
        lbl_pil = lbl_pil.split()[0]
    lbl_pil = lbl_pil.resize((img_w, img_h), Image.NEAREST)
    lbl_np = np.array(lbl_pil, dtype=np.int64)
    return class_indices_to_onehot(lbl_np, num_classes)


def encoding_stats(label_t: torch.Tensor) -> dict:
    x = label_t.float()
    ch_mean = x.mean(dim=(1, 2)).tolist()
    return {
        'dtype': str(label_t.dtype),
        'shape': list(label_t.shape),
        'min': float(x.min()),
        'max': float(x.max()),
        'per_channel_mean': ch_mean,
    }
