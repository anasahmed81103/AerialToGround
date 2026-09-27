"""
crossnet_dataset.py  —  PyTorch Dataset for CVPR (CVUSA subset) cross-view data
========================================================================
CSV format (one sample per line):
    aerial_path, ground_path, label_path

Images are loaded as RGB uint8, then:
  • Aerial  : centre-cropped to (224, 224) and normalised to [-1, 1]
  • Ground  : bilinear-resized to (224, 1232) and normalised to [-1, 1]
  • Label   : nearest-resized to (224, 1232), kept as uint8 integer class ids
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CVPRDataset(Dataset):
    """Dataset for aerial ↔ ground semantic segmentation transfer (CVUSA subset)."""

    def __init__(self,
                 csv_path:    str,
                 aerial_size: tuple = (224, 224),
                 ground_size: tuple = (224, 1232),
                 root:        str   = ''):
        """
        Parameters
        ----------
        csv_path    : path to the CSV manifest
        aerial_size : (H, W) target size for aerial images (centre-cropped)
        ground_size : (H, W) target size for ground / label images
        root        : optional path prefix prepended to every entry in the CSV
        """
        self.aerial_size = aerial_size
        self.ground_size = ground_size
        self.root        = root

        self.samples: list[dict] = []
        with open(csv_path, 'r') as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) != 3:
                    continue
                if root:
                    parts = [os.path.join(root, p) for p in parts]
                self.samples.append({'aerial': parts[0],
                                     'ground': parts[1],
                                     'label':  parts[2]})

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _center_crop(img: Image.Image, hw: tuple) -> Image.Image:
        """Centre-crop a PIL image to (H, W)."""
        H, W = hw
        w_orig, h_orig = img.size           # PIL: (width, height)
        left   = (w_orig - W) // 2
        top    = (h_orig - H) // 2
        return img.crop((left, top, left + W, top + H))

    @staticmethod
    def _to_tensor_norm(img: Image.Image) -> torch.Tensor:
        """Convert PIL RGB image to float32 tensor in [-1, 1]."""
        t = TF.to_tensor(img)               # [0, 1]  float32  (C, H, W)
        t = TF.normalize(t,
                         mean=[0.5, 0.5, 0.5],
                         std= [0.5, 0.5, 0.5])   # → [-1, 1]
        return t

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # ── Aerial ────────────────────────────────────────────────────
        aerial_pil = Image.open(s['aerial']).convert('RGB')
        # Centre-crop to exactly (224, 224)
        aH, aW = self.aerial_size
        if aerial_pil.size != (aW, aH):
            if aerial_pil.size[0] < aW or aerial_pil.size[1] < aH:
                aerial_pil = aerial_pil.resize((aW, aH), Image.BILINEAR)
            else:
                aerial_pil = self._center_crop(aerial_pil, self.aerial_size)
        aerial = self._to_tensor_norm(aerial_pil)   # (3, 224, 224)

        # ── Ground panorama ───────────────────────────────────────────
        gH, gW = self.ground_size
        ground_pil = Image.open(s['ground']).convert('RGB')
        if ground_pil.size != (gW, gH):
            ground_pil = ground_pil.resize((gW, gH), Image.BILINEAR)
        ground = self._to_tensor_norm(ground_pil)   # (3, 224, 1232)

        # ── Semantic label ────────────────────────────────────────────
        label_pil = Image.open(s['label'])
        if label_pil.mode == 'RGB':
            label_pil = label_pil.convert('L')
        elif label_pil.mode == 'RGBA':
            label_pil = label_pil.split()[0]        # take R channel
        if label_pil.size != (gW, gH):
            label_pil = label_pil.resize((gW, gH), Image.NEAREST)
        label = torch.from_numpy(np.array(label_pil)).long()   # (224, 1232)

        return aerial, ground, label
