"""
crossnet_dataset.py  —  PyTorch Dataset for CVPR (CVUSA subset) cross-view data
========================================================================
CSV format (one sample per line):
    aerial_path, ground_path, label_path
    aerial_path, ground_path, ground_label, aerial_label   (v2 / Phase 1)

Images are loaded as RGB uint8, then:
  • Aerial  : centre-cropped to (224, 224) and normalised to [-1, 1]
  • Ground  : bilinear-resized to (224, 1232) and normalised to [-1, 1]
  • Label   : nearest-resized to (224, 1232), kept as uint8 integer class ids
  • Aerial label (optional): nearest-resized to (224, 224); -1 if missing
"""

import os
import random
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
                 root:        str   = '',
                 augment:     bool  = False):
        """
        Parameters
        ----------
        csv_path    : path to the CSV manifest (3 or 4 columns)
        aerial_size : (H, W) target size for aerial images (centre-cropped)
        ground_size : (H, W) target size for ground / label images
        root        : optional path prefix prepended to every entry in the CSV
        augment     : if True, 50% horizontal flip (aerial + left-right reverse of pano)
        """
        self.aerial_size = aerial_size
        self.ground_size = ground_size
        self.root        = root
        self.augment     = augment

        self.samples: list[dict] = []
        with open(csv_path, 'r') as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) < 3:
                    continue
                if root:
                    parts = [os.path.join(root, p) for p in parts]
                sample = {
                    'aerial': parts[0],
                    'ground': parts[1],
                    'label':  parts[2],
                    'aerial_label': parts[3] if len(parts) >= 4 else '',
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _center_crop(img: Image.Image, hw: tuple) -> Image.Image:
        H, W = hw
        w_orig, h_orig = img.size
        left   = (w_orig - W) // 2
        top    = (h_orig - H) // 2
        return img.crop((left, top, left + W, top + H))

    @staticmethod
    def _to_tensor_norm(img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return t

    @staticmethod
    def _load_label(path: str, hw: tuple) -> torch.Tensor:
        H, W = hw
        pil = Image.open(path)
        if pil.mode == 'RGB':
            pil = pil.convert('L')
        elif pil.mode == 'RGBA':
            pil = pil.split()[0]
        if pil.size != (W, H):
            pil = pil.resize((W, H), Image.NEAREST)
        return torch.from_numpy(np.array(pil)).long()

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        aH, aW = self.aerial_size
        gH, gW = self.ground_size

        aerial_pil = Image.open(s['aerial']).convert('RGB')
        if aerial_pil.size != (aW, aH):
            if aerial_pil.size[0] < aW or aerial_pil.size[1] < aH:
                aerial_pil = aerial_pil.resize((aW, aH), Image.BILINEAR)
            else:
                aerial_pil = self._center_crop(aerial_pil, self.aerial_size)
        aerial = self._to_tensor_norm(aerial_pil)

        ground_pil = Image.open(s['ground']).convert('RGB')
        if ground_pil.size != (gW, gH):
            ground_pil = ground_pil.resize((gW, gH), Image.BILINEAR)
        ground = self._to_tensor_norm(ground_pil)

        label = self._load_label(s['label'], self.ground_size)

        if s['aerial_label'] and os.path.isfile(s['aerial_label']):
            aerial_label = self._load_label(s['aerial_label'], self.aerial_size)
        else:
            aerial_label = torch.full(self.aerial_size, -1, dtype=torch.long)

        if self.augment and random.random() < 0.5:
            aerial = TF.hflip(aerial)
            ground = TF.hflip(ground)
            label = torch.flip(label, dims=[-1])
            aerial_label = torch.flip(aerial_label, dims=[-1])

        return aerial, ground, label, aerial_label
