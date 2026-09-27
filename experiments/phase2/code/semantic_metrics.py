"""
Shared 4-class palette, colorization, and segmentation metrics for CrossNet.

IoU is reported at two resolutions:
  • full  — logits bilinear-upsampled to the label size (224×1232)
  • low   — argmax at the native CrossNet grid (8×40), labels nearest-downsampled
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.amp import autocast


CLASS_NAMES = ('sky', 'vegetation', 'road', 'building')

# red=sky, green=vegetation, blue=road, yellow=building
CLASS_COLORS = np.array([
    [255,  64,  64],
    [ 64, 200,  64],
    [ 64,  64, 255],
    [255, 220,  30],
], dtype=np.uint8)


def colorize_label(indices, num_classes: int = 4) -> np.ndarray:
    """Map a (H, W) class-id map to an (H, W, 3) uint8 RGB image."""
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    idx = np.asarray(indices).astype(np.int64)
    idx = np.clip(idx, 0, num_classes - 1)
    return CLASS_COLORS[:num_classes][idx]


def denorm_image(t: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) tensor in [-1, 1] to (H, W, 3) uint8."""
    arr = t.detach().float().cpu().numpy()
    arr = (arr * 0.5 + 0.5).clip(0.0, 1.0)
    return (arr * 255.0).astype(np.uint8).transpose(1, 2, 0)


class ConfusionMeter:
    """Accumulates a (C, C) confusion matrix. Rows = GT, cols = prediction."""

    def __init__(self, num_classes: int):
        self.n = num_classes
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().reshape(-1)
        target = target.detach().reshape(-1)
        valid = (target >= 0) & (target < self.n)
        pred = pred[valid]
        target = target[valid]
        if pred.numel() == 0:
            return
        idx = target * self.n + pred
        bc = torch.bincount(idx.cpu(), minlength=self.n * self.n)
        self.cm += bc.reshape(self.n, self.n)

    def compute(self) -> dict:
        cm = self.cm.float()
        tp = torch.diag(cm)
        support = cm.sum(dim=1)
        denom = (cm.sum(dim=1) + cm.sum(dim=0) - tp).clamp(min=1e-6)
        iou = tp / denom
        present = support > 0
        miou = iou[present].mean().item() if bool(present.any()) else 0.0
        acc = (tp.sum() / cm.sum().clamp(min=1.0)).item()
        return {
            'miou': miou,
            'iou': iou,
            'acc': acc,
            'cm': self.cm.clone(),
            'support': support,
        }


@torch.inference_mode()
def evaluate_crossnet(model,
                      loader,
                      device: torch.device,
                      num_classes: int = 4,
                      use_amp: bool = False) -> dict:
    """Run CrossNet over a loader and return full-res + native-grid metrics."""
    was_training = model.training
    model.eval()

    meter_full = ConfusionMeter(num_classes)
    meter_low = ConfusionMeter(num_classes)
    n_seen = 0

    amp_on = bool(use_amp and device.type == 'cuda')
    n_batches = len(loader)
    for bi, batch in enumerate(loader, start=1):
        aerial, _ground, label = batch[0], batch[1], batch[2]
        aerial = aerial.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with autocast('cuda', enabled=amp_on):
            _la, lg = model(aerial)
        lg = lg.float()

        pred_full = F.interpolate(
            lg, size=label.shape[-2:], mode='bilinear', align_corners=True,
        ).argmax(dim=1)
        meter_full.update(pred_full, label)

        low_hw = lg.shape[-2:]
        pred_low = lg.argmax(dim=1)
        label_low = F.interpolate(
            label.float().unsqueeze(1), size=low_hw, mode='nearest',
        ).squeeze(1).long()
        meter_low.update(pred_low, label_low)
        n_seen += aerial.size(0)
        if bi == 1 or bi % 50 == 0 or bi == n_batches:
            print(f'    eval batch {bi}/{n_batches}  ({n_seen} images)', flush=True)

    if was_training:
        model.train()

    return {
        'n': n_seen,
        'full': meter_full.compute(),
        'low': meter_low.compute(),
    }


def format_metrics(metrics: dict, class_names=CLASS_NAMES) -> str:
    full = metrics['full']
    low = metrics['low']
    n = metrics['n']
    lines = [
        f"  Validation  ({n} images)",
        f"  mIoU @ full-res : {full['miou']:.4f}",
        f"  mIoU @ native   : {low['miou']:.4f}   (8x40)",
        f"  pixel acc full  : {full['acc']:.4f}",
    ]
    support = full['support'].float()
    support_pct = 100.0 * support / support.sum().clamp(min=1.0)
    iou = full['iou']
    for i, name in enumerate(class_names[:len(iou)]):
        lines.append(
            f"    {name:<12} IoU {iou[i].item():.4f}   "
            f"(GT {support_pct[i].item():5.1f}%)"
        )
    return '\n'.join(lines)


def metrics_row(metrics: dict, epoch: int, step: int, class_names=CLASS_NAMES) -> dict:
    """Flat dict suitable for a CSV row."""
    full = metrics['full']
    low = metrics['low']
    row = {
        'epoch': epoch,
        'step': step,
        'n': metrics['n'],
        'miou_full': f"{full['miou']:.6f}",
        'miou_native': f"{low['miou']:.6f}",
        'acc_full': f"{full['acc']:.6f}",
    }
    for i, name in enumerate(class_names[:full['iou'].numel()]):
        row[f'iou_{name}'] = f"{full['iou'][i].item():.6f}"
    return row


def save_eval_montage(aerial: torch.Tensor,
                      ground: torch.Tensor,
                      label: torch.Tensor,
                      la: torch.Tensor,
                      lg: torch.Tensor,
                      step: int,
                      dump_dir: str,
                      prefix: str = 'train',
                      num_classes: int = 4) -> str:
    """
    Save aerial | ground photo | GT label | pred ground | pred aerial.

    Semantic columns use argmax + CLASS_COLORS (nearest upsample).
    """
    import os

    pred_g = lg[0].argmax(dim=0)
    pred_a = la[0].argmax(dim=0)
    gt = label[0]

    photos = [
        ('AERIAL', denorm_image(aerial[0]), Image.BILINEAR),
        ('GROUND', denorm_image(ground[0]), Image.BILINEAR),
    ]
    maps = [
        ('GT', colorize_label(gt, num_classes), Image.NEAREST),
        ('PRED-G', colorize_label(pred_g, num_classes), Image.NEAREST),
        ('PRED-A', colorize_label(pred_a, num_classes), Image.NEAREST),
    ]

    thumb_h = 112
    header_h = 20
    parts = []
    captions = []
    for caption, arr, resample in photos + maps:
        img = Image.fromarray(arr)
        w = max(1, int(img.width * thumb_h / img.height))
        parts.append(img.resize((w, thumb_h), resample))
        captions.append(caption)

    gap = 4
    total_w = sum(p.width for p in parts) + gap * (len(parts) - 1)
    canvas = Image.new('RGB', (total_w, thumb_h + header_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x_off = 0
    for cap, part in zip(captions, parts):
        draw.text((x_off + 2, 2), cap, fill=(210, 210, 210), font=font)
        canvas.paste(part, (x_off, header_h))
        x_off += part.width + gap

    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f'{prefix}_{step:07d}.jpg')
    canvas.save(path, quality=92)
    return path
