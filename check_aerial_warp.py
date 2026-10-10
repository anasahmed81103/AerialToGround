"""
Visual gate before SPADE+aerial training: 8 val pairs, 3-panel JPGs.

  raw aerial (256 px high) | warped aerial | real pano (GAN resize)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aerial_warp import PanoAerialWarper


def _load_csv_rows(csv_path: str) -> list[list[str]]:
    rows = []
    with open(csv_path, encoding='utf-8') as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) >= 2:
                rows.append(parts)
    return rows


def _pano_like_gan(path: str, img_h: int, img_w: int) -> np.ndarray:
    g = Image.open(path).convert('RGB')
    g = g.resize((img_w, img_h), Image.BILINEAR)
    return np.array(g, dtype=np.uint8)


def _aerial_strip(path: str, strip_h: int = 256) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    w, h = img.size
    new_w = max(1, int(round(w * strip_h / h)))
    img = img.resize((new_w, strip_h), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().cpu().numpy().transpose(1, 2, 0)
    arr = ((arr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return arr


def _pad_to_height(img: np.ndarray, target_h: int, fill=(32, 32, 32)) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    canvas = np.full((target_h, w, 3), fill, dtype=np.uint8)
    y0 = (target_h - h) // 2
    canvas[y0 : y0 + h] = img
    return canvas


def main():
    p = argparse.ArgumentParser(description='Check aerial polar warp vs real panos')
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--out_dir', default='experiments/stage2_aerial/warp_check')
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--n', type=int, default=8)
    args = p.parse_args()

    rows = _load_csv_rows(args.val_csv)
    if not rows:
        raise SystemExit(f'No rows in {args.val_csv}')

    idx = np.linspace(0, len(rows) - 1, args.n).astype(int)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warper = PanoAerialWarper((args.img_h, args.img_w), device=device)

    for j, i in enumerate(idx):
        aerial_path, pano_path = rows[i][0], rows[i][1]
        if not os.path.isfile(aerial_path):
            print(f'[!] missing aerial: {aerial_path}')
            continue
        if not os.path.isfile(pano_path):
            print(f'[!] missing pano: {pano_path}')
            continue

        aerial_u8 = _aerial_strip(aerial_path, args.img_h)
        warped = warper.warp_aerial_to_pano(aerial_path)
        warped_u8 = _tensor_to_uint8_rgb(warped)
        pano_u8 = _pano_like_gan(pano_path, args.img_h, args.img_w)

        h = args.img_h
        aerial_u8 = _pad_to_height(aerial_u8, h)
        warped_u8 = _pad_to_height(warped_u8, h)
        montage = np.concatenate([aerial_u8, warped_u8, pano_u8], axis=1)
        out = os.path.join(args.out_dir, f'warp_check_{j:02d}_idx{i}.jpg')
        Image.fromarray(montage).save(out, quality=92)
        print(f'[*] wrote {out}')

    # Timing: batch of 16 warps
    sample_paths = [rows[i][0] for i in idx if os.path.isfile(rows[i][0])]
    while len(sample_paths) < 16 and rows:
        sample_paths.extend([r[0] for r in rows[: 16 - len(sample_paths)]])
    sample_paths = sample_paths[:16]

    tiles = []
    from aerial_warp import _aerial_to_tensor

    for path in sample_paths:
        tiles.append(_aerial_to_tensor(path))
    batch = torch.stack(tiles, dim=0).to(device)
    grid = warper.grid.expand(batch.size(0), -1, -1, -1)

    for label, dev in [('CPU', torch.device('cpu')), ('GPU', device)]:
        if label == 'GPU' and not torch.cuda.is_available():
            continue
        b = batch.to(dev)
        g = grid.to(dev)
        v0 = warper.v0
        shift = warper.shift
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            band = F.grid_sample(b, g, mode='bilinear', align_corners=False)
            band = torch.roll(band, shifts=shift, dims=-1)
            out = torch.zeros(b.size(0), 3, args.img_h, args.img_w, device=dev)
            out[:, :, v0:, :] = band
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 3
        print(f'[*] warp batch=16 on {label}: {elapsed * 1000:.2f} ms/iter')

    print('[*] Inspect montages: road/veg in warped panel should roughly match real pano.')


if __name__ == '__main__':
    main()
