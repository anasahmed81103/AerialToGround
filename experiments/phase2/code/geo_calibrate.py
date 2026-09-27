"""
geo_calibrate.py  —  Measure the aerial→panorama polar alignment on CVUSA.

Polar-warps the aerial semantic map into the lower (ground) half of the
panorama and searches over azimuth offset and handedness for the setting
that best matches the panorama's own semantic map.

Answers:
  • which panorama column faces north (azimuth offset)
  • clockwise vs counter-clockwise azimuth (handedness)
  • whether "hflip aerial + reverse panorama" is a geometrically valid augmentation

Usage:
    python geo_calibrate.py --n 400
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from semantic_metrics import CLASS_NAMES


def polar_grid(hp, wp, horizon_frac, clockwise=True):
    """grid_sample grid mapping panorama ground-band pixels to aerial coords.

    Bottom panorama row -> aerial centre (camera), horizon row -> aerial edge.
    Column u -> azimuth 2*pi*u/W measured from aerial "up" (north).
    """
    v0 = int(round(hp * horizon_frac))
    rows = torch.arange(v0, hp, dtype=torch.float32) + 0.5
    r = (hp - rows) / (hp - v0)
    cols = torch.arange(wp, dtype=torch.float32) + 0.5
    theta = 2 * np.pi * cols / wp
    s = 1.0 if clockwise else -1.0
    x = s * r[:, None] * torch.sin(theta)[None, :]
    y = -r[:, None] * torch.cos(theta)[None, :]
    return torch.stack([x, y], dim=-1), v0


def load_label(path, hw=None):
    img = Image.open(path)
    if hw is not None:
        img = img.resize((hw[1], hw[0]), Image.NEAREST)
    return torch.from_numpy(np.array(img)).long()


def search(aer, gnd, grid, v0, cls):
    """Return per-shift pooled IoU for class `cls` (shift k = north at column k)."""
    n = aer.size(0)
    a = (aer == cls).float().unsqueeze(1)
    g = grid.unsqueeze(0).expand(n, -1, -1, -1)
    warped = F.grid_sample(a, g, mode='nearest', align_corners=False).squeeze(1) > 0.5
    band = gnd[:, v0:, :] == cls
    wp = band.size(-1)
    ious = torch.zeros(wp)
    for k in range(wp):
        w = torch.roll(warped, shifts=k, dims=-1)
        inter = (w & band).sum().item()
        union = (w | band).sum().item()
        ious[k] = inter / max(union, 1)
    return ious


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='cvpr_train_v2.csv')
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--pano_h', type=int, default=56)
    p.add_argument('--pano_w', type=int, default=308)
    p.add_argument('--horizon', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default=os.path.join('experiments', 'phase2'))
    return p.parse_args()


def main():
    args = get_args()
    rows = [l.strip().split(',') for l in open(args.csv) if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]

    hw = (args.pano_h, args.pano_w)
    gnd = torch.stack([load_label(r[2], hw) for r in rows])
    aer = torch.stack([load_label(r[3]) for r in rows])

    results = {'n': len(rows), 'pano_grid': list(hw), 'horizon_frac': args.horizon}
    best = None
    road = CLASS_NAMES.index('road')
    for clockwise in (True, False):
        grid, v0 = polar_grid(*hw, args.horizon, clockwise)
        ious = search(aer, gnd, grid, v0, road)
        k = int(ious.argmax())
        entry = {
            'best_shift_col': k,
            'north_at_frac': k / args.pano_w,
            'road_iou_best': float(ious.max()),
            'road_iou_mean_over_shifts': float(ious.mean()),
        }
        results['clockwise' if clockwise else 'counterclockwise'] = entry
        print(f"{'cw ' if clockwise else 'ccw'}  best shift {k:3d}/{args.pano_w}"
              f"  road IoU {ious.max():.3f}  (chance {ious.mean():.3f})")
        if best is None or entry['road_iou_best'] > best[1]['road_iou_best']:
            best = (clockwise, entry)

    clockwise, entry = best
    grid, v0 = polar_grid(*hw, args.horizon, clockwise)
    per_class = {}
    for ci, name in enumerate(CLASS_NAMES):
        ious = search(aer, gnd, grid, v0, ci)
        per_class[name] = {
            'iou_at_best_shift': float(ious[entry['best_shift_col']]),
            'mean_over_shifts': float(ious.mean()),
        }
    results['per_class_at_best'] = per_class

    # Flip test: hflip aerial + reverse panorama must keep the same best shift.
    ious_f = search(torch.flip(aer, dims=[-1]), torch.flip(gnd, dims=[-1]), grid, v0, road)
    kf = int(ious_f.argmax())
    results['flip_test'] = {
        'best_shift_after_flip': kf,
        'road_iou_after_flip': float(ious_f.max()),
        'road_iou_after_flip_at_original_shift': float(ious_f[entry['best_shift_col']]),
        'consistent': abs(kf - entry['best_shift_col']) <= 2
                      or abs(kf - entry['best_shift_col']) >= args.pano_w - 2,
    }
    print(f"flip test: best shift {kf} vs {entry['best_shift_col']}  "
          f"consistent={results['flip_test']['consistent']}")
    for name, v in per_class.items():
        print(f"  {name:<11} IoU {v['iou_at_best_shift']:.3f}  (chance {v['mean_over_shifts']:.3f})")

    # Visual check on the full RGB tile
    os.makedirs(os.path.join(args.out_dir, 'showcase'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'results'), exist_ok=True)
    vis_hw = (224, 1232)
    vgrid, vv0 = polar_grid(*vis_hw, args.horizon, clockwise)
    shift_vis = round(entry['best_shift_col'] * vis_hw[1] / args.pano_w)
    strips = []
    for r in rows[:3]:
        pano = np.asarray(Image.open(r[1]).convert('RGB').resize((vis_hw[1], vis_hw[0])))
        tile = torch.from_numpy(np.asarray(Image.open(r[0]).convert('RGB'))).permute(2, 0, 1)
        tile = tile.float().unsqueeze(0) / 255.0
        warped = F.grid_sample(tile, vgrid.unsqueeze(0), mode='bilinear',
                               align_corners=False)[0]
        warped = torch.roll(warped, shifts=shift_vis, dims=-1)
        warped = (warped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        synth = np.zeros_like(pano)
        synth[vv0:] = warped
        strips.append(np.concatenate([pano, synth], axis=0))
    out_img = os.path.join(args.out_dir, 'showcase', 'geo_calibration.jpg')
    Image.fromarray(np.concatenate(strips, axis=0)).save(out_img, quality=90)

    results['best'] = {'clockwise': clockwise, **entry}
    out_json = os.path.join(args.out_dir, 'results', 'geo_calibration.json')
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'[*] Wrote {out_json}\n[*] Wrote {out_img}')


if __name__ == '__main__':
    main()
