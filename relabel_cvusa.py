"""
relabel_cvusa.py  —  Phase 1: rewrite CVUSA 4-class maps with SegFormer-ADE20K.

Leaves the original SegNet annotations untouched. Writes:
    cvpr_subset/labels_v2/ground/<id>.png
    cvpr_subset/labels_v2/aerial/<id>.png
    cvpr_train_v2.csv
    cvpr_val_v2.csv

Safe to interrupt and rerun — existing PNGs are skipped.

Usage:
    python relabel_cvusa.py
    python relabel_cvusa.py --max_samples 64          # smoke test
    python relabel_cvusa.py --split val               # val only
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from semantic_metrics import colorize_label


# ADE20K (150) → 4-class CrossNet ids
# 0 sky/other  1 vegetation  2 road  3 building
_ADE_TO4 = np.zeros(150, dtype=np.uint8)  # default 0 = sky/other

for _i in (2, 21, 26, 60, 113, 128):          # sky, water, sea, river, waterfall, lake
    _ADE_TO4[_i] = 0
for _i in (4, 9, 16, 17, 29, 34, 46, 66, 68, 72, 94):  # tree/grass/plant/field/...
    _ADE_TO4[_i] = 1
for _i in (6, 11, 13, 52, 54, 91, 20, 80, 83, 102):    # road/sidewalk/earth + vehicles
    _ADE_TO4[_i] = 2
for _i in (0, 1, 8, 14, 25, 32, 38, 42, 48, 61, 79, 84, 86):  # building/wall/house/...
    _ADE_TO4[_i] = 3


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _center_crop(img: Image.Image, hw: tuple[int, int]) -> Image.Image:
    H, W = hw
    w0, h0 = img.size
    if w0 < W or h0 < H:
        return img.resize((W, H), Image.BILINEAR)
    left = (w0 - W) // 2
    top = (h0 - H) // 2
    return img.crop((left, top, left + W, top + H))


def _load_segformer(device: torch.device):
    try:
        # Import SegFormer directly. Auto* pulls torchaudio via audio_utils,
        # and a CPU torchaudio wheel will crash on Windows (WinError 127).
        from transformers.models.segformer.image_processing_segformer import (
            SegformerImageProcessor,
        )
        from transformers.models.segformer.modeling_segformer import (
            SegformerForSemanticSegmentation,
        )
    except ImportError as exc:
        raise SystemExit(
            'transformers is required for relabeling. Install with:\n'
            '    pip install transformers\n'
        ) from exc

    model_id = 'nvidia/segformer-b0-finetuned-ade-512-512'
    print(f'[*] Loading {model_id} ...')
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


@torch.inference_mode()
def _segment(processor, model, pil_img: Image.Image, out_hw: tuple[int, int],
             device: torch.device) -> np.ndarray:
    """Return a (H, W) uint8 4-class map."""
    inputs = processor(images=pil_img.convert('RGB'), return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits  # (1, 150, h, w)
    logits = F.interpolate(logits, size=out_hw, mode='bilinear', align_corners=False)
    ade = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    ade = np.clip(ade, 0, 149)
    return _ADE_TO4[ade]


def _read_csv(path: str) -> list[tuple[str, str, str]]:
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


def _save_png(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode='L').save(path)


def get_args():
    p = argparse.ArgumentParser(description='Relabel CVUSA with SegFormer-ADE20K')
    p.add_argument('--train_csv', default='cvpr_train.csv')
    p.add_argument('--val_csv', default='cvpr_val.csv')
    p.add_argument('--split', choices=('all', 'train', 'val'), default='all')
    p.add_argument('--out_root', default=os.path.join('cvpr_subset', 'labels_v2'))
    p.add_argument('--max_samples', type=int, default=0)
    p.add_argument('--ground_size', type=int, nargs=2, default=(224, 1232))
    p.add_argument('--aerial_size', type=int, nargs=2, default=(224, 224))
    p.add_argument('--preview_n', type=int, default=8)
    p.add_argument('--preview_dir', default=os.path.join('outputs', 'relabel_preview'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--aerial_mode', choices=('crop', 'full'), default='crop',
                   help='crop = centre 224 crop (Phase 1); full = whole tile resized (Phase 2)')
    p.add_argument('--csv_tag', default='v2',
                   help='Output manifests are cvpr_{train,val}_<tag>.csv')
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] Device : {device}')
    if device.type == 'cpu':
        print('[!] CPU relabel will be slow. A GPU is strongly recommended.')

    processor, model = _load_segformer(device)

    jobs = []
    if args.split in ('all', 'train'):
        jobs.append(('train', args.train_csv, f'cvpr_train_{args.csv_tag}.csv'))
    if args.split in ('all', 'val'):
        jobs.append(('val', args.val_csv, f'cvpr_val_{args.csv_tag}.csv'))

    gH, gW = args.ground_size
    aH, aW = args.aerial_size
    ground_dir = os.path.join(args.out_root, 'ground')
    aerial_dir = os.path.join(args.out_root,
                              'aerial' if args.aerial_mode == 'crop' else 'aerial_full')
    os.makedirs(ground_dir, exist_ok=True)
    os.makedirs(aerial_dir, exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)

    previews = 0
    t0 = time.time()
    done = skipped = failed = 0

    for split_name, src_csv, dst_csv in jobs:
        rows = _read_csv(src_csv)
        if args.max_samples > 0:
            rows = rows[:args.max_samples]
        print(f'[*] {split_name}: {len(rows)} pairs from {src_csv}')
        out_rows = []

        for i, (aerial_p, ground_p, *_rest) in enumerate(rows, start=1):
            sid = _stem(ground_p)
            g_out = os.path.join(ground_dir, f'{sid}.png').replace('\\', '/')
            a_out = os.path.join(aerial_dir, f'{sid}.png').replace('\\', '/')
            out_rows.append(f'{aerial_p},{ground_p},{g_out},{a_out}')

            need_g = args.overwrite or not os.path.isfile(g_out)
            need_a = args.overwrite or not os.path.isfile(a_out)
            if not need_g and not need_a:
                skipped += 1
            else:
                try:
                    if need_g:
                        gp = Image.open(ground_p).convert('RGB')
                        if gp.size != (gW, gH):
                            gp = gp.resize((gW, gH), Image.BILINEAR)
                        gmap = _segment(processor, model, gp, (gH, gW), device)
                        _save_png(gmap, g_out)
                        if previews < args.preview_n:
                            vis = np.concatenate([
                                np.asarray(gp),
                                colorize_label(gmap),
                            ], axis=0)
                            Image.fromarray(vis).save(
                                os.path.join(args.preview_dir, f'{sid}_ground.jpg'),
                                quality=90,
                            )
                    if need_a:
                        ap = Image.open(aerial_p).convert('RGB')
                        if args.aerial_mode == 'crop':
                            ap = _center_crop(ap, (aH, aW))
                        else:
                            ap = ap.resize((aW, aH), Image.BICUBIC)
                        amap = _segment(processor, model, ap, (aH, aW), device)
                        _save_png(amap, a_out)
                        if previews < args.preview_n:
                            vis = np.concatenate([
                                np.asarray(ap),
                                colorize_label(amap),
                            ], axis=0)
                            Image.fromarray(vis).save(
                                os.path.join(args.preview_dir, f'{sid}_aerial.jpg'),
                                quality=90,
                            )
                            previews += 1
                    done += 1
                except Exception as exc:
                    failed += 1
                    print(f'    [fail] {sid}: {exc}')

            if i % 25 == 0 or i == len(rows):
                elapsed = time.time() - t0
                rate = (done + skipped) / max(elapsed, 1e-6)
                print(f'  [{split_name}] {i}/{len(rows)}  '
                      f'new {done}  skip {skipped}  fail {failed}  '
                      f'{rate:.2f} img/s  ({elapsed/60:.1f} min)')

        with open(dst_csv, 'w', newline='') as fh:
            fh.write('\n'.join(out_rows) + ('\n' if out_rows else ''))
        print(f'[*] Wrote {len(out_rows)} rows -> {dst_csv}')

    print(f'\n[*] Relabel finished. new={done} skipped={skipped} failed={failed}  '
          f'time {(time.time()-t0)/60:.1f} min')
    print(f'    Ground labels : {ground_dir}')
    print(f'    Aerial labels : {aerial_dir}')
    print(f'    Previews      : {args.preview_dir}')


if __name__ == '__main__':
    main()
