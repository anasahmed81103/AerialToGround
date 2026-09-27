"""
cache_dino_features.py  —  Phase 2: precompute frozen DINOv2 features for full aerial tiles.

The backbone never trains, so features are extracted once and read from disk.
Each split becomes one fp16 .npy memmap of shape (N, C, g, g), in CSV row order.

    cache/<tag>/train.npy   cache/<tag>/val.npy   cache/<tag>/meta.json

Safe to interrupt and rerun — resumes from the last finished batch.

Usage:
    python cache_dino_features.py
    python cache_dino_features.py --max_samples 64 --tag smoke
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TileDataset(Dataset):
    def __init__(self, paths, size, crop=0):
        self.paths = paths
        self.size = size
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.crop > 0:
            w, h = img.size
            l, t = (w - self.crop) // 2, (h - self.crop) // 2
            img = img.crop((l, t, l + self.crop, t + self.crop))
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return (t - IMAGENET_MEAN) / IMAGENET_STD


class VGGConv5(nn.Module):
    """Frozen ImageNet VGG-16 up to relu5_3 (stride 16, 512 ch) — CrossNet's backbone."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        self.body = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:30]

    def forward(self, x):
        return self.body(x)


def load_backbone(model_id, device):
    """Returns (model, n_register_tokens, channels, stride). n_reg=None means a CNN."""
    if model_id == 'vgg16':
        model = VGGConv5().to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, None, 512, 16
    if 'registers' in model_id:
        from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
            Dinov2WithRegistersModel as Model,
        )
    else:
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Model as Model
    model = Model.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_reg = int(getattr(model.config, 'num_register_tokens', 0))
    return model, n_reg, model.config.hidden_size, 14


def get_args():
    p = argparse.ArgumentParser(description='Cache frozen DINOv2 aerial features')
    p.add_argument('--train_csv', default='cvpr_train_v2.csv')
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--model_id', default='facebook/dinov2-with-registers-small',
                   help="HF DINOv2 id, or 'vgg16' for frozen ImageNet VGG-16 relu5_3")
    p.add_argument('--size', type=int, default=336,
                   help='Tile is resized to this (multiple of the backbone stride)')
    p.add_argument('--crop', type=int, default=0,
                   help='Center-crop this many px of the 750px tile first (0 = full tile)')
    p.add_argument('--tag', default='dinov2s_reg_336')
    p.add_argument('--out_root', default='cache')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_samples', type=int, default=0)
    return p.parse_args()


@torch.inference_mode()
def cache_split(name, csv_path, model, n_reg, c, stride, args, device, out_dir):
    paths = [l.split(',')[0].strip() for l in open(csv_path) if l.strip()]
    if args.max_samples > 0:
        paths = paths[:args.max_samples]
    n = len(paths)
    g = args.size // stride

    npy_path = os.path.join(out_dir, f'{name}.npy')
    prog_path = os.path.join(out_dir, f'{name}.progress.json')
    done = 0
    if os.path.isfile(npy_path) and os.path.isfile(prog_path):
        arr = np.load(npy_path, mmap_mode='r+')
        if arr.shape != (n, c, g, g):
            raise SystemExit(f'{npy_path} has shape {arr.shape}, expected {(n, c, g, g)}')
        done = json.load(open(prog_path))['done']
    else:
        arr = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.float16,
                                        shape=(n, c, g, g))
    if done >= n:
        print(f'[*] {name}: already complete ({n})')
        return
    print(f'[*] {name}: {n} tiles -> {npy_path}  (resume at {done})', flush=True)

    ds = TileDataset(paths[done:], args.size, args.crop)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    t0 = time.time()
    idx = done
    for bi, x in enumerate(dl):
        if n_reg is None:
            feat = model(x.to(device))
            b = feat.size(0)
        else:
            tokens = model(pixel_values=x.to(device)).last_hidden_state
            patches = tokens[:, 1 + n_reg:, :]
            b = patches.size(0)
            feat = patches.transpose(1, 2).reshape(b, c, g, g)
        arr[idx:idx + b] = feat.half().cpu().numpy()
        idx += b
        if bi % 50 == 0 or idx == n:
            arr.flush()
            json.dump({'done': idx}, open(prog_path, 'w'))
            rate = (idx - done) / max(time.time() - t0, 1e-6)
            eta = (n - idx) / max(rate, 1e-6) / 60
            print(f'  [{name}] {idx}/{n}  {rate:.1f} img/s  eta {eta:.1f} min', flush=True)
    arr.flush()
    json.dump({'done': idx}, open(prog_path, 'w'))


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    model, n_reg, c, stride = load_backbone(args.model_id, device)
    g = args.size // stride
    print(f'[*] {args.model_id} on {device}  registers={n_reg}  crop={args.crop}  '
          f'grid={g}x{g}  dim={c}')

    splits = [('train', args.train_csv), ('val', args.val_csv)]
    for name, csv_path in splits:
        cache_split(name, csv_path, model, n_reg, c, stride, args, device, out_dir)

    meta = {
        'model_id': args.model_id,
        'input': (f'center {args.crop}px crop of 750px tile' if args.crop else
                  'full 750x750 tile') + f', resized to {args.size}',
        'crop': args.crop,
        'size': args.size,
        'grid': g,
        'dim': c,
        'dtype': 'float16',
        'layout': 'N, C, H, W  (row order = CSV order)',
        'train_csv': args.train_csv,
        'val_csv': args.val_csv,
        'max_samples': args.max_samples,
    }
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)
    print(f'[*] Done -> {out_dir}')


if __name__ == '__main__':
    main()
