"""
train_polar.py  —  Phase 2 Stage-1 trainer: cached frozen DINOv2 features -> PolarLayoutNet.

Protocol
  • model selection on a minival carved from TRAIN (never the test split)
  • final numbers on the full CVUSA test split (cvpr_val_v2.csv, 8884) at 224x1232
  • everything for a run lands in experiments/phase2/runs/<name>/

Usage:
    python train_polar.py --name polar_full
    python train_polar.py --name no_prior --prior none
    python train_polar.py --name smoke --cache cache/smoke --max_samples 32 --epochs 1
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from polar_layout_model import PolarLayoutNet
from semantic_metrics import CLASS_NAMES, ConfusionMeter, colorize_label
from train_crossnet import dice_loss, focal_ce


def get_args():
    p = argparse.ArgumentParser(description='Phase 2 polar layout trainer')
    p.add_argument('--name', required=True)
    p.add_argument('--train_csv', default='cvpr_train_v2.csv')
    p.add_argument('--test_csv', default='cvpr_val_v2.csv')
    p.add_argument('--cache', default=os.path.join('cache', 'dinov2s_reg_336'))
    p.add_argument('--runs_dir', default=os.path.join('experiments', 'phase2', 'runs'))
    p.add_argument('--max_samples', type=int, default=0)
    p.add_argument('--minival', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no_flip', action='store_true')
    p.add_argument('--focal_gamma', type=float, default=2.0)

    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--radial_bins', type=int, default=32)
    p.add_argument('--out_h', type=int, default=32)
    p.add_argument('--out_w', type=int, default=160)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--horizon', type=float, default=0.5)
    p.add_argument('--prior', choices=('learned', 'fixed', 'none'), default='learned')
    p.add_argument('--no_offsets', action='store_true')
    p.add_argument('--mapping', choices=('polar', 'dense'), default='polar')

    p.add_argument('--skip_test', action='store_true')
    p.add_argument('--save_n', type=int, default=8)
    return p.parse_args()


def read_rows(path, max_samples=0):
    rows = [l.strip().split(',') for l in open(path) if l.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def label_cache(rows, hw, path):
    if os.path.isfile(path):
        arr = np.load(path, mmap_mode='r')
        if arr.shape == (len(rows), *hw):
            return path
    print(f'[*] Building label cache {path} ...', flush=True)
    arr = np.zeros((len(rows), *hw), dtype=np.uint8)
    for i, r in enumerate(rows):
        arr[i] = np.array(Image.open(r[2]).resize((hw[1], hw[0]), Image.NEAREST))
        if i % 5000 == 0:
            print(f'    {i}/{len(rows)}', flush=True)
    np.save(path, arr)
    return path


class FeatDataset(Dataset):
    """Returns (features, label). Memmaps are opened lazily per worker."""

    def __init__(self, feat_path, indices, label_path=None, label_pngs=None, flip=False):
        self.feat_path = feat_path
        self.label_path = label_path
        self.label_pngs = label_pngs
        self.indices = np.asarray(indices)
        self.flip = flip
        self._feats = None
        self._labels = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._feats is None:
            self._feats = np.load(self.feat_path, mmap_mode='r')
            if self.label_path:
                self._labels = np.load(self.label_path, mmap_mode='r')
        j = int(self.indices[i])
        f = torch.from_numpy(np.array(self._feats[j], dtype=np.float32))
        if self.label_pngs is not None:
            y = torch.from_numpy(np.array(Image.open(self.label_pngs[j]))).long()
        else:
            y = torch.from_numpy(np.array(self._labels[j])).long()
        if self.flip and random.random() < 0.5:
            f = torch.flip(f, dims=[-1])
            y = torch.flip(y, dims=[-1])
        return f, y


@torch.inference_mode()
def evaluate(model, loader, device, amp=False):
    was_training = model.training
    model.eval()
    meter = ConfusionMeter(len(CLASS_NAMES))
    for f, y in loader:
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp and device.type == 'cuda'):
            logits = model(f)
        logits = logits.float()
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear',
                                   align_corners=True)
        meter.update(logits.argmax(1), y)
    if was_training:
        model.train()
    return meter.compute()


def metrics_dict(m):
    return {
        'miou': m['miou'],
        'acc': m['acc'],
        'iou': {n: m['iou'][i].item() for i, n in enumerate(CLASS_NAMES)},
    }


def save_montages(model, rows, feat_path, device, out_dir, n):
    feats = np.load(feat_path, mmap_mode='r')
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i in range(min(n, len(rows))):
        f = torch.from_numpy(np.array(feats[i], dtype=np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(f).float()
        logits = F.interpolate(logits, size=(224, 1232), mode='bilinear', align_corners=True)
        pred = colorize_label(logits.argmax(1)[0])
        gt = colorize_label(np.array(Image.open(rows[i][2])))
        pano = np.asarray(Image.open(rows[i][1]).convert('RGB').resize((1232, 224)))
        aerial = np.asarray(Image.open(rows[i][0]).convert('RGB').resize((224, 224)))
        pad = np.zeros((224, 224, 3), dtype=np.uint8)
        grid = np.concatenate([
            np.concatenate([aerial, pano], axis=1),
            np.concatenate([pad, gt], axis=1),
            np.concatenate([pad, pred], axis=1),
        ], axis=0)
        Image.fromarray(grid).save(os.path.join(out_dir, f'test_{i:04d}.jpg'), quality=90)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, 'config.json'), 'w'), indent=2)

    hw = (args.out_h, args.out_w)
    train_rows = read_rows(args.train_csv, args.max_samples)
    test_rows = read_rows(args.test_csv, args.max_samples)
    train_feats = os.path.join(args.cache, 'train.npy')
    test_feats = os.path.join(args.cache, 'val.npy')
    n_cached = np.load(train_feats, mmap_mode='r').shape[0]
    if n_cached != len(train_rows):
        raise SystemExit(f'Cache has {n_cached} train rows, CSV has {len(train_rows)}. '
                         'Use matching --max_samples.')
    train_labels = label_cache(train_rows, hw, os.path.join(
        args.cache, f'labels_{hw[0]}x{hw[1]}_train.npy'))

    perm = np.random.RandomState(args.seed).permutation(len(train_rows))
    n_mv = min(args.minival, len(train_rows) // 4)
    mv_idx, tr_idx = perm[:n_mv], perm[n_mv:]
    np.save(os.path.join(run_dir, 'minival_indices.npy'), mv_idx)

    train_dl = DataLoader(
        FeatDataset(train_feats, tr_idx, label_path=train_labels, flip=not args.no_flip),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    mv_dl = DataLoader(
        FeatDataset(train_feats, mv_idx, label_path=train_labels),
        batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    feat_shape = np.load(train_feats, mmap_mode='r').shape
    in_ch = feat_shape[1]
    model = PolarLayoutNet(
        in_ch=in_ch, dim=args.dim, num_classes=len(CLASS_NAMES),
        radial_bins=args.radial_bins, out_hw=hw, horizon_frac=args.horizon,
        heads=args.heads, layers=args.layers,
        use_offsets=not args.no_offsets, prior_mode=args.prior,
        mapping=args.mapping, in_grid=feat_shape[2],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[*] {args.name}: {n_params / 1e6:.2f} M trainable params  '
          f'train {len(tr_idx)}  minival {len(mv_idx)}  test {len(test_rows)}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    curve_path = os.path.join(run_dir, 'minival_curve.csv')
    best_miou, step, t_start = -1.0, 0, time.time()
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        for bi, (f, y) in enumerate(train_dl, start=1):
            f = f.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=args.amp and device.type == 'cuda'):
                logits = model(f)
            logits = logits.float()
            loss = focal_ce(logits, y, gamma=args.focal_gamma) + dice_loss(
                logits, y, len(CLASS_NAMES))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            if bi % 200 == 0 or bi == len(train_dl):
                print(f'  [ep {epoch:02d}] {bi}/{len(train_dl)}  loss {running / bi:.4f}  '
                      f'lr {sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.0f}s)',
                      flush=True)

        epoch_times.append(time.time() - t0)
        m = evaluate(model, mv_dl, device, args.amp)
        row = {'epoch': epoch, 'step': step, 'loss': running / len(train_dl),
               'minival_miou': m['miou'], 'minival_acc': m['acc'],
               **{f'iou_{n}': m['iou'][i].item() for i, n in enumerate(CLASS_NAMES)}}
        write_header = not os.path.isfile(curve_path)
        with open(curve_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f'[ep {epoch:02d}] minival mIoU {m["miou"]:.4f}  ' + '  '.join(
            f'{n} {m["iou"][i]:.3f}' for i, n in enumerate(CLASS_NAMES)), flush=True)

        ck = {'model': model.state_dict(), 'epoch': epoch, 'step': step,
              'minival_miou': m['miou'], 'config': vars(args)}
        torch.save(ck, os.path.join(run_dir, 'last.pt'))
        if m['miou'] > best_miou:
            best_miou = m['miou']
            torch.save(ck, os.path.join(run_dir, 'best.pt'))
            print(f'    new best -> {run_dir}/best.pt', flush=True)

    summary = {
        'name': args.name,
        'params_trainable': n_params,
        'peak_vram_gb': (torch.cuda.max_memory_allocated() / 1024 ** 3
                         if device.type == 'cuda' else 0.0),
        'sec_per_epoch': float(np.mean(epoch_times)),
        'train_minutes': (time.time() - t_start) / 60,
        'best_minival_miou': best_miou,
    }

    if not args.skip_test:
        best = torch.load(os.path.join(run_dir, 'best.pt'), map_location=device,
                          weights_only=False)
        model.load_state_dict(best['model'])
        test_dl = DataLoader(
            FeatDataset(test_feats, np.arange(len(test_rows)),
                        label_pngs=[r[2] for r in test_rows]),
            batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        print(f'[*] Test eval on {len(test_rows)} images at 224x1232 ...', flush=True)
        t = evaluate(model, test_dl, device, args.amp)
        summary['test_full_res'] = {'n': len(test_rows), 'best_epoch': best['epoch'],
                                    **metrics_dict(t)}
        print(f'[test] mIoU {t["miou"]:.4f}  ' + '  '.join(
            f'{n} {t["iou"][i]:.3f}' for i, n in enumerate(CLASS_NAMES)), flush=True)
        save_montages(model, test_rows, test_feats, device,
                      os.path.join(run_dir, 'showcase'), args.save_n)

    json.dump(summary, open(os.path.join(run_dir, 'results.json'), 'w'), indent=2)
    print(f'[*] Wrote {run_dir}/results.json', flush=True)


if __name__ == '__main__':
    main()
