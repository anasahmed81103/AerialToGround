"""
final_eval.py — Publication-grade val harness (no training).

Configs (order E → C → D → B → A), checkpoints from experiments/phase2/MODELS.txt.
Writes per-config NPZ + JSON after each config completes.

Smoke (config E, 64 samples only):
  python final_eval.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from eval_gan_pix2pix import load_generator
from gan_layout_codec import (
    class_indices_to_onehot,
    encoding_stats,
    load_gt_label_tensor,
)
from gan_metrics import (
    per_image_lpips,
    per_image_psnr,
    per_image_sharpness_diff,
    per_image_ssim,
    tensor_to_uint8_nhwc,
)
from polar_layout_model import PolarLayoutNet
from train_gan_pix2pix import generator_forward


@dataclass(frozen=True)
class EvalConfig:
    key: str
    slug: str
    gan_ckpt: str
    generator: str
    aux_aerial: bool
    layout: str  # 'pred' | 'gt'


CONFIG_ORDER: list[EvalConfig] = [
    EvalConfig(
        'E', 'spade_aerial_v2_pred',
        'experiments/phase2/runs/spade_aerial_v2/ckpts/G_step0032000.pt',
        'spade', True, 'pred',
    ),
    EvalConfig(
        'C', 'spade_v2_pred',
        'experiments/phase2/runs/spade_v2/ckpts/G_step0033300.pt',
        'spade', False, 'pred',
    ),
    EvalConfig(
        'D', 'spade_aerial_v2_gt',
        'experiments/phase2/runs/spade_aerial_v2/ckpts/G_step0032000.pt',
        'spade', True, 'gt',
    ),
    EvalConfig(
        'B', 'spade_v2_gt',
        'experiments/phase2/runs/spade_v2/ckpts/G_step0033300.pt',
        'spade', False, 'gt',
    ),
    EvalConfig(
        'A', 'pix2pix_random_d_gt',
        'experiments/phase2/runs/gan_v2_random_d/ckpts/G_step0231066.pt',
        'unet', False, 'gt',
    ),
]

DEFAULT_POLAR_CKPT = 'experiments/phase2/runs/polar_full/best.pt'
DEFAULT_FEAT_VAL = os.path.join('cache', 'dinov2s_reg_336', 'val.npy')
REAL_SUBDIR = 'real_256x512'
KEEP_N = 24


def read_csv_rows(csv_path: str, max_samples: int | None) -> list[dict]:
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 3:
                continue
            rows.append({'aerial': parts[0], 'ground': parts[1], 'label': parts[2]})
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def save_chw_png_rgb(t: torch.Tensor, path: str) -> None:
    """Single image (3, H, W) in [-1, 1]."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    u = tensor_to_uint8_nhwc(t.unsqueeze(0))[0]
    Image.fromarray(u).save(path)


def load_polar_model(polar_ckpt: str, feat_val_path: str, device: torch.device) -> PolarLayoutNet:
    ck = torch.load(polar_ckpt, map_location=device, weights_only=False)
    cfg = ck['config']
    feat_shape = np.load(feat_val_path, mmap_mode='r').shape
    model = PolarLayoutNet(
        in_ch=int(feat_shape[1]),
        dim=int(cfg['dim']),
        num_classes=4,
        radial_bins=int(cfg['radial_bins']),
        out_hw=(int(cfg['out_h']), int(cfg['out_w'])),
        horizon_frac=float(cfg['horizon']),
        heads=int(cfg['heads']),
        layers=int(cfg['layers']),
        use_offsets=not bool(cfg.get('no_offsets', False)),
        prior_mode=str(cfg['prior']),
        mapping=str(cfg.get('mapping', 'polar')),
        in_grid=int(feat_shape[2]),
    ).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model


@torch.inference_mode()
def predict_layout_maps(
    polar: PolarLayoutNet,
    feat_val_path: str,
    n: int,
    img_h: int,
    img_w: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Stage-1 predicted class indices (N, H, W) uint8."""
    feats = np.load(feat_val_path, mmap_mode='r')
    if feats.shape[0] < n:
        raise SystemExit(f'{feat_val_path} has {feats.shape[0]} rows, need {n}')
    out = np.zeros((n, img_h, img_w), dtype=np.uint8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        f = torch.from_numpy(np.array(feats[start:end], dtype=np.float32)).to(device)
        logits = polar(f).float()
        if logits.shape[-2:] != (img_h, img_w):
            logits = F.interpolate(
                logits, size=(img_h, img_w), mode='bilinear', align_corners=True,
            )
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        out[start:end] = pred
        if start == 0:
            print(f'[*] polar logits -> {tuple(logits.shape)} (after upsample)', flush=True)
    return out


def run_encoding_sanity(
    rows: list[dict],
    pred_maps: np.ndarray,
    num_classes: int,
    img_h: int,
    img_w: int,
    n_check: int = 8,
) -> None:
    print('[*] Layout encoding sanity (predicted vs GT loader) on '
          f'{min(n_check, len(rows))} samples:', flush=True)
    for i in range(min(n_check, len(rows))):
        pred_t = class_indices_to_onehot(pred_maps[i], num_classes)
        gt_t = load_gt_label_tensor(rows[i]['label'], num_classes, img_h, img_w)
        sp, sg = encoding_stats(pred_t), encoding_stats(gt_t)
        print(f'    idx {i}: pred {sp}', flush=True)
        print(f'    idx {i}: gt   {sg}', flush=True)
        for key in ('dtype', 'shape'):
            if sp[key] != sg[key]:
                raise SystemExit(f'Encoding mismatch idx {i}: {key} pred={sp[key]} gt={sg[key]}')
        for tag, st in ('pred', sp), ('gt', sg):
            if st['min'] < -1.01 or st['max'] > 1.01:
                raise SystemExit(f'{tag} idx {i}: values outside [-1,1]: min={st["min"]} max={st["max"]}')
    print('[*] Encoding check passed (dtype/shape/range).', flush=True)


class FinalValDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        layout: str,
        pred_maps: np.ndarray | None,
        num_classes: int,
        img_h: int,
        img_w: int,
        aux_aerial: bool,
    ):
        self.rows = rows
        self.layout = layout
        self.pred_maps = pred_maps
        self.num_classes = num_classes
        self.img_h = img_h
        self.img_w = img_w
        self.aux_aerial = aux_aerial
        self._warper_obj = None

    def _get_warper(self):
        if self._warper_obj is None:
            from aerial_warp import PanoAerialWarper
            self._warper_obj = PanoAerialWarper((self.img_h, self.img_w))
        return self._warper_obj

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        s = self.rows[idx]
        if self.layout == 'gt':
            label_t = load_gt_label_tensor(
                s['label'], self.num_classes, self.img_h, self.img_w,
            )
        else:
            label_t = class_indices_to_onehot(self.pred_maps[idx], self.num_classes)

        g_pil = Image.open(s['ground']).convert('RGB')
        g_pil = g_pil.resize((self.img_w, self.img_h), Image.BILINEAR)
        g_np = np.array(g_pil, dtype=np.float32) / 127.5 - 1.0
        ground_t = torch.from_numpy(g_np.transpose(2, 0, 1))

        if self.aux_aerial:
            aerial_t = self._get_warper().warp_aerial_to_pano(s['aerial'])
            return idx, label_t, ground_t, aerial_t
        return idx, label_t, ground_t


def ensure_real_pngs(
    rows: list[dict],
    out_dir: str,
    img_h: int,
    img_w: int,
) -> str:
    real_dir = os.path.join(out_dir, REAL_SUBDIR)
    os.makedirs(real_dir, exist_ok=True)
    for i, s in enumerate(rows):
        path = os.path.join(real_dir, f'{i:05d}.png')
        if os.path.isfile(path):
            continue
        g_pil = Image.open(s['ground']).convert('RGB')
        g_pil = g_pil.resize((img_w, img_h), Image.BILINEAR)
        g_pil.save(path)
    return real_dir


def compute_clean_fid_kid(gen_dir: str, real_dir: str, device: torch.device) -> dict:
    try:
        from cleanfid import fid as clean_fid
    except ImportError as e:
        raise ImportError('pip install clean-fid') from e

    n_gen = len([f for f in os.listdir(gen_dir) if f.lower().endswith('.png')])
    n_real = len([f for f in os.listdir(real_dir) if f.lower().endswith('.png')])
    fid_score = clean_fid.compute_fid(
        gen_dir, real_dir, mode='clean', device=device,
        num_workers=0, batch_size=32,
    )
    kid_score = clean_fid.compute_kid(
        gen_dir, real_dir, mode='clean', device=device,
        num_workers=0, batch_size=32,
    )
    return {
        'fid_clean': float(fid_score),
        'kid_clean': float(kid_score),
        'n_images_gen': n_gen,
        'n_images_real': n_real,
    }


def prune_gen_dir_keep_samples(gen_dir: str, keep_indices: np.ndarray, samples_dir: str) -> None:
    os.makedirs(samples_dir, exist_ok=True)
    keep_set = {f'{int(i):05d}.png' for i in keep_indices}
    for name in os.listdir(gen_dir):
        if not name.lower().endswith('.png'):
            continue
        src = os.path.join(gen_dir, name)
        if name in keep_set:
            shutil.copy2(src, os.path.join(samples_dir, name))
        os.remove(src)


@torch.inference_mode()
def run_one_config(
    cfg: EvalConfig,
    args: argparse.Namespace,
    device: torch.device,
    rows: list[dict],
    pred_maps: np.ndarray | None,
    real_dir: str,
) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    gen_dir = os.path.join(out_dir, f'_tmp_gen_{cfg.slug}')
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir)

    G, step = load_generator(
        cfg.gan_ckpt,
        args.num_classes,
        args.ngf,
        device,
        generator=cfg.generator,
        img_h=args.img_h,
        img_w=args.img_w,
        aux_aerial=cfg.aux_aerial,
    )
    print(f'[*] Config {cfg.key} ({cfg.slug})  G step={step}  layout={cfg.layout}', flush=True)

    ds = FinalValDataset(
        rows, cfg.layout, pred_maps,
        args.num_classes, args.img_h, args.img_w, cfg.aux_aerial,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n = len(rows)
    keep_idx = np.linspace(0, max(n - 1, 0), KEEP_N).astype(int)
    all_idx: list[int] = []
    ssim_l: list[np.ndarray] = []
    psnr_l: list[np.ndarray] = []
    lpips_l: list[np.ndarray] = []
    sd_l: list[np.ndarray] = []
    timings: list[float] = []

    t_all = time.time()
    for bi, batch in enumerate(dl):
        if cfg.aux_aerial:
            idx, label, real, aerial = batch
            aerial = aerial.to(device)
        else:
            idx, label, real = batch
            aerial = None
        label = label.to(device)
        real = real.to(device)

        t0 = time.perf_counter()
        fake = generator_forward(G, label, aerial, cfg.aux_aerial)
        dt = time.perf_counter() - t0
        per = dt / max(real.shape[0], 1)
        timings.extend([per] * real.shape[0])

        fake_cpu = fake.cpu()
        real_cpu = real.cpu()

        ssim_l.append(per_image_ssim(real_cpu, fake_cpu))
        psnr_l.append(per_image_psnr(real_cpu, fake_cpu))
        lpips_l.append(per_image_lpips(real_cpu, fake_cpu, device))
        sd_l.append(per_image_sharpness_diff(real_cpu, fake_cpu))

        idx_list = idx.tolist() if torch.is_tensor(idx) else list(idx)
        all_idx.extend(int(i) for i in idx_list)
        for j, global_i in enumerate(idx_list):
            save_chw_png_rgb(fake_cpu[j], os.path.join(gen_dir, f'{int(global_i):05d}.png'))

        if args.smoke and (bi + 1) <= 3:
            print(f'    batch {bi}: label {tuple(label.shape)} fake {tuple(fake.shape)} '
                  f'({per * 1000:.1f} ms/img)', flush=True)

    ssim_a = np.concatenate(ssim_l)
    psnr_a = np.concatenate(psnr_l)
    lpips_a = np.concatenate(lpips_l)
    sd_a = np.concatenate(sd_l)
    idx_a = np.asarray(all_idx, dtype=np.int32)

    npz_path = os.path.join(out_dir, f'{cfg.slug}_per_image.npz')
    np.savez(
        npz_path,
        index=idx_a,
        ssim=ssim_a,
        psnr=psnr_a,
        lpips=lpips_a,
        sd=sd_a,
    )
    print(f'[*] wrote {npz_path}', flush=True)

    fid_block = {}
    if not args.skip_fid and n >= 64:
        fid_block = compute_clean_fid_kid(gen_dir, real_dir, device)
    elif args.skip_fid:
        fid_block = {'skipped': True}
    else:
        fid_block = {'skipped': True, 'note': 'n < 64'}

    metrics = {
        'config_key': cfg.key,
        'slug': cfg.slug,
        'gan_ckpt': cfg.gan_ckpt,
        'generator': cfg.generator,
        'aux_aerial': cfg.aux_aerial,
        'layout': cfg.layout,
        'generator_step': step,
        'val_csv': args.val_csv,
        'n_images': n,
        'mean_ssim': float(ssim_a.mean()),
        'mean_psnr': float(psnr_a.mean()),
        'mean_lpips': float(lpips_a.mean()),
        'mean_sd': float(sd_a.mean()),
        'elapsed_sec': round(time.time() - t_all, 2),
        **fid_block,
    }
    if args.smoke:
        metrics['smoke'] = True
        metrics['per_image_sec_mean'] = float(np.mean(timings))
        metrics['per_image_sec_p50'] = float(np.percentile(timings, 50))
        print(f'[*] smoke timing: mean {metrics["per_image_sec_mean"]:.4f}s/img  '
              f'p50 {metrics["per_image_sec_p50"]:.4f}s/img', flush=True)

    json_path = os.path.join(out_dir, f'{cfg.slug}_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'[*] wrote {json_path}', flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    samples_dir = os.path.join(out_dir, f'{cfg.slug}_samples')
    prune_gen_dir_keep_samples(gen_dir, keep_idx, samples_dir)
    try:
        os.rmdir(gen_dir)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Final val eval harness (E,C,D,B,A)')
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--out_dir', default=os.path.join('experiments', 'final_eval'))
    p.add_argument('--polar_ckpt', default=DEFAULT_POLAR_CKPT)
    p.add_argument('--feat_val', default=DEFAULT_FEAT_VAL)
    p.add_argument('--num_classes', type=int, default=4)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--config', default='all',
                   help='Config key E,C,D,B,A or all (order preserved)')
    p.add_argument('--smoke', action='store_true',
                   help='Config E only, 64 samples, extra timing prints')
    p.add_argument('--skip_fid', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_samples = 64
        args.config = 'E'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] device: {device}', flush=True)

    rows = read_csv_rows(args.val_csv, args.max_samples)
    n = len(rows)
    print(f'[*] val rows: {n}', flush=True)
    if n == 0:
        raise SystemExit('No val rows')

    real_dir = ensure_real_pngs(rows, args.out_dir, args.img_h, args.img_w)

    if args.config == 'all':
        configs = CONFIG_ORDER
    else:
        keys = [k.strip().upper() for k in args.config.split(',')]
        by_key = {c.key: c for c in CONFIG_ORDER}
        configs = [by_key[k] for k in keys]

    need_pred = any(c.layout == 'pred' for c in configs)
    pred_maps = None
    if need_pred:
        if not os.path.isfile(args.polar_ckpt):
            raise SystemExit(f'Missing polar ckpt: {args.polar_ckpt}')
        if not os.path.isfile(args.feat_val):
            raise SystemExit(f'Missing val features: {args.feat_val}')
        polar = load_polar_model(args.polar_ckpt, args.feat_val, device)
        pred_maps = predict_layout_maps(
            polar, args.feat_val, n, args.img_h, args.img_w, device, args.batch_size,
        )
        run_encoding_sanity(rows, pred_maps, args.num_classes, args.img_h, args.img_w)

    for cfg in configs:
        if cfg.layout == 'pred' and pred_maps is None:
            raise SystemExit(f'Config {cfg.key} needs predicted layouts')
        run_one_config(cfg, args, device, rows, pred_maps, real_dir)

    if args.smoke:
        print('[*] Smoke finished — not running full 8884×5 until you drop --smoke.', flush=True)


if __name__ == '__main__':
    main()
