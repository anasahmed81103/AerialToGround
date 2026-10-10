"""
eval_gan_pix2pix.py — SSIM / LPIPS / FID on GT semantic maps -> GAN RGB.

Training uses ground-truth labels_v2/ground/ (standard Pix2Pix). This script
evaluates the same pairing: GT layout in, compare synthesised RGB to real pano.

Streams batches (does not load the full val set into RAM).

Usage
-----
  python eval_gan_pix2pix.py \\
      --gan_ckpt outputs/ckpts_gan/G_step0197766.pt \\
      --val_csv cvpr_val_v2.csv \\
      --max_samples 512

Full val (8,884): omit --max_samples (GPU / RunPod).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from gan_metrics import (
    compute_fid_from_features,
    fid_features,
    lpips_sum,
    ssim_sum,
)
from train_gan_pix2pix import GANDataset, build_generator, generator_forward


def load_generator(
    ckpt_path: str,
    num_classes: int,
    ngf: int,
    device: torch.device,
    *,
    generator: str = 'unet',
    img_h: int = 256,
    img_w: int = 512,
    aux_aerial: bool = False,
):
    class _Args:
        pass

    a = _Args()
    a.generator = generator
    a.num_classes = num_classes
    a.ngf = ngf
    a.img_h = img_h
    a.img_w = img_w
    a.aux_aerial = aux_aerial
    G = build_generator(a).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(ck['model'])
    G.eval()
    step = ck.get('step', None)
    return G, step


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description='Evaluate Pix2Pix GAN (GT labels -> RGB)')
    p.add_argument('--gan_ckpt', required=True)
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_classes', type=int, default=4)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--generator', choices=('unet', 'spade'), default='unet')
    p.add_argument('--aux_aerial', action='store_true',
                   help='SPADE: also condition on warped aerial RGB (CSV col 0).')
    p.add_argument('--skip_fid', action='store_true',
                   help='Skip FID (faster; use for tiny subsets).')
    p.add_argument('--skip_lpips', action='store_true')
    p.add_argument('--out_json', default='')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] device: {device}')

    G, step = load_generator(
        args.gan_ckpt,
        args.num_classes,
        args.ngf,
        device,
        generator=args.generator,
        img_h=args.img_h,
        img_w=args.img_w,
        aux_aerial=args.aux_aerial,
    )
    print(f'[*] generator step: {step}')

    ds = GANDataset(
        csv_path=args.val_csv,
        num_classes=args.num_classes,
        img_h=args.img_h,
        img_w=args.img_w,
        max_samples=args.max_samples,
        aux_aerial=args.aux_aerial,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ssim_total = lpips_total = 0.0
    n_ssim = n_lpips = 0
    real_fid_chunks: list[np.ndarray] = []
    fake_fid_chunks: list[np.ndarray] = []
    n_images = 0

    t0 = time.time()
    for bi, batch in enumerate(dl):
        if args.aux_aerial:
            label, real, aerial = batch
            aerial = aerial.to(device)
        else:
            label, real = batch
            aerial = None
        label = label.to(device)
        fake = generator_forward(G, label, aerial, args.aux_aerial)
        n_images += real.shape[0]

        st, sc = ssim_sum(real, fake.cpu())
        ssim_total += st
        n_ssim += sc

        if not args.skip_lpips:
            lt, lc = lpips_sum(real, fake, device)
            lpips_total += lt
            n_lpips += lc

        if not args.skip_fid:
            real_fid_chunks.append(fid_features(real, device, args.batch_size))
            fake_fid_chunks.append(fid_features(fake.cpu(), device, args.batch_size))

        if (bi + 1) % 100 == 0:
            print(f'[*] processed {n_images} images...')

    metrics = {
        'gan_ckpt': args.gan_ckpt,
        'val_csv': args.val_csv,
        'n_images': n_images,
        'generator_step': step,
        'ssim': ssim_total / max(n_ssim, 1),
    }
    if not args.skip_lpips:
        metrics['lpips'] = lpips_total / max(n_lpips, 1)
    if not args.skip_fid and n_images >= 64:
        real_feats = np.concatenate(real_fid_chunks, axis=0)
        fake_feats = np.concatenate(fake_fid_chunks, axis=0)
        metrics['fid'] = compute_fid_from_features(real_feats, fake_feats)
    elif not args.skip_fid:
        metrics['fid'] = None
        metrics['fid_note'] = 'need >= 64 images for stable FID'

    metrics['elapsed_sec'] = round(time.time() - t0, 2)

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f'[*] wrote {args.out_json}')


if __name__ == '__main__':
    main()
