"""
eval_gan_pix2pix.py — SSIM / LPIPS / FID on GT semantic maps -> GAN RGB.

Training uses ground-truth labels_v2/ground/ (standard Pix2Pix). This script
evaluates the same pairing: GT layout in, compare synthesised RGB to real pano.

Usage
-----
  python eval_gan_pix2pix.py \\
      --gan_ckpt outputs/ckpts_gan/G_step0197766.pt \\
      --val_csv cvpr_val_v2.csv \\
      --max_samples 512

Full val (8,884): omit --max_samples (slow on CPU; use GPU / RunPod).
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from gan_metrics import compute_fid, mean_lpips, mean_ssim
from gan_unet_model import UNetGenerator
from train_gan_pix2pix import GANDataset


def load_generator(ckpt_path: str, num_classes: int, ngf: int, device: torch.device):
    G = UNetGenerator(in_ch=num_classes, ngf=ngf).to(device)
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
    p.add_argument('--skip_fid', action='store_true',
                   help='Skip FID (faster; use for tiny subsets).')
    p.add_argument('--skip_lpips', action='store_true')
    p.add_argument('--out_json', default='')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] device: {device}')

    G, step = load_generator(args.gan_ckpt, args.num_classes, args.ngf, device)
    print(f'[*] generator step: {step}')

    ds = GANDataset(
        csv_path=args.val_csv,
        num_classes=args.num_classes,
        img_h=args.img_h,
        img_w=args.img_w,
        max_samples=args.max_samples,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_real, all_fake = [], []
    t0 = time.time()
    for label, real in dl:
        label = label.to(device)
        fake = G(label)
        all_real.append(real)
        all_fake.append(fake.cpu())

    real_cat = torch.cat(all_real, dim=0)
    fake_cat = torch.cat(all_fake, dim=0)

    metrics = {
        'gan_ckpt': args.gan_ckpt,
        'val_csv': args.val_csv,
        'n_images': int(real_cat.shape[0]),
        'generator_step': step,
        'ssim': mean_ssim(real_cat, fake_cat),
    }
    if not args.skip_lpips:
        metrics['lpips'] = mean_lpips(real_cat, fake_cat, device)
    if not args.skip_fid and real_cat.shape[0] >= 64:
        metrics['fid'] = compute_fid(real_cat, fake_cat, device, batch_size=args.batch_size)
    elif not args.skip_fid:
        metrics['fid'] = None
        metrics['fid_note'] = 'need >= 64 images for stable FID; use full val on RunPod'

    metrics['elapsed_sec'] = round(time.time() - t0, 2)

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f'[*] wrote {args.out_json}')


if __name__ == '__main__':
    main()
