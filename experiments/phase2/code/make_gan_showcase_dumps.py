#!/usr/bin/env python3
"""Save a few label|real|fake montages for Stage-2 GAN runs (local, no FID)."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import DataLoader

from eval_gan_pix2pix import load_generator
from train_gan_pix2pix import GANDataset, generator_forward, save_visualisation

# Runs with empty/missing dump/ only (do not regenerate gan_v2_* / spade_v2 dumps).
PRESETS = {
    'spade_aerial_v2': {
        'gan_ckpt': 'experiments/phase2/runs/spade_aerial_v2/ckpts/G_step0032000.pt',
        'generator': 'spade',
        'aux_aerial': True,
        'dump_dir': 'experiments/phase2/runs/spade_aerial_v2/dump',
    },
}


@torch.no_grad()
def dump_run(
    *,
    gan_ckpt: str,
    dump_dir: str,
    generator: str,
    aux_aerial: bool,
    val_csv: str,
    n_images: int,
    img_h: int,
    img_w: int,
    ngf: int,
) -> int:
    if not os.path.isfile(gan_ckpt):
        raise FileNotFoundError(gan_ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G, step = load_generator(
        gan_ckpt, 4, ngf, device,
        generator=generator, img_h=img_h, img_w=img_w, aux_aerial=aux_aerial,
    )
    ds = GANDataset(
        val_csv, num_classes=4, img_h=img_h, img_w=img_w,
        max_samples=n_images, aux_aerial=aux_aerial,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(dump_dir, exist_ok=True)
    saved = 0
    for i, batch in enumerate(dl):
        if aux_aerial:
            label, real, aerial = batch
            aerial = aerial.to(device)
        else:
            label, real = batch
            aerial = None
        label = label.to(device)
        real = real.to(device)
        fake = generator_forward(G, label, aerial, aux_aerial)
        save_visualisation(label, real, fake, i + 1, dump_dir)
        saved += 1
    print(f'[*] {dump_dir}: {saved} montages (G step {step}, {device})')
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=list(PRESETS.keys()), action='append')
    p.add_argument('--all-empty', action='store_true', help='Run every preset in PRESETS')
    p.add_argument('--gan_ckpt', default='')
    p.add_argument('--dump_dir', default='')
    p.add_argument('--generator', choices=('unet', 'spade'), default='spade')
    p.add_argument('--aux_aerial', action='store_true')
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--n_images', type=int, default=8)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--ngf', type=int, default=64)
    args = p.parse_args()

    if args.gan_ckpt and args.dump_dir:
        dump_run(
            gan_ckpt=args.gan_ckpt,
            dump_dir=args.dump_dir,
            generator=args.generator,
            aux_aerial=args.aux_aerial,
            val_csv=args.val_csv,
            n_images=args.n_images,
            img_h=args.img_h,
            img_w=args.img_w,
            ngf=args.ngf,
        )
        return

    names = list(PRESETS.keys()) if args.all_empty else (args.preset or [])
    if not names:
        p.error('Use --all-empty, --preset NAME, or --gan_ckpt + --dump_dir')
    for name in names:
        cfg = PRESETS[name]
        dump_run(
            val_csv=args.val_csv,
            n_images=args.n_images,
            img_h=args.img_h,
            img_w=args.img_w,
            ngf=args.ngf,
            **cfg,
        )


if __name__ == '__main__':
    main()
