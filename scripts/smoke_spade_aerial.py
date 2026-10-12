"""Smoke test: layout-only vs aux-aerial SPADE shapes, params, short train step."""

import argparse
import os

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from spade_model import SPADEGenerator
from train_gan_pix2pix import (
    GANDataset,
    build_generator,
    discriminator_loss,
    generator_adv_loss,
    generator_forward,
    save_visualisation,
)


class _Args:
    generator = 'spade'
    num_classes = 4
    ngf = 64
    img_h = 256
    img_w = 512
    aux_aerial = False


def _param_m(G: torch.nn.Module) -> float:
    return sum(p.numel() for p in G.parameters() if p.requires_grad) / 1e6


def _cond_shapes(G: SPADEGenerator, label, aerial):
    with torch.no_grad():
        sz_first = (G.sh, G.sw)
        sz_last = (G.sh * 32, G.sw * 32)
        c0 = G._condition_at(label, aerial, sz_first)
        c1 = G._condition_at(label, aerial, sz_last)
    return c0.shape, c1.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', default='cvpr_train_v2.csv')
    p.add_argument('--max_samples', type=int, default=32)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=2)
    args = p.parse_args()

    if not os.path.isfile(args.train_csv):
        raise SystemExit(f'Missing {args.train_csv} — run from repo root with data.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a_off = _Args()
    a_off.aux_aerial = False
    G0 = build_generator(a_off).to(device)
    print(f'[*] G params aux_aerial=OFF: {_param_m(G0):.2f} M')

    a_on = _Args()
    a_on.aux_aerial = True
    G1 = build_generator(a_on).to(device)
    print(f'[*] G params aux_aerial=ON:  {_param_m(G1):.2f} M')

    ds = GANDataset(
        args.train_csv, max_samples=args.max_samples, aux_aerial=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    label, real, aerial = next(iter(dl))
    label, real, aerial = label.to(device), real.to(device), aerial.to(device)

    sh0, sh1 = _cond_shapes(G1, label[:1], aerial[:1])
    print(f'[*] first SPADE block cond shape: {tuple(sh0)}  (4 lbl + 3 aerial when ON)')
    print(f'[*] last SPADE block cond shape:  {tuple(sh1)}')

    from gan_unet_model import PatchDiscriminator

    D = PatchDiscriminator(in_ch=4, ndf=64).to(device)
    opt_G = torch.optim.Adam(G1.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler_G = GradScaler('cuda', enabled=device.type == 'cuda')
    scaler_D = GradScaler('cuda', enabled=device.type == 'cuda')

    dump_dir = 'experiments/phase2/runs/spade_aerial_v2/smoke_dump'
    os.makedirs(dump_dir, exist_ok=True)

    it = iter(dl)
    for step in range(1, args.steps + 1):
        try:
            label, real, aerial = next(it)
        except StopIteration:
            it = iter(dl)
            label, real, aerial = next(it)
        label, real, aerial = label.to(device), real.to(device), aerial.to(device)

        with autocast('cuda', enabled=device.type == 'cuda'):
            fake = generator_forward(G1, label, aerial, True)
            loss_D = discriminator_loss(D(label, real), D(label, fake.detach()))
        opt_D.zero_grad(set_to_none=True)
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        with autocast('cuda', enabled=device.type == 'cuda'):
            fake = generator_forward(G1, label, aerial, True)
            loss_G = generator_adv_loss(D(label, fake)) + torch.nn.functional.l1_loss(fake, real) * 10
        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        if step == args.steps:
            print(f'[*] final step losses: D={float(loss_D):.4f} G={float(loss_G):.4f}')
            assert torch.isfinite(loss_D) and torch.isfinite(loss_G)
            save_visualisation(label, real, fake, step, dump_dir)
            print(f'[*] montage -> {dump_dir}/{step:07d}.jpg')

    print('[OK] smoke_spade_aerial')


if __name__ == '__main__':
    main()
