"""
Verify two SPADE checkpoints produce different outputs when weights differ.

Uses eval_gan_pix2pix.load_generator (same path as full val eval).
"""

from __future__ import annotations

import argparse
import os

import torch

from eval_gan_pix2pix import load_generator
from train_gan_pix2pix import GANDataset, generator_forward


def total_abs_weight_diff(path_a: str, path_b: str) -> float:
    ck_a = torch.load(path_a, map_location='cpu', weights_only=False)['model']
    ck_b = torch.load(path_b, map_location='cpu', weights_only=False)['model']
    if ck_a.keys() != ck_b.keys():
        raise KeyError('state_dict keys differ between checkpoints')
    return sum((ck_a[k].float() - ck_b[k].float()).abs().sum().item() for k in ck_a)


@torch.no_grad()
def pixel_diff_stats(fake_a: torch.Tensor, fake_b: torch.Tensor) -> tuple[float, float]:
    d = (fake_a.float() - fake_b.float()).abs()
    return d.mean().item(), d.max().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--ckpt_a',
        default='experiments/phase2/runs/spade_v2/ckpts/G_step0031080.pt',
    )
    p.add_argument(
        '--ckpt_b',
        default='experiments/phase2/runs/spade_v2/ckpts/G_step0033300.pt',
    )
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--n', type=int, default=8)
    args = p.parse_args()

    for path in (args.ckpt_a, args.ckpt_b):
        if not os.path.isfile(path):
            raise SystemExit(f'Missing checkpoint: {path}')

    wdiff = total_abs_weight_diff(args.ckpt_a, args.ckpt_b)
    print(f'(a) total abs weight difference: {wdiff:.6f}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_a, step_a = load_generator(
        args.ckpt_a, 4, 64, device, generator='spade', aux_aerial=False,
    )
    G_b, step_b = load_generator(
        args.ckpt_b, 4, 64, device, generator='spade', aux_aerial=False,
    )
    print(f'[*] loaded steps: {step_a}, {step_b}')

    ds = GANDataset(args.val_csv, aux_aerial=False)
    labels = torch.stack([ds[i][0] for i in range(args.n)]).to(device)

    fake_a = generator_forward(G_a, labels, None, False)
    fake_b = generator_forward(G_b, labels, None, False)
    mean_pd, max_pd = pixel_diff_stats(fake_a, fake_b)
    print(f'(b) mean abs pixel difference (same {args.n} val labels): {mean_pd:.9f}')
    print(f'(c) max abs pixel difference: {max_pd:.9f}')
    print('[*] eval_gan_pix2pix.py does not read cached JPGs; it always runs G(label) forward.')

    if wdiff > 0 and max_pd == 0.0:
        print(
            '\n[!] Weights differ but outputs are identical — inspect load_generator:\n'
            '    eval_gan_pix2pix.py load_generator -> build_generator + load_state_dict(ck["model"])'
        )


if __name__ == '__main__':
    main()
