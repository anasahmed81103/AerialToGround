"""
Lightweight Stage 2 smoke test (local or RunPod).

1) v2 CSV paths resolve (GT ground labels + panos)
2) Warm-start G/D from baseline checkpoints
3) Two training steps on a tiny subset
4) SSIM/LPIPS on 16 val images (FID skipped)

  python smoke_gan_v2.py
  python smoke_gan_v2.py --resume_g path/to/G.pt --resume_d path/to/D.pt
"""

import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from eval_gan_pix2pix import load_generator
from gan_metrics import mean_lpips, mean_ssim
from gan_unet_model import PatchDiscriminator, UNetGenerator
from train_gan_pix2pix import GANDataset, discriminator_loss, generator_adv_loss, load_checkpoint


def weight_fingerprint(model: torch.nn.Module) -> dict:
    sd = model.state_dict()
    key = sorted(sd.keys())[0]
    t = sd[key]
    return {
        'first_key': key,
        'first_shape': list(t.shape),
        'first_tensor_sum': round(float(t.float().sum().item()), 6),
        'abs_sum_all_params': round(
            sum(p.float().abs().sum().item() for p in sd.values()), 4
        ),
        'n_tensors': len(sd),
    }


def verify_warmstart(resume_g: str, resume_d: str) -> dict:
    """Prove checkpoints load with strict key/shape match (not random init)."""
    out = {'resume_g': resume_g, 'resume_d': resume_d}
    G = UNetGenerator(in_ch=4, ngf=64)
    D = PatchDiscriminator(in_ch=4, ndf=64)
    out['G_before_random_init'] = weight_fingerprint(G)
    out['D_before_random_init'] = weight_fingerprint(D)

    ck_g = torch.load(resume_g, map_location='cpu', weights_only=False)
    ck_d = torch.load(resume_d, map_location='cpu', weights_only=False)
    out['ckpt_G_step'] = ck_g.get('step')
    out['ckpt_D_step'] = ck_d.get('step')

    G.load_state_dict(ck_g['model'], strict=True)
    D.load_state_dict(ck_d['model'], strict=True)
    out['G_after_load'] = weight_fingerprint(G)
    out['D_after_load'] = weight_fingerprint(D)

    fg = out['G_after_load']['first_tensor_sum']
    fd = out['D_after_load']['first_tensor_sum']
    file_fg = round(float(ck_g['model'][out['G_after_load']['first_key']].float().sum().item()), 6)
    file_fd = round(float(ck_d['model'][out['D_after_load']['first_key']].float().sum().item()), 6)

    out['G_weights_changed_from_init'] = (
        out['G_before_random_init']['abs_sum_all_params']
        != out['G_after_load']['abs_sum_all_params']
    )
    out['D_weights_changed_from_init'] = (
        out['D_before_random_init']['abs_sum_all_params']
        != out['D_after_load']['abs_sum_all_params']
    )
    out['G_matches_ckpt_file'] = fg == file_fg
    out['D_matches_ckpt_file'] = fd == file_fd
    out['warmstart_ok'] = all([
        out['G_weights_changed_from_init'],
        out['D_weights_changed_from_init'],
        out['G_matches_ckpt_file'],
        out['D_matches_ckpt_file'],
    ])
    return out


def _pip_lpips():
    try:
        import lpips  # noqa: F401
        return True
    except ImportError:
        print('[smoke] installing lpips (one-time)...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lpips', '-q'])
        return True


def check_csv_paths(csv_path: str, n: int = 50) -> dict:
    missing = {'ground': 0, 'label': 0}
    checked = 0
    with open(csv_path, encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 3:
                continue
            checked += 1
            if not os.path.isfile(parts[1]):
                missing['ground'] += 1
            if not os.path.isfile(parts[2]):
                missing['label'] += 1
            if checked >= n:
                break
    return {'csv': csv_path, 'checked': checked, 'missing': missing}


def run_train_smoke(args, device):
    G = UNetGenerator(in_ch=4, ngf=64).to(device)
    D = PatchDiscriminator(in_ch=4, ndf=64).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler_G = GradScaler('cuda', enabled=device.type == 'cuda')
    scaler_D = GradScaler('cuda', enabled=device.type == 'cuda')

    start_step = 0
    if args.resume_g and os.path.isfile(args.resume_g):
        start_step, _ = load_checkpoint(args.resume_g, G, opt_G, scaler_G, device)
    if args.resume_d and os.path.isfile(args.resume_d):
        load_checkpoint(args.resume_d, D, opt_D, scaler_D, device)

    ds = GANDataset(
        csv_path=args.train_csv,
        num_classes=4,
        img_h=256,
        img_w=512,
        max_samples=args.max_samples,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    label, real_g = next(iter(dl))
    label = label.to(device)
    real_g = real_g.to(device)

    use_amp = device.type == 'cuda'
    losses = []
    for _ in range(args.train_steps):
        with autocast('cuda', enabled=use_amp):
            fake_g = G(label)
            loss_D = discriminator_loss(D(label, real_g), D(label, fake_g.detach()))
        opt_D.zero_grad(set_to_none=True)
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        with autocast('cuda', enabled=use_amp):
            fake_g = G(label)
            loss_G = generator_adv_loss(D(label, fake_g)) + F.l1_loss(fake_g, real_g) * 100.0
        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()
        losses.append({'D': float(loss_D.item()), 'G': float(loss_G.item())})

    return {
        'resume_g': args.resume_g,
        'resume_step': start_step,
        'train_samples': len(ds),
        'train_steps': args.train_steps,
        'losses': losses,
    }


def run_eval_smoke(args, device):
    G, step = load_generator(args.resume_g, 4, 64, device)
    ds = GANDataset(
        csv_path=args.val_csv,
        num_classes=4,
        img_h=256,
        img_w=512,
        max_samples=args.eval_samples,
    )
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    reals, fakes = [], []
    with torch.no_grad():
        for label, real in dl:
            label = label.to(device)
            fake = G(label)
            reals.append(real)
            fakes.append(fake.cpu())
    real_cat = torch.cat(reals, dim=0)
    fake_cat = torch.cat(fakes, dim=0)
    return {
        'generator_step': step,
        'eval_n': int(real_cat.shape[0]),
        'ssim': mean_ssim(real_cat, fake_cat),
        'lpips': mean_lpips(real_cat, fake_cat, device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', default='cvpr_train_v2.csv')
    p.add_argument('--val_csv', default='cvpr_val_v2.csv')
    p.add_argument('--resume_g', default='outputs/ckpts_gan/G_step0197766.pt')
    p.add_argument('--resume_d', default='outputs/ckpts_gan/D_step0197766.pt')
    p.add_argument('--max_samples', type=int, default=32)
    p.add_argument('--train_steps', type=int, default=2)
    p.add_argument('--eval_samples', type=int, default=16)
    args = p.parse_args()

    _pip_lpips()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    report = {
        'device': str(device),
        'path_check_train': check_csv_paths(args.train_csv),
        'path_check_val': check_csv_paths(args.val_csv, n=20),
        'note': (
            'GAN trains on GT labels_v2/ground/ + panos. '
            'polar_full layouts are inference-only for the demo pipeline.'
        ),
    }

    for key in ('path_check_train', 'path_check_val'):
        m = report[key]['missing']
        if m['ground'] or m['label']:
            print(json.dumps(report, indent=2))
            raise SystemExit(f'[smoke] missing files in {report[key]["csv"]}: {m}')

    if not os.path.isfile(args.resume_g):
        report['warmstart_verify'] = {'skipped': 'resume_g not found'}
        report['train_smoke'] = {'skipped': 'resume_g not found'}
        report['eval_smoke'] = {'skipped': 'no generator ckpt'}
    else:
        if os.path.isfile(args.resume_d):
            report['warmstart_verify'] = verify_warmstart(args.resume_g, args.resume_d)
        else:
            report['warmstart_verify'] = {'error': f'missing {args.resume_d}'}
        report['train_smoke'] = run_train_smoke(args, device)
        report['eval_smoke'] = run_eval_smoke(args, device)
        report['eval_smoke']['fid'] = None
        report['eval_smoke']['fid_note'] = (
            f'intentionally skipped: smoke uses eval_n={args.eval_samples} '
            f'(< 64); run eval_gan_pix2pix.py on full val for FID'
        )

    print('\n' + '=' * 70)
    print('  Stage 2 GAN v2 smoke test')
    print('=' * 70)
    print(json.dumps(report, indent=2))
    print('=' * 70)


if __name__ == '__main__':
    main()
