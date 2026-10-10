"""
train_gan_pix2pix.py  —  Pix2Pix GAN: semantic labels → real ground-view panoramas
===========================================================================
Stage 2 of the pipeline
  CrossNet  (already trained) : aerial image  → semantic segmentation map
  This GAN                    : semantic map  → synthesised RGB ground photo

The GAN is trained on ground-truth semantic annotations (not CrossNet's
predictions), so it learns a clean label→photo mapping.  At inference time,
CrossNet's predicted labels are fed in as a drop-in replacement.

Usage
-----
  # Full training (35 k samples, ~3 h/epoch on P1000 — use --max_samples for faster runs)
  python train_gan_pix2pix.py

  # Quick demo: 5 000 samples, 20 epochs ≈ 1–2 h total
  python train_gan_pix2pix.py --max_samples 5000 --epochs 20

  # Resume from a checkpoint pair
  python train_gan_pix2pix.py --resume_g outputs/ckpts_gan/G_step0010000.pt \
                               --resume_d outputs/ckpts_gan/D_step0010000.pt

Outputs
-------
  outputs/ckpts_gan/   — G_stepXXXXXXX.pt  +  D_stepXXXXXXX.pt  (last 3 kept)
  outputs/dump_gan/    — montage JPEGs: [semantic | real ground | synthesised]
"""

import os
import argparse
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from gan_unet_model import UNetGenerator, PatchDiscriminator
from spade_model import SPADEGenerator
from gan_perceptual_loss import VGGPerceptualLoss


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Pix2Pix GAN — semantic map → ground RGB')
    p.add_argument('--train_csv',    default='cvpr_train_v2.csv',
                   help='GT semantic maps (labels_v2/ground/) + panos — not Stage-1 predictions.')
    p.add_argument('--val_csv',      default='cvpr_val_v2.csv')
    p.add_argument('--max_samples',  type=int, default=None,
                   help='Cap training set size (None = use all). '
                        'Tip: --max_samples 5000 for a 1-2 h quick run.')
    p.add_argument('--batch_size',   type=int,   default=2,
                   help='Batch size. Default 2 uses ~2.3 GB VRAM on a 4 GB card (safe). '
                        'Raise to 3 (~3.5 GB) for even better gradient stability.')
    p.add_argument('--epochs',       type=int,   default=10)
    p.add_argument('--lr',           type=float, default=2e-4,
                   help='Adam LR for both G and D (decays linearly after epoch=epochs//2)')
    p.add_argument('--generator',    choices=('unet', 'spade'), default='unet',
                   help='Generator architecture (spade = layout-conditioned SPADE net).')
    p.add_argument('--lambda_l1',    type=float, default=100.0,
                   help='Weight of pixel-wise L1 loss relative to adversarial loss. '
                        'SPADE runs often use 10–20 (see run_spade_v2.sh).')
    p.add_argument('--lambda_perceptual', type=float, default=0.0,
                   help='Weight of VGG perceptual loss (typical SPADE: 10). 0 = off.')
    p.add_argument('--ngf',          type=int,   default=64,
                   help='Generator base filters. Halving to 32 cuts VRAM ~4× at a quality cost.')
    p.add_argument('--ndf',          type=int,   default=64)
    p.add_argument('--num_classes',  type=int,   default=4)
    p.add_argument('--img_h',        type=int,   default=256,
                   help='Training image height (power-of-2 recommended).')
    p.add_argument('--img_w',        type=int,   default=512,
                   help='Training image width. Inference upsamples back to 224×1232.')
    p.add_argument('--num_workers',  type=int,   default=0,
                   help='DataLoader workers (0 = main process, safest on Windows).')
    p.add_argument('--ckpt_dir',     default='outputs/ckpts_gan')
    p.add_argument('--dump_dir',     default='outputs/dump_gan')
    p.add_argument('--log_every',    type=int,   default=100)
    p.add_argument('--vis_every',    type=int,   default=500)
    p.add_argument('--save_every',   type=int,   default=2000)
    p.add_argument('--resume_g',     default='', help='Generator checkpoint to resume from.')
    p.add_argument('--resume_d',     default='', help='Discriminator checkpoint to resume from.')
    p.add_argument('--fresh_d_optimizer', action='store_true',
                   help='Load D weights from --resume_d but use a fresh Adam + GradScaler '
                        '(no optimizer momentum from checkpoint). Useful when fine-tuning '
                        'on a new label distribution after D has saturated.')
    p.add_argument('--random_init_d', action='store_true',
                   help='Do not load --resume_d; keep D at random init (G still uses '
                        '--resume_g). Use when warm-started D is saturated (D_loss ~ 0) on '
                        'a new label distribution.')
    p.add_argument('--g_steps',      type=int,   default=2,
                   help='Number of Generator updates per Discriminator update. '
                        'Raising this weakens D dominance (recommended: 2-3).')
    p.add_argument('--label_smooth', type=float, default=0.9,
                   help='One-sided label smoothing for D real targets (0.9 = soft real). '
                        'Prevents D from becoming overconfident. Set to 1.0 to disable.')
    p.add_argument('--no_amp',       action='store_true', help='Disable AMP (use FP32 throughout).')
    p.add_argument('--d_diag_max_steps', type=int, default=500,
                   help='Print [D_diag] through this many run-steps after resume (0=off). '
                        'Use ~4500 for a 2-epoch gan_v2_diag3 run.')
    p.add_argument('--aux_aerial', action='store_true',
                   help='SPADE only: warp aerial RGB (CSV col 0) as extra G conditioning.')
    p.add_argument('--keep_ckpts', type=int, default=3,
                   help='Rolling checkpoint retention per G/D (default 3).')
    return p.parse_args()


def build_generator(args) -> nn.Module:
    if args.generator == 'unet':
        if getattr(args, 'aux_aerial', False):
            raise ValueError('--aux_aerial requires --generator spade')
        return UNetGenerator(in_ch=args.num_classes, ngf=args.ngf)
    return SPADEGenerator(
        label_nc=args.num_classes,
        ngf=args.ngf,
        img_h=args.img_h,
        img_w=args.img_w,
        aux_aerial=getattr(args, 'aux_aerial', False),
    )


def generator_forward(G: nn.Module, label: torch.Tensor,
                      aerial: torch.Tensor | None, aux_aerial: bool) -> torch.Tensor:
    if aux_aerial:
        return G(label, aerial)
    return G(label)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GANDataset(Dataset):
    """
    Returns (label_tensor, ground_tensor) pairs.

    label_tensor : (num_classes, H, W) float32 — one-hot in [−1, 1]
                   (class-present → +1, class-absent → −1)
    ground_tensor: (3, H, W) float32 — RGB in [−1, 1]

    CSV format: aerial_path, ground_path, label_path  (one sample per line)
    """

    def __init__(self, csv_path: str, num_classes: int = 4,
                 img_h: int = 256, img_w: int = 512,
                 root: str = '', max_samples: int = None,
                 aux_aerial: bool = False):
        self.num_classes = num_classes
        self.img_h = img_h
        self.img_w = img_w
        self.aux_aerial = aux_aerial
        self._warper = None

        samples = []
        with open(csv_path, encoding='utf-8') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) < 3:
                    continue
                if root:
                    parts = [os.path.join(root, p) for p in parts]
                rec = {'ground': parts[1], 'label': parts[2]}
                if aux_aerial:
                    rec['aerial'] = parts[0]
                samples.append(rec)

        if max_samples is not None:
            import random
            random.shuffle(samples)
            samples = samples[:max_samples]

        self.samples = samples

    def _get_warper(self):
        if self._warper is None:
            from aerial_warp import PanoAerialWarper
            self._warper = PanoAerialWarper((self.img_h, self.img_w))
        return self._warper

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # ── Label PNG → one-hot in [−1, 1] ────────────────────────────────────
        lbl_pil = Image.open(s['label'])
        if lbl_pil.mode == 'RGB':
            lbl_pil = lbl_pil.convert('L')
        elif lbl_pil.mode == 'RGBA':
            lbl_pil = lbl_pil.split()[0]
        lbl_pil = lbl_pil.resize((self.img_w, self.img_h), Image.NEAREST)

        lbl_np  = np.array(lbl_pil, dtype=np.int64).clip(0, self.num_classes - 1)
        # one-hot: (H, W, C) then (C, H, W)
        one_hot = np.eye(self.num_classes, dtype=np.float32)[lbl_np].transpose(2, 0, 1)
        # map 0 → −1, 1 → +1
        label_t = torch.from_numpy(one_hot * 2.0 - 1.0)

        # ── Ground RGB → [−1, 1] ───────────────────────────────────────────────
        g_pil   = Image.open(s['ground']).convert('RGB')
        g_pil   = g_pil.resize((self.img_w, self.img_h), Image.BILINEAR)
        g_np    = np.array(g_pil, dtype=np.float32) / 127.5 - 1.0   # [−1, 1]
        ground_t = torch.from_numpy(g_np.transpose(2, 0, 1))

        if self.aux_aerial:
            aerial_t = self._get_warper().warp_aerial_to_pano(s['aerial'])
            return label_t, ground_t, aerial_t
        return label_t, ground_t


# ──────────────────────────────────────────────────────────────────────────────
# Loss (LSGAN — least-squares GAN, more stable than vanilla BCE)
# ──────────────────────────────────────────────────────────────────────────────

def discriminator_loss(real_scores: torch.Tensor,
                       fake_scores: torch.Tensor,
                       label_smooth: float = 0.9) -> torch.Tensor:
    """
    D wants real → label_smooth, fake → 0.  Factor 0.5 slows D relative to G.

    One-sided label smoothing (label_smooth < 1.0) prevents D from becoming
    overconfident on real samples, which keeps the adversarial gradient alive
    for G.  Fake target stays at 0 (no smoothing needed on the fake side).
    """
    real_target = torch.full_like(real_scores, label_smooth)
    return 0.5 * (
        F.mse_loss(real_scores, real_target)
        + F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    )


def generator_adv_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """G wants D to predict fake images as real (score → 1)."""
    return F.mse_loss(fake_scores, torch.ones_like(fake_scores))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoints(G, D, opt_G, opt_D, scaler_G, scaler_D,
                     step: int, epoch: int, ckpt_dir: str, keep: int = 3):
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f'step{step:07d}'

    for prefix, model, opt, scaler in [
        ('G', G, opt_G, scaler_G), ('D', D, opt_D, scaler_D)
    ]:
        path = os.path.join(ckpt_dir, f'{prefix}_{tag}.pt')
        torch.save({
            'step':      step,
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': opt.state_dict(),
            'scaler':    scaler.state_dict(),
        }, path)
        # Keep only the last 3 checkpoints per model
        ckpts = sorted(
            f for f in os.listdir(ckpt_dir)
            if f.startswith(prefix + '_') and f.endswith('.pt')
        )
        keep_n = max(1, keep)
        for old in ckpts[:-keep_n]:
            os.remove(os.path.join(ckpt_dir, old))

    print(f'    [ckpt] saved → {ckpt_dir}/{tag}')


def load_checkpoint(
    path: str,
    model: nn.Module,
    opt,
    scaler,
    device,
    *,
    load_optimizer: bool = True,
):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    if load_optimizer:
        opt.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
    return ck['step'], ck['epoch']


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = np.array(
    [[255, 0,   0  ],   # class 0 — red
     [0,   255, 0  ],   # class 1 — green
     [0,   0,   255],   # class 2 — blue
     [255, 255, 0  ]],  # class 3 — yellow
    dtype=np.uint8
)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Float tensor in [−1, 1] → uint8 numpy (H, W, 3)."""
    arr = t.detach().float().cpu().numpy()   # (3, H, W)
    arr = ((arr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)           # (H, W, 3)


def save_visualisation(label: torch.Tensor, real: torch.Tensor,
                       fake: torch.Tensor, step: int, dump_dir: str):
    """Save side-by-side: [semantic colour map | real ground | synthesised]."""
    lbl_np  = label[0].cpu().float().numpy()       # (C, H, W)
    cls_map = lbl_np.argmax(axis=0)                # (H, W)
    sem_img  = Image.fromarray(_CLASS_COLORS[cls_map])
    real_img = Image.fromarray(_tensor_to_uint8(real[0]))
    fake_img = Image.fromarray(_tensor_to_uint8(fake[0]))

    total_w = sem_img.width + real_img.width + fake_img.width
    montage = Image.new('RGB', (total_w, sem_img.height), (32, 32, 32))
    montage.paste(sem_img,  (0,                            0))
    montage.paste(real_img, (sem_img.width,                0))
    montage.paste(fake_img, (sem_img.width + real_img.width, 0))

    os.makedirs(dump_dir, exist_ok=True)
    montage.save(os.path.join(dump_dir, f'{step:07d}.jpg'))


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device   = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'[*] GPU  : {gpu_name}  ({vram_gb:.1f} GB VRAM)')
    else:
        device = torch.device('cpu')
        print('[!] No GPU — training on CPU (will be very slow).')

    use_amp = (not args.no_amp) and (device.type == 'cuda')
    print(f'[*] AMP  : {"ON (FP16 forward + FP32 master weights)" if use_amp else "OFF"}')

    # ── Models ────────────────────────────────────────────────────────────────
    G = build_generator(args).to(device)
    D = PatchDiscriminator(in_ch=args.num_classes, ndf=args.ndf).to(device)
    perceptual = None
    if args.lambda_perceptual > 0:
        perceptual = VGGPerceptualLoss().to(device)
        print(f'[*] VGG perceptual loss weight: {args.lambda_perceptual}')

    n_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f'[*] Generator params    : {n_G / 1e6:.2f} M')
    print(f'[*] Discriminator params: {n_D / 1e6:.2f} M')

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_G    = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D    = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler_G = GradScaler('cuda', enabled=use_amp)
    scaler_D = GradScaler('cuda', enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, start_epoch = 0, 0
    if args.resume_g and os.path.isfile(args.resume_g):
        try:
            start_step, start_epoch = load_checkpoint(
                args.resume_g, G, opt_G, scaler_G, device)
            print(f'[*] Resumed G from step {start_step}, epoch {start_epoch}')
        except RuntimeError as e:
            if args.generator == 'spade':
                print('[!] --resume_g failed (use a SPADE G checkpoint, not U-Net):', e)
            else:
                raise
    if args.random_init_d:
        print('[*] Discriminator: random init (ignoring --resume_d)')
    elif args.resume_d and os.path.isfile(args.resume_d):
        load_checkpoint(
            args.resume_d, D, opt_D, scaler_D, device,
            load_optimizer=not args.fresh_d_optimizer,
        )
        if args.fresh_d_optimizer:
            print('[*] Resumed D weights only (fresh Adam + GradScaler for D)')
        else:
            print('[*] Resumed D from checkpoint')

    # --epochs means "run this many MORE epochs from the resume point".
    end_epoch = start_epoch + args.epochs

    # LR decays linearly to 0 over the second half of the *additional* epochs.
    # Must be defined after start_epoch is known (i.e. after resume).
    decay_start = start_epoch + args.epochs // 2

    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, args.epochs // 2))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # ── Data ──────────────────────────────────────────────────────────────────
    ds = GANDataset(
        csv_path    = args.train_csv,
        num_classes = args.num_classes,
        img_h       = args.img_h,
        img_w       = args.img_w,
        max_samples = args.max_samples,
        aux_aerial  = args.aux_aerial,
    )
    dl = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == 'cuda') and (args.num_workers == 0),
        drop_last   = True,
    )

    steps_per_epoch = len(dl)
    total_steps     = steps_per_epoch * args.epochs   # additional steps to run
    # batch=2 + g_steps updates ≈ 0.50 s/outer-step on P1000 + AMP
    sec_per_step    = 0.30 * (1 + args.g_steps)

    print(f'[*] Training samples : {len(ds)}')
    print(f'[*] Batch size       : {args.batch_size}  (VRAM ≈ {args.batch_size * 1.16:.1f} GB est.)')
    print(f'[*] G updates / D    : {args.g_steps}  (D label_smooth={args.label_smooth})')
    print(f'[*] Steps per epoch  : {steps_per_epoch}')
    print(f'[*] Total steps      : {total_steps}')
    print(f'[*] Est. total time  : {total_steps * sec_per_step / 3600:.1f} h  '
          f'(rough estimate; use --max_samples N to shorten)')
    print(f'[*] Healthy training : D_loss in [0.05–0.50], G_adv in [0.5–1.5]')
    print()

    # ── Training ──────────────────────────────────────────────────────────────
    global_step = start_step

    for epoch in range(start_epoch, end_epoch):
        G.train()
        D.train()
        epoch_loss_G = epoch_loss_D = 0.0
        epoch_t0 = time.time()

        for batch_idx, batch in enumerate(dl):
            if args.aux_aerial:
                label, real_g, aerial = batch
                aerial = aerial.to(device, non_blocking=True)
            else:
                label, real_g = batch
                aerial = None
            label  = label.to(device,  non_blocking=True)
            real_g = real_g.to(device, non_blocking=True)

            # ── Step D (once per batch) ───────────────────────────────────────
            with autocast('cuda', enabled=use_amp):
                fake_g      = generator_forward(G, label, aerial, args.aux_aerial)
                real_score  = D(label, real_g)
                fake_score  = D(label, fake_g.detach())
                loss_D      = discriminator_loss(real_score, fake_score,
                                                 args.label_smooth)

            opt_D.zero_grad(set_to_none=True)
            scaler_D.scale(loss_D).backward()

            run_step = global_step - start_step + 1
            d_diag_cap = args.d_diag_max_steps
            if d_diag_cap > 0 and run_step <= d_diag_cap and (
                run_step <= 10 or run_step % 50 == 0 or run_step % args.log_every == 0
            ):
                if use_amp:
                    scaler_D.unscale_(opt_D)
                d_grad_sq = 0.0
                n_grad = 0
                for p in D.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        d_grad_sq += g.pow(2).sum().item()
                        n_grad += 1
                d_grad_norm = d_grad_sq ** 0.5
                rs = real_score.detach().float()
                fs = fake_score.detach().float()
                print(
                    f'  [D_diag run_step {run_step:4d} global {global_step + 1:07d}] '
                    f'grad_L2={d_grad_norm:.6e} (params_with_grad={n_grad}) amp={use_amp} '
                    f'real(mean/min/max)={rs.mean().item():.3f}/'
                    f'{rs.min().item():.3f}/{rs.max().item():.3f} '
                    f'fake(mean/min/max)={fs.mean().item():.3f}/'
                    f'{fs.min().item():.3f}/{fs.max().item():.3f}',
                    flush=True,
                )

            scaler_D.step(opt_D)
            scaler_D.update()

            # ── Step G (g_steps times per D step) ────────────────────────────
            # Updating G more often than D prevents D from dominating and keeps
            # the adversarial gradient useful throughout training.
            for g_iter in range(args.g_steps):
                if g_iter > 0:
                    # Only recompute fake_g on subsequent G steps
                    with autocast('cuda', enabled=use_amp):
                        fake_g = generator_forward(G, label, aerial, args.aux_aerial)

                with autocast('cuda', enabled=use_amp):
                    fake_score_g = D(label, fake_g)
                    loss_G_adv   = generator_adv_loss(fake_score_g)
                    loss_G_l1    = F.l1_loss(fake_g, real_g) * args.lambda_l1
                    loss_G       = loss_G_adv + loss_G_l1

                loss_G_perc = fake_g.new_zeros(())
                if perceptual is not None:
                    with autocast('cuda', enabled=False):
                        loss_G_perc = (
                            perceptual(fake_g.float(), real_g.float())
                            * args.lambda_perceptual
                        )
                    loss_G = loss_G + loss_G_perc

                opt_G.zero_grad(set_to_none=True)
                scaler_G.scale(loss_G).backward()
                scaler_G.step(opt_G)
                scaler_G.update()

            global_step  += 1
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                elapsed = time.time() - epoch_t0
                avg_G   = epoch_loss_G / (batch_idx + 1)
                avg_D   = epoch_loss_D / (batch_idx + 1)
                mem_gb  = (torch.cuda.memory_reserved(0) / 1024**3
                           if device.type == 'cuda' else 0.)
                # Healthy GAN: D in [0.1–0.4], G_adv in [0.5–1.5]
                d_health = ('✓' if 0.05 < loss_D.item() < 0.5
                            else ('D_weak' if loss_D.item() < 0.05 else 'D_strong'))
                perc_str = (
                    f' perc={loss_G_perc.item():.4f}'
                    if perceptual is not None else ''
                )
                print(
                    f'  [epoch {epoch+1:03d}] step {global_step:07d}  '
                    f'G={loss_G.item():.4f} '
                    f'(adv={loss_G_adv.item():.4f} l1={loss_G_l1.item():.4f}{perc_str})  '
                    f'D={loss_D.item():.4f}[{d_health}]  '
                    f'avg_G={avg_G:.4f}  avg_D={avg_D:.4f}  '
                    f'VRAM={mem_gb:.2f}GB  ({elapsed:.0f}s)'
                )

            # ── Visualisation ─────────────────────────────────────────────────
            if global_step % args.vis_every == 1:
                with torch.no_grad():
                    save_visualisation(label, real_g, fake_g, global_step, args.dump_dir)

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % args.save_every == 0:
                save_checkpoints(G, D, opt_G, opt_D, scaler_G, scaler_D,
                                 global_step, epoch, args.ckpt_dir, args.keep_ckpts)

        # ── End-of-epoch ──────────────────────────────────────────────────────
        n_b      = max(len(dl), 1)
        dur      = time.time() - epoch_t0
        cur_lr   = opt_G.param_groups[0]['lr']
        sched_G.step()
        sched_D.step()

        print(
            f'\n[epoch {epoch+1:03d} done]  '
            f'avg G={epoch_loss_G/n_b:.4f}  avg D={epoch_loss_D/n_b:.4f}  '
            f'time={dur/60:.1f} min  lr={cur_lr:.2e}\n'
        )

        save_checkpoints(G, D, opt_G, opt_D, scaler_G, scaler_D,
                         global_step, epoch + 1, args.ckpt_dir, args.keep_ckpts)

    print('[*] GAN training complete.')


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()

    print('\n' + '='*70)
    print('  Stage 2 GAN — semantic labels → ground-view RGB  (configuration)')
    print('='*70)
    for k, v in vars(args).items():
        print(f'  {k:<22} {v}')
    print('='*70 + '\n')

    train(args)
