"""
train_crossnet.py  —  CrossNet PyTorch training script
======================================================
Usage:
    python train_crossnet.py                              # default: cvpr_train.csv, batch=4, 10 epochs
    python train_crossnet.py --csv cvpr_train.csv --batch_size 2 --epochs 20
    python train_crossnet.py --no_conditioned             # unconditioned variant

Score a checkpoint without training:
    python eval_crossnet.py --ckpt outputs/ckpts_pt/crossnet_best.pt

GPU:  automatically uses CUDA if available (Quadro P1000 with CUDA 12.x)
AMP:  mixed-precision (FP16 forward + FP32 master weights) to fit 4 GB VRAM

Best checkpoint is selected by validation mIoU (full-res), not train loss.
Dumps are argmax + 4-class palette: aerial | ground | GT | pred ground | pred aerial.
"""

import os, argparse, time, csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast

from crossnet_model   import CrossNet
from crossnet_dataset import CVPRDataset
from semantic_metrics import evaluate_crossnet, format_metrics, metrics_row, save_eval_montage


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='CrossNet PyTorch trainer')
    p.add_argument('--train_csv',      default='cvpr_train.csv')
    p.add_argument('--val_csv',        default='cvpr_val.csv')
    p.add_argument('--batch_size',     type=int,   default=4)
    p.add_argument('--epochs',         type=int,   default=10)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--lr_decay_steps', type=int,   default=5000)
    p.add_argument('--lr_decay_rate',  type=float, default=0.7)
    p.add_argument('--num_classes',    type=int,   default=4)
    p.add_argument('--num_workers',    type=int,   default=0,
                   help='DataLoader workers (0 = main process, safest on Windows)')
    p.add_argument('--no_pretrained',  action='store_true',
                   help='Skip ImageNet weight initialisation for VGG16')
    p.add_argument('--no_conditioned', action='store_true',
                   help='Use unconditioned transformation network')
    p.add_argument('--no_amp',         action='store_true',
                   help='Disable automatic mixed precision (use FP32 throughout)')
    p.add_argument('--ckpt_dir',       default='outputs/ckpts_pt')
    p.add_argument('--dump_dir',       default='outputs/dump_pt')
    p.add_argument('--log_every',      type=int,   default=10,
                   help='Print loss every N steps')
    p.add_argument('--vis_every',      type=int,   default=100,
                   help='Save colourised visualisation every N steps')
    p.add_argument('--save_every',     type=int,   default=500,
                   help='Save checkpoint every N steps')
    p.add_argument('--val_every',      type=int,   default=0,
                   help='Also run validation every N steps (0 = end of epoch only)')
    p.add_argument('--val_max_samples', type=int,  default=0,
                   help='Cap val images for a faster sanity check (0 = full val split)')
    p.add_argument('--val_batch_size', type=int,   default=0,
                   help='Val batch size (0 = same as --batch_size)')
    p.add_argument('--val_at_start',   action='store_true',
                   help='Run validation once before the first training step')
    p.add_argument('--resume',         default='',
                   help='Path to checkpoint to resume from')
    p.add_argument('--flip',           action='store_true',
                   help='Horizontal-flip aerial + left-right reverse the panorama')
    p.add_argument('--aux_weight',     type=float, default=0.5,
                   help='Weight of aerial auxiliary loss (0 disables it)')
    p.add_argument('--loss_h',         type=int,   default=32,
                   help='Height at which ground logits are supervised')
    p.add_argument('--loss_w',         type=int,   default=160,
                   help='Width at which ground logits are supervised')
    p.add_argument('--focal_gamma',    type=float, default=2.0)
    p.add_argument('--legacy_loss',    action='store_true',
                   help='Use original soft-CE at 8x40 instead of focal+dice')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Loss: soft cross-entropy (matches original TF implementation)
# ──────────────────────────────────────────────────────────────────────

def soft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                        target_hw: tuple, num_classes: int) -> torch.Tensor:
    """
    Parameters
    ----------
    logits      : (B, C, Hg, Wg) — raw network outputs
    labels      : (B, H, W)  long — per-pixel class indices (full ground size)
    target_hw   : (Hg, Wg)   — spatial size of logits
    num_classes : C
    """
    # One-hot encode at full resolution
    B, H, W   = labels.shape
    one_hot   = F.one_hot(labels.clamp(0, num_classes - 1), num_classes)  # (B, H, W, C)
    one_hot   = one_hot.permute(0, 3, 1, 2).float()                        # (B, C, H, W)

    # Bilinear downsample to match logits spatial size
    soft_lbl  = F.interpolate(one_hot, size=target_hw,
                               mode='bilinear', align_corners=True)         # (B, C, Hg, Wg)

    # Soft cross-entropy: -Σ p_soft * log_softmax(logits)
    log_prob  = F.log_softmax(logits, dim=1)                                # (B, C, Hg, Wg)
    loss      = -(soft_lbl * log_prob).sum(dim=1).mean()
    return loss


def _resize_labels(labels: torch.Tensor, hw: tuple) -> torch.Tensor:
    return F.interpolate(
        labels.float().unsqueeze(1), size=hw, mode='nearest',
    ).squeeze(1).long()


def focal_ce(logits: torch.Tensor, target: torch.Tensor,
             gamma: float = 2.0, ignore_index: int = -1) -> torch.Tensor:
    """Per-pixel focal cross-entropy. `target` is (B, H, W) class ids."""
    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    logp = F.log_softmax(logits, dim=1)
    tgt = target.clamp(0, logits.size(1) - 1)
    logpt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    loss = -(1.0 - pt).pow(gamma) * logpt
    return loss[valid].mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              num_classes: int, ignore_index: int = -1,
              smooth: float = 1.0) -> torch.Tensor:
    valid = target != ignore_index
    tgt = target.clamp(0, num_classes - 1)
    pred = F.softmax(logits, dim=1)
    oh = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()
    pred = pred * mask
    oh = oh * mask
    dims = (0, 2, 3)
    inter = (pred * oh).sum(dim=dims)
    den = pred.sum(dim=dims) + oh.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (den + smooth)
    return 1.0 - dice.mean()


def phase1_loss(la, lg, label, aerial_label, num_classes,
                loss_hw, aux_weight, gamma, legacy=False):
    """Ground loss at `loss_hw` + optional aerial auxiliary loss."""
    if legacy:
        loss_g = soft_cross_entropy(lg, label, lg.shape[-2:], num_classes)
    else:
        lg_up = F.interpolate(lg, size=loss_hw, mode='bilinear', align_corners=True)
        g_lbl = _resize_labels(label, loss_hw)
        loss_g = focal_ce(lg_up, g_lbl, gamma=gamma) + dice_loss(
            lg_up, g_lbl, num_classes,
        )

    loss_a = lg.new_zeros(())
    if aux_weight > 0 and aerial_label is not None and bool((aerial_label >= 0).any()):
        a_hw = aerial_label.shape[-2:]
        la_up = F.interpolate(la, size=a_hw, mode='bilinear', align_corners=True)
        loss_a = focal_ce(la_up, aerial_label, gamma=gamma) + 0.5 * dice_loss(
            la_up, aerial_label, num_classes,
        )
    total = loss_g + aux_weight * loss_a
    return total, loss_g, loss_a


# ──────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────

def _ckpt_payload(model, optimizer, scaler, step, epoch, extra=None):
    payload = {
        'step':      step,
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler':    scaler.state_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def save_ckpt(model, optimizer, scaler, step, epoch, ckpt_dir, extra=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'crossnet_step{step:07d}.pt')
    torch.save(_ckpt_payload(model, optimizer, scaler, step, epoch, extra), path)
    # Keep only the last 3 step checkpoints; never delete crossnet_best.pt
    steps = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.startswith('crossnet_step') and f.endswith('.pt')
    )
    for old in steps[:-3]:
        os.remove(os.path.join(ckpt_dir, old))
    print(f'    [ckpt] saved -> {path}')


def save_best_ckpt(model, optimizer, scaler, step, epoch, ckpt_dir, extra=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'crossnet_best.pt')
    torch.save(_ckpt_payload(model, optimizer, scaler, step, epoch, extra), path)
    print(f'    [ckpt] new best -> {path}')


def load_ckpt(path, model, optimizer, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    if scaler is not None and ck.get('scaler') is not None:
        scaler.load_state_dict(ck['scaler'])
    return ck


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f'[*] GPU detected : {gpu_name}')
        print(f'    VRAM total   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        device = torch.device('cpu')
        print('[!] No GPU detected — training on CPU (will be slow).')

    use_amp = (not args.no_amp) and (device.type == 'cuda')
    print(f'[*] Mixed precision (AMP) : {"ON" if use_amp else "OFF"}')

    # ── Model ─────────────────────────────────────────────────────────
    model = CrossNet(
        num_classes = args.num_classes,
        conditioned  = not args.no_conditioned,
        pretrained   = not args.no_pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[*] Trainable parameters : {n_params / 1e6:.2f} M')

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = GradScaler('cuda', enabled=use_amp)

    start_step  = 0
    start_epoch = 0
    best_miou   = -1.0
    if args.resume:
        ck = load_ckpt(args.resume, model, optimizer, scaler, device)
        start_step  = int(ck.get('step', 0))
        start_epoch = int(ck.get('epoch', 0))
        best_miou   = float(ck.get('val_miou', -1.0))
        print(f'[*] Resumed from step {start_step}, epoch {start_epoch}'
              f'  (best val mIoU {best_miou:.4f})')

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = CVPRDataset(args.train_csv, augment=args.flip)
    train_dl = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == 'cuda') and (args.num_workers == 0),
        drop_last   = True,
    )
    print(f'[*] Training samples : {len(train_ds)}')
    print(f'[*] Batch size       : {args.batch_size}')
    print(f'[*] Steps per epoch  : {len(train_dl)}')

    val_dl = None
    if args.val_csv and os.path.isfile(args.val_csv):
        val_ds = CVPRDataset(args.val_csv, augment=False)
        if args.val_max_samples > 0:
            val_ds = Subset(val_ds, range(min(args.val_max_samples, len(val_ds))))
        val_bs = args.val_batch_size or args.batch_size
        val_dl = DataLoader(
            val_ds,
            batch_size  = val_bs,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = (device.type == 'cuda') and (args.num_workers == 0),
        )
        print(f'[*] Validation samples : {len(val_ds)}  (batch {val_bs})')
    else:
        print('[!] No val CSV — validation disabled. Pass --val_csv cvpr_val.csv')

    val_log_path = os.path.join(args.dump_dir, 'val_metrics.csv')

    def run_validation(epoch, step):
        nonlocal best_miou
        if val_dl is None:
            return None
        print(f'\n[*] Validating at step {step:07d} ...')
        t0 = time.time()
        metrics = evaluate_crossnet(
            model, val_dl, device,
            num_classes=args.num_classes,
            use_amp=use_amp,
        )
        print(format_metrics(metrics))
        print(f'    val time {time.time() - t0:.1f}s\n')

        os.makedirs(args.dump_dir, exist_ok=True)
        row = metrics_row(metrics, epoch=epoch, step=step)
        write_header = not os.path.isfile(val_log_path)
        with open(val_log_path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        extra = {
            'val_miou': metrics['full']['miou'],
            'val_miou_native': metrics['low']['miou'],
        }
        if metrics['full']['miou'] > best_miou:
            best_miou = metrics['full']['miou']
            save_best_ckpt(model, optimizer, scaler, step, epoch, args.ckpt_dir, extra)
        return metrics

    print(f'[*] Flip augment     : {args.flip}')
    print(f'[*] Loss grid        : {args.loss_h}x{args.loss_w}  aux={args.aux_weight}')

    # ── Training ──────────────────────────────────────────────────────
    # ── Training ──────────────────────────────────────────────────────
    loss_hw     = (args.loss_h, args.loss_w)
    global_step = start_step

    if args.val_at_start:
        run_validation(start_epoch, global_step)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_t0   = time.time()

        for batch_idx, batch in enumerate(train_dl):
            aerial, ground, label, aerial_label = batch
            aerial = aerial.to(device, non_blocking=True)
            ground = ground.to(device, non_blocking=True)
            label  = label.to(device,  non_blocking=True)
            aerial_label = aerial_label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=use_amp):
                La, Lg = model(aerial)
                loss, loss_g, loss_a = phase1_loss(
                    La, Lg, label, aerial_label, args.num_classes,
                    loss_hw, args.aux_weight, args.focal_gamma,
                    legacy=args.legacy_loss,
                )

            scaler.scale(loss).backward()

            # Gradient clipping (prevents exploding gradients early in training)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(optimizer)
            scaler.update()

            # LR schedule: exponential decay every lr_decay_steps
            global_step += 1
            if global_step % args.lr_decay_steps == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] *= args.lr_decay_rate
                print(f'    [lr] decayed -> {optimizer.param_groups[0]["lr"]:.2e}')

            loss_val   = loss.item()
            epoch_loss += loss_val

            # ── Logging ───────────────────────────────────────────────
            if global_step % args.log_every == 0:
                elapsed = time.time() - epoch_t0
                avg     = epoch_loss / (batch_idx + 1)
                mem_gb  = (torch.cuda.memory_reserved(0) / 1024**3
                           if device.type == 'cuda' else 0.)
                print(f'  [epoch {epoch+1:02d}] '
                      f'step {global_step:07d}  '
                      f'loss {loss_val:.4f}  '
                      f'Lg {loss_g.item():.3f}  '
                      f'La {loss_a.item():.3f}  '
                      f'avg {avg:.4f}  '
                      f'VRAM {mem_gb:.2f} GB  '
                      f'({elapsed:.0f}s)', flush=True)

            # ── Visualisation (argmax + class palette, includes GT) ──
            if args.vis_every > 0 and global_step % args.vis_every == 1:
                save_eval_montage(
                    aerial, ground, label, La, Lg,
                    global_step, args.dump_dir,
                    prefix='train', num_classes=args.num_classes,
                )

            # ── Checkpoint ────────────────────────────────────────────
            if global_step % args.save_every == 0:
                save_ckpt(model, optimizer, scaler, global_step, epoch, args.ckpt_dir)

            # ── Mid-epoch validation ──────────────────────────────────
            if args.val_every > 0 and global_step % args.val_every == 0:
                run_validation(epoch, global_step)
                model.train()

        # End-of-epoch summary
        epoch_avg = epoch_loss / max(len(train_dl), 1)
        epoch_dur = time.time() - epoch_t0
        print(f'\n[epoch {epoch+1:02d} done]  avg loss {epoch_avg:.4f}  '
              f'time {epoch_dur/60:.1f} min\n')

        # Save at epoch end, then validate (best ckpt is selected here)
        save_ckpt(model, optimizer, scaler, global_step, epoch + 1, args.ckpt_dir)
        run_validation(epoch + 1, global_step)

    print('[*] Training complete.')


# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()
    print('\n' + '='*60)
    print('  CrossNet PyTorch — configuration')
    print('='*60)
    for k, v in vars(args).items():
        print(f'  {k:<22} {v}')
    print('='*60 + '\n')
    train(args)
