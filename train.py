"""
train.py  —  CrossNet PyTorch training script
=============================================
Usage:
    python train.py                              # default: cvpr_train.csv, batch=4, 10 epochs
    python train.py --csv cvpr_train.csv --batch_size 2 --epochs 20
    python train.py --no_conditioned             # unconditioned variant

GPU:  automatically uses CUDA if available (Quadro P1000 with CUDA 12.x)
AMP:  mixed-precision (FP16 forward + FP32 master weights) to fit 4 GB VRAM
"""

import os, argparse, time, math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from model   import CrossNet
from dataset import CVPRDataset


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
                   help='Save visualisation every N steps')
    p.add_argument('--save_every',     type=int,   default=500,
                   help='Save checkpoint every N steps')
    p.add_argument('--resume',         default='',
                   help='Path to checkpoint to resume from')
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


# ──────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────

def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert any float tensor to uint8 numpy array in [0, 255]."""
    a  = t.detach().float().cpu().numpy()
    lo = a.min(); hi = a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo) * 255.
    return a.clip(0, 255).astype(np.uint8)


def save_montage(aerial, ground, prob_a, prob_g, step, dump_dir):
    """
    Save a side-by-side montage of the first sample in the batch.
    Columns: aerial | ground | aerial_pred | ground_pred
    """
    def pil_from(t, hw=None):
        arr = _to_uint8(t)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)          # CHW -> HWC
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]                      # collapse single channel
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] > 3:
            arr = arr[..., :3]
        img = Image.fromarray(arr)
        if hw:
            img = img.resize((hw[1], hw[0]), Image.BILINEAR)
        return img

    thumb_h = 112
    parts = []
    for t in [aerial[0], ground[0], prob_a[0], prob_g[0]]:
        w = int(t.shape[-1] * thumb_h / t.shape[-2])
        parts.append(pil_from(t, (thumb_h, w)))

    total_w = sum(p.width for p in parts)
    canvas  = Image.new('RGB', (total_w, thumb_h), (20, 20, 20))
    x_off   = 0
    for p in parts:
        canvas.paste(p, (x_off, 0))
        x_off += p.width

    os.makedirs(dump_dir, exist_ok=True)
    canvas.save(os.path.join(dump_dir, f'{step:07d}.jpg'))


# ──────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────

def save_ckpt(model, optimizer, scaler, step, epoch, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'crossnet_step{step:07d}.pt')
    torch.save({
        'step':            step,
        'epoch':           epoch,
        'model':           model.state_dict(),
        'optimizer':       optimizer.state_dict(),
        'scaler':          scaler.state_dict(),
    }, path)
    # Keep only the last 3 checkpoints
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    for old in ckpts[:-3]:
        os.remove(os.path.join(ckpt_dir, old))
    print(f'    [ckpt] saved -> {path}')


def load_ckpt(path, model, optimizer, scaler, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler'])
    return ck['step'], ck['epoch']


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
    if args.resume:
        start_step, start_epoch = load_ckpt(args.resume, model, optimizer, scaler, device)
        print(f'[*] Resumed from step {start_step}, epoch {start_epoch}')

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = CVPRDataset(args.train_csv)
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

    # ── Training ──────────────────────────────────────────────────────
    target_hw   = model.target_size   # (8, 40)
    global_step = start_step
    best_loss   = math.inf

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_t0   = time.time()

        for batch_idx, (aerial, ground, label) in enumerate(train_dl):
            aerial = aerial.to(device, non_blocking=True)
            ground = ground.to(device, non_blocking=True)
            label  = label.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=use_amp):
                La, Lg = model(aerial)
                loss   = soft_cross_entropy(Lg, label, target_hw, args.num_classes)

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
                      f'avg {avg:.4f}  '
                      f'VRAM {mem_gb:.2f} GB  '
                      f'({elapsed:.0f}s)')

            # ── Visualisation ─────────────────────────────────────────
            if global_step % args.vis_every == 1:
                with torch.no_grad():
                    prob_a = torch.softmax(La,  dim=1)
                    prob_g = torch.softmax(Lg,  dim=1)
                save_montage(aerial.cpu(), ground.cpu(),
                             prob_a.cpu(), prob_g.cpu(),
                             global_step, args.dump_dir)

            # ── Checkpoint ────────────────────────────────────────────
            if global_step % args.save_every == 0:
                save_ckpt(model, optimizer, scaler, global_step, epoch, args.ckpt_dir)

        # End-of-epoch summary
        epoch_avg = epoch_loss / max(len(train_dl), 1)
        epoch_dur = time.time() - epoch_t0
        print(f'\n[epoch {epoch+1:02d} done]  avg loss {epoch_avg:.4f}  '
              f'time {epoch_dur/60:.1f} min\n')

        # Save at epoch end
        save_ckpt(model, optimizer, scaler, global_step, epoch + 1, args.ckpt_dir)

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
