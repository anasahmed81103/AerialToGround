"""
aerial_to_ground_final.py  —  End-to-end Aerial → Ground Synthesis Pipeline
=============================================================================
Runs the full two-stage pipeline on any aerial-view image:

  Stage 1 — CrossNet   : aerial image (224×224)
                         → semantic segmentation map (8×40, upscaled for display)
  Stage 2 — GAN        : semantic map (256×512)
                         → synthesised RGB ground-view panorama

Displays three panels on screen:
  [ Aerial Input ]  [ CrossNet Semantic Map ]  [ GAN Synthesised Ground View ]

Also saves the result montage to  outputs/pipeline_results/

Usage
-----
  # Use any aerial image — auto-detects latest checkpoints
  python aerial_to_ground_final.py --input path/to/aerial.jpg

  # Specify checkpoints explicitly
  python aerial_to_ground_final.py --input path/to/aerial.jpg \\
      --crossnet_ckpt outputs/ckpts_pt/crossnet_step0088830.pt \\
      --gan_ckpt      outputs/ckpts_gan/G_step0197766.pt

  # Run without displaying (save only)
  python aerial_to_ground_final.py --input path/to/aerial.jpg --no_display
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# ── Local modules ────────────────────────────────────────────────────────────
from crossnet_model import CrossNet
from gan_unet_model  import UNetGenerator

ROOT         = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES  = 4
GAN_IMG_H    = 256
GAN_IMG_W    = 512

# Colour palette for semantic classes (up to 8 classes supported)
CLASS_COLORS = np.array([
    [255,  64,  64],   # class 0 — red
    [ 64, 200,  64],   # class 1 — green
    [ 64,  64, 255],   # class 2 — blue
    [255, 220,  30],   # class 3 — yellow
    [220,  64, 220],   # class 4 — magenta
    [ 64, 220, 220],   # class 5 — cyan
    [255, 140,   0],   # class 6 — orange
    [160, 160, 160],   # class 7 — grey
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Aerial → Semantic Map → Ground Panorama  (full pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--input', '-i', required=True,
                   help='Path to aerial-view input image (any common format)')
    p.add_argument('--crossnet_ckpt', default='',
                   help='CrossNet checkpoint (.pt). Default: latest in outputs/ckpts_pt/')
    p.add_argument('--gan_ckpt', default='',
                   help='GAN generator checkpoint (.pt). Default: latest in outputs/ckpts_gan/')
    p.add_argument('--no_display', action='store_true',
                   help='Skip the on-screen display (save result only)')
    p.add_argument('--save_dir', default=os.path.join(ROOT, 'outputs', 'pipeline_results'),
                   help='Directory to save the result montage')
    p.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_crossnet_ckpt() -> str:
    pattern = os.path.join(ROOT, 'outputs', 'ckpts_pt', 'crossnet_step*.pt')
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        sys.exit('[ERROR] No CrossNet checkpoint found in outputs/ckpts_pt/.\n'
                 '        Train first with:  python train_crossnet.py')
    return candidates[-1]


def _latest_gan_ckpt() -> str:
    pattern = os.path.join(ROOT, 'outputs', 'ckpts_gan', 'G_step*.pt')
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return ''   # GAN is optional — caller checks for empty string
    return candidates[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_aerial(path: str, size: int = 224) -> torch.Tensor:
    """Load and centre-crop an aerial image to (size × size), normalise to [-1, 1]."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if w < size or h < size:
        img = img.resize((size, size), Image.BILINEAR)
    else:
        left = (w - size) // 2
        top  = (h - size) // 2
        img  = img.crop((left, top, left + size, top + size))
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return t.unsqueeze(0)   # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: CrossNet — aerial → semantic map
# ─────────────────────────────────────────────────────────────────────────────

def load_crossnet(ckpt_path: str, num_classes: int, device: torch.device) -> CrossNet:
    print(f'[Stage 1] Loading CrossNet from:\n          {ckpt_path}')
    model = CrossNet(num_classes=num_classes, conditioned=True, pretrained=False).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def run_crossnet(model: CrossNet, aerial_tensor: torch.Tensor,
                 device: torch.device) -> tuple:
    """
    Returns
    -------
    Lg          : (1, C, 8, 40)  raw ground logits
    sem_map_pil : PIL image of the upscaled semantic map (display resolution)
    """
    aerial_tensor = aerial_tensor.to(device)
    with torch.no_grad():
        _La, Lg = model(aerial_tensor)

    # Upsample semantic logits to a readable resolution for display
    prob_g  = torch.softmax(Lg, dim=1)
    prob_up = F.interpolate(prob_g, size=(224, 1232), mode='bilinear', align_corners=True)
    pred_up = prob_up.argmax(dim=1)[0].cpu().numpy()   # (224, 1232)

    colors  = CLASS_COLORS[:model.num_classes]
    sem_rgb = colors[pred_up]                           # (224, 1232, 3)
    sem_pil = Image.fromarray(sem_rgb)

    return Lg, sem_pil


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: GAN — semantic map → synthesised RGB panorama
# ─────────────────────────────────────────────────────────────────────────────

def load_gan(ckpt_path: str, num_classes: int, device: torch.device):
    """Load GAN generator; returns None if no checkpoint given."""
    if not ckpt_path:
        print('[Stage 2] No GAN checkpoint found — skipping synthesis stage.\n'
              '          Train with:  python train_gan_pix2pix.py')
        return None
    print(f'[Stage 2] Loading GAN generator from:\n          {ckpt_path}')
    G = UNetGenerator(in_ch=num_classes, ngf=64).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(ck['model'])
    G.eval()
    return G


def run_gan(G, Lg: torch.Tensor, device: torch.device,
            out_size: tuple = (224, 1232)) -> Image.Image:
    """
    Parameters
    ----------
    Lg       : (1, C, 8, 40)  ground logits from CrossNet
    out_size : (H, W) for the output PIL image
    """
    # One-hot encode and upsample to GAN training resolution
    cls_map = Lg.argmax(dim=1)                           # (1, 8, 40)
    B, C, H, W = Lg.shape
    one_hot = torch.zeros(B, C, H, W, device=device)
    one_hot.scatter_(1, cls_map.unsqueeze(1), 1.0)
    one_hot = F.interpolate(one_hot, size=(GAN_IMG_H, GAN_IMG_W), mode='nearest')
    gan_input = one_hot * 2.0 - 1.0                      # [0,1] → [-1,1]

    with torch.no_grad():
        fake = G(gan_input)                              # (1, 3, 256, 512) in [-1,1]

    fake_up = F.interpolate(fake, size=out_size, mode='bilinear', align_corners=True)
    arr = ((fake_up[0].cpu().float().numpy() * 0.5 + 0.5) * 255
           ).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Display / save
# ─────────────────────────────────────────────────────────────────────────────

def build_display_montage(aerial_path: str,
                          sem_pil: Image.Image,
                          gan_pil) -> Image.Image:
    """
    Compose a side-by-side montage for on-screen display and saving.

    Layout (all panels scaled to 224 px height):
      [ Aerial Input (224×224) ] [ CrossNet Semantic Map (224×1232) ] [ GAN Output (224×1232) ]
      (if GAN output is None, only the first two panels are shown)
    """
    row_h = 224

    aerial_img = Image.open(aerial_path).convert('RGB')
    aw, ah = aerial_img.size
    if aw < 224 or ah < 224:
        aerial_img = aerial_img.resize((224, 224), Image.BILINEAR)
    else:
        left = (aw - 224) // 2
        top  = (ah - 224) // 2
        aerial_img = aerial_img.crop((left, top, left + 224, top + 224))

    sem_pil  = sem_pil.resize((sem_pil.width, row_h), Image.NEAREST)

    panels = [aerial_img, sem_pil]
    if gan_pil is not None:
        gan_pil = gan_pil.resize((gan_pil.width, row_h), Image.BILINEAR)
        panels.append(gan_pil)

    total_w = sum(p.width for p in panels)
    montage = Image.new('RGB', (total_w, row_h), (20, 20, 20))
    x = 0
    for p in panels:
        montage.paste(p, (x, 0))
        x += p.width
    return montage


def display_results(aerial_path: str,
                    sem_pil: Image.Image,
                    gan_pil,
                    save_dir: str):
    """Show three-panel result in a matplotlib window and save to disk."""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    aerial_img = Image.open(aerial_path).convert('RGB')
    aw, ah = aerial_img.size
    if aw < 224 or ah < 224:
        aerial_img = aerial_img.resize((224, 224), Image.BILINEAR)
    else:
        left = (aw - 224) // 2
        top  = (ah - 224) // 2
        aerial_img = aerial_img.crop((left, top, left + 224, top + 224))

    has_gan = gan_pil is not None
    ncols   = 3 if has_gan else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                             gridspec_kw={'wspace': 0.04})

    axes[0].imshow(aerial_img)
    axes[0].set_title('Aerial Input', fontsize=13, fontweight='bold', pad=8)
    axes[0].axis('off')

    axes[1].imshow(sem_pil)
    axes[1].set_title('CrossNet — Semantic Map', fontsize=13, fontweight='bold', pad=8)
    axes[1].axis('off')

    if has_gan:
        axes[2].imshow(gan_pil)
        axes[2].set_title('GAN — Synthesised Ground View', fontsize=13,
                          fontweight='bold', pad=8)
        axes[2].axis('off')

    fig.suptitle('Aerial → Ground Synthesis Pipeline', fontsize=15,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    base     = os.path.splitext(os.path.basename(aerial_path))[0]
    out_path = os.path.join(save_dir, f'{base}_pipeline_result.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'\n[*] Result saved → {out_path}')

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Validate input ───────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        sys.exit(f'[ERROR] Input file not found: {args.input}')

    # ── Resolve checkpoints ──────────────────────────────────────────────────
    crossnet_ckpt = args.crossnet_ckpt or _latest_crossnet_ckpt()
    gan_ckpt      = args.gan_ckpt      or _latest_gan_ckpt()

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print('  Aerial -> Ground Synthesis Pipeline')
    print(f'{"="*60}')
    print(f'  Input image   : {args.input}')
    print(f'  CrossNet ckpt : {crossnet_ckpt}')
    print(f'  GAN ckpt      : {gan_ckpt or "(none — synthesis stage skipped)"}')
    print(f'  Device        : {device}')
    print(f'{"="*60}\n')

    # ── Stage 1: CrossNet ────────────────────────────────────────────────────
    crossnet_model = load_crossnet(crossnet_ckpt, args.num_classes, device)
    aerial_tensor  = preprocess_aerial(args.input)
    Lg, sem_pil    = run_crossnet(crossnet_model, aerial_tensor, device)
    print('[Stage 1] Done — semantic map generated.')

    # ── Stage 2: GAN ─────────────────────────────────────────────────────────
    G       = load_gan(gan_ckpt, args.num_classes, device)
    gan_pil = run_gan(G, Lg, device) if G is not None else None
    if gan_pil is not None:
        print('[Stage 2] Done — ground-view panorama synthesised.')

    # ── Save raw outputs ─────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    sem_pil.save(os.path.join(args.save_dir, f'{base}_semantic_map.png'))
    if gan_pil is not None:
        gan_pil.save(os.path.join(args.save_dir, f'{base}_ground_synthesised.png'))

    # ── Montage (always saved) ───────────────────────────────────────────────
    montage = build_display_montage(args.input, sem_pil, gan_pil)
    montage_path = os.path.join(args.save_dir, f'{base}_full_pipeline_montage.png')
    montage.save(montage_path)
    print(f'[*] Montage saved → {montage_path}')

    # ── On-screen display ────────────────────────────────────────────────────
    if not args.no_display:
        display_results(args.input, sem_pil, gan_pil, args.save_dir)
    else:
        print(f'[*] Display skipped (--no_display).  All outputs in: {args.save_dir}')


if __name__ == '__main__':
    main()
