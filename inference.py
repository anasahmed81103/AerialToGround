"""
inference.py  --  CrossViewNet Quickstart Inference Script
===========================================================
Single-command entry point:  feed any aerial image and receive the
full two-stage ground-level synthesis result.

Quickstart
----------
  python inference.py --source demo/sample_aerial_road.jpg
  python inference.py --source demo/           # batch: all images in folder
  python inference.py --source demo/ --no_display  # headless / server mode

Explicit checkpoint selection
------------------------------
  python inference.py --source demo/sample_aerial_road.jpg \\
      --weights checkpoints/best.pt            # single combined checkpoint
  python inference.py --source demo/ \\
      --crossnet outputs/ckpts_pt/crossnet_step0088830.pt \\
      --gan     outputs/ckpts_gan/G_step0197766.pt

Output
------
  outputs/pipeline_results/<input_stem>_full_pipeline_montage.png
  outputs/pipeline_results/<input_stem>_semantic_map.png
  outputs/pipeline_results/<input_stem>_ground_synthesised.png
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# -- local modules --------------------------------------------------------
from crossnet_model import CrossNet
from gan_unet_model import UNetGenerator
from semantic_metrics import CLASS_COLORS

# -------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = 4
GAN_H, GAN_W = 256, 512


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='CrossViewNet: Aerial -> Ground Synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--source', '-s', required=True,
                   help='Path to an aerial image OR a folder of images')
    p.add_argument('--crossnet', default='',
                   help='CrossNet checkpoint (.pt).  Default: auto-detect latest')
    p.add_argument('--gan', default='',
                   help='GAN generator checkpoint (.pt).  Default: auto-detect latest')
    p.add_argument('--weights', default='',
                   help='Combined checkpoint (overrides --crossnet / --gan if provided)')
    p.add_argument('--save_dir', default=os.path.join(ROOT, 'outputs', 'pipeline_results'),
                   help='Where to write result images')
    p.add_argument('--no_display', action='store_true',
                   help='Do not open a matplotlib window (save-only mode)')
    p.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    return p.parse_args()


# -------------------------------------------------------------------------
# Checkpoint auto-discovery
# -------------------------------------------------------------------------

def _latest(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else ''


def resolve_checkpoints(args):
    """Return (crossnet_ckpt, gan_ckpt) strings."""
    if args.weights:
        # A single combined-checkpoint file (future-proofing)
        return args.weights, args.weights

    crossnet = args.crossnet or _latest(
        os.path.join(ROOT, 'outputs', 'ckpts_pt', 'crossnet_step*.pt'))
    if not crossnet:
        sys.exit('[ERROR] No CrossNet checkpoint found.\n'
                 '        Train with:  python train_crossnet.py')

    gan = args.gan or _latest(
        os.path.join(ROOT, 'outputs', 'ckpts_gan', 'G_step*.pt'))

    return crossnet, gan


# -------------------------------------------------------------------------
# Image helpers
# -------------------------------------------------------------------------

def load_aerial(path: str, size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if w < size or h < size:
        img = img.resize((size, size), Image.BILINEAR)
    else:
        img = img.crop(((w - size)//2, (h - size)//2,
                        (w - size)//2 + size, (h - size)//2 + size))
    t = TF.to_tensor(img)
    t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return t.unsqueeze(0)   # (1, 3, H, W)


def collect_inputs(source: str):
    """Return a list of image file paths from a file or directory."""
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    p = Path(source)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in EXTS)
    # Glob pattern
    return sorted(glob.glob(source))


# -------------------------------------------------------------------------
# Stage 1 — CrossNet
# -------------------------------------------------------------------------

def load_crossnet(ckpt: str, num_classes: int, device: torch.device) -> CrossNet:
    print(f'[CrossNet] Loading: {ckpt}')
    m = CrossNet(num_classes=num_classes, conditioned=True, pretrained=False).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


@torch.no_grad()
def run_crossnet(model: CrossNet, t: torch.Tensor, device: torch.device):
    t = t.to(device)
    _La, Lg = model(t)
    prob_up = F.interpolate(torch.softmax(Lg, 1), (224, 1232), mode='bilinear', align_corners=True)
    pred    = prob_up.argmax(1)[0].cpu().numpy()
    sem_pil = Image.fromarray(CLASS_COLORS[:model.num_classes][pred])
    return Lg, sem_pil


# -------------------------------------------------------------------------
# Stage 2 — GAN
# -------------------------------------------------------------------------

def load_gan(ckpt: str, num_classes: int, device: torch.device):
    if not ckpt:
        print('[GAN] No checkpoint — synthesis stage skipped.')
        return None
    print(f'[GAN    ] Loading: {ckpt}')
    G = UNetGenerator(in_ch=num_classes, ngf=64).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    G.load_state_dict(ck['model'])
    G.eval()
    return G


@torch.no_grad()
def run_gan(G, Lg: torch.Tensor, device: torch.device) -> Image.Image:
    B, C, H, W = Lg.shape
    one_hot = torch.zeros(B, C, H, W, device=device)
    one_hot.scatter_(1, Lg.argmax(1, keepdim=True), 1.0)
    one_hot = F.interpolate(one_hot, (GAN_H, GAN_W), mode='nearest') * 2.0 - 1.0
    fake = G(one_hot)
    fake = F.interpolate(fake, (224, 1232), mode='bilinear', align_corners=True)
    arr  = ((fake[0].cpu().float().numpy() * 0.5 + 0.5) * 255
            ).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


# -------------------------------------------------------------------------
# Visualisation
# -------------------------------------------------------------------------

def save_results(aerial_path, sem_pil, gan_pil, save_dir):
    """Save montage + individual panels; optionally show matplotlib window."""
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(aerial_path).stem

    sem_pil.save(os.path.join(save_dir, f'{stem}_semantic_map.png'))
    if gan_pil:
        gan_pil.save(os.path.join(save_dir, f'{stem}_ground_synthesised.png'))

    # Build side-by-side montage
    aerial_img = Image.open(aerial_path).convert('RGB')
    aw, ah = aerial_img.size
    s = 224
    if aw < s or ah < s:
        aerial_img = aerial_img.resize((s, s), Image.BILINEAR)
    else:
        aerial_img = aerial_img.crop(((aw-s)//2, (ah-s)//2, (aw-s)//2+s, (ah-s)//2+s))

    panels = [aerial_img,
              sem_pil.resize((sem_pil.width, 224), Image.NEAREST)]
    if gan_pil:
        panels.append(gan_pil.resize((gan_pil.width, 224), Image.BILINEAR))

    montage = Image.new('RGB', (sum(p.width for p in panels), 224), (20, 20, 20))
    x = 0
    for panel in panels:
        montage.paste(panel, (x, 0))
        x += panel.width

    montage_path = os.path.join(save_dir, f'{stem}_full_pipeline_montage.png')
    montage.save(montage_path)
    print(f'[*] Saved -> {montage_path}')
    return montage_path


def show_result(aerial_path, sem_pil, gan_pil, save_dir):
    """Display using matplotlib and save."""
    import matplotlib.pyplot as plt

    aerial_img = Image.open(aerial_path).convert('RGB')
    aw, ah = aerial_img.size
    s = 224
    if aw < s or ah < s:
        aerial_img = aerial_img.resize((s, s), Image.BILINEAR)
    else:
        aerial_img = aerial_img.crop(((aw-s)//2, (ah-s)//2, (aw-s)//2+s, (ah-s)//2+s))

    ncols = 3 if gan_pil else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 4))
    axes[0].imshow(aerial_img);    axes[0].set_title('Aerial Input',            fontweight='bold'); axes[0].axis('off')
    axes[1].imshow(sem_pil);       axes[1].set_title('CrossNet Semantic Map',   fontweight='bold'); axes[1].axis('off')
    if gan_pil:
        axes[2].imshow(gan_pil);   axes[2].set_title('GAN Synthesised Ground',  fontweight='bold'); axes[2].axis('off')

    fig.suptitle('CrossViewNet: Aerial -> Ground Synthesis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    stem = Path(aerial_path).stem
    fig.savefig(os.path.join(save_dir, f'{stem}_pipeline_result.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args    = get_args()
    inputs  = collect_inputs(args.source)
    if not inputs:
        sys.exit(f'[ERROR] No images found at: {args.source}')

    crossnet_ckpt, gan_ckpt = resolve_checkpoints(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"="*60}')
    print('  CrossViewNet: Aerial -> Ground Synthesis')
    print(f'{"="*60}')
    print(f'  Images found : {len(inputs)}')
    print(f'  CrossNet     : {crossnet_ckpt}')
    print(f'  GAN          : {gan_ckpt or "(none - synthesis skipped)"}')
    print(f'  Device       : {device}')
    print(f'{"="*60}\n')

    # Load models once
    crossnet = load_crossnet(crossnet_ckpt, args.num_classes, device)
    G        = load_gan(gan_ckpt, args.num_classes, device)

    for img_path in inputs:
        print(f'\n[*] Processing: {img_path}')
        t          = load_aerial(img_path)
        Lg, sem_pil = run_crossnet(crossnet, t, device)
        gan_pil    = run_gan(G, Lg, device) if G else None

        save_results(img_path, sem_pil, gan_pil, args.save_dir)
        if not args.no_display:
            show_result(img_path, sem_pil, gan_pil, args.save_dir)

    print(f'\n[Done] All results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
