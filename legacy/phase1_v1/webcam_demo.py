"""
webcam_demo.py  --  Live Webcam / Video Inference Demo
=======================================================
Captures frames from a webcam (or video file), runs the CrossViewNet
pipeline on each frame, and displays a live side-by-side window:

  [ Webcam / Video Frame ]  |  [ CrossNet Semantic Map ]  |  [ GAN Synthesised Ground ]

Note on semantic meaning
------------------------
This demo treats *any* image feed as if it were aerial imagery.  For best
visual results, use top-down (bird's-eye) imagery.  You can, for example,
point a phone camera down from a high window, use a drone feed, or simply
pass in aerial satellite image files (--source path/to/aerial.jpg).

Usage
-----
  python webcam_demo.py                         # default webcam (index 0)
  python webcam_demo.py --source 1              # second webcam
  python webcam_demo.py --source path/video.mp4 # video file
  python webcam_demo.py --no_gan                # CrossNet only (faster)
  python webcam_demo.py --save_frames           # save every synthesised frame

Controls (while window is open)
---------------------------------
  Q / Esc  — quit
  S        — save current frame to outputs/webcam_captures/

Requirements
------------
  pip install opencv-python
  (all other requirements already covered by requirements.txt)
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))

# -- local modules -------------------------------------------------------
from crossnet_model import CrossNet
from gan_unet_model  import UNetGenerator

NUM_CLASSES  = 4
GAN_H, GAN_W = 256, 512
INFER_SIZE   = 224   # CrossNet input size

CLASS_COLORS_BGR = np.array([
    [ 64,  64, 255],  # 0  sky    (red in RGB → blue in BGR)
    [ 64, 200,  64],  # 1  vegetation (green)
    [255,  64,  64],  # 2  road   (blue in RGB → red in BGR)
    [ 30, 220, 255],  # 3  building (yellow in RGB → cyan-ish in BGR)
], dtype=np.uint8)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='CrossViewNet live webcam / video inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--source', default='0',
                   help='Webcam index (int), video file path, or aerial image path')
    p.add_argument('--crossnet', default='',
                   help='CrossNet checkpoint (.pt). Default: auto-detect latest')
    p.add_argument('--gan', default='',
                   help='GAN generator checkpoint (.pt). Default: auto-detect latest')
    p.add_argument('--no_gan', action='store_true',
                   help='Skip GAN stage (faster, CrossNet only)')
    p.add_argument('--save_frames', action='store_true',
                   help='Save each synthesised frame to outputs/webcam_captures/')
    p.add_argument('--fps', type=int, default=0,
                   help='Target FPS cap (0 = as fast as possible)')
    return p.parse_args()


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def _latest(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else ''


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def load_models(crossnet_ckpt, gan_ckpt, num_classes, device):
    print(f'[CrossNet] {crossnet_ckpt}')
    model = CrossNet(num_classes=num_classes, conditioned=True, pretrained=False).to(device)
    ck    = torch.load(crossnet_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    G = None
    if gan_ckpt:
        print(f'[GAN    ] {gan_ckpt}')
        G = UNetGenerator(in_ch=num_classes, ngf=64).to(device)
        ck2 = torch.load(gan_ckpt, map_location=device, weights_only=False)
        G.load_state_dict(ck2['model'])
        G.eval()
    else:
        print('[GAN    ] Skipped (no checkpoint or --no_gan)')

    return model, G


# -------------------------------------------------------------------------
# Per-frame inference
# -------------------------------------------------------------------------

@torch.no_grad()
def process_frame(bgr_frame, crossnet, G, device):
    """
    bgr_frame : HxWx3 uint8 numpy BGR (from OpenCV)
    Returns   : (sem_bgr, gan_bgr)  — each HxWx3 uint8 numpy BGR, or None
    """
    # Convert to PIL, resize/crop to 224x224
    from PIL import Image
    import torchvision.transforms.functional as TF

    pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    w, h = pil.size
    s = INFER_SIZE
    if w < s or h < s:
        pil = pil.resize((s, s), Image.BILINEAR)
    else:
        pil = pil.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))

    t = TF.normalize(TF.to_tensor(pil), [0.5]*3, [0.5]*3).unsqueeze(0).to(device)

    _La, Lg = crossnet(t)

    # Semantic map
    prob_up = F.interpolate(torch.softmax(Lg, 1), (s, s), mode='bilinear', align_corners=True)
    pred    = prob_up.argmax(1)[0].cpu().numpy()   # (224, 224)
    sem_bgr = cv2.cvtColor(CLASS_COLORS_BGR[pred], cv2.COLOR_RGB2BGR)

    if G is None:
        return sem_bgr, None

    # GAN synthesis
    B, C, H, W = Lg.shape
    one_hot = torch.zeros(B, C, H, W, device=device)
    one_hot.scatter_(1, Lg.argmax(1, keepdim=True), 1.0)
    one_hot = F.interpolate(one_hot, (GAN_H, GAN_W), mode='nearest') * 2.0 - 1.0
    fake    = G(one_hot)
    fake    = F.interpolate(fake, (s, s), mode='bilinear', align_corners=True)
    arr     = ((fake[0].cpu().float().numpy() * 0.5 + 0.5) * 255
               ).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    gan_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return sem_bgr, gan_bgr


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------

def main():
    args = get_args()

    # Resolve checkpoints
    crossnet_ckpt = args.crossnet or _latest(
        os.path.join(ROOT, 'outputs', 'ckpts_pt', 'crossnet_step*.pt'))
    if not crossnet_ckpt:
        sys.exit('[ERROR] No CrossNet checkpoint found.  Train first.')

    gan_ckpt = ''
    if not args.no_gan:
        gan_ckpt = args.gan or _latest(
            os.path.join(ROOT, 'outputs', 'ckpts_gan', 'G_step*.pt'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    crossnet, G = load_models(crossnet_ckpt, gan_ckpt, NUM_CLASSES, device)

    # Open capture
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Could not open video source: {src}')

    print('\n[Running] Press  Q / Esc  to quit    S  to save current frame.\n')

    save_dir   = os.path.join(ROOT, 'outputs', 'webcam_captures')
    frame_count = 0
    t_last      = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS cap
        if args.fps > 0:
            elapsed = time.time() - t_last
            if elapsed < 1.0 / args.fps:
                continue

        t0 = time.time()
        sem_bgr, gan_bgr = process_frame(frame, crossnet, G, device)
        infer_ms = (time.time() - t0) * 1000

        # Resize source frame for display
        disp_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))

        # Compose display: [source | semantic | gan(opt)]
        panels = [disp_frame, sem_bgr]
        if gan_bgr is not None:
            panels.append(gan_bgr)
        display = np.concatenate(panels, axis=1)

        # Overlay FPS / latency
        fps_display = 0.9 * fps_display + 0.1 * (1000.0 / max(infer_ms, 1))
        cv2.putText(display, f'{infer_ms:.0f} ms  ({fps_display:.1f} FPS)',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'Q=Quit  S=Save',
                    (8, display.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)

        col_labels = ['Input', 'Semantic Map', 'GAN Synthesis']
        for ci, lbl in enumerate(col_labels[:len(panels)]):
            cv2.putText(display, lbl,
                        (ci * INFER_SIZE + 4, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 120), 1)

        cv2.imshow('CrossViewNet Live Demo', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or Esc
            break
        if key == ord('s'):
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f'frame_{frame_count:06d}.png')
            cv2.imwrite(fname, display)
            print(f'[Saved] {fname}')

        if args.save_frames:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f'frame_{frame_count:06d}.png'), display)

        frame_count += 1
        t_last = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f'[Done] Processed {frame_count} frames.')


if __name__ == '__main__':
    main()
