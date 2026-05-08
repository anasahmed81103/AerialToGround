"""
One-off inference: random aerial → CrossNet semantic map → GAN ground synthesis.

Stages
  1. CrossNet  : aerial image (224×224) → semantic logits (8×40)
  2. GAN       : semantic map (256×512) → synthesised RGB ground panorama  [optional]
                 (skipped if no GAN checkpoint is found in outputs/ckpts_gan/)
"""
import os
import glob
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from model     import CrossNet
from gan_model import UNetGenerator

ROOT          = os.path.dirname(os.path.abspath(__file__))
CROSSNET_CKPT = os.path.join(ROOT, "outputs", "ckpts_pt", "crossnet_step0088830.pt")
GAN_CKPT_DIR  = os.path.join(ROOT, "outputs", "ckpts_gan")
GAN_IMG_H, GAN_IMG_W = 256, 512   # resolution GAN was trained at
NUM_CLASSES   = 4


def preprocess_aerial(path: str) -> torch.Tensor:
    aerial_size = (224, 224)
    img = Image.open(path).convert("RGB")
    aH, aW = aerial_size
    if img.size != (aW, aH):
        if img.size[0] < aW or img.size[1] < aH:
            img = img.resize((aW, aH), Image.BILINEAR)
        else:
            w_orig, h_orig = img.size
            left = (w_orig - aW) // 2
            top = (h_orig - aH) // 2
            img = img.crop((left, top, left + aW, top + aH))
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return t.unsqueeze(0)


def _load_gan(device: torch.device):
    """Load the most-recent GAN generator checkpoint; return None if none found."""
    g_ckpts = sorted(glob.glob(os.path.join(GAN_CKPT_DIR, "G_step*.pt")))
    if not g_ckpts:
        print("[!] No GAN checkpoint found in outputs/ckpts_gan/  "
              "— skipping synthesis stage.  Train first with:  python train_gan.py")
        return None
    path = g_ckpts[-1]
    print(f"[*] Loading GAN generator from {path}")
    G = UNetGenerator(in_ch=NUM_CLASSES, ngf=64).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    G.load_state_dict(ck['model'])
    G.eval()
    return G


def _semantic_logits_to_gan_input(Lg: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert CrossNet ground logits → GAN input tensor.

    Lg shape : (1, C, 8, 40)   — raw ground logits from CrossNet
    Returns  : (1, C, 256, 512) float32 one-hot in [−1, 1]
    """
    cls_map = Lg.argmax(dim=1)                                  # (1, 8, 40)  long
    # One-hot: (1, C, 8, 40)
    B, C, H, W = Lg.shape
    one_hot = torch.zeros(B, C, H, W, device=device)
    one_hot.scatter_(1, cls_map.unsqueeze(1), 1.0)
    # Upsample to GAN resolution using nearest to keep hard boundaries
    one_hot = F.interpolate(one_hot, size=(GAN_IMG_H, GAN_IMG_W),
                            mode='nearest')                      # (1, C, 256, 512)
    # Normalise: 0 → −1, 1 → +1
    return one_hot * 2.0 - 1.0


def _synthesise_rgb(G: UNetGenerator, gan_input: torch.Tensor,
                    out_size: tuple) -> Image.Image:
    """Run GAN generator and return a PIL image resized to out_size (H, W)."""
    with torch.no_grad():
        fake = G(gan_input)                                      # (1, 3, 256, 512) in [−1,1]
    fake_up = F.interpolate(fake, size=out_size, mode='bilinear', align_corners=True)
    arr = ((fake_up[0].cpu().float().numpy() * 0.5 + 0.5) * 255
           ).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


def _load_ground_panorama(path: str) -> Image.Image:
    """RGB ground panorama at dataset resolution (H=224, W=1232), PIL size (W,H)."""
    g = Image.open(path).convert("RGB")
    gW, gH = 1232, 224
    if g.size != (gW, gH):
        g = g.resize((gW, gH), Image.BILINEAR)
    return g


def main():
    with open(os.path.join(ROOT, "cvpr_train.csv"), encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and len(ln.split(",")) >= 3]
    parts = random.choice(lines).split(",")
    aerial_rel = parts[0].strip()
    ground_rel = parts[1].strip()
    img_path = os.path.join(ROOT, aerial_rel)
    ground_path = os.path.join(ROOT, ground_rel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Stage 1: CrossNet ────────────────────────────────────────────────────
    model = CrossNet(
        num_classes=NUM_CLASSES, conditioned=True, pretrained=False
    ).to(device)
    ck = torch.load(CROSSNET_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # ── Stage 2: GAN generator (optional) ───────────────────────────────────
    G = _load_gan(device)

    x = preprocess_aerial(img_path).to(device)
    with torch.no_grad():
        La, Lg = model(x)
        prob_g = torch.softmax(Lg, dim=1)
        prob_a = torch.softmax(La, dim=1)

    out_dir = os.path.join(ROOT, "outputs", "infer_demo")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]

    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8
    )
    pred = Lg.argmax(dim=1)[0].cpu().numpy()
    Image.fromarray(colors[pred]).save(
        os.path.join(out_dir, f"{base}_ground_pred_8x40.png")
    )

    prob_up = F.interpolate(
        prob_g, size=(224, 1232), mode="bilinear", align_corners=True
    )
    pred_up = prob_up.argmax(dim=1)[0].cpu().numpy()
    pred_rgb = Image.fromarray(colors[pred_up])
    pred_rgb.save(os.path.join(out_dir, f"{base}_ground_pred_semantic_224x1232.png"))

    # Real ground-view RGB (paired street panorama — model does not synthesize photos)
    ground_real = _load_ground_panorama(ground_path)
    ground_real.save(os.path.join(out_dir, f"{base}_ground_real_224x1232.png"))

    # Blend: real photo + predicted classes (50/50) for alignment check
    gr = np.array(ground_real, dtype=np.float32)
    pr = np.array(pred_rgb, dtype=np.float32)
    blend = (0.55 * gr + 0.45 * pr).clip(0, 255).astype(np.uint8)
    Image.fromarray(blend).save(
        os.path.join(out_dir, f"{base}_ground_real_plus_pred_overlay.png")
    )

    # Side-by-side: aerial | real ground | semantic prediction
    aerial_show = Image.open(img_path).convert("RGB")
    aH, aW = 224, 224
    if aerial_show.size != (aW, aH):
        if aerial_show.size[0] < aW or aerial_show.size[1] < aH:
            aerial_show = aerial_show.resize((aW, aH), Image.BILINEAR)
        else:
            w0, h0 = aerial_show.size
            left = (w0 - aW) // 2
            top = (h0 - aH) // 2
            aerial_show = aerial_show.crop((left, top, left + aW, top + aH))
    row_h = 224
    w1, w2, w3 = aerial_show.width, ground_real.width, pred_rgb.width
    montage = Image.new("RGB", (w1 + w2 + w3, row_h), (32, 32, 32))
    montage.paste(aerial_show, (0, 0))
    montage.paste(ground_real, (w1, 0))
    montage.paste(pred_rgb, (w1 + w2, 0))
    montage.save(os.path.join(out_dir, f"{base}_aerial__ground_real__pred.png"))

    # ── GAN synthesis: semantic → RGB ground view ────────────────────────────
    if G is not None:
        gan_input   = _semantic_logits_to_gan_input(Lg, device)
        synth_img   = _synthesise_rgb(G, gan_input, out_size=(224, 1232))
        synth_path  = os.path.join(out_dir, f"{base}_ground_synthesised_gan.png")
        synth_img.save(synth_path)

        # Side-by-side comparison: aerial | real ground | semantic pred | GAN synthesis
        montage4 = Image.new("RGB",
                             (aerial_show.width + ground_real.width
                              + pred_rgb.width + synth_img.width, 224),
                             (32, 32, 32))
        montage4.paste(aerial_show, (0, 0))
        montage4.paste(ground_real, (aerial_show.width, 0))
        montage4.paste(pred_rgb,    (aerial_show.width + ground_real.width, 0))
        montage4.paste(synth_img,   (aerial_show.width + ground_real.width + pred_rgb.width, 0))
        montage4.save(os.path.join(out_dir, f"{base}_aerial__real__sem__gan.png"))
        print(f"[*] GAN synthesis saved → {synth_path}")

    pa = prob_a[0].detach().cpu().numpy()
    pa_img = (pa * 255.0 / pa.max(axis=0, keepdims=True).clip(1e-6)).astype(np.uint8)
    pa_img = np.transpose(pa_img, (1, 2, 0))
    if pa_img.shape[2] >= 3:
        Image.fromarray(pa_img[:, :, :3]).save(
            os.path.join(out_dir, f"{base}_aerial_sem_17x17.png")
        )

    print(f"[*] device        : {device}")
    print(f"[*] aerial        : {img_path}")
    print(f"[*] ground (real) : {ground_path}")
    print(f"[*] saved         : {out_dir}")
    if G is None:
        print("[*] Tip: run 'python train_gan.py --max_samples 5000 --epochs 20' "
              "to train the GAN, then re-run this script for full RGB synthesis.")


if __name__ == "__main__":
    main()
