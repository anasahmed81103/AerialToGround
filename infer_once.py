"""One-off inference: random aerial from cvpr_train.csv, save predictions under outputs/infer_demo."""
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from model import CrossNet

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "outputs", "ckpts_pt", "crossnet_step0088830.pt")


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
    model = CrossNet(
        num_classes=4, conditioned=True, pretrained=False
    ).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

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

    pa = prob_a[0].detach().cpu().numpy()
    pa_img = (pa * 255.0 / pa.max(axis=0, keepdims=True).clip(1e-6)).astype(np.uint8)
    pa_img = np.transpose(pa_img, (1, 2, 0))
    if pa_img.shape[2] >= 3:
        Image.fromarray(pa_img[:, :, :3]).save(
            os.path.join(out_dir, f"{base}_aerial_sem_17x17.png")
        )

    print(f"[*] device: {device}")
    print(f"[*] aerial: {img_path}")
    print(f"[*] ground (real RGB): {ground_path}")
    print(f"[*] saved: {out_dir}")
    print(
        "[*] Note: CrossNet predicts semantic maps only; "
        "'ground_real' is the dataset street-view photo for this aerial pair."
    )


if __name__ == "__main__":
    main()
