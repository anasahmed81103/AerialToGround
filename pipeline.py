"""
End-to-end inference: aerial tile -> predicted layout + warped aerial + RGB pano.

Usage:
  python pipeline.py --aerial path/to/tile.jpg --out outputs/demo_run/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aerial_warp import PanoAerialWarper
from cache_dino_features import IMAGENET_MEAN, IMAGENET_STD, load_backbone
from eval_gan_pix2pix import load_generator
from gan_layout_codec import class_indices_to_onehot
from polar_layout_model import PolarLayoutNet
from semantic_metrics import colorize_label, denorm_image
from train_gan_pix2pix import generator_forward

DEFAULT_STAGE1 = os.path.join('experiments', 'phase2', 'runs', 'polar_full', 'best.pt')
DEFAULT_GAN = os.path.join(
    'experiments', 'phase2', 'runs', 'spade_aerial_v2', 'ckpts', 'G_step0032000.pt',
)
DEFAULT_DINO_ID = 'facebook/dinov2-with-registers-small'
DEFAULT_DINO_SIZE = 336
DEFAULT_DINO_CROP = 0
IMG_H, IMG_W = 256, 512
NUM_CLASSES = 4


def preprocess_aerial_tile(
    aerial: Union[str, Image.Image],
    size: int = DEFAULT_DINO_SIZE,
    crop: int = DEFAULT_DINO_CROP,
) -> torch.Tensor:
    """Same preprocessing as cache_dino_features.TileDataset (single image)."""
    if isinstance(aerial, (str, os.PathLike)):
        img = Image.open(aerial).convert('RGB')
    else:
        img = aerial.convert('RGB')
    if crop > 0:
        w, h = img.size
        l, t = (w - crop) // 2, (h - crop) // 2
        img = img.crop((l, t, l + crop, t + crop))
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD


@torch.inference_mode()
def extract_dino_features(
    model,
    n_reg: int | None,
    channels: int,
    stride: int,
    size: int,
    batch_nchw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Match cache_dino_features forward; returns float32 (N, C, g, g)."""
    g = size // stride
    x = batch_nchw.to(device)
    if n_reg is None:
        feat = model(x)
    else:
        tokens = model(pixel_values=x).last_hidden_state
        patches = tokens[:, 1 + n_reg :, :]
        b = patches.size(0)
        feat = patches.transpose(1, 2).reshape(b, channels, g, g)
    return feat.float()


def load_polar_from_checkpoint(
    stage1_ckpt: str, in_ch: int, in_grid: int, device: torch.device,
) -> PolarLayoutNet:
    ck = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    run_dir = os.path.dirname(os.path.abspath(stage1_ckpt))
    cfg_path = os.path.join(run_dir, 'config.json')
    cfg = ck.get('config') if isinstance(ck.get('config'), dict) else None
    if cfg is None and os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path, encoding='utf-8'))
    if cfg is None:
        raise FileNotFoundError(f'No config in ckpt or {cfg_path}')

    model = PolarLayoutNet(
        in_ch=in_ch,
        dim=int(cfg['dim']),
        num_classes=NUM_CLASSES,
        radial_bins=int(cfg['radial_bins']),
        out_hw=(int(cfg['out_h']), int(cfg['out_w'])),
        horizon_frac=float(cfg['horizon']),
        heads=int(cfg['heads']),
        layers=int(cfg['layers']),
        use_offsets=not bool(cfg.get('no_offsets', False)),
        prior_mode=str(cfg['prior']),
        mapping=str(cfg.get('mapping', 'polar')),
        in_grid=in_grid,
    ).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class AerialToGround:
    """Stage-1 PolarLayoutNet + SPADE+aerial Stage-2 (config E)."""

    def __init__(
        self,
        stage1_ckpt: str = DEFAULT_STAGE1,
        gan_ckpt: str = DEFAULT_GAN,
        device: str = 'auto',
        dino_model_id: str = DEFAULT_DINO_ID,
        dino_size: int = DEFAULT_DINO_SIZE,
        dino_crop: int = DEFAULT_DINO_CROP,
    ):
        self.device = resolve_device(device)
        self.dino_size = dino_size
        self.dino_crop = dino_crop
        self.dino_model_id = dino_model_id

        self.dino, self.n_reg, self.dino_c, self.dino_stride = load_backbone(
            dino_model_id, self.device,
        )
        in_grid = dino_size // self.dino_stride
        self.polar = load_polar_from_checkpoint(
            stage1_ckpt, self.dino_c, in_grid, self.device,
        )
        self.G, self.g_step = load_generator(
            gan_ckpt,
            NUM_CLASSES,
            64,
            self.device,
            generator='spade',
            img_h=IMG_H,
            img_w=IMG_W,
            aux_aerial=True,
        )
        self.warper = PanoAerialWarper((IMG_H, IMG_W), device=self.device)

    @torch.inference_mode()
    def predict(self, aerial_image: Union[str, Image.Image]) -> dict:
        """
        Returns dict with uint8 HWC images:
          layout_rgb, warped_aerial_rgb, panorama_rgb
        """
        aerial_path = aerial_image if isinstance(aerial_image, (str, os.PathLike)) else None
        x = preprocess_aerial_tile(aerial_image, self.dino_size, self.dino_crop).unsqueeze(0)
        feat = extract_dino_features(
            self.dino, self.n_reg, self.dino_c, self.dino_stride,
            self.dino_size, x, self.device,
        )

        logits = self.polar(feat).float()
        logits = F.interpolate(
            logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=True,
        )
        cls_map = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        layout_rgb = colorize_label(cls_map, NUM_CLASSES)

        label_t = class_indices_to_onehot(cls_map, NUM_CLASSES).unsqueeze(0).to(self.device)
        warp_src = aerial_path if aerial_path is not None else aerial_image
        aerial_t = self.warper.warp_aerial_to_pano(warp_src).unsqueeze(0).to(self.device)

        fake = generator_forward(self.G, label_t, aerial_t, True)
        pano_rgb = denorm_image(fake[0])
        warped_rgb = denorm_image(aerial_t[0])

        return {
            'layout_rgb': layout_rgb,
            'warped_aerial_rgb': warped_rgb,
            'panorama_rgb': pano_rgb,
            'class_map': cls_map,
        }


def read_val_row(csv_path: str, index: int) -> dict:
    with open(csv_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                parts = [p.strip() for p in line.strip().split(',')]
                return {'aerial': parts[0], 'ground': parts[1], 'label': parts[2]}
    raise IndexError(f'Row {index} not in {csv_path}')


def save_montage(aerial_rgb: np.ndarray, layout_rgb: np.ndarray,
                 warp_rgb: np.ndarray, pano_rgb: np.ndarray, path: str) -> None:
    """Horizontal strip: aerial | layout | warp | pano (resize aerial to pano H)."""
    h, w = pano_rgb.shape[:2]
    def _fit(img):
        pil = Image.fromarray(img).resize((w, h), Image.BILINEAR)
        return np.array(pil)
    strip = np.concatenate(
        [_fit(aerial_rgb), _fit(layout_rgb), warp_rgb, pano_rgb], axis=1,
    )
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(strip).save(path)


def load_aerial_uint8(aerial: Union[str, Image.Image]) -> np.ndarray:
    if isinstance(aerial, (str, os.PathLike)):
        return np.array(Image.open(aerial).convert('RGB'))
    return np.array(aerial.convert('RGB'))


def main() -> None:
    p = argparse.ArgumentParser(description='Aerial2Ground E2E inference')
    p.add_argument('--aerial', required=True, help='Path to aerial RGB tile')
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--stage1_ckpt', default=DEFAULT_STAGE1)
    p.add_argument('--gan_ckpt', default=DEFAULT_GAN)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    pipe = AerialToGround(args.stage1_ckpt, args.gan_ckpt, args.device)
    out = pipe.predict(args.aerial)
    os.makedirs(args.out, exist_ok=True)

    aerial_u8 = load_aerial_uint8(args.aerial)
    Image.fromarray(out['layout_rgb']).save(os.path.join(args.out, 'layout.png'))
    Image.fromarray(out['warped_aerial_rgb']).save(os.path.join(args.out, 'warped_aerial.png'))
    Image.fromarray(out['panorama_rgb']).save(os.path.join(args.out, 'panorama.png'))
    save_montage(
        aerial_u8, out['layout_rgb'], out['warped_aerial_rgb'], out['panorama_rgb'],
        os.path.join(args.out, 'montage.png'),
    )
    print(f'[*] wrote {args.out}')


if __name__ == '__main__':
    main()
