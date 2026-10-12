"""
Parity checks for pipeline.py — run before demo:
  python -m pytest tests/test_pipeline_parity.py -s
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from pipeline import (
    DEFAULT_DINO_CROP,
    DEFAULT_DINO_SIZE,
    extract_dino_features,
    preprocess_aerial_tile,
    read_val_row,
    AerialToGround,
)
from cache_dino_features import load_backbone

VAL_CSV = os.path.join(ROOT, 'cvpr_val_v2.csv')
VAL_NPY = os.path.join(ROOT, 'cache', 'dinov2s_reg_336', 'val.npy')
SAMPLES_DIR = os.path.join(
    ROOT, 'experiments', 'final_eval', 'spade_aerial_v2_pred_samples',
)


def _require_paths(*paths: str) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        msg = 'Missing: ' + ', '.join(missing)
        if pytest is not None:
            pytest.skip(msg)
        raise SystemExit(msg)


def _dino_bundle(device):
    model, n_reg, c, stride = load_backbone(
        'facebook/dinov2-with-registers-small', device,
    )
    return model, n_reg, c, stride, device


def test_live_dino_matches_val_cache(dino_bundle=None):
    """(a) Live DINOv2 vs cached val.npy for 8 val rows."""
    _require_paths(VAL_CSV, VAL_NPY)
    if dino_bundle is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dino_bundle = _dino_bundle(device)
    model, n_reg, c, stride, device = dino_bundle
    feats = np.load(VAL_NPY, mmap_mode='r')
    indices = [0, 1, 2, 3, 4, 5, 6, 7]

    print('\n=== (a) Live DINO vs cache/dinov2s_reg_336/val.npy ===')
    for idx in indices:
        row = read_val_row(VAL_CSV, idx)
        x = preprocess_aerial_tile(row['aerial'], DEFAULT_DINO_SIZE, DEFAULT_DINO_CROP)
        live = extract_dino_features(
            model, n_reg, c, stride, DEFAULT_DINO_SIZE, x.unsqueeze(0), device,
        )[0].cpu().numpy()
        cached = np.array(feats[idx], dtype=np.float32)
        diff = np.abs(live - cached)
        max_abs = float(diff.max())
        cos = float(
            np.dot(live.ravel(), cached.ravel())
            / (np.linalg.norm(live.ravel()) * np.linalg.norm(cached.ravel()) + 1e-8)
        )
        print(f'  idx {idx}: max_abs_diff={max_abs:.6f}  cosine_sim={cos:.8f}')
        assert max_abs < 0.05, f'idx {idx}: DINO drift too large (max_abs={max_abs})'
        assert cos > 0.999, f'idx {idx}: cosine too low ({cos})'


def test_pipeline_pano_matches_final_eval_samples(device=None):
    """(b) Pipeline pano vs saved final_eval sample PNGs."""
    _require_paths(VAL_CSV)
    if not os.path.isdir(SAMPLES_DIR):
        msg = f'No samples dir: {SAMPLES_DIR}'
        if pytest is not None:
            pytest.skip(msg)
        raise SystemExit(msg)

    sample_files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith('.png'))
    if not sample_files:
        msg = 'No sample PNGs'
        if pytest is not None:
            pytest.skip(msg)
        raise SystemExit(msg)

    stage1 = os.path.join(ROOT, 'experiments', 'phase2', 'runs', 'polar_full', 'best.pt')
    gan = os.path.join(
        ROOT, 'experiments', 'phase2', 'runs', 'spade_aerial_v2', 'ckpts', 'G_step0032000.pt',
    )
    _require_paths(stage1, gan)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = AerialToGround(stage1, gan, device=str(device))
    print('\n=== (b) Pipeline pano vs spade_aerial_v2_pred_samples ===')
    threshold = 1.0 / 255.0
    flagged = []

    for name in sample_files:
        m = re.match(r'(\d+)\.png', name, re.I)
        if not m:
            continue
        idx = int(m.group(1))
        row = read_val_row(VAL_CSV, idx)
        out = pipe.predict(row['aerial'])
        pred = out['panorama_rgb'].astype(np.float32) / 255.0
        ref = np.array(Image.open(os.path.join(SAMPLES_DIR, name)).convert('RGB')).astype(np.float32) / 255.0
        if pred.shape != ref.shape:
            flagged.append((idx, 'shape mismatch', pred.shape, ref.shape))
            print(f'  idx {idx}: SHAPE MISMATCH {pred.shape} vs {ref.shape}')
            continue
        mad = float(np.abs(pred - ref).mean())
        print(f'  idx {idx}: mean_abs_pixel_diff={mad:.6f}  (flag if > {threshold:.6f})')
        if mad > threshold:
            flagged.append((idx, mad))

    if flagged:
        print('  FLAGGED:', flagged)
    assert not flagged, f'Pano mismatch above 1/255: {flagged}'


def run_all() -> None:
    """Run parity checks without pytest (prints only)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_live_dino_matches_val_cache(_dino_bundle(device))
    test_pipeline_pano_matches_final_eval_samples(device)
    print('\n[*] All parity checks passed.')


if __name__ == '__main__':
    run_all()
