"""
bootstrap_ci.py — 95% CIs for final_eval per-image metrics.

Image-level bootstrap (1000 resamples, seed 0). Paired CIs for D−B and E−C
on SSIM, PSNR, LPIPS (aligned by index column in NPZ files).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from final_eval import CONFIG_ORDER

N_BOOT = 1000
SEED = 0
METRICS = ('ssim', 'psnr', 'lpips', 'sd')

PAIRED = (
    ('D', 'B', 'spade_aerial_v2_gt', 'spade_v2_gt'),
    ('E', 'C', 'spade_aerial_v2_pred', 'spade_v2_pred'),
)


def load_metric_by_index(eval_dir: str, slug: str) -> dict[str, np.ndarray]:
    path = os.path.join(eval_dir, f'{slug}_per_image.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    idx = data['index'].astype(np.int64)
    out = {m: data[m].astype(np.float64) for m in METRICS if m in data}
    order = np.argsort(idx)
    idx = idx[order]
    for m in out:
        out[m] = out[m][order]
    out['_index'] = idx
    return out


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        boots[b] = values[pick].mean()
    return {
        'mean': float(values.mean()),
        'ci95_low': float(np.percentile(boots, 2.5)),
        'ci95_high': float(np.percentile(boots, 97.5)),
        'n': int(n),
    }


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch {a.shape} vs {b.shape}')
    diff = a - b
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    boots = np.empty(n_boot, dtype=np.float64)
    for t in range(n_boot):
        pick = rng.integers(0, n, size=n)
        boots[t] = diff[pick].mean()
    return {
        'mean_diff': float(diff.mean()),
        'ci95_low': float(np.percentile(boots, 2.5)),
        'ci95_high': float(np.percentile(boots, 97.5)),
        'n': int(n),
    }


def align_pair(slug_a: str, slug_b: str, eval_dir: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    da = load_metric_by_index(eval_dir, slug_a)
    db = load_metric_by_index(eval_dir, slug_b)
    common = np.intersect1d(da['_index'], db['_index'])
    if common.size == 0:
        raise SystemExit(f'No overlapping indices for {slug_a} vs {slug_b}')
    ia = {int(i): k for k, i in enumerate(da['_index'])}
    ib = {int(i): k for k, i in enumerate(db['_index'])}
    out_a = {m: np.array([da[m][ia[int(i)]] for i in common]) for m in METRICS if m in da}
    out_b = {m: np.array([db[m][ib[int(i)]] for i in common]) for m in METRICS if m in db}
    return out_a, out_b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--eval_dir', default=os.path.join('experiments', 'final_eval'))
    p.add_argument('--out_json', default='')
    args = p.parse_args()

    summary: dict = {'per_config': {}, 'paired': {}}

    for cfg in CONFIG_ORDER:
        try:
            blob = load_metric_by_index(args.eval_dir, cfg.slug)
        except FileNotFoundError:
            print(f'[!] skip {cfg.key}: missing NPZ', flush=True)
            continue
        summary['per_config'][cfg.key] = {
            m: bootstrap_ci_mean(blob[m]) for m in METRICS if m in blob
        }

    for key_a, key_b, slug_a, slug_b in PAIRED:
        try:
            a, b = align_pair(slug_a, slug_b, args.eval_dir)
        except (FileNotFoundError, SystemExit) as e:
            print(f'[!] skip pair {key_a}-{key_b}: {e}', flush=True)
            continue
        pair_name = f'{key_a}_minus_{key_b}'
        summary['paired'][pair_name] = {
            m: paired_bootstrap_ci(a[m], b[m])
            for m in ('ssim', 'psnr', 'lpips')
            if m in a and m in b
        }

    text = json.dumps(summary, indent=2)
    print(text, flush=True)
    out_path = args.out_json or os.path.join(args.eval_dir, 'bootstrap_ci.json')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'[*] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
