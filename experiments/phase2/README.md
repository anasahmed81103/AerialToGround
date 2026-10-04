# Phase 2 — research log (Stage 1: aerial -> ground semantic layout)

Every row changes **one thing** and records its measured effect. All numbers: full
test set (8,884 pairs, `cvpr_val_v2.csv`, SegFormer v2 labels), mIoU at 224x1232,
best epoch picked on a 2,000-image minival carved from train (seed 0), never on test.
Hardware: one Quadro P1000 (4 GB). GPU spend: $0.

## 1. Changes vs Phase 1 and why

| # | Change | Why | Where |
|---|---|---|---|
| C1 | Fixed eval protocol: minival from train for selection, full 8,884 test for reporting | Phase 1 picked its checkpoint on the same 1,024 test images it reported | `train_polar.py` |
| C2 | Full 750px aerial tile (resized to 336) instead of the 224px center crop | Old loader saw ~9% of the tile; ground panorama sees far beyond that | `cache_dino_features.py` |
| C3 | Frozen DINOv2-S/14 + registers, features cached once | Strong generic features, no backbone training, cheap epochs | `cache_dino_features.py` |
| C4 | Polar resampling of aerial features (tile centre -> bottom row, edge -> horizon, north at centre column, clockwise) | CVUSA geometry measured, not assumed | `geo_calibrate.py`, `polar_layout_model.py` |
| C5 | Column decoder: row queries attend along each azimuth ray, weights shared over azimuth | Rotation-equivariant, 1.5M params instead of a dense per-pixel map | `polar_layout_model.py` |
| C6 | Ground-plane attention prior + learned sampling offsets | Hypothesis: helps the decoder find the right radius | `polar_layout_model.py` |
| C7 | Dropped the aerial auxiliary loss | SegFormer-ADE20K labels on nadir tiles are wrong (forest -> building, field -> road) | finding, see `../phase1/showcase/relabel_aerial.jpg` |

## 2. Main result (fair, same labels, same 8,884 test images)

| Model | mIoU | sky | veg | road | building | pixel acc |
|---|---|---|---|---|---|---|
| Baseline CrossNet (VGG-16, dense M) | 0.299 | 0.638 | 0.026 | 0.438 | 0.095 | 0.434 |
| Phase 1 (v2 labels, focal+dice, flip, aux) | 0.402 | 0.633 | 0.533 | 0.362 | 0.080 | 0.665 |
| **Phase 2 `polar_full`** | **0.514** | **0.689** | **0.661** | **0.494** | **0.209** | **0.745** |

+0.215 mIoU (+72% relative) over the baseline. 234 min training, 1.98 GB peak VRAM.

## 3. Ablations

Round 1 — decoder components (done):

| Variant | Changed | mIoU | Impact |
|---|---|---|---|
| `polar_full` | — | 0.5135 | reference |
| `polar_prior_none` | attention prior removed | 0.5139 | none |
| `polar_prior_fixed` | prior frozen, not learned | 0.5133 | none |
| `polar_no_offsets` | learned offsets removed | 0.5142 | none |

Finding: C6 does not help (all within +/-0.001). It is not claimed as a contribution;
the simpler model (no prior, no offsets) is equally good.

Round 2 — where the gain comes from (done):

| Variant | Changed | mIoU | Impact |
|---|---|---|---|
| `polar_dense_mapping` | polar geometry -> learned dense matrix (CrossNet idea), 4.5M params | 0.5138 | **~0** — geometry not required once DINOv2 + column decoder |
| `polar_crop224` | full tile -> 224px center crop, 16x16 tokens | 0.5019 | **~−0.012** — full context helps |
| `polar_vgg16` | DINOv2 -> frozen ImageNet VGG-16 relu5_3, 21x21 | 0.4852 | **~−0.028** — backbone is the largest single factor |

**Stage 1 conclusion:** keep `polar_full` (0.514) as the showcase checkpoint; cite DINOv2 + full tile + shared column decoder. Do not over-claim polar mapping, attention prior, or offsets. Class-weighted loss (`polar_full_classweighted`, 0.501) trades overall mIoU for slightly higher building IoU — **Stage 1 locked on `polar_full`.**

Round 3 — native head resolution (see `code/run_polar_full_64x320.ps1`):

| Variant | Changed | mIoU | Impact |
|---|---|---|---|
| `polar_full_64x320` | native head 32x160 -> **64x320** (same test upsample to 224x1232) | **0.5126** | **~0 vs 32x160** — finer native grid does not help at full-res eval |

Architecture note: `PolarLayoutNet` has **no** fixed upsampling stages; `out_hw` sets `row_queries` (H), polar grid width W, and ground-plane prior shape directly. No code change required for 64x320.

Round 4 — class-weighted loss (last Stage 1 experiment, pending RunPod):

| Variant | Changed | mIoU | Impact |
|---|---|---|---|
| `polar_full_classweighted` | `--class_weights auto` on focal+dice (inverse freq from `labels_32x160_train.npy`) | **0.5006** | building **0.237** (+0.03) but overall mIoU **−0.013** vs polar_full; not used for deploy |

New modules: `phase2_class_weights.py`, `phase2_weighted_loss.py`. Launch: `experiments/phase2/code/run_polar_full_classweighted.sh`.

---

## Stage 2 (Pix2Pix GAN) — next

| Item | Choice |
|------|--------|
| Stage 1 layout for **paper / mIoU** | `polar_full/best.pt` (0.514) |
| GAN **training** supervision | GT `labels_v2/ground/` + panos via `cvpr_train_v2.csv` |
| GAN **demo / E2E** | `polar_full` predicted layouts at inference only |
| Warm-start | `G_step0197766.pt` / `D_step0197766.pt` (v1 labels) |
| Metrics | `eval_gan_pix2pix.py` — SSIM, LPIPS, FID on val |
| Smoke (local) | `python smoke_gan_v2.py` |
| Full retrain (RunPod) | `experiments/phase2/code/run_gan_v2_retrain.sh` |

Paper one-liner for class-weighted: inverse-frequency loss improves building IoU (+0.03) but lowers mIoU (−0.013); main Stage 1 result stays unweighted `polar_full`.

## 4. Files

- `runs/<name>/`: `config.json`, `minival_curve.csv`, `best.pt`, `results.json`, `showcase/`
- `results/geo_calibration.json`, `showcase/geo_calibration.jpg`: measured CVUSA geometry
- `code/`: frozen snapshot of the Phase 2 scripts
- Feature caches (`cache/`) are gitignored and regenerable with `cache_dino_features.py`
