# Aerial2Ground — Full research handoff (for paper framing & next steps)

**Repo:** `AerialToGround` (GitHub: anasahmed81103/AerialToGround)  
**Task:** CVUSA-style **aerial tile → ground-view semantic layout → synthesised ground RGB panorama**  
**Dataset:** CVUSA subset in repo via CSVs; **35,532 train / 8,884 val** pairs in `cvpr_train_v2.csv` / `cvpr_val_v2.csv`  
**Labels (v2):** SegFormer-ADE20K pseudo-labels on panos + aligned maps in `labels_v2/` (4 classes: sky, vegetation, road, building)  
**Status (Oct 2025):** **Training complete.** **Final publication eval complete** (`final_eval.py`, all 5 configs + bootstrap). **No further GPU training planned.**

---

## 1. Where we started (university Phase 1 / baseline)

### 1.1 Original baseline (Zhai CVPR 2017 reimplementation)

- **CrossNet:** VGG-16, learned dense cross-view mapping, **8×40** native output grid, soft CE.
- **Trained on:** original CVUSA SegNet-style 4-class annotations (`cvpr_train.csv`).
- **Best ckpt:** `crossnet_step0088830.pt` (step 88830, 10 epochs).
- **Eval on old labels, full val 8884:** mIoU **0.454** (native 8×40: 0.412). Strong sky/road/building; **vegetation IoU ~0.087** (old label noise).

### 1.2 Phase 1 upgrade (same CrossNet architecture, better labels & training)

**Motivation:** Old labels mislabel vegetation; fair comparison requires **same v2 labels** on all models.

| Change | Rationale |
|--------|-----------|
| **SegFormer relabel** (`relabel_cvusa.py`, nvidia/segformer-b0-finetuned-ade-512-512) | Consistent 4-class pano semantics; 44,416 pairs, ~132 min |
| **32×160 grid + focal + dice + flip** | Finer layout than 8×40; harder loss for imbalance |
| **Aerial auxiliary head (0.5 weight)** | Extra supervision from nadir view (later abandoned in Phase 2 — aerial ADE labels wrong) |
| **Fresh VGG-16 init** | Not from collapsed 88830 ckpt |
| **20 epochs, ~177k steps**, bs=4 | Local GPU training |

**Phase 1 on v2 labels (fair n=1024 val during training):** final mIoU **~0.403** (veg IoU **~0.535**, building still **~0.083**).

**Fair comparison old vs new on v2 (n=1024):**

| Model | mIoU | veg IoU | building IoU |
|-------|------|---------|--------------|
| Baseline 88830 on v2 | 0.299 | 0.030 | 0.096 |
| Phase 1 best on v2 | 0.403 | 0.535 | 0.083 |
| **Delta** | **+0.10** | **+0.50** | ~flat / weak |

**Full val 8884 (v2):** Phase 1 **0.402** mIoU (`experiments/phase1/results/eval_v2_labels_full8884/`).

**Do not compare** baseline 0.45 (old labels) to Phase 1 0.40 (v2 labels) directly.

---

## 2. Phase 2 Stage 1 — layout from aerial (PolarLayoutNet)

### 2.1 Architecture (`polar_layout_model.py`)

Pipeline:

1. **Frozen DINOv2-S/14+reg** on **750px tile → 336px** input; features cached `(N, 384, 24, 24)` per CSV row (`cache/dinov2s_reg_336/`).
2. **1×1 proj + ResBlocks** on aerial tokens.
3. **Polar resampling** along rays (CVUSA geometry from `geo_calibrate.py`: north at centre column, clockwise azimuth, horizon at mid-height).
4. **Column decoder:** shared row queries attend over radial samples per azimuth (+ optional ground-plane prior & offsets).
5. **Head:** native **32×160** logits (4 classes); at test **bilinear upsample** to **224×1232** for mIoU vs GT PNGs.

**Trainable params:** ~**1.53M**. **Trainer:** `train_polar.py`. **Selection:** 2000-image **minival from train** (seed 0); **never** tune on test.

**Main run:** `polar_full` — 8 epochs, bs=16, focal(γ=2)+dice, flip, P1000 ~234 min, ~2 GB VRAM.

### 2.2 Stage 1 results (full test 8884, v2 labels)

| Model | mIoU | sky | veg | road | building |
|-------|------|-----|-----|------|----------|
| Baseline CrossNet (v2) | 0.299 | 0.638 | 0.026 | 0.438 | 0.095 |
| Phase 1 CrossNet (v2) | 0.402 | 0.633 | 0.533 | 0.362 | 0.080 |
| **polar_full (paper Stage 1)** | **0.514** | **0.689** | **0.661** | **0.494** | **0.209** |

**Checkpoint:** `experiments/phase2/runs/polar_full/best.pt` (minival best mIoU 0.5125, test 0.5135).

### 2.3 Ablations (honest null results — report in paper)

| Finding | Detail |
|---------|--------|
| Attention prior & offsets | **No effect** (~±0.001 mIoU): `polar_prior_none`, `polar_prior_fixed`, `polar_no_offsets` |
| Polar vs dense mapping | **~0**: `polar_dense_mapping` 0.514 @ 4.5M params |
| Full tile vs 224 crop | **~+0.012** mIoU for full tile |
| DINOv2 vs frozen VGG-16 | **~+0.028** mIoU |
| Native 64×320 head | **No gain** vs 32×160 at full-res eval (0.5126 vs 0.5135) |
| Class-weighted loss | Building IoU +0.03 but mIoU −0.013 → **not used** |

**Paper claim:** frozen foundation features + full-tile context + lightweight shared decoder — **not** polar prior/offsets as novel mechanism.

**Building IoU 0.21** — GT is SegFormer pseudo-labels; disclose weakness.

---

## 3. Phase 2 Stage 2 — layout → RGB synthesis

### 3.1 Training protocol (all Stage 2 GANs)

- **Supervision:** **GT** semantic maps from `labels_v2/ground/` + real panos (`cvpr_train_v2.csv`), **not** Stage-1 predictions during training.
- **Resolution:** **256×512** RGB; labels one-hot **4 × H × W** in **[-1, 1]** (`GANDataset` in `train_gan_pix2pix.py`).
- **Inference / E2E:** Stage-1 **predicted** layouts plugged in at test (see final eval E, C).
- **Warm-start:** Pix2Pix G from `G_step0197766.pt` (v1 labels). **SPADE trains from scratch.**

### 3.2 Architectures

| Generator | Class | Conditioning | Notes |
|-----------|-------|--------------|--------|
| **UNetGenerator** | Pix2Pix | 4-ch layout | Baseline Stage 2 |
| **SPADEGenerator** | SPADE | layout | λ_L1=10, λ_VGG perceptual=10 |
| **SPADE + `--aux_aerial`** | SPADE | layout + **warped aerial RGB** | `aerial_warp.py` (`PanoAerialWarper`) |

**Discriminator:** PatchGAN; **LSGAN**. Critical fix: **`--random_init_d`** when fine-tuning G on v2 — warm-started D **saturated** (D≈0, G_adv≈1). Diagnostic chain: `gan_v2_diag`, `diag2`, `diag3` → `gan_v2_random_d`.

**Training infra fixes:** weights-only ckpts by default, rolling `keep_ckpts=1`, prune before save, `vis_every=0` for long runs, disk-safe scripts on RunPod.

### 3.3 Stage 2 eval (legacy harness: `eval_gan_pix2pix.py`)

**Protocol:** GT layouts, n=8884, SSIM + LPIPS (alex) + **FID (torchvision Inception pool3 — not Clean-FID)**.

| Run | Ckpt | SSIM | LPIPS | FID (legacy) |
|-----|------|------|-------|--------------|
| gan_v2_retrain (warm D) | G step 231066 | 0.516 | 0.618 | **164.4** |
| **gan_v2_random_d** | **G_step0231066** | 0.516 | 0.610 | **157.6** |
| spade_v2 | G_step0033300 | 0.507 | 0.591 | **129.7** |
| **spade_aerial_v2** | **G_step0032000** | **0.509** | **0.565** | **116.2** |

**SPADE training notes:** D often **D_weak** late epochs; **pick ckpt by val FID**, not last step. `spade_aerial_v2`: 15 epochs total, **resumed step 12000**, training ended ~34200, **reported ckpt at save step 32000**. Steps 31080/32000/33300 on spade_v2 gave **same FID** (duplicate saves).

**Adversarial caveat for paper:** describe late training as **L1 + perceptual dominated** when D collapsed; not a stable long-run GAN game.

---

## 4. Final publication eval (Oct 2025 — DONE)

### 4.1 Harness

- **Scripts:** `final_eval.py`, `bootstrap_ci.py` (1000 image bootstrap, seed 0; paired **D−B**, **E−C**).
- **Val:** all **8884** rows `cvpr_val_v2.csv`.
- **Predicted layout:** `polar_full/best.pt` on `cache/dinov2s_reg_336/val.npy` → logits **(4,32,160)** → bilinear **`align_corners=True`** → **256×512** → argmax → same one-hot as GT (`gan_layout_codec.py`).
- **Metrics per config:** SSIM, PSNR, LPIPS(alex), SD (Laplacian-var sharpness diff), **FID/KID (`clean-fid`, mode=`clean`)**.
- **Configs (order run):**

| Key | Slug | Generator | Layout | Aerial aux |
|-----|------|-----------|--------|------------|
| **E** | spade_aerial_v2_pred | SPADE | Stage-1 **pred** | yes |
| **C** | spade_v2_pred | SPADE | Stage-1 **pred** | no |
| **D** | spade_aerial_v2_gt | SPADE | **GT** | yes |
| **B** | spade_v2_gt | SPADE | **GT** | no |
| **A** | pix2pix_random_d_gt | U-Net | **GT** | no |

**Artifacts:** `experiments/final_eval/` (also archived `final_eval_results.tgz` locally).  
**Pod run:** RTX 4090, ~**6 h** total for 5 configs + bootstrap; scipy pinned `<1.14` for clean-fid.

### 4.2 Final numbers (n=8884, clean-FID)

| Config | SSIM | PSNR | LPIPS↓ | FID_clean↓ | KID↓ |
|--------|------|------|--------|------------|------|
| **E** pred+aerial | 0.459 | 14.82 | 0.650 | **182.1** | 0.209 |
| **C** pred layout | 0.458 | 14.65 | 0.671 | 195.0 | 0.221 |
| **D** GT+aerial | 0.509 | 17.28 | 0.565 | **141.0** | 0.163 |
| **B** GT SPADE | 0.507 | 17.06 | 0.591 | 155.4 | 0.179 |
| **A** GT Pix2Pix | 0.516 | 17.07 | 0.610 | 167.4 | 0.180 |

**95% CIs:** `bootstrap_ci.json` (all means tight; n=8884).

**Paired (statistically meaningful):**

- **E − C (aerial on predicted layout):** LPIPS **−0.021** (CI excludes 0); PSNR **+0.17**; SSIM diff ~0 (CI barely includes 0).
- **D − B (aerial on GT layout):** LPIPS **−0.026**; PSNR **+0.23**; FID **141 vs 155**.

**Interpretation:**

- **Oracle gap:** GT layouts (D FID **141**) vs predicted (E **182**) — Stage-1 layout error dominates synthesis FID.
- **Legacy FID 116.2 (D, old Inception)** ≠ **141.0 (D, clean-fid)** — do not mix in one table without relabeling column.
- **Do not compare** our SSIM ~0.51 (GT layout) to CrossViewDiff ~0.37 — we use **oracle segmentation of target pano** in B/D/A; E/C is closer to true pipeline but FID still >> SOTA (~24–41) on clean-fid.
- **Headline E2E row for paper:** **Config E**.

### 4.3 Metric caveats (disclose)

- **SD:** |Var(Laplacian luma, real) − Var(Laplacian, fake)| — verify vs cited Regmi/Mathieu if reviewers ask.
- **SSIM inflation** when layout is GT (B,D,A).
- **Single seed / single run** per config; bootstrap CIs are over **images**, not training rerolls.
- **Stage 1 GT:** SegFormer pseudo-labels, not human segmentation.

---

## 5. End-to-end pipeline (intended demo)

```
Aerial PNG (CSV col 0)
  → DINOv2 cache OR live extract
  → PolarLayoutNet (polar_full/best.pt)
  → layout 256×512 one-hot [-1,1]
  → (+ warped aerial for spade_aerial_v2)
  → SPADE G (G_step0032000.pt)
  → RGB panorama 256×512
```

**E2E inference:** repo-root `pipeline.py` (DINO → PolarLayoutNet → SPADE+aerial); parity tests in `tests/test_pipeline_parity.py`; Gradio `app.py`.

---

## 6. Repository map (high signal)

| Path | Content |
|------|---------|
| `experiments/results_table.json` | Locked numeric summary (Stage 1 + legacy Stage 2 FID) |
| `experiments/phase2/MODELS.txt` | Checkpoint paths |
| `experiments/phase2/README.md` | Phase 2 research log |
| `experiments/final_eval/*` | **Publication metrics + NPZ + 24 sample PNGs + real_256x512** |
| `experiments/phase2/runs/` | Stage 1/2 runs (`.pt` gitignored) |
| `logs/*.txt` | Long training logs (spade, gan, polar) |
| `final_eval.py`, `bootstrap_ci.py`, `gan_layout_codec.py`, `gan_metrics.py` | Eval harness |

---

## 7. Compute & cost narrative

- **Stage 1 polar_full:** consumer **P1000 4GB**, ~4 h, **$0** cloud.
- **Stage 2:** RunPod **4090** for SPADE / Pix2Pix finetunes; disk crashes / checkpoint hygiene were recurring issues.
- **Final eval:** one **4090** session ~**6 h** (~$3–8 depending on $/hr) — **last planned GPU job**.

**Efficiency story (defensible):** small Stage-1 head on cached DINOv2; Stage-2 on one GPU — **not** unique vs Sat2Density (~20 h V100), **much cheaper** than diffusion baselines (CrossViewDiff 8×A100).

---

## 8. Literature positioning (from prior advisor review)

- **CrossViewDiff / Sat2Density / S2SP / CDTE** report FID ~24–41, SSIM ~0.28–0.37 on CVUSA with **different protocols** (true satellite-only or different FID code).
- Our work fits **efficiency + diagnostics + reproducibility**, not SOTA benchmark chase.
- **Novelty thin:** polar conditioning lineage (CDTE, S2SP, Sat2Density); **possible niche:** frozen DINOv2 + polar column decoder at 1.5M params — needs careful related-work.
- **Venue judgment (external):** CVPR main long shot; **workshop / IGARSS / arXiv** more realistic. **IGARSS 2027** deadline Jan 11, 2027 mentioned; avoid double-submit with CVPR.

---

## 9. What is DONE vs NEXT (no training)

### Done

- [x] Baseline + Phase 1 + Phase 2 Stage 1 ablations  
- [x] Stage 2 Pix2Pix + SPADE + SPADE+aerial training & checkpoint selection  
- [x] Final eval ladder E,C,D,B,A + bootstrap CIs  
- [x] Artifacts downloaded (`final_eval_results.tgz`)  
- [x] Pod eval complete — **terminate pod**

### Next (writing & packaging — CPU / human time)

1. **Paper framing** from **Config E** + ladder table + paired CIs; separate **oracle (D)** vs **E2E (E)** rows; never mix legacy FID 116 with clean-fid 141 without footnote.  
2. **Update `results_table.json`** with final_eval block + clean-fid columns.  
3. **Figures:** 24-sample folders per config; montages (`make_gan_showcase_dumps.py`); failure cases (layout errors → RGB errors).  
4. **Related work** pass (DINO polar layout, Sat2Density, CrossViewDiff critique of oracle layout).  
5. **Limitations paragraph:** pseudo-labels, D collapse, confounded ladder, no Sat2Density number in harness.  
6. **Optional:** HF Hub for ckpts; Gradio demo; single E2E inference script.  
7. **Optional (extra GPU):** Sat2Density eval in same harness — high setup, not required to close project.

---

## 10. Ask for Claude (explicit)

Given all of the above:

1. Propose **paper title, abstract sketch, and table structure** (main results + ablations + diagnostic).  
2. Decide **primary claim** (efficiency? E2E aerial synthesis? reproducible Stage-1/2 ladder?).  
3. Map **Config E** numbers to honest **comparison language** vs literature (what we can and cannot claim).  
4. Outline **IGARSS 4-page** vs **workshop** structure using existing figures/logs.  
5. List **reviewer objections** and **prebuttal** text from our null ablations and D-collapse logs.  
6. Prioritize **1-week writing checklist** (no new experiments unless absolutely necessary).

---

*Generated for handoff Oct 2025. Source of truth for numbers: `experiments/final_eval/*_metrics.json`, `bootstrap_ci.json`, `experiments/results_table.json`.*
