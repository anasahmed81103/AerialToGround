# Experiments — research showcase archive

This folder freezes **baseline** and **Phase 1** so later work cannot overwrite them.

| | Baseline (original) | Phase 1 (current improvement) |
|---|---|---|
| Code snapshot | `baseline/code/` | `phase1/code/` |
| Stage 1 checkpoints | `baseline/checkpoints/` | `phase1/checkpoints/` |
| Training dumps | `baseline/dumps/` | `phase1/dumps/` |
| Numbers | `baseline/results/` | `phase1/results/` |
| Still images | `baseline/showcase/` | `phase1/showcase/` |
| Labels | `cvpr_train.csv` + original annotations | `cvpr_train_v2.csv` + `cvpr_subset/labels_v2/` |
| Full number sheet | [`results_table.json`](results_table.json) | same |

Old paths still work. `outputs/ckpts_pt`, `outputs/ckpts_pt_v2`, `outputs/dump_pt`, `outputs/dump_pt_v2`, and `outputs/ckpts_gan` are junctions to the folders above.

## Headline numbers (do not lose)

**Baseline CrossNet** `step 88830`, **old labels**, full val **8884**:

- mIoU full **0.4535** · native 0.4117 · acc 0.7046
- IoU: sky 0.679 · **veg 0.087** · road 0.515 · building 0.533

**Fair test, same new labels, n=1024:**

| Model | mIoU | sky | veg | road | building |
|---|---|---|---|---|---|
| Baseline 88830 | 0.299 | 0.639 | 0.030 | 0.432 | 0.096 |
| Phase 1 epoch 20 | **0.403** | 0.633 | **0.535** | 0.361 | 0.083 |

**Fair test, same new labels, full 8,884** (Phase 2 selected on a minival from train):

| Model | mIoU | sky | veg | road | building |
|---|---|---|---|---|---|
| Baseline 88830 | 0.299 | 0.638 | 0.026 | 0.438 | 0.095 |
| Phase 1 best | 0.402 | 0.633 | 0.533 | 0.362 | 0.080 |
| Phase 2 `polar_full` | **0.514** | **0.689** | **0.661** | **0.494** | **0.209** |

Phase 2 changes, ablations and their impact: [`phase2/README.md`](phase2/README.md).

Phase 1 train curve (all 20 epochs): `phase1/results/val_curve_20epochs_n1024.csv`.

## How to re-run without mixing things

```bash
# Baseline eval (old labels)
python experiments/phase1/code/eval_crossnet.py --ckpt experiments/baseline/checkpoints/crossnet_step0088830.pt --val_csv cvpr_val.csv

# Phase 1 eval (new labels)
python experiments/phase1/code/eval_crossnet.py --ckpt experiments/phase1/checkpoints/crossnet_best.pt --val_csv cvpr_val_v2.csv
```

Start the **next** method in a new folder, e.g. `experiments/phase2/`. Do not write into `baseline/` or `phase1/`.
