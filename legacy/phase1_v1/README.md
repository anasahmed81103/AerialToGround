# Phase 1 v1 — CrossNet + Pix2Pix (legacy quickstart)

Run from **this directory** so imports resolve:

```bash
cd legacy/phase1_v1
# From this directory (so local crossnet_* imports resolve):
cd legacy/phase1_v1
python inference.py --source ../../../demo/sample_aerial_road.jpg
python train_crossnet.py --csv ../../../cvpr_train_v2.csv
```

Checkpoints: see `experiments/phase1/checkpoints/` or `outputs/ckpts_pt_v2/` (local, gitignored).

For the **current** system use repo-root `pipeline.py` instead.
