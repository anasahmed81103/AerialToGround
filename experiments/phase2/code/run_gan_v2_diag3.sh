#!/usr/bin/env bash
# diag3: random-init D + g_steps=3 + FP32; D_diag through 2 epochs (~4440 steps).
# Does not touch gan_v2_retrain/, gan_v2_diag/, or gan_v2_diag2/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p logs experiments/phase2/runs/gan_v2_diag3/ckpts experiments/phase2/runs/gan_v2_diag3/dump

python -u train_gan_pix2pix.py \
  --train_csv cvpr_train_v2.csv \
  --resume_g outputs/ckpts_gan/G_step0197766.pt \
  --random_init_d \
  --g_steps 3 \
  --no_amp \
  --epochs 2 \
  --batch_size 16 \
  --num_workers 4 \
  --d_diag_max_steps 4500 \
  --ckpt_dir experiments/phase2/runs/gan_v2_diag3/ckpts \
  --dump_dir experiments/phase2/runs/gan_v2_diag3/dump \
  2>&1 | tee logs/gan_v2_diag3.txt
