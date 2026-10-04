#!/usr/bin/env bash
# 2-epoch diagnostic: fresh D optimizer + g_steps=1 (does not touch gan_v2_retrain/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p logs experiments/phase2/runs/gan_v2_diag/ckpts experiments/phase2/runs/gan_v2_diag/dump

python -u train_gan_pix2pix.py \
  --train_csv cvpr_train_v2.csv \
  --resume_g outputs/ckpts_gan/G_step0197766.pt \
  --resume_d outputs/ckpts_gan/D_step0197766.pt \
  --fresh_d_optimizer \
  --g_steps 1 \
  --epochs 2 \
  --batch_size 16 \
  --num_workers 4 \
  --ckpt_dir experiments/phase2/runs/gan_v2_diag/ckpts \
  --dump_dir experiments/phase2/runs/gan_v2_diag/dump \
  2>&1 | tee logs/gan_v2_diag.txt
