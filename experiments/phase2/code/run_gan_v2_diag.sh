#!/usr/bin/env bash
# Discriminator-collapse diagnostic (~2 epochs). Does not touch gan_v2_retrain/.
#
# If D stays ~0 with warm D + fresh_d_optimizer, use random_init_d + g_steps=3:
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p logs experiments/phase2/runs/gan_v2_diag/ckpts experiments/phase2/runs/gan_v2_diag/dump

python -u train_gan_pix2pix.py \
  --train_csv cvpr_train_v2.csv \
  --resume_g outputs/ckpts_gan/G_step0197766.pt \
  --random_init_d \
  --g_steps 3 \
  --epochs 2 \
  --batch_size 16 \
  --num_workers 4 \
  --ckpt_dir experiments/phase2/runs/gan_v2_diag/ckpts \
  --dump_dir experiments/phase2/runs/gan_v2_diag/dump \
  2>&1 | tee logs/gan_v2_diag.txt
