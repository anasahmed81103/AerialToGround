#!/usr/bin/env bash
# Test AMP / D gradient underflow (1 epoch, FP32). Does not touch gan_v2_retrain/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p logs experiments/phase2/runs/gan_v2_diag2/ckpts experiments/phase2/runs/gan_v2_diag2/dump

python -u train_gan_pix2pix.py \
  --train_csv cvpr_train_v2.csv \
  --resume_g outputs/ckpts_gan/G_step0197766.pt \
  --resume_d outputs/ckpts_gan/D_step0197766.pt \
  --fresh_d_optimizer \
  --g_steps 1 \
  --no_amp \
  --epochs 1 \
  --batch_size 16 \
  --num_workers 4 \
  --ckpt_dir experiments/phase2/runs/gan_v2_diag2/ckpts \
  --dump_dir experiments/phase2/runs/gan_v2_diag2/dump \
  2>&1 | tee logs/gan_v2_diag2.txt
