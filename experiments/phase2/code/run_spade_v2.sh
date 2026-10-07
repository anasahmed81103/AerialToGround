#!/usr/bin/env bash
# Stage 2 SPADE: random G + random D, g_steps=3, lower L1 + VGG perceptual.
# Beat Pix2Pix baseline FID 157.6 (gan_v2_random_d @ G_step0231066).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-experiments/phase2/runs/spade_v2}"
mkdir -p logs "$OUT/ckpts" "$OUT/dump"

python -u train_gan_pix2pix.py \
  --generator spade \
  --train_csv cvpr_train_v2.csv \
  --random_init_d \
  --g_steps 3 \
  --lambda_l1 "${LAMBDA_L1:-10}" \
  --lambda_perceptual "${LAMBDA_VGG:-10}" \
  --epochs "${EPOCHS:-15}" \
  --batch_size "${BATCH:-8}" \
  --num_workers "${WORKERS:-4}" \
  --d_diag_max_steps "${D_DIAG:-300}" \
  --ckpt_dir "$OUT/ckpts" \
  --dump_dir "$OUT/dump" \
  2>&1 | tee logs/spade_v2.txt
