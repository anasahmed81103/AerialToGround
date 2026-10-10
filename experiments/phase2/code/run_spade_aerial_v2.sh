#!/usr/bin/env bash
# SPADE + warped aerial conditioning (7-ch G input). Same recipe as spade_v2 except --aux_aerial.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-experiments/phase2/runs/spade_aerial_v2}"
mkdir -p logs "$OUT/ckpts" "$OUT/dump"

python -u train_gan_pix2pix.py \
  --generator spade \
  --aux_aerial \
  --train_csv cvpr_train_v2.csv \
  --random_init_d \
  --g_steps 3 \
  --lambda_l1 "${LAMBDA_L1:-10}" \
  --lambda_perceptual "${LAMBDA_VGG:-10}" \
  --epochs "${EPOCHS:-15}" \
  --batch_size "${BATCH:-16}" \
  --num_workers "${WORKERS:-8}" \
  --d_diag_max_steps "${D_DIAG:-300}" \
  --keep_ckpts "${KEEP_CKPTS:-5}" \
  --ckpt_dir "$OUT/ckpts" \
  --dump_dir "$OUT/dump" \
  2>&1 | tee logs/spade_aerial_v2.txt
