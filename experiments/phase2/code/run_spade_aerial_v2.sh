#!/usr/bin/env bash
# SPADE + warped aerial conditioning. Disk-safe defaults (weights-only ckpts, no montage flood).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-experiments/phase2/runs/spade_aerial_v2}"
mkdir -p logs "$OUT/ckpts"

# Append log only — do not tee (duplicates every line on quota-sensitive pods).
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
  --keep_ckpts "${KEEP_CKPTS:-1}" \
  --save_every "${SAVE_EVERY:-4000}" \
  --vis_every 0 \
  --ckpt_dir "$OUT/ckpts" \
  --dump_dir "$OUT/dump" \
  >> "logs/spade_aerial_v2.txt" 2>&1
