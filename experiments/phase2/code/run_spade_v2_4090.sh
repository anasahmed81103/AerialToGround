#!/usr/bin/env bash
# SPADE on 24 GB GPUs (RTX 4090 / A5000): large batch, full data, ~3–4 h wall time.
# 22 h in train_gan_pix2pix.py banner is a P1000 guess — ignore it on 4090.
#
# Usage (RunPod):
#   chmod +x experiments/phase2/code/run_spade_v2_4090.sh
#   bash experiments/phase2/code/run_spade_v2_4090.sh
#
# Hard cap ~2.5 h (slightly less training): EPOCHS=10 bash ...
# Resume: pass RESUME_G and RESUME_D to train_gan_pix2pix (SPADE G only from spade ckpts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-experiments/phase2/runs/spade_v2}"
mkdir -p logs "$OUT/ckpts" "$OUT/dump"

EXTRA=()
if [[ -n "${RESUME_G:-}" ]]; then EXTRA+=(--resume_g "$RESUME_G"); fi
if [[ -n "${RESUME_D:-}" ]]; then EXTRA+=(--resume_d "$RESUME_D"); fi

python -u train_gan_pix2pix.py \
  --generator spade \
  --train_csv cvpr_train_v2.csv \
  --random_init_d \
  --g_steps 3 \
  --lambda_l1 "${LAMBDA_L1:-10}" \
  --lambda_perceptual "${LAMBDA_VGG:-10}" \
  --epochs "${EPOCHS:-12}" \
  --batch_size "${BATCH:-32}" \
  --num_workers "${WORKERS:-8}" \
  --save_every "${SAVE_EVERY:-1000}" \
  --d_diag_max_steps "${D_DIAG:-300}" \
  --ckpt_dir "$OUT/ckpts" \
  --dump_dir "$OUT/dump" \
  "${EXTRA[@]}" \
  2>&1 | tee -a logs/spade_v2.txt
