#!/usr/bin/env bash
# Stage 2 Pix2Pix baseline: random-init D + g_steps=3, G warm-start from v1 (197766).
# Train 15 epochs on GT v2 layouts; eval separately with eval_gan_pix2pix.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-experiments/phase2/runs/gan_v2_random_d}"
mkdir -p logs "$OUT/ckpts" "$OUT/dump"

python -u train_gan_pix2pix.py \
  --train_csv cvpr_train_v2.csv \
  --resume_g outputs/ckpts_gan/G_step0197766.pt \
  --random_init_d \
  --g_steps 3 \
  --epochs "${EPOCHS:-15}" \
  --batch_size "${BATCH:-16}" \
  --num_workers "${WORKERS:-4}" \
  --d_diag_max_steps "${D_DIAG:-300}" \
  --ckpt_dir "$OUT/ckpts" \
  --dump_dir "$OUT/dump" \
  2>&1 | tee logs/gan_v2_random_d.txt
