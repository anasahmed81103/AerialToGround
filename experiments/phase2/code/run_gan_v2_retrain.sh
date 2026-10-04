#!/usr/bin/env bash
# Fair v2 GAN retrain: warm-start from v1 checkpoints, GT v2 layouts for supervision.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

CKPT_G="${CKPT_G:-outputs/ckpts_gan/G_step0197766.pt}"
CKPT_D="${CKPT_D:-outputs/ckpts_gan/D_step0197766.pt}"
OUT="${OUT:-experiments/phase2/runs/gan_v2_retrain}"
mkdir -p "$OUT/ckpts" "$OUT/dump"

python -u train_gan_pix2pix.py \
  --train_csv cvpr_train_v2.csv \
  --batch_size "${BATCH:-4}" \
  --epochs "${EPOCHS:-10}" \
  --num_workers "${WORKERS:-4}" \
  --ckpt_dir "$OUT/ckpts" \
  --dump_dir "$OUT/dump" \
  --resume_g "$CKPT_G" \
  --resume_d "$CKPT_D" \
  "$@"

# Full val metrics (GT labels -> GAN), pick latest G checkpoint or set GAN_CKPT=
GAN_CKPT="${GAN_CKPT:-$(ls -t "$OUT/ckpts"/G_step*.pt | head -1)}"
python eval_gan_pix2pix.py \
  --gan_ckpt "$GAN_CKPT" \
  --val_csv cvpr_val_v2.csv \
  --batch_size 8 \
  --out_json "$OUT/val_metrics.json"
