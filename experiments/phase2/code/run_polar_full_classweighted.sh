#!/usr/bin/env bash
# Same as polar_full + inverse-frequency class weights. RunPod / Linux.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -u train_polar.py \
  --name polar_full_classweighted \
  --cache cache/dinov2s_reg_336 \
  --out_h 32 --out_w 160 \
  --batch_size 16 --epochs 8 --num_workers 4 --amp \
  --class_weights auto \
  --save_n 8
python experiments/phase2/code/append_polar_full_classweighted_results.py
