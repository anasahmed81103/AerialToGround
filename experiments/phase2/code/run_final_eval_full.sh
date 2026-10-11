#!/usr/bin/env bash
# Full publication eval: E,C,D,B,A on all cvpr_val_v2 rows (~8884 each).
set -euo pipefail
cd "$(dirname "$0")/../../.."
pip install -q clean-fid lpips scikit-image
python final_eval.py --batch_size 4 --config all
python bootstrap_ci.py --eval_dir experiments/final_eval
