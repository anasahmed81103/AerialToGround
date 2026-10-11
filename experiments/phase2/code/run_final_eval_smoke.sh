#!/usr/bin/env bash
# Smoke: config E, 64 val samples (no full 8884 run).
set -euo pipefail
cd "$(dirname "$0")/../../.."
pip install -q clean-fid lpips scikit-image
python final_eval.py --smoke --batch_size 4
