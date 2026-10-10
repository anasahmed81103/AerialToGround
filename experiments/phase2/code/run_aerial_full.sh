#!/usr/bin/env bash
# 1) Optional weight sanity on existing spade_v2 ckpts (fast)
# 2) SPADE + warped aerial training (same as run_spade_aerial_v2.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

CKA=experiments/phase2/runs/spade_v2/ckpts/G_step0031080.pt
CKB=experiments/phase2/runs/spade_v2/ckpts/G_step0033300.pt
if [ -f "$CKA" ] && [ -f "$CKB" ]; then
  echo "[*] check_eval_weights.py (spade_v2 ckpts on disk)..."
  python check_eval_weights.py --ckpt_a "$CKA" --ckpt_b "$CKB" || true
else
  echo "[*] skip weight check — spade_v2 ckpts not on this pod (download or skip)."
fi

exec bash experiments/phase2/code/run_spade_aerial_v2.sh
