#!/usr/bin/env bash
# Move pod-local untracked files that often block `git pull`, then pull.
# Does NOT delete ckpts/*.pt or gan_checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

BACKUP="${BACKUP:-pod_pull_backup_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BACKUP"

for f in logs/spade_v2.txt logs/spade_aerial_v2.txt logs/gan_v2_random_d.txt \
         logs/gan_v2_diag3.txt logs/gan_v2_retrain.txt; do
  [ -f "$f" ] && mv "$f" "$BACKUP/" 2>/dev/null || true
done

if [ -d experiments/phase2/runs/gan_v2_random_d/dump ]; then
  mkdir -p "$BACKUP/gan_v2_random_d_dump"
  mv experiments/phase2/runs/gan_v2_random_d/dump/*.jpg "$BACKUP/gan_v2_random_d_dump/" 2>/dev/null || true
fi

for f in experiments/phase2/runs/gan_v2_random_d/val_metrics_*.json \
         experiments/phase2/runs/spade_v2/val_metrics_*.json; do
  [ -f "$f" ] && mv "$f" "$BACKUP/" 2>/dev/null || true
done

git pull
echo "[*] Backup (if any): $BACKUP"
git status -sb
