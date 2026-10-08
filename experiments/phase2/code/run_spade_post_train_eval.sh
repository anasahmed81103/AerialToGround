#!/usr/bin/env bash
# After SPADE training finishes: eval final + last 2 G checkpoints, write JSON metrics.
# Does NOT stop the RunPod — stop the pod yourself in the UI to save money.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RUN="${OUT:-experiments/phase2/runs/spade_v2}"
CKPT_DIR="$RUN/ckpts"
VAL_CSV="${VAL_CSV:-cvpr_val_v2.csv}"
BATCH="${EVAL_BATCH:-8}"

mapfile -t G_CKPTS < <(ls -1 "$CKPT_DIR"/G_step*.pt 2>/dev/null | sort -V)
if [[ ${#G_CKPTS[@]} -eq 0 ]]; then
  echo "No G_step*.pt in $CKPT_DIR"
  exit 1
fi

# Final + up to two earlier checkpoints (same idea as Pix2Pix FID sweep)
N=${#G_CKPTS[@]}
PICK=("${G_CKPTS[$((N-1))]}")
[[ "$N" -ge 2 ]] && PICK+=("${G_CKPTS[$((N-2))]}")
[[ "$N" -ge 3 ]] && PICK+=("${G_CKPTS[$((N-3))]}")

echo "[*] Evaluating ${#PICK[@]} checkpoint(s)..."
for CKPT in "${PICK[@]}"; do
  STEP=$(basename "$CKPT" .pt | sed 's/G_step//')
  OUT_JSON="$RUN/val_metrics_${STEP}.json"
  echo "[*] $CKPT -> $OUT_JSON"
  python eval_gan_pix2pix.py --generator spade \
    --gan_ckpt "$CKPT" \
    --val_csv "$VAL_CSV" \
    --batch_size "$BATCH" \
    --out_json "$OUT_JSON"
done

BEST=$(RUN_DIR="$RUN" python - <<'PY'
import glob, json, os
run = os.environ["RUN_DIR"]
best = None
for p in glob.glob(os.path.join(run, "val_metrics_*.json")):
    with open(p) as f:
        m = json.load(f)
    fid = m.get("fid")
    if fid is None:
        continue
    if best is None or fid < best[0]:
        best = (fid, p, m.get("gan_ckpt"))
if best:
    print(f"best_fid={best[0]:.4f} json={best[1]} ckpt={best[2]}")
else:
    print("no_fid")
PY
)
echo "$BEST"

# Optional tarball of small artifacts (no .pt)
TAR="${RUN}/spade_v2_artifacts_$(date +%Y%m%d_%H%M%S).tgz"
tar czf "$TAR" \
  logs/spade_v2.txt \
  "$RUN"/val_metrics_*.json \
  "$RUN"/dump/*.jpg 2>/dev/null || true
echo "[*] Wrote $TAR (metrics + dumps + log; ckpts stay in $CKPT_DIR)"
echo "[*] Download ckpts + tarball, then STOP the pod in RunPod UI."
