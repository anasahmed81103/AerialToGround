#!/usr/bin/env bash
# List project artifacts on RunPod (or locally) — skips bulky dataset trees.
# Usage:
#   cd /workspace/AerialToGround
#   bash experiments/phase2/code/list_workspace_artifacts.sh
#   bash experiments/phase2/code/list_workspace_artifacts.sh --write-manifest
#
# Compare: copy pod manifest to local, then:
#   bash experiments/phase2/code/list_workspace_artifacts.sh --diff pod_manifest.txt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

WRITE_MANIFEST=0
DIFF_AGAINST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --write-manifest) WRITE_MANIFEST=1; shift ;;
    --diff) DIFF_AGAINST="${2:-}"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Paths we never recurse into (images / caches / git)
PRUNE=(
  -path './.git/*' -o
  -path './cvpr_subset/*' -o
  -path './**/node_modules/*' -o
  -path './**/__pycache__/*' -o
  -path './**/.ipynb_checkpoints/*'
)

should_skip() {
  local rel="$1"
  case "$rel" in
    cvpr_subset/*|*/node_modules/*|*/__pycache__/*|*/.ipynb_checkpoints/*) return 0 ;;
  esac
  return 1
}

list_tree() {
  echo "======================================================================"
  echo "  Workspace artifact inventory"
  echo "  root: $ROOT"
  echo "  date: $(date -Iseconds 2>/dev/null || date)"
  echo "======================================================================"
  echo

  echo "--- Disk (workspace mount) ---"
  df -h . 2>/dev/null || df -h "$ROOT"
  echo

  echo "--- Top-level project size (excl. cvpr_subset) ---"
  du -sh . 2>/dev/null
  du -sh cvpr_subset 2>/dev/null || echo "cvpr_subset: (not present)"
  echo

  echo "--- experiments/phase2/runs (per run) ---"
  if [ -d experiments/phase2/runs ]; then
    du -sh experiments/phase2/runs/* 2>/dev/null | sort -h || true
  else
    echo "(missing experiments/phase2/runs)"
  fi
  echo

  echo "--- Checkpoints (*.pt) ---"
  find experiments outputs experiments/phase2/gan_checkpoints -name '*.pt' -type f 2>/dev/null \
    | while read -r f; do ls -lh "$f"; done | sort -k9 || echo "(none)"
  echo

  echo "--- Eval metrics (*.json under phase2 runs) ---"
  find experiments/phase2/runs -name 'val_metrics*.json' -o -name 'results.json' 2>/dev/null \
    | sort | while read -r f; do
      [ -f "$f" ] && ls -lh "$f"
    done || echo "(none)"
  echo

  echo "--- Training logs (logs/*.txt) ---"
  ls -lh logs/*.txt 2>/dev/null || echo "(none)"
  echo

  echo "--- Dump folders (jpg count + size) ---"
  find experiments/phase2/runs -type d -name dump 2>/dev/null | while read -r d; do
    n=$(find "$d" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
    echo "$d  jpg=$n  $(du -sh "$d" 2>/dev/null | cut -f1)"
  done
  echo "  (note: spade_aerial with vis_every=0 → empty dump; old typo path was runs/spa/)"
  if [ -d experiments/phase2/runs/spa ]; then
    echo "  FOUND legacy dump: experiments/phase2/runs/spa $(du -sh experiments/phase2/runs/spa)"
  fi
  echo

  echo "--- Stage-1 polar checkpoints ---"
  find experiments/phase2/runs -maxdepth 2 -name 'best.pt' -o -name 'last.pt' 2>/dev/null \
    | while read -r f; do [ -f "$f" ] && ls -lh "$f"; done
  echo

  echo "--- Other outputs/ ckpts ---"
  find outputs -name '*.pt' -type f 2>/dev/null | head -20
  echo

  echo "--- CSVs (paths only, not listing image files) ---"
  ls -lh cvpr_*.csv 2>/dev/null || true
  echo
}

write_manifest() {
  local out="${1:-experiments/phase2/runs/_pod_manifest.txt}"
  echo "[*] Writing manifest: $out"
  {
    echo "# manifest root=$ROOT date=$(date -Iseconds 2>/dev/null || date)"
    find experiments logs outputs \
      \( "${PRUNE[@]}" \) -prune -o \
      -type f \( \
        -name '*.pt' -o -name '*.json' -o -name '*.txt' -o -name '*.csv' -o \
        -name '*.jpg' -o -name '*.png' -o -name '*.tgz' -o -name '*.tar.gz' \
      \) -print 2>/dev/null \
    | while read -r f; do
      rel="./${f#./}"
      should_skip "$rel" && continue
      # skip huge csv-adjacent image dirs if any slipped through
      case "$rel" in
        ./cvpr_subset/*) continue ;;
      esac
      if [ -f "$f" ]; then
        sz=$(stat -c '%s' "$f" 2>/dev/null || stat -f '%z' "$f" 2>/dev/null)
        echo "${sz}	${rel}"
      fi
    done | sort -t$'\t' -k2
  } > "$out"
  wc -l "$out"
  ls -lh "$out"
}

diff_manifest() {
  local pod_file="$1"
  local local_file="experiments/phase2/runs/_local_manifest.txt"
  if [ ! -f "$pod_file" ]; then
    echo "Missing pod manifest: $pod_file"
    exit 1
  fi
  write_manifest "$local_file"
  echo
  echo "=== On POD but missing locally (or size mismatch) ==="
  python3 - <<PY
import sys
pod, loc = "$pod_file", "$local_file"

def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sz, path = line.split("\t", 1)
            d[path] = int(sz)
    return d

P, L = load(pod), load(loc)
only_pod = sorted(set(P) - set(L))
only_loc = sorted(set(L) - set(P))
mismatch = sorted(p for p in set(P) & set(L) if P[p] != L[p])

for p in only_pod:
    print(f"MISSING_LOCAL  {P[p]:>12}  {p}")
for p in only_loc:
    if p.startswith("./experiments/") or p.startswith("./logs/"):
        print(f"EXTRA_LOCAL    {L[p]:>12}  {p}")
for p in mismatch:
    print(f"SIZE_DIFF      pod={P[p]} local={L[p]}  {p}")

if not only_pod and not mismatch:
    print("(all pod manifest files present locally with same size)")
PY
}

list_tree

if [ "$WRITE_MANIFEST" -eq 1 ]; then
  write_manifest "experiments/phase2/runs/_pod_manifest.txt"
  echo "[*] scp this file to your PC, then run locally:"
  echo "    bash experiments/phase2/code/list_workspace_artifacts.sh --diff experiments/phase2/runs/_pod_manifest.txt"
fi

if [ -n "$DIFF_AGAINST" ]; then
  diff_manifest "$DIFF_AGAINST"
fi
