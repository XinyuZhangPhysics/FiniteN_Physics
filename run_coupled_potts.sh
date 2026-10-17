#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
WORKERS=${WORKERS:-10}
OUT=${1:-"$ROOT/reproduced/coupled_potts_n5"}
mkdir -p "$(dirname "$OUT")"
"$PY" "$ROOT/coupled_potts_n5/source/n5_coupled_potts_mixed_closure_v5b.py" \
  --N 5 --j-grid 0.005:0.0244:55 --hessian-samples 15 \
  --audit-starts 128 --audit-seeds 3 --workers "$WORKERS" --seed 20261015 --out "$OUT"
