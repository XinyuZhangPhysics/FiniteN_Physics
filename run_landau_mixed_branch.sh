#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
WORKERS=${WORKERS:-12}
OUT=${1:-"$ROOT/reproduced/landau_n5_mixed_branch"}
mkdir -p "$OUT"
"$PY" "$ROOT/landau_n5_mixed_branch/source/n5_mixed_branch_continuation_v1.py" \
  --omega-seed 0.14 --seed-starts 2048 --workers "$WORKERS" \
  --step 0.001 --min-step 2e-6 --outdir "$OUT"
"$PY" "$ROOT/landau_n5_mixed_branch/analysis/analyze_spinodals.py" \
  --table "$OUT/mixed_branch.csv" --expected "$ROOT/landau_n5_mixed_branch/expected.json" \
  --output "$OUT/LINEAR_ENDPOINT_FITS.json"
