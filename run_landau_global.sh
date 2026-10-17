#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
WORKERS=${WORKERS:-12}
OUT=${1:-"$ROOT/reproduced/landau_n5_global"}
mkdir -p "$OUT"
SRC="$ROOT/landau_n5_global/source/n5_publication_numerical_audit_v1.py"
for seed in 20261015 31415926 27182818; do
  "$PY" "$SRC" --mode all --preset publication --workers "$WORKERS" --seed "$seed" --outdir "$OUT/seed_$seed"
done
