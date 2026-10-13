#!/usr/bin/env bash
# End-to-end NicheRFT quick-start on a small synthetic dataset.
#
#   representation learning + retrieval cache  ->  conditional flow matching
#
# Usage:
#   bash examples/run_example.sh            # uses python from PATH / $PYTHON
#   PYTHON=python3 bash examples/run_example.sh
#
# The example dataset is synthetic (see make_example_data.py); no real data and
# no identifying information are required.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python}"

cd "$ROOT"

DATA="$HERE/data/example.h5ad"
CACHE="outputs/example_cache"
FLOW="outputs/example_flow"

# Start from a clean slate: both stages refuse to overwrite an existing cache.
rm -rf "$CACHE" "$FLOW"

# 1) Generate the synthetic dataset if it is not already present.
if [ ! -f "$DATA" ]; then
  echo "[1/3] generating synthetic example data -> $DATA"
  "$PY" examples/make_example_data.py --output_file "$DATA"
else
  echo "[1/3] found existing example data at $DATA"
fi

# 2) Stage I/II: niche-aware RNA representation + split-aware retrieval cache.
#    Small data needs a small batch (the default 1024 under-trains tiny inputs).
echo "[2/3] caching representation + retrieval -> $CACHE"
"$PY" -u experiments/cache_stage12.py \
    --dataset example \
    --data_path "$DATA" \
    --output_dir "$CACHE" \
    --k 128 --batch_size 32 \
    --stage1_epochs 20 \
    --seed 42 --split_seed 42 \
    --device cuda

# 3) Stage III: fit the DiT conditional flow field and evaluate.
#    The default is fully inductive (no query-slice statistics). Add
#    --slice_relative_center to calibrate the source on each slice's own prior mean.
echo "[3/3] training conditional flow matching -> $FLOW"
"$PY" -u experiments/finalize_stage3.py \
    --cache_dir "$CACHE" \
    --output_dir "$FLOW" \
    --seed 42 --k 128 \
    --prediction_nfe 8 --epochs 20

echo
echo "Done. Predictions and metrics are under $FLOW/fold0/seed42/"
