#!/usr/bin/env bash
# Regenerate zeroD, ml_inputs, jax_arrays, and split_indices under data/.../<RUN_CONFIG>/.
# Requires the same physics flags as the suffix (e.g. stenosis_off + symmetric for
# stenosis_off_symmetric_gen_loss, symmetric_gen_loss (--symmetric-loss), or symmetric_penalty_off_gen_loss).
#
# Usage:
#   SET_NAME=VMR_pulmo RUN_CONFIG=stenosis_off_symmetric_gen_loss \
#     ./util/zerod_calibration/regenerate_run_config_data.sh
#
# For RUN_CONFIG=symmetric_penalty_off_gen_loss, replace --stenosis-off with --penalty-off
# in CMD_BATCH below (keep --symmetric-loss and --run-config).
#
# Optional: pass geometry names as extra args (forwarded to both scripts), e.g.:
#   .../regenerate_run_config_data.sh 0063_1001 0155_0001

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SET_NAME="${SET_NAME:-VMR}"
RUN_CONFIG="${RUN_CONFIG:-stenosis_off_symmetric_gen_loss}"
EXTRA_GEO=("$@")

echo "==> Batch zeroD / calibration (set=${SET_NAME}, run-config=${RUN_CONFIG})"
CMD_BATCH=(python util/zerod_calibration/batch_generate_zerod_inputs_vmr.py
  --set-name "$SET_NAME"
  --stenosis-off
  --symmetric-loss
  --run-config "$RUN_CONFIG")
if ((${#EXTRA_GEO[@]})); then
  CMD_BATCH+=(--geometries "${EXTRA_GEO[@]}")
fi
"${CMD_BATCH[@]}"

echo "==> ML inputs + jax + splits (geometry-variant all)"
CMD_DP=(python util/data_processing/run_data_processing.py
  --set-name "$SET_NAME"
  --geometry-variant all
  --run-config "$RUN_CONFIG")
if ((${#EXTRA_GEO[@]})); then
  CMD_DP+=(--geometries "${EXTRA_GEO[@]}")
fi
"${CMD_DP[@]}"

echo "Done. Data lives under data/zeroD/${SET_NAME}/${RUN_CONFIG}/, data/ml_inputs/..., data/jax_arrays/..., data/split_indices/..."
