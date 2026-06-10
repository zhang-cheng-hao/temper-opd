#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/flashopd_constant_sweep_$RUN_ID}"

ACTION_GRID=(
  "repair_like:0.7:0.90"
  "dwell_like:1.0:0.95"
  "mild_escape_like:1.2:0.97"
  "escape_like:1.4:0.98"
)

mkdir -p "$OUT_ROOT"

for row in "${ACTION_GRID[@]}"; do
  IFS=":" read -r action_name temperature top_p <<< "$row"
  echo "==> FlashOPD action=$action_name temperature=$temperature top_p=$top_p"
  OUTPUT_DIR="$OUT_ROOT/$action_name" \
    "$ROOT_DIR/scripts/run_flashopd_smoke.sh" \
      --rollout_temperature "$temperature" \
      --rollout_top_p "$top_p"
done

echo "Sweep complete: $OUT_ROOT"
