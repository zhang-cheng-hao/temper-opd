#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
FLASHOPD_DIR="${FLASHOPD_DIR:-$ROOT_DIR/baselines/flash-opd}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/baselines/flashopd_qwen25_05b_smoke.yaml}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/flashopd_qwen25_05b_smoke_$RUN_ID}"

if [ ! -d "$FLASHOPD_DIR/flashopd" ]; then
  echo "Missing FlashOPD source: $FLASHOPD_DIR/flashopd" >&2
  echo "Download or restore baselines/flash-opd before running this script." >&2
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "GPU=$GPU"
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"

export PYTHONPATH="$FLASHOPD_DIR${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m flashopd.cli \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
