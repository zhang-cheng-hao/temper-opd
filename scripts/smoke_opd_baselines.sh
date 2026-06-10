#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/opd_baseline_smoke_$RUN_ID}"
RUN_FLASHOPD="${RUN_FLASHOPD:-1}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/smoke.log"
SUMMARY="$OUT_DIR/summary.tsv"
: > "$LOG"
printf "name\tstatus\tdetail\n" > "$SUMMARY"

record() {
  printf "%s\t%s\t%s\n" "$1" "$2" "$3" | tee -a "$SUMMARY"
}

run_logged() {
  local name="$1"
  shift
  echo "==> $name" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    record "$name" "PASS" "$*"
  else
    record "$name" "FAIL" "$*"
    return 1
  fi
}

cd "$ROOT_DIR"

run_logged "python_imports" "$PYTHON_BIN" -c "import torch, transformers, datasets, accelerate, yaml; print(torch.__version__, transformers.__version__)"
run_logged "flashopd_compile" "$PYTHON_BIN" -m compileall -q baselines/flash-opd/flashopd
run_logged "ta_opd_compile" "$PYTHON_BIN" -m compileall -q baselines/ta-opd/ta_opd baselines/ta-opd/tools
run_logged "opsd_compile" "$PYTHON_BIN" -m compileall -q baselines/opsd
run_logged "tinker_compile" "$PYTHON_BIN" -m compileall -q baselines/tinker-cookbook/tinker_cookbook

if [ -d baselines/hybrid-policy-distillation/verl/recipe/HPD ]; then
  run_logged "hpd_recipe_compile" "$PYTHON_BIN" -m compileall -q baselines/hybrid-policy-distillation/verl/recipe/HPD
else
  record "hpd_recipe_compile" "SKIP" "missing baselines/hybrid-policy-distillation/verl/recipe/HPD"
fi

run_logged "thunlp_opd_shell" bash -n baselines/thunlp-opd/on_policy_distillation.sh
run_logged "opsd_shell" bash -n baselines/opsd/scripts/run_opsd_1b.sh

if [ -f baselines/hybrid-policy-distillation/verl/recipe/HPD/run_hpd.sh ]; then
  run_logged "hpd_shell" bash -n baselines/hybrid-policy-distillation/verl/recipe/HPD/run_hpd.sh
else
  record "hpd_shell" "SKIP" "missing HPD run_hpd.sh"
fi

if [ -f baselines/tropd/README.md ] && grep -q "Training and evaluation code" baselines/tropd/README.md; then
  record "tropd_readme" "PASS" "README present; upstream marks training/eval code as TODO"
else
  record "tropd_readme" "FAIL" "README missing or unexpected"
fi

if [ "$RUN_FLASHOPD" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" GPU="$GPU" RUN_ID="baseline_smoke_$RUN_ID" \
    "$ROOT_DIR/scripts/run_flashopd_smoke.sh" >> "$LOG" 2>&1
  record "flashopd_12step" "PASS" "actual 12-step run via scripts/run_flashopd_smoke.sh"
else
  record "flashopd_12step" "SKIP" "RUN_FLASHOPD=$RUN_FLASHOPD"
fi

echo "Smoke summary: $SUMMARY"
