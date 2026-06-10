#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${1:-$ROOT_DIR/configs/reproduction/baseline_paths.env}"
PATCH_FILE="$ROOT_DIR/patches/thunlp-opd-env-overrides.patch"
DRY_RUN="${DRY_RUN:-1}"
FAIL_ON_TOKENIZER_MISMATCH="${FAIL_ON_TOKENIZER_MISMATCH:-0}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  echo "Copy configs/reproduction/baseline_paths.env.template and fill paths first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

THUNLP_OPD_DIR="${THUNLP_OPD_DIR:-$ROOT_DIR/baselines/thunlp-opd}"
PROJECT_PATH="${THUNLP_PROJECT_PATH:-${REPRO_OUTPUT_ROOT:-$ROOT_DIR/runs/repro}/thunlp_opd/checkpoint}"

required=(
  THUNLP_ACTOR_MODEL_PATH
  THUNLP_REWARD_MODEL_PATH
  THUNLP_TRAIN_DATASET
  THUNLP_TEST_DATA_DIR
)
for var in "${required[@]}"; do
  if [ -z "${!var:-}" ]; then
    echo "Missing required env var: $var" >&2
    exit 1
  fi
done

echo "==> THUNLP OPD dir: $THUNLP_OPD_DIR"
cd "$THUNLP_OPD_DIR"

if git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "==> Env override patch already applied"
else
  echo "==> Applying env override patch"
  git apply "$PATCH_FILE"
fi

tokenizer_args=(
  "$ROOT_DIR/scripts/repro/check_tokenizer_stop_tokens.py"
  --student "$THUNLP_ACTOR_MODEL_PATH"
  --teacher "$THUNLP_REWARD_MODEL_PATH"
)
if [ "$FAIL_ON_TOKENIZER_MISMATCH" = "1" ]; then
  tokenizer_args+=(--fail-on-mismatch)
fi
python "${tokenizer_args[@]}"

export ACTOR_MODEL_PATH="$THUNLP_ACTOR_MODEL_PATH"
export REWARD_MODEL_PATH="$THUNLP_REWARD_MODEL_PATH"
export TRAIN_DATASET="$THUNLP_TRAIN_DATASET"
export TEST_DATA_DIR="$THUNLP_TEST_DATA_DIR"
export PROJECT_PATH
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-1}"

echo "==> Resolved THUNLP OPD run"
echo "ACTOR_MODEL_PATH=$ACTOR_MODEL_PATH"
echo "REWARD_MODEL_PATH=$REWARD_MODEL_PATH"
echo "TRAIN_DATASET=$TRAIN_DATASET"
echo "TEST_DATA_DIR=$TEST_DATA_DIR"
echo "PROJECT_PATH=$PROJECT_PATH"
echo "N_GPUS_PER_NODE=$N_GPUS_PER_NODE"
echo "NNODES=$NNODES"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1, not launching training. Set DRY_RUN=0 to run bash on_policy_distillation.sh"
  exit 0
fi

bash on_policy_distillation.sh
