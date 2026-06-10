#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${1:-$ROOT_DIR/configs/reproduction/baseline_paths.env}"
PATCH_FILE="$ROOT_DIR/patches/thunlp-opd-env-overrides.patch"
DRY_RUN="${DRY_RUN:-1}"
FAIL_ON_TOKENIZER_MISMATCH="${FAIL_ON_TOKENIZER_MISMATCH:-0}"
STRICT_ENV_CHECK="${STRICT_ENV_CHECK:-0}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  echo "Copy configs/reproduction/baseline_paths.env.template and fill paths first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

THUNLP_OPD_DIR="${THUNLP_OPD_DIR:-$ROOT_DIR/baselines/thunlp-opd}"
PROJECT_PATH="${THUNLP_PROJECT_PATH:-${REPRO_OUTPUT_ROOT:-$ROOT_DIR/runs/repro}/thunlp_opd/checkpoint}"

if [ -n "${THUNLP_CONDA_PREFIX:-}" ] && [ "${CONDA_PREFIX:-}" != "$THUNLP_CONDA_PREFIX" ]; then
  echo "WARN: current CONDA_PREFIX='${CONDA_PREFIX:-unset}', expected THUNLP_CONDA_PREFIX='$THUNLP_CONDA_PREFIX'" >&2
fi

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

env_check_args=("$ROOT_DIR/scripts/repro/check_thunlp_env.py" --min-gpus "${N_GPUS_PER_NODE:-8}")
if [ "$STRICT_ENV_CHECK" = "1" ]; then
  env_check_args+=(--fail-on-missing)
fi
python "${env_check_args[@]}"

tokenizer_args=(
  "$ROOT_DIR/scripts/repro/check_tokenizer_stop_tokens.py"
  --student "$THUNLP_ACTOR_MODEL_PATH"
  --teacher "$THUNLP_REWARD_MODEL_PATH"
)
if [ "$FAIL_ON_TOKENIZER_MISMATCH" = "1" ]; then
  tokenizer_args+=(--fail-on-mismatch)
fi
python "${tokenizer_args[@]}"

data_check_args=(
  "$ROOT_DIR/scripts/repro/check_thunlp_opd_data.py"
  --train-file "$THUNLP_TRAIN_DATASET"
  --test-data-dir "$THUNLP_TEST_DATA_DIR"
  --tokenizer "$THUNLP_ACTOR_MODEL_PATH"
  --max-prompt-length "${MAX_PROMPT_LENGTH:-1024}"
  --token-length-sample-rows "${TOKEN_LENGTH_SAMPLE_ROWS:-1024}"
  --sample-rows 1
  --fail-on-warning
)
if [ -n "${THUNLP_TEST_DATASET:-}" ]; then
  data_check_args+=(--val-files "$THUNLP_TEST_DATASET")
fi
if [ "${FAIL_ON_OVERLONG_PROMPT:-0}" = "1" ]; then
  data_check_args+=(--fail-on-overlong)
fi
python "${data_check_args[@]}"

export ACTOR_MODEL_PATH="$THUNLP_ACTOR_MODEL_PATH"
export REWARD_MODEL_PATH="$THUNLP_REWARD_MODEL_PATH"
export TRAIN_DATASET="$THUNLP_TRAIN_DATASET"
export TEST_DATA_DIR="$THUNLP_TEST_DATA_DIR"
if [ -n "${THUNLP_TEST_DATASET:-}" ]; then
  export TEST_DATASET="$THUNLP_TEST_DATASET"
fi
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
