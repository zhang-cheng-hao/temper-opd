#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${1:-$ROOT_DIR/configs/reproduction/baseline_paths.env}"
PATCH_FILE="$ROOT_DIR/patches/thunlp-opd-env-overrides.patch"
Y_OPD_PATCH_FILE="$ROOT_DIR/patches/thunlp-opd-y-opd.patch"
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

prepend_env_path() {
  local var_name="$1"
  local path_value="$2"
  local current_value="${!var_name:-}"
  if [ -z "$path_value" ]; then
    return
  fi
  case ":$current_value:" in
    *":$path_value:"*) ;;
    *) export "$var_name=$path_value${current_value:+:$current_value}" ;;
  esac
}

python_probe="python"
if [ -n "${THUNLP_CONDA_PREFIX:-}" ] && [ -x "$THUNLP_CONDA_PREFIX/bin/python" ]; then
  python_probe="$THUNLP_CONDA_PREFIX/bin/python"
fi
cuda_compat_dir="${THUNLP_CUDA_COMPAT_DIR:-}"
if [ -z "$cuda_compat_dir" ] && [ -n "${THUNLP_CONDA_PREFIX:-}" ]; then
  default_cuda_compat="$THUNLP_CONDA_PREFIX/cuda-compat-12-8/usr/local/cuda-12.8/compat"
  if [ -f "$default_cuda_compat/libcuda.so.1" ]; then
    cuda_compat_dir="$default_cuda_compat"
  fi
fi
if [ -n "$cuda_compat_dir" ]; then
  if [ ! -f "$cuda_compat_dir/libcuda.so.1" ]; then
    echo "THUNLP_CUDA_COMPAT_DIR is set but libcuda.so.1 is missing: $cuda_compat_dir" >&2
    exit 1
  fi
  prepend_env_path LD_LIBRARY_PATH "$cuda_compat_dir"
  echo "==> Added CUDA forward-compat driver path: $cuda_compat_dir"
fi
if [ -n "${THUNLP_CONDA_PREFIX:-}" ] && [ -d "$THUNLP_CONDA_PREFIX/lib" ]; then
  prepend_env_path LIBRARY_PATH "$THUNLP_CONDA_PREFIX/lib"
  prepend_env_path LD_LIBRARY_PATH "$THUNLP_CONDA_PREFIX/lib"
fi
curand_paths="$("$python_probe" - <<'PY' 2>/dev/null || true
from pathlib import Path

try:
    import nvidia.curand
except Exception:
    raise SystemExit(0)

root = Path(nvidia.curand.__file__).resolve().parent
include = root / "include"
lib = root / "lib"
print(include if (include / "curand.h").exists() else "")
print(lib if any(lib.glob("libcurand*")) else "")
PY
)"
curand_include="$(printf '%s\n' "$curand_paths" | sed -n '1p')"
curand_lib="$(printf '%s\n' "$curand_paths" | sed -n '2p')"
if [ -n "$curand_include" ]; then
  prepend_env_path CPATH "$curand_include"
  prepend_env_path C_INCLUDE_PATH "$curand_include"
  prepend_env_path CPLUS_INCLUDE_PATH "$curand_include"
  echo "==> Added CUDA curand include path for JIT builds: $curand_include"
fi
if [ -n "$curand_lib" ]; then
  prepend_env_path LIBRARY_PATH "$curand_lib"
  prepend_env_path LD_LIBRARY_PATH "$curand_lib"
  echo "==> Added CUDA curand library path for JIT builds: $curand_lib"
fi
cuda_runtime_paths="$("$python_probe" - <<'PY' 2>/dev/null || true
from pathlib import Path

try:
    import nvidia.cuda_runtime
except Exception:
    raise SystemExit(0)

root = Path(nvidia.cuda_runtime.__file__).resolve().parent
include = root / "include"
lib = root / "lib"
print(include if (include / "cuda_runtime.h").exists() else "")
print(lib if any(lib.glob("libcudart*")) else "")
PY
)"
cuda_runtime_include="$(printf '%s\n' "$cuda_runtime_paths" | sed -n '1p')"
cuda_runtime_lib="$(printf '%s\n' "$cuda_runtime_paths" | sed -n '2p')"
if [ -n "$cuda_runtime_include" ]; then
  prepend_env_path CPATH "$cuda_runtime_include"
  prepend_env_path C_INCLUDE_PATH "$cuda_runtime_include"
  prepend_env_path CPLUS_INCLUDE_PATH "$cuda_runtime_include"
  echo "==> Added CUDA runtime include path for JIT builds: $cuda_runtime_include"
fi
if [ -n "$cuda_runtime_lib" ]; then
  prepend_env_path LIBRARY_PATH "$cuda_runtime_lib"
  prepend_env_path LD_LIBRARY_PATH "$cuda_runtime_lib"
  echo "==> Added CUDA runtime library path for JIT builds: $cuda_runtime_lib"
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

if grep -Fq 'export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-' on_policy_distillation.sh \
  && grep -Fq 'export PROJECT_PATH=${PROJECT_PATH:-' on_policy_distillation.sh; then
  echo "==> Env override patch already applied"
elif git apply --unidiff-zero --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "==> Applying env override patch"
  git apply --unidiff-zero "$PATCH_FILE"
else
  echo "Env override patch is neither applied nor cleanly applicable: $PATCH_FILE" >&2
  exit 1
fi

if ! grep -Fq 'export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-' on_policy_distillation.sh \
  || ! grep -Fq 'export PROJECT_PATH=${PROJECT_PATH:-' on_policy_distillation.sh; then
  echo "Env override patch check failed after patch step." >&2
  exit 1
fi

case "${Y_OPD_ENABLED:-false}" in
  1|true|True|TRUE|yes|Yes|YES)
    if [ -f verl/verl/trainer/ppo/y_opd_controller.py ] \
      && grep -Fq 'y_opd: dict = field(default_factory=dict)' verl/verl/workers/config/rollout.py \
      && grep -Fq 'actor_rollout_ref.rollout.y_opd.enabled' on_policy_distillation.sh; then
      echo "==> Y-OPD patch already applied"
    elif git apply --unidiff-zero --check "$Y_OPD_PATCH_FILE" >/dev/null 2>&1; then
      echo "==> Applying Y-OPD patch"
      git apply --unidiff-zero "$Y_OPD_PATCH_FILE"
    else
      echo "Y-OPD patch is neither applied nor cleanly applicable: $Y_OPD_PATCH_FILE" >&2
      exit 1
    fi

    if [ ! -f verl/verl/trainer/ppo/y_opd_controller.py ] \
      || ! grep -Fq 'y_opd: dict = field(default_factory=dict)' verl/verl/workers/config/rollout.py \
      || ! grep -Fq 'actor_rollout_ref.rollout.y_opd.enabled' on_policy_distillation.sh; then
      echo "Y-OPD patch check failed after patch step." >&2
      exit 1
    fi
    ;;
esac

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
