#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${1:-$ROOT_DIR/configs/reproduction/baseline_paths.env}"
DEFAULT_ENV_ROOT="/mmu_mllm_hdd/zhangchenghao05/envs"
INSTALL_PREFLIGHT_DEPS="${INSTALL_PREFLIGHT_DEPS:-1}"
INSTALL_THUNLP_DEPS="${INSTALL_THUNLP_DEPS:-0}"

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

OPD_ENV_ROOT="${OPD_ENV_ROOT:-$DEFAULT_ENV_ROOT}"
THUNLP_CONDA_PREFIX="${THUNLP_CONDA_PREFIX:-$OPD_ENV_ROOT/opd-thunlp}"
THUNLP_OPD_DIR="${THUNLP_OPD_DIR:-$ROOT_DIR/baselines/thunlp-opd}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

mkdir -p "$OPD_ENV_ROOT"

if [ ! -x "$THUNLP_CONDA_PREFIX/bin/python" ]; then
  echo "==> Creating THUNLP conda prefix: $THUNLP_CONDA_PREFIX"
  conda create -y -p "$THUNLP_CONDA_PREFIX" python=3.12
else
  echo "==> THUNLP conda prefix already exists: $THUNLP_CONDA_PREFIX"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$THUNLP_CONDA_PREFIX"

python -V
python -m pip -V

if [ "$INSTALL_PREFLIGHT_DEPS" = "1" ]; then
  echo "==> Installing lightweight THUNLP preflight dependencies"
  python -m pip install pandas pyarrow transformers sentencepiece protobuf jinja2
fi

if [ "$INSTALL_THUNLP_DEPS" != "1" ]; then
  echo "INSTALL_THUNLP_DEPS=0, full THUNLP training dependencies were not installed."
  echo "Set INSTALL_THUNLP_DEPS=1 on the 8-GPU machine to install THUNLP verl/vLLM dependencies."
  exit 0
fi

echo "==> Installing THUNLP OPD dependencies in: $THUNLP_CONDA_PREFIX"
cd "$THUNLP_OPD_DIR/verl"
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install math-verify
python -m pip install -e .
