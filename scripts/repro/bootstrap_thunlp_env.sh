#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${1:-$ROOT_DIR/configs/reproduction/baseline_paths.env}"
DEFAULT_ENV_ROOT="/mmu_mllm_hdd/zhangchenghao05/envs"
RECREATE_ENV="${RECREATE_ENV:-0}"
INSTALL_PREFLIGHT_DEPS="${INSTALL_PREFLIGHT_DEPS:-1}"
INSTALL_THUNLP_DEPS="${INSTALL_THUNLP_DEPS:-0}"
FIX_THUNLP_PIP_CONFLICTS="${FIX_THUNLP_PIP_CONFLICTS:-1}"
BUILD_FLASH_ATTN_FROM_SOURCE="${BUILD_FLASH_ATTN_FROM_SOURCE:-auto}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.1}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION:-12.8.93}"

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

if [ "$RECREATE_ENV" = "1" ] && [ -d "$THUNLP_CONDA_PREFIX" ]; then
  echo "==> Removing existing THUNLP conda prefix: $THUNLP_CONDA_PREFIX"
  conda remove -y -p "$THUNLP_CONDA_PREFIX" --all
fi

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

if [ "$INSTALL_PREFLIGHT_DEPS" = "1" ] && [ "$INSTALL_THUNLP_DEPS" != "1" ]; then
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
python -m pip install math-verify jinja2
python -m pip install -e .

if [ "$FIX_THUNLP_PIP_CONFLICTS" = "1" ]; then
  echo "==> Fixing known THUNLP pip dependency conflicts"
  python -m pip uninstall -y outlines decord || true
  python -m pip install \
    cupy-cuda12x==13.6.0 \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.11.0.86
fi

needs_flash_attn_source=0
if [ "$BUILD_FLASH_ATTN_FROM_SOURCE" = "1" ]; then
  needs_flash_attn_source=1
elif [ "$BUILD_FLASH_ATTN_FROM_SOURCE" = "auto" ]; then
  if ! python - <<'PY'
import flash_attn  # noqa: F401
PY
  then
    needs_flash_attn_source=1
  fi
fi

if [ "$needs_flash_attn_source" = "1" ]; then
  echo "==> Building flash-attn from source for CUDA arch(s): $FLASH_ATTN_CUDA_ARCHS"
  conda install -y -c nvidia "cuda-nvcc=$CUDA_NVCC_VERSION"
  CUDA_TARGET_DIR="$THUNLP_CONDA_PREFIX/targets/x86_64-linux"
  python -m pip uninstall -y flash-attn || true
  CUDA_HOME="$THUNLP_CONDA_PREFIX" \
    CPATH="$CUDA_TARGET_DIR/include:${CPATH:-}" \
    C_INCLUDE_PATH="$CUDA_TARGET_DIR/include:${C_INCLUDE_PATH:-}" \
    CPLUS_INCLUDE_PATH="$CUDA_TARGET_DIR/include:${CPLUS_INCLUDE_PATH:-}" \
    LIBRARY_PATH="$CUDA_TARGET_DIR/lib:$THUNLP_CONDA_PREFIX/lib:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="$CUDA_TARGET_DIR/lib:$THUNLP_CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    CC="$THUNLP_CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
    CXX="$THUNLP_CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
    MAX_JOBS="$FLASH_ATTN_MAX_JOBS" \
    FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    python -m pip install --no-build-isolation --no-binary flash-attn --no-cache-dir "flash-attn==$FLASH_ATTN_VERSION"
fi

python -m pip check
