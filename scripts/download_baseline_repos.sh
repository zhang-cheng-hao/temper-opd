#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p baselines

clone_if_missing() {
  local path="$1"
  local url="$2"
  if [ -d "$path/.git" ]; then
    echo "exists: $path"
  else
    echo "clone: $url -> $path"
    git clone --depth 1 "$url" "$path"
  fi
}

clone_if_missing baselines/flash-opd https://github.com/china10s/flash-opd.git
clone_if_missing baselines/opsd https://github.com/siyan-zhao/OPSD.git
clone_if_missing baselines/ta-opd https://github.com/wyy-code/TA-OPD.git
clone_if_missing baselines/tropd https://github.com/Xingrun-Xing2/TrOPD.git
clone_if_missing baselines/hybrid-policy-distillation https://github.com/zwhong714/Hybrid-Policy-Distillation.git
clone_if_missing baselines/tinker-cookbook https://github.com/thinking-machines-lab/tinker-cookbook.git

if [ ! -d baselines/thunlp-opd ]; then
  echo "missing: baselines/thunlp-opd"
  echo "Use https://github.com/thunlp/OPD. This workspace currently used an archive/unpacked copy."
fi
