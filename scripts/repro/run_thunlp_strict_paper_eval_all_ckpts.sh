#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

THUNLP_CONDA_PREFIX="${THUNLP_CONDA_PREFIX:-/mmu_mllm_hdd/zhangchenghao05/envs/opd-thunlp}"
PYTHON_BIN="${PYTHON_BIN:-$THUNLP_CONDA_PREFIX/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$THUNLP_CONDA_PREFIX/bin/torchrun}"

BASE_MODEL="${BASE_MODEL:-/mmu_mllm_hdd/zhangchenghao05/models/DeepSeek-R1-Distill-Qwen-1.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-/mmu_mllm_hdd/zhangchenghao05/models/JustRL-DeepSeek-1.5B}"
CKPT_ROOT="${CKPT_ROOT:-/mmu_mllm_hdd/zhangchenghao05/output/temper-opd-repro/thunlp_opd/checkpoint/token_reward_direct_DAPO-Math-17k_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-10_18-33-11}"

RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/thunlp_strict_paper_eval_20260610}"
HF_ROOT="${HF_ROOT:-$RUN_DIR/hf}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_DIR/eval_outputs}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/baselines/thunlp-opd/scripts/val/data}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
INCLUDE_TEACHER="${INCLUDE_TEACHER:-1}"
EVAL_REPLACE="${EVAL_REPLACE:-0}"

mkdir -p "$HF_ROOT" "$EVAL_ROOT" "$LOG_DIR"

prepend_path() {
  local var_name="$1"
  local dir="$2"
  if [ -d "$dir" ]; then
    local current="${!var_name:-}"
    if [ -n "$current" ]; then
      export "$var_name=$dir:$current"
    else
      export "$var_name=$dir"
    fi
  fi
}

prepare_cuda_env() {
  export CUDA_HOME="${CUDA_HOME:-$THUNLP_CONDA_PREFIX}"
  export CUDA_PATH="${CUDA_PATH:-$THUNLP_CONDA_PREFIX}"
  export CUDA_TARGET_DIR="${CUDA_TARGET_DIR:-$THUNLP_CONDA_PREFIX/targets/x86_64-linux}"
  export SITE_NVIDIA="${SITE_NVIDIA:-$THUNLP_CONDA_PREFIX/lib/python3.12/site-packages/nvidia}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
  export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-$RUN_DIR/flashinfer_jit_cache}"

  prepend_path PATH "$THUNLP_CONDA_PREFIX/bin"
  prepend_path PATH "$CUDA_TARGET_DIR/bin"

  prepend_path CPATH "$CUDA_TARGET_DIR/include"
  prepend_path CPATH "$SITE_NVIDIA/cuda_runtime/include"
  prepend_path CPATH "$SITE_NVIDIA/cuda_nvcc/include"
  prepend_path CPATH "$SITE_NVIDIA/curand/include"
  prepend_path C_INCLUDE_PATH "$CUDA_TARGET_DIR/include"
  prepend_path C_INCLUDE_PATH "$SITE_NVIDIA/cuda_runtime/include"
  prepend_path C_INCLUDE_PATH "$SITE_NVIDIA/cuda_nvcc/include"
  prepend_path C_INCLUDE_PATH "$SITE_NVIDIA/curand/include"
  prepend_path CPLUS_INCLUDE_PATH "$CUDA_TARGET_DIR/include"
  prepend_path CPLUS_INCLUDE_PATH "$SITE_NVIDIA/cuda_runtime/include"
  prepend_path CPLUS_INCLUDE_PATH "$SITE_NVIDIA/cuda_nvcc/include"
  prepend_path CPLUS_INCLUDE_PATH "$SITE_NVIDIA/curand/include"

  prepend_path LIBRARY_PATH "$CUDA_TARGET_DIR/lib"
  prepend_path LIBRARY_PATH "$THUNLP_CONDA_PREFIX/lib"
  prepend_path LIBRARY_PATH "$SITE_NVIDIA/cuda_runtime/lib"
  prepend_path LIBRARY_PATH "$SITE_NVIDIA/curand/lib"

  prepend_path LD_LIBRARY_PATH "$CUDA_TARGET_DIR/lib"
  prepend_path LD_LIBRARY_PATH "$THUNLP_CONDA_PREFIX/lib"
  prepend_path LD_LIBRARY_PATH "$SITE_NVIDIA/cuda_runtime/lib"
  prepend_path LD_LIBRARY_PATH "$SITE_NVIDIA/curand/lib"
  prepend_path LD_LIBRARY_PATH "$THUNLP_CONDA_PREFIX/cuda-compat-12-8/usr/local/cuda-12.8/compat"
}

is_hf_ready() {
  local model_dir="$1"
  [ -f "$model_dir/config.json" ] && \
    { [ -f "$model_dir/model.safetensors.index.json" ] || [ -f "$model_dir/model.safetensors" ]; }
}

prepare_cuda_env

"$PYTHON_BIN" -m py_compile \
  scripts/repro/merge_thunlp_fsdp_actor_to_hf.py \
  scripts/repro/run_thunlp_readme_eval.py

mapfile -t CKPT_DIRS < <(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
if [ "${#CKPT_DIRS[@]}" -eq 0 ]; then
  echo "No global_step_* checkpoints found under $CKPT_ROOT" >&2
  exit 1
fi

printf '%s\n' "${CKPT_DIRS[@]}" > "$RUN_DIR/checkpoints_to_eval.txt"

echo "==> Strict paper eval setting"
echo "data_dir=$DATA_DIR"
echo "tasks=AIME24,AIME25,AMC23 n=16 temp=0.7 top_p=0.95 max_tokens=31744 enable_thinking=false"
echo "ckpts=${#CKPT_DIRS[@]} include_base=$INCLUDE_BASE include_teacher=$INCLUDE_TEACHER"

for ckpt_dir in "${CKPT_DIRS[@]}"; do
  step_name="$(basename "$ckpt_dir")"
  actor_dir="$ckpt_dir/actor"
  hf_dir="$HF_ROOT/$step_name"

  if is_hf_ready "$hf_dir"; then
    echo "==> HF already ready: $hf_dir"
    continue
  fi

  echo "==> Merging $step_name to $hf_dir"
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    scripts/repro/merge_thunlp_fsdp_actor_to_hf.py \
    --actor-dir "$actor_dir" \
    --base-model "$BASE_MODEL" \
    --output-dir "$hf_dir" \
    --trust-remote-code \
    >"$LOG_DIR/merge_${step_name}.log" 2>&1
done

declare -a MODEL_NAMES=()
declare -a MODEL_PATHS=()

if [ "$INCLUDE_BASE" = "1" ]; then
  MODEL_NAMES+=("base_orig")
  MODEL_PATHS+=("$BASE_MODEL")
fi

if [ "$INCLUDE_TEACHER" = "1" ]; then
  MODEL_NAMES+=("teacher_justrl")
  MODEL_PATHS+=("$TEACHER_MODEL")
fi

for ckpt_dir in "${CKPT_DIRS[@]}"; do
  step_name="$(basename "$ckpt_dir")"
  MODEL_NAMES+=("$step_name")
  MODEL_PATHS+=("$HF_ROOT/$step_name")
done

for i in "${!MODEL_NAMES[@]}"; do
  name="${MODEL_NAMES[$i]}"
  model="${MODEL_PATHS[$i]}"
  echo "==> Evaluating $name: $model"

  replace_args=()
  if [ "$EVAL_REPLACE" = "1" ]; then
    replace_args+=(--replace)
  fi

  "$PYTHON_BIN" scripts/repro/run_thunlp_readme_eval.py \
    --model "$model" \
    --name "$name" \
    --data-dir "$DATA_DIR" \
    --output-root "$EVAL_ROOT" \
    --tasks AIME24 AIME25 AMC23 \
    --n 16 \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-tokens 31744 \
    --gpu-ids "$GPU_IDS" \
    "${replace_args[@]}" \
    >"$LOG_DIR/eval_${name}.log" 2>&1
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ.get("RUN_DIR", "runs/thunlp_strict_paper_eval_20260610"))
eval_root = Path(os.environ.get("EVAL_ROOT", str(run_dir / "eval_outputs")))
rows = []
for result_path in sorted(eval_root.glob("*/grading_results.json")):
    model = result_path.parent.name
    with result_path.open() as f:
        results = json.load(f)
    weighted_correct = 0.0
    weighted_total = 0
    for item in results:
        questions = int(item["questions"])
        weighted_correct += float(item["mean_score"]) * questions
        weighted_total += questions
        rows.append({
            "model": model,
            "task": item["task"],
            "questions": questions,
            "rollouts": int(item["rollouts"]),
            "mean_score": float(item["mean_score"]),
            "best_score": float(item["best_score"]),
            "solve_none": int(item["solve_none"]),
            "solve_all": int(item["solve_all"]),
            "avg_output_length": float(item["avg_output_length"]),
            "format_error_rollouts": int(item["format_error_rollouts"]),
        })
    if weighted_total:
        rows.append({
            "model": model,
            "task": "OVERALL_WEIGHTED",
            "questions": weighted_total,
            "rollouts": "",
            "mean_score": weighted_correct / weighted_total,
            "best_score": "",
            "solve_none": "",
            "solve_all": "",
            "avg_output_length": "",
            "format_error_rollouts": "",
        })

summary_path = run_dir / "summary.tsv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
headers = [
    "model",
    "task",
    "questions",
    "rollouts",
    "mean_score",
    "best_score",
    "solve_none",
    "solve_all",
    "avg_output_length",
    "format_error_rollouts",
]
with summary_path.open("w", encoding="utf-8") as f:
    f.write("\t".join(headers) + "\n")
    for row in rows:
        f.write("\t".join(str(row[h]) for h in headers) + "\n")
print(f"summary={summary_path}")
PY

echo "==> Done. Summary: $RUN_DIR/summary.tsv"
