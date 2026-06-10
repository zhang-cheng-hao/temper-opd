#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${1:-}"
FAILURES=0

if [ -n "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

check_path() {
  local name="$1"
  local path="$2"
  if [ -e "$path" ]; then
    printf "%s\tPASS\t%s\n" "$name" "$path"
  else
    printf "%s\tFAIL\t%s\n" "$name" "$path"
    FAILURES=$((FAILURES + 1))
  fi
}

check_optional_path() {
  local name="$1"
  local path="$2"
  if [ -e "$path" ]; then
    printf "%s\tPASS\t%s\n" "$name" "$path"
  else
    printf "%s\tWARN\t%s\n" "$name" "$path"
  fi
}

check_cmd() {
  local name="$1"
  local cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf "%s\tPASS\t%s\n" "$name" "$(command -v "$cmd")"
  else
    printf "%s\tFAIL\tmissing command: %s\n" "$name" "$cmd"
    FAILURES=$((FAILURES + 1))
  fi
}

printf "name\tstatus\tdetail\n"
check_cmd conda conda
check_cmd git git
check_cmd python python

check_path repo_root "$ROOT_DIR"
check_path flash_opd "$ROOT_DIR/baselines/flash-opd"
check_path thunlp_opd "${THUNLP_OPD_DIR:-$ROOT_DIR/baselines/thunlp-opd}"
check_path ta_opd "${TA_OPD_DIR:-$ROOT_DIR/baselines/ta-opd}"
check_path opsd "${OPSD_DIR:-$ROOT_DIR/baselines/opsd}"

if [ -n "${OPD_ENV_ROOT:-}" ]; then
  check_path opd_env_root "$OPD_ENV_ROOT"
fi
if [ -n "${THUNLP_CONDA_PREFIX:-}" ]; then
  check_optional_path thunlp_conda_prefix "$THUNLP_CONDA_PREFIX"
  check_optional_path thunlp_python "$THUNLP_CONDA_PREFIX/bin/python"
fi
if [ -n "${TA_OPD_CONDA_PREFIX:-}" ]; then
  check_optional_path ta_opd_conda_prefix "$TA_OPD_CONDA_PREFIX"
fi
if [ -n "${OPSD_CONDA_PREFIX:-}" ]; then
  check_optional_path opsd_conda_prefix "$OPSD_CONDA_PREFIX"
fi

if [ -n "${THUNLP_ACTOR_MODEL_PATH:-}" ]; then
  check_path thunlp_actor_model "$THUNLP_ACTOR_MODEL_PATH"
fi
if [ -n "${THUNLP_REWARD_MODEL_PATH:-}" ]; then
  check_path thunlp_reward_model "$THUNLP_REWARD_MODEL_PATH"
fi
if [ -n "${TA_STUDENT_HF:-}" ]; then
  check_path ta_student_hf "$TA_STUDENT_HF"
fi
if [ -n "${TA_STUDENT_TORCH_DIST:-}" ]; then
  check_path ta_student_torch_dist "$TA_STUDENT_TORCH_DIST"
  check_path ta_student_torch_dist_latest "$TA_STUDENT_TORCH_DIST/latest_checkpointed_iteration.txt"
fi
if [ -n "${TA_TEACHER_MODEL:-}" ]; then
  check_path ta_teacher_model "$TA_TEACHER_MODEL"
fi
if [ -n "${TA_PROMPT_DATA:-}" ]; then
  check_path ta_prompt_data "$TA_PROMPT_DATA"
fi
if [ -n "${OPSD_MODEL_NAME_OR_PATH:-}" ]; then
  check_path opsd_model "$OPSD_MODEL_NAME_OR_PATH"
fi

exit "$FAILURES"
