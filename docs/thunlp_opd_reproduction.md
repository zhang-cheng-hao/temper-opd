# THUNLP OPD Reproduction Prep

目标：在本机完成代码/数据/模型/环境边界验证，在 8 卡机器上复现 THUNLP OPD / vanilla OPD baseline。

## Current Status

本机已完成：

| 项 | 状态 |
|---|---|
| code | `baselines/thunlp-opd` 已下载，HEAD `83063cf62293` |
| data | `datasets/dapo-math-17k.parquet` 和默认 AIME25/AMC23/AIME24 eval parquet 已读通 |
| local validation | import/compile/shell syntax 通过，见 `runs/opd_baseline_smoke_local_validation_20260610_144946/summary.tsv` |
| shared env prefix | `/mmu_mllm_hdd/zhangchenghao05/envs/opd-thunlp`；本机和 8 卡训练机共享文件系统，正式训练也用这个 prefix |
| wrapper | `scripts/repro/run_thunlp_opd_8gpu.sh` 已准备，默认 `DRY_RUN=1` |
| patch | `patches/thunlp-opd-env-overrides.patch` 已验证可 clean apply |
| blocker | 仍需准备官方默认 actor/reward 模型，并在 8 卡机器上做 `--min-gpus 8` 环境验证 |

## Environment

THUNLP OPD 单独使用一个 conda prefix，不和 TA-OPD/OPSD 共用。因为本机和 8 卡训练机共享文件系统，先在本机把完整环境装到：

```bash
/mmu_mllm_hdd/zhangchenghao05/envs/opd-thunlp
```

统一安装命令：

```bash
export OPD_ENV_ROOT=/mmu_mllm_hdd/zhangchenghao05/envs
export THUNLP_CONDA_PREFIX="${OPD_ENV_ROOT}/opd-thunlp"

RECREATE_ENV=1 INSTALL_THUNLP_DEPS=1 \
  scripts/repro/bootstrap_thunlp_env.sh configs/reproduction/baseline_paths.env
```

安装后检查：

```bash
cd /path/to/temper-opd
"$THUNLP_CONDA_PREFIX/bin/python" scripts/repro/check_thunlp_env.py --min-gpus 0 --fail-on-missing
```

到 8 卡机器后不重装环境，只用同一个 prefix 做 GPU 数检查：

```bash
"$THUNLP_CONDA_PREFIX/bin/python" scripts/repro/check_thunlp_env.py --min-gpus 8 --fail-on-missing
```

关键版本/风险：

| 项 | 期望/注意 |
|---|---|
| Python | `3.12` |
| vendored verl | `0.7.0.dev` |
| vLLM | 当前安装脚本钉 `0.11.0`，不要退到 `0.7.x` |
| Torch | 应为 `2.8.x`；SGLang extra 钉 `torch==2.8.0` |
| FlashAttention | 官方 wheel 可能依赖较新 glibc；当前共享环境优先用 `FLASH_ATTN_CUDA_ARCHS=80` 源码构建 |
| Ray | setup.py 要 `ray[default]>=2.41.0` |
| PyArrow | 建议确认 `>=19.0.0` |
| Megatron | OPD/FSDP/vLLM 路线用 `USE_MEGATRON=0`，避免额外复杂依赖 |
| DeepSpeed | 不是当前 OPD 脚本核心依赖，不要为了 OPD 额外升级 |

如果 `flash_attn` import 报 `GLIBC_2.32 not found`，不要换整个环境；直接让
`bootstrap_thunlp_env.sh` 的自动 fallback 从源码编译，或手动设置：

```bash
BUILD_FLASH_ATTN_FROM_SOURCE=1 FLASH_ATTN_CUDA_ARCHS=80 \
  INSTALL_THUNLP_DEPS=1 scripts/repro/bootstrap_thunlp_env.sh configs/reproduction/baseline_paths.env
```

## Data

正式默认：

| 用途 | 路径 | 行数 | 核心列 |
|---|---|---:|---|
| train | `baselines/thunlp-opd/datasets/dapo-math-17k.parquet` | 17,917 | `data_source,prompt,ability,reward_model,extra_info` |
| eval | `datasets/test_data/AIME25/test.parquet` | 30 | `prompt,data_source,ability,reward_model,extra_info` |
| eval | `datasets/test_data/AMC23/test.parquet` | 83 | `prompt,source,id,data_source,ability,reward_model,extra_info` |
| eval | `datasets/test_data/AIME24/test.parquet` | 30 | `prompt,source,id,data_source,ability,reward_model,extra_info` |

Schema 期望：

```text
prompt: list[{"role": "user", "content": "..."}]
reward_model: {"ground_truth": str, "style": str}
data_source: str
extra_info.index: str
```

本机检查：

```bash
python scripts/repro/check_thunlp_opd_data.py \
  --train-file baselines/thunlp-opd/datasets/dapo-math-17k.parquet \
  --test-data-dir baselines/thunlp-opd/datasets/test_data \
  --fail-on-warning
```

带 tokenizer prompt length 检查：

```bash
python scripts/repro/check_thunlp_opd_data.py \
  --train-file "$THUNLP_TRAIN_DATASET" \
  --test-data-dir "$THUNLP_TEST_DATA_DIR" \
  --tokenizer "$THUNLP_ACTOR_MODEL_PATH" \
  --max-prompt-length 1024 \
  --token-length-sample-rows 1024 \
  --fail-on-warning
```

默认情况下，prompt 超过 `--max-prompt-length` 只报告不失败，因为 THUNLP 配置会
`filter_overlong_prompts=True`。如果要在正式长跑前强制失败，加
`--fail-on-overlong`。

已知数据注意：

| 项 | 说明 |
|---|---|
| 主训练文件 | `dapo-math-17k.parquet` 核心字段无缺失 |
| 默认 eval | AIME25/AMC23/AIME24 核心字段无缺失 |
| extra train | `DeepScaler/train.parquet`、`MATH/train.parquet`、`MATH-8k/train.parquet` 有少量 missing `ground_truth`，不要误作默认训练集 |
| `verl_example/opd.sh` | 默认指向不存在的 `datasets/DAPO-Math-17k/DAPO-Math.parquet`，不要直接用 |

## Code

代码入口：

| 项 | 路径 |
|---|---|
| official script | `baselines/thunlp-opd/on_policy_distillation.sh` |
| wrapper | `scripts/repro/run_thunlp_opd_8gpu.sh` |
| env override patch | `patches/thunlp-opd-env-overrides.patch` |
| path template | `configs/reproduction/baseline_paths.env.template` |

wrapper 行为：

- 自动应用 env override patch。
- 检查环境 import/GPU。
- 检查 teacher/student tokenizer stop token。
- 检查 train/eval parquet schema。
- `DRY_RUN=1` 只检查并打印配置，不启动训练。
- `DRY_RUN=0` 才执行 `bash on_policy_distillation.sh`。

## Models

THUNLP 官方脚本默认：

| 角色 | 默认路径 | 当前本机状态 |
|---|---|---|
| actor/student | `model/DeepSeek-R1-Distill-Qwen-1.5B` | 缺失 |
| reward/teacher | `model/JustRL-DeepSeek-1.5B` | 缺失 |

8 卡机器优先准备这两个模型，并填入：

```bash
THUNLP_ACTOR_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
THUNLP_REWARD_MODEL_PATH=/path/to/JustRL-DeepSeek-1.5B
```

本机可用模型不能替代官方默认复现：

| 本机模型 | 备注 |
|---|---|
| `Qwen2.5-0.5B` | 可做 smoke，不是论文 baseline |
| `Qwen3-8B` | 可作为 OPSD 迁移候选；和 Qwen2.5 混配有 EOS 风险 |
| `Llama-3.2-1B/3B` | 非 THUNLP 默认 |
| Revela checkpoints | adapter/DeepSpeed checkpoint 形态，不是直接 HF CausalLM 全量模型 |

必须跑 tokenizer preflight：

```bash
python scripts/repro/check_tokenizer_stop_tokens.py \
  --student "$THUNLP_ACTOR_MODEL_PATH" \
  --teacher "$THUNLP_REWARD_MODEL_PATH" \
  --fail-on-mismatch
```

已知 stop-token 风险：

- `Qwen2.5-0.5B -> Qwen2.5-0.5B`：通过，EOS `<|endoftext|>` / `151643`。
- `Qwen3-8B -> Qwen3-8B`：通过，EOS `<|im_end|>` / `151645`。
- `Qwen2.5-0.5B -> Qwen3-8B`：风险，vocab size 和 EOS token/id 不一致。

经验规则：base 蒸 base，instruct 蒸 instruct。不要把 base/instruct 混配结果当作干净 baseline。

## 8-GPU Execution

1. 准备路径文件：

```bash
cp configs/reproduction/baseline_paths.env.template configs/reproduction/baseline_paths.env
vim configs/reproduction/baseline_paths.env
```

2. 检查基础路径：

```bash
scripts/repro/check_8gpu_prereqs.sh configs/reproduction/baseline_paths.env
```

3. dry-run：

```bash
source configs/reproduction/baseline_paths.env
conda activate "$THUNLP_CONDA_PREFIX"
DRY_RUN=1 STRICT_ENV_CHECK=1 FAIL_ON_TOKENIZER_MISMATCH=1 \
  scripts/repro/run_thunlp_opd_8gpu.sh configs/reproduction/baseline_paths.env
```

4. 正式跑：

```bash
source configs/reproduction/baseline_paths.env
conda activate "$THUNLP_CONDA_PREFIX"
DRY_RUN=0 STRICT_ENV_CHECK=1 FAIL_ON_TOKENIZER_MISMATCH=1 \
  scripts/repro/run_thunlp_opd_8gpu.sh configs/reproduction/baseline_paths.env
```

## Known Pitfalls

| Pitfall | Action |
|---|---|
| 一个环境装所有 baseline | 不做；THUNLP/TA-OPD/OPSD 分环境 |
| EOS/stop token 不一致 | 跑 tokenizer preflight；不一致先修再长跑 |
| top-k RKL 在 EOS 位压低停止概率 | 监控 response length、EOS rate、repetition rate |
| THUNLP 脚本硬编码路径/GPU | 使用 env override patch + wrapper |
| built-in validation under verl v0.7 | README 说会低估 5-7 points；保留 `trainer.test_freq=-1`，最终用 scripts/val 单独评估 |
| 默认模型缺失 | 8 卡机器先准备官方默认 actor/reward，或明确声明为非默认复现 |
