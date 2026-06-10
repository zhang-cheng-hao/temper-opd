# 8-GPU Baseline Reproduction Runbook

目标：本机只做代码路径和环境边界验证；正式论文 baseline 复现放到 8 卡机器上跑。

THUNLP OPD 的完整准备流程见：

```text
docs/thunlp_opd_reproduction.md
```

## 当前本机验证

2026-06-10 本机已完成轻量 validation：

```text
runs/opd_baseline_smoke_local_validation_20260610_144946/summary.tsv
```

结果：`python_imports`、FlashOPD/TA-OPD/OPSD/Tinker/HPD compile、THUNLP/OPSD/HPD shell syntax、TrOPD README 状态检查均通过；FlashOPD 12-step 在这次 validation 中按计划跳过。

## 8 卡机器准备

1. 复制并填写路径模板：

```bash
cp configs/reproduction/baseline_paths.env.template configs/reproduction/baseline_paths.env
vim configs/reproduction/baseline_paths.env
```

2. 检查路径：

```bash
scripts/repro/check_8gpu_prereqs.sh configs/reproduction/baseline_paths.env
```

3. 检查 teacher/student tokenizer 和 stop-token 对齐：

```bash
python scripts/repro/check_tokenizer_stop_tokens.py \
  --student "$THUNLP_ACTOR_MODEL_PATH" \
  --teacher "$THUNLP_REWARD_MODEL_PATH"
```

4. 检查 THUNLP OPD 数据：

```bash
python scripts/repro/check_thunlp_opd_data.py \
  --train-file "$THUNLP_TRAIN_DATASET" \
  --test-data-dir "$THUNLP_TEST_DATA_DIR"
```

5. 不要用一个 conda 环境跑所有 baseline。按下面顺序拆环境。

## OPD Stop-Token Pitfall

OPD 复现前必须检查 teacher/student 的模型形态和终止 token。一个常见训崩模式是：
student 在少数 step 后生成长度暴涨、验证集暴跌、case 里出现无限重复。分析 top-k RKL
位置时，最大分歧常集中在 EOS/stop token：例如 student 倾向 `<|endoftext|>`，teacher
倾向 `<|im_end|>`。如果 top-k RKL 只在 student top-k 上重归一化，teacher 在这些 token
上的概率可能变成长尾噪音，梯度会错误压低 student 停止概率。

复现策略：

| 检查 | 要求 |
|---|---|
| model family | 优先 base 蒸 base、instruct 蒸 instruct，不混 chat template |
| tokenizer | 记录 `eos_token_id`、`pad_token_id`、chat template、候选 stop token 编码 |
| OPD loss | top-k RKL 近似要特别检查终止位；若做 EOS mask/remap 必须写入实验配置 |
| runtime metrics | 必须 log 平均 response length、EOS rate、repetition rate、stop-token prob/overlap |
| failure response | 长度暴涨或重复率上升时先停 run，检查 EOS 分歧，不要把结果当作方法失败 |

如果必须混用 base/instruct 或不同 stop token 约定，至少预先实现并记录一种处理方式：
采样到 EOS 时 mask loss，或在 logits/probability 层把 student/teacher 的 stop token 做显式
映射。前者可能只延缓训崩；后者更适合长期复现，但需要在 baseline 和 RPI 中一致使用。

## Priority 1: THUNLP OPD / Vanilla OPD

用途：先复现 vanilla OPD，作为 RPI 的主地基。

官方环境边界：

```bash
source configs/reproduction/baseline_paths.env
INSTALL_THUNLP_DEPS=1 scripts/repro/bootstrap_thunlp_env.sh configs/reproduction/baseline_paths.env
```

正式入口：

```bash
DRY_RUN=1 scripts/repro/run_thunlp_opd_8gpu.sh configs/reproduction/baseline_paths.env
DRY_RUN=0 scripts/repro/run_thunlp_opd_8gpu.sh configs/reproduction/baseline_paths.env
```

这个 wrapper 会应用 `patches/thunlp-opd-env-overrides.patch`，让官方脚本的模型、数据、
输出目录、GPU 数、response 长度等关键项能通过环境变量覆盖；`DRY_RUN=1` 只检查并打印
最终配置，不启动训练。

必须改的内容：

| 项 | 位置 | 动作 |
|---|---|---|
| actor model | `on_policy_distillation.sh` 的 `ACTOR_MODEL_PATH` | 改成 `$THUNLP_ACTOR_MODEL_PATH` |
| reward/teacher model | `on_policy_distillation.sh` 的 `REWARD_MODEL_PATH` | 改成 `$THUNLP_REWARD_MODEL_PATH` |
| train data | `TRAIN_DATASET` | 默认仓库内 `datasets/dapo-math-17k.parquet` 存在，可保留 |
| test data | `TEST_DATA_DIR` | 默认仓库内 test data 存在，可保留 |
| GPU count | `trainer.n_gpus_per_node=8` | 8 卡机器保留 |
| stop token | actor/reward tokenizer | 用 `check_tokenizer_stop_tokens.py` 检查，不一致先修 |

本机不建议直接跑 `on_policy_distillation.sh`：它硬编码 8 卡、长 response、`N_RESPONSES=4`，更适合 8 卡正式复现。

## Priority 2: TA-OPD

用途：复现最关键 loss-side / token-selection 对照，并产出 E0 可借鉴的 fixed-context diagnostic。

训练环境不是外层 `requirements.txt` 能覆盖的轻环境。按 upstream 的 slime stack：

```bash
cd baselines/ta-opd/slime_ta_opd
bash build_conda.sh
```

它会准备 Python 3.12、CUDA 12.9、Torch 2.9.1 cu129、SGLang、Megatron-LM、flash-attn、Transformer Engine、Apex 等。建议在 8 卡机器独立执行，避免污染其他 baseline 环境。

正式主方法 suite 模板：

```bash
source configs/reproduction/baseline_paths.env
cd "$TA_OPD_DIR/slime_ta_opd"

export SLIME_DIR="$PWD"
export MEGATRON_LM_DIR="$MEGATRON_LM_DIR"
export OUTPUT_ROOT="${REPRO_OUTPUT_ROOT}/ta_opd"
export TEACHER_MODEL="$TA_TEACHER_MODEL"
export STUDENT_HF="$TA_STUDENT_HF"
export STUDENT_TORCH_DIST="$TA_STUDENT_TORCH_DIST"
export PROMPT_DATA="$TA_PROMPT_DATA"
export COMMON_CONTEXT="$TA_COMMON_CONTEXT"
export BASELINE_METRICS="$TA_BASELINE_METRICS"

export TAG=main_methods_k16_ratio10_seed1
export SEED_LABEL=seed1
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40 tip:0.10:50"
export TEACHER_GPU=0
export RAY_GPUS=1,2,3
export EVAL_GPUS=4,5
export ACTOR_NUM_GPUS_PER_NODE=2
export ROLLOUT_NUM_GPUS=1
export OPD_TOPK_METRICS_K=16

bash ../scripts/train/run_teachability_opd_method_suite.sh
```

Blockers to resolve before formal run:

| Blocker | Why it matters |
|---|---|
| `STUDENT_TORCH_DIST` | Must be Megatron/slime format with `latest_checkpointed_iteration.txt` |
| `COMMON_CONTEXT` / `BASELINE_METRICS` | Main suite eval expects fixed-context bank and theta0 metrics |
| path placeholders | Upstream scripts contain `/path/to/...`; fill env or patch wrapper |
| Ray isolation | Scripts may `ray stop --force`; isolate ports/temp dirs on shared machines |

## Priority 3: OPSD

用途：teacher-free self-distillation 对照，放在 vanilla OPD 和 TA-OPD 后面。

官方环境：

```bash
cd baselines/opsd
conda env create -p "$OPSD_CONDA_PREFIX" -f environment.yml
conda activate "$OPSD_CONDA_PREFIX"
pip install flash-attn==2.8.3 --no-build-isolation
```

关键版本：Python 3.10、Torch 2.8.0、Transformers 4.57.1、TRL 0.26.0、DeepSpeed 0.18.2、vLLM 0.11.0。

8 卡入口：

```bash
cd "$OPSD_DIR"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_opsd_8b.sh
```

必须改的内容：

| 项 | 默认 | 动作 |
|---|---|---|
| `--model_name_or_path` | `/data0/shared/Qwen3-8B` | 改成 `$OPSD_MODEL_NAME_OR_PATH` |
| `--output_dir` | `/data0/siyanz/opsd/` | 改成 `$OPSD_OUTPUT_DIR` |
| `--main_process_port` | `12949` | 共享机器改成空闲端口 |
| dataset | HF `siyanzhao/Openthoughts_math_30k_opsd` | 确认 8 卡机器可联网或预缓存 |

## Optional: HPD

HPD 更适合作为 cheap distillation baseline，不适合作为第一版 horizon baseline 底座。

LlamaFactory 版适合轻量验证；verl 版更接近在线/Ray/FSDP，但工程重。

8 卡 verl 入口：

```bash
cd "$HPD_DIR/verl"
NNODES=1 NGPUS_PER_NODE=8 CKPTS_DIR="$HPD_CKPTS_DIR" bash recipe/HPD/run_hpd.sh
```

先不要把 HPD 排在 OPD/TA-OPD/OPSD 前面。

## Not Blocking

| Baseline | 状态 |
|---|---|
| TrOPD | 官方 README 当前写着 training/evaluation code TODO；不阻塞第一阶段 |
| TRB | 未找到官方训练代码；先记录，不阻塞 |
| POPD/TOPD | 未找到官方代码；后续可在选定底座中实现 horizon-control baseline |
