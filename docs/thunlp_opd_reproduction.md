# THUNLP OPD Reproduction Prep

目标：在本机完成代码/数据/模型/环境边界验证，在 8 卡机器上复现 THUNLP OPD / vanilla OPD baseline。

## Current Status

本机已完成：

| 项 | 状态 |
|---|---|
| code | `baselines/thunlp-opd` 已下载，HEAD `83063cf62293` |
| data | `dapo-math-17k.parquet` train 17,917 rows；eval AIME25 30、AMC23 83、AIME24 30；prompt overlong 0 |
| local validation | import/compile/shell syntax 通过，见 `runs/opd_baseline_smoke_local_validation_20260610_144946/summary.tsv`；8 卡 strict dry-run 已通过 |
| shared env prefix | `/mmu_mllm_hdd/zhangchenghao05/envs/opd-thunlp`；本机和 8 卡训练机共享文件系统，正式训练也用这个 prefix |
| private paths | `configs/reproduction/baseline_paths.env` 本地存在且被 git 忽略；不要提交私有 token 或密钥 |
| actor/student | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 已下载到 `/mmu_mllm_hdd/zhangchenghao05/models/DeepSeek-R1-Distill-Qwen-1.5B` |
| teacher/reward | `thunlp/JustRL-DeepSeek-1.5B` 官方仓库 gated；当前使用公共 mirror `hbx/JustRL-DeepSeek-1.5B`，下载到 `/mmu_mllm_hdd/zhangchenghao05/models/JustRL-DeepSeek-1.5B` |
| tokenizer | actor/student 与 teacher/reward 的 tokenizer、EOS、vocab 一致；stop-token check 已通过 |
| wrapper | `scripts/repro/run_thunlp_opd_8gpu.sh` 已准备，默认 `DRY_RUN=1` |
| patch | `patches/thunlp-opd-env-overrides.patch` 已验证可 clean apply；手动应用用 `git apply --unidiff-zero` |
| formal run | `logs/thunlp_opd_full_20260610_183210.log` 正在 8 卡跑；已到 `global_step=21/279`，`global_step_20` checkpoint 已写出 |
| remaining | 等完整 279 step 结束后跑独立 eval；论文复现需在报告中注明 teacher mirror，或申请官方 `thunlp/JustRL-DeepSeek-1.5B` 访问后重跑确认 |

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

已验证主栈版本/风险：

| 项 | 期望/注意 |
|---|---|
| Python | `3.12` |
| Host driver | 当前 8 卡机器 `535.129.03` / `nvidia-smi CUDA Version 12.2` |
| Torch | `2.8.0+cu128` |
| FlashAttention | `flash_attn 2.8.1`，按 A800 `sm80` 源码构建 |
| FlashInfer | `flashinfer-python 0.3.1` |
| vLLM | `0.11.0`，不要退到 `0.7.x` |
| SGLang | `0.5.2` |
| Ray | `2.55.1` |
| vendored verl | `0.7.0.dev` |
| SwanLab | `swanlab 0.8.1`；THUNLP 默认 logger 需要，当前用 `SWANLAB_MODE=offline` |
| Matplotlib | `matplotlib 3.10.7`；THUNLP 的 candidate-count plot/logging 会 import |
| PyArrow | 建议确认 `>=19.0.0` |
| Megatron | OPD/FSDP/vLLM 路线用 `USE_MEGATRON=0`，避免额外复杂依赖 |
| DeepSpeed | 不是当前 OPD 脚本核心依赖，不要为了 OPD 额外升级 |
| CUDA compat | host driver 低于 CUDA 12.8 时使用 `cuda-compat-12-8`，当前解包到 `${THUNLP_CONDA_PREFIX}/cuda-compat-12-8` |

核心运行栈 pins 记录在：

```bash
configs/env/thunlp_opd_environment.yml
```

从零重建时仍以 `scripts/repro/bootstrap_thunlp_env.sh` 为准，因为 THUNLP 的
vendored verl 安装脚本会拉平台相关的 CUDA/vLLM/flash-attn 依赖；yml 用于迁移时核对
已验证版本。

如果 `flash_attn` import 报 `GLIBC_2.32 not found`，不要换整个环境；直接让
`bootstrap_thunlp_env.sh` 的自动 fallback 从源码编译，或手动设置：

```bash
BUILD_FLASH_ATTN_FROM_SOURCE=1 FLASH_ATTN_CUDA_ARCHS=80 \
  INSTALL_THUNLP_DEPS=1 scripts/repro/bootstrap_thunlp_env.sh configs/reproduction/baseline_paths.env
```

正式 run 还遇到过 FlashInfer/vLLM JIT 找不到 CUDA 头文件/运行库：

| 报错 | 原因 | 当前处理 |
|---|---|---|
| `fatal error: curand.h: No such file or directory` | pip 的 `nvidia-curand-cu12` 把 header 放在 site-packages 内，不在系统默认 include path | wrapper 自动把 `nvidia/curand/include` 加到 `CPATH/C_INCLUDE_PATH/CPLUS_INCLUDE_PATH` |
| `/usr/bin/ld: cannot find -lcudart` | `libcudart` 在 conda prefix 或 pip 的 `nvidia-cuda-runtime-cu12` 目录内，不在默认 linker path | wrapper 自动把 conda lib 和 `nvidia/cuda_runtime/lib` 加到 `LIBRARY_PATH/LD_LIBRARY_PATH` |

修复后独立 FlashInfer sampling JIT 检查已通过，正式 run 也已越过 vLLM CUDA graph
初始化。

正式 run 第一步 `compute_log_prob` 又暴露了 flash-attn 扩展和 host driver 的兼容问题：

| 报错 | 原因 | 当前处理 |
|---|---|---|
| `CUDA error: device kernel image is invalid` | A800 是 `sm80` 没错；问题是 host driver `535.129.03` 只到 CUDA 12.2，而环境是 torch/vLLM/flash-attn CUDA 12.8 栈 | 下载 `cuda-compat-12-8=570.211.01-0ubuntu1`，解包到 conda prefix；wrapper 自动把 compat `libcuda.so` 目录置于 `LD_LIBRARY_PATH` 前面 |

验证命令：

```bash
COMPAT="$THUNLP_CONDA_PREFIX/cuda-compat-12-8/usr/local/cuda-12.8/compat"
LD_LIBRARY_PATH="$COMPAT:$THUNLP_CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  "$THUNLP_CONDA_PREFIX/bin/python" - <<'PY'
import torch
from flash_attn import flash_attn_varlen_func

q = torch.randn(16, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(16, 4, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(16, 4, 64, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, k, v, cu, cu, 8, 8, 0.0, causal=True)
torch.cuda.synchronize()
print("flash_attn_varlen_ok", out.shape, out.dtype)
PY
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
| 主训练文件 | `dapo-math-17k.parquet` 核心字段无缺失；17,917 rows |
| 默认 eval | AIME25 30 rows、AMC23 83 rows、AIME24 30 rows；核心字段无缺失 |
| prompt length | 使用 DeepSeek tokenizer 检查，`max_prompt_length=1024` 下 overlong 0 |
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
| private paths file | `configs/reproduction/baseline_paths.env`，本地 ignored，只放路径和非敏感开关 |

wrapper 行为：

- 自动应用 env override patch。
- 检查环境 import/GPU。
- 检查 teacher/student tokenizer stop token。
- 检查 train/eval parquet schema。
- `DRY_RUN=1` 只检查并打印配置，不启动训练。
- `DRY_RUN=0` 才执行 `bash on_policy_distillation.sh`。

## Models

当前正式复现使用：

| 角色 | 来源 | 本机路径 | 注意 |
|---|---|---|---|
| actor/student | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `/mmu_mllm_hdd/zhangchenghao05/models/DeepSeek-R1-Distill-Qwen-1.5B` | THUNLP 默认 actor |
| reward/teacher | `hbx/JustRL-DeepSeek-1.5B` mirror | `/mmu_mllm_hdd/zhangchenghao05/models/JustRL-DeepSeek-1.5B` | 官方 `thunlp/JustRL-DeepSeek-1.5B` gated；论文复现需注明 mirror 或申请官方访问 |

路径写入本地 ignored 文件：

```bash
THUNLP_ACTOR_MODEL_PATH=/mmu_mllm_hdd/zhangchenghao05/models/DeepSeek-R1-Distill-Qwen-1.5B
THUNLP_REWARD_MODEL_PATH=/mmu_mllm_hdd/zhangchenghao05/models/JustRL-DeepSeek-1.5B
```

模型准备记录：

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir /mmu_mllm_hdd/zhangchenghao05/models/DeepSeek-R1-Distill-Qwen-1.5B

huggingface-cli download hbx/JustRL-DeepSeek-1.5B \
  --local-dir /mmu_mllm_hdd/zhangchenghao05/models/JustRL-DeepSeek-1.5B
```

大文件下载中断时可用断点续传。当前这两个模型最终都落到了单个
`model.safetensors`，约 3.4G；实际遇到 HF CLI 长时间卡住时，用 `wget -c`
续传 `model.safetensors`，保留已经下载好的 tokenizer/config 小文件。

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
- `DeepSeek-R1-Distill-Qwen-1.5B -> JustRL-DeepSeek-1.5B mirror`：通过，vocab size 151665，EOS `<｜end▁of▁sentence｜>` / `151643`。

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
SWANLAB_MODE=offline \
DRY_RUN=0 STRICT_ENV_CHECK=1 FAIL_ON_TOKENIZER_MISMATCH=1 \
  scripts/repro/run_thunlp_opd_8gpu.sh configs/reproduction/baseline_paths.env
```

当前后台启动约定：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG="logs/thunlp_opd_full_${RUN_ID}.log"
OUTLINES_DIR="/mmu_mllm_hdd/zhangchenghao05/cache/outlines/${RUN_ID}"
mkdir -p logs /mmu_mllm_hdd/zhangchenghao05/tmp/ray "$OUTLINES_DIR"

setsid env \
  CONDA_PREFIX="$THUNLP_CONDA_PREFIX" \
  PATH="$THUNLP_CONDA_PREFIX/bin:$PATH" \
  PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCH_CUDA_ARCH_LIST=8.0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  RAY_TMPDIR=/mmu_mllm_hdd/zhangchenghao05/tmp/ray \
  RAY_DISABLE_DISK_MONITOR=1 \
  OUTLINES_CACHE_DIR="$OUTLINES_DIR" \
  SWANLAB_MODE=offline \
  DRY_RUN=0 STRICT_ENV_CHECK=1 FAIL_ON_TOKENIZER_MISMATCH=1 \
  bash scripts/repro/run_thunlp_opd_8gpu.sh configs/reproduction/baseline_paths.env \
  > "$LOG" 2>&1 < /dev/null &
```

`STRICT_ENV_CHECK=1` 会让环境/GPU 检查失败时直接退出；
`FAIL_ON_TOKENIZER_MISMATCH=1` 会在 teacher/student tokenizer 或 stop token 不一致时直接退出。
在当前工具会话里不要用普通 `nohup ... &`，因为会话清理可能杀掉子进程；`setsid` 已验证可让
训练保持在后台。

正式 run 记录：

| 时间 | 日志 | 结果 |
|---|---|---|
| 2026-06-10 18:13 CST | `logs/thunlp_opd_full_20260610_181327.log` | 越过 vLLM CUDA graph 后因缺 `swanlab` 失败 |
| 2026-06-10 18:18 CST | `logs/thunlp_opd_full_20260610_181857.log` | 已进入训练循环，第一步 `compute_log_prob` 因 flash-attn / driver CUDA 兼容失败；加 `cuda-compat-12-8` 后最小 flash-attn 测试通过 |
| 2026-06-10 18:32 CST | `logs/thunlp_opd_full_20260610_183210.log` | 正在跑；已通过 strict env/tokenizer、vLLM CUDA graph、`compute_log_prob` 和 actor update；截至 2026-06-10 19:20 CST 到 `global_step=21/279`，新增 fatal/error 计数 0；首个保存点 `global_step_20` 已写出，`latest_checkpointed_iteration.txt=20`；当前估计剩余约 9h15m |

当前首个 checkpoint：

```bash
/mmu_mllm_hdd/zhangchenghao05/output/temper-opd-repro/thunlp_opd/checkpoint/token_reward_direct_DAPO-Math-17k_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-10_18-33-11/global_step_20
```

`global_step_20/actor` 下已有 8 个 rank 的 `model_world_size_8_rank_*.pt`、
`optim_world_size_8_rank_*.pt` 和 `extra_state_world_size_8_rank_*.pt`，说明保存链路正常。

## Known Pitfalls

| Pitfall | Action |
|---|---|
| 一个环境装所有 baseline | 不做；THUNLP/TA-OPD/OPSD 分环境 |
| EOS/stop token 不一致 | 跑 tokenizer preflight；不一致先修再长跑 |
| top-k RKL 在 EOS 位压低停止概率 | 监控 response length、EOS rate、repetition rate |
| THUNLP 脚本硬编码路径/GPU | 使用 env override patch + wrapper |
| built-in validation under verl v0.7 | README 说会低估 5-7 points；保留 `trainer.test_freq=-1`，最终用 scripts/val 单独评估 |
| 官方 teacher gated | 当前用 `hbx/JustRL-DeepSeek-1.5B` 公共 mirror；论文报告中注明，或申请官方访问后复核 |
| 下载中断 | HF CLI 可能长时间卡在大 safetensors；保留 partial 后用断点续传，不要删已下载分片 |
| `swanlab` 缺失 | THUNLP 默认 `trainer.logger=[console,swanlab]`，环境需安装 `swanlab==0.8.1`；本地设 `SWANLAB_MODE=offline` |
| `matplotlib` 缺失 | 不会中断训练，但 candidate-count plot/logging 会每步报错；环境需安装 `matplotlib==3.10.7` |
| FlashInfer JIT 找不到 CUDA 组件 | wrapper 自动补 `nvidia-curand-cu12` 和 `nvidia-cuda-runtime-cu12` 的 include/lib path |
| flash-attn `device kernel image is invalid` | host driver CUDA 12.2 跑 CUDA 12.8 扩展；使用 `cuda-compat-12-8` forward-compat lib |
| Ray disk 95% warning | 当前共享盘还有约 21TB 可用，像是容量/配额识别问题；先观察，只有 object spilling 失败才处理 |
| conda `libtinfo` warning | 由 wrapper 暂时把 conda lib 放进 `LD_LIBRARY_PATH` 引起；不影响训练，后续可精简为仅对子进程/JIT 生效 |
