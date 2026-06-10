# Baseline 复现 Runbook

本文档把 baseline 跑法分成两层：

1. **Smoke**：先用小模型和 12 条 demo 数据验证代码能跑通。
2. **正式 baseline**：按论文或对照实验设置替换模型、数据和训练规模。

## 当前机器状态

- GPU：2 x A800 80GB；当前 GPU1 空闲度较高。
- 可用 Python：`/opt/duyong_ssd/envs/llama_factory/bin/python`
- 可用本地小模型：`/mmu_mllm_hdd/zhangchenghao05/models/Qwen2.5-0.5B`
- 可用本地大模型：`/mmu_mllm_hdd/zhangchenghao05/models/Qwen3-8B`
- base Python 当前不适合训练：导入 torch 会报 `libcudnn.so.9` 缺失。

## 1. FlashOPD Smoke

目标：确认 OPD 基本数据流能跑通。

可迁移入口已固化为：

```bash
PYTHON_BIN=/opt/duyong_ssd/envs/llama_factory/bin/python \
GPU=1 \
scripts/run_flashopd_smoke.sh
```

环境文件见：

```text
configs/env/flashopd_smoke_requirements.txt
configs/env/flashopd_smoke_environment.yml
docs/migration_assets.md
```

如果 `baselines/flash-opd` 是重新下载的原始版本，先应用本仓库记录的兼容补丁：

```bash
cd /mmu_mllm_hdd/zhangchenghao05/code/temper-opd/baselines/flash-opd
git apply ../../patches/flash-opd-transformers-compat.patch
```

原因：当前可用环境的 `transformers==4.45.0` 使用 `torch_dtype=` 和 `tokenizer=`
接口，FlashOPD 原始代码使用了不兼容的 `dtype=` / `processing_class=`。

启动 smoke：

```bash
cd /mmu_mllm_hdd/zhangchenghao05/code/temper-opd
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=baselines/flash-opd \
/opt/duyong_ssd/envs/llama_factory/bin/python -m flashopd.cli \
  --config configs/baselines/flashopd_qwen25_05b_smoke.yaml
```

这个 smoke 使用同一个 `Qwen2.5-0.5B` 同时作为 student 和 teacher，只跑 12 条 demo，
所以只能证明 pipeline 可运行，不能作为论文结果。

本机已验证通过：

```text
steps: 12/12
train_runtime: 10.2055s
train_loss: 2.2509
last rollout_len: 8
output_dir: runs/flashopd_qwen25_05b_smoke
```

## 2. Vanilla OPD 正式 Baseline

优先使用 `baselines/thunlp-opd`，这是最贴近 OPD 论文设置的实现。

### 2.0 Stop-token / EOS 踩坑记录

OPD 的 teacher/student 不能只看模型大小是否合适，还要看模型形态和终止 token 是否一致。
一个已知失败模式是：base student 蒸 instruct/non-thinking teacher，top-k RKL 在 EOS 位置
出现严重分歧，梯度把 student 的停止概率压低，随后 response length 暴涨、验证集崩掉、
case 里无限重复。

复现前先跑：

```bash
python scripts/repro/check_tokenizer_stop_tokens.py \
  --student "$THUNLP_ACTOR_MODEL_PATH" \
  --teacher "$THUNLP_REWARD_MODEL_PATH"
```

硬性记录：

```text
student/teacher 是否同为 base 或同为 instruct
eos_token_id / pad_token_id
chat_template 是否存在且一致
<|endoftext|> / <|im_end|> / <|eot_id|> 等 stop token 编码
训练中的平均生成长度、EOS rate、repetition rate
```

经验规则：base 蒸 base，instruct 蒸 instruct。若必须混用，需要在 loss 里显式做
EOS mask/remap，并让所有 baseline 和 RPI 使用同一规则。

官方脚本：

```bash
cd /mmu_mllm_hdd/zhangchenghao05/code/temper-opd/baselines/thunlp-opd
bash on_policy_distillation.sh
```

正式跑之前必须改脚本中的路径：

```bash
ACTOR_MODEL_PATH=...
REWARD_MODEL_PATH=...
TRAIN_DATASET=...
TEST_DATA_DIR=...
```

默认脚本假设 8 卡，并依赖 `verl`、`ray`、`vllm`、OPD 定制 reward/logprob 逻辑。
当前机器只有 2 张 A800，因此建议先做 1-2 卡缩小版复现，再转到 8 卡机器跑正式数值。

最小缩小版需要改：

```bash
trainer.n_gpus_per_node=1 或 2
N_RESPONSES=1
MAX_RESP_LENGTH=512
MAX_VAL_RESP_LENGTH=512
MINI_BATCH_SIZE=1
PARALLEL_SIZE=1
trainer.total_epochs=1
trainer.save_freq=较大值
trainer.test_freq=-1
```

## 3. Constant Decoding Sweep

这是 RPI-OPD 必须打掉的强 baseline。先在 FlashOPD 或 THUNLP OPD 中固定其余设置，只改 rollout decoding：

```text
(T=0.7, p=0.90)
(T=1.0, p=0.95)
(T=1.2, p=0.97)
(T=1.4, p=0.98)
```

FlashOPD 对应字段：

```yaml
rollout_temperature: 1.0
rollout_top_p: 0.95
```

THUNLP OPD 对应字段：

```bash
TEMPERATURE=1.0
# top_p 需要看 verl rollout 配置是否暴露；若未暴露，需要在 hydra 参数里补。
```

## 4. Fixed Schedule Baseline

先不用 router，只做固定 schedule，例如前 30% 高温探索，后 70% 默认温度：

```text
early: T=1.2, p=0.97
late:  T=1.0, p=0.95
```

如果 fixed schedule 已经接近 RPI-OPD，说明 online feedback 的贡献不足。

## 5. Teacher-Free OPSD Baseline

使用 `baselines/opsd`。官方 1B 启动脚本：

```bash
cd /mmu_mllm_hdd/zhangchenghao05/code/temper-opd/baselines/opsd
bash scripts/run_opsd_1b.sh
```

注意：OPSD 官方脚本默认 4 卡 H100 和 `/data0/shared/Qwen3-1.7B` 这类路径。当前机器需要改：

```bash
--model_name_or_path /mmu_mllm_hdd/zhangchenghao05/models/Qwen3-8B
--num_processes 1 或 2
--max_completion_length 512
--per_device_train_batch_size 1
--output_dir /mmu_mllm_hdd/zhangchenghao05/code/temper-opd/runs/opsd
```

## 推荐顺序

1. 跑通 FlashOPD smoke。
2. 用 FlashOPD 做 4 个 constant decoding smoke，确认温度/`top_p` 对 OPD loss 有可见影响。
3. 准备 THUNLP OPD 环境，先跑 1-2 卡小步数。
4. 跑 THUNLP OPD 正式 vanilla baseline。
5. 跑 tuned constant、fixed schedule、global bandit。
6. 跑 OPSD teacher-free 对照。
