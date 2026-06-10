# Migration Assets

目标：换环境时先恢复可运行的 baseline 和实验矩阵资产，再进入 E0/P0/RPI 实验。

## 已验证入口

| 层级 | 目标 | 文件/命令 | 当前状态 | 换环境判据 |
|---|---|---|---|---|
| Env | FlashOPD smoke 最小环境 | `configs/env/flashopd_smoke_requirements.txt` / `configs/env/flashopd_smoke_environment.yml` | 已从当前可运行环境抽取 | `python -c "import torch, transformers, datasets, accelerate"` 通过 |
| Smoke | OPD 数据流跑通 | `scripts/run_flashopd_smoke.sh` | 2026-06-10 复验通过 | `global_step=12` 且保存 `checkpoint-12` |
| Actuator | constant decoding smoke sweep | `scripts/run_flashopd_constant_sweep.sh` | 2026-06-10 复验通过 | 4 个 action 都能落盘 |
| Matrix | decoding action ladder | `configs/baselines/flashopd_constant_actions.yaml` | 已固化 | 和脚本中的 4 个 action 一致 |
| Baseline repos | OPD/TA-OPD/TrOPD/OPSD/HPD 代码 manifest | `configs/baselines/opd_baseline_repos.yaml` | 已固化 | `scripts/download_baseline_repos.sh` 可恢复 |
| Local validation | 多 baseline 轻量路径检查 | `scripts/smoke_opd_baselines.sh` | 已准备 | summary.tsv 全部 PASS 或有解释性 SKIP |
| THUNLP | OPD 正式 baseline 入口 | `baselines/thunlp-opd/on_policy_distillation.sh` | 需要新环境缩小版复验 | 1-2 卡小步数可跑通 |

## 当前可用环境快照

| 项 | 值 |
|---|---|
| Python | `3.11.9` |
| GPU smoke | `CUDA_VISIBLE_DEVICES=1` on A800 |
| Torch | `2.7.1+cu126` |
| Transformers | `4.45.0` |
| Datasets | `2.20.0` |
| Accelerate | `0.34.0` |
| DeepSpeed | `0.14.4` |
| NumPy / Pandas | `1.26.4` / `2.2.3` |
| PyArrow import version | `19.0.0` |

注意：当前环境 import `pyarrow` 时有 binary-compatibility warning，但 FlashOPD smoke 已跑通。
全量 `pip freeze` 含大量本机 `file:///croot/...` 路径，不适合作为迁移主文件。

## 2026-06-10 验证记录

| 入口 | 输出目录 | 结果 |
|---|---|---|
| `scripts/run_flashopd_smoke.sh` | `runs/flashopd_qwen25_05b_smoke_scriptcheck` | `global_step=12`, `train_loss=2.250938` |
| `scripts/run_flashopd_constant_sweep.sh` | `runs/flashopd_constant_sweep_scriptcheck/repair_like` | `train_loss=2.249720` |
| `scripts/run_flashopd_constant_sweep.sh` | `runs/flashopd_constant_sweep_scriptcheck/dwell_like` | `train_loss=2.249577` |
| `scripts/run_flashopd_constant_sweep.sh` | `runs/flashopd_constant_sweep_scriptcheck/mild_escape_like` | `train_loss=2.249591` |
| `scripts/run_flashopd_constant_sweep.sh` | `runs/flashopd_constant_sweep_scriptcheck/escape_like` | `train_loss=2.248961` |

## 新环境 bring-up 顺序

1. 恢复外部 baseline 目录，至少需要 `baselines/flash-opd`。
2. 创建环境：

```bash
export OPD_ENV_ROOT=/mmu_mllm_hdd/zhangchenghao05/envs
conda env create -p "${OPD_ENV_ROOT}/temper-opd-flashopd" \
  -f configs/env/flashopd_smoke_environment.yml
conda activate "${OPD_ENV_ROOT}/temper-opd-flashopd"
```

或：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r configs/env/flashopd_smoke_requirements.txt
```

3. 若 `baselines/flash-opd` 是原始版本，应用兼容补丁：

```bash
cd baselines/flash-opd
git apply ../../patches/flash-opd-transformers-compat.patch
```

4. 设置模型路径。默认 smoke 需要：

```text
/mmu_mllm_hdd/zhangchenghao05/models/Qwen2.5-0.5B
```

如果新机器路径不同，先改 `configs/baselines/flashopd_qwen25_05b_smoke.yaml`
里的 `student_model` 和 `teacher_model`。

5. 跑最小 smoke：

```bash
PYTHON_BIN=/path/to/python GPU=0 scripts/run_flashopd_smoke.sh
```

6. 跑 constant action sweep：

```bash
PYTHON_BIN=/path/to/python GPU=0 scripts/run_flashopd_constant_sweep.sh
```

7. 通过后再准备 THUNLP OPD 的 1-2 卡缩小版，不要直接上全规模。

正式 baseline 论文复现不使用统一 smoke 环境；按 `docs/baseline_smoke_report.md`
中的环境策略，在 8 卡机器上为 THUNLP OPD、TA-OPD、OPSD 分别建环境。

## Baseline/RPI 矩阵

| 阶段 | 实验 | 目的 | 依赖 | 产物 |
|---|---|---|---|---|
| 0 | FlashOPD smoke | 环境和 OPD loop 连通性 | 小模型、本地 demo 数据 | checkpoint、trainer_state |
| 1 | Constant decoding sweep | 证明 actuator 可控且有可测输出 | FlashOPD smoke | 4 个 action 的 run 目录 |
| 2 | THUNLP vanilla OPD mini | 接近正式 OPD 实现的最小复现 | verl/ray/vllm 环境 | mini baseline 日志 |
| 3 | THUNLP vanilla OPD full | 正式 vanilla baseline | 训练数据、评测集、多卡 | baseline checkpoint/eval |
| 4 | E0 reward validity | 验证 `J/r` 预测真实 gain | 固定 checkpoint、bucket update | high/low reward bucket 表 |
| 5 | P0 proxy/oracle/gap | 验证 deployable state-conditioned routing 信号 | E0 后的数据采集 infra | interaction/gap/proxy 质量表 |
| 6 | P1 kappa residual | 验证 `kappa` 条件化增量 | P0 infra | nested model comparison 表 |

## 可迁移路径变量

| 变量 | 默认/当前值 | 迁移时动作 |
|---|---|---|
| `OPD_ENV_ROOT` | `/mmu_mllm_hdd/zhangchenghao05/envs` | 所有 conda prefix 放在这里 |
| `PYTHON_BIN` | 当前为 `/opt/duyong_ssd/envs/llama_factory/bin/python` | 新环境显式传入 |
| `GPU` | smoke 用 `1` | 新机器按空闲卡设置 |
| `ROOT_DIR` | 仓库根目录 | 通常无需设置 |
| `FLASHOPD_DIR` | `baselines/flash-opd` | baseline 放在别处时设置 |
| `CONFIG` | `configs/baselines/flashopd_qwen25_05b_smoke.yaml` | 换模型/数据时复制并改新 config |
| `OUT_ROOT` | `runs/flashopd_constant_sweep_$RUN_ID` | sweep 结果目录 |

## 不要迁移的东西

| 内容 | 原因 |
|---|---|
| `runs/` 下的大 checkpoint | 体积大，只保留关键日志/表格即可 |
| 全量 conda/pip freeze | 本机路径和 ABI 绑定较多 |
| 外部 baseline 源码直接提交进主仓库 | 已被 `.gitignore` 忽略，后续需要可复现再用 submodule |
