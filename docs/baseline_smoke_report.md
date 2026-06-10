# Baseline Reproduction Plan

本报告记录 baseline 代码准备、环境边界，以及“本机验证”和“8 卡正式复现”的分工。

核心原则：

- 本机只验证代码路径、依赖边界、配置改造点和最小运行入口。
- 论文复现结果是否对齐，在 8 卡机器上跑，不在本机用缩小 smoke 判断。
- 不强行把所有 baseline 塞进一个环境；full reproduction 按 baseline 拆环境。

8 卡正式执行命令和路径检查见：

```text
docs/8gpu_baseline_reproduction.md
configs/reproduction/baseline_paths.env.template
scripts/repro/check_8gpu_prereqs.sh
```

## 本机验证环境

环境文件：

```text
configs/env/temper_opd_local_validation.yml
```

创建方式：

```bash
conda env create -p /mmu_mllm_hdd/zhangchenghao05/envs/temper-opd-local-validation \
  -f configs/env/temper_opd_local_validation.yml
conda activate /mmu_mllm_hdd/zhangchenghao05/envs/temper-opd-local-validation
```

设计原则：这个环境用于本机 lightweight validation 和 FlashOPD 12-step 路径检查，不试图同时满足
OPSD 官方 `vllm==0.11.0`/`torch==2.8.0`、FlashOPD 当前验证栈
`transformers==4.45.0`、以及各 verl fork 的 full training 依赖。

这个环境不是正式复现实验环境。

## 已下载代码

可恢复 manifest：

```text
configs/baselines/opd_baseline_repos.yaml
scripts/download_baseline_repos.sh
```

| Baseline | 路径 | HEAD | 本机验证 |
|---|---|---:|---|
| FlashOPD | `baselines/flash-opd` | `f2485a646dbd` | actual 12-step local run |
| THUNLP OPD | `baselines/thunlp-opd` | `83063cf62293` | config/script inspection; full env on 8卡 |
| TA-OPD | `baselines/ta-opd` | `ccdf21d20664` | tools/package compile; full env on 8卡 |
| TrOPD | `baselines/tropd` | `b4bee6fb9816` | README/TODO check; upstream code not released |
| OPSD | `baselines/opsd` | `7448751f307a` | env/spec inspection; official env on 8卡 |
| HPD / ESR-related candidate | `baselines/hybrid-policy-distillation` | `492b98c58dc6` | recipe inspection; full env on 8卡 if selected |
| Tinker cookbook | `baselines/tinker-cookbook` | `14374b5377e3` | package compile only |

## 本机验证命令

```bash
PYTHON_BIN="$(which python)" GPU=0 scripts/smoke_opd_baselines.sh
```

输出写到：

```text
runs/opd_baseline_smoke_<RUN_ID>/summary.tsv
runs/opd_baseline_smoke_<RUN_ID>/smoke.log
```

这条命令只回答“代码路径/依赖导入/脚本语法是否基本通”，不回答论文数值是否复现。

## 正式复现顺序

| 顺序 | Baseline | 8 卡机器目标 | 成功判据 |
|---:|---|---|---|
| 1 | THUNLP OPD / vanilla OPD | 跑出 vanilla OPD 主结果或论文可比 mini/full setting | 曲线和最终指标接近论文，日志可复查 |
| 2 | TA-OPD | 跑出 TA-OPD diagnostic 和主 baseline | fixed-context gain / downstream trend 对齐论文 |
| 3 | OPSD | 跑 teacher-free 对照 | 官方或论文设置下结果趋势可复现 |
| 4 | Horizon-control/ESR 类 | 若有官方代码则复现；否则在已选底座实现 | matched budget 下形成 cheap baseline |
| 5 | TRB/TrOPD | 代码可用后再跑；当前先记录不可用状态 | 不阻塞第一阶段 |

## 环境策略

| 环境 | Baseline | 核心栈 | 用途 |
|---|---|---|---|
| `/mmu_mllm_hdd/zhangchenghao05/envs/temper-opd-flashopd` 或 local-validation env | FlashOPD smoke/dev | Python 3.11, torch 2.7.1, transformers 4.45.0, deepspeed 0.14.4, datasets 2.20.0 | 快速跑通 OPD loop、小模型 smoke、controller 原型 |
| `/mmu_mllm_hdd/zhangchenghao05/envs/opd-thunlp` | THUNLP OPD / vanilla OPD | Python 3.12, torch 2.8.0+cu128, vLLM 0.11.0, SGLang 0.5.2, ray, vendored verl 0.7.0.dev0, numpy<2 | 主 baseline，正式 8 卡复现 |
| `/mmu_mllm_hdd/zhangchenghao05/envs/ta-opd` | TA-OPD | Python 3.12, CUDA 12.9, torch 2.9.1+cu129, SGLang pinned commit, Megatron-LM pinned commit, flash-attn 2.7.4.post1, Transformer Engine, Apex, slime | TA-OPD 复现；Megatron/slime 格式专用 |
| `/mmu_mllm_hdd/zhangchenghao05/envs/opsd` | OPSD | Python 3.10, torch 2.8.0, transformers 4.57.1, TRL 0.26.0, DeepSpeed 0.18.2, vLLM 0.11.0, xFormers 0.0.32.post1 | teacher-free/self-distillation 对照 |
| `/mmu_mllm_hdd/zhangchenghao05/envs/hpd` | HPD candidate | verl-style stack, SGLang 0.4.9.post6, flash-attn, ray/vLLM deps | 可选 cheap distillation baseline，不排第一批 |
| `/mmu_mllm_hdd/zhangchenghao05/envs/tinker-cookbook` | Tinker cookbook | Python >=3.11, tinker >=0.22.3, transformers >=4.57.6 | API/cookbook 候选，不和本地多卡训练栈混用 |
| `/mmu_mllm_hdd/zhangchenghao05/envs/rpi-dev` | 我们自己的方法开发 | 在最终选定底座后再锁 | RPI-OPD 开发环境 |

不建议一个环境装所有 baseline。冲突来自 `torch`/`transformers`/`deepspeed`/`vllm`/`ray`
这些训练栈的强版本耦合，尤其是不同 repo 的 verl fork 和 CUDA 扩展依赖。

## 当前局限

| 项 | 状态 |
|---|---|
| TRB | 搜到论文和引用，但未找到官方训练代码；manifest 中记录为 `no_official_code_found` |
| POPD/TOPD | 搜到论文，未找到官方代码；后续可直接在 OPD/FlashOPD/THUNLP loop 中实现 horizon-control baseline |
| TrOPD | 官方 README 显示 training/evaluation code 仍是 TODO，当前只能作为代码未释放状态记录 |
| THUNLP full run | 共享 prefix 已准备环境；还需要官方 actor/reward 模型和 8 卡机器 `--min-gpus 8` 验证 |
| HPD/OPSD full run | 需要各自训练栈环境和本地路径改造；应在 8 卡机器上正式复现 |
