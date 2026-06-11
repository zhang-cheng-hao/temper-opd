# Temper-OPD

Temper-OPD 是 **RPI-OPD: Teachability-Driven Behavior Policy Improvement for
On-Policy Distillation** 的启动仓库。

核心 thesis：OPD 的训练数据来自 student-induced states。Vanilla OPD 隐含假设
训练期采集状态的 **behavior policy** 和最终部署的 **target policy** 相同，即
`mu = pi_theta`。RPI-OPD 提出受控的 behavior-target decoupling：训练期学习一个
轻量 behavior controller `mu_{theta,phi}`，让 rollout 更常访问可教状态；部署期
移除 `phi`，inference recipe 不变。

```text
target policy:    pi_theta        # student，最终部署
behavior policy:  mu_theta_phi    # rollout-only controller，只在训练期使用
```

Vanilla OPD 是 `mu_theta_phi = pi_theta` 的特例。RPI-OPD 的问题不是“换一个推理
温度是否更好”，而是：

> 训练期受控偏离 target policy 访问更可教的状态，带来的 teachability gain 是否
> 足以超过 distribution mismatch 的代价？

## 方法定位

RPI-OPD 的身份必须保持清楚：

- **rollout-only**：只改变训练期状态采集，不改变部署期推理。
- **teacher-conditioned**：controller 使用 teacher-supervised reachability 信号。
- **online**：router 随 student 训练动态更新。
- **state-targeted**：把 rollout budget 花在更可教的状态，而不是只调一个全局温度。

它不是 adaptive inference temperature、post-hoc filtering、lookahead candidate
selection，也不是 teacher-free self-distillation。

和近期 teacher-free self-distillation/SSD-style 方法相比，RPI-OPD 是它的
teacher-conditioned、online、state-targeted 对应版本：SSD 回答如何 reshape
distribution，RPI-OPD 回答 rollout budget 应该花在哪里。

## 方法草图

RPI-OPD 从当前 student 分布出发，用状态条件化的 decoding config 生成 behavior
rollout：

```text
mu_theta_phi(a | s) = Decode(pi_theta(. | s); phi(z(s), kappa(s)))
```

router 是一个轻量查表控制器：

```text
phi: (z, kappa) -> (T, p)
```

两个 conditioning 轴：

- `z`：teachability type，例如 `Frontier`、`Mastered`、`Drift`、`Other`。
- `kappa`：position type，用 teacher support 集中度估计，例如 teacher top-1
  probability 或 teacher entropy。

引入 `kappa` 的原因是 precision-exploration conflict 更像 position 属性，而不是
整段 state 的属性：lock-like token 需要低温压 distractor，fork-like token 可能需要
更高多样性。`phi(z, kappa)` 把这个冲突显式交给 router，同时仍然保持查表结构。

默认 action ladder：

```text
(T=0.7, p=0.90)  # repair-like
(T=1.0, p=0.95)  # dwell-like
(T=1.2, p=0.97)  # mild escape-like
(T=1.4, p=0.98)  # escape-like
```

`repair / dwell / escape` 只是解释性标签，不是手写策略。

## Teachability Utility

utility 的测量要和 actuator 解耦：sampling temperature 只影响 prefix 生成，状态测量
固定在 `T_eval = 1`，避免“用哪个温度生成”污染“这个状态是否可教”的判断。

对状态 `s`：

```text
g(s) = clip(G(s), c)
J(s) = g(s) * max(R(s) - rho_t, 0)
```

- `g(s)`：student 仍有多少学习空间。
- `R(s)`：teacher correction 是否 reachable。
- `rho_t`：quantile-calibrated reachability gate，由周期性刷新的 calibration
  prefix pool 估计。

batch utility 在每个 `(z, kappa)` cell 内部做 window-normalized 估计：

```text
U_t(action | z, kappa)
  = norm_{z,kappa}(mean_{s in B_{t,z,kappa}} J(s))
```

这样 router 比较的是“给定 cell 内哪个 action 更好”，不依赖不同 cell 的绝对尺度。
mean utility 和 tail utility，例如 CVaR 或 top-quartile mean，应该作为
curriculum-shape ablation，而不只是 robustness check。

## Controller

每个 cell 维护 decoding action 上的分布：

```text
q_t(action | z, kappa)
```

每个 OPD window 后用 discounted/sliding-window exponentiated-gradient 更新：

```text
q_{t+1}(action | z, kappa)
  proportional_to q_t(action | z, kappa) * exp(beta * U_hat_t(action | z, kappa))
```

必须保留 exploration floor，确保每个 `(action, z, kappa)` pair 持续拿到 feedback。
实验报告里要包含 per-cell feedback count，避免把“没访问到”误判为“已经学会”。

## 实现 Caveat

router 在选择下一步 rollout action 之前需要知道当前 prefix 属于哪个 cell。如果 `z`
或 `kappa` 依赖当前 prefix 的 teacher logits，那么普通“生成完整 rollout 后再 teacher
scoring”的 OPD pipeline 不能无代价地把这些特征用于当前 token 的控制。

可行实现路线：

- rollout 时在线查询 teacher prefix distribution，并报告额外 overhead。
- teacher-derived `z/kappa` 只用于 post-hoc attribution，实际 router 使用 student 可见
  proxy feature。
- 把 routing 做到 chunk/window 级别，使用上一窗口已经得到的 teacher feature 控制下一
  窗口。

不要在 baseline 框架没有提前算 teacher prefix distribution 的情况下宣称
“controller 不引入额外 teacher query”。

## 实验优先级

headline 必须是 final task performance，`U` 只做机制解释。

必要对照：

1. **Vanilla OPD**：明确 default decoding recipe。
2. **Tuned constant decoding config `T*`**：排除“一个最优常数温度就够”。
3. **Handcrafted fixed schedule**：例如 high-to-low anneal，排除“固定 schedule 就够”。
4. **Global decoding bandit**：验证 state conditioning 的增量。
5. **Teacher-free SSD/OPSD-style baseline**：隔离 teacher-conditioned reachability 的价值。
6. **RPI-OPD `phi(z)` vs `phi(z,kappa)`**：验证 `kappa` 轴是否真的有增量。

机制分析：

- teachability utility 曲线。
- Frontier-state 访问率。
- distribution mismatch 诊断。
- reachability gate 的 `alpha` sweep。
- mean vs tail aggregator。
- per-cell action distribution 和 feedback count。
- local-control pilot 按 `kappa` 分层。

## Matched Compute

至少匹配：

- total generated tokens。
- teacher-supervised tokens。
- base student checkpoint。
- training windows 和 optimizer budget。

controller overhead 必须报告。如果 overhead 明显，要通过减少 RPI-OPD rollout budget
做补偿。

## Baseline 下载位置

所有外部 baseline 统一放在：

```text
baselines/
```

当前建议下载：

```text
baselines/thunlp-opd        https://github.com/thunlp/OPD
baselines/tinker-cookbook   https://github.com/thinking-machines-lab/tinker-cookbook
baselines/flash-opd         https://github.com/china10s/flash-opd
baselines/opsd              https://github.com/siyan-zhao/OPSD
```

建议用途：

- `thunlp-opd`：优先作为 vanilla OPD 和 OPD diagnostics 参考。
- `tinker-cookbook`：参考实际 OPD workflow。
- `flash-opd`：小实现，适合快速读懂和改 controller。
- `opsd`：teacher-free self-distillation 对照。

默认不把这些外部仓库的完整代码提交到本项目 GitHub；主仓库只保留 baseline manifest。
如果后面需要可复现地锁定版本，再改成 git submodule。

## Y-OPD 入口链路

当前 Y-OPD 原型接在 THUNLP OPD 训练 loop 上，用全局序列级温度控制器做
training-time rollout policy 调整。启动方式：

```bash
Y_OPD_ENABLED=true bash baselines/thunlp-opd/on_policy_distillation.sh
```

在本仓库的复现 wrapper 中，先应用
`patches/thunlp-opd-env-overrides.patch`；当 `Y_OPD_ENABLED` 为 truthy 时，再应用
`patches/thunlp-opd-y-opd.patch`，把下面这条链路补到外部
`baselines/thunlp-opd` 代码里。

关键链路：

- 启动入口：`baselines/thunlp-opd/on_policy_distillation.sh` 读取
  `Y_OPD_ENABLED`、`Y_OPD_TEMPERATURES`、`Y_OPD_TOP_P` 等环境变量。
- Hydra 配置注入：同一脚本把 `+actor_rollout_ref.rollout.y_opd.*` 传入
  rollout config。
- trainer 初始化：`baselines/thunlp-opd/verl/verl/trainer/ppo/ray_trainer.py`
  创建 `GlobalYOPDController(y_opd_config)`。
- 生成前 snapshot：trainer 在每个 step 生成前写入
  `gen_batch_output.meta_info["y_opd_policy"]`。
- rollout 执行：`baselines/thunlp-opd/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
  通过 `_build_y_opd_sampling_params(...)` 为每个样本选择温度，并在 vLLM
  `generate` 前替换成 per-sample `SamplingParams`。
- controller 更新：teacher/student 批注后，trainer 调用
  `self.y_opd_controller.update_from_batch(...)`，使用 `y_opd_temp_id`、
  `old_log_probs`、student top-k 和 teacher-on-student log-prob 等张量更新温度分布。
- 核心逻辑：`baselines/thunlp-opd/verl/verl/trainer/ppo/y_opd_controller.py`
  定义温度空间、A/B yield、virtual likelihood credit、`sparsemax/softmax`
  normalizer 和 logits 更新；checkpoint 中保存 `y_opd_controller.pt`。

## 推荐启动顺序

1. 下载 baseline 到 `baselines/`。
2. 先跑通 vanilla OPD，固定 default decoding。
3. 用同一训练 loop 跑 action ladder 的 constant decoding sweep，选出 `T*`。
4. 跑 fixed schedule，例如 high-to-low。
5. 跑 global bandit。
6. 跑 local-control pilot，并按 `kappa` 分层。
7. pilot 通过后再实现 `phi(z)`，最后实现 `phi(z,kappa)`。

## Go/No-Go 标准

进入完整 RPI-OPD 的最低条件：

- repair-like action 在 high-`kappa` position 上有更强 local-control leverage；或
- escape-like action 在 low-`kappa` position 上有更强 local-control leverage；或
- 在固定 measurement temperature 下，decoding action 对 teachability 有可测影响。

fallback：

- 如果 `kappa` 分层不明显，退回 `phi(z)`。
- 如果 state conditioning 不明显但 global action 有效，改名为
  **Teachability-Driven Rollout Control for OPD**。
- 如果 teacher-free OPSD/SSD-style baseline 拿到大部分 gain，teacher-conditioned
  routing 的 claim 暂时不成立。

## 主要风险

- **Distribution mismatch**：训练期访问更可教状态，但偏离部署分布过远。
- **Measurement contamination**：utility 被 sampling temperature 污染。
- **Sparse cell feedback**：部分 `(z, kappa, action)` cell 样本太少。
- **Constant-temperature explanation**：gain 可能被一个 tuned constant `T*` 解释掉。
- **Teacher-free explanation**：gain 可能来自普通 distribution reshaping，而不是
  teacher-conditioned routing。

## 当前状态

- [x] baseline OPD 代码下载完成。
- [x] THUNLP-OPD strict paper eval 复现完成，结果见
  `docs/results/2026-06-11_thunlp_strict_paper_eval.md`。
- [ ] constant decoding sweep 完成。
- [ ] local-control pilot 完成。
- [ ] global bandit baseline 完成。
- [ ] Y-OPD 与 THUNLP-OPD strict eval 对比完成。
- [ ] `phi(z)` controller 实现完成。
- [ ] `phi(z,kappa)` controller 实现完成。
- [ ] matched-compute downstream evaluation 完成。
