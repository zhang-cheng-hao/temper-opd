# RPI-OPD Controller 设计规格 v1(实现级)

配套 proposal v4。目标:把 controller 从数学骨架落到可实现的系统组件。所有默认值标注 [D],均可被 P0/E0 结果覆盖。

---

## 0. 一个必须先做的系统决策:teacher 打分的调度模式

$\hat z$ 用"上一 chunk 的 teacher 统计"作 lagged 特征,这要求 **teacher 在 chunk $i{+}1$ 生成前完成 chunk $i$ 的打分**。标准 OPD 是整条 rollout 结束后批量打分——两者冲突。二选一:

**Mode A — 流水线 teacher(推荐,默认)[D]**
teacher scorer 作为异步 worker:student 生成 chunk $i{+}1$ 的**同时**,scorer 对 chunk $i$ 打分并把统计写回 feature store。
- compute 记账:**总 teacher forward 数不变**(同样的 token 反正都要打分,只是提前打),额外代价仅为流水线气泡与调度复杂度;
- wall-clock:student 生成与 teacher 打分重叠,理想情况近零开销;chunk 长度 $L_c$ 要足够长让打分跟得上生成(见 §5);
- 这是"无额外 teacher query"声明在 Mode A 下严格成立的原因——**改变的是调度,不是查询量**。

**Mode B — 窗口滞后(降级备选)**
within-trajectory 不传 teacher 信号;$\hat z$ 来自**上一个 OPD window** 在相同/相似 prompt 上的 prompt 级 $z$ 统计(prompt-conditioned prior)。
- 零调度改动,teacher 完全维持事后批量;
- 信号更弱:$\hat z$ 退化为 prompt 级先验,state 级分辨率只剩 $\hat\kappa$;
- 若 Mode A 工程上做不动,Mode B + 降级阶梯 $\phi(\hat\kappa,d)$ 是诚实的退路。

**P0 顺手测的裁决量:** $z$ 的 chunk 间自相关。自相关高 ⇒ Mode A 的 lag-1 外推有效;自相关低 ⇒ Mode A 也救不了 $\hat z$,直接 Mode B + $\phi(\hat\kappa,d)$,省一套异步工程。

---

## 1. 系统架构(组件与数据流)

```
┌─────────────┐  per-chunk SamplingParams   ┌──────────────┐
│  Rollout    │◄────────────────────────────│  Controller  │
│  Engine     │                             │  (本规格)     │
│  (vLLM)     │── chunk tokens + student ──►│              │
└─────┬───────┘   logit stats (κ̂ 用)        └──▲────┬──────┘
      │ finished chunks                        │    │ q_t, b, ρ, λ 更新
      ▼                                        │    │ (每 OPD window)
┌─────────────┐   token-level p_T stats    ┌──┴────▼──────┐
│  Teacher    │──────────────────────────► │ Feature Store │
│  Scorer     │   (Mode A: 异步流式;        │ (per-traj,    │
│  (async)    │    Mode B: 窗口后批量)       │  per-chunk)   │
└─────────────┘                             └───────┬──────┘
                                                    ▼
                                            OPD Trainer (loss 不变)
```

Controller 对 OPD loss **零侵入**:它只改 rollout 的采样参数,trainer 照常消费 (trajectory, teacher logits)。

## 2. Controller 状态(数据结构)

```python
class ControllerState:
    # conditioning spec —— 降级阶梯靠换这个 config,不改代码
    cells: CellSpec           # 默认 ẑ∈{F,M,D,O} × κ̂∈{hi,lo} = 8 cells [D]
                              # 降级: (κ̂×d桶) / (κ̂) / (global)
    actions: list[Config]     # 路由集 A,默认 4 个 [D]:
                              # {(0.7,.90),(1.0,.95),(1.2,.97),(1.0,1.0)}
                              # (P0/P1 离线用满 3×3 grid;路由集按 P0 结果重选)

    eta:   float[cells, |A|]  # EG logits
    b:     float[cells]       # per-cell reward baseline (EMA)
    Dbar:  float[cells, |A|]  # per cell-action 的 per-chunk KL 代价 (EMA)
    lam:   float              # Lagrangian dual 变量
    rho:   float[d_buckets]   # depth-stratified reachability gate(probe pool 分位数)
    kappa_thr: float[d_buckets]  # κ̂ 分桶阈值(probe pool 中位数,按 depth)[D]
    buf:   RingBuffer         # 最近 N_w 个 window 的 (cell, a, r, propensity, D_chunk)
    counts: int[cells, |A|]   # ESS / 饥饿检测用
```

## 3. 在线决策路径(chunk 边界,热路径,必须零额外前向)

```python
def route(traj, controller):
    # ---- κ̂:从 student 生成时已有的 logits 顺手算,免费 ----
    H = ema_entropy(traj.last_chunk_student_entropies, W=32)   # [D] W=32 token 窗
    kappa_hat = HI if H < controller.kappa_thr[d_bucket(traj.depth)] else LO
    # 注意方向:低熵 = 高集中度 = high-κ̂(lock-like)

    # ---- ẑ:Mode A 取 lag-1 chunk 的 teacher 统计;Mode B 取 prompt 先验 ----
    z_hat = feature_store.lagged_z(traj.id) or prompt_prior_z(traj.prompt_id)
    # 首 chunk:prompt 先验;先验也缺:'Other'

    cell = controller.cells.index(z_hat, kappa_hat)

    # ---- 饥饿回退:cell 样本太少时退到边际分布 ----
    if controller.counts[cell].sum() < n_min:        # [D] n_min=64 chunks
        q = marginal_q(controller)                   # 跨 cell 聚合的 q(a)
    else:
        q = softmax(controller.eta[cell])

    q_tilde = (1 - eps) * q + eps / len(A)           # ε-floor,propensity 有下界
    a = sample(q_tilde)
    log_propensity(traj.id, chunk_i, a, q_tilde[a])  # IPW 用
    return SamplingParams(T=a.T, top_p=a.p)
```

vLLM 落地:**chunked decode loop**——每次 `generate(max_tokens=L_c)`,返回后换 SamplingParams 续生成,prefix cache 保证重发成本仅为调度开销。不要用 logit processor 内嵌切换(propensity 记录和 chunk 对齐都会变脏)。

## 4. Reward 回填与更新(每个 OPD window 一次,冷路径)

```python
def update(controller, window_data):
    # ---- 1. 状态打分(teacher 统计已到齐;T_eval=1)----
    for chunk in window_data:
        J = [g(s) * max(R(s) - controller.rho[d_bucket(s)], 0) for s in chunk.gen_states]
        r_raw = mean(J)
        r = r_raw - controller.b[chunk.cell]                  # baseline 修正
        controller.b[chunk.cell] = ema(controller.b[chunk.cell], r_raw, m=0.9)  # [D]
        D_chunk = sum(kl_mu_pi(s) for s in chunk.gen_states)  # 从 student logits 精确算
        controller.Dbar[chunk.cell, chunk.a] = ema(..., D_chunk, m=0.9)
        controller.buf.push(chunk.cell, chunk.a, r, chunk.propensity, D_chunk)

    # ---- 2. per-cell utility(IPW + cell 内 z-score 归一)----
    for cell in cells:
        U = ipw_mean(controller.buf[cell])        # r/propensity 的加权均值, per action
        U = zscore_within(cell, U)                # 窗口归一,吸收 ρ 漂移
        U_prime = U - controller.lam * normalize(controller.Dbar[cell])

        # ---- 3. discounted EG ----
        controller.eta[cell] = gamma_disc * controller.eta[cell] + beta * U_prime
        #                      [D] gamma_disc=0.9, beta=0.5(z-score 后)

    # ---- 4. dual update(trajectory-KL rail)----
    D_traj = mean_per_traj_cumulative_KL(window_data)
    controller.lam = max(0, controller.lam + alpha_lam * (D_traj - delta))
    #                      [D] alpha_lam=0.05, delta 两档进 ablation

    # ---- 5. probe & 校准(每 N_probe 个 window)----
    if window_idx % N_probe == 0:                 # [D] N_probe=4,probe 预算 ~5%
        pool = run_native_probe_rollouts(pi_theta)  # 默认 decoding,不经 router
        controller.rho = depth_quantiles(pool, alpha)      # [D] alpha=0.3
        controller.kappa_thr = depth_medians_of_student_entropy(pool)
        log_proxy_confusion(window_data)          # macro-F1 / MI,oracle 标签免费(事后已有)
        log_state_freq_diff(pool, window_data)    # d_μ vs d_π 的 Frontier/Drift/κ 频率
```

## 5. 超参默认表

| 参数 | 默认 [D] | 依据/敏感性 |
|---|---|---|
| $L_c$ chunk 长度 | math: 128 tok;agentic: turn 边界 | Mode A 下需 ≥ teacher 打分延迟 × 生成速率;P0 扫 {64,128,256} |
| 路由动作集 $|\mathcal A|$ | 4 | 样本效率优先;**由 P0 在 3×3 grid 上的结果重选**(取各 cell 最优的并集) |
| ε-floor | 0.10,前 3 window 从 0.5 退火 | 每 action propensity ≥ 2.5%,IPW 方差可控 |
| β(EG 步长) | 0.5 | U 已 z-score,量纲稳定 |
| $\gamma_{disc}$ | 0.9 / window | 等效记忆 ~10 window;θ 漂移快则降 |
| baseline/Dbar EMA | 0.9 | — |
| $K$($M_K$) | 10 | 与 TA-OPD 对齐,便于 E5 可比 |
| $g$ 截断 $c$ | 5 nats | E0 回归后可调 |
| $z$ 阈值 | probe pool 分位数(g 中位 × R-ρ 符号) | 预注册,不手调 |
| $\alpha$($\rho$ 分位) | 0.3 | sweep {0.2,0.3,0.5} 进机制分析 |
| depth 桶数 | 4(等 token 数分桶) | — |
| $n_{min}$(饥饿阈值) | 64 chunks | 低于此回退边际 q |
| $N_w$(buffer 窗口) | 5 windows | 与 $\gamma_{disc}$ 二选一即可,都开冗余 |
| probe 预算 | 5% rollout tokens | 三用:ρ 校准 + κ̂ 阈值 + mismatch 诊断 |
| λ 初值 / $\alpha_\lambda$ | 0 / 0.05 | dual ascent 标准设置 |

## 6. 冷启动 = P0 热启动(免费的初始化)

P0 的 paired 估计给出每个 oracle cell 上各 action 的干净 $\hat U$。初始化:

$$\eta_0(a|c) = \beta_0 \cdot \mathrm{zscore}\big(\hat U_{P0}(a|c)\big),\quad \beta_0=0.5\ \text{[D]}$$

P0 不只是 go/no-go gate,**也是 controller 的先验**。ε 退火(0.5→0.1)保证热启动错了也能纠正。对照实验加一组 cold-start(η=0)验证热启动增益,顺便测 controller 的自学习能力。

## 7. 必须落日志的诊断量(v4 协议的实现承接)

每 window 落盘:`q_t` 全表、per cell-action counts(ESS)、IPW 与朴素均值的差(混杂信号)、$\widehat D^{\mathrm{traj}}$ 按 depth、λ 轨迹、proxy confusion(macro-F1/MI)、b 与 ρ 轨迹、饥饿回退触发次数。
训练后离线:online 学到的 per-cell argmax action vs P0 paired 估计的一致率(v4 的对齐检查);$q_t$ 轨迹图(不振荡背书)。

## 8. 已知的设计风险与守护

- **chunk 边界效应**:T 在边界跳变可能制造风格断裂;守护:相邻 chunk action 变化时在首 2 token 线性插值 T(可关,作 ablation);
- **Mode A 流水线失速**:teacher 慢于生成时 ẑ 退化为 lag-2/lag-3;feature store 记录实际 lag,lag>1 的 chunk 在分析中分层;
- **数值守护**:η clip 到 [-10,10];r 截断到 ±5σ;任何 NaN → 该 window 跳过更新并报警;
- **cell 合并预案**:Drift×hi-κ̂ 这类天然稀疏 cell 若长期饥饿,按预注册规则并入相邻 cell(F/M/D/O → F/非F 的二值化),属降级阶梯的细粒度版本。

## 9. 实现顺序建议

1. 离线部分先行:J/R/g/z/κ 的打分管线 + probe 校准(E0、P0 直接复用,**不依赖任何在线组件**);
2. chunked decode loop + 固定 schedule(不学习)跑通,验证 chunk 切换无质量退化;
3. Mode B controller(全离线更新,最简);
4. P0 结果出来后再决定是否上 Mode A 异步 scorer——若 $z$ 自相关低,Mode A 整个不用做。

这个顺序保证:**E0 和 P0 在 controller 在线部分写完之前就能跑**,关键路径不被工程阻塞。
