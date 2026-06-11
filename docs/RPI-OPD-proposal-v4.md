# RPI-OPD v4: Proxy-Conditioned, Teachability-Rewarded Rollout Control for On-Policy Distillation

> **v4 修订说明(相对 v3 的七处变化):**
> ① **验证层级对齐部署**:P0/P1 改三层(oracle / proxy / gap),go/no-go 判据移到 **proxy 层**;
> ② **$\hat z$ 可辨识性协议**:macro-F1 + 互信息 + chunk 间自相关 + 逐轴 ablation,降级阶梯写死;
> ③ **新增 E0(reward validity)**:证明 $J/r$ 预测真实 learning gain,而非仅 teachable-looking;闭合 action→$J$→learning 的最后一环;
> ④ **KL rail 降调**:从"有保证的约束"改为"信息论上界 + probe 实证评估松紧";补 $\mu\ll\pi$、EOS、计算方式、TV 换算四个技术说明;
> ⑤ **bandit 去偏协议**:ε-floor、ESS、IPW/DR 敏感性、checkpoint fixed effect、online-vs-paired 对齐;
> ⑥ **baseline 按基础设施分三层**,新增 oracle-RPI、random-cell router、proxy heuristic、depth-only schedule;
> ⑦ **$\kappa$ 降为 conditional modifier**:P1 改为固定 $z$ 后的残差解释力检验;related work 补 TrOPD 双重定位与 TIP。

---

## Thesis(主 claim)

> We test whether **deployable, proxy-conditioned decoding control**, trained with **delayed teacher-measured teachability rewards**, can improve on-policy distillation **beyond loss-side token selection (TA-OPD), teacher-directed behavior intervention (TRB; TrOPD's off-policy guidance), and depth/entropy-based rollout heuristics**, under matched trajectory-KL and teacher-token budgets.

这个表述同时承认三件事:router 用的是 deployable proxy 而非 teacher-exact oracle;reward 是延迟的、事后 teacher 测量的;最强的替代解释来自 TA-OPD、TRB、depth/entropy heuristic——本文的实验矩阵就是针对这三类替代解释逐一构造的可证伪对照。

证据链(每环一个实验,环环相扣):

```
E0: J/r 预测真实 learning gain        (reward 是 training-useful 的,不只是 teachable-looking)
P0: proxy cell × action 交互存在      (在 router 实际可见的特征下,最优 action 是 state-dependent 的)
P1: κ̂ 在固定 z 后有残差解释力          (position-type 是有效的 conditional modifier)
E*: matched budget 下超过三类替代解释   (rollout-side teachability control 有独立增量)
```

## Positioning(更新)

「干预作用点 × 导向信号」taxonomy 保持,两处精确化:

- **TrOPD 双重定位**:其 trust-region masking + outlier FKL 属 loss-side teacher-reliability;但其第三组件 off-policy guidance(student 从 teacher prefix 继续生成 + FKL 模仿,鼓励探索走向 teacher 可靠区)**已部分进入状态干预空间**。区分:TrOPD 的状态干预靠**注入 teacher 数据**、方向是 **teacher-reliable** 区域;本文靠 **student rollout 内的 decoding controller**、方向是 **teachable** 状态(teacher 可靠 ≠ student 可学:Mastered 区 teacher 极可靠但无可教内容)。TrOPD 因此与 TRB 同属 teacher-directed 状态干预,机制不同(数据注入 vs 行为混合)。
- **TIP(Xu et al., 2026)加入 related work**:token importance = student entropy × teacher-student divergence,显示 **student entropy 是 token 重要性的强一阶 proxy**(loss-side)。这削弱 $\hat\kappa$ 用 student entropy 的新颖性,但同时是其**可行性的文献支撑**——本文的差异在把该 proxy 用于 rollout-side 在线控制并以 teacher-measured reward 校准,而非 loss 加权。

最近邻差异(TA-OPD / TRB / TAMPO / OmniOPD)同 v3,保留。

## Method(骨架同 v3,接缝修复)

chunk-committed control、$\mu_{\theta,\phi}=\mathrm{Decode}(\pi_\theta;\phi(\hat z,\hat\kappa))$、transition-level reward $r(c_i,a_i)=\overline{J}(c_i^{\mathrm{gen}})-b(\hat z_i,\hat\kappa_i,d_i)$、$T_{\mathrm{eval}}{=}1$ 测量解耦、操作性定义($R\equiv M_K$、$G\equiv D_{\mathrm{KL}}(p_T\|\pi_\theta)$、分位数化 $z$、$\kappa=p_T(\text{top-1})$)——全部沿用 v3。

### Proxy 特征与可辨识性协议(v4 修复之一)

在线特征:$\hat\kappa(s)$ = student entropy / top-1 margin;$\hat z$ = 上一 chunk teacher 打分聚合的 lagged 标签 + prompt 级先验。$\hat z$ 的有效性依赖 $z$ 的 chunk 间自相关——这是可测的经验事实,不是假设:

**必报指标(P0 数据顺手产出):**
- $z_i\to z_{i+1}$、$\kappa_i\to\kappa_{i+1}$ 的 chunk-level transition matrix 与自相关;
- proxy 质量:**macro-F1**(非 accuracy,防类别不平衡)+ 互信息 $I((\hat z,\hat\kappa);(z,\kappa))$;
- 逐轴 ablation:$\hat\kappa$-only / $\hat z$-only / $\hat z{+}\hat\kappa$ / oracle $z{+}\kappa$ 四档的 P0 交互效应与 downstream。

**降级阶梯(预注册):** $\hat z$ 预测性弱(macro-F1 或 MI 低于预注册阈值,或 $\hat z$-only 无交互)⇒ 主方法降为 $\phi(\hat\kappa,d)$(student entropy × depth,皆为强在线信号);再弱 ⇒ $\phi(\hat\kappa)$;$\hat\kappa$ 也无交互 ⇒ global bandit 论文。**不强行保留双轴。**

### Mismatch accounting(v4 降调)

由链式法则,$\widehat D^{\mathrm{traj}}=\mathbb E_\mu[\sum_t D_{\mathrm{KL}}(\mu(\cdot|s_t)\|\pi_\theta(\cdot|s_t))]$ 等于 trajectory KL,并(经 data processing)给出固定 depth 的 state-marginal 偏移的**信息论上界;该上界在长轨迹下可能很松(累积量随长度近线性增长),其实际松紧由 probe rollouts 实证评估**($d_\mu$ vs $d_{\pi_\theta}$ 的 state-type 频率直接对比)。KL rail 的定位:**mismatch accounting + safety rail,不是 downstream performance guarantee。**

技术说明(回应可复现性):
- **绝对连续性**:温度变换不改支撑,top-p 只收缩支撑,故 $\mathrm{supp}(\mu)\subseteq\mathrm{supp}(\pi_\theta)$,$D_{\mathrm{KL}}(\mu\|\pi_\theta)$ 恒有限,无需额外平滑;
- **计算**:per-step KL 从 student logits 精确可算($O(V)$,零额外前向);非"闭式"而是"精确可计算";
- **EOS/终止**:EOS 作普通 token 计入;累积按 $\mu$ 轨迹的实际终止截断;按长度分桶报告,另报长度归一化版本;
- **直观换算**:同时报 Pinsker 界 $\mathrm{TV}\le\sqrt{\widehat D/2}$,给 rail 一个可解释的尺度。

Lagrangian 形式 $U'=U-\lambda\bar D(a|\cdot)$、dual update 维持 $\widehat D^{\mathrm{traj}}\le\delta$、$\delta$ 两档——同 v3。

### Online controller 去偏协议(v4 修复之二)

EG 更新的 reward 估计混杂 checkpoint、time、state-visitation 漂移(action 改变后续状态分布 → 改变各 cell 的样本构成)。协议:

- 每个 OPD window 内保持 **ε-floor 随机化**(每 cell 每 action 的最小采样率,即 propensity 已知且有下界);
- 报告 per cell-action **effective sample size**;
- utility 估计的 **IPW / doubly-robust 敏感性检验**(propensity 由 $q_t$ + floor 显式可得,IPW 免费);
- 机制分析的 mixed-effects 模型加入 **checkpoint/time fixed effect**(或 random slope);
- **online 估计 vs P0 paired 估计对齐检查**:训练中 controller 学到的 per-cell 最优 action 与 P0 干净估计的一致率——不一致本身是 finding(non-stationarity 或混杂的证据)。

## E0 — Reward validity(v4 新增,最早跑)

**目的:证明 $J/r$ 是 training-useful,不只是 teachable-looking。** TA-OPD 用 fixed-context KL-reduction diagnostic 闭合了他们的环;本文 setting 更强(主动改变访问状态),验证义务更重。

设计:固定 checkpoint,收集一批 generated chunks,按 $r(c,a)$ 分位数分桶;对每桶分别做 one-step / few-step distillation update(同 batch 规模),测:
1. **same-context KL reduction**(直接借 TA-OPD 的 diagnostic,顺便建立两文 utility 的可比性);
2. **held-out $\pi_\theta$-native rollout loss**(防"只在 $\mu$ 分布上好看");
3. downstream:answer correctness / verifier score(子集即可);
4. 退化检查:repetition rate、degenerate continuation、format drift;
5. 回归分析:$J,R,g,d,\kappa$ 各自对真实 improvement 的边际解释力(决定 utility 里哪些项配拥有权重)。

**预注册判据:** 高 $r$ 桶的真实 improvement 显著高于低 $r$ 桶且无退化代价;否则 utility 重设计(候选:直接以 same-context KL-reduction 的可预测代理为 reward),在重设计收敛前不进入主实验。

## Pilots(三层化,v4 修复之三)

**P0 — cell-specific 最优 action 存在性:**
- **L1 Oracle**:teacher-exact $(z,\kappa)$ 分层,same-prefix paired,cell×action 交互(mixed-effects + checkpoint fixed effect)⇒ 方法**上界**;
- **L2 Proxy(主判据)**:同一数据按 $(\hat z,\hat\kappa)$ 重分层重估交互 ⇒ **deployable 信号是否存在**;
- **L3 Gap**:oracle→proxy 交互效应的衰减比例 + per-cell 最优 action 的一致率 ⇒ proxy regret 是否可接受。
**go/no-go:proxy 层交互不显著 ⇒ 不 claim deployable state-conditioned routing,即使 oracle 显著**(oracle-only 结果可降级为机制分析短文)。

**P1 — $\kappa$ 作为 conditional modifier(重设计):**
不测全局单调("高 $\kappa$ ⇒ 低熵最优"过强:高 $\kappa$ 可能是 Mastered 也可能是 confident correction)。改测:**固定 $z$(或以 $R,g,d$ 为协变量)后,$\kappa$ 对最优 $(T,p)$ 的残差解释力**(nested model comparison:$a^*\sim z$ vs $a^*\sim z+\kappa+z{\times}\kappa$)。factorial grid $T\in\{0.7,1.0,1.3\}\times p\in\{0.90,0.95,1.0\}$ 同 v3。oracle 与 proxy 两层都做。
判据:$\kappa$ 主效应或 $z{\times}\kappa$ 交互显著且方向跨 checkpoint 稳定 ⇒ 保留双轴;否则按降级阶梯走。

## Experiments(分层重组,v4 修复之四)

主 suite:math reasoning,Qwen3 系(同 v3);agentic 附录级。全部 matched teacher-token & generated-token budget,controller 开销报告并补偿。

**Tier-1:RPI 家族内部 ablation(同一套基础设施,换 conditioning/reward,边际成本低):**

| 变体 | 排除的替代解释 |
|---|---|
| Oracle-RPI(teacher-exact cells) | 方法上界(报告用,非对手) |
| **Proxy-RPI(主方法)** | — |
| Random-cell router(随机分桶,同结构) | cell 分桶本身的正则化效应 |
| Proxy heuristic router(student-entropy 手写规则,无学习) | 收益只是 entropy heuristic |
| Depth-only schedule $\phi(d)$ | $(\hat z,\hat\kappa)$ 只是隐式学了 depth |
| $\phi(\hat z)$-only / $\phi(\hat\kappa)$-only | 各轴增量 |
| Global teachability bandit | state-conditioning 增量 |

**Tier-2:外部 baseline:**
vanilla OPD;tuned constant $(T^*,p^*)$;**TA-OPD**(官方/强复现);**TRB**;horizon-control 家族(ESR 必做;TOPD/POPD 在 infra 允许时加)。

**Tier-3:互补性:** TA-OPD + RPI;TRB + RPI(至少一个组合有稳定叠加收益,是"两个作用点/两个方向正交"的直接证据)。

**预注册成功判据(成文门槛):**
- Proxy-RPI > {tuned constant, global bandit, depth-only, proxy heuristic, ESR}(缺一即"廉价替代成立");
- Proxy-RPI 相对 TA-OPD、TRB 至少不劣,且 Tier-3 至少一个正叠加;
- 仅超过 vanilla + tuned constant ⇒ 不足以成文,转机制研究/可行性分析短文。

## Contributions(微调)

1. **Deployable rollout-side teachability control 的可证伪检验。** proxy-conditioned、chunk-committed、delayed teacher-measured reward、trajectory-KL accounting 的完整控制回路,oracle/proxy/gap 三层验证将"信号存在"与"信号可部署"分离。
2. **作用点 × 导向信号 × 信息可得性的三重分解。** 在统一 utility 族下分离:loss-side vs rollout-side(TA-OPD)、teachability- vs teacher-directed(TRB/TrOPD-guidance)、oracle vs proxy conditioning;每条对照预注册判据,含 reward validity(E0)与互补性(Tier-3)。
3. **Position-type 作为 rollout-side conditional modifier。** 在 SSD/EGRSD/TIP 的 loss-side 加权之外,以 paired factorial intervention 检验 position-type 在固定 teachability state 后对最优采样配置的残差解释力,并经 student-proxy 用于在线路由。

## Risk register(更新)

| 风险 | 触发 | 预案 |
|---|---|---|
| E0 失败:$r$ 不预测真实 gain | 高低桶 improvement 无差 | utility 重设计(KL-reduction 代理);重设计前冻结主实验 |
| P0-proxy 失败(oracle 成立) | L2 交互不显著 | 转 oracle 机制分析短文,或按降级阶梯换 conditioning |
| $\hat z$ 不可辨识 | macro-F1/MI 低、自相关弱 | 降 $\phi(\hat\kappa,d)$;TIP 先例支持 $\hat\kappa$ 路线 |
| cheap baseline 等效 | depth-only / heuristic ≈ RPI | 预注册失败条件,主动降级,不硬推 |
| bandit 混杂 | online vs P0 估计不一致 | IPW/DR 校正后重估;不一致本身入文为 finding |
| 再撞车 | — | E0+P0 结果尽早 arXiv;投稿前复核 OPDHub |

## 执行顺序(最低证据门槛,写正文前完成)

1. **E0**(reward validity)——单 checkpoint,一周量级;失败则一切重来,必须最先跑;
2. **P0 三层 + $\hat z$ 可辨识性**(同一批数据全部产出:交互、macro-F1、MI、自相关、gap);
3. **P1**(κ 条件化检验,复用 P0 infra);
4. Tier-1 家族小规模对跑(同一套代码);
5. 通过以上后才进入 Tier-2/3 全规模与正文写作。

## 核心参考(增量)

v3 列表全部保留,新增/修订:
- TrOPD: arXiv 2606.01249 — **双重定位**:loss-side trust region + §4.3 off-policy guidance(teacher-prefix continuation + FKL,teacher-directed 状态干预);related work 必须按此双重身份处理
- TIP (Xu et al., 2026) — student entropy × teacher-student divergence 的 token importance;$\hat\kappa$ proxy 的先例与支撑,loss-side
- TOPD / POPD(truncated rollouts)— horizon-control baseline 家族(infra 允许时)
