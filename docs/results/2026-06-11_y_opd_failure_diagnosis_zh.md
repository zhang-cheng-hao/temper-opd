# Y-OPD 失败复盘中文说明

日期: 2026-06-11

这份文档是
[英文复盘](2026-06-11_y_opd_failure_diagnosis.md)
和
[Y-OPD v1 design freeze](../Y-OPD-v1-design-freeze.md)
的中文解释版。它不是新的实现方案，主要用于解释术语、判断逻辑和下一步为什么这样排。

## 一句话结论

这次 Y-OPD run 不是证明 controller 坏了，而是证明 controller 很忠实地优化了一个没有被 rail
约束、没有经过 M1/M2 验证的 reward。

失败机制不是“温度伪造了 entropy”。entropy 是学生模型在某个状态上的真实困惑度，温度不能直接伪造它。
真正发生的是: 高温采样把轨迹带进了坏前缀，学生在那些状态上真的困惑，所以 A-rate 真的升高。
问题是这些状态不是有价值的训练数据，属于 off-distribution 状态访问劫持。

随后出现双环正反馈:

1. controller 环: 高温访问更坏的状态，A-rate 和 Y 上升，controller 更偏向高温。
2. student 环: 学生在高温坏轨迹上被蒸馏更新，grad norm 爆炸，学生本身退化，整体 entropy 继续升高。

所以 “Y 上升但 true reward 很低” 不是好信号，而是分布偏移加学生损伤。

## 这次数据怎么读

早期 warmup 还正常。step 1 和 step 10 时 Y-OPD 强制温度为 1.0，和 vanilla OPD 接近。

warmup 后立刻坏掉:

- step 20: Y-OPD actor entropy 是 vanilla 的 5.3 倍。
- step 20: Y-OPD grad norm 是 vanilla 的 26 倍。
- step 20: true reward 只有 vanilla 同步值的大约 33%。

B-rate 从 0.00181 涨到 0.01258 不能当成 “B 被温度操纵成功” 的证据。因为 P(T=1.7)
只从 0.0807 到 0.0956，变化很小；更可能是学生已经退化后，到处都变成 teacher 高分歧状态，
于是 B 的条件被污染地满足。这条曲线双向禁引: 既不能当正证据，也不能当反证据。

## 关键术语

### Y-OPD v1

当前要复现和重启的方案版本。它的 online conditioning 只允许用 `(hat kappa, d)`:

- `hat kappa`: 学生侧的局部置信/不确定性 proxy。
- `d`: 深度或位置 bucket。

teacher 信号只进入延迟 reward，不进入在线 routing 特征。Y-OPD v1 不包含 `hat z`，也不包含 Mode A。

### RPI-OPD v4

另一个更复杂的旧 proposal。它包含 `hat z` 和 Mode A 异步 teacher 特征。现在的纪律是:
不要把 RPI-OPD v4 的器官重新长到 Y-OPD v1 里。

### Scope freeze

设计冻结。意思是先把 Y-OPD v1 到底包含什么、不包含什么写死，避免实现过程中不断混入别的 proposal。
本 repo 里单独放了 [Y-OPD v1 design freeze](../Y-OPD-v1-design-freeze.md)。

### State visitation hijacking

状态访问劫持。temperature 没有直接改变学生模型的 entropy 公式，但它改变了模型会走到哪些前缀状态。
如果高温更容易走到垃圾前缀，学生在那里真的会更困惑，A-rate 就会真的变高。
这不是 proxy 被伪造，而是访问分布被劫持。

### Off-distribution

训练或评估时访问到了自然策略很少访问的状态。Y-OPD 本来想找“有教学价值的困难状态”，但这次没有 rail，
高温直接把采样带到了坏状态。坏状态上的高 yield 不等于好训练数据。

### KL rail

KL rail 是“可行域约束”，不是普通 reward penalty。它要限制温度改变采样分布的幅度。

直观上: 如果 T=1.3 生成分布和 T=1.0 很接近，可以放行；如果 T=1.7 让分布偏太远，就要被截断。
这次 run 没有 rail，所以高温可以无约束地把训练带偏。

KL 是按 response 累计的，长度越长自然越大，所以预算要按长度 bucket 设置，或者按长度归一。
只记录长度 bucket 不够，预算本身也要考虑长度。

### Lagrangian rail

rail 的软约束版本。形式上像 `U' = U - lambda * KL`。但当前建议先做硬约束 rail，
因为硬约束更容易排查问题: 超预算就不让采样，不先引入新的调参自由度。

### Temperature arm

controller 可以选择的温度动作，比如 0.7、1.0、1.3。每个温度就是一个 arm。
这次默认动作集到 1.7，太宽，等于预置了一个容易把模型带出分布的高温按钮。

### Temperature bucket

把很多 arm 合并成粗桶，比如低温/中温/高温。M1-lite 里建议先用 bucket，
因为 B 是稀有事件，10 个 step 摊到 13 个 arm 上，per-arm B 计数可能太少。

### A-rate

A 是“学生高 entropy”的事件。A-rate 是 A 在 token 或样本上的比例。
这次 A-rate 到 0.39 到 0.51，表示接近一半 token 被判成高 entropy，说明阈值或学生状态都很可疑。

### B-rate

B 是“学生低 entropy 但和 teacher 高分歧，且 teacher 偏好还在学生 top-k 里”的事件。
它本来想捕捉“学生自信但错，teacher 可以教”的位置。

注意: 退化循环也可能满足 B，所以只看 B-rate 上升不够。必须用 M1/M2 判断它是否真能带来学习增益。

### `tau_H` 和 `tau_D`

A/B 判定用的阈值:

- `tau_H`: high entropy 阈值。
- `tau_D`: teacher disagreement 阈值。

协议要求它们来自固定 probe pool 的分位数，并记录校准 checkpoint 和时间。
不能从退化中的 online batch 动态估，否则 A/B 的定义会漂。

### Probe pool

固定的一组校准样本。用途是先把阈值定死，避免每次 online run 都因为模型状态不同而改掉 A/B 的刻度。

### Propensity

某个样本实际被分配到某个温度 arm 的概率。做 IPW 时必须知道这个概率。
这次早期窗口是在 `policy_top_p=0.7` 下采的，部分低概率 arm 可能被直接截断成 0 propensity。
所以 M1-lite 第一步必须先审计每个 arm 实际采了多少样本。

### IPW

Inverse Propensity Weighting，逆倾向加权。它用 `1 / propensity` 给样本加权，试图纠正不同 arm 被采样概率不同的问题。
但如果某个 arm 的 propensity 是 0，就无法用 IPW 补回来，因为根本没有样本。

### ESS

Effective Sample Size，有效样本量。IPW 权重差异太大时，表面样本很多，实际有效样本可能很少。
所以 state-conditioned controller 后续要同时记录 per-cell action counts、ESS 和 IPW。

### M1

干净的操纵性实验。理想形式是 same-prefix paired: 对同一个前缀，用不同温度生成或打分，
看 A/B 是否真的随温度变化。它是判断 B 是否可被 actuator 操纵的正式证据。

### M1-lite

现在能先做的弱版本。只用早期 post-warmup 数据，大约 step 11-20，按 step 分层，
审计 propensity，再按低/中/高温 bucket 看 A/B 曲线。

M1-lite 只能给方向性证据，不能替代正式 paired M1 下最终结论。

### M2

判断 yield 是否真的对应学习增益。做法是在 vanilla/on-distribution 轨迹上按 candidate utility 分桶，
然后做少量蒸馏，看高 yield bucket 是否真的带来更好的 KL 降低、held-out loss、answer score 等。

M2 必须匹配桶间长度分布和 prompt 难度分层。否则长轨迹会机械地积累更多 A/B，难题也天然更容易高分歧。

### Why M2 uses step 100-150, not step 220

`global_step_220` 已经接近 teacher，weighted 0.5944 对 teacher 0.6012。
这时 teacher-student gap 太小，yield 是否预测 gain 很难测出来。

所以 M2 应该主看 100-150 这样的中段 checkpoint。step 220 可以作为“接近收敛时 yield 信号衰减”的附加观察。

### True reward gate

`true_reward >= 0.9 * vanilla` 不能单独当硬门槛。Y-OPD 的目标之一是暴露学生自信错误，
所以 rollout task reward 轻微下降可能是正常的。

但如果 KL rail 已经启用，分布偏移应该被限制在小半径内。这时 true reward 大跌就不正常，
可以作为 rail 联用的硬 gate。

### Student-health gate

真正要优先保护的是学生健康:

- actor entropy 不应超过 vanilla 的 1.5 倍。
- actor grad norm 不应超过 vanilla 的 5 倍。
- 第一批 smoke checkpoint 应该立刻跑小规模 eval。

这次 run 在 step 20 已经同时破了 entropy 和 grad norm gate。

### Global bandit

整条 response 只选一个温度。它最多能说明“全局哪个温度更好”，不能证明 state-conditioned 控制有效。
当前 sequence-level controller 只是 smoke，不是 Y-OPD v1 主 claim。

### State-conditioned control

按局部状态选动作，比如根据 `(hat kappa, d)` 的 cell 来选择温度。
这是 Y-OPD v1 真正要验证的方向，但必须在 M1、M2、rail、epsilon-floor、reward normalization 都就位后再做。

### Epsilon-floor

探索下界。做法是把 controller 策略和均匀分布混合:

```text
q_tilde = (1 - eps) * q + eps / |A|
```

这样每个 arm 都有非零概率，后续 IPW 和纠偏才成立。

### Arm top-p

把 controller 的 arm 分布按概率截断，只保留累计概率 top-p 的 arm。
这和 epsilon-floor 相反: 它会把低概率 arm 的 propensity 变成 0。
一旦 controller 漂向高温，低温 arm 可能没有样本，后续很难纠正。

### Z-score normalization

把不同 base rate 的指标先标准化再合成。A 是常见事件，B 是稀有事件。
直接用 `A_rate + 5 * B_rate` 会被 A 接管，因为 A 的数量级大得多。

### Reward search

不断凭直觉改 reward，再跑实验，看哪个结果好。这是现在要避免的事情。
正确顺序是先 rail 限定可行域，再用 probe pool 定阈值，再用 M1/M2 裁决 A/B 怎么合成。

## 下一步优先级

当前优先级是:

1. 先审计早期窗口每个温度 arm 的实际采样数和 propensity。
2. 做 M1-lite，只给方向性结论，不代替正式 M1。
3. 做 M2 on-distribution 分桶蒸馏，主用 100-150 中段 checkpoint，并匹配长度和 prompt 难度。
4. 实现 hard KL rail，预算按长度 bucket 或长度归一。
5. 用 epsilon-floor 替换 arm top-p。
6. 再回到在线控制 smoke。

现在最值得花算力的是 M2。它回答的是最关键的问题: 这些 yield 指标在自然分布上到底能不能预测真实学习增益。
