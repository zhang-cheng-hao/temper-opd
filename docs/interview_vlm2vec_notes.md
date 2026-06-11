# 本地 VLM2Vec / Goods / Better SID 面试补充材料

生成日期: 2026-06-11

用途: 这份文档用于补充“项目深挖面试手册”的 CREM 章节。重点是把本地两个项目里的真实工程资产、实验路线和安全口径整理出来:

- `/mmu_mllm_hdd/zhangchenghao05/code/vlm2vec`
- `/mmu_mllm_hdd/zhangchenghao05/code/vlm2vec-better-sid`

注意: 这不是官方 VLM2Vec 论文总结。官方 VLM2Vec 只放在最后作为 related work。面试时不要把本地 goods/Better SID 探索讲成官方 VLM2Vec，也不要把未确认的候选 checkpoint 讲成正式上线模型。

## 0. 当前本地项目状态

| 项目 | 路径 | 当前分支/HEAD | 作用 |
|---|---|---|---|
| Goods / unified embedding | `/mmu_mllm_hdd/zhangchenghao05/code/vlm2vec` | `goods-strong300-emb-compare`, HEAD `5810c89 restore cached projector workflow` | 商品/视频/直播相关 unified embedding、retrieval + OpenQA 混训、cached projector |
| Better SID / XSeq | `/mmu_mllm_hdd/zhangchenghao05/code/vlm2vec-better-sid` | `musid`, HEAD `4e25f06 feat: configure xseq token scopes` | SID/OpenQA 压缩生成、XSeq 跨样本 attention、embedding 相似度梯度探索 |

本地状态提醒:

- `vlm2vec` 当前比旧项目总结更新，`src/cached_projector/` 和 `experiments/hetu/cached_projector/` 已恢复。
- `vlm2vec` 工作树有本地未提交文档/计划变更；`vlm2vec-better-sid` 有 `experiments/musid/scripts/train_musid_multitask.sh` 本地改动。
- 当前已确认有训练候选 checkpoint 和 encode-only embedding 产物，但缺统一全量 recall / NDCG / AUC 表、最终 release note、线上服务 ID。

## 1. 可插入 CREM 的 30 秒口径

我本地的 VLM2Vec 相关工作不是单纯复现官方 VLM2Vec，而是把 VLM-to-embedding 的思路迁移到推荐域 goods unified embedding。具体做法是在 Qwen2.5-Omni / Qwen-VL 系列上训练少量 compression tokens，用 `retrieval_goods` 对比学习做商品/视频/图文 pair 表征，同时混入 OpenQA/SID 压缩生成任务，让压缩 token 承载更高密度的多模态语义。

工程上我探索过多槽位 embedding、`slot_aligned` / `maxsim_topk` 相似度、orthogonal loss 防 slot collapse、Matryoshka 低维截断、cached projector 低维输出，以及 Better SID / XSeq 这种用生成 loss 反向约束 embedding 相似度的路线。它和 CREM 的关系是: CREM 是业务上线口径；本地 VLM2Vec/Goods 线提供了模型适配、压缩表征、训练和评测工具链。

## 2. 2 分钟口径

这个项目的基础是 VLM2Vec: 把多模态大模型训练成固定向量 embedding，而不是只做生成。但我们的本地目标更偏推荐业务，不是公开 MMEB benchmark。

主线有两部分。第一条是 goods unified embedding。输入包括商品、商品图、短视频/直播相关多模态信息，训练任务包括 `retrieval_goods` 和 `openqa`。`retrieval_goods` 负责把 item/item、item/photo 等正 pair 拉近；OpenQA/SID 任务负责让少量 compression tokens 能承载可生成、可还原的多模态信息。典型配置是 4 个 compression tokens，`VLM_COMPRESS_TOKEN_MODE=distinct`，`TASK_TYPES="retrieval_goods openqa"`，检索 batch 和 QA batch 按权重混训。

第二条是 Better SID / XSeq。它不是推理时依赖外部样本，而是在训练期把 batch 内或 memory bank 中的相似样本 hidden slots 注入到 OpenQA/SID 生成路径里，让 token-level generation loss 通过 `sample_gate` / `sample_cos` 等相似度权重回传到当前样本 embedding 和 slot 表征。目标是让生成任务也给 embedding 学习提供梯度，而不只是靠 InfoNCE。

我会强调边界: 当前本地 repo 已经有候选 checkpoint、encode-only embedding、cached projector smoke 和部分 MMEB ret+qa 对照实验，但还缺最终 release note 和统一业务指标表。因此面试里可以讲“技术探索和工程实现”，但线上收益仍优先使用 CREM 正式简历口径。

## 3. 技术链路

| 环节 | 本地项目实现 | 面试要点 |
|---|---|---|
| Backbone | Qwen2.5-Omni / Qwen-VL 系列 | 不是只用 CLIP encoder；可覆盖图文、视频、音频/OCR/ASR 等业务输入 |
| Compression tokens | goods 主线多用 4 个 token，`distinct` 模式 | 不说无损压缩；说任务导向的信息瓶颈 |
| 检索任务 | `retrieval_goods` | item/item、item/photo、goods pair 对比学习 |
| 生成任务 | `openqa` / SID | 用 QA LM loss 逼 compression tokens 保留任务相关语义 |
| 混训 | `TASK_TYPES="retrieval_goods openqa"`，`DATA_WEIGHTS="1 4"`，`LOSS_WEIGHTS="1 0.5"` | 读数据比例和 loss 权重分开 |
| 相似度 | `dense`、`maxsim_topk`、`slot_aligned` | 多槽位 embedding 不等于单向量 dense |
| 辅助约束 | orthogonal / volume / HSIC diagnostics | 主要防 slot collapse、分析槽位冗余 |
| 低维输出 | cached projector / `multi_head_mlp` | 先缓存 4 x 2048 slot，再训练 256/128/64 或 64/32/16 head |
| XSeq | memory bank top-k external slots 注入 OpenQA forward | 训练期相似度梯度，不是推理期外部 memory 依赖 |
| 蒸馏 | openqa-only nat teacher -> comp student logits KL | 不动 retrieval / GradCache 分支 |

## 4. Goods unified embedding

### 4.1 任务定义

Goods 线目标是给商品、商品图、商品描述、短视频/直播相关内容训练统一 embedding，用于 item/item、item/photo 等召回或粗排前置表征。

典型训练入口:

```bash
cd /mmu_mllm_hdd/zhangchenghao05/code/vlm2vec
bash experiments/hetu/train_goods_0402_twostage.sh <NNODES> <NODE_RANK> \
  STAGE=full \
  VARIANT=slotalign_ortho \
  TOTAL_COMPRESS_TOKENS=4 \
  VLM_COMPRESS_TOKEN_MODE=distinct
```

0429 curated 主线分两阶段:

- Stage1: `IA_Pair_0429` retrieval + goods/item/photo-item 相关 OpenQA。
- Stage2: `IA_Pair_0429`、`LF_Pair_0429`、`XL_Pair_0429`、`LF_I2P_Pair_0429` 四源 retrieval + 同域 OpenQA。
- 明确排除泛短视频理解数据 `Hetu_Videos_Compress_*` 和纯主播画像 `ParallelSID_Author_*`，避免 QA 分布偏离 goods 检索目标。

### 4.2 数据口径

当前能确认的 retrieval 源:

- `IA_Pair_0429`
- `LF_Pair_0429`
- `XL_Pair_0429`
- `LF_I2P_Pair_0429`
- `Goods_Strong300`

OpenQA/SID 源包括:

- `Hetu_Goods_Compress_seed`
- `Hetu_Goods_Compress_self`
- `Hetu_Goods_Compress_gpt_caption_short`
- `Hetu_Goods_Compress_gpt_caption_long`
- `ParallelSID_Item_*`
- `ParallelSID_Photo_Item_*`
- `ParallelSID_Photo_Author_Item_*`
- `ParallelSID_Photo_50K_READY`

面试安全说法:

> 数据上不是只拿公开 image-text pair，而是推荐域真实 pair 和同域 QA/SID 数据混训。retrieval 负责推荐相似度，OpenQA/SID 负责让压缩 token 承载可解释的商品/视频语义。当前本地文档还缺每份数据的最终规模、过滤规则和 fixed split，所以具体规模不要临场硬报。

### 4.3 Compression token 和 mask 口径

本地 goods 主线常用:

```bash
TOTAL_COMPRESS_TOKENS=4
VLM_COMPRESS_TOKEN_MODE=distinct
```

解释:

- `repeat_first`: 4 个位置重复同一个 `<SP0>` token。
- `distinct`: 使用 `<SP0><SP1><SP2><SP3>`，让不同 slot 更容易分工。
- 当前推荐口径是 4 个 distinct compression tokens。

回答方式:

> 这里的 compression tokens 是任务导向的有损压缩，不是无损记忆所有输入。检索 loss 让它们对推荐相似度有用，OpenQA loss 让它们保留可生成的多模态语义，orthogonal loss/几何诊断用来避免 4 个 slot collapse 成同一个向量。

### 4.4 Pooling / similarity 变体

| Variant | Pooling | Similarity | 语义 |
|---|---|---|---|
| `mean` | `avg` | `dense` | 4 个 token 平均成单向量 |
| `maxtop4_ortho` | `none` | `maxsim_topk` | 多槽位 token pair top-k 相似度 |
| `slotalign_ortho` | `none` | `slot_aligned` | 第 i 个 slot 对第 i 个 slot，平均相似度 |
| `attn + dense` | `attn` | `dense` | attention pooling 后输出单向量 |
| `multi_head_mlp` | projector | dense / slot-aligned | 多个低维 head 并行训练 |

不要混淆:

- `POOLING=none` 通常是多 token 输出，不能当单向量 embedding。
- `slot_aligned` 是多槽位服务形态；线上如果只支持单向量 ANN，需要 projector 或 pooling。
- `multi_head_mlp` 是独立多 head projector；MRL/Matryoshka 是共享向量前缀截断，不是一回事。

## 5. 结果与候选产物

### 5.1 0429 goods 候选 checkpoint

本地总结中确认过三类候选:

| 候选 | 形态 | 路径口径 |
|---|---|---|
| `slot_aligned` 多槽位 | `pooling=none`, `similarity_type=slot_aligned` | `goods_0429_distinct_slotalign_ortho_retqa.../stage2/checkpoint-4000` |
| `attn + dense` 单向量 | `pooling=attn`, `similarity_type=dense` | `goods_0429_distinct_attnpool_ortho_retqa.../stage2/checkpoint-4000` |
| `attn + dense 64d` | attention pooling + low dim | `goods_0429_distinct_attnpool_ortho_retqa...s2-4000/stage2/checkpoint-4000` |

安全边界:

> 这些是已完成训练/推理的候选，不要说成最终 release checkpoint。当前还缺统一全量 recall / NDCG / AUC 表和业务 release note。

### 5.2 0402 SP padding 几何分析

历史几何诊断结论:

- `repeat0_ia checkpoint-400`: sample-level similarity 最低，`pooled=0.1328`、`maxsim=0.3984`。
- `distinct_ia checkpoint-400`: token decorrelation 最强，`same-token=0.0043`、`internal-max=0.0351`。
- `repeat0_4src checkpoint-200`: 明显 collapse，`same-token=0.3781`、`internal-max=0.9954`。
- `repeat0_4src checkpoint-400`: within-sample collapse 修复，但 sample-level similarity 仍偏高，`pooled=0.2875`、`maxsim=0.5967`。

面试说法:

> 这组不是业务指标，而是 embedding 几何诊断。它说明 token 设计和数据混合会影响 slot collapse，因此后来更倾向 distinct token 和正交约束。

### 5.3 MMEB ret+qa vs orthogonal 对照

本地 `docs/plan/2026-06-03/20260603_mmeb_retqa_2machine8g_results.md` 记录了一组公开 MMEB image 任务对照:

| setting | Classification | VQA | Retrieval | VG | IND | OOD | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ret+qa | 0.6506 | 0.6152 | 0.6825 | 0.7870 | 0.7020 | 0.6222 | 0.6666 |
| ret+qa+orthogonal | 0.6484 | 0.6180 | 0.6825 | 0.7838 | 0.7011 | 0.6229 | 0.6664 |

结论口径:

> 这组说明 orthogonal loss 在 MMEB image overall 上基本持平，不是一个必然涨点的 magic trick。它对 VQA/OOD 有小幅正向，对 Classification/VG 有小幅负向。goods 场景里使用 orthogonal 更主要是为了 slot 几何和 collapse 控制，仍需看推荐域指标。

## 6. Cached projector / 低维输出

当前 `vlm2vec` 已恢复 cached projector 三段链路:

1. `build_cache`: 主模型无梯度推理 retrieval pair，落盘 `rankNNN_shardXXXXXX.pt`，内容包括 `qry`、`tgt`、metadata。
2. `train`: 从 cache shard 读取 `B x 4 x 2048` slot embedding，训练 `multi_head_mlp` 低维 projector。
3. `apply`: 把 projector 应用到已推理 embedding pickle，导出低维 head。

恢复文件:

- `src/cached_projector/build_cache.py`
- `src/cached_projector/train.py`
- `src/cached_projector/apply.py`
- `experiments/hetu/cached_projector/build_goods_cache.sh`
- `experiments/hetu/cached_projector/train_cached_projector.sh`
- `experiments/hetu/cached_projector/apply_cached_projector.sh`

smoke 结果:

- 从历史 cache 训练 1 step projector，识别 `token_count=4`、`hidden_dim=2048`。
- 输出 `head_64`、`head_32` standalone heads。
- 从新构建 cache 训练 `HEAD_DIMS='16 8'`，`slot_aligned` smoke 中 `Acc@1#16=60.0`、`Acc@1#8=40.0`，这只是 1-step smoke，不是正式指标。

面试说法:

> 低维 projector 是为服务形态准备的。多槽位 4 x 2048 表达能力强，但线上 ANN 和存储成本高；cached projector 先把大模型特征缓存下来，再快速训练 256/128/64 或 64/32/16 维 head，用来比较效果和服务成本。这个链路也让低维 head 的迭代不必每次重跑大模型训练。

## 7. Better SID / XSeq

### 7.1 目标

Better SID / XSeq 不是推理期 retrieval-augmented generation。它的目标是在训练期把“相似样本应该互相可借鉴”这个结构加入生成路径，让 OpenQA/SID token loss 通过相似度权重回传到 embedding / slot。

直观说:

- InfoNCE 只直接监督 embedding 相似度。
- OpenQA 只监督压缩 token 能不能支持生成。
- XSeq 试图把二者连接起来: 如果样本 A 的 slot 能帮助样本 B 生成，那么 A/B 的 embedding 或 slot 相似度也应该被奖励。

### 7.2 当前实现

关键文件:

- `/mmu_mllm_hdd/zhangchenghao05/code/vlm2vec-better-sid/src/xseq.py`
- `/mmu_mllm_hdd/zhangchenghao05/code/vlm2vec-better-sid/src/trainer.py`
- Qwen2.5-Omni attention 注入: `src/model/vlm_backbone/qwen2_5_omni/modeling_qwen2_5_omni.py`

流程:

1. 当前 batch 正常 forward，得到 local loss、pooled embedding、slot tokens、指定层 hidden slots。
2. 从本地 memory bank 中按 dense 或 `slot_aligned` 相似度取 top-k 外部样本。
3. 构造 `xseq_state`: `sample_gate`、`sample_cos`、`external_slots`、`token_mask`、`cls_indices`。
4. 在指定 decoder layers 注入 external slots attention，再 forward 一次。
5. 记录 `xseq/loss_local_only`、`xseq/loss_with_xseq`、`xseq/sample_gate_mean`、`xseq/layer_*_hsic_mean` 等诊断。
6. 当前 batch 的 embedding/slot 写入 memory bank，作为后续样本 stop-grad context。

当前支持 token scope:

- `c` / `chorus` / `slot`
- `q` / `question` / `query`
- `a` / `answer`
- `none`

常见脚本:

- `train_musid_xseq_ret_openqa_3task.sh`
- `train_musid_xseq_openqa_2task.sh`
- `train_musid_baseline_ret_openqa_ortho1.sh`
- `train_musid_multitask.sh`

### 7.3 限制

- 当前是本地进程 memory-bank 版本，不是严格 BxB in-batch attention。
- bank entries 是 stop-grad，top-k 选择离散，梯度主要回到当前 query 分支和 `sample_gate/sample_cos` 路径。
- 多一次 local forward + xseq forward，训练成本高。
- 目前还缺正式 retrieval / embedding 指标表，不能只看 xseq loss 是否下降。

面试安全说法:

> XSeq 是我探索的 research 线，动机是让生成任务也给 embedding 相似度提供梯度。但当前版本还需要正式 ablation 验证它是否提升最终静态 embedding；不能只凭训练期 xseq loss 下降下结论。

## 8. OpenQA nat->comp 蒸馏

本地 `蒸馏实现文档.md` 记录的设计:

- 只蒸馏 `openqa`，不影响 retrieval / GradCache。
- Student 用 comp，`prob=0.0`，压缩生效。
- Teacher 用 nat，`prob=1.0`，跳过压缩。
- 蒸馏损失用答案段 logits KL。

公式:

```text
L_total = L_openqa_ce + lambda * KL(logits_comp, logits_nat)
```

关键实现点:

- collator 同时产出 `comp` 和 `nat` dict。
- trainer 在 openqa 分支分别 forward student 和 teacher。
- 只在 `labels != -100` 的答案段算 KL。
- 建议前 N step 开启 comp/nat answer token 对齐断言。

面试说法:

> 这个蒸馏不是为了让 retrieval 直接学 teacher，而是让压缩后的 OpenQA 输出接近不压缩的 nat teacher。它只插在 openqa 分支，避免破坏 retrieval 的 GradCache 流程。

## 9. MRL / Matryoshka

本地 `vlm2vec` 支持:

```bash
--matryoshkas 2048 1024 512 256 128 64 32 16 8
```

实现方式:

- 完整维度先算 contrastive loss。
- 对每个维度 `dim` 做 `x[..., :dim]`、`y[..., :dim]`、`neg[..., :dim]` 前缀截断。
- 每个维度都累加 CE loss。
- eval 时按维度截断 embedding 后重算检索指标。

面试边界:

- MRL 是共享前缀维度的低维兼容训练。
- `multi_head_mlp` 是独立多 head projector。
- 两者都服务低维输出，但工程形态和参数共享方式不同。

## 10. 和 CREM 主项目怎么连

建议把这段作为 CREM 的补充，不要独立抢主项目位置:

> CREM 是业务主线，强调推荐粗排/召回、LLM judge 样本构造、Qwen2.5-Omni 适配、InfoNCE + OpenQA 辅助训练和线上 embedding 服务。VLM2Vec/Goods 本地项目是这个方向的工程和研究沉淀，里面包含多槽位 embedding、compression token、cached projector、XSeq 等探索。面试时我会把它作为支撑 CREM 的技术细节展开，而不是另起一个完全无关项目。

如果被问“你到底做了什么”:

> 我参与的是模型适配、训练 pipeline、compression token/attention mask、检索与 OpenQA 混训、slot/pooling/similarity 变体、低维 projector 和 case/几何分析。XSeq/Better SID 这条线更偏研究探索，我会明确它还缺正式业务指标闭环。

## 11. 高频追问

### Q1: 这个和官方 VLM2Vec 是什么关系?

答: 官方 VLM2Vec 是通用多模态 embedding benchmark 和训练框架，核心是 instruction-guided contrastive learning。本地项目借鉴的是“VLM 可以转成 embedding model”的方向，但问题设定是推荐域 goods unified embedding，数据、负样本、OpenQA 压缩监督和服务形态都不同。

### Q2: 为什么需要 OpenQA，检索 loss 不够吗?

答: 检索 loss 只告诉模型哪些 pair 近，监督比较稀疏，而且推荐 pair 有噪声。OpenQA 让 compression tokens 必须保留能回答问题的多模态语义，相当于给 embedding 一个更密集的语义约束。它不是直接优化 AUC，而是提升表示质量和训练稳定性。

### Q3: 为什么 4 个 compression tokens?

答: 当前 goods/musid 主线多用 4 个，是效果、显存和服务形态之间的折中。历史 public/MMEB 或其他配置可能用 16，所以面试里要说“本地 goods 主线默认 4 个 distinct tokens”，不要泛化成所有项目都 4 个。

### Q4: `slot_aligned` 比 mean pooling 好在哪里?

答: mean pooling 把 4 个 slot 压成一个向量，简单但可能丢掉不同语义槽位分工。`slot_aligned` 保留 4 个 slot，按位置对齐计算相似度，可以表达多方面语义。但服务复杂度更高，所以需要 low-dim projector 或单向量候选做业务折中。

### Q5: Orthogonal loss 有没有明确涨点?

答: 不能说必然涨点。MMEB 对照里 overall 基本持平。它更像是 slot 几何约束，用来降低 collapse 和冗余；最终是否提升推荐指标还要看 goods 全量 eval。

### Q6: XSeq 是不是推理时要查 memory?

答: 不是默认推理方案。当前目标是在训练期通过相似样本 attention 给 embedding/slot 提供额外梯度，最终仍希望服务静态 embedding。当前实现用了 local memory bank，但这是训练机制，不是线上依赖。

### Q7: cached projector 为什么不是直接端到端训练?

答: 端到端训练每次都要跑大模型，成本高。cached projector 先把 4 x 2048 slot 缓存下来，再训练低维 head，可以快速比较 256/128/64 或 64/32/16 等输出维度，适合服务侧成本/效果折中。

### Q8: 当前最大缺口是什么?

答: 最大缺口是结果闭环: 固定 eval 数据、固定 checkpoint、统一 recall/NDCG/AUC 表、选 release candidate、补服务形态和 release note。现有代码和候选产物不少，但不能把候选探索直接说成最终线上结论。

## 12. 官方 VLM2Vec related work 速记

官方 VLM2Vec:

- 论文: <https://arxiv.org/abs/2410.05160>
- 项目页: <https://tiger-ai-lab.github.io/VLM2Vec/>
- 官方仓库: <https://github.com/TIGER-AI-Lab/VLM2Vec>

核心事实:

- VLM2Vec 是 ICLR 2025 工作，把 VLM 训练成通用多模态 embedding model。
- MMEB v1 覆盖 36 个数据集、4 个 meta-task: classification、VQA、retrieval、visual grounding。
- 训练方式是 instruction-guided contrastive learning，query/target 可以是 text、image、image+text。
- 论文主表中 LoRA bs=1024 的 VLM2Vec overall Precision@1 为 60.1，OOD 为 52.0。
- VLM2Vec-V2 扩到 video 和 visual document。公开项目页、论文摘要和本地 README 对任务/数据集总数口径有过差异，面试中不主动硬报总数；只说它把评测扩展到视频、视觉文档、temporal grounding、video QA 等方向。

和 CREM 的对比:

| 维度 | 官方 VLM2Vec | 本地 Goods / CREM 线 |
|---|---|---|
| 目标 | 通用多模态 embedding benchmark | 推荐粗排/召回 goods embedding |
| 数据 | MMEB 公开多任务 | 用户交互 pair、goods/OpenQA/SID 同域数据 |
| Backbone | Phi-3.5-V、LLaVA、Qwen2VL 等 | Qwen2.5-Omni / Qwen-VL 系列 |
| 训练 | InfoNCE + in-batch negatives | retrieval_goods + OpenQA/SID + orthogonal/projector/XSeq 探索 |
| 指标 | MMEB Precision@1 | 推荐离线 AUC/召回指标、线上业务指标 |

不要说:

- “我参与过官方 VLM2Vec。”
- “官方 VLM2Vec 的 MMEB 数字就是 CREM 效果。”
- “本地 goods 线已经有最终上线 release”，除非后续补齐 release note 和业务记录。
