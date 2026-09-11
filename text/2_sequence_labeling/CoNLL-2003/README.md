# CoNLL-2003：BiLSTM、Transformer 与 BERT 命名实体识别实验

> 结果更新：2026-09-10。当前共 **11 组完整训练实验**：8 组 GloVe 系列（20 epoch）和 3 组 BERT 微调（4 epoch），均为 seed=42。
> 8 组 GloVe 系列已评估完整 train/dev/test；2026-09-10 新增 3 组 BERT 的完整 dev 复核和 test 评估，**三组 dev F1 均精确重现历史最佳值**。BERT 尚未补跑 eval 模式下的 train F1。
> 此前按 dev 选定的模型仍是 **BERT-base-cased，峰值 LR=5e-5：dev F1 95.01%，test F1 91.56%**。相比 GloVe 系列最好的 BiLSTM（test F1 83.16%），同一 test 上提高 **8.41pp**；未根据 test 重新挑选模型。

这是词级 NER，而不是整句分类：每个原始词预测一个 BIO2 标签。LSTM 和小 Transformer 共用 GloVe、大小写特征和逐词线性分类头，编码器随机初始化；BERT 使用配套 WordPiece tokenizer 和预训练编码器，提取每个词首个子词的特征，仍输出词级 `[B,W,9]`，共用实体评分逻辑。

## 1. 全部实验结果

### 1.1 主结果

下表 P/R/F1 均为**完整实体精确匹配**的百分数，F1 默认指 micro-F1；`pp` 表示百分点。每个 checkpoint 都由训练时的 **dev micro-F1** 选出，test 只用于报告，不用于挑 epoch 或决定下一轮超参。

| 实验 / 原始评估记录 | 配置差异 | 总参数 | 最佳 epoch | Dev F1 | Test F1 | Test P | Test R |
|---|---|---:|---:|---:|---:|---:|---:|
| [LSTM](reports/2026-09-08/outputs_lstm.json) | 双向 H=256，2 层，dropout=0.5 | 4.449M | 18 | 87.72 | 83.16 | 81.39 | 85.00 |
| [LSTM-no-case](reports/2026-09-08/outputs_lstm_nocase.json) | 去掉大小写特征 | 4.416M | 20 | 79.42 | 73.90 | 71.17 | 76.84 |
| [LSTM-no-GloVe](reports/2026-09-08/outputs_lstm_noglove.json) | 随机词向量，stage 1=0 / stage 2=20 | 4.449M | 17 | 76.51 | 72.44 | 71.15 | 73.78 |
| [Transformer](reports/2026-09-08/outputs_transformer.json) | D=128，4 头，2 层，Pre-LN，dropout=0.1 | 2.514M | 17 | 86.22 | 79.97 | 77.47 | 82.63 |
| [Transformer-big](reports/2026-09-08/outputs_trf_big.json) | D=256，4 层，其余同基线 | 5.293M | 18 | 86.42 | 80.79 | 78.37 | 83.37 |
| [Transformer-d03](reports/2026-09-08/outputs_trf_d03.json) | dropout=0.3 | 2.514M | 19 | 81.64 | 74.55 | 71.16 | 78.28 |
| [Transformer-d05](reports/2026-09-08/outputs_trf_d05.json) | dropout=0.5 | 2.514M | 18 | 74.22 | 67.89 | 63.42 | 73.03 |
| [Transformer-post](reports/2026-09-08/outputs_trf_post.json) | Post-LN，其余同基线 | 2.514M | 17 | 85.46 | 78.79 | 75.92 | 81.89 |
| [BERT-LR2e-5](outputs_bert_lr2e-5/eval_test.json) | bert-base-cased，全量微调，峰值 LR=2e-5 | 107.727M | 4 | 94.73 | 90.93 | 90.07 | 91.82 |
| [BERT-LR3e-5](outputs_bert/eval_test.json) | 默认配置，峰值 LR=3e-5 | 107.727M | 3 | 94.92 | 91.16 | 90.35 | 91.98 |
| [BERT-LR5e-5](outputs_bert_lr5e-5/eval_test.json) | 峰值 LR=5e-5 | 107.727M | 4 | **95.01** | **91.56** | **90.79** | **92.35** |

主表统一列出**总参数量**；旧实验的非 embedding 参数仍保留在原报告及下文分析中。8 组 GloVe 系列的完整精度、配置、训练时长、混淆矩阵和文件 SHA-256 见 [summary.json](reports/2026-09-08/summary.json)，该历史快照不含 BERT。BERT 新评估记录为各运行目录的 `eval_test.json`，包含完整精度的 `evaluations.valid/test`、训练元数据、命令、环境和文件 SHA-256；原 `training_log.json` 保持不变。表格按实验身份排列，不按 test 分数重新选模型。

### 1.2 全部模型：Test 分实体类型与标签合法性

| 实验 | PER F1 | LOC F1 | ORG F1 | MISC F1 | Macro F1 | Token accuracy | 非法 I- 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 85.37 | 88.48 | 80.52 | 71.49 | 81.46 | 96.89 | 114 |
| LSTM-no-case | 71.84 | 84.10 | 67.94 | 69.30 | 73.29 | 94.20 | 137 |
| LSTM-no-GloVe | 77.48 | 76.84 | 68.29 | 58.82 | 70.36 | 94.86 | 162 |
| Transformer | 80.31 | 87.74 | 76.78 | 68.68 | 78.38 | 96.50 | 294 |
| Transformer-big | 81.59 | 87.70 | 77.69 | 70.17 | 79.29 | 96.60 | 251 |
| Transformer-d03 | 70.45 | 83.91 | 72.34 | 69.92 | 74.15 | 95.64 | 469 |
| Transformer-d05 | 62.34 | 79.34 | 64.80 | 65.23 | 67.93 | 94.27 | 950 |
| Transformer-post | 77.34 | 87.27 | 76.64 | 68.44 | 77.42 | 96.31 | 253 |
| BERT-LR2e-5 | 96.35 | 92.96 | 89.01 | 78.76 | 89.27 | 98.20 | 48 |
| BERT-LR3e-5 | 96.12 | 93.24 | 89.15 | 80.08 | 89.65 | 98.24 | 45 |
| BERT-LR5e-5 | 95.86 | 93.44 | 90.21 | 80.77 | 90.07 | 98.30 | 46 |

非法 I- 数统计不符合 BIO2 的位置，例如句首 `I-PER`、`O → I-PER`。分母是 46,435 个真实 test token，不是实体数。它是解码诊断，**不等于可被 CRF 挽回的 F1**。

### 1.3 BERT 与主要基线：Dev 分类型对比

以下均为各自 dev 最佳 checkpoint，分母是相同的 5,942 个实体 / 51,362 个真实词；不要与上一张 test 表混用。

| 模型 | Dev P | Dev R | PER F1 | LOC F1 | ORG F1 | MISC F1 | 非法 I- 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 86.65 | 88.81 | 89.86 | 92.97 | 82.74 | 79.66 | 98 |
| Transformer | 84.29 | 88.24 | 86.63 | 92.60 | 80.50 | 80.98 | 271 |
| BERT-LR2e-5 | 94.30 | 95.17 | 97.41 | 96.47 | 92.50 | 89.26 | 37 |
| BERT-LR3e-5 | 94.51 | 95.34 | 97.33 | 96.60 | 93.13 | 89.47 | 35 |
| BERT-LR5e-5 | 94.64 | 95.39 | 97.32 | 96.86 | 92.68 | 90.20 | 23 |

## 2. 实验说明了什么

### 2.1 GloVe 系列中 BiLSTM 最好，但不是架构的普遍排名

在前 8 组 word-level 实验中，BiLSTM 相比小 Transformer，dev 高 **1.50pp**；相比大 Transformer，dev 高 **1.30pp**。新增 BERT 已超过这些结果，旧基线仍保留用于教学对照。

需要区分拟合能力和泛化。以下是在同一个最佳 checkpoint 上关闭 dropout 后重新评估的结果，不是训练循环里一边更新权重一边统计的 loss：

| 模型 | Train F1（eval 模式） | Dev F1 | Train−Dev |
|---|---:|---:|---:|
| LSTM | 93.09 | 87.72 | 5.38pp |
| Transformer | 96.55 | 86.22 | 10.33pp |
| Transformer-big | 97.12 | 86.42 | 10.71pp |
| LSTM-no-GloVe | 81.06 | 76.51 | 4.55pp |

**观察：随机初始化的 Transformer 编码器已能更好地拟合训练集，却没有获得更好的 dev 表现。** 因此，这组结果不支持把主要问题简单归结为“容量不够”。局部顺序建模、词表示、优化策略等都可能有关，尚未逐项隔离。BERT 尚未补跑 eval 模式下的 train F1，不能用它的训练 loss 填入此表。

这也不是严格的等参数比较：默认 LSTM / Transformer 总参数为 **4.449M / 2.514M**，非 embedding 参数相差约 5.7 倍，dropout 和计算方式也不同。各模型都只有一次训练，不能据此宣称 BiLSTM 在 NER 上普遍优于 Transformer。

### 2.2 大小写特征有明显价值，不只是装饰

去掉 case 后，LSTM 的 dev F1 降 **8.30pp**，test 降 **9.26pp**。从 test 分类型看，PER 和 ORG 的下降尤其明显。

dev 错误诊断也支持这个方向：没有任何预测实体与 gold 重叠的漏检，由 **43 → 311**；边界重叠但未精确匹配的 gold 实体由 **253 → 439**。

结论限定为：**本实现、本次训练中，保留大小写有明显收益**。case 表本身只有 `7×16=112` 个参数，但去掉它也使 LSTM 输入从 116 维变为 100 维，连同第一层权重共减少 32,880 个参数；这不是严格保持参数布局不变的实验。Transformer 的 no-case 实验尚未做，不能直接把这组差值移植过去。

### 2.3 No-GloVe 降幅大，但不是干净的初始化消融

LSTM-no-GloVe 的 dev 比基线低 **11.21pp**，test 低 **10.72pp**。它的 train F1 也只有 **81.06%**，说明在当前训练配置下，训练集拟合本身就明显落后，不能把差距全部解释为“新名字泛化失败”。

这组实验确实跳过了冻结阶段，但同时改变了优化过程：

- 基线先训练 10 epoch：encoder/head/case 学习率从 `1e-3` 开始，再解冻微调。
- No-GloVe 直接进入 20 epoch 的 stage 2：encoder/head 从 `3e-4` 开始，随机 word/case embedding 从 `5e-5` 开始。

因此它测到的是**现有 no-GloVe 训练方案与基线方案的差异**，不是预训练知识的独立因果贡献。随机向量也能学到任务特征，“这个规模几乎学不到东西”不成立；应先补一组适合从零训练的学习率实验。

### 2.4 扩容、dropout、LayerNorm：哪些尝试有效

- **扩容收益小**：非 embedding 参数从 0.413M 增到 3.192M，约 7.7 倍；dev 只提高 **0.20pp**。这是同时改变宽度和深度的一组结果，不足以确认提升可靠，也不足以证明扩容永远无效。
- **统一提高 dropout 明显变差**：`0.1 → 0.3 → 0.5` 时 dev 为 `86.22 → 81.64 → 74.22`。最高 dropout 的 train F1 也只有 82.19%，与过强正则化或当前预算下拟合不足相符。这里一个参数同时控制输入、注意力、FFN、残差和 head 多处 dropout，不能解读成某一处 dropout 的独立效果。
- **Pre-LN 暂时保留**：比 Post-LN 的 dev 高 **0.76pp**。两者第一个 epoch 的 dev F1 已分别为 71.56% / 70.94%，没有出现“Post-LN 完全不学习”。小差距需要多 seed；本项目的 Pre-LN 还包含栈末 LayerNorm。

历史日志的平均 epoch 时间：LSTM 9.03 秒、小 Transformer 2.96 秒、大 Transformer 3.59 秒。它们不是受控性能基准，不能直接当作稳定加速比；2026-09-08 的重新评估没有测训练吞吐。

### 2.5 BERT 全量微调：提升主要体现在误报、漏报一起减少

默认 `3e-5` 的 BERT 仅第一轮就达到 **93.48% dev F1**，超过所有旧基线；最佳 94.92% 比 LSTM 高 **7.21pp**。学习率消融中的最佳 `5e-5` 达到 **95.01%**，比 LSTM / 小 Transformer / 大 Transformer 分别高 **7.30 / 8.79 / 8.60pp**。

同样的 dev 实体上，LSTM → 默认 BERT → BERT-LR5e-5 的 TP 为 **5,277 → 5,665 → 5,668**，FP 为 **813 → 329 → 321**，FN 为 **665 → 277 → 274**。提升并不是单靠多预测实体换取召回。默认 BERT 的 PER precision 从 LSTM 的 83.37% 提高到 96.58%，recall 仍达到 98.10%；ORG 和 MISC 的 F1 也明显改善。

这是**完整训练方案的对比**，不是等参数、等预训练资源的架构消融：旧模型仅 GloVe 词向量经过预训练，BERT 则有预训练的 12 层编码器、WordPiece 和更大的参数量（107,726,601，约为 LSTM 的 24 倍）。不能把全部差距归因于架构、预训练或子词中的某一个因素。

默认 BERT 的四轮训练加逐轮 dev 评估合计 **68.2 秒**（按日志 `time_sec` 求和），不含预训练成本及首次下载；与旧实验并非受控速度测试。

### 2.6 BERT 学习率消融：5e-5 暂时领先，差距很小

三组日志中的模型、数据、seed、batch、训练轮数及其他优化设置一致，仅**全局峰值学习率**不同。各轮 dev F1 为：

| Epoch | LR=2e-5 | LR=3e-5 | LR=5e-5 |
|---|---:|---:|---:|
| 1 | 92.85 | 93.48 | 93.56 |
| 2 | 94.33 | 94.40 | 94.27 |
| 3 | 94.68 | **94.92** | 94.78 |
| 4 | **94.73** | 94.79 | **95.01** |

- **按本次 dev F1 选择 `outputs_bert_lr5e-5/best.pt`**，但它只比 `3e-5` 高 **0.09pp**、比 `2e-5` 高 **0.28pp**。每组只有 seed=42，不能据此认定稳定排名；代码默认值仍保留 `3e-5`。
- `2e-5` 初期适应较慢；`5e-5` 没有明显训练失败，但也不是每一轮都最好。当前结论只适用于这套四轮预算与 warmup/衰减调度。
- 训练 loss 的最后一轮值为 **0.0114 / 0.0075 / 0.0050**，下降幅度并不等于泛化收益。`5e-5` 的 dev loss 从第 2 轮的 **0.0366** 上升到第 4 轮的 **0.0397**，F1 却继续提高；交叉熵关注概率，实体 F1 关注最终标签与边界，选择指标仍是 dev F1。
- 默认 `3e-5` 在第 4 轮 train loss 继续下降、dev loss 上升、F1 略降，是轻微过拟合的迹象，不代表严重不稳定，也不支持盲目增加到 10～20 轮。
- 三组均同时改变 backbone 和 head 的 LR，**没有验证“head 单独用更大 LR”或“先冻结 backbone”**。不能将小幅提升直接解释为 head 学得更快或 backbone 受到了保护。

### 2.7 补跑 Test：BERT 的提升保留下来了，但学习率排名仍需谨慎

评估的是各自**已由 dev 选定**的 checkpoint：`2e-5 / 3e-5 / 5e-5` 分别为第 `4 / 3 / 4` 轮，未重新训练，也未用 test 选择 epoch。

- **BERT 的优势不仅出现在 dev。** 此前 dev 选中的 `5e-5`，test F1 为 **91.56%**，比 LSTM / 小 Transformer / 大 Transformer 分别高 **8.41 / 11.60 / 10.77pp**；三个 BERT 配置均超过旧基线。仍应理解为完整训练方案的对比，而非纯架构优势。
- **精确命中增多，同时减少误报和漏报。** 在相同的 5,648 个 test 实体上，LSTM → BERT-LR5e-5 的 TP 为 **4,801 → 5,216**，FP 为 **1,098 → 529**，FN 为 **847 → 432**。不是靠大幅增加预测实体数换取召回。
- **学习率顺序在本次 test 上一致，但差异不大。** `2e-5 / 3e-5 / 5e-5` 的 test F1 为 **90.93 / 91.16 / 91.56%**；`5e-5` 比 `3e-5` 高 **0.41pp**，比 `2e-5` 高 **0.63pp**。这是单 seed、同一测试集的描述性结果，不证明跨 seed 的稳定排名；不因 test 改默认值或追加测试集调参。
- **Test 仍明显低于 dev。** 三组 Dev−Test 分别为 **3.80 / 3.77 / 3.45pp**。这说明 dev 数字不能代替最终 test 表现；语料分布差异、开发集选择效应等可能影响差距，不能只归因于某一个原因。
- **MISC 仍是相对弱项。** `5e-5` 的 test PER/LOC/ORG/MISC F1 为 **95.86 / 93.44 / 90.21 / 80.77%**；它也并非每类都胜出，`2e-5` 的 PER F1 更高。后续可在 dev 上分析类型混淆及边界错误，不根据 test 错例定制规则。46 个非法 I- 只是一项诊断，不代表加入 CRF 就能自动挽回全部错误。

## 3. 错误在哪里：先用 dev 决定下一步

### 3.1 GloVe 系列的未登录词仍是弱点，但不等于所有新实体

将 dev 的 5,942 个 gold 实体按是否含 `<unk>` token 分组，计算**完整实体召回率**：

| 模型 | 所有 token 都在词表（4,660 实体） | 至少一个 token 为 UNK（1,282 实体） |
|---|---:|---:|
| LSTM | 92.96% | 73.71% |
| LSTM-no-case | 87.21% | 66.46% |
| LSTM-no-GloVe | 78.78% | 70.59% |
| Transformer | 93.05% | 70.75% |
| Transformer-big | 93.76% | 69.81% |

这是固定 gold 子集的 recall，不是分组 F1，也不是“训练未见过实体名称”的评估：一个新名字可以由已知 token 组成。大 Transformer 对已知词实体稍好，但这一轮并没有改善含 UNK 实体的召回。

GloVe 路径的词表仅来自 train；不在词表里的词，即使存在于完整 GloVe 文件，也会映射到 UNK。这两种模型没有字符/子词编码，所以这些位置主要依赖上下文和大小写。

BERT 的 dev `[UNK]` word rate 日志为 **0.0000**，因为许多训练词表外的词可以拆成已有 WordPiece；这不意味着它在训练中见过所有实体。以上固定 gold 子集上的 BERT recall 尚未计算，不能用整体 F1 代替该项比较。

### 3.2 类型判断和边界都有错误，不能只看非法 BIO

以下按 gold 实体做互斥分类：先精确匹配；否则判断是否同边界错类型；再判断是否与任何预测实体重叠；最后为完全漏检。每行总和都是 5,942，**不是 FP 的分类统计**。

| 模型 | 完全正确 | 同边界、错类型 | 有重叠但边界不匹配 | 无重叠漏检 |
|---|---:|---:|---:|---:|
| LSTM | 5,277 | 369 | 253 | 43 |
| LSTM-no-case | 4,916 | 276 | 439 | 311 |
| LSTM-no-GloVe | 4,576 | 780 | 332 | 254 |
| Transformer | 5,243 | 366 | 277 | 56 |

LSTM 在 dev 上的 PER precision/recall 是 **83.37% / 97.45%**，有过多预测为人名的现象。实际 dev 错例包括 `Kent`（gold ORG）和 `ex-England`（gold MISC）被标成 PER；`Leicester`（gold LOC）被标成 ORG。这些不只是 BIO 转移是否合法的问题。

LSTM / Transformer 的 dev 非法 I- 数是 98 / 271。可以研究带约束解码，但**普通可学习 CRF 不会自动禁止非法转移**，需要显式的起始/转移约束。评分器已将孤立 `I-X` 宽容地解释为实体开头；只把它改写成 `B-X` 不改变 span，因此不会自动提高实体 F1。

## 4. 下一步方向（尚未执行）

按优先级推进，所有选择以 dev 为依据。**BERT 的评估入口及三组 test 报告已完成**，不再列为待办；后续不要用已查看的 test 结果继续调参，也不要一口气同时改容量、学习率和解码器。

| 优先级 | 要回答的问题 | 建议实验 | 判断依据 |
|---|---|---|---|
| P0 | BERT 的 0.09pp 学习率差距是否稳定？ | `3e-5 / 5e-5` 各补 seed=43/44，与已有 seed=42 汇总 | dev 最佳 F1 的均值、标准差，不只挑最高 seed |
| P1 | 随机 head 是否需要更大的 LR？ | backbone=`3e-5`、head=`1e-4`，其他条件与默认基线一致 | 先实现参数组 LR 控制；按 dev 比较，不假定一定更好 |
| P2 | 旧 word-level 模型差距是否稳定？ | 基线 LSTM、小 Transformer、大 Transformer 各做 seed=42/43/44 | dev 最佳 F1 的均值、标准差；保留为独立教学对照 |
| P1 | No-GloVe 是信息不足还是训练设置过于保守？ | 20 epoch 全程解冻；分别试 word embedding `3e-4 / 1e-3`，case/encoder/head `3e-4`；保留现有 run 作对照 | train/dev F1、类型错误；这是调优对比，不冒充纯初始化消融 |
| P1 | 词向量冻结 10 epoch 是否太久？ | 有 GloVe 的原基线改成 `3+17`，其他参数保持不变 | dev F1；目前第一阶段连未命中 GloVe 的随机词向量也一起冻结 |
| P2 | GloVe 模型能否更好处理低频词和未见词？ | 增加字符 CNN/LSTM 或子词表示，先固定一个编码器做对照 | dev 总体 F1、含 UNK 实体 recall，再补按名称是否见过的分组 |
| P2 | 边界约束能改善多少？ | 先比较独立 argmax 与 BIO2 约束 Viterbi，再考虑带显式约束的 CRF | dev 实体 F1、边界错误与非法率；类型混淆可能仍存在 |
| P3 | 基线能否更小或更简单？ | LSTM H=128；另做 H=100/1 层；`min_freq=1/2/5`；Transformer no-case | 分开改变变量，衡量质量与开销 |

实现前要注意两点：

1. GloVe 的 `train.py` CLI 没有学习率参数，word/case 共用 embedding 参数组。BERT 的 `train_bert.py` 有 `--lr`，但 backbone/head 共用 LR，现有两组仅区分是否 weight decay。上述分组 LR 对照都需要先增加相应控制；本次没有改训练代码或默认值。
2. 两个训练入口的 `--seed` 都用于全局随机种子，但 DataLoader 的 shuffle generator 仍直接取 `config.SEED`。所以现状下多 seed 会改变随机初始化/dropout，**不会改变 shuffle 种子**；若想测完整训练随机性，应先显式传递 shuffle seed，并将协议差异记录下来。

字符表示和 CRF 是值得验证的结构方向，不保证一定改善当前模型；可参考 [Lample et al. 的 NER 架构](https://aclanthology.org/N16-1030/)。小幅差异应报告分布，而非单次最高分，参见 [Reimers & Gurevych](https://aclanthology.org/D17-1035/)。BERT 分支现已完成三组全量微调，它改变了预训练资源，不再是 word-level 编码器的直接消融。

## 5. 数据、标签与模型约定

### 数据与评估

本地完整读取后的统计如下，均排除 `-DOCSTART-`：

| split | 句子 | token | 实体 | 最长句 | UNK token 比例 | 全预测 O 的 token accuracy |
|---|---:|---:|---:|---:|---:|---:|
| train | 14,041 | 203,621 | 23,499 | 113 | 0.00% | 83.28% |
| valid / testa | 3,250 | 51,362 | 5,942 | 109 | 7.54% | 83.25% |
| test / testb | 3,453 | 46,435 | 5,648 | 124 | 10.87% | 82.53% |

此表按**原始词**计数，UNK 比例属于 GloVe 路径。`MAX_LEN=128` 在这三个本地 split 上均未截断；不能通过裁掉长句或 gold 标签来改善评分。UNK 比例是 token 频次比例，不能与 GloVe 的词表类型覆盖率混淆。

BERT 仍读取相同的原始词和标签。三组日志均记录 train 的 **203,621 个词 → 272,595 个子词**（约 1.34 子词/词，不含特殊标记），train/dev 最长序列为 **173 个位置**（含 `[CLS]/[SEP]`）。新评估确认 test 为 **46,435 个词 → 63,461 个子词**（约 1.37 子词/词，不含特殊标记），最长 **148 个位置**（含特殊标记），`[UNK]` word rate 为 **0**。三个 split 均未超过 `BERT_MAX_LEN=512`，test 未截断句子或 gold 标签。

原论文 **Table 1** 是文章/句子/token 统计，**Table 2** 是实体统计。Table 1 的句子数为 14,987 / 3,466 / 3,684，与本地正文句子数的差值恰好为文章数 946 / 216 / 231，符合将文档边界块计为句子的计数差异；不要把它误写为模型漏读数据。train/dev 来自 1996 年 8 月，test 来自 12 月；时间与内容分布变化可能影响表现，但不能把所有 dev/test 差距一概归因于分布偏移。[数据集原论文](https://aclanthology.org/W03-0419.pdf)

文件格式：每行一个 token，空行分句。

```text
EU       NNP  I-NP  I-ORG
rejects  VBZ  I-VP  O
German   JJ   I-NP  I-MISC
```

四列分别是词、POS、chunk、NER。本项目只取第一列为输入、第四列为目标。**POS/chunk 是自动工具产生的，不是人工金标注**；真实系统也可以运行工具产生这些特征。这里不使用它们，是为了保持文本输入设置，不是因为使用就属于作弊。[原论文 §2.2](https://aclanthology.org/W03-0419.pdf)

loader 将原始 IOB1 转为 BIO2：每个实体以 `B-TYPE` 开头，后续为 `I-TYPE`，非实体为 `O`。转换保持实体 span；IOB1 本身并非不能正确评分，重要的是标签规则与解析器一致。

四种实体类型是 `PER/LOC/ORG/MISC`，合计 9 个 BIO2 标签。实体边界和类型都正确才算 TP；边界或类型错误会产生 FP 和 FN。micro-F1 汇总四类的计数后计算 `2TP/(2TP+FP+FN)`；macro-F1 平均四类 F1，不包含 O。Token accuracy 和非法率只是辅助诊断。

### GloVe 词表示与模型

词表由 train 的小写 token 建立，`min_freq=1`、无容量上限，共 **21,011** 项；有 GloVe 的日志记录类型覆盖率 **87.65%**。保留低频词能保留其独立表示及可能已有的预训练信息，但不保证泛化更好；映射到 UNK 不会删除该位置的标签。阈值应由消融决定，不能声称单次词全是名字或全是拼写错误。

大小写从原始 token 提取为 `lower/title/upper/mixed/digit/other` 六类，另加 PAD。两个模型默认都拼接 `word 100维 + case 16维`，得到 `[B,L,116]`，**L 不变，不是一个词变成两个 token**。

```text
词 ID + case ID
  → TokenEmbedding [B,L,116]
  → BiLSTM [B,L,512]
     或 Linear + 正弦位置编码 + Transformer [B,L,128]
  → 逐 token Linear [B,L,9]
```

Transformer 使用 `nn.TransformerEncoderLayer`，4 头、FFN 宽度 `4*dim`、GELU、`batch_first=True`；默认 Pre-LN 并带栈末 LayerNorm。手写 `transformer_naive.py` 保留供学习，当前训练入口不使用它。无 pooling、无字符编码、无 CRF，预测默认逐位置 argmax。

语料已经分词，训练 loader 不重新 tokenize；自由文本预测才调用 tokenizer。词/case PAD ID 都是 0，标签 PAD 是 -100；交叉熵忽略标签 PAD。LSTM 用真实长度 packing，Transformer 用 key padding mask，评估仅统计真实位置。

### BERT 词表示与模型

`AutoTokenizer` 与 `AutoModel` 均加载 `google-bert/bert-base-cased`。不再根据 train 建词表，也不额外拼接 case embedding；输入保留大小写，词表固定为预训练时的 WordPiece 词表。

```text
原始词 + BIO2 标签
  → WordPiece input_ids [B,T] + 首子词位置 word_index [B,W]
  → BERT [B,T,768]
  → gather 首子词特征 [B,W,768]
  → dropout + Linear [B,W,9]
```

`T` 是含 `[CLS]/[SEP]` 的子词序列长度，`W` 是原始词数。后续子词参与 attention，但不单独产生词级标签；`lengths` 仍按词计数。输入按 batch 最长子词序列动态 padding，标签按最长词序列 padding，因此上限 512 不代表每个 batch 都计算 512 个位置。超长输入当前报错，不静默丢弃 gold 标签。

模型关闭 pooler，BERT 主干使用预训练权重，NER head 的 **6,921** 个参数随机初始化；全部 **107,726,601** 个参数从第一步共同训练，无冻结阶段。`attention_mask` 的 1 表示有效子词（包括特殊标记），标签 padding 为 -100；评估仍由相同的 `TaggingMetrics` 计算词级实体 span。

## 6. 训练协议与复现

### GloVe 系列默认训练设置

| 项目 | Stage 1（10 epoch） | Stage 2（10 epoch） |
|---|---|---|
| Word embedding | 冻结整张词表 | 解冻，初始 LR `5e-5` |
| Case embedding | 初始 LR `1e-3` | 初始 LR `5e-5` |
| Encoder / head | 初始 LR `1e-3` | 初始 LR `3e-4` |

每阶段重新创建 Adam 和 cosine scheduler；weight decay=`1e-4`，gradient clip=`5`，batch size=32，label smoothing=0，无类别加权。以上是实验设置，不代表唯一正确方案；不能用“smoothing 会把大部分概率分给 O”来解释关闭它，均匀平滑并不会按类别频率分配概率。

### GloVe 系列训练与普通 eval

本机可使用 `C:\Users\Xiaochuan\miniconda3\envs\dev\python.exe`；下面假定已激活含 PyTorch 的环境，并从项目目录运行。使用新 output 目录，避免覆盖现有实验。

```powershell
cd C:\code\beatDL\text\2_sequence_labeling\CoNLL-2003
python train.py --model lstm --output-dir outputs_lstm_rerun
python train.py --model transformer --output-dir outputs_transformer_rerun
python train.py --model lstm --no-case --output-dir outputs_lstm_nocase_rerun
python train.py --model lstm --no-glove --epochs-stage1 0 --epochs-stage2 20 --output-dir outputs_lstm_noglove_rerun
python train.py --model transformer --dim 256 --layers 4 --output-dir outputs_trf_big_rerun
python train.py --model transformer --norm post --output-dir outputs_trf_post_rerun
python train.py --model transformer --dropout 0.3 --output-dir outputs_trf_d03_rerun
python train.py --model transformer --dropout 0.5 --output-dir outputs_trf_d05_rerun

python eval.py --weights outputs_lstm/best.pt --split valid --show-errors 20
python eval.py --weights outputs_lstm/best.pt --split test --save-cm
```

No-GloVe 命令复现的是本报告中的保守学习率方案，不是推荐的最优 scratch 配置。只传 `--no-glove` **不会自动跳过 stage 1**，必须显式指定 `--epochs-stage1 0`。

训练在数据/GloVe 缺失时可能自动下载；eval 从 checkpoint 读词向量，不需要重新下载 GloVe。普通 eval 从相邻日志恢复模型结构，词表应与 `best.pt`、`training_log.json` 一起保存；默认只打印报告，`--save-cm` 才额外保存图。

### BERT 微调设置与复现

| 项目 | 三组实验的共同设置 |
|---|---|
| 模型 / tokenizer | `google-bert/bert-base-cased` |
| Epoch / batch / seed | 4 / 32 / 42；dev batch=128 |
| 优化器 | AdamW；weight decay=0.01，bias/LayerNorm 不衰减 |
| 学习率 | 全参数共同使用峰值 `2e-5 / 3e-5 / 5e-5`；配置默认仍为 `3e-5` |
| 调度 | 1,756 次更新，前 175 次 warmup，之后线性衰减到 0；每次更新后 step |
| 正则化 | Head dropout=0.1，BERT 内部沿用 checkpoint 配置；gradient clip=1.0，label smoothing=0 |
| 精度 | 这三组训练使用 BF16 autocast，dev 评估使用 FP32 / no-grad |
| 最佳模型 | 按 dev 实体 micro-F1 保存，同一目录中的 `best.pt` 随最佳结果更新 |

需在同一个 Python 环境安装 `transformers`。下面训练命令在项目目录执行，均使用新目录；本次仅执行评估，未重新训练或安装依赖。

```powershell
python -m pip install transformers
python train_bert.py --lr 2e-5 --epochs 4 --batch-size 32 --seed 42 --output-dir outputs_bert_lr2e-5_rerun
python train_bert.py --lr 3e-5 --epochs 4 --batch-size 32 --seed 42 --output-dir outputs_bert_lr3e-5_rerun
python train_bert.py --lr 5e-5 --epochs 4 --batch-size 32 --seed 42 --output-dir outputs_bert_lr5e-5_rerun
```

首次加载会下载模型及 tokenizer，之后复用 Hugging Face 缓存。原始默认 `3e-5` 实验保存在 `outputs_bert/`，不是 `outputs_bert_lr3e-5/`。各运行目录保存 `best.pt`、`training_log.json`、loss/F1 曲线和 **dev** 混淆矩阵；没有自行生成的 `vocab.json`，重建时仍需配套 tokenizer/config。

### BERT 独立评估（已执行）

新增 [eval_bert.py](eval_bert.py)，与 `train_bert.py` 配套。**`eval.py` 仍用于 LSTM/小 Transformer；BERT 权重应交给 `eval_bert.py`。** 后者从权重旁的 `training_log.json` 恢复模型名、head、PAD、标签顺序和长度上限，严格加载完整微调权重，复用 WordPiece Dataset、collate 和实体评分函数。

运行 test 前会先评估完整 dev，与日志 `history` 中的最佳 F1 按 `1e-8` 绝对容差核对；不符则停止，不继续读取 test。`--split valid` 可单独复核 dev。全部使用 FP32、`eval()`、无梯度、完整 split；超长句直接报错，不静默裁掉标签。

本次在 **dev Conda 环境**（Python 3.12.13、PyTorch 2.11.0+cu128、Transformers 5.17.0、RTX 5080）依次执行以下评估，batch=128、num_workers=0。开启离线模式，仅复用已有 Hugging Face 缓存，没有下载模型或修改训练权重：

```powershell
conda activate dev
cd C:\code\beatDL\text\2_sequence_labeling\CoNLL-2003
$env:HF_HUB_OFFLINE = "1"
python eval_bert.py --weights outputs_bert/best.pt --split test --device cuda --save-cm
python eval_bert.py --weights outputs_bert_lr2e-5/best.pt --split test --device cuda --save-cm
python eval_bert.py --weights outputs_bert_lr5e-5/best.pt --split test --device cuda --save-cm
```

实际使用解释器 `C:\Users\Xiaochuan\miniconda3\envs\dev\python.exe`；每份 JSON 的 `command` 保存了具体调用。每组新增 `eval_test.json` 和 `confusion_matrix_test.png`，JSON 包含完整 dev/test 指标、token 混淆矩阵、句子/词/子词计数、训练元数据、环境、tokenizer 词表摘要，以及权重、日志、语料和 Python 源码 SHA-256。旧 `best.pt`、`training_log.json` 和 dev 图均保留。

评估记录已存在时脚本会拒绝覆盖。确需重跑时用新路径，例如 `--output reports/bert_recheck/lr5e-5/eval_test.json --save-cm`；不要删除旧报告来覆盖结果。本次三组 dev 均精确重现，完整 test 均覆盖 **3,453 句 / 46,435 词 / 5,648 实体**。另执行 `python -m unittest test_eval_bert -v`，元数据和 dev 分数校验的 3 项测试通过。

### 2026-09-08 的 GloVe 批量报告（历史快照）

以下是生成旧 8 组报告时的命令，不是当前 11 组实验可直接使用的统一入口：

```powershell
python report_experiments.py --output reports/new_snapshot --device auto
```

[report_experiments.py](report_experiments.py) 扫描 `outputs_*/training_log.json`，但当前实现依赖 GloVe 系列的两阶段元数据、`vocab.json` 和 `eval.py` 加载函数。**现有目录含 BERT 后，直接运行会因不兼容的元数据失败**，需先增加显式的运行筛选或 BERT 支持；不要为绕过它删除或移动实验目录。

旧报告流程先锁定按历史 dev 选择的结果，再逐组复核 dev、评估 train/test；拒绝缺失工件、不完整训练、标签顺序不符、gold 被截断或 dev F1 不匹配的情况。输出目录必须不存在，旧报告不会被覆盖。不要在日常调参时反复跑全量 test 报告。

2026-09-08 保存的 [reports/2026-09-08/](reports/2026-09-08/) 下 8 份单组 JSON 和 1 份汇总 JSON 包含：

- 训练元数据、历史最佳 epoch、参数量、历史平均 epoch 时间。
- 完整 train/dev/test P/R/F1、分类型指标和 token 混淆矩阵。
- dev 的 gold 实体错误分类、按 UNK token 分组的实体召回。
- checkpoint、词表、训练日志、评估源代码及语料 SHA-256；Python/PyTorch/GPU 信息。

该次评估环境为 Python 3.12.13、PyTorch 2.11.0+cu128、RTX 5080。所有评估均为 `eval()`/无梯度、batch=128、完整 split。另执行了 `utils/metrics.py` 的手算评分检查，以及原 `eval.py` 的 LSTM dev 评估/错例查看，结果一致。报告只能记录评估时的源代码身份；历史训练日志没有保存源码版本，不能反推出每次训练的精确代码版本。

2026-09-10 的 BERT 评估由独立 `eval_bert.py` 完成，记录位于各 BERT 运行目录，不追加到 2026-09-08 的历史快照。未补跑 BERT train F1，未修改训练配置或原 checkpoint；原 GloVe 评估快照保持不变。

## 7. 文件导航

- [config.py](config.py)：默认参数、标签和路径。
- [dataset/conll2003.py](dataset/conll2003.py)：解析、BIO 转换、Dataset、padding。
- [dataset/vocab.py](dataset/vocab.py)、[dataset/glove.py](dataset/glove.py)：词表、case 特征与预训练矩阵。
- [model/embedding.py](model/embedding.py)：两路向量拼接；[model/lstm_tagger.py](model/lstm_tagger.py)：BiLSTM 模型。
- [model_transformer/encoder.py](model_transformer/encoder.py)、[transformer_tagger.py](model_transformer/transformer_tagger.py)：Transformer 模型。
- [model_bert/wordpiece.py](model_bert/wordpiece.py)：子词编码、首子词索引、词级标签对齐与 batch padding。
- [model_bert/bert_tagger.py](model_bert/bert_tagger.py)：BERT 主干、gather 与 NER head；[train_bert.py](train_bert.py)：全量微调和 dev 评估。
- [eval_bert.py](eval_bert.py)：BERT checkpoint 的 dev 复核、完整 test 评估及独立 JSON 报告；[test_eval_bert.py](test_eval_bert.py)：评估校验逻辑的离线测试。
- [utils/metrics.py](utils/metrics.py)：实体评分与诊断；[train.py](train.py)、[eval.py](eval.py)：训练和评估入口。
- [predict/predict.py](predict/predict.py)：自由文本预测；注意其结果目录可能在运行时重建。
- `outputs_*/`：原始实验工件，含三组 `outputs_bert*`；`reports/2026-09-08/`：8 组 GloVe 系列的不可覆盖评估快照。
