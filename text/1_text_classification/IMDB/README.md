# IMDB：RNN 与手写 Transformer 情感分类实验

> **状态：12 次运行已完成，结果与结论见第 7 节。** 最好的单模型是 BiLSTM 跑满
> 18 个 epoch，test 0.8846。本文所有数字都是在本机语料 / 本机 GPU 上实测的，不是
> 估计值（第 9 节列出每个数字的来源命令）。
>
> 三条主要结论：**序列从 128 拉长到 400，vanilla RNN 相对 GRU 的差距从 3.7pp 扩大到
> 20.4pp**（7.2）；**本项目的训练噪声地板约 0.4pp，小于它的差异一律不是结论**
> （7.3）；**固定 epoch 预算下比较架构，比的是收敛速度而不是最终质量——GRU/LSTM 的
> 排序在 9 epoch 和 18 epoch 下符号相反**（7.4）。

## 1. 这个项目在整个文本系列里的位置

这是 `text/1_text_classification/` 下的第三个项目，结构完全沿用 SST-2 和 AG-News：
同样的 embedding / encoder / head 三段划分、同样的两阶段分层学习率、同样的
per-epoch JSON 日志与曲线、同样那份手写 Transformer。**唯一被推动的变量是数据。**

| 项目 | 类别数 | 训练文档 | 中位长度 | 主要考验 |
|---|---:|---:|---:|---|
| SST-2 | 2 | 67k 片段 | 7 token | 几乎没有；太短 |
| AG News | 4 | 114k 新闻 | 44 token | 中等上下文 |
| **IMDB** | **2** | **22.5k 影评** | **201 token** | **长程记忆 + 数据稀缺** |

这一行同时变了两件事，两件都重要：

**（1）文档长了 5 倍。** 用 `last` pooling 时，预测依赖的隐状态要走完几百步。
SST-2 上三种 cell 差距不到 2 个点，AG News 上拉开了但没有崩，因为主题分类基本是
**词汇任务**——"midfielder" 出现在哪都意味着体育。这里不一样。

**（2）监督信号少了 10 倍。** 22,500 个二分类标签 = 22.5 kbit，而 AG News 是
114,000 个四分类标签 = 228 kbit。下面每一个容量决策，本质上都是在处理这个比值。

而且任务性质不同：情感是**组合性**的，而且经常在后半段翻转——
"演技出色，摄影漂亮，可惜再好也救不了这么懒的剧本"是一条几乎全由褒义词构成的差评。
这就是为什么 [model/head.py](model/head.py) 里关于 pooling 的讨论比两个兄弟项目都长。

---

## 2. 数据分析

### 2.1 语料本身

Maas et al. (2011) 的 Large Movie Review Dataset。按星级划分极性，**中间段被刻意剔除**：
`≤4 星 → negative`，`≥7 星 → positive`，5–6 星丢弃。所以 5 万条影评只产出
25,000 train + 25,000 test，且**完全均衡**（每类 12,500），accuracy 就是诚实指标，
随机基线 0.50。中间段为空也是这个任务比它的长度看起来要容易的原因——没有骑墙的样本。

原始发布是 **5 万个独立 .txt 文件**。打开 5 万个文件比读完它们还慢，在 Windows 上尤其如此，
所以下载步骤会把目录树**合并成两个 csv**（`train.csv` / `test.csv`，即 AG News 的原生格式），
然后删掉解压出来的文件树。之后这个项目读数据的方式和 AG-News 完全一样。

### 2.2 语料噪声：只有一种，但很密集

| 现象 | 实测占比 | 处理 |
|---|---:|---|
| `<br />` 标签 | **58.7%** 的影评含有，共 101,870 个，平均 4.1 个/条 | 替换成**空格** |
| HTML 实体（`&quot;` 等） | 5 / 25,000 = **0.02%** | `html.unescape()`（基本没用） |
| `x/10` 形式的评分 | 6.0%（另有 2.7% 写成 "x out of 10"） | **保留** |

`<br />` 是唯一真正要紧的。不处理的话 tokenizer 会为每个标签吐出 `["<", "br", "/", ">"]`，
在 4,000 条样本上实测**占全部 token 的 5.8%**，而且 `br` 会成为词表里最高频的"词"之一。

替换成空格而不是空字符串，是因为影评里写的是 `great movie<br />I loved it`，
删掉标签会粘成 `moviei`。

对比 AG News：那边四分之一的行带着残缺的 `#39;`，需要自定义正则；IMDB 的抓取把实体解码了、
只把标签重新编码回去，所以标准库就够。

`x/10` 保留，因为真人读者也看得到它，而且这是 benchmark 一直包含的特征。注意
**起作用的是数字而不是这个模式**：含有 x/10 的影评里 55.1% 是正面的，只比 50% 的基准率
高一点点，光识别"这里有个评分"学不到东西。

### 2.3 长度分布（清洗 + 分词后，train 部分 22,500 条）

```
mean 269.7   min 11   p10 105   p25 146   p50 201   p75 328
p90 527   p95 688   p99 1042   max 2737
```

正负两类长度几乎一样（negative mean 266.7 / positive mean 272.7，中位数都是 201），
所以长度本身不是可利用的泄漏特征。

**这是仓库里跨度最大的分布**：p10 到 p99 差了 10 倍。

### 2.4 `MAX_LEN` 怎么定的

| MAX_LEN | 完整保留的影评 | 保留的 token | RNN 相对开销 | attention 相对开销 |
|---:|---:|---:|---:|---:|
| 256 | 64.0% | 71.8% | 2.00x | 4.00x |
| **400** | **82.3%** | **85.5%** | **3.12x** | **9.77x** |
| 512 | 89.4% | 91.3% | 4.00x | 16.00x |
| 800 | 96.8% | 97.8% | 6.16x | 38.01x |

（开销以 AG News 用的 `MAX_LEN=128` 为 1.0）

**选 400**：保留 82% 的完整影评和 86% 的 token；升到 512 只多换来 5.8 个百分点的
token 覆盖率，却要多付 28% 的循环计算和 64% 的注意力计算。400 也足够长，
让 vanilla RNN 真的必须记住点什么。

**保留哪 400 个 token：head+tail（前 300 + 后 100），不是前 400。**

这是本项目唯一一处偏离 SST-2 / AG-News 流水线的地方，理由是数据本身不同。
AG News 的一行是"标题 + 导语"，主题在头十几个词里就定了；而一条影评是一段**论证**，
结论常常落在最后一句——"……但总的来说，这是浪费掉的两个小时"。
只保留头部，等于恰好丢掉人类读者最先去找的那句话。

代价是一道**接缝**：第 299 号和第 300 号 token 其实并不相邻，但下游全部按相邻处理
（RNN 的隐状态直接跨过去，Transformer 的位置编码把拼接后的序列编成连续的 0..399）。
没有插入分隔符——那需要一个 GloVe 里没有向量的新词表项，得单独论证。
换来的是模型能看到每条影评的两端。

`HEAD_LEN=300` 是相对 head-only 更小的改动：保住大部分开头，补回缺失的收尾约 100 个 token。
文献的结论其实反过来——Sun et al., *How to Fine-Tune BERT for Text Classification*
在同一个语料上发现 head 128 + tail 382（25/75）最好——但那是 BERT@512 而不是 BiLSTM@400，
所以它是一个待验证的假设，不是可以照抄的设定。`--head-len 100` 复现他们的比例，
是第一个值得跑的 sweep（见第 6 节）。

只有 17.7% 的影评会走到这条分支，所以这是个"只影响六分之一样本、
但影响的恰好是最难的那六分之一"的改动。

### 2.5 一个反直觉的实测：动态 padding 在这里几乎不省钱

`collate_batch` 按每个 batch 内最长的样本 padding，而不是 padding 到 `MAX_LEN`。
在 AG News 上这很有用。这里实测：

| batch size | 平均 batch 宽度 | padding 浪费 | 若按长度排序分桶 |
|---:|---:|---:|---|
| 32 | 399.9 | 42.3% | 0.1%（总计算量 → 0.58x） |
| 64 | 400.0 | 42.3% | 0.2%（→ 0.58x） |
| 128 | 400.0 | 42.3% | 0.4%（→ 0.58x） |

因为 **17.7% 的影评超过 400**，随机洗牌后几乎每个 batch 都至少含一条，
所以 batch 宽度永远顶到上限。平均每条影评只有 230.7 个真 token，**42.3% 的算力花在 `<pad>` 上**。

真正的解法是**按长度分桶**（能把整个 epoch 降到 0.58x）。这里没做，因为那会让
batch 的构成和影评长度相关，需要单独论证，而这个项目的目的是隔离**数据**的影响，
不是引入两个兄弟项目没有的采样策略。这个数字记在这里，是因为它值得知道。

### 2.6 词表

train 部分共 6,067,532 个 token，**83,232 个不同的类型**，其中 **40.5% 只出现一次**。

| min_freq | 类型数 | token 覆盖率 | embedding 参数 |
|---:|---:|---:|---:|
| 1 | 83,232 | 100.00% | 8.32M |
| 2 | 49,559 | 99.44% | 4.96M |
| 3 | 38,982 | 99.10% | 3.90M |
| **5** | **29,107** | **98.54%** | **2.91M** |
| 10 | 19,529 | 97.50% | 1.95M |
| 20 | 12,634 | 95.95% | 1.26M |

**选 min_freq=5**（AG News 是 2，SST-2 是 1）。保留那些只出现一次的词，会让 embedding 表
几乎翻三倍，而它们每个 epoch 只收到一次梯度更新——在只有 22,500 个标签的情况下不划算。
让出 1.5% 的 token 给 `<unk>` 也不是纯损失：它**教会**模型 `<unk>` 长什么样，
而不是让那一行停在初始化状态。

最终词表 **29,109**（含 `<pad>`/`<unk>`）。

### 2.7 GloVe 覆盖率

复用 SST-2 已下载的 `glove.6B.100d.txt`（`config.GLOVE_PATH` 会先扫描同级项目，
找不到才下载）。实测：

- 类型覆盖 **27,002 / 29,109 = 92.76%**
- token 覆盖 **96.82%**
- `<unk>` 率：train 1.46% / val 1.83% / test 2.17%

顺便一个值得记住的性质（`python dataset/glove.py` 会打印）：

```
cos(dreadful, awful)     = 0.8266   同义词：高
cos(dreadful, wonderful) = 0.4885   反义词：也高
cos(dreadful, sandwich)  = 0.0816   无关词：低
```

GloVe 训练的是**共现**，而反义词天天共现（"到底是好是坏？"）。所以预训练词向量给的是
一个好的初始几何结构，**不是情感轴本身**——找到那个轴是 encoder 的活，
这也正是 stage 2（解冻 embedding）存在的理由。

### 2.8 语料里的重复影评（实测，值得知道）

按论文设计，train/test 使用**不相交的电影**，且没有用户在任一半贡献超过 30 条，
所以不存在作者或片名层面的泄漏。但**逐字重复的影评是存在的**：

| | 属于重复组的行 |
|---|---:|
| train.csv | 188 行（0.75%） |
| test.csv | 390 行（1.56%） |
| **test 中逐字出现在 train 里的** | **123 行（0.49%）** |

没有任何一对重复样本标签冲突，所以这是**可记忆的余量**而不是标签噪声，最多能让
报告的 accuracy 偏高约 0.05 个点。**不做去重**（那会让结果和所有已发表的 IMDB 数字不可比），
但在把 0.2 个点的差异当真之前，应该先想起这一条。

> 因此 `python dataset/imdb.py` 自检里 "val reviews whose text also occurs in train" 
> 不为 0 是**正常的**——切分是对行做划分，重复来自语料本身。

---

## 3. 模型大小设计

### 3.1 实测参数量（vocab = 29,109，embed_dim = 100）

**RNN 路径**

| 配置 | 总参数 | embedding | encoder |
|---|---:|---:|---:|
| BiRNN h=128 ×2 | 3.07M | 2.91M | 0.158M |
| BiGRU h=128 ×2 | 3.38M | 2.91M | 0.473M |
| BiLSTM h=128 ×2 | 3.54M | 2.91M | 0.631M |
| BiRNN h=256 ×2 | 3.49M | 2.91M | 0.578M |
| BiGRU h=256 ×2 | 4.64M | 2.91M | 1.733M |
| **BiLSTM h=256 ×2**（默认） | **5.22M** | 2.91M | 2.310M |
| BiLSTM h=384 ×2 | 7.95M | 2.91M | 5.038M |
| BiLSTM h=256 ×1 | 3.65M | 2.91M | 0.733M |
| BiLSTM h=256 ×3 | 6.80M | 2.91M | 3.887M |

**Transformer 路径**（max_len=400）

| 配置 | 总参数 | embedding | encoder |
|---|---:|---:|---:|
| **d=128 h=4 ×2**（默认） | **3.32M** | 2.91M | 0.409M |
| d=128 h=4 ×4 | 3.72M | 2.91M | 0.806M |
| d=192 h=4 ×2 | 3.82M | 2.91M | 0.909M |
| d=256 h=4 ×2 | 4.52M | 2.91M | 1.605M |
| d=256 h=8 ×4 | 6.10M | 2.91M | 3.185M |

### 3.2 设计结论

**保持 `HIDDEN_SIZE=256`、`NUM_LAYERS=2`、双向；Transformer 保持 `d=128 / 4 头 / 2 层`。**
理由是这两个配置在 SST-2 和 AG News 上原封不动地用过，只有全部固定住，
结果的差异才能归因于数据而不是重新调过的超参。

但这张表真正说明的是另一件事，而且它比 `HIDDEN_SIZE` 重要得多：

> **embedding 表占了整个模型的 56%，而且它的大小与 `HIDDEN_SIZE` 无关。**
> 在这个数据集上真正的容量旋钮是 **`MIN_FREQ`**。
> 把 hidden width 减半（256→128）去掉 1.7M 参数；
> 把 `min_freq` 从 5 提到 20 同样去掉 1.7M，而且代价更小（token 覆盖率 98.5%→96.0%）。

所以如果曲线显示过拟合，**先动 `MIN_FREQ`，再动 `HIDDEN_SIZE`**。

两条需要留意的不对称（做跨架构比较时必须记住）：

1. **Transformer 的 encoder 只有 BiLSTM 的六分之一大**（0.41M vs 2.31M）。
   如果它赢了，那不是"同等容量下更好"。
2. **正则化强度不同**：RNN dropout=0.5，Transformer dropout=0.1。

`--hidden-size` 已经加进 `train.py`（对应 Transformer 侧的 `--dim`），
配合 `--output-dir` 就能跑容量消融，见第 6 节。

### 3.3 显存与 batch size

`BATCH_SIZE=64`（AG News 是 128），理由是**优化**而不是显存：
22,500 条影评在 batch=64 下每个 epoch 只有 **352 步**（AG News 有 891 步），
再大的 batch 会让 9 个 epoch 的预算里更新次数太少。

显存不是约束。本机 RTX 5080（17.1 GB），实测单次前向峰值：

| 模型 | 前向峰值显存 |
|---|---:|
| BiLSTM h=256 ×2 | 0.29 GB |
| BiRNN h=256 ×2 | 0.22 GB |
| Transformer d=128 ×2 | 0.62 GB |

注意力矩阵在 L=400、batch=64、4 头下约 0.16 GB，反向保留约 0.66 GB。

### 3.4 实测训练速度

微基准（12 步取后 9 步均值，batch=(64, 400)，352 步/epoch）：

| 模型 | ms/步 | 秒/epoch | 9 个 epoch |
|---|---:|---:|---:|
| BiRNN h=256 ×2 | 85.4 | 30.1 | ~4.5 min |
| BiGRU h=256 ×2 | 70.9 | 24.9 | ~3.7 min |
| BiLSTM h=256 ×2 | 71.5 | 25.2 | ~3.8 min |
| Transformer d=128 ×2 | 21.7 | 7.6 | ~1.1 min |

**四个模型全跑完约 13 分钟训练**，加上每轮验证和最后的 test 评估，
`run_all.ps1` 整体大约 20–25 分钟。

> vanilla RNN 反而最慢，这不是笔误：cuDNN 对 LSTM/GRU 有融合核，对 `nn.RNN` 的优化差得多。
> AG News 上是同样的现象。

---

## 4. 训练协议（与两个兄弟项目一致）

**两阶段分层学习率**，把 GloVe 词表当作预训练 backbone：

| 阶段 | epoch | embedding | encoder | head |
|---|---:|---|---|---|
| Stage 1 | 6 | **冻结** | 1e-3 | 1e-3 |
| Stage 2 | 3 | 5e-5 | 3e-4 | 3e-4 |

每个阶段各自跑一条 cosine 退火。其余：`Adam`、`weight_decay=1e-4`、`grad_clip=5.0`、
`label_smoothing=0.05`、`seed=42`。

6+3 而不是 AG News 的 5+3：这里一个 epoch 只有 352 步而不是 891 步，同样的 epoch 数
训练量少得多。IMDB 也比两个兄弟项目更早开始过拟合——保护报告数字的是**在 val 上选最佳
checkpoint**，所以末尾多跑几轮浪费的是时间，不是准确率。

**评估口径（重要）：** IMDB 的 test 标签是公开的。如果每个 epoch 都在 test 上打分、
再挑最好的那个 epoch，报出来的就是 best-of-N。所以：

```
train 部分  22,500 条   梯度更新 + 建词表
val   部分   2,500 条   每轮曲线、best.pt 选择
test.csv    25,000 条   最后由 eval.py 读一次
```

val 用 10% 而不是 AG News 的 5%：train.csv 只有那边的五分之一，5% 就只剩 1,250 条，
±1.8 个点的置信区间根本没法用来选 checkpoint。2,500 条把它收窄到 ±1.3 个点——
**仍然不算窄**，两个 epoch 差半个点的时候要记得这件事。代价是 10% 的训练数据，
这就是"拒绝在 test 上选模型"的真实价格。

反过来，**test 有 25,000 条**，accuracy 在 0.88 附近的 95% 置信区间只有 **±0.4 个点**，
所以最终数字之间的差异是值得当真的（前提是超过 0.4 个点）。

---

## 5. 怎么运行

数据已经下载并合并好了（`dataset/data/imdb/{train,test}.csv`，各 25,000 行），
GloVe 复用 `../SST-2/dataset/data/glove/glove.6B.100d.txt`，都不需要再下载。

### 5.1 一条命令跑完全部四个实验

在 IMDB 项目根目录，激活 torch 环境后：

```powershell
.\run_all.ps1
```

它会依次训练 `rnn → gru → lstm → transformer`，然后依次在 test 上评估这四个模型
并保存混淆矩阵。任何一步非零退出就停下。

如果 PowerShell 拒绝执行脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

子集或者透传参数：

```powershell
.\run_all.ps1 -Models lstm,transformer
.\run_all.ps1 -TrainArgs '--seed',7
```

### 5.2 或者手动一条条来

**训练：**

```powershell
python train.py --cell rnn                # → outputs_rnn/
python train.py --cell gru                # → outputs_gru/
python train.py --cell lstm               # → outputs_lstm/
python train.py --model transformer       # → outputs_transformer/
```

**在 test 上评估（每个模型跑一次，最后的报告数字）：**

```powershell
python eval.py --cell rnn  --split test --save-cm
python eval.py --cell gru  --split test --save-cm
python eval.py --cell lstm --split test --save-cm
python eval.py --weights outputs_transformer/best.pt --split test --save-cm
```

`--save-cm` 会在 checkpoint 旁边写 `confusion_matrix_test.png`。
模型结构从各自的 `training_log.json` 读回，不依赖 `config.py` 有没有被改过。

### 5.3 看模型在错哪

```powershell
python predict/predict.py --cell lstm --test-mistakes 15
python predict/predict.py --cell lstm --test-random 20 --seed 0
python predict/predict.py --text "A dull, lifeless remake of a classic."
```

`--test-mistakes` 按"错得最自信"排序，在 IMDB 上这个视图特别值得读：
高置信度的错误基本是反讽、和正面语气复述压抑剧情。
（"结论落在截断窗口之外"曾经是第三类；`TRUNCATION="head_tail"` 就是为消掉它加的。）

### 5.4 各模块自检（不需要训练）

```powershell
python dataset/imdb.py       # 长度分布、<unk> 率、截断率、重复统计
python dataset/vocab.py      # 分词与词表
python dataset/glove.py      # GloVe 覆盖率与词向量语义探针
python model/head.py         # masked pooling 的手算校验
python model/encoder.py      # packing 与 padding 不变性
python model/rnn_classifier.py
python utils/metrics.py      # 混淆矩阵指标的手算校验
```

---

## 6. 建议的后续消融（可选）

按性价比排序。**下列命令中截断方式、pooling、epoch 预算、LayerNorm 位置四组已经跑过，结果见第 7 节**；其余仍待做。

```powershell
# 容量：embedding 是大头，但 hidden width 也值得一试
python train.py --cell lstm --hidden-size 128 --output-dir outputs_lstm_h128

# pooling：情感任务上 last / mean / max 的取舍和主题分类不同（见 head.py）
python train.py --cell lstm --pooling mean --output-dir outputs_lstm_mean
python train.py --model transformer --pooling last --output-dir outputs_transformer_last

# Transformer 加宽 / 加深
python train.py --model transformer --dim 256 --output-dir outputs_transformer_d256
python train.py --model transformer --layers 4 --output-dir outputs_transformer_l4

# 多 seed —— 在把任何排序当结论之前，这一项的信息量最高
python train.py --cell lstm --seed 7  --output-dir outputs_lstm_s7
python train.py --cell lstm --seed 13 --output-dir outputs_lstm_s13

# 截断方式 —— head+tail 已是默认，这两条是它的对照组和文献比例
python train.py --cell lstm --truncation head --output-dir outputs_lstm_head
python train.py --cell lstm --head-len 100  --output-dir outputs_lstm_h100

# LayerNorm 位置 —— Pre-LN vs Post-LN，见下方
python train.py --model transformer --norm pre --dim 256 --layers 4 `
    --epochs-stage1 12 --epochs-stage2 6 --output-dir outputs_trf_big_pre
python train.py --model transformer --norm pre `
    --epochs-stage1 12 --epochs-stage2 6 --output-dir outputs_trf_pre
```

### Pre-LN vs Post-LN

`model_transformer/transformer_naive.py` 原本是 **Post-LN**（2017 原论文的排布）：

```python
x = LN(x + Sublayer(x))      # LayerNorm 压在残差通路上
```

跑 `--dim 256 --layers 4` 时它**前 5 个 epoch 完全没在学**——val_acc 精确等于
0.5000，loss 卡在 0.693 = ln 2，模型塌成"永远输出同一类"，第 6 个 epoch 才自己
爬出来，最终 0.8556 还低于 2 层的 0.8586。2 层 / dim=128 从来没出现过这个现象。

这是 Post-LN 的经典失效模式：LayerNorm 在残差通路上，底层收到的梯度被上面每一层
各缩放一次，所以初期必须靠 **learning-rate warmup** 压住，而且层数越深越严重。
Pre-LN 把 LayerNorm 挪进残差分支：

```python
x = x + Sublayer(LN(x))      # 残差通路是一条不被打断的恒等映射
```

梯度可以无缩放地直达第一层，不需要 warmup——这就是 2020 年之后几乎所有
Transformer 都改用 Pre-LN 的原因。代价是残差流本身永远不被归一化，量级随深度增长，
所以 Pre-LN 需要在整个 stack 之后补一个 LayerNorm（`TransformerEncoder.final_norm`，
只在 `norm="pre"` 时创建）。漏掉它是让 Pre-LN 显得比实际差的常见做法。

`--norm post` 仍是默认，已有的 `outputs_transformer*` 全部保持可复现（实测两个旧
checkpoint 重新评估，准确率一字不差）。**判据很简单：Pre-LN 版第 1 个 epoch 的
val_acc 如果不再是 0.5000，假设就验证了。**

`--truncation head` 是 head+tail 的**对照组**：词表由未截断的全文建立，
所以切换这个开关不会改变任何一个词 id，两者的差异可以干净地归因到"看不看得到结尾"。
预期差异只会出现在那 17.7% 的长影评上，所以把整体准确率的变化除以 0.177
才是这个改动在受影响子集上的真实效果。

`--head-len 100` 复现 Sun et al. 的 25/75 比例（见第 2.4 节）。

---

## 7. 结果

12 次运行，全部 seed=42、`MAX_LEN=400`、head+tail 截断（除非另注）、`MIN_FREQ=5`、
vocab=29,109、GloVe 100d、train batch=64。Test = 25,000 条，val = 2,500 条。

### 7.1 全部运行

| 运行 | 参数量 | epoch | 秒/ep | 第1个epoch val | 最佳ep | Val Acc | **Test Acc** | Test F1 | 短(≤400) | 长(>400) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BiRNN | 3.49M | 9 | 24.7 | 0.5148 | 6 | 0.6708 | **0.6677** | 0.6610 | 0.6672 | 0.6701 |
| BiGRU | 4.64M | 9 | 22.9 | 0.7908 | 8 | 0.8752 | **0.8769** | 0.8769 | 0.8814 | 0.8548 |
| BiLSTM | 5.22M | 9 | 23.0 | 0.7124 | 9 | 0.8520 | **0.8495** | 0.8495 | 0.8553 | 0.8206 |
| Transformer d128 L2 post-LN | 3.32M | 9 | 7.6 | 0.8260 | 8 | 0.8588 | **0.8563** | 0.8562 | 0.8613 | 0.8311 |
| GRU `--truncation head` | 4.64M | 9 | 22.6 | 0.8184 | 8 | 0.8744 | **0.8693** | 0.8692 | 0.8772 | 0.8299 |
| GRU `--pooling mean` | 4.64M | 9 | 26.3 | 0.8228 | 8 | 0.8772 | **0.8732** | 0.8732 | 0.8769 | 0.8548 |
| BiGRU 18ep | 4.64M | 18 | 22.9 | — | 14 | 0.8820 | **0.8772** | 0.8769 | 0.8817 | 0.8553 |
| **BiLSTM 18ep** | 5.22M | 18 | 23.8 | — | 17 | 0.8828 | **0.8846** | 0.8845 | 0.8889 | 0.8634 |
| TRF d128 L2 post 18ep | 3.32M | 18 | 8.2 | — | 16 | 0.8716 | **0.8586** | 0.8586 | 0.8633 | 0.8349 |
| TRF d256 L4 post 18ep | 6.10M | 18 | 23.2 | **0.5000** | 18 | 0.8636 | **0.8556** | 0.8556 | 0.8613 | 0.8273 |
| TRF d128 L2 **pre** 18ep | 3.32M | 18 | 8.2 | 0.8296 | 13 | 0.8664 | **0.8566** | 0.8566 | 0.8618 | 0.8309 |
| TRF d256 L4 **pre** 18ep | 6.10M | 18 | 23.2 | 0.8224 | 12 | 0.8600 | **0.8545** | 0.8545 | 0.8590 | 0.8321 |

**最好的单模型是 BiLSTM 跑满 18 个 epoch，test 0.8846。**

### 7.2 结论 1：vanilla RNN 在 400 步的序列上塌了

这是本项目最主要的结果，也是"只变数据、不变模型"这个设计换来的东西。把 IMDB 和
AG-News 放在一起（两边都用 val，AG-News 项目没做 test 汇总）：

| | 序列长度 | BiRNN | BiGRU | 差距 |
|---|---:|---:|---:|---:|
| AG-News | 128 | 0.8860 | 0.9232 | −3.72pp |
| **IMDB** | **400** | **0.6708** | **0.8752** | **−20.44pp** |

同样的模型代码、同样的两阶段 LR、同样的 hidden size，序列从 128 变成 400，
vanilla RNN 就从"能用"变成"基本不工作"。

它不是精度差，是**根本没优化起来**：9 个 epoch 里 train loss 只从 0.7018 挪到
0.6339（BiGRU 同期到 0.3761），混淆矩阵也偏（negative 召回 0.808 / positive 0.527），
在往一边猜。这就是梯度消失，现在有了本机测出来的数字。

### 7.3 结论 2：噪声地板 ≈ 0.4pp（本项目自测）

`outputs_gru` 和 `outputs_gru_head` 的唯一差别是那 17.7% 长影评保留哪 400 个 token。
两次运行的 `vocab.json` md5 相同、seed 都是 42、split_seed 都是 1234，所以
**长度 ≤400 的 20,820 条影评在训练和测试时输入逐字节相同**。

理论上这个子集应该毫无差别。实测差了 **0.41pp，p=0.00058**。

差异来自训练过程本身：长影评喂进去的内容不同 → 权重走上不同轨迹 → 在从未受影响的
短影评上也给出不同预测。**这就是单次训练的随机性在这个测试集上的量级。**

用它重新审视所有比较：

| 比较 | 差值 | 相对 0.41pp 地板 |
|---|---:|---|
| GRU 9→18 epoch | +0.03pp | 远低于，无意义 |
| pre vs post（d256L4） | −0.11pp | 远低于，无意义 |
| pre vs post（d128L2） | −0.19pp | 低于，无意义 |
| d256L4 vs d128L2 | −0.21pp | 低于，无意义 |
| TRF 9→18 epoch | +0.23pp | 低于，无意义 |
| mean vs last pooling | −0.37pp | 约等于地板，无意义 |
| GRU vs LSTM @18ep | −0.74pp | 1.8× —— 弱证据 |
| head+tail vs head（长影评） | +2.49pp | 6× —— 可信 |
| LSTM vs Transformer @18ep | +2.60pp | 6× —— 可信 |
| LSTM 9→18 epoch | +3.51pp | 8.5× —— 可信 |

顺带一条方法论：短影评那一行 p=0.00058，"显著"得很漂亮，可它测的恰恰是一个
**从构造上就与干预无关**的效果。**p 值小只说明"两个函数不同"，不说明"你的改动
起作用了"。**

### 7.4 结论 3：固定 epoch 预算下比架构，比的是收敛速度

```
 9 epoch:  GRU − LSTM = +2.74pp  (p=6.5e-57，GRU 赢)
18 epoch:  GRU − LSTM = −0.74pp  (p=1.1e-06，LSTM 赢)
```

同一对架构、同一 seed、同一批测试数据，**只是训练预算不同，结论符号就翻了**，
而且两次都"高度显著"。

加倍预算只改变了一个模型：

| 架构 | 9 ep | 18 ep | 差值 | p |
|---|---:|---:|---:|---:|
| BiGRU | 0.8769 | 0.8772 | +0.03pp | 0.843 不显著 |
| Transformer | 0.8563 | 0.8586 | +0.23pp | 0.091 不显著 |
| **BiLSTM** | 0.8495 | **0.8846** | **+3.51pp** | 1.2e-85 |

GRU 和 Transformer 在第 9 个 epoch 就真收敛了，只有 LSTM 没有——它有 4 个门、
2.31M 的 encoder（GRU 1.73M），同样的 1e-3 学得更慢。首轮那个 p=6.5e-57 完全正确地
回答了"这两个 checkpoint 有差别吗"，而我们想问的是"这两个架构有差别吗"，它一个字
都没回答。

而 18 epoch 下的 −0.74pp 只有噪声地板的 1.8 倍，**等预算下 LSTM 和 GRU 基本打平**。

### 7.5 结论 4：head+tail 截断有效，效果精确落在预测的位置

| 子集 | head+tail | head | 差值 | p |
|---|---:|---:|---:|---:|
| 长影评（真被截，n=4,180） | 0.8548 | 0.8299 | **+2.49pp** | 1.6e-06 |
| 短影评（输入相同，n=20,820） | 0.8814 | 0.8772 | +0.41pp | 5.8e-04 |
| 全部 | 0.8769 | 0.8693 | +0.76pp | 7.6e-09 |

拿回结尾那句话，在受影响的那 16.7% 上值 2.5 个点——正是 2.4 节预测的机制。

但整体收益要老实分解：`0.167 × 2.49 = +0.42pp` 是机制带来的，剩下 +0.34pp 是溢出
噪声（见 7.3），换个 seed 符号都可能反过来。**head+tail 值得留作默认，但它的整体
效果和噪声同量级。**

### 7.6 结论 5：mean pooling 没用（一个失败的预测）

设计阶段我判断 `last` 在 400 步上是最可疑的选择。实测相反：

| 子集 | mean | last | 差值 |
|---|---:|---:|---:|
| 全部 | 0.8732 | 0.8769 | −0.37pp |
| 短 | 0.8769 | 0.8814 | −0.44pp |
| **长（最该受益）** | 0.8548 | 0.8548 | **+0.00pp**（146 对 146） |

长影评上一个精确的零。**双向 GRU 的 "last" 是正向末端 + 反向末端拼接，本来就同时
看到了两端**，我把它当成单向 RNN 的 last 来担心了。而且 mean 还慢 15%
（26.3 vs 22.9 秒/epoch）。

### 7.7 结论 6：Post-LN 的深度不稳定性（在本仓库自己写的 Transformer 里）

`--dim 256 --layers 4` 的 Post-LN 版本**前 5 个 epoch 完全没有学习**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Post-LN d256L4 | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | 0.8260 |
| Pre-LN d256L4 | **0.8224** | 0.8340 | 0.8424 | 0.8320 | 0.8524 | 0.8384 |

val_acc 精确等于 0.5000、loss 卡在 0.693 = ln 2，模型塌成"永远输出同一类"，第 6 个
epoch 才自己爬出来。d=128 / 2 层从来没出现过——**这是深度的函数**，正是 Post-LN
文献预测的失效模式（见 6 节的机制说明）。换成 Pre-LN，第一个 epoch 就正常收敛。

**但 Pre-LN 没有提高精度：**

| 形状 | Post-LN | Pre-LN | 差值 | p |
|---|---:|---:|---:|---:|
| d=128 L=2 | 0.8586 | 0.8566 | −0.19pp | 0.211 不显著 |
| d=256 L=4 | 0.8556 | 0.8545 | −0.11pp | 0.528 不显著 |

两个都在噪声地板以下。**Pre-LN 买到的是"能可靠训练深栈"，不是"更高的准确率"**——
在 2–4 层这个规模上，天花板不由优化能力决定。它的价值要到十几层才会显现。

副产品：**Pre-LN 造出了这个项目唯一一次过拟合。** `outputs_trf_big_pre` 的 train loss
降到全部运行最低的 0.2917，val_loss 从第 15 个 epoch 起回升，val_acc 从 0.8600(ep12)
退到 0.8516(ep18)。Post-LN 的迟钝在无意中起了正则化作用；修好优化路径后，6.1M 的
模型终于开始背那 22,500 条影评。

### 7.8 结论 7：Transformer 落后 2.6pp，排除法只剩一个解释

等预算（18 epoch）下 BiLSTM 0.8846 vs 最好的 Transformer 0.8586，差 **2.60pp**
（p=3.5e-38，噪声地板的 6 倍）。四个候选解释死了三个：

| 解释 | 证据 | 状态 |
|---|---|---|
| 容量太小 | d256L4（6.1M）= −0.21pp，且开始过拟合 | 排除 |
| 没训练收敛 | 9→18 epoch = +0.23pp（p=0.09） | 排除 |
| 优化路径有问题 | Pre-LN 修好训练，精度不变 | 排除 |
| **架构与数据规模不匹配** | — | **剩下的解释** |

22,500 条训练文档对一个从零学的注意力机制来说太少。BiLSTM 自带"顺序 + 局部性"的
归纳偏置，Transformer 得从数据里学出来；它的输入只是 100 维 GloVe 投影到 128 维，
位置信息来自固定正弦编码。**Transformer 是在数据规模上来之后才赢的，本项目的规模
恰好在它不占优的一侧。**

一个反面的注脚：Transformer 的性价比很高——3.32M 参数、8.2 秒/epoch（BiLSTM 的
1/3 时间），换来低 2.6pp 的准确率。

### 7.9 一个必要的校准

**0.8846 相对外部基线偏低。** IMDB 上一个调好的线性模型（tf-idf bigram + 逻辑回归 /
NB-SVM）通常在 0.89–0.91，BERT 在 0.94–0.95。也就是说本项目最好的神经网络很可能
打不过一个词袋模型。

这不是失败——GloVe 100d + 2 层 BiLSTM 不是为了刷榜。但它是这个项目最值得学的一课：
**情感分类里"哪些词出现了"贡献了绝大部分信号，序列结构的边际收益远小于直觉。**
线性基线还没跑，见 6 节。

### 7.10 所有配对检验（McNemar，同一批 25,000 条 test）

| 比较 | 差值 | 只有A对/只有B对 | p | |
|---|---:|---:|---:|---|
| head+tail vs head（全部） | +0.76pp | 630 / 440 | 7.6e-09 | 显著 |
| head+tail vs head（**仅长影评**） | +2.49pp | 282 / 178 | 1.6e-06 | 显著 |
| head+tail vs head（仅短影评，输入相同） | +0.41pp | 348 / 262 | 5.8e-04 | 显著但**是噪声** |
| mean vs last pooling | −0.37pp | 691 / 783 | 0.018 | 显著但在地板内 |
| GRU 9→18 epoch | +0.03pp | 629 / 621 | 0.843 | 不显著 |
| LSTM 9→18 epoch | +3.51pp | 1436 / 559 | 1.2e-85 | 显著 |
| Transformer 9→18 epoch | +0.23pp | 576 / 519 | 0.091 | 不显著 |
| GRU vs LSTM @ 9 epoch | +2.74pp | 1268 / 583 | 6.5e-57 | 显著（**预算假象**） |
| GRU vs LSTM @ 18 epoch | −0.74pp | 615 / 799 | 1.1e-06 | 显著但仅 1.8× 地板 |
| LSTM vs Transformer @ 18 epoch | +2.60pp | 1591 / 940 | 3.5e-38 | 显著 |
| Pre vs Post-LN（d128L2） | −0.19pp | 681 / 729 | 0.211 | 不显著 |
| Pre vs Post-LN（d256L4） | −0.11pp | 901 / 929 | 0.528 | 不显著 |
| d256L4 vs d128L2（均 Pre-LN） | −0.21pp | 807 / 860 | 0.203 | 不显著 |

McNemar 只在两个模型**意见不一致**的样本上做检验：一致的样本（都对或都错）对
"谁更强"零信息，全部丢弃，剩下的变成一个抛硬币问题。它处理掉了测试集的抽样噪声，
但**没有处理训练的随机性**——所以它检验的是"这两个 checkpoint 有差别吗"，不是
"这两个架构有差别吗"。7.3 的噪声地板就是补上这一块的。

### 7.11 还欠着的

- **多 seed。** 全部 12 次运行都是 seed=42。LSTM 领先 GRU 只有 0.74pp，不跑多 seed
  就说不出这两个架构谁更好。这是目前信息量最高的一项。
- **线性基线**（7.9）。
- **Transformer 加正则**：大 Pre-LN 模型过拟合了，而它的 dropout 只有 0.1（RNN 是
  0.5）。`--dropout 0.3` 能同时回答"能不能压住"和"压住后容量是否终于有用"。
- **更深的 Pre/Post 对照**：2→4 层看不出 Pre-LN 的价值，8 层大概率能看出来。

---

## 8. 文件结构

```
IMDB/
├── config.py                       所有超参 + 测得的语料统计（带出处）
├── train.py                        训练入口（--cell / --model / --hidden-size ...）
├── eval.py                         评估入口，唯一读 test.csv 的地方
├── run_all.ps1                     四个实验串行跑完 + 评估
├── dataset/
│   ├── imdb.py                     下载、合并成 csv、清洗、切分、Dataset、collate
│   ├── vocab.py                    分词器 + 词表（与两个兄弟项目一致）
│   ├── glove.py                    GloVe 下载 + 构造 embedding 矩阵
│   └── data/imdb/{train,test}.csv  合并后的语料（各 25,000 行）
├── model/                          RNN 路径：embedding / encoder / head
├── model_transformer/              手写 Transformer（transformer_naive.py 是核心）
├── predict/predict.py              推理 + 错误样本查看
└── utils/                          下载器、指标、可视化
```

GloVe 不在本项目里，复用 `../SST-2/dataset/data/glove/`。

---

## 9. 本文数字的来源

所有统计都在本机语料上实测。可复现的入口：

| 数据 | 来源 |
|---|---|
| 长度分布、`<unk>` 率、截断率、重复统计 | `python dataset/imdb.py` |
| GloVe 覆盖率、词向量语义探针 | `python dataset/glove.py` |
| 词表 min_freq 扫描、`<br />` 占比、MAX_LEN 权衡表 | 一次性统计脚本，结论已写入 [config.py](config.py) 注释 |
| 参数量、显存峰值 | 直接实例化模型后 `sum(p.numel() ...)` |
| ms/步、秒/epoch | 12 步微基准取后 9 步均值（第 3.4 节）；第 7 节的秒/epoch 是训练日志里的实测均值 |
| Val Acc / 最佳 epoch / 每轮曲线 | 各 `outputs_*/training_log.json` |
| Test Acc / F1 / 混淆矩阵 | `python eval.py ... --split test --save-cm` |
| 长短影评分组准确率、McNemar 配对检验 | 逐条推理已训练的 checkpoint 后在预测数组上统计（不训练） |

第 7 节的每个数字都对应 `outputs_*/` 里的一次真实运行，目录名在表格首列。

---

### 环境备注：Windows 上 cuDNN RNN 的退出问题

`train.py` 里的 `clean_exit()` 是从 AG-News 项目继承过来的：本机
（Windows 11 + torch 2.11.0+cu128）上，用过带 dropout 的 cuDNN RNN 且处于 train 模式的进程，
会在 `main()` 返回很久之后以 `-1073740791`（`STATUS_STACK_BUFFER_OVERRUN`）崩溃。

产物（checkpoint、日志、PNG）在崩溃发生时已全部写完关闭，所以这纯属退出码问题——
但它会让 `run_all.ps1` 在第一个模型之后中止，所以必须处理。
`os._exit()` 不够（Windows 上它会走到 `ExitProcess`，仍然执行 `DLL_PROCESS_DETACH`，
崩溃就在那里），只有 `TerminateProcess` 能拿到 0。详见 `train.py` 中 `clean_exit()` 的注释。

Transformer 路径不触发这个问题（没有 cuDNN RNN），所以只有 RNN 训练结束时才调用它。
