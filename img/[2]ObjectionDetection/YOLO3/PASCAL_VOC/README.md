# YOLOv3 on PASCAL VOC(手写复现 + 调参记录)

从零手写的 YOLOv3 目标检测器(ResNet backbone + FPN neck + 3 个检测头),在 PASCAL VOC 上训练。
**目标**:做到"接近 YOLOv3 水平"(~0.55–0.65 mAP@0.5)。**结果:达成。**

- 训练集:VOC2007 trainval + VOC2012 trainval(~16.5k 张)
- 验证集:VOC2007 test(4952 张,标准协议)
- 最终:**mAP@0.5 = 0.603,mAP@0.75 = 0.355,mAP[.5:.95] = 0.346**(`outputs/best.pt`)
  - 训练过程中峰值 **mAP@0.5 = 0.607**(focal loss 那版,瘦 neck);当前 `best.pt` 是后续厚 neck 版 0.603,两者统计上同一个数(见死路清单)。

> 这是按 YOLOv3 *思想* 实现的(Darknet 风格 backbone、多尺度特征金字塔、anchor-based 检测头、objectness/box/class 组合 loss)。backbone、neck、loss 在能让代码更清晰或更好训练的地方对原论文做了改动。

---

## 快速开始

所有命令在本目录(`PASCAL_VOC/`)下运行。

```powershell
# 1. 下载数据(VOC07 trainval+test、VOC12 trainval,约 2–3 GB,落在 dataset/data/)
python dataset/voc.py --download

# 2. 训练(两阶段 finetune,默认 stage1=20 + stage2=60 epoch)
python train.py
#   云上可调:--num-workers 8 --batch-size 16 --epochs-stage1 26 --epochs-stage2 60

# 3. 全量评估(VOC07 test 4952 张,输出 mAP / mAP@0.5 / mAP@0.75)
python eval.py
python eval.py --max-batches 20          # 快速抽查

# 4. 按类别诊断(每类 AP@0.5 + recall@0.5,最差的排在最前)
python eval_per_class.py

# 5. 可视化检测(pred 和 GT 并排存到 detect/results/)
python detect/detect.py --voc-random 10  # 随机抽 10 张 VOC test
python detect/detect.py --img path/to/image.jpg
```

### 依赖

```bash
pip install torch torchvision numpy matplotlib tqdm pillow torchmetrics pycocotools
```

| package | 用途 |
|---|---|
| `torch`, `torchvision` | 模型、数据(`VOCDetection`)、`ops.batched_nms` |
| `numpy`, `matplotlib` | 随机种子、loss/mAP 曲线 |
| `tqdm` | 进度条(可选,缺了会自动降级) |
| `pillow` | 图像加载 / 画框 |
| `torchmetrics`, `pycocotools` | mAP 计算(`utils/metrics.py`) |

> **设备**:训练自动选 `cuda` > `mps`(Apple Silicon)> `cpu`。macOS 上若遇到 MPS 不支持的算子,用 `PYTORCH_ENABLE_MPS_FALLBACK=1` 让其回落到 CPU。

---

## 架构

```
image [B,3,416,416]
  └─ backbone (ResNet-34, ImageNet 预训练)  ──►  c3,c4,c5  (stride 8/16/32, 通道 128/256/512)
       └─ neck (FPN 自顶向下融合, ConvSet 5-conv)  ──►  p3,p4,p5  (通道 128/192/256)
            └─ heads ×3 (每尺度一个)  ──►  3 个 raw 预测 [B,3,H,W,25]
```

- **3 个尺度 × 3 个 anchor**,每个 cell 预测 `[tx,ty,tw,th,obj,cls×20]`(VOC 20 类),共 9 个 anchor。
- **anchor**:沿用 COCO-416 的 9 个 anchor(实测比 VOC 上 k-means 重聚类更好,见死路)。
- **正样本分配**:多 anchor 匹配 —— GT 分给所有形状 IoU>`ANCHOR_MATCH_THRESH` 的 anchor(单最佳作兜底),比经典"单最佳 anchor"监督更密。
- **负样本 / ignore**:负样本里 decoded-box 与任意 GT IoU>`IGNORE_THRESH` 的被 ignore(既不算正也不算负)。
- **objectness bias-init**:头部 objectness logit 初始化成 `log(0.01/0.99)≈-4.6`(RetinaNet 先验),开局假设"处处是背景",只在有证据处抬高 —— 抑制早期满屏高置信度假阳性。
- **loss**:CIoU box loss + **focal** objectness + BCE 分类。
- **优化器/调度**:Adam + 每阶段 CosineAnnealingLR;Stage2 用两个 param group(backbone 小 LR、neck+head 大 LR)。
- **推理**:decode → 置信度过滤 → 按类 NMS;评估用 torchmetrics 的 mAP。

两阶段 finetune:

| 阶段 | 做什么 | epoch | LR |
|---|---|---|---|
| Stage 1 | 冻结整个 backbone,只训 neck+head(预热,别让乱梯度污染 backbone) | 20 | 1e-3 |
| Stage 2 | **解冻整个 backbone** 端到端 finetune | 60 | head 1e-4 / backbone 3e-5 |

---

## 实验记录:0.22 → 0.61

每一步都是**单变量**改动,对照前一版的全量 eval。

| # | 改动 | mAP@0.5 | 结论 |
|---|---|---|---|
| 0 | 冻结 backbone,只训 neck+head | ~0.37 | 起点 |
| 1 | objectness bias-init(RetinaNet 先验) | ~0.37 | 修早期假阳性,稳定训练 |
| 2 | `LAMBDA_NOOBJ=2` + 延长 stage2 | 0.44 | 压负样本权重,有效 |
| 3 | **多 anchor 正样本匹配**(`ANCHOR_MATCH_THRESH=0.3`) | 0.47 | 更密的监督,有效 |
| 4 | **解冻整个 backbone**(`STAGE2_UNFREEZE="all"`) | 0.53 | **关键**:冻结的 ImageNet backbone 才是真瓶颈 |
| 5 | **换 ResNet-34**(+WD 1e-3) | 0.53 | @0.5 仅 +0.005,但 **@0.75 +0.074**(定位更准) |
| 6 | **focal loss on objectness**(γ=2, α=0.25) | **0.607** | **最大单点收益**,打破 ~0.53 的天花板 |
| 7 | 加厚 neck `(64,128,256)→(128,192,256)` | 0.603 | ❌ 无增益(见死路) |

### 有效的杠杆(保留)

真正起决定作用的是 **#4 全解冻 backbone** 和 **#6 focal loss**:

1. **全解冻 backbone**(#4)—— 此前一直怀疑是分辨率/anchor/标签的问题,实测都不是。冻着的 ImageNet 特征对检测任务"水土不服",解冻让它充分适配,@0.75 定位精度大涨。**这是从 0.47 到 0.53 的根因。**
2. **focal loss**(#6)—— `α(1-p_t)^γ·BCE`,down-weight 海量易分背景,把梯度集中到难样本。直接抬升最差的几个类:**boat +0.20、plant +0.16、cow +0.15、bottle +0.14**,与 per-class 诊断完全吻合。**这是从 0.53 到 0.607 的根因。**
3. 多 anchor 匹配(#3)、ResNet-34(#5,主要赚 @0.75)、objectness bias-init —— 各有小贡献。

### 死路清单(没有新理由别再试)

| 改动 | 结果 | 为什么没用 |
|---|---|---|
| VOC k-means 重聚类 anchor | 0.354 → 0.222 ❌ | COCO anchor 已经够好,重聚类反而劣化 |
| 更强的颜色/几何增强 | 无增益 | 过拟合**不是** mAP 的瓶颈,加正则没用 |
| multi-scale 训练 | 0.525 → 0.504 ❌ | 且实现有缺陷(从已是 416 的张量缩放,>416 是模糊上采样)。已 `MULTISCALE_TRAIN=False` |
| 固定 512 分辨率 | 0.472 → 0.464 ❌ | 慢 ~4×,冻结 backbone 用不上更高分辨率 |
| 加厚 neck (128,192,256) | 0.607 → 0.603 ❌ | **+2M 参数、train_loss 砸到 0.28,但 mAP 不动** → 检测头容量从来不是瓶颈 |

---

## 关键经验 / 坑

- **val_loss 和 mAP 会解耦**:训练后期 val_loss 因过拟合上升,但 mAP 仍在涨。所以 `best.pt` **按 mAP@0.5 选**,不是按 val_loss。
- **proxy vs 全量 eval**:训练日志里每 epoch 的 mAP 是只用前 60 batch 的**有偏 proxy**(`MAP_EVAL_MAX_BATCHES=60`,省时间);真实数字以 `eval.py` 全量为准,两者可差 ±0.01–0.02。
- **stage1 会饱和**:实测 stage1 在 ~ep30 就到顶(proxy ~0.46),跑到 46 epoch 是白烧算力。stage1 给 25–30 足够。
- **检测头/neck 容量不是瓶颈**:#7 实测,加宽只会让模型更狠地拟合训练集,mAP 不动。

---

## 还差在哪(per-class 诊断)

最终模型每类 AP@0.5(`eval_per_class.py`,最差在前):

| class | AP@0.5 | recall@0.5 | 病因 |
|---|---|---|---|
| cow | 0.338 | 0.477 | 漏检为主(一半没找到) |
| bottle | 0.393 | 0.519 | 漏检 + 低精度 |
| boat | 0.422 | 0.575 | 漏检 + 低精度 |
| pottedplant | 0.424 | 0.568 | 漏检 + 低精度 |
| sheep | 0.443 | 0.701 | 精度为主(找到 70% 但误报多) |
| chair | 0.483 | 0.643 | 精度为主 |
| … | … | … | car/person/dog/train/cat 均 0.74–0.81 |

两种病,**都不是当前架构能修的**:
1. **召回型**(cow/bottle):小、稀有,ResNet-34 特征到顶 → 要更多数据 / 更强 backbone。
2. **精度型**(sheep/chair):相似类互串(sheep↔cow、chair↔sofa)→ 数据 / 分类问题。

## 天花板

- **这套实现**(YOLOv3 思想 + ResNet-34 + VOC07+12)的实际天花板 ≈ **0.60–0.61 mAP@0.5**,已到顶。架构内继续加容量无效。
- **YOLOv3 范式**在 VOC 上的理论天花板 ≈ **0.80–0.83**。再往上要靠范式外的东西:mosaic/mixup、更多数据、或更现代的检测范式(anchor-free、DETR 系)。

---

## 文件结构

```
PASCAL_VOC/
├─ config.py              # 所有超参 / 路径 / anchor(单一配置入口)
├─ train.py               # 两阶段训练,日志/曲线/checkpoint 写到 outputs/
├─ eval.py                # 全量 mAP 评估
├─ eval_per_class.py      # 每类 AP@0.5 + recall 诊断
├─ dataset/
│  ├─ voc.py              # VOCDataset + 下载 + collate
│  └─ transforms.py       # resize / 增强 / 归一化(box 同步变换)
├─ models/
│  ├─ yolov3.py           # 组装 backbone→neck→head + finetune 解冻开关
│  ├─ backbone.py         # ResNet-18/34 backbone(3 个 tap)
│  ├─ neck.py             # FPN 自顶向下融合(ConvSet)
│  └─ head.py             # 每尺度检测头 + objectness bias-init
├─ losses/yolo_loss.py    # CIoU + focal objectness + 多 anchor 匹配 + ignore
├─ utils/                 # nms / metrics(mAP)/ bbox / anchors(k-means)/ viz
├─ detect/detect.py       # 推理可视化(pred vs GT 并排,结果存 detect/results/)
└─ outputs/               # best.pt / last.pt / training_log.json / *.png(gitignore)
```

## 最终配置(`config.py` 要点)

```python
IMG_SIZE = 416                       # COCO-416 anchors
BACKBONE = "resnet34"
NECK_CHANNELS = (128, 192, 256)      # 厚 neck 版(与瘦 neck 64/128/256 精度相同)
BATCH_SIZE = 16 ;  WEIGHT_DECAY = 1e-3
STAGE1_EPOCHS = 20 ; STAGE1_LR = 1e-3
STAGE2_EPOCHS = 60 ; STAGE2_LR_HEAD = 1e-4 ; STAGE2_LR_BACKBONE = 3e-5
STAGE2_UNFREEZE = "all"              # 全解冻 backbone(关键杠杆)
ANCHOR_MATCH_THRESH = 0.3 ; IGNORE_THRESH = 0.5   # 多 anchor 匹配
FOCAL_OBJ = True ; FOCAL_GAMMA = 2.0 ; FOCAL_ALPHA = 0.25  # focal(关键杠杆)
```
