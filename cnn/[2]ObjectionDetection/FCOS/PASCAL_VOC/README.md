# FCOS on PASCAL VOC(手写复现)

从零手写的 **FCOS**(Fully Convolutional One-Stage,anchor-free)目标检测器,在 PASCAL VOC 上训练。
与本仓库的 [YOLO3 复现](../../YOLO3/PASCAL_VOC/) 完全平行:**同一个 backbone(ResNet-18/34)、同一套数据管线、同一个两阶段训练协议、同一套评估脚本**,唯一的本质区别是"每个位置怎么定义正样本、预测什么"。目的就是把 anchor-based 和 anchor-free 两条路线放在同一实验条件下对比。

- 训练集:VOC2007 trainval + VOC2012 trainval(~16.5k 张)
- 验证集:VOC2007 test(4952 张,标准协议)
- 结果:**待训练**(YOLO3 版同条件下 mAP@0.5 = 0.603,作为对照基线)

> 这是按 FCOS *思想* 实现的(anchor-free 逐位置回归、centerness、按尺度范围分层)。原论文用 P3–P7 五层 / 800px 输入;这里为了与 YOLO3 复现可比,保持 416 输入 + P3–P5 三层。

---

## 快速开始

所有命令在本目录(`PASCAL_VOC/`)下运行。

```powershell
# 1. 数据:如果 YOLO3 项目已经下载过 VOC,会自动复用(见 config.DATA_ROOT),
#    无需重新下载;否则:
python dataset/voc.py --download

# 2. 训练(两阶段 finetune,默认 stage1=20 + stage2=60 epoch)
python train.py
#   云上可调:--num-workers 8 --batch-size 16 --epochs-stage1 20 --epochs-stage2 60

# 3. 全量评估(VOC07 test 4952 张,输出 mAP / mAP@0.5 / mAP@0.75)
python eval.py
python eval.py --max-batches 20          # 快速抽查

# 4. 按类别诊断(每类 AP@0.5 + recall@0.5,最差的排在最前)
python eval_per_class.py

# 5. 可视化检测(pred 和 GT 并排存到 detect/results/)
python detect/detect.py --voc-random 10  # 随机抽 10 张 VOC test
python detect/detect.py --img path/to/image.jpg
```

每个模块文件都带自测入口,可以单独跑来验证形状/逻辑:

```bash
python models/backbone.py ; python models/neck.py ; python models/head.py
python models/fcos.py ; python losses/fcos_loss.py ; python utils/locations.py
```

### 依赖

与 YOLO3 项目完全相同:

```bash
pip install torch torchvision numpy matplotlib tqdm pillow torchmetrics pycocotools
```

---

## 架构

```
image [B,3,416,416]
  └─ backbone (ResNet-34, ImageNet 预训练)  ──►  c3,c4,c5  (stride 8/16/32, 通道 128/256/512)
       └─ neck (经典 FPN: lateral 1x1 + 自顶向下相加 + 3x3 smooth)  ──►  p3,p4,p5  (通道统一 256)
            └─ head ×1 (三层共享同一个头)  ──►  3 个 raw 预测 [B,H,W,25]
```

每个位置(不是 anchor!)预测 `[l,t,r,b, centerness, cls×20]`:

- **l,t,r,b**:该位置到框四条边的像素距离,`exp()` 解码保证为正;head 里每层有一个可学习标量 `Scale` 调节量级(P3 回归 ≤64px,P5 能到 400+px,共享分支靠它适配)。
- **centerness**:该位置距离所属目标中心有多近,`sqrt(min(l,r)/max(l,r) · min(t,b)/max(t,b))`;推理时乘进分类分数,压掉边缘位置回归出来的低质量框——**接替 YOLOv3 objectness 的角色**。
- **cls**:20 类 sigmoid 多标签,focal loss 直面前景/背景不平衡(FCOS 没有 objectness,focal 是核心设计而非可选项)。

### 正样本分配(anchor matching 的替代品)

一个位置是某个 GT 的正样本,须同时满足三条:

| 条件 | 内容 | 替代了 YOLOv3 的什么 |
|---|---|---|
| 1. 在框内 | 四个距离 l,t,r,b 全部 > 0 | "GT 中心落在哪个 cell" |
| 2. 中心采样 | 且落在 GT 中心 ±1.5×stride 内(可关) | —(FCOS 自己的改进) |
| 3. 尺度范围 | max(l,t,r,b) 落在该层的范围内:P3 (0,64] / P4 (64,128] / P5 (128,∞) | **9 个 anchor 的形状匹配** |

一个位置同时命中多个 GT 时,取**面积最小**的那个(小目标只有这里能表达,大目标别处有的是位置)。

### Loss(三项)

| 项 | 公式 | 范围 | 归一化 |
|---|---|---|---|
| cls | sigmoid focal(γ=2, α=0.25) | 全部位置 | / num_pos |
| reg | 1 − GIoU(解码框 vs GT 框) | 仅正样本 | centerness 加权,/ Σcenterness |
| ctr | BCE(centerness 预测 vs 目标) | 仅正样本 | / num_pos |

reg 用 centerness 目标加权是官方 FCOS 的细化:中心位置回归得准、推理时也真会被用到,让它们主导梯度。

### 两阶段 finetune(与 YOLO3 完全一致)

| 阶段 | 做什么 | epoch | LR |
|---|---|---|---|
| Stage 1 | 冻结整个 backbone,只训 neck+head | 20 | 1e-3 |
| Stage 2 | 解冻整个 backbone 端到端 finetune | 60 | head 1e-4 / backbone 3e-5 |

(YOLO3 的实验已经证明"冻结的 ImageNet backbone 才是瓶颈",这里直接沿用结论,`STAGE2_UNFREEZE="all"`。)

---

## 与 YOLOv3 复现的逐点对照

| | YOLO3 版 | FCOS 版(本项目) |
|---|---|---|
| 每个 cell 预测 | 3 个 anchor × [tx,ty,tw,th,obj,cls] | **1 个** [l,t,r,b,ctr,cls](无 anchor 轴) |
| 框参数化 | sigmoid(txy)+cell、anchor·exp(twh) | 位置 ± exp(ltrb) 四边距离 |
| 正样本定义 | GT 与 anchor 形状 IoU > 0.3(多 anchor 匹配) | 框内 + 中心采样 + 尺度范围 |
| 大小目标分层 | 9 个 anchor 的形状 | REGRESSION_RANGES(像素区间) |
| 质量分支 | objectness(+ignore 机制) | centerness(无需 ignore) |
| 分类失衡 | focal 加在 objectness 上(调参调出来的) | focal 加在 cls 上(原生设计) |
| neck | 逐级变宽 concat 融合 (128/192/256) | 等宽 sum 融合 FPN(256,共享头前提) |
| head | 每尺度各一个 | **三层共享一个**(+每层 1 个 Scale 标量) |
| head 归一化 | BatchNorm | GroupNorm(共享头 + 小 batch 友好) |
| 超参手感 | anchor 是主要调参对象 | 没有 anchor 可调,换成 ranges/半径 |
| 坐标单位 | 全程归一化 [0,1] | loss/解码用输入像素(ranges 天然是像素) |

数据管线(`dataset/`)、评估(`utils/metrics.py`)、可视化、训练循环骨架与 YOLO3 项目逐行相同——diff 一下两个项目就能精确看到"anchor-free 到底改了哪些东西"。

---

## 最终配置(`config.py` 要点)

```python
IMG_SIZE = 416
BACKBONE = "resnet34"
STRIDES = [8, 16, 32]
REGRESSION_RANGES = ((0, 64), (64, 128), (128, inf))   # 层级分配,替代 anchor
CENTER_SAMPLING = True ; CENTER_SAMPLING_RADIUS = 1.5
FPN_CHANNELS = 256 ; NUM_HEAD_CONVS = 4                # 共享头:4 conv + GN 塔
FOCAL_GAMMA = 2.0 ; FOCAL_ALPHA = 0.25 ; CLS_PRIOR = 0.01
BATCH_SIZE = 16 ; WEIGHT_DECAY = 1e-3
STAGE1_EPOCHS = 20 ; STAGE1_LR = 1e-3
STAGE2_EPOCHS = 60 ; STAGE2_LR_HEAD = 1e-4 ; STAGE2_LR_BACKBONE = 3e-5
STAGE2_UNFREEZE = "all"
```

## 文件结构

```
PASCAL_VOC/
├─ config.py              # 所有超参 / 路径 / 回归范围(单一配置入口)
├─ train.py               # 两阶段训练,日志/曲线/checkpoint 写到 outputs/
├─ eval.py                # 全量 mAP 评估
├─ eval_per_class.py      # 每类 AP@0.5 + recall 诊断
├─ dataset/
│  ├─ voc.py              # VOCDataset + 下载 + collate(与 YOLO3 相同)
│  └─ transforms.py       # resize / 增强 / 归一化(与 YOLO3 相同)
├─ models/
│  ├─ fcos.py             # 组装 backbone→FPN→共享头 + finetune 解冻开关
│  ├─ backbone.py         # ResNet-18/34 backbone(3 个 tap,与 YOLO3 相同)
│  ├─ neck.py             # 等宽 FPN(lateral + sum + smooth)
│  └─ head.py             # 共享 FCOS 头(cls/reg 双塔 + centerness + Scale)
├─ losses/fcos_loss.py    # 正样本分配 + focal cls + GIoU reg + centerness BCE
├─ utils/
│  ├─ locations.py        # 生成每层的 (x,y) 位置点(anchor 的替代品)
│  ├─ bbox.py             # xywh↔xyxy / IoU / GIoU
│  ├─ nms.py              # 解码(必须与 loss 一致)+ 置信度过滤 + 按类 NMS
│  ├─ metrics.py          # torchmetrics mAP(与 YOLO3 相同)
│  └─ viz.py              # 画框(与 YOLO3 相同)
├─ detect/detect.py       # 推理可视化(pred vs GT 并排)
└─ outputs/               # best.pt / last.pt / training_log.json / *.png(gitignore)
```

## 待办 / 实验计划

- [ ] 跑通两阶段训练,记录 mAP@0.5 / @0.75 曲线
- [ ] 与 YOLO3 版(0.603)同条件对比:整体 mAP、per-class 差异(尤其 YOLO3 最弱的 cow/bottle/boat/pottedplant——anchor-free 的分配方式对小目标是否更友好)
- [ ] 消融:关掉 CENTER_SAMPLING / centerness 加权,看各自贡献
- [ ] 调 REGRESSION_RANGES(416 输入下 (0,64)/(64,128) 的切分是否最优)
