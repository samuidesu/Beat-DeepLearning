# YOLOv3 on PASCAL VOC (PyTorch)

A from-scratch, modular implementation of an object detector based on the
**YOLOv3** idea, trained on **PASCAL VOC 2007/2012** (20 classes).

> This follows the *spirit* of YOLOv3 (Darknet-style backbone, multi-scale
> feature pyramid, anchor-based heads, and a combined objectness / box / class
> loss). The backbone, neck, and loss may deviate from the original paper where
> it makes the code clearer or easier to train.

## Core ideas of YOLOv3 (what we are implementing)

1. **Backbone** — ResNet-18 (ImageNet-pretrained) extracts features at 3 strides
   8, 16, 32 (channels 128/256/512; e.g. 52x52, 26x26, 13x13 for a 416x416 input).
2. **Neck** — an FPN-style top-down path that fuses deep semantic features with
   shallow high-resolution features, producing 3 pyramid levels.
3. **Heads** — each of the 3 levels predicts, per anchor:
   `[tx, ty, tw, th, objectness, class_1..class_20]`.
   3 anchors per level => 9 anchors total.
4. **Target assignment** — each ground-truth box is assigned to the anchor (and
   grid cell) whose shape best matches it (highest IoU).
5. **Loss** — sum over scales of:
   - box regression (coordinates of positive anchors),
   - objectness (BCE, positive vs. negative anchors),
   - classification (BCE per class for positive anchors).
6. **Inference** — decode predictions to absolute boxes, filter by confidence,
   then apply per-class **NMS**. Evaluate with **mAP@0.5**.

## Directory layout

```
PASCAL_VOC/
├── README.md             # this file
├── config.py             # hyperparameters, anchors, class names, paths
├── dataset/
│   ├── __init__.py
│   ├── voc.py            # VOCDataset: load images + parse XML -> normalized boxes
│   ├── transforms.py     # resize / augmentation (flip, color jitter, ...)
│   └── data/             # VOC downloaded here (gitignored)
├── models/
│   ├── __init__.py
│   ├── backbone.py       # ResNet-18 backbone (ImageNet-pretrained)
│   ├── neck.py           # FPN-style top-down feature fusion
│   ├── head.py           # 3 detection heads (one per scale)
│   └── yolov3.py         # assembles backbone + neck + heads
├── losses/
│   ├── __init__.py
│   └── yolo_loss.py      # multi-scale loss (CIoU box + objectness + class)
├── utils/
│   ├── __init__.py
│   ├── bbox.py           # IoU/CIoU, (cx,cy,w,h) <-> (x1,y1,x2,y2)
│   ├── anchors.py        # (optional) k-means; default anchors live in config.py
│   ├── nms.py            # decode + per-class NMS (torchvision.ops.batched_nms)
│   ├── metrics.py        # mAP (torchmetrics MeanAveragePrecision)
│   └── viz.py            # draw predicted boxes on images
├── train.py              # two-stage training (saves logs + curves to outputs/)
├── eval.py               # compute mAP on VOC2007 test
├── detect.py             # run inference on an image / folder
└── outputs/              # training_log.json, loss_curve.png, checkpoints (*.pt gitignored)
```

## Dependencies

```bash
pip install torch torchvision numpy matplotlib tqdm pillow torchmetrics pycocotools
```

| package | used for |
|---|---|
| `torch`, `torchvision` | model, data (`VOCDetection`), `ops.batched_nms` |
| `numpy`, `matplotlib` | seeding, loss-curve plots |
| `tqdm` | progress bars (optional; code falls back without it) |
| `pillow` | image loading / drawing detections |
| `torchmetrics`, `pycocotools` | mAP computation (`utils/metrics.py`) |

> **Device:** training auto-selects `cuda` > `mps` (Apple-Silicon GPU) > `cpu`.
> On macOS, MPS uses the Mac's GPU; if you hit an unsupported-op error, run with
> `PYTORCH_ENABLE_MPS_FALLBACK=1` to let those ops fall back to the CPU.

## Data

PASCAL VOC is downloaded via `torchvision.datasets.VOCDetection` into
`dataset/data/` (gitignored). See `config.py` for paths and `dataset/voc.py`
for loading. Train = VOC2007+2012 trainval, eval = VOC2007 test.

```bash
python dataset/voc.py --download    # download VOC (~2-3 GB), or:
python train.py --download          # download then train
```

## Usage

```bash
python train.py                          # two-stage training
python eval.py                           # mAP on VOC2007 test (uses outputs/best.pt)
python eval.py --max-batches 20          # quick mAP spot-check
python detect.py --img path/to/image.jpg # run detection on one image
python detect.py --dir path/to/folder    # ... or a whole folder
```
