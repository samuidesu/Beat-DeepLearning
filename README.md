# Beat-DeepLearning

A personal PyTorch playground with one goal: **re-implement and solve as many deep-learning problems as possible from scratch** — one task, one dataset, one (or several) models at a time. Every folder is a self-contained experiment with training scripts, tuning notes, and results.

The repo includes **CNN** vision experiments and **NLP** experiments with recurrent models, Transformers, and pretrained BERT.

---

## 1. CNN — Image Classification

Working from simple baselines up to transfer learning on fine-grained datasets. Models range from hand-built CNNs to pretrained backbones fine-tuned in two stages (freeze → head-only → unfreeze last block).

| Dataset | Model(s) used | Approach | Result |
|---|---|---|---|
| **MNIST** | MLP → small VGG-style CNN | from scratch | baseline sanity check |
| **SVHN** (full) | WideResNet | from scratch | digit classification |
| **SVHN** (single-digit) | Custom CNN, WideResNet | from scratch | — |
| **CIFAR-10** | Plain CNN, ResNet, WideResNet | from scratch | architecture comparison |
| **CIFAR-100** | WideResNet + MixUp / CutMix | from scratch + augmentation | — |
| **Dogs vs Cats** | VGG-style CNN (scratch), ResNet-18 (pretrained) | scratch vs transfer learning | — |
| **STL-10** | ResNet-18 / ResNet-50 / DenseNet-121 / EfficientNet-B0 | two-stage fine-tune | up to **97.9%** test acc (ResNet-50) |
| **Oxford-IIIT Pet** | ResNet-50 | two-stage fine-tune | fine-grained (37 classes) |
| **Flowers-102** | EfficientNet-B0 | two-stage fine-tune | fine-grained (102 classes) |

**STL-10 model comparison** (same two-stage recipe): ResNet-50 **97.9%** · EfficientNet-B0 **97.2%** · DenseNet-121 **95.9%** · ResNet-18 **95.4%**.

Recurring themes across these experiments: two-stage fine-tuning, cosine LR scheduling, AdamW, MixUp/CutMix augmentation, and per-run training logs (loss/accuracy curves + JSON history).

---

## 2. CNN — Object Detection

| Dataset | Model | Approach | Result |
|---|---|---|---|
| **PASCAL VOC** (2007 + 2012) | **YOLOv3** (from scratch) | ResNet backbone + FPN neck + 3 detection heads | **mAP@0.5 = 0.603**, mAP@0.75 = 0.355, mAP[.5:.95] = 0.346 |

A hand-written YOLOv3-style detector built on YOLOv3's core ideas (Darknet-style multi-scale backbone, anchor-based heads, combined objectness/box/class loss). Trained on VOC2007 + VOC2012 trainval (~16.5k images) and evaluated on VOC2007 test (4952 images, standard protocol), reaching near-reference YOLOv3 quality. Includes training, full and per-class evaluation, and side-by-side prediction/GT visualization.

See [`cnn/[2]ObjectionDetection/YOLO3/PASCAL_VOC/README.md`](cnn/%5B2%5DObjectionDetection/YOLO3/PASCAL_VOC/README.md) for the full write-up and tuning log.

---

## 3. NLP — Text Classification and Sequence Labeling

Experiments with recurrent models, Transformers, pretrained word vectors, and BERT fine-tuning.

- **Text classification:** sentiment analysis on [SST-2](text/1_text_classification/SST-2/README.md) and [IMDB](text/1_text_classification/IMDB/README.md), and news categorization on [AG-News](text/1_text_classification/AG-News/README.md).
- **Sequence labeling:** named entity recognition on [CoNLL-2003](text/2_sequence_labeling/CoNLL-2003/README.md), using BiLSTM, Transformer, and BERT models.

Each experiment's README contains its setup, running instructions, and results.

---

## Repository Layout

```
Beat-DeepLearing/
├── README.md
└── cnn/
    ├── [1] Image Classification/
    │   ├── mnist/          SVHN/         SVHN_single/
    │   ├── cifar10/        cifar100/     dogs_vs_cats/
    │   ├── STL-10/         Oxford-IIIT Pet/   Flowers102/
    │   └── ...             (notebooks + two-stage fine-tune scripts)
    └── [2]ObjectionDetection/
        └── YOLO3/PASCAL_VOC/   (from-scratch YOLOv3: train/eval/detect)
```

## Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm pillow torchmetrics pycocotools
```

See each experiment's README for additional dependencies and environment setup.

## Roadmap

Continue the "implement, compare, and record what worked" approach across vision and NLP, expanding to more tasks and model families.
