"""Central configuration: paths, dataset info, and hyperparameters.

Only the data-related settings are filled in for now. Anchors, loss weights,
and training hyperparameters will be added when we write the loss / training.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../YOLO3/PASCAL_VOC).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# VOC is downloaded here (dataset/data/). This folder is gitignored.
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "data")
# Logs, curves, checkpoints go here.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# -----------------------------------------------------------------------------
# Dataset: PASCAL VOC (20 classes)
# -----------------------------------------------------------------------------
# The order of this list defines the integer class id of each class
# (aeroplane -> 0, bicycle -> 1, ..., tvmonitor -> 19). Do NOT reorder.
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)  # 20

# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------
# Square network input size. Must be a multiple of 32 (the largest stride),
# so the stride-32 feature map is an integer grid (416 / 32 = 13).
IMG_SIZE = 416

# ImageNet normalization stats. The ResNet-18 backbone was pretrained with
# these, so inputs must be normalized the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Anchors
# -----------------------------------------------------------------------------
# Anchor box sizes in PIXELS at the IMG_SIZE=416 input scale, 3 per scale.
# Order MUST match the model's output order: stride 8 (P3, small objects),
# stride 16 (P4, medium), stride 32 (P5, large).
#
# These are k-means clustered on VOC07+12 trainval (47223 boxes, IoU metric);
# regenerate with `python utils/anchors.py`. Mean IoU 0.655 vs 0.639 for the
# classic COCO anchors -- VOC objects run larger, so the small-scale anchors
# are bigger than COCO's. To revert, swap in the COCO block below:
#   [(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)],
#   [(116, 90), (156, 198), (373, 326)]
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],       # stride 8  -> P3 (small)
    [(30, 61), (62, 45), (59, 119)],      # stride 16 -> P4 (medium)
    [(116, 90), (156, 198), (373, 326)],  # stride 32 -> P5 (large)
]
STRIDES = [8, 16, 32]
NUM_ANCHORS_PER_SCALE = 3

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"          # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 16
NUM_WORKERS = 4
WEIGHT_DECAY = 5e-4

# Two-stage finetuning schedule:
#   Stage 1: freeze the whole backbone, train only neck + head.
#   Stage 2: unfreeze the high backbone stages (layer3/layer4) and finetune
#            them with a smaller LR while continuing to train neck + head.
STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 30
STAGE2_LR_HEAD = 1e-4         # lr for neck + head in stage 2
STAGE2_LR_BACKBONE = 1e-5     # smaller lr for the unfrozen backbone layers
STAGE2_UNFREEZE = ("layer3", "layer4")

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
# Weights balancing the loss terms (all O(1) since each term is a mean).
# Tunable; raise LAMBDA_BOX to emphasize localization, lower LAMBDA_NOOBJ if the
# model becomes too conservative about predicting objects.
LAMBDA_BOX = 1.0      # CIoU box loss (positives only)
LAMBDA_OBJ = 1.0      # objectness BCE on positive cells
LAMBDA_NOOBJ = 2    # objectness BCE on negative (non-ignored) cells
LAMBDA_CLS = 1.0      # classification BCE (positives only)
# A negative anchor whose decoded box has IoU > IGNORE_THRESH with any GT is
# "ignored": neither positive nor counted as a negative for objectness.
IGNORE_THRESH = 0.5

# -----------------------------------------------------------------------------
# Inference / evaluation thresholds
# -----------------------------------------------------------------------------
CONF_THRESH = 0.05       # drop predictions below this objectness*class score
NMS_IOU_THRESH = 0.45    # IoU threshold for non-maximum suppression

# Per-epoch mAP monitoring: full VOC2007 test eval adds ~1min/epoch. Set to an
# int N to estimate mAP on only the first N val batches (faster, biased proxy);
# None = full eval. The final best-checkpoint mAP is always computed in full.
MAP_EVAL_MAX_BATCHES = 60

