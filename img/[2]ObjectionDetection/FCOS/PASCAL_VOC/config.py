"""Central configuration for FCOS on PASCAL VOC: paths, dataset info, and
hyperparameters.

FCOS (Fully Convolutional One-Stage detection) is ANCHOR-FREE: instead of
matching ground-truth boxes to pre-defined anchor shapes (YOLOv3), every
location on every pyramid level directly predicts
    - the 4 distances (l, t, r, b) from itself to the box sides,
    - a per-class score,
    - a "centerness" score that down-weights low-quality boxes predicted far
      from an object's center.
So there is NO anchor list here -- the analog of the anchor design is the
per-level REGRESSION_RANGES that decide which pyramid level owns an object.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../FCOS/PASCAL_VOC).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# VOC data location. To avoid downloading the ~2-3 GB dataset twice, we REUSE
# the copy already downloaded by the YOLO3 project when it exists; otherwise
# data is downloaded into this project's own dataset/data/ (gitignored).
_LOCAL_DATA = os.path.join(PROJECT_ROOT, "dataset", "data")
_YOLO3_DATA = os.path.normpath(os.path.join(
    PROJECT_ROOT, "..", "..", "YOLO3", "PASCAL_VOC", "dataset", "data"))
if (not os.path.isdir(os.path.join(_LOCAL_DATA, "VOCdevkit"))
        and os.path.isdir(os.path.join(_YOLO3_DATA, "VOCdevkit"))):
    DATA_ROOT = _YOLO3_DATA   # reuse the YOLO3 download
else:
    DATA_ROOT = _LOCAL_DATA   # this project's own copy

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
# so the stride-32 feature map is an integer grid (416 / 32 = 13). Kept at 416
# to stay comparable with the YOLO3 project.
IMG_SIZE = 416

# ImageNet normalization stats. The ResNet backbone was pretrained with these,
# so inputs must be normalized the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Pyramid levels (the FCOS replacement for anchors)
# -----------------------------------------------------------------------------
# We use 3 levels P3/P4/P5 (strides 8/16/32) -- same as the YOLO3 project,
# rather than the original FCOS's P3-P7. (At 416 input, stride-64/128 grids
# would be 6.5x6.5 / 3.25x3.25 -- not even integer -- and VOC's large objects
# are already covered by the 13x13 stride-32 grid, exactly as in YOLOv3.)
STRIDES = [8, 16, 32]

# Which pyramid level "owns" a ground-truth box is decided by the size of the
# regression target: a location is only a positive for a GT if
#     low <= max(l, t, r, b) <= high      (distances in INPUT-IMAGE PIXELS)
# for its level's (low, high) below. Small boxes go to the fine stride-8 level,
# big boxes to the coarse stride-32 level. This plays the role YOLOv3's 9
# anchor shapes played. Original FCOS at 800px input uses (0,64), (64,128),
# (128,256), (256,512), (512,inf) for P3-P7; with 3 levels at 416 we merge the
# top ranges into P5.
REGRESSION_RANGES = (
    (0, 64),              # P3, stride 8:  small objects
    (64, 128),            # P4, stride 16: medium objects
    (128, float("inf")),  # P5, stride 32: large objects
)

# Center sampling (an FCOS refinement that helps a lot): a location is only a
# positive if it falls within CENTER_SAMPLING_RADIUS * stride of the GT box
# CENTER (clipped to the box), not merely anywhere inside the box. Locations
# near a box's border produce poor regressions; excluding them gives cleaner
# positives. Set CENTER_SAMPLING = False for the plain "anywhere inside" rule.
CENTER_SAMPLING = True
CENTER_SAMPLING_RADIUS = 1.5

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Backbone architecture: "resnet18" or "resnet34" (same options as the YOLO3
# project; both tap C3/C4/C5 at channels 128/256/512).
BACKBONE = "resnet34"

# FPN output width. ONE number (not per-level like YOLOv3's neck): FCOS runs a
# single SHARED head over all pyramid levels, so every level must have the same
# channel count. 256 is the FCOS/RetinaNet default. Must be divisible by 32
# (the head's GroupNorm group count).
FPN_CHANNELS = 256

# Number of conv+GN+ReLU blocks in each head tower (classification tower and
# regression tower). Original FCOS uses 4.
NUM_HEAD_CONVS = 4

# Classification bias init prior (the RetinaNet trick, also used by the YOLO3
# head for objectness): start the head assuming P(any class)=0.01 everywhere so
# training isn't flooded by early false positives.
CLS_PRIOR = 0.01

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"          # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 16
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-3      # same as the YOLO3 ResNet-34 setting

# Two-stage finetuning schedule (identical protocol to the YOLO3 project):
#   Stage 1: freeze the whole backbone, train only neck + head (warm them up so
#            they don't push garbage gradients into the backbone).
#   Stage 2: unfreeze the backbone (see STAGE2_UNFREEZE) and finetune it end-to-
#            end with a smaller LR while continuing to train neck + head.
STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 60            # a full-backbone finetune needs more epochs
STAGE2_LR_HEAD = 1e-4         # lr for neck + head in stage 2
STAGE2_LR_BACKBONE = 3e-5     # lr for the unfrozen backbone; the key knob --
                              # raise toward 1e-4 if the backbone barely
                              # improves, lower if training destabilizes.
# Which backbone stages to unfreeze in stage 2. "all" unfreezes the ENTIRE
# backbone -- the YOLO3 experiments showed the frozen ImageNet backbone (not
# resolution or label assignment) is the accuracy bottleneck, so default "all".
# A tuple like ("layer3", "layer4") unfreezes only those stages.
STAGE2_UNFREEZE = "all"

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
# The FCOS loss has 3 terms (see losses/fcos_loss.py):
#   cls : sigmoid focal loss over ALL locations (positives + background)
#   reg : GIoU loss on the decoded (l,t,r,b) box, positives only,
#         weighted by the centerness target (better-centered = larger weight)
#   ctr : BCE on the centerness logit, positives only
LAMBDA_CLS = 1.0
LAMBDA_REG = 1.0
LAMBDA_CTR = 1.0

# Focal loss parameters for the classification term. Unlike YOLOv3 (which has a
# separate objectness output where focal was bolted on), FCOS has NO objectness:
# the per-class scores directly face the huge foreground/background imbalance,
# so focal loss on classification is part of the original design, not an add-on.
FOCAL_GAMMA = 2.0     # focusing strength (RetinaNet default 2; 0 = plain BCE)
FOCAL_ALPHA = 0.25    # positive-class weight (RetinaNet default 0.25)

# -----------------------------------------------------------------------------
# Inference / evaluation thresholds
# -----------------------------------------------------------------------------
# Final detection score = sqrt-free product: sigmoid(cls) * sigmoid(centerness).
# The centerness factor is FCOS's substitute for NMS-quality ranking: boxes
# predicted far from an object's center get their score pulled down and are
# then removed by NMS/thresholding.
CONF_THRESH = 0.05       # drop predictions below this cls*centerness score
NMS_IOU_THRESH = 0.45    # IoU threshold for non-maximum suppression

# Per-epoch mAP monitoring: full VOC2007 test eval adds ~1min/epoch. Set to an
# int N to estimate mAP on only the first N val batches (faster, biased proxy);
# None = full eval. The final best-checkpoint mAP is always computed in full.
MAP_EVAL_MAX_BATCHES = 60
