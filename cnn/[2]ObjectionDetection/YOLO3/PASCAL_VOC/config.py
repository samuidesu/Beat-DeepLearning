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

# Multi-scale training: every MULTISCALE_INTERVAL batches, randomly resize the
# input to a new square size in [MULTISCALE_MIN, MULTISCALE_MAX] (multiples of
# 32). This is YOLOv3's signature regularizer -- it stops the (now fully
# unfrozen, high-capacity) backbone from memorizing absolute object scales, the
# main lever against the heavy overfitting that appears once the whole backbone
# is finetuned. Eval always stays at the fixed IMG_SIZE. Set False to disable.
MULTISCALE_TRAIN = False  # disabled: the resize-from-416 impl only upscaled the
                          # >416 sizes, and it didn't help (slightly hurt mAP).
MULTISCALE_MIN = 320      # smallest training size (multiple of 32)
MULTISCALE_MAX = 512      # largest training size (multiple of 32)
MULTISCALE_INTERVAL = 10  # resample the size every N batches (YOLOv3 uses 10)

# -----------------------------------------------------------------------------
# Anchors
# -----------------------------------------------------------------------------
# Anchor box sizes in PIXELS at the IMG_SIZE=416 input scale, 3 per scale.
# Order MUST match the model's output order: stride 8 (P3, small objects),
# stride 16 (P4, medium), stride 32 (P5, large).
#
# These are the classic COCO YOLOv3 anchors. (Tried bumping resolution to 512
# with anchors scaled x512/416; it plateaued at the same mAP and ran ~4x slower
# on MPS, so reverted -- the bottleneck is the frozen backbone, not resolution.)
# To re-cluster on VOC instead, run `python utils/anchors.py`.
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],       # stride 8  -> P3 (small)
    [(30, 61), (62, 45), (59, 119)],      # stride 16 -> P4 (medium)
    [(116, 90), (156, 198), (373, 326)],  # stride 32 -> P5 (large)
]
STRIDES = [8, 16, 32]
NUM_ANCHORS_PER_SCALE = 3

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Backbone architecture: "resnet18" or "resnet34". ResNet-34 (~2x the backbone
# params) has stronger features -> a higher mAP ceiling, at the cost of somewhat
# heavier overfitting on VOC's small train set (managed via mAP-based checkpoint
# selection + weight decay). Both use BasicBlock with identical tap channels
# (128/256/512), so the neck/head are unchanged when switching.
BACKBONE = "resnet34"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"          # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 16
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-3      # raised from 5e-4 to regularize the larger ResNet-34

# Two-stage finetuning schedule:
#   Stage 1: freeze the whole backbone, train only neck + head (warm them up so
#            they don't push garbage gradients into the backbone).
#   Stage 2: unfreeze the backbone (see STAGE2_UNFREEZE) and finetune it end-to-
#            end with a smaller LR while continuing to train neck + head.
STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 60           # longer: a full-backbone finetune needs more epochs
STAGE2_LR_HEAD = 1e-4         # lr for neck + head in stage 2
STAGE2_LR_BACKBONE = 3e-5     # lr for the unfrozen backbone (full finetune); the
                              # key knob -- raise toward 1e-4 if the backbone
                              # barely improves, lower if training destabilizes.
# Which backbone stages to unfreeze in stage 2. "all" unfreezes the ENTIRE
# backbone (stem + layer1-4) for a full finetune -- needed because the frozen
# ImageNet backbone (not resolution or label assignment) is the accuracy
# bottleneck. A tuple like ("layer3", "layer4") unfreezes only those stages.
STAGE2_UNFREEZE = "all"

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
# Weights balancing the loss terms (all O(1) since each term is a mean).
# Tunable; raise LAMBDA_BOX to emphasize localization, lower LAMBDA_NOOBJ if the
# model becomes too conservative about predicting objects.
LAMBDA_BOX = 1.0      # CIoU box loss (positives only)
LAMBDA_OBJ = 1.0      # objectness BCE on positive cells
LAMBDA_NOOBJ = 1      # objectness loss weight on negatives. Back to 1: focal
                      # loss's alpha now handles the pos/neg balance (see below).
LAMBDA_CLS = 1.0      # classification BCE (positives only)
# A negative anchor whose decoded box has IoU > IGNORE_THRESH with any GT is
# "ignored": neither positive nor counted as a negative for objectness.
IGNORE_THRESH = 0.5

# Multi-anchor matching: a GT is assigned as a POSITIVE to every anchor whose
# shape (w,h) IoU with it exceeds this threshold (its single best anchor is
# always kept as a fallback). Lower -> more positives per GT = denser
# supervision. The classic single-best-anchor YOLOv3 rule is the limit
# THRESH -> 1.0; 0.2 typically yields ~2-3 positives per GT.
ANCHOR_MATCH_THRESH = 0.3

# Focal loss for the OBJECTNESS term: scales each cell's loss by (1 - p_t)^gamma,
# down-weighting EASY cells (the flood of obvious background) so the gradient
# concentrates on HARD ones -- targets the low precision on hard classes that a
# uniform LAMBDA_NOOBJ can't fix (it weights every negative the same). alpha
# balances positives vs negatives (taking over LAMBDA_NOOBJ's old job, hence
# that is back to 1). With focal on, the objectness loss is normalized by the
# number of positives (RetinaNet convention). FOCAL_OBJ=False or gamma=0 -> BCE.
FOCAL_OBJ = True
FOCAL_GAMMA = 2.0     # focusing strength (RetinaNet default 2; 0 = plain BCE)
FOCAL_ALPHA = 0.25    # positive-class weight (RetinaNet default 0.25)

# -----------------------------------------------------------------------------
# Inference / evaluation thresholds
# -----------------------------------------------------------------------------
CONF_THRESH = 0.05       # drop predictions below this objectness*class score
NMS_IOU_THRESH = 0.45    # IoU threshold for non-maximum suppression

# Per-epoch mAP monitoring: full VOC2007 test eval adds ~1min/epoch. Set to an
# int N to estimate mAP on only the first N val batches (faster, biased proxy);
# None = full eval. The final best-checkpoint mAP is always computed in full.
MAP_EVAL_MAX_BATCHES = 60

