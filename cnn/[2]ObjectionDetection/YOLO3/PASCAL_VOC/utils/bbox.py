"""Bounding-box utilities for YOLOv3.

Box format conventions used throughout:
  - "xywh"  = (center_x, center_y, width, height)
  - "xyxy"  = (x_min, y_min, x_max, y_max)
Coordinates may be normalized ([0,1]) or in pixels; the functions are unit-
agnostic as long as both inputs use the SAME units.

Functions:
  xywh_to_xyxy : convert center boxes -> corner boxes
  box_iou      : pairwise IoU between two box sets (used for the ignore mask)
  bbox_ciou    : Complete-IoU for aligned box pairs (used for the box loss)
  wh_iou       : shape-only IoU (used to match GT boxes to anchors)
"""

import math

import torch

_EPS = 1e-7


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert center boxes to corner boxes.

    Input:  boxes [..., 4] = (cx, cy, w, h).
    Output: boxes [..., 4] = (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of boxes (corner format).

    Input:
        boxes1: [N, 4] xyxy.
        boxes2: [M, 4] xyxy.
    Output:
        iou: [N, M], iou[i, j] = IoU(boxes1[i], boxes2[j]).
    """
    # Areas of each box: [N] and [M].
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersection rectangle corners, broadcast to [N, M, 2].
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # top-left
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # bottom-right
    wh = (rb - lt).clamp(min=0)                               # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]                           # [N, M]

    # IoU = inter / (areaA + areaB - inter).
    union = area1[:, None] + area2[None, :] - inter + _EPS
    return inter / union


def bbox_ciou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complete-IoU (CIoU) for ALIGNED box pairs (center format).

    CIoU = IoU - center_distance_term - aspect_ratio_term, which adds two
    penalties on top of IoU:
      - center distance normalized by the diagonal of the smallest enclosing box,
      - a width/height aspect-ratio consistency term.
    These give useful gradients even when boxes don't overlap. The box loss is
    then (1 - CIoU).

    Input:
        pred:   [N, 4] = (cx, cy, w, h), the predicted boxes (requires grad).
        target: [N, 4] = (cx, cy, w, h), the matched GT boxes.
    Output:
        ciou: [N], the CIoU of each pair (higher is better, range ~(-1, 1]).
    """
    # Corner coords for intersection / enclosing-box computations.
    p = xywh_to_xyxy(pred)
    t = xywh_to_xyxy(target)

    # --- IoU ---
    lt = torch.max(p[:, :2], t[:, :2])
    rb = torch.min(p[:, 2:], t[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = pred[:, 2] * pred[:, 3]
    area_t = target[:, 2] * target[:, 3]
    union = area_p + area_t - inter + _EPS
    iou = inter / union

    # --- Center-distance term: rho^2 / c^2 ---
    # rho^2 = squared distance between the two box centers.
    rho2 = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2
    # c^2 = squared diagonal of the smallest box enclosing both boxes.
    elt = torch.min(p[:, :2], t[:, :2])
    erb = torch.max(p[:, 2:], t[:, 2:])
    ewh = (erb - elt).clamp(min=0)
    c2 = ewh[:, 0] ** 2 + ewh[:, 1] ** 2 + _EPS

    # --- Aspect-ratio term: alpha * v ---
    # v measures how different the two boxes' aspect ratios are.
    v = (4 / (math.pi ** 2)) * (
        torch.atan(target[:, 2] / (target[:, 3] + _EPS))
        - torch.atan(pred[:, 2] / (pred[:, 3] + _EPS))
    ) ** 2
    # alpha is a positive trade-off weight; treated as a constant (no grad),
    # which is how the CIoU paper defines it.
    with torch.no_grad():
        alpha = v / (1 - iou + v + _EPS)

    return iou - rho2 / c2 - alpha * v


def wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """Shape-only IoU between width/height pairs (centers assumed aligned).

    Used to match each GT box to the best anchor: it ignores position and only
    compares box shapes.

    Input:
        wh1: [N, 2] = (w, h) of GT boxes.
        wh2: [M, 2] = (w, h) of anchors.
    Output:
        iou: [N, M], shape-IoU of each (gt, anchor) pair.
    """
    wh1 = wh1[:, None, :]  # [N, 1, 2]
    wh2 = wh2[None, :, :]  # [1, M, 2]
    # Intersection area assuming boxes share a center = product of min sides.
    inter = torch.min(wh1, wh2).prod(dim=2)        # [N, M]
    area1 = wh1.prod(dim=2)                         # [N, 1]
    area2 = wh2.prod(dim=2)                         # [1, M]
    return inter / (area1 + area2 - inter + _EPS)
