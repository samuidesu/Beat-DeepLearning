"""Bounding-box utilities for FCOS.

Box format conventions used throughout:
  - "xywh"  = (center_x, center_y, width, height)
  - "xyxy"  = (x_min, y_min, x_max, y_max)
Coordinates may be normalized ([0,1]) or in pixels; the functions are unit-
agnostic as long as both inputs use the SAME units.

Functions:
  xywh_to_xyxy : convert center boxes -> corner boxes
  box_iou      : pairwise IoU between two box sets
  bbox_giou    : Generalized IoU for ALIGNED xyxy pairs (the FCOS box loss)

(The YOLO3 project's wh_iou / bbox_ciou are absent on purpose: wh_iou existed
only to match GT shapes against anchors -- FCOS has no anchors -- and FCOS's
standard regression loss is GIoU on the decoded corner boxes.)
"""

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


def bbox_giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU (GIoU) for ALIGNED box pairs (corner format).

    GIoU = IoU - (enclosing_area - union) / enclosing_area.

    The extra term measures how much "wasted" space the smallest box enclosing
    both boxes contains: it is 0 when the boxes coincide and approaches -1 as
    they move far apart, so -- unlike plain IoU, which is flat at 0 for any
    non-overlapping pair -- GIoU still provides a useful gradient when the
    prediction misses the target entirely. The box loss is then (1 - GIoU),
    with range [0, 2].

    Input:
        pred:   [N, 4] xyxy, the decoded predicted boxes (requires grad).
        target: [N, 4] xyxy, the matched GT boxes.
    Output:
        giou: [N], the GIoU of each pair (higher is better, range (-1, 1]).
    """
    # --- IoU ---
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + _EPS
    iou = inter / union

    # --- Smallest enclosing box ---
    elt = torch.min(pred[:, :2], target[:, :2])
    erb = torch.max(pred[:, 2:], target[:, 2:])
    ewh = (erb - elt).clamp(min=0)
    enclose = ewh[:, 0] * ewh[:, 1] + _EPS

    return iou - (enclose - union) / enclose
