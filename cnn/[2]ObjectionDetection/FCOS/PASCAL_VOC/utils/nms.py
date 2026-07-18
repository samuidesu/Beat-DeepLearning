"""Decode raw FCOS outputs into final detections (decode + confidence filter
+ non-maximum suppression).

The decode here MUST match the decode used in the loss (losses/fcos_loss.py):
    d = exp(raw_ltrb)                (POSITIVE pixel distances; the raw values
                                      already include the head's per-level Scale)
    x1 = loc_x - d_l ,  y1 = loc_y - d_t     (corner box around the location)
    x2 = loc_x + d_r ,  y2 = loc_y + d_b
    ctr   = sigmoid(centerness logit)
    cls   = sigmoid(class logits)            (multi-label)
    score = max_class_prob * ctr ; label = argmax class

The `* ctr` factor is the FCOS trick: locations near a box border regress
poorly, but their class score alone doesn't know that. Multiplying by the
predicted centerness pulls those low-quality boxes' scores down so NMS/
thresholding removes them. (YOLOv3's `obj * cls` plays the same role.)

NMS is done per class via torchvision.ops.batched_nms. Output boxes are in
PIXEL coordinates at the network input size (0..img_size), corner (xyxy) format
-- the same contract as the YOLO3 project's postprocess().
"""

import math
import os
import sys

import torch
from torchvision.ops import batched_nms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.locations import make_locations  # noqa: E402

# Overflow guard for exp(): caps decoded distances at e^9.2 ~= 10,000 px, far
# beyond any real box at 416 input. This is purely numerical protection against
# inf/NaN from an unlucky early-training logit, NOT a modeling choice.
_LOG_MAX = math.log(1e4)


def _decode_one_level(pred: torch.Tensor, stride: int, img_size: int) -> torch.Tensor:
    """Decode one pyramid level's raw predictions into candidate detections.

    Input:
        pred: raw predictions [B, H, W, 5 + C]
              (last dim = [l, t, r, b, centerness, class logits...]).
        stride: this level's stride (8 / 16 / 32) -- generates the locations.
        img_size: network input size in pixels (for clamping boxes).
    Output:
        dets: [B, H*W, 6] = [x1, y1, x2, y2, score, label] in pixel coords.
    """
    B, H, W, D = pred.shape
    C = D - 5

    # Locations [H*W, 2]; row-major order matches reshape(B, H*W, ...) below.
    locs = make_locations(H, W, stride, pred.device)
    xs = locs[:, 0].unsqueeze(0)                    # [1, H*W] broadcast over B
    ys = locs[:, 1].unsqueeze(0)

    # ltrb distances: exp maps the raw values to positive pixels.
    d = pred[..., 0:4].reshape(B, -1, 4).clamp(max=_LOG_MAX).exp()  # [B,H*W,4]

    # Location +/- distances -> corner box, clamped to the image frame.
    x1 = (xs - d[..., 0]).clamp(0, img_size)
    y1 = (ys - d[..., 1]).clamp(0, img_size)
    x2 = (xs + d[..., 2]).clamp(0, img_size)
    y2 = (ys + d[..., 3]).clamp(0, img_size)

    # Detection confidence = best class prob * centerness.
    ctr = pred[..., 4].reshape(B, -1).sigmoid()             # [B, H*W]
    cls_prob = pred[..., 5:].reshape(B, -1, C).sigmoid()    # [B, H*W, C]
    cls_score, cls_label = cls_prob.max(dim=-1)             # [B, H*W] each
    score = cls_score * ctr

    dets = torch.stack([x1, y1, x2, y2, score, cls_label.float()], dim=-1)
    return dets                                             # [B, H*W, 6]


@torch.no_grad()
def postprocess(predictions, conf_thresh=config.CONF_THRESH,
                iou_thresh=config.NMS_IOU_THRESH, img_size=config.IMG_SIZE,
                max_det=300):
    """Turn raw model outputs into per-image detections after conf filter + NMS.

    Input:
        predictions: list of 3 raw outputs, each [B, H_l, W_l, 5 + C], ordered
            by stride to match config.STRIDES (8, 16, 32).
        conf_thresh: drop detections with score below this.
        iou_thresh: IoU threshold for NMS.
        img_size: input size (for clamping boxes to the frame).
        max_det: cap on detections kept per image.

    Output:
        list of length B, each a tensor [K, 6] = [x1, y1, x2, y2, score, label]
        (pixel coords, xyxy). K may be 0.
    """
    device = predictions[0].device

    # Decode every level and concatenate candidates per image -> [B, total, 6].
    dets = torch.cat(
        [_decode_one_level(p, s, img_size)
         for p, s in zip(predictions, config.STRIDES)],
        dim=1,
    )

    results = []
    for b in range(dets.shape[0]):
        d = dets[b]
        # 1) confidence threshold
        d = d[d[:, 4] > conf_thresh]
        if d.numel() == 0:
            results.append(torch.zeros((0, 6), device=device))
            continue
        # 2) per-class NMS (batched_nms offsets boxes by class so classes don't
        #    suppress each other).
        keep = batched_nms(d[:, :4], d[:, 4], d[:, 5].long(), iou_thresh)
        results.append(d[keep[:max_det]])
    return results
