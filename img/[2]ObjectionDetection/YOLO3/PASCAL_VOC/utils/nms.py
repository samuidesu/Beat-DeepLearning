"""Decode raw YOLOv3 outputs into final detections (decode + confidence filter
+ non-maximum suppression).

The decode here MUST match the decode used in the loss (losses/yolo_loss.py):
    bx = (sigmoid(tx) + cell_x) / W          (normalized center x)
    by = (sigmoid(ty) + cell_y) / H
    bw = anchor_w_norm * exp(tw)             (normalized width)
    bh = anchor_h_norm * exp(th)
    obj  = sigmoid(objectness logit)
    cls  = sigmoid(class logits)             (multi-label)
    score = obj * max_class_prob ; label = argmax class

NMS is done per class via torchvision.ops.batched_nms. Output boxes are in
PIXEL coordinates at the network input size (0..img_size), corner (xyxy) format.
"""

import os
import sys

import torch
from torchvision.ops import batched_nms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402


def get_anchors_norm(device) -> torch.Tensor:
    """Return config anchors normalized to [0,1], shape [S, A, 2], on `device`."""
    a = torch.tensor(config.ANCHORS, dtype=torch.float32, device=device)
    return a / config.IMG_SIZE


def _decode_one_scale(pred: torch.Tensor, anchors_s: torch.Tensor, img_size: int) -> torch.Tensor:
    """Decode one scale's raw predictions into candidate detections.

    Input:
        pred: raw predictions [B, A, H, W, 5 + C].
        anchors_s: this scale's normalized anchors [A, 2] = (w, h).
        img_size: network input size in pixels.
    Output:
        dets: [B, A*H*W, 6] = [x1, y1, x2, y2, score, label] in pixel coords.
    """
    B, A, H, W, _ = pred.shape
    device = pred.device

    # Grid cell indices: gx = column (x), gy = row (y). Each [H, W].
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")

    # Decode center + size to NORMALIZED [0,1] coordinates.
    bx = (pred[..., 0].sigmoid() + gx) / W
    by = (pred[..., 1].sigmoid() + gy) / H
    bw = anchors_s[:, 0].view(1, A, 1, 1) * pred[..., 2].exp()
    bh = anchors_s[:, 1].view(1, A, 1, 1) * pred[..., 3].exp()

    # Objectness * best class probability = detection confidence.
    obj = pred[..., 4].sigmoid()                    # [B,A,H,W]
    cls_prob = pred[..., 5:].sigmoid()              # [B,A,H,W,C]
    cls_score, cls_label = cls_prob.max(dim=-1)     # [B,A,H,W] each
    score = obj * cls_score

    # Center box -> corner box, scaled from normalized to pixels.
    x1 = (bx - bw / 2) * img_size
    y1 = (by - bh / 2) * img_size
    x2 = (bx + bw / 2) * img_size
    y2 = (by + bh / 2) * img_size

    dets = torch.stack([x1, y1, x2, y2, score, cls_label.float()], dim=-1)  # [B,A,H,W,6]
    return dets.reshape(B, -1, 6)


@torch.no_grad()
def postprocess(predictions, conf_thresh=config.CONF_THRESH,
                iou_thresh=config.NMS_IOU_THRESH, img_size=config.IMG_SIZE,
                max_det=300):
    """Turn raw model outputs into per-image detections after conf filter + NMS.

    Input:
        predictions: list of 3 raw outputs, each [B, A, H, W, 5 + C].
        conf_thresh: drop detections with score below this.
        iou_thresh: IoU threshold for NMS.
        img_size: input size (for scaling boxes to pixels).
        max_det: cap on detections kept per image.

    Output:
        list of length B, each a tensor [K, 6] = [x1, y1, x2, y2, score, label]
        (pixel coords, xyxy). K may be 0.
    """
    device = predictions[0].device
    anchors_norm = get_anchors_norm(device)  # [S, A, 2]

    # Decode every scale and concatenate candidates per image -> [B, total, 6].
    dets = torch.cat(
        [_decode_one_scale(p, anchors_norm[s], img_size) for s, p in enumerate(predictions)],
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
