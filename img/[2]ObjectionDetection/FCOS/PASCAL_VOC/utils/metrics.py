"""Evaluation metric: mean Average Precision (mAP).

We delegate the actual AP computation to torchmetrics'
`MeanAveragePrecision` (COCO-style), which is well-tested and avoids
re-implementing the matching / PR-curve logic ourselves.

It reports several numbers; for VOC the headline is mAP@0.5 (`map_50`).

(Identical to the YOLO3 project's metrics module -- the metric only sees the
postprocessed [x1,y1,x2,y2,score,label] detections, which both projects emit
in the same format; all the anchor-based vs anchor-free differences are
upstream of here.)
"""

import os
import sys

import torch
from torchmetrics.detection import MeanAveragePrecision

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.bbox import xywh_to_xyxy  # noqa: E402
from utils.nms import postprocess  # noqa: E402


def _targets_to_dicts(targets, batch_size, img_size, device):
    """Convert batched GT targets into torchmetrics' per-image dict format.

    Input:
        targets: [M, 6] = [batch_idx, class, cx, cy, w, h] (normalized).
        batch_size: number of images in this batch.
    Output:
        list of length batch_size, each {"boxes": [G,4] xyxy px, "labels": [G]}.
    """
    out = []
    for b in range(batch_size):
        t = targets[targets[:, 0] == b]
        if t.numel() == 0:
            out.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
            continue
        # Normalized (cx,cy,w,h) -> pixel (cx,cy,w,h) -> pixel xyxy.
        boxes = xywh_to_xyxy(t[:, 2:6] * img_size)
        out.append({"boxes": boxes.cpu(), "labels": t[:, 1].long().cpu()})
    return out


def _preds_to_dicts(detections):
    """Convert postprocess() output into torchmetrics' per-image dict format.

    Input:
        detections: list of [K,6] = [x1,y1,x2,y2,score,label] (pixel xyxy).
    Output:
        list of {"boxes": [K,4], "scores": [K], "labels": [K]}.
    """
    out = []
    for d in detections:
        out.append({
            "boxes": d[:, :4].cpu(),
            "scores": d[:, 4].cpu(),
            "labels": d[:, 5].long().cpu(),
        })
    return out


@torch.no_grad()
def compute_map(model, loader, device, conf_thresh=config.CONF_THRESH,
                iou_thresh=config.NMS_IOU_THRESH, img_size=config.IMG_SIZE,
                max_batches=None, verbose=True):
    """Run the model over `loader` and compute mAP.

    Input:
        model, loader, device as usual.
        conf_thresh / iou_thresh: postprocessing thresholds.
        max_batches: if set, only evaluate this many batches (quick check).
    Output:
        dict with {"map", "map_50", "map_75"} (floats).
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    # We feed many low-confidence boxes (needed for a full PR curve); torchmetrics
    # keeps the top-100 per image by COCO convention. Silence its info warning.
    metric.warn_on_many_detections = False

    try:
        from tqdm import tqdm
        iterator = tqdm(loader, desc="mAP eval", leave=False)
    except ImportError:
        iterator = loader

    for i, (images, targets) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        # Forward -> raw preds -> decode + NMS.
        preds = model(images)
        detections = postprocess(preds, conf_thresh=conf_thresh,
                                 iou_thresh=iou_thresh, img_size=img_size)
        # Feed this batch to the metric.
        metric.update(_preds_to_dicts(detections),
                      _targets_to_dicts(targets, images.shape[0], img_size, device))
        if max_batches is not None and (i + 1) >= max_batches:
            break

    res = metric.compute()
    out = {
        "map": float(res["map"]),
        "map_50": float(res["map_50"]),
        "map_75": float(res["map_75"]),
    }
    if verbose:
        print(f"mAP={out['map']:.4f}  mAP@0.5={out['map_50']:.4f}  mAP@0.75={out['map_75']:.4f}")
    return out
