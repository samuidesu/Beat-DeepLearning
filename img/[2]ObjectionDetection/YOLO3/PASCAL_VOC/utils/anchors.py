"""Anchor boxes: k-means clustering of VOC box shapes (IoU metric).

The default config anchors are the classic COCO YOLOv3 anchors, whose size /
aspect-ratio distribution does not match PASCAL VOC. Re-clustering the 9 anchors
on the VOC train set (VOC2007 + VOC2012 trainval) yields anchors that fit the
data, which improves localization (notably mAP@0.75).

Boxes are clustered with the IoU distance (1 - IoU), matched at a shared corner
so only the shape (w, h) matters -- the standard YOLO anchor metric. Sizes are
in PIXELS at config.IMG_SIZE, the same space the model's anchors live in.

Run directly to compute anchors and print a config-ready ANCHORS block:
    python utils/anchors.py
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from torchvision.datasets import VOCDetection

# Make the project root importable whether run as a script or imported.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from dataset.voc import _parse_target  # noqa: E402


def collect_box_wh(splits=(("2007", "trainval"), ("2012", "trainval")),
                   keep_difficult=True):
    """Gather every GT box's (w, h) in pixels at config.IMG_SIZE.

    Boxes are parsed straight from the VOC XML (no image decode), so this is
    fast. Output: [N, 2] float array of (w, h) in the square IMG_SIZE space.
    """
    whs = []
    for year, image_set in splits:
        voc = VOCDetection(root=config.DATA_ROOT, year=year,
                           image_set=image_set, download=False)
        # torchvision stores the XML paths in `.targets` (new) or `.annotations`.
        ann_paths = getattr(voc, "targets", None) or getattr(voc, "annotations")
        for p in ann_paths:
            target = voc.parse_voc_xml(ET.parse(p).getroot())
            boxes = _parse_target(target, keep_difficult=keep_difficult)
            if boxes.numel() == 0:
                continue
            # normalized (w, h) -> pixels at IMG_SIZE (cols 3:5 are w, h).
            whs.append(boxes[:, 3:5].numpy() * config.IMG_SIZE)
    return np.concatenate(whs, axis=0)


def _iou_wh(boxes, clusters):
    """Shape-only IoU between boxes and clusters (matched at a shared corner).

    boxes [N,2], clusters [K,2] -> [N,K].
    """
    inter_w = np.minimum(boxes[:, None, 0], clusters[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 1], clusters[None, :, 1])
    inter = inter_w * inter_h
    area_b = (boxes[:, 0] * boxes[:, 1])[:, None]
    area_c = (clusters[:, 0] * clusters[:, 1])[None, :]
    return inter / (area_b + area_c - inter)


def kmeans_iou(boxes, k=9, seed=42, max_iter=300):
    """K-means with the IoU distance (1 - IoU). Returns clusters [k, 2] (w, h)."""
    rng = np.random.default_rng(seed)
    n = boxes.shape[0]
    clusters = boxes[rng.choice(n, k, replace=False)].astype(np.float64)
    last = np.full(n, -1)
    for _ in range(max_iter):
        nearest = (1.0 - _iou_wh(boxes, clusters)).argmin(axis=1)
        if (nearest == last).all():
            break
        for j in range(k):
            members = boxes[nearest == j]
            if len(members):
                clusters[j] = members.mean(axis=0)   # recompute centroid
        last = nearest
    return clusters


def avg_iou(boxes, clusters):
    """Mean best-IoU of every box to its nearest cluster (higher = better fit)."""
    return float(_iou_wh(boxes, clusters).max(axis=1).mean())


def to_config_anchors(clusters):
    """Sort clusters by area and split into 3 scales (small -> large).

    Output: list of 3 lists of 3 (w, h) int tuples, in config.ANCHORS layout
    (scale 0 = stride 8 = smallest objects).
    """
    clusters = clusters[np.argsort(clusters[:, 0] * clusters[:, 1])]
    rounded = [(int(round(w)), int(round(h))) for w, h in clusters]
    return [rounded[0:3], rounded[3:6], rounded[6:9]]


if __name__ == "__main__":
    wh = collect_box_wh()
    print(f"Collected {len(wh)} GT boxes from VOC07+12 trainval.")

    clusters = kmeans_iou(wh, k=9, seed=42)
    anchors = to_config_anchors(clusters)

    print(f"\nMean IoU (clustered) : {avg_iou(wh, clusters):.4f}")
    cur = np.array([a for scale in config.ANCHORS for a in scale], dtype=float)
    print(f"Mean IoU (current)   : {avg_iou(wh, cur):.4f}")

    print("\nConfig-ready ANCHORS (paste into config.py):")
    print("ANCHORS = [")
    names = ["stride 8  -> P3 (small)", "stride 16 -> P4 (medium)", "stride 32 -> P5 (large)"]
    for scale, name in zip(anchors, names):
        row = ", ".join(f"({w}, {h})" for w, h in scale)
        print(f"    [{row}],".ljust(40) + f"# {name}")
    print("]")
