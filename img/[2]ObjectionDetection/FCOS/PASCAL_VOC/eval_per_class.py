"""Per-class AP@0.5 diagnostic on VOC2007 test.

Runs a trained checkpoint and reports, per class, AP@0.5 and recall@0.5 (sorted
worst-first) to show WHERE the overall mAP@0.5 is leaking. Reading it:

    low AP + low recall  -> the model MISSES these objects (a recall/detection
                            problem: capacity, level assignment, or just
                            hard/rare classes).
    low AP + high recall -> it finds them but with many false positives
                            (a precision/confidence/classification problem).

Also handy for comparing against the YOLO3 project's per-class table: the
weak classes there were bottle/pottedplant/cow/boat (small or clustered
objects) -- interesting to see whether anchor-free assignment changes that.

Usage:
    python eval_per_class.py                       # uses outputs/best.pt
    python eval_per_class.py --weights outputs/last.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

import config
from models.fcos import FCOS
from dataset.voc import VOCDataset, voc_collate_fn
from utils.nms import postprocess
from utils.metrics import _targets_to_dicts, _preds_to_dicts
from train import get_device  # reuse the device picker


def parse_args():
    p = argparse.ArgumentParser(description="Per-class AP@0.5 on VOC2007 test")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--conf", type=float, default=config.CONF_THRESH)
    p.add_argument("--iou", type=float, default=config.NMS_IOU_THRESH)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    val_set = VOCDataset(year="2007", image_set="test", train=False)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=voc_collate_fn,
    )
    print(f"Val images: {len(val_set)}  batches: {len(val_loader)}")

    model = FCOS(num_classes=config.NUM_CLASSES,
                 pretrained=False, backbone=config.BACKBONE,
                 fpn_channels=config.FPN_CHANNELS,
                 num_head_convs=config.NUM_HEAD_CONVS).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights: {args.weights}  (backbone={config.BACKBONE})")

    # iou_thresholds=[0.5] -> map_per_class IS AP@0.5 per class; class_metrics
    # turns on the per-class breakdown (incl. per-class recall via mar_100).
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  iou_thresholds=[0.5], class_metrics=True)
    metric.warn_on_many_detections = False

    try:
        from tqdm import tqdm
        iterator = tqdm(val_loader, desc="per-class AP", leave=False)
    except ImportError:
        iterator = val_loader

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        preds = model(images)
        dets = postprocess(preds, conf_thresh=args.conf, iou_thresh=args.iou,
                           img_size=config.IMG_SIZE)
        metric.update(_preds_to_dicts(dets),
                      _targets_to_dicts(targets, images.shape[0], config.IMG_SIZE, device))

    res = metric.compute()
    ap = res["map_per_class"]            # AP@0.5 per class
    rec = res.get("mar_100_per_class")   # recall@0.5 (<=100 dets) per class
    classes = res["classes"].tolist()

    rows = []
    for i, c in enumerate(classes):
        name = config.VOC_CLASSES[int(c)]
        a = float(ap[i])
        r = float(rec[i]) if rec is not None else float("nan")
        rows.append((name, a, r))
    rows.sort(key=lambda x: x[1])  # worst AP first

    print(f"\nOverall mAP@0.5 = {float(res['map']):.4f}\n")
    print(f"{'class':>14} {'AP@0.5':>8} {'recall@0.5':>11}")
    print("-" * 36)
    for name, a, r in rows:
        print(f"{name:>14} {a:>8.3f} {r:>11.3f}")


if __name__ == "__main__":
    main()
