"""Evaluation entry point: compute mAP of a trained checkpoint on VOC2007 test.

Usage:
    python eval.py                          # uses outputs/best.pt
    python eval.py --weights outputs/last.pt --conf 0.01 --device cpu
"""

import argparse

import torch
from torch.utils.data import DataLoader

import config
from models.yolov3 import YOLOv3
from dataset.voc import VOCDataset, voc_collate_fn
from utils.metrics import compute_map
from train import get_device  # reuse the device picker


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv3 (mAP) on VOC2007 test")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt", help="checkpoint path")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--conf", type=float, default=config.CONF_THRESH)
    p.add_argument("--iou", type=float, default=config.NMS_IOU_THRESH)
    p.add_argument("--max-batches", type=int, default=None, help="limit batches (quick check)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Val set = VOC2007 test (no augmentation).
    val_set = VOCDataset(year="2007", image_set="test", train=False)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=voc_collate_fn,
    )
    print(f"Val images: {len(val_set)}  batches: {len(val_loader)}")

    # Build model and load weights.
    model = YOLOv3(num_classes=config.NUM_CLASSES,
                   num_anchors=config.NUM_ANCHORS_PER_SCALE,
                   pretrained=False).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    compute_map(model, val_loader, device, conf_thresh=args.conf,
                iou_thresh=args.iou, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
