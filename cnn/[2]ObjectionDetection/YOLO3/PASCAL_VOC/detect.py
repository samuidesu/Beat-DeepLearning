"""Inference entry point: run a trained YOLOv3 on image(s) and save the results
with boxes drawn.

Usage:
    python detect.py --img path/to/image.jpg
    python detect.py --dir path/to/folder --weights outputs/best.pt --conf 0.3
Outputs go to outputs/detections/ by default.
"""

import os
import glob
import argparse

import torch
from PIL import Image

import config
from models.yolov3 import YOLOv3
from dataset.transforms import get_eval_transforms
from utils.nms import postprocess
from utils.viz import draw_detections
from train import get_device  # reuse the device picker

# Image extensions we accept when given a folder.
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv3 detection on images")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", help="path to a single image")
    src.add_argument("--dir", help="path to a folder of images")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt")
    p.add_argument("--out", default=f"{config.OUTPUT_DIR}/detections", help="output folder")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    p.add_argument("--iou", type=float, default=config.NMS_IOU_THRESH, help="NMS IoU threshold")
    return p.parse_args()


def gather_images(args):
    """Return the list of image paths to process from --img or --dir."""
    if args.img:
        return [args.img]
    paths = []
    for ext in _IMG_EXTS:
        paths += glob.glob(os.path.join(args.dir, f"*{ext}"))
        paths += glob.glob(os.path.join(args.dir, f"*{ext.upper()}"))
    return sorted(paths)


@torch.no_grad()
def detect_one(model, transform, device, path, conf, iou):
    """Run detection on one image file.

    Input:
        model, transform, device; path to image; conf/iou thresholds.
    Output:
        (result_image, dets) where dets is [K,6] = [x1,y1,x2,y2,score,label]
        in the ORIGINAL image's pixel coordinates.
    """
    # Load the original image and remember its size for rescaling boxes back.
    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size

    # Preprocess (resize to IMG_SIZE + normalize). transform expects (img, boxes);
    # we pass an empty boxes tensor since there are no labels at inference.
    img_t, _ = transform(image, torch.zeros((0, 5)))
    img_t = img_t.unsqueeze(0).to(device)  # [1, 3, IMG_SIZE, IMG_SIZE]

    # Forward -> decode + confidence filter + NMS. dets are in IMG_SIZE pixels.
    preds = model(img_t)
    dets = postprocess(preds, conf_thresh=conf, iou_thresh=iou, img_size=config.IMG_SIZE)[0]

    # Rescale boxes from the (square) network input back to the original image.
    # We used a plain resize (not letterbox), so x and y scale independently.
    if dets.numel() > 0:
        sx = orig_w / config.IMG_SIZE
        sy = orig_h / config.IMG_SIZE
        dets = dets.clone().cpu()
        dets[:, [0, 2]] *= sx
        dets[:, [1, 3]] *= sy

    result = draw_detections(image, dets)
    return result, dets


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")

    # Build model + load weights.
    model = YOLOv3(num_classes=config.NUM_CLASSES,
                   num_anchors=config.NUM_ANCHORS_PER_SCALE,
                   pretrained=False).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights: {args.weights}")

    transform = get_eval_transforms(config.IMG_SIZE, config.IMAGENET_MEAN, config.IMAGENET_STD)

    paths = gather_images(args)
    if not paths:
        print("No images found.")
        return

    for path in paths:
        result, dets = detect_one(model, transform, device, path, args.conf, args.iou)
        out_path = os.path.join(args.out, os.path.basename(path))
        result.save(out_path)
        print(f"{os.path.basename(path)}: {dets.shape[0]} detections -> {out_path}")


if __name__ == "__main__":
    main()
