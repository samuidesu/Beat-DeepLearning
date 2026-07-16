"""Inference entry point: run a trained YOLOv3 on image(s) and save the results
with boxes drawn.

This script lives in PASCAL_VOC/detect/. Results are written to
PASCAL_VOC/detect/results/, which is wiped and recreated fresh on every run.

For each input image we save up to two files, named so the pair sorts together:
    <id>_pred.<ext>  -- the model's predicted boxes
    <id>_gt.<ext>    -- the ground-truth boxes (only for VOC images that have a
                        matching .xml annotation)
Browsing results/ by filename then shows each image's prediction and GT
side by side.

Usage (run from the PASCAL_VOC project root):
    python detect/detect.py --img path/to/image.jpg
    python detect/detect.py --voc-random 10           # 10 random VOC2007 test images
    python detect/detect.py --dir path/to/folder --conf 0.3
"""

import os
import sys
import glob
import random
import shutil
import argparse
import xml.etree.ElementTree as ET

import torch
from PIL import Image

# This file now sits in PASCAL_VOC/detect/, so the project root is its parent's
# parent. Put it on sys.path so `import config`, `models`, `utils`, `train` ...
# resolve regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from models.yolov3 import YOLOv3  # noqa: E402
from dataset.transforms import get_eval_transforms  # noqa: E402
from utils.nms import postprocess  # noqa: E402
from utils.viz import draw_detections  # noqa: E402
from train import get_device  # noqa: E402  reuse the device picker

# Image extensions we accept when given a folder.
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# Default output folder: PASCAL_VOC/detect/results/ (sits next to this file).
_DETECT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_DETECT_DIR, "results")
# class name -> integer id (same order/ids as config.VOC_CLASSES), for GT parsing.
_CLASS_TO_IDX = {name: i for i, name in enumerate(config.VOC_CLASSES)}


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv3 detection on images")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", help="path to a single image")
    src.add_argument("--dir", help="path to a folder of images")
    src.add_argument("--voc-random", type=int, metavar="N",
                     help="randomly sample N images from the VOC2007 test split")
    p.add_argument("--seed", type=int, default=None,
                   help="optional random seed for reproducible --voc-random sampling")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt")
    p.add_argument("--out", default=_RESULTS_DIR,
                   help="output folder (wiped and recreated fresh each run)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    p.add_argument("--iou", type=float, default=config.NMS_IOU_THRESH, help="NMS IoU threshold")
    return p.parse_args()


def sample_voc_test(n, seed):
    """Randomly pick `n` image paths from the VOC2007 test split.

    Reads the official test id list at
        dataset/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt
    (one image id per line, e.g. "000001") and maps each id to its image file
    JPEGImages/<id>.jpg.

    Input:
        n: how many images to sample (capped at the split size).
        seed: optional RNG seed. None gives a different sample each run.
    Output:
        sorted list of image file paths (length <= n).
    """
    voc07 = os.path.join(config.DATA_ROOT, "VOCdevkit", "VOC2007")
    split_file = os.path.join(voc07, "ImageSets", "Main", "test.txt")
    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    # A supplied seed makes sampling reproducible; None uses system randomness.
    # A local RNG keeps the global random state untouched.
    random.Random(seed).shuffle(ids)
    ids = ids[:n]
    img_dir = os.path.join(voc07, "JPEGImages")
    return sorted(os.path.join(img_dir, f"{i}.jpg") for i in ids)


def load_voc_gt(image_path):
    """Load the ground-truth boxes for a VOC image, ready for draw_detections.

    Maps the image path to its annotation by swapping the folder and extension:
        .../VOC2007/JPEGImages/000001.jpg -> .../VOC2007/Annotations/000001.xml
    and parses the objects out of the XML.

    Input:
        image_path: path to a JPEGImages/<id>.jpg file.
    Output:
        gt: tensor [G, 6] = [x1, y1, x2, y2, score, label] in the ORIGINAL
            image's pixel coords. score is a constant 1.0 so the result can be
            fed straight to draw_detections. Returns None if no .xml exists
            (e.g. a user-supplied image that isn't part of VOC).
    """
    # JPEGImages -> Annotations, .jpg -> .xml
    ann_path = image_path.replace(
        os.sep + "JPEGImages" + os.sep, os.sep + "Annotations" + os.sep)
    ann_path = os.path.splitext(ann_path)[0] + ".xml"
    if not os.path.isfile(ann_path):
        return None

    root = ET.parse(ann_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in _CLASS_TO_IDX:
            continue
        bb = obj.find("bndbox")
        # VOC corners are 1-indexed pixels; subtract 1 to make them 0-based
        # (matches how dataset/voc.py parses GT for training).
        x1 = float(bb.findtext("xmin")) - 1.0
        y1 = float(bb.findtext("ymin")) - 1.0
        x2 = float(bb.findtext("xmax")) - 1.0
        y2 = float(bb.findtext("ymax")) - 1.0
        boxes.append([x1, y1, x2, y2, 1.0, _CLASS_TO_IDX[name]])

    if not boxes:
        return torch.zeros((0, 6))
    return torch.tensor(boxes, dtype=torch.float32)


def gather_images(args):
    """Return the list of image paths to process from --img / --dir / --voc-random."""
    if args.img:
        return [args.img]
    if args.voc_random:
        return sample_voc_test(args.voc_random, args.seed)
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
    # Start from a clean results folder: delete the previous one if it exists,
    # then recreate it, so each run's output isn't mixed with stale files.
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")

    # Build model + load weights.
    model = YOLOv3(num_classes=config.NUM_CLASSES,
                   num_anchors=config.NUM_ANCHORS_PER_SCALE,
                   pretrained=False, backbone=config.BACKBONE,
                   neck_channels=config.NECK_CHANNELS).to(device)
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
        # File stem (e.g. "000001") + extension, used to name the _pred/_gt pair
        # so the two files sort right next to each other when browsing.
        stem, ext = os.path.splitext(os.path.basename(path))

        # --- Prediction image ---
        result, dets = detect_one(model, transform, device, path, args.conf, args.iou)
        result.save(os.path.join(args.out, f"{stem}_pred{ext}"))
        msg = f"{stem}: {dets.shape[0]} pred"

        # --- Ground-truth image (only if a VOC annotation exists) ---
        gt = load_voc_gt(path)
        if gt is not None:
            gt_img = draw_detections(Image.open(path).convert("RGB"), gt, show_score=False)
            gt_img.save(os.path.join(args.out, f"{stem}_gt{ext}"))
            msg += f", {gt.shape[0]} gt"

        print(msg)


if __name__ == "__main__":
    main()
