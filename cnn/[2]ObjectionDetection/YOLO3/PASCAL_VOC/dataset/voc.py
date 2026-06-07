"""PASCAL VOC dataset for YOLOv3.

Responsibilities of this file:
  1. Download VOC into dataset/data/ (via torchvision).
  2. Wrap torchvision's VOCDetection and convert each sample into
        (image, boxes) where boxes is a tensor [N, 5] = [class, cx, cy, w, h]
     with cx, cy, w, h NORMALIZED to [0, 1].
  3. Provide a collate_fn that batches variable-numbers-of-boxes into a single
     targets tensor tagged with the sample index in the batch.

The anchor-based target assignment (which cell/anchor is responsible for each
box) is intentionally NOT done here -- it happens in the loss, so all anchor
logic lives in one place.
How to download: python dataset/voc.py --download
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

# Make the project root importable so `import config` works whether this file is
# run directly (python dataset/voc.py) or imported as a package (dataset.voc).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402

# Import our transforms (package-relative when imported, plain when run as script)
try:
    from .transforms import get_train_transforms, get_eval_transforms
except ImportError:
    from transforms import get_train_transforms, get_eval_transforms


# Map class name -> integer id, built once from config.VOC_CLASSES.
CLASS_TO_IDX = {name: i for i, name in enumerate(config.VOC_CLASSES)}


def _parse_target(target: dict, keep_difficult: bool = True) -> torch.Tensor:
    """Convert one VOC XML annotation (as a dict) into a boxes tensor.

    Input:
        target: the dict torchvision returns, of the form
            {"annotation": {"size": {"width","height",...},
                            "object": [{"name","difficult","bndbox":{...}}, ...]}}
        keep_difficult: if False, drop objects flagged 'difficult'.

    Output:
        boxes: tensor [N, 5], each row = [class_id, cx, cy, w, h] with
               cx, cy, w, h normalized to [0, 1]. N may be 0 (-> shape [0, 5]).
    """
    ann = target["annotation"]
    # Image size in pixels (strings in the XML -> float).
    img_w = float(ann["size"]["width"])
    img_h = float(ann["size"]["height"])

    # torchvision gives a list for multiple objects, but a single dict when
    # there is exactly one object. Normalize to a list either way.
    objs = ann.get("object", [])
    if isinstance(objs, dict):
        objs = [objs]

    boxes = []
    for obj in objs:
        # Optionally skip "difficult" objects (ambiguous / heavily occluded).
        if not keep_difficult and obj.get("difficult", "0") == "1":
            continue

        cls_id = CLASS_TO_IDX[obj["name"]]
        bb = obj["bndbox"]
        # VOC boxes are 1-indexed pixel corners; subtract 1 to make them 0-based.
        xmin = float(bb["xmin"]) - 1.0
        ymin = float(bb["ymin"]) - 1.0
        xmax = float(bb["xmax"]) - 1.0
        ymax = float(bb["ymax"]) - 1.0

        # Corner box (x1,y1,x2,y2) -> normalized center box (cx,cy,w,h).
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Drop degenerate boxes (zero/negative area can appear after rounding).
        if w <= 0 or h <= 0:
            continue
        boxes.append([cls_id, cx, cy, w, h])

    if len(boxes) == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


class VOCDataset(Dataset):
    """PASCAL VOC detection dataset returning (image_tensor, boxes[N,5]).

    Args:
        year (str): "2007" or "2012".
        image_set (str): "train" / "val" / "trainval" / "test" (test=2007 only).
        train (bool): use training augmentation if True, else eval transforms.
        download (bool): download the data if missing.
        keep_difficult (bool): keep objects flagged 'difficult'.
    """

    def __init__(self, year="2007", image_set="trainval", train=True,
                 download=False, keep_difficult=True):
        # torchvision handles the actual download + XML parsing of file lists.
        self.voc = VOCDetection(
            root=config.DATA_ROOT, year=year, image_set=image_set,
            download=download,
        )
        self.keep_difficult = keep_difficult
        # Pick the transform pipeline based on train vs. eval.
        if train:
            self.transforms = get_train_transforms(
                config.IMG_SIZE, config.IMAGENET_MEAN, config.IMAGENET_STD)
        else:
            self.transforms = get_eval_transforms(
                config.IMG_SIZE, config.IMAGENET_MEAN, config.IMAGENET_STD)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        """Return one processed sample.

        Output:
            img:   normalized tensor [3, IMG_SIZE, IMG_SIZE].
            boxes: tensor [N, 5] = [class, cx, cy, w, h], normalized.
        """
        # torchvision returns (PIL image, annotation dict).
        img, target = self.voc[idx]
        boxes = _parse_target(target, self.keep_difficult)
        # Apply resize + (augmentation) + normalize, keeping boxes in sync.
        img, boxes = self.transforms(img, boxes)
        return img, boxes


def voc_collate_fn(batch):
    """Collate a list of (img, boxes) into batched tensors.

    Different images have different numbers of boxes, so we cannot stack the
    box tensors directly. Instead we concatenate them and prepend a column with
    the image's index within the batch, so the loss can tell which boxes belong
    to which image.

    Input:
        batch: list of length B, each item = (img[3,S,S], boxes[N_i, 5]).

    Output:
        images:  tensor [B, 3, S, S].
        targets: tensor [M, 6] = [batch_idx, class, cx, cy, w, h],
                 where M = sum_i N_i (total boxes in the batch).
    """
    images, targets = [], []
    for batch_idx, (img, boxes) in enumerate(batch):
        images.append(img)
        if boxes.numel() > 0:
            # Prepend a column filled with this image's batch index.
            idx_col = torch.full((boxes.shape[0], 1), float(batch_idx))
            targets.append(torch.cat([idx_col, boxes], dim=1))  # [N_i, 6]

    images = torch.stack(images, dim=0)  # [B, 3, S, S]
    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)  # [M, 6]
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    return images, targets


def download_voc():
    """Download the VOC splits we need into dataset/data/.

    Downloads VOC2007 trainval + test and VOC2012 trainval (~2-3 GB total).
    Common protocol: train on VOC07+12 trainval, evaluate on VOC07 test.
    """
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    print(f"Downloading VOC into: {config.DATA_ROOT}")
    VOCDetection(config.DATA_ROOT, year="2007", image_set="trainval", download=True)
    VOCDetection(config.DATA_ROOT, year="2007", image_set="test", download=True)
    VOCDetection(config.DATA_ROOT, year="2012", image_set="trainval", download=True)
    print("Done.")


# ---- Run directly --------------------------------------------------------
#   python dataset/voc.py --download   # download the dataset
#   python dataset/voc.py              # offline self-test (no download)
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        download_voc()
        raise SystemExit

    # ---- Offline self-test: exercise parsing / transforms / collate ----
    from PIL import Image

    # 1) A hand-built VOC annotation with two objects on a 500x375 image.
    fake_target = {"annotation": {
        "size": {"width": "500", "height": "375", "depth": "3"},
        "object": [
            {"name": "dog", "difficult": "0",
             "bndbox": {"xmin": "100", "ymin": "50", "xmax": "300", "ymax": "350"}},
            {"name": "person", "difficult": "0",
             "bndbox": {"xmin": "10", "ymin": "20", "xmax": "120", "ymax": "300"}},
        ],
    }}
    boxes = _parse_target(fake_target)
    print("parsed boxes [class, cx, cy, w, h] (normalized):")
    print(boxes)

    # 2) Run the training transforms on a dummy image of the right size.
    dummy_img = Image.new("RGB", (500, 375), color=(123, 116, 103))
    tfm = get_train_transforms(config.IMG_SIZE, config.IMAGENET_MEAN, config.IMAGENET_STD)
    img_t, boxes_t = tfm(dummy_img, boxes.clone())
    print("\nafter transforms:")
    print("  image tensor:", tuple(img_t.shape), "(expected (3, 416, 416))")
    print("  boxes:", tuple(boxes_t.shape), "(still normalized, [N,5])")

    # 3) Collate two samples into a batch.
    images, targets = voc_collate_fn([(img_t, boxes_t), (img_t, boxes_t.clone())])
    print("\nafter collate:")
    print("  images:", tuple(images.shape), "(expected (2, 3, 416, 416))")
    print("  targets:", tuple(targets.shape), "(expected (4, 6): [batch_idx, class, cx, cy, w, h])")
    print(targets)
