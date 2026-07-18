"""Image + bounding-box transforms / augmentation.

Boxes are stored as a tensor of shape [N, 5] = [class_id, cx, cy, w, h], where
cx, cy, w, h are NORMALIZED to [0, 1] (relative to the image). Because the
coordinates are normalized, resizing the image does NOT change them -- only
geometric augmentations like horizontal flip do.

Every transform is a callable `transform(img, boxes) -> (img, boxes)` so the
image and its boxes always stay in sync. We Compose them into a pipeline.

Pipeline order matters:
    PIL-space ops (Resize, Flip, ColorJitter)  ->  ToTensor  ->  Normalize

(This file is shared verbatim with the YOLO3 project -- the data pipeline is
model-agnostic; only the loss/target assignment differs between the two.)
"""

import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as tvt
from PIL import Image


class Compose:
    """Chain several (img, boxes) transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes):
        # Apply each transform in order, threading both img and boxes through.
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class Resize:
    """Resize a PIL image to a fixed square size.

    Boxes are normalized, so they are left unchanged. Note this does NOT
    preserve aspect ratio (a simple, common choice). Letterbox padding could
    be added later if aspect-ratio distortion hurts accuracy.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, boxes):
        # Input img: PIL image; Output: PIL image of (size, size).
        img = TF.resize(img, [self.size, self.size])
        return img, boxes


class RandomHorizontalFlip:
    """Randomly mirror the image left<->right (and the boxes' x coordinate)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, boxes):
        # With probability p, flip the PIL image and mirror box centers:
        # a center at normalized cx moves to (1 - cx); width/height unchanged.
        if random.random() < self.p:
            img = TF.hflip(img)
            if boxes.numel() > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # column 1 is cx
        return img, boxes


class RandomAffine:
    """Random scale + translation jitter, keeping boxes in sync.

    Operates on the already-square PIL image (run it after Resize). This adds
    the positional/scale variety that plain flip + color jitter lack -- the
    main lever against the detection head overfitting to object *positions*.

    Each call samples a scale `s` and a pixel translation (tx, ty) and applies
    the forward map  x' = s*x + bx,  y' = s*y + by  to both the image and the
    box corners. `bx/by` keep the image centered (plus the random shift), so at
    s=1, t=0 this is the identity. Boxes are clipped to the frame and dropped if
    too little of them survives the crop.

    Args:
        scale: (min, max) multiplicative scale range. <1 zooms out (gray pad),
               >1 zooms in (crops the borders).
        translate: max shift as a fraction of image size, each axis.
        p: probability of applying the transform at all.
        fill: RGB pad color for revealed borders (114 = standard YOLO gray).
        min_visibility: drop a box if its clipped area is below this fraction
                        of its pre-clip (scaled) area.
        min_size_px: also drop boxes thinner than this many pixels.
    """

    def __init__(self, scale=(0.8, 1.2), translate=0.1, p=1.0,
                 fill=(114, 114, 114), min_visibility=0.2, min_size_px=2):
        self.scale = scale
        self.translate = translate
        self.p = p
        self.fill = fill
        self.min_visibility = min_visibility
        self.min_size_px = min_size_px

    def __call__(self, img, boxes):
        if random.random() >= self.p:
            return img, boxes
        S = img.size[0]  # square after Resize: width == height

        # Sample the forward map x' = s*x + bx (and likewise for y).
        s = random.uniform(self.scale[0], self.scale[1])
        max_t = self.translate * S
        bx = (1.0 - s) * S / 2.0 + random.uniform(-max_t, max_t)
        by = (1.0 - s) * S / 2.0 + random.uniform(-max_t, max_t)

        # PIL wants the INVERSE map (dest -> src): x = (x' - bx) / s.
        inv = (1.0 / s, 0.0, -bx / s,
               0.0, 1.0 / s, -by / s)
        img = img.transform((S, S), Image.AFFINE, inv,
                            resample=Image.BILINEAR, fillcolor=self.fill)

        if boxes.numel() == 0:
            return img, boxes

        # Move box corners with the same forward map (in pixel space).
        cx, cy = boxes[:, 1] * S, boxes[:, 2] * S
        w, h = boxes[:, 3] * S, boxes[:, 4] * S
        x1, y1 = s * (cx - w / 2) + bx, s * (cy - h / 2) + by
        x2, y2 = s * (cx + w / 2) + bx, s * (cy + h / 2) + by

        area_before = (x2 - x1) * (y2 - y1)            # pre-clip scaled area
        x1c, y1c = x1.clamp(0, S), y1.clamp(0, S)
        x2c, y2c = x2.clamp(0, S), y2.clamp(0, S)
        new_w, new_h = (x2c - x1c), (y2c - y1c)
        area_after = new_w * new_h

        keep = (
            (new_w >= self.min_size_px)
            & (new_h >= self.min_size_px)
            & (area_after >= self.min_visibility * area_before.clamp(min=1e-6))
        )

        new = torch.stack([
            boxes[:, 0],                       # class id (unchanged)
            (x1c + x2c) / 2 / S,               # cx
            (y1c + y2c) / 2 / S,               # cy
            new_w / S,                         # w
            new_h / S,                         # h
        ], dim=1)
        return img, new[keep]


class ColorJitter:
    """Randomly perturb brightness/contrast/saturation/hue (image only)."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        # Wrap torchvision's ColorJitter, which operates on the PIL image.
        self.jitter = tvt.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, boxes):
        # Photometric only: boxes are untouched.
        return self.jitter(img), boxes


class ToTensor:
    """Convert a PIL image to a float tensor and ensure boxes are a tensor."""

    def __call__(self, img, boxes):
        # img: PIL [H,W,3] -> tensor [3,H,W] with values scaled to [0, 1].
        img = TF.to_tensor(img)
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return img, boxes


class Normalize:
    """Normalize an image tensor with the given per-channel mean/std."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, boxes):
        # img: tensor [3,H,W] -> normalized tensor [3,H,W]. Boxes unchanged.
        img = TF.normalize(img, self.mean, self.std)
        return img, boxes


def get_train_transforms(img_size: int, mean, std) -> Compose:
    """Training pipeline: resize + light augmentation + normalize.

    Input:  img_size (int), mean/std (per-channel lists for normalization).
    Output: a Compose that maps (PIL image, boxes[N,5]) ->
            (normalized tensor [3,img_size,img_size], boxes[N,5]).
    """
    return Compose([
        Resize(img_size),
        # Geometric jitter: scale + translate. The key augmentation for
        # detection -- it varies object position/size so the model can't just
        # memorize where objects sit. Same settings as the YOLO3 project
        # (wider ranges were tried there and did NOT help).
        RandomAffine(scale=(0.8, 1.2), translate=0.1),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(),
        ToTensor(),
        Normalize(mean, std),
    ])


def get_eval_transforms(img_size: int, mean, std) -> Compose:
    """Eval/inference pipeline: resize + normalize only (no augmentation)."""
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(mean, std),
    ])
