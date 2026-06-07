"""Image + bounding-box transforms / augmentation.

Boxes are stored as a tensor of shape [N, 5] = [class_id, cx, cy, w, h], where
cx, cy, w, h are NORMALIZED to [0, 1] (relative to the image). Because the
coordinates are normalized, resizing the image does NOT change them -- only
geometric augmentations like horizontal flip do.

Every transform is a callable `transform(img, boxes) -> (img, boxes)` so the
image and its boxes always stay in sync. We Compose them into a pipeline.

Pipeline order matters:
    PIL-space ops (Resize, Flip, ColorJitter)  ->  ToTensor  ->  Normalize
"""

import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as tvt


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
