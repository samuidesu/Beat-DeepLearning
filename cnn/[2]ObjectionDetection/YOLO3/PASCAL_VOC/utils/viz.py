"""Visualization helpers: draw predicted boxes on images.

(Training-curve plotting lives in train.py; this module is for detection
results.)
"""

import os
import sys
import colorsys

from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402


def _class_colors(n: int):
    """Return n visually-distinct RGB colors by spreading hues evenly."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


# One color per VOC class, computed once.
_COLORS = _class_colors(config.NUM_CLASSES)


def draw_detections(image: Image.Image, dets, class_names=config.VOC_CLASSES) -> Image.Image:
    """Draw detection boxes + labels onto a copy of `image`.

    Input:
        image: a PIL RGB image.
        dets: tensor/array [K, 6] = [x1, y1, x2, y2, score, label], with box
              coordinates already in THIS image's pixel space.
        class_names: list mapping class id -> name.

    Output:
        a new PIL image with the boxes drawn.
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for row in dets:
        x1, y1, x2, y2, score, label = [float(v) for v in row]
        label = int(label)
        color = _COLORS[label % len(_COLORS)]
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        text = f"{name} {score:.2f}"

        # Box.
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label background + text, placed just above the top-left corner.
        if font is not None:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            tw, th = 8 * len(text), 11
        ty = max(0, y1 - th - 2)
        draw.rectangle([x1, ty, x1 + tw + 2, ty + th + 2], fill=color)
        draw.text((x1 + 1, ty + 1), text, fill=(0, 0, 0), font=font)

    return img
