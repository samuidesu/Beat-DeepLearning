"""Detection heads for YOLOv3 (PASCAL VOC, 20 classes).

The neck gives us 3 fused feature maps (p3, p4, p5) with channels (64, 128, 256).
One head is attached to each scale. A head predicts, for every grid cell and
every anchor, a vector of length (5 + num_classes):

    [tx, ty, tw, th, objectness, class_0, ..., class_{C-1}]

      tx, ty      : raw box-center offset inside the cell (decoded later via sigmoid)
      tw, th      : raw box size (decoded later via anchor * exp(t))
      objectness  : raw "is there an object here?" score (decoded via sigmoid)
      class_0..   : raw per-class scores (decoded via sigmoid, multi-label)

These are RAW values: no sigmoid/exp is applied here. Decoding happens in the
model / loss so that training can use numerically-stable losses on the logits.

Output layout per scale is [B, num_anchors, H, W, 5 + num_classes], which makes
the loss easy to write (slice the last dim).
"""

import math

import torch
import torch.nn as nn

# Reuse conv_bn_leaky from the neck. The try/except lets this file work both as
# part of the `models` package (from .neck) and when run directly as a script
# `python models/head.py` (in which case `models/` is on sys.path, so `neck`).
try:
    from .neck import conv_bn_leaky
except ImportError:
    from neck import conv_bn_leaky


class YOLOHead(nn.Module):
    """Detection head for a SINGLE scale.

    Args:
        in_ch (int): channels of this scale's neck feature (e.g. 64/128/256).
        num_anchors (int): anchors predicted per grid cell (default 3).
        num_classes (int): number of object classes (VOC = 20).
    """

    def __init__(self, in_ch: int, num_anchors: int = 3, num_classes: int = 20):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Length of the per-anchor prediction vector: 4 box + 1 obj + C classes.
        self.num_outputs = 5 + num_classes  # = 25 for VOC

        # 3x3 conv expands channels (in_ch -> 2*in_ch) to add capacity/context.
        self.conv = conv_bn_leaky(in_ch, in_ch * 2, 3)
        # 1x1 conv maps to the raw predictions. Plain Conv2d (bias=True, no BN,
        # no activation) because these are regression values / logits.
        self.pred = nn.Conv2d(in_ch * 2, num_anchors * self.num_outputs, kernel_size=1)

        self._init_obj_bias(prior=0.01)

    def _init_obj_bias(self, prior: float = 0.01):
        """Bias-init the objectness logit to a low prior (the RetinaNet trick).

        Default init leaves the objectness bias at ~0, so the head starts by
        predicting sigmoid(0)=0.5 objectness on EVERY cell (~10k per image).
        With the large negative/positive imbalance the model can't suppress all
        that background, and you get a flood of high-confidence false positives.
        Starting the objectness logit at log(prior/(1-prior)) instead means the
        head begins assuming "background everywhere" (P(obj)=prior=0.01) and only
        raises objectness where there's evidence -- the key fix for precision.

        The pred conv's output channels are laid out per anchor as
        [tx, ty, tw, th, obj, cls...], so the objectness channel is index 4
        within each anchor's block of `num_outputs`.
        """
        with torch.no_grad():
            b = self.pred.bias.view(self.num_anchors, self.num_outputs)
            b[:, 4].fill_(math.log(prior / (1.0 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the head on one feature map.

        Input:
            x: neck feature for this scale, [B, in_ch, H, W].

        Output:
            raw predictions, [B, num_anchors, H, W, 5 + num_classes].
            (For VOC: [B, 3, H, W, 25].)
        """
        B = x.shape[0]
        # Expand then project to raw predictions.
        x = self.conv(x)               # [B, 2*in_ch, H, W]
        x = self.pred(x)               # [B, A*(5+C), H, W]
        H, W = x.shape[2], x.shape[3]

        # Reshape so anchors and the (5+C) vector get their own axes.
        # [B, A*(5+C), H, W] -> [B, A, (5+C), H, W]
        x = x.view(B, self.num_anchors, self.num_outputs, H, W)
        # Move the (5+C) axis to the end: [B, A, H, W, (5+C)].
        # .contiguous() because the loss will index/reshape this tensor.
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


class DetectionHead(nn.Module):
    """Holds one YOLOHead per scale and runs them on the 3 neck features.

    Args:
        in_channels (tuple): channels of (p3, p4, p5) from the neck,
            default (64, 128, 256).
        num_anchors (int): anchors per cell per scale (default 3).
        num_classes (int): number of classes (VOC = 20).
    """

    def __init__(self, in_channels=(64, 128, 256), num_anchors: int = 3, num_classes: int = 20):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # One head per scale; ModuleList so parameters register correctly.
        self.heads = nn.ModuleList(
            [YOLOHead(c, num_anchors, num_classes) for c in in_channels]
        )

    def forward(self, feats):
        """Run all 3 heads.

        Input:
            feats: tuple (p3, p4, p5) from the neck
                p3: [B,  64, H/8,  W/8]
                p4: [B, 128, H/16, W/16]
                p5: [B, 256, H/32, W/32]

        Output:
            list of 3 raw prediction tensors, one per scale:
                out[0]: [B, 3, H/8,  W/8,  25]   (small objects)
                out[1]: [B, 3, H/16, W/16, 25]   (medium objects)
                out[2]: [B, 3, H/32, W/32, 25]   (large objects)
        """
        # Apply each scale's head to its matching feature map.
        return [head(f) for head, f in zip(self.heads, feats)]


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python models/head.py
if __name__ == "__main__":
    # Fake neck outputs for a 416x416 input (batch of 2).
    p3 = torch.randn(2, 64, 52, 52)
    p4 = torch.randn(2, 128, 26, 26)
    p5 = torch.randn(2, 256, 13, 13)

    head = DetectionHead(in_channels=(64, 128, 256), num_anchors=3, num_classes=20)
    outs = head((p3, p4, p5))

    for i, o in enumerate(outs):
        print(f"scale {i}: {tuple(o.shape)}")
    print("expected: (2,3,52,52,25), (2,3,26,26,25), (2,3,13,13,25)")
