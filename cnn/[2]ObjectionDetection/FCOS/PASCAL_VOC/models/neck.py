"""Neck: classic FPN (lateral 1x1 + top-down sum + 3x3 smooth) for FCOS.

The backbone gives us 3 feature maps:
    c3: [B, 128, H/8,  W/8]   (stride 8,  fine resolution, weak semantics)
    c4: [B, 256, H/16, W/16]  (stride 16)
    c5: [B, 512, H/32, W/32]  (stride 32, coarse resolution, strong semantics)

Like the YOLOv3 neck, information flows TOP-DOWN: deep, semantically-rich c5
is upsampled and merged into the shallower maps. Two deliberate differences
from the YOLO3 project's neck:

  1. UNIFORM output width. FCOS runs a single SHARED detection head over all
     pyramid levels, so every level must end up with the SAME channel count
     (default 256). The YOLOv3 neck used per-scale widths (128/192/256) because
     each scale had its own head.
  2. SUM fusion + plain convs (the original FPN recipe): a 1x1 "lateral" conv
     projects each backbone map to the common width, the deeper level is
     upsampled and ADDED (not concatenated), and a final 3x3 "smooth" conv
     cleans up the upsampling artifacts. No BN and no activation on these convs
     (FPN/RetinaNet/FCOS convention -- normalization lives in the head's
     GroupNorm instead); compare the YOLOv3 neck's conv+BN+LeakyReLU ConvSets.
"""

import torch
import torch.nn as nn


class FPNNeck(nn.Module):
    """Top-down FPN that turns (c3, c4, c5) into (p3, p4, p5) of equal width.

    Args:
        in_channels: channel counts of the backbone outputs (c3, c4, c5).
            Defaults to ResNet-18/34's (128, 256, 512).
        out_channels (int): the SINGLE width shared by all pyramid outputs
            (default 256, the FCOS/RetinaNet standard).

    Attributes:
        out_channels (int): the pyramid width in use; the shared detection
            head reads this.
    """

    def __init__(self, in_channels=(128, 256, 512), out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        c3, c4, c5 = in_channels

        # Lateral 1x1 convs: project each backbone map to the common width.
        # bias=True because there is no norm layer after these convs.
        self.lateral3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5, out_channels, kernel_size=1)

        # 3x3 smoothing convs: applied AFTER the top-down sum to reduce the
        # aliasing/checkerboard artifacts of nearest-neighbor upsampling.
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Nearest-neighbor upsampling doubles H,W (stride 32 -> 16 -> 8 grids).
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, feats):
        """Fuse the 3 backbone feature maps top-down.

        Input:
            feats: tuple (c3, c4, c5) from the backbone
                c3: [B, 128, H/8,  W/8]
                c4: [B, 256, H/16, W/16]
                c5: [B, 512, H/32, W/32]

        Output:
            (p3, p4, p5): fused pyramid features for the shared FCOS head,
                ALL with `out_channels` channels:
                p3: [B, 256, H/8,  W/8]   (stride 8,  small objects)
                p4: [B, 256, H/16, W/16]  (stride 16, medium objects)
                p5: [B, 256, H/32, W/32]  (stride 32, large objects)
        """
        c3, c4, c5 = feats

        # Project every level to the common width first.
        m5 = self.lateral5(c5)                    # [B, 256, H/32, W/32]
        m4 = self.lateral4(c4)                    # [B, 256, H/16, W/16]
        m3 = self.lateral3(c3)                    # [B, 256, H/8,  W/8]

        # Top-down pathway: upsample the deeper map and ADD it in. Because all
        # maps share the same width, sum fusion works without extra convs.
        m4 = m4 + self.upsample(m5)               # [B, 256, H/16, W/16]
        m3 = m3 + self.upsample(m4)               # [B, 256, H/8,  W/8]

        # Smooth each fused map with a 3x3 conv.
        p5 = self.smooth5(m5)
        p4 = self.smooth4(m4)
        p3 = self.smooth3(m3)

        return p3, p4, p5


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python models/neck.py
if __name__ == "__main__":
    # Fake backbone outputs for a 416x416 input (batch of 2).
    c3 = torch.randn(2, 128, 52, 52)
    c4 = torch.randn(2, 256, 26, 26)
    c5 = torch.randn(2, 512, 13, 13)

    neck = FPNNeck(in_channels=(128, 256, 512), out_channels=256)
    p3, p4, p5 = neck((c3, c4, c5))

    print("p3:", tuple(p3.shape), "(expected (2, 256, 52, 52))")
    print("p4:", tuple(p4.shape), "(expected (2, 256, 26, 26))")
    print("p5:", tuple(p5.shape), "(expected (2, 256, 13, 13))")
    print("out_channels:", neck.out_channels)
