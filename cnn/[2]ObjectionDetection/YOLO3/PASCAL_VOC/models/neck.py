"""Neck: feature pyramid (FPN-style top-down fusion) for YOLOv3.

The backbone gives us 3 feature maps:
    c3: [B, 128, H/8,  W/8]   (stride 8,  fine resolution, weak semantics)
    c4: [B, 256, H/16, W/16]  (stride 16)
    c5: [B, 512, H/32, W/32]  (stride 32, coarse resolution, strong semantics)

The FPN sends information TOP-DOWN: it takes the deep, semantically-rich c5,
upsamples it, and merges it into the shallower maps so every output level has
both good semantics and good resolution.

Channel design follows the original YOLOv3 ratio 512:256:128 (= 4:2:1) for the
three scales, halved to suit the lighter ResNet-18 backbone:
    P5 = 256, P4 = 128, P3 = 64.
Each scale is fused by a Darknet "conv set": 1x1 (squeeze) and 3x3 (expand)
convs alternating, ending on a 1x1 so the output channel count is the squeezed
width. The 3 outputs (P3, P4, P5) are what the detection heads consume.
"""

import torch
import torch.nn as nn


def conv_bn_leaky(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    """A single Conv -> BatchNorm -> LeakyReLU(0.1) block.

    Input:
        in_ch, out_ch: input / output channel counts.
        kernel_size: 1 or 3 here. Padding = kernel_size // 2 keeps H,W unchanged.
    Output:
        nn.Sequential applying conv+bn+activation; preserves spatial size,
        i.e. [B, in_ch, H, W] -> [B, out_ch, H, W].
    """
    padding = kernel_size // 2
    return nn.Sequential(
        # bias=False because the following BatchNorm has its own shift term.
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ConvSet(nn.Module):
    """The YOLOv3 5-conv 'set' that compresses/fuses a feature map.

    Pattern: 1x1 squeeze -> 3x3 expand -> 1x1 squeeze -> 3x3 expand -> 1x1
    squeeze. The 1x1 layers reduce channels to `out_ch`; the 3x3 layers expand
    to 2*out_ch to mix spatial context, then squeeze back. Final output has
    `out_ch` channels and the same H,W as the input.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_leaky(in_ch,       out_ch,     1),  # squeeze: in_ch  -> out_ch
            conv_bn_leaky(out_ch,      out_ch * 2, 3),  # expand:  out_ch -> 2*out_ch
            conv_bn_leaky(out_ch * 2,  out_ch,     1),  # squeeze: 2*out  -> out_ch
            conv_bn_leaky(out_ch,      out_ch * 2, 3),  # expand:  out_ch -> 2*out_ch
            conv_bn_leaky(out_ch * 2,  out_ch,     1),  # squeeze: 2*out  -> out_ch (output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [B, in_ch, H, W]. Output: [B, out_ch, H, W] (H,W unchanged)."""
        return self.block(x)


class FPNNeck(nn.Module):
    """Top-down FPN that turns (c3, c4, c5) into (p3, p4, p5).

    Args:
        in_channels: channel counts of the backbone outputs (c3, c4, c5).
            Defaults to ResNet-18's (128, 256, 512).

    Attributes:
        out_channels (tuple[int,int,int]): channels of (p3, p4, p5) =
            (64, 128, 256). The detection heads read this.
    """

    out_channels = (64, 128, 256)  # (P3, P4, P5)

    def __init__(self, in_channels=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = in_channels          # backbone widths: 128, 256, 512
        p3, p4, p5 = self.out_channels    # neck widths:      64, 128, 256

        # --- Stride-32 branch (deepest): just compress c5 ---------------------
        self.conv_set5 = ConvSet(c5, p5)              # 512 -> 256

        # Lateral 1x1 to halve channels before sending features down to stride 16.
        self.lateral5 = conv_bn_leaky(p5, p5 // 2, 1)  # 256 -> 128
        # Nearest-neighbor upsampling doubles H,W (stride 32 -> stride 16 grid).
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Stride-16 branch: fuse upsampled-P5 with c4 ---------------------
        # Input channels = c4 + (p5 // 2) = 256 + 128 = 384.
        self.conv_set4 = ConvSet(c4 + p5 // 2, p4)     # 384 -> 128
        self.lateral4 = conv_bn_leaky(p4, p4 // 2, 1)  # 128 -> 64

        # --- Stride-8 branch: fuse upsampled-P4 with c3 ----------------------
        # Input channels = c3 + (p4 // 2) = 128 + 64 = 192.
        self.conv_set3 = ConvSet(c3 + p4 // 2, p3)     # 192 -> 64

    def forward(self, feats):
        """Fuse the 3 backbone feature maps top-down.

        Input:
            feats: tuple (c3, c4, c5) from the backbone
                c3: [B, 128, H/8,  W/8]
                c4: [B, 256, H/16, W/16]
                c5: [B, 512, H/32, W/32]

        Output:
            (p3, p4, p5): fused pyramid features for the 3 detection heads
                p3: [B,  64, H/8,  W/8]   (stride 8,  small objects)
                p4: [B, 128, H/16, W/16]  (stride 16, medium objects)
                p5: [B, 256, H/32, W/32]  (stride 32, large objects)
        """
        c3, c4, c5 = feats

        # Deepest level: compress c5 into the stride-32 detection feature.
        p5 = self.conv_set5(c5)                       # [B, 256, H/32, W/32]

        # Go down to stride 16: reduce channels, upsample, concat with c4.
        up5 = self.upsample(self.lateral5(p5))        # [B, 128, H/16, W/16]
        x4 = torch.cat([up5, c4], dim=1)              # [B, 128+256=384, H/16, W/16]
        p4 = self.conv_set4(x4)                       # [B, 128, H/16, W/16]

        # Go down to stride 8: reduce channels, upsample, concat with c3.
        up4 = self.upsample(self.lateral4(p4))        # [B, 64, H/8, W/8]
        x3 = torch.cat([up4, c3], dim=1)              # [B, 64+128=192, H/8, W/8]
        p3 = self.conv_set3(x3)                       # [B, 64, H/8, W/8]

        return p3, p4, p5


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python models/neck.py
if __name__ == "__main__":
    # Fake backbone outputs for a 416x416 input (batch of 2).
    c3 = torch.randn(2, 128, 52, 52)
    c4 = torch.randn(2, 256, 26, 26)
    c5 = torch.randn(2, 512, 13, 13)

    neck = FPNNeck(in_channels=(128, 256, 512))
    p3, p4, p5 = neck((c3, c4, c5))

    print("p3:", tuple(p3.shape), "(expected (2, 64, 52, 52))")
    print("p4:", tuple(p4.shape), "(expected (2, 128, 26, 26))")
    print("p5:", tuple(p5.shape), "(expected (2, 256, 13, 13))")
    print("out_channels:", neck.out_channels)
