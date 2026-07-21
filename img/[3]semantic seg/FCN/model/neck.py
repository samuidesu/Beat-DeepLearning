"""Neck: FPN-style top-down fusion producing ONE stride-8 map for FCN.

The backbone gives us 3 feature maps:
    c3: [B, 128, H/8,  W/8]   (stride 8,  fine resolution, weak semantics)
    c4: [B, 256, H/16, W/16]  (stride 16)
    c5: [B, 512, H/32, W/32]  (stride 32, coarse resolution, strong semantics)

Like the FCOS/YOLOv3 necks, information flows TOP-DOWN: deep,
semantically-rich c5 is upsampled and merged into the shallower maps. The
difference is WHAT we keep. Detection ran heads on every pyramid level
(p3/p4/p5, one per object size range); segmentation has no size ranges -- it
just wants the finest map with everything fused in -- so this neck returns
ONLY p3 [B, 256, H/8, W/8].

This top-down feature fusion is the modern equivalent of FCN-8s' skip
connections: FCN-8s predicted class scores at stride 32/16/8 and SUMMED the
score maps; we instead sum FEATURES (richer than 21-channel scores) and
predict once at the end, in the head.

Recipe per merge step: a 1x1 "lateral" conv projects the backbone map to the
common width (256), the deeper level is 2x nearest-upsampled and ADDED (not
concatenated, so channels stay 256), then a ConvSet (the YOLOv3 5-conv
conv+BN+LeakyReLU block) cleans up the upsampling artifacts and mixes the two
sources. NOTE: the original FPN paper used a single plain 3x3 conv with no
norm/activation here; the heavier ConvSet is fine for segmentation (conv+BN+
ReLU stacks are the norm in seg decoders), it just costs more compute.
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
    """Top-down FPN that fuses (c3, c4, c5) into a single stride-8 map p3.

    Args:
        in_channels: channel counts of the backbone outputs (c3, c4, c5).
            Defaults to ResNet-18/34's (128, 256, 512).
        out_channels (int): width of the internal pyramid and of the returned
            p3 (default 256).

    Attributes:
        out_channels (int): channel count of the returned p3; the FCN head
            reads this to size its convs.
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
        self.smooth3 = ConvSet(out_channels, out_channels)
        self.smooth4 = ConvSet(out_channels, out_channels)
        self.smooth5 = ConvSet(out_channels, out_channels)

        # Nearest-neighbor upsampling doubles H,W (stride 32 -> 16 -> 8 grids).
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        

    def forward(self, feats):
        """Fuse the 3 backbone feature maps top-down into one stride-8 map.

        Input:
            feats: tuple (c3, c4, c5) from the backbone
                c3: [B, 128, H/8,  W/8]
                c4: [B, 256, H/16, W/16]
                c5: [B, 512, H/32, W/32]

        Output:
            p3: [B, out_channels, H/8, W/8] -- the single fused stride-8
                feature map, carrying c5/c4 semantics merged into c3's
                resolution. The FCN head turns it into per-pixel class
                logits and upsamples 8x back to input resolution.
                (p4/p5 are intermediate values only and are not returned.)
        """
        c3, c4, c5 = feats

        # Project every level to the common width first.
        m5 = self.lateral5(c5)                    # [B, 256, H/32, W/32]
        p5 = self.smooth5(m5)


        m4 = self.lateral4(c4) + self.upsample(p5)                     # [B, 256, H/16, W/16]
        p4 = self.smooth4(m4)

        m3 = self.lateral3(c3) + self.upsample(p4)                # [B, 256, H/8,  W/8]
        p3 = self.smooth3(m3)

        return p3


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/neck.py
if __name__ == "__main__":
    # Fake backbone outputs for a 416x416 input (batch of 2).
    c3 = torch.randn(2, 128, 52, 52)
    c4 = torch.randn(2, 256, 26, 26)
    c5 = torch.randn(2, 512, 13, 13)

    neck = FPNNeck(in_channels=(128, 256, 512), out_channels=256)
    p3 = neck((c3, c4, c5))  # a single fused map now, NOT a 3-tuple

    print("p3:", tuple(p3.shape), "(expected (2, 256, 52, 52))")
    print("out_channels:", neck.out_channels)
