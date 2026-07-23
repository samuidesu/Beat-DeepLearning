"""U-Net segmentation head (PASCAL VOC: 21 classes = 20 objects + background).

The decoder already did all the feature mixing (every UpBlock ends in a
DoubleConv), so the head is deliberately tiny -- the classic U-Net "OutConv":

    p1  [B, 64, H/2, W/2]
     -> 1x1 conv "classifier"  [B, 21, H/2, W/2]  (per-pixel class logits)
     -> 2x bilinear upsample   [B, 21, H,   W  ]  (back to input resolution)

Contrast with the FCN head: that one still ran a heavy ConvSet before
classifying, because its neck output needed one more round of fusion, and it
upsampled 4x (neck stopped at stride 4). Here the decoder climbs to stride 2,
so only a single 2x upsample remains -- the cheapest possible "last mile".

Predict-then-upsample (NOT upsample-then-predict) on purpose: upsampling the
21-channel logit map is ~3x cheaper than upsampling the 64-channel feature map.

Outputs are RAW logits, no softmax here (same "the loss decodes the raw
output" contract as before). The loss is nn.CrossEntropyLoss(ignore_index=255):
per-pixel softmax over the 21 classes, skipping the white object-boundary
pixels that VOC label PNGs mark as 255.

Output stays channel-first [B, 21, H, W]: CrossEntropyLoss wants [B, C, ...].
"""

import torch
import torch.nn as nn


class UNetHead(nn.Module):
    """U-Net head: fused stride-2 features -> full-resolution class logits.

    Args:
        in_ch (int): channels of the decoder output p1 (matches
            UNetDecoder.out_channels, 64 for ResNet-18/34).
        num_classes (int): 21 for VOC segmentation -- 20 object classes PLUS
            an explicit background class (index 0 in the VOC label PNGs).
    """

    def __init__(self, in_ch: int = 64, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes

        # The "classifier": a 1x1 conv mapping each pixel's 64-d feature
        # vector to 21 class logits. 1x1 is enough -- all spatial reasoning
        # already happened in the encoder/decoder; this is per-pixel linear
        # classification (exactly the original U-Net's final layer).
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

        # 2x BILINEAR upsample of the logits back to input resolution.
        # Bilinear, not nearest: nearest would copy every stride-2 logit into
        # a blocky 2x2 tile; bilinear interpolates smoothly between grid points.
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear",
                                     align_corners=False)

    def forward(self, p1: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel class logits from the decoder's fused map.

        Input:
            p1: [B, in_ch, H/2, W/2] from the decoder.

        Output:
            raw logits [B, num_classes, H, W], channel-first, ready for
            nn.CrossEntropyLoss(ignore_index=255). At inference, argmax over
            dim 1 gives the predicted class id per pixel.
        """
        return self.upsample2(self.classifier(p1))


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/head.py
if __name__ == "__main__":
    # Fake decoder output (stride-2 p1) for a 416x416 input (batch of 2).
    p1 = torch.randn(2, 64, 208, 208)

    head = UNetHead(in_ch=64, num_classes=21)
    out = head(p1)

    print("out:", tuple(out.shape), "(expected (2, 21, 416, 416))")
