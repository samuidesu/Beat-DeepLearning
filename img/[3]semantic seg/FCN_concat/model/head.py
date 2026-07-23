"""FCN segmentation head (PASCAL VOC: 21 classes = 20 objects + background).

Detection predicted a VECTOR per grid location (box + objectness/centerness +
class scores); segmentation is plain per-pixel classification: ONE class-logit
vector PER PIXEL. The head takes the neck's single fused stride-4 map and:

    p2  [B, 256, H/4, W/4]
     -> ConvSet fusion block   [B, 128, H/4, W/4]  (mix features once more)
     -> 1x1 conv "classifier"  [B, 21,  H/4, W/4]  (per-pixel class logits)
     -> 4x bilinear upsample   [B, 21,  H,   W]    (back to input resolution)

Predict-then-upsample (NOT upsample-then-predict) on purpose: upsampling the
21-channel logit map is ~12x cheaper than upsampling the 256-channel feature
map, and it is exactly what the original FCN did.

Outputs are RAW logits, no softmax here (same "the loss decodes the raw
output" contract as the detection projects). The loss will be
nn.CrossEntropyLoss(ignore_index=255): per-pixel softmax over the 21 classes,
skipping the white object-boundary pixels that VOC label PNGs mark as 255.
Two contrasts with the FCOS head worth remembering:

  * single-label softmax per pixel, NOT multi-label sigmoids per location --
    each pixel belongs to exactly one class;
  * "background" is an EXPLICIT class (index 0), not the absence of a
    prediction -- every pixel must get a label, so background cannot stay
    implicit like in detection.

Output stays channel-first [B, 21, H, W]: CrossEntropyLoss wants [B, C, ...],
so unlike the detection heads there is NO permute to channels-last.
"""

import torch
import torch.nn as nn

# Package-relative import when used as `model.head`; plain import when this
# file is run directly as a script.
try:
    from .neck import ConvSet
except ImportError:
    from neck import ConvSet


class FCNHead(nn.Module):
    """FCN head: fused stride-4 features -> full-resolution class logits.

    Args:
        in_ch (int): channels of the neck output p2 (matches
            FPNNeck.out_channels, default 256).
        num_classes (int): 21 for VOC segmentation -- 20 object classes PLUS
            an explicit background class (index 0 in the VOC label PNGs).
            NOT 20 like the detection projects!
    """

    def __init__(self, in_ch: int = 256, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes

        # ConvSet: one more round of context mixing at stride 4 (the same
        # 5-conv block the neck uses), then the "classifier": a 1x1 conv
        # mapping each pixel's 256-d feature vector to 21 class logits.
        # 1x1 is enough for the last layer -- all spatial reasoning already
        # happened in backbone/neck/ConvSet; this is per-pixel linear
        # classification.
        self.predict = nn.Sequential(
            ConvSet(in_ch, 128),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        # 4x BILINEAR upsample of the logits back to input resolution.
        # Bilinear, not nearest: nearest would copy every stride-4 logit into
        # a blocky 4x4 tile (jagged mask borders); bilinear interpolates
        # smoothly between grid points. The original FCN used a transposed
        # conv initialized to bilinear -- same effect, this is simpler.
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear",
                                     align_corners=False)

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel class logits from the neck's single fused map.

        Input:
            p2: [B, in_ch, H/4, W/4] from the neck (a single tensor -- no
                per-level tuple and no per-level loop like detection).

        Output:
            raw logits [B, num_classes, H, W], channel-first, ready for
            nn.CrossEntropyLoss(ignore_index=255). At inference, argmax over
            dim 1 gives the predicted class id per pixel.
        """
        return self.upsample4(self.predict(p2))


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/head.py
if __name__ == "__main__":
    # Fake neck output (stride-4 p2) for a 416x416 input (batch of 2).
    p2 = torch.randn(2, 256, 104, 104)

    head = FCNHead(in_ch=256, num_classes=21)
    out = head(p2)

    print("out:", tuple(out.shape), "(expected (2, 21, 416, 416))")
