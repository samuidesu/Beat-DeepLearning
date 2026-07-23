"""Full U-Net model: ResNet encoder -> U-Net decoder -> segmentation head.

Forward pass:
    image [B, 3, H, W]
      --encoder--> (c1, c2, c3, c4, c5)      strides 2 / 4 / 8 / 16 / 32
      --decoder--> p1 [B, 64, H/2, W/2]      ONE fused stride-2 map
      --head-----> logits [B, 21, H, W]      per-pixel class logits

The model returns RAW logits (no softmax) at full input resolution. Training
feeds them straight into nn.CrossEntropyLoss(ignore_index=255); inference
takes argmax over dim 1 to get the per-pixel class-id mask.

What changed vs. the FCN project (same task, same data, same training loop):
  * neck (FPN, add + nearest-upsample, constant 256 channels, stops at
    stride 4)  ->  decoder (concat + learned ConvTranspose upsample, channels
    mirror the encoder, climbs to stride 2);
  * the encoder taps ONE extra map (c1, stride 2) to feed that last step;
  * the head shrinks to a single 1x1 classifier + 2x upsample.

The encoder attribute is still named `backbone` on purpose: train.py splits
stage-2 param groups by the "backbone." name prefix, and the freeze/unfreeze
recipe is identical to the FCN/FCOS/YOLO3 projects.
"""

import torch
import torch.nn as nn

# Package-relative imports when used as `model.unet`; plain imports when this
# file is run directly as a script.
try:
    from .encoder import ResNetBackbone
    from .decoder import UNetDecoder
    from .head import UNetHead
except ImportError:
    from encoder import ResNetBackbone
    from decoder import UNetDecoder
    from head import UNetHead


class UNet(nn.Module):
    """ResNet-encoder U-Net for PASCAL VOC 2012 semantic segmentation.

    Args:
        num_classes (int): 21 for VOC seg (20 object classes + background).
        pretrained (bool): load ImageNet-pretrained encoder weights.
        backbone (str): encoder arch, "resnet18" or "resnet34".
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True,
                 backbone: str = "resnet18"):
        super().__init__()
        self.num_classes = num_classes

        # Encoder -> 5 feature maps (strides 2/4/8/16/32). resnet18/34 share
        # the same tap channels (64/64/128/256/512), so the decoder is
        # unaffected by the arch choice.
        self.backbone = ResNetBackbone(arch=backbone, pretrained=pretrained)
        # Decoder: 4 UpBlocks (upsample + concat skip + DoubleConv) back to
        # ONE stride-2 map. Channel plan is read off the encoder.
        self.decoder = UNetDecoder(in_channels=self.backbone.out_channels)
        # Head: 1x1 per-pixel classifier + 2x bilinear upsample to input size.
        self.head = UNetHead(in_ch=self.decoder.out_channels,
                             num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the segmenter.

        Input:
            x: image batch [B, 3, H, W] (H, W multiples of 32).

        Output:
            raw per-pixel class logits [B, num_classes, H, W].
        """
        feats = self.backbone(x)    # (c1, c2, c3, c4, c5)
        p1 = self.decoder(feats)    # [B, 64, H/2, W/2]
        logits = self.head(p1)      # [B, 21, H, W]
        return logits

    # ---- Two-stage finetuning helpers ---------------------------------------
    # (Identical protocol to the FCN/FCOS/YOLO3 projects so the training recipe
    # carries over -- backbone-unfreeze was the biggest accuracy lever there.)
    def freeze_backbone(self):
        """Stage 1: freeze the entire encoder (train only decoder + head)."""
        self.backbone.freeze()

    def unfreeze_backbone_high(self, layers=("layer3", "layer4")):
        """Stage 2: unfreeze the high encoder stages (e.g. layer3/layer4)."""
        self.backbone.unfreeze_high_layers(layers)

    def unfreeze_backbone_all(self):
        """Stage 2 (full finetune): unfreeze the ENTIRE encoder (stem + all
        layers) so the pretrained features can fully adapt to segmentation.
        In the detection projects this full unfreeze beat the partial one."""
        self.backbone.unfreeze()

    def set_bn_eval_on_frozen(self):
        """Keep BatchNorm layers whose params are frozen in eval mode.

        Call this AFTER model.train() each epoch. model.train() flips every BN
        back to training mode, which would let frozen layers keep updating
        their running mean/var -- undesirable. This restores eval mode for any
        BN whose affine weights are frozen, preserving the pretrained stats.
        (The decoder's DoubleConvs also contain BatchNorm, but those are never
        frozen, so in practice only encoder BNs are affected.)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # A frozen BN has requires_grad == False on its weight.
                if m.weight is not None and not m.weight.requires_grad:
                    m.eval()

    def trainable_parameters(self):
        """Yield only the parameters that currently require gradients."""
        return (p for p in self.parameters() if p.requires_grad)


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/unet.py
if __name__ == "__main__":
    # pretrained=False avoids a network download for this shape check.
    model = UNet(num_classes=21, pretrained=False)
    dummy = torch.randn(2, 3, 416, 416)
    out = model(dummy)
    print("logits:", tuple(out.shape), "(expected (2, 21, 416, 416))")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"params: total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
