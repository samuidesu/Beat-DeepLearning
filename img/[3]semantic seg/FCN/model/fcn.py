"""Full FCN model: backbone -> FPN neck -> segmentation head.

Forward pass:
    image [B, 3, H, W]
      --backbone-->  (c3, c4, c5)            strides 8 / 16 / 32
      --neck (FPN)-> p3 [B, 256, H/8, W/8]   ONE fused stride-8 map
      --head-------> logits [B, 21, H, W]    per-pixel class logits

The model returns RAW logits (no softmax) at full input resolution. Training
feeds them straight into nn.CrossEntropyLoss(ignore_index=255); inference
takes argmax over dim 1 to get the per-pixel class-id mask.

Note the contrast with the FCOS/YOLO3 projects: ONE output tensor, not one
per pyramid level -- the pyramid is fused inside the neck (FCN-8s style)
instead of being predicted from at every level.
"""

import torch
import torch.nn as nn

# Package-relative imports when used as `model.fcn`; plain imports when this
# file is run directly as a script.
try:
    from .backbone import ResNetBackbone
    from .neck import FPNNeck
    from .head import FCNHead
except ImportError:
    from backbone import ResNetBackbone
    from neck import FPNNeck
    from head import FCNHead


class FCN(nn.Module):
    """ResNet-backbone FCN for PASCAL VOC 2012 semantic segmentation.

    Args:
        num_classes (int): 21 for VOC seg (20 object classes + background).
        pretrained (bool): load ImageNet-pretrained backbone weights.
        backbone (str): backbone arch, "resnet18" or "resnet34".
        fpn_channels (int): width of the neck's fused output map (default 256).
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True,
                 backbone: str = "resnet18", fpn_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes

        # Backbone -> 3 feature maps (strides 8/16/32). resnet18/34 share the
        # same tap channels (128/256/512), so neck/head are unaffected by arch.
        self.backbone = ResNetBackbone(arch=backbone, pretrained=pretrained)
        # Neck fuses them top-down into ONE stride-8 map of fpn_channels width.
        self.neck = FPNNeck(in_channels=self.backbone.out_channels,
                            out_channels=fpn_channels)
        # Head: per-pixel classifier + 8x bilinear upsample to input size.
        self.head = FCNHead(in_ch=self.neck.out_channels,
                            num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the segmenter.

        Input:
            x: image batch [B, 3, H, W] (H, W multiples of 32).

        Output:
            raw per-pixel class logits [B, num_classes, H, W].
        """
        feats = self.backbone(x)    # (c3, c4, c5)
        p3 = self.neck(feats)       # [B, 256, H/8, W/8]
        logits = self.head(p3)      # [B, 21, H, W]
        return logits

    # ---- Two-stage finetuning helpers ---------------------------------------
    # (Identical protocol to the FCOS/YOLO3 projects so the training recipe
    # carries over -- backbone-unfreeze was the biggest accuracy lever there.)
    def freeze_backbone(self):
        """Stage 1: freeze the entire backbone (train only neck + head)."""
        self.backbone.freeze()

    def unfreeze_backbone_high(self, layers=("layer3", "layer4")):
        """Stage 2: unfreeze the high backbone stages (e.g. layer3/layer4)."""
        self.backbone.unfreeze_high_layers(layers)

    def unfreeze_backbone_all(self):
        """Stage 2 (full finetune): unfreeze the ENTIRE backbone (stem + all
        layers) so the pretrained features can fully adapt to segmentation.
        In the detection projects this full unfreeze beat the partial one."""
        self.backbone.unfreeze()

    def set_bn_eval_on_frozen(self):
        """Keep BatchNorm layers whose params are frozen in eval mode.

        Call this AFTER model.train() each epoch. model.train() flips every BN
        back to training mode, which would let frozen layers keep updating
        their running mean/var -- undesirable. This restores eval mode for any
        BN whose affine weights are frozen, preserving the pretrained stats.
        (Unlike FCOS -- whose head used GroupNorm -- the neck/head ConvSets
        here ALSO contain BatchNorm, but those are never frozen, so in
        practice only backbone BNs are affected.)
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
# python model/fcn.py
if __name__ == "__main__":
    # pretrained=False avoids a network download for this shape check.
    model = FCN(num_classes=21, pretrained=False)
    dummy = torch.randn(2, 3, 416, 416)
    out = model(dummy)
    print("logits:", tuple(out.shape), "(expected (2, 21, 416, 416))")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"params: total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
