"""Full FCOS model: backbone -> FPN neck -> shared head.

Forward pass:
    image [B, 3, H, W]
      --backbone-->  (c3, c4, c5)
      --neck (FPN)-> (p3, p4, p5)     (all the same width, e.g. 256)
      --shared head-> 3 raw prediction tensors, each [B, H_l, W_l, 5 + C]
                      (last dim = [l, t, r, b, centerness, classes...])

The model returns the RAW predictions (no sigmoid/exp). Decoding into absolute
boxes happens later (in the loss for training, in detect/eval for inference).
Note the contrast with YOLOv3's output [B, A, H, W, 5+C]: no anchor axis A --
each location makes exactly ONE prediction.
"""

import torch
import torch.nn as nn

# Package-relative imports when used as `models.fcos`; plain imports when this
# file is run directly as a script.
try:
    from .backbone import ResNetBackbone
    from .neck import FPNNeck
    from .head import FCOSHead
except ImportError:
    from backbone import ResNetBackbone
    from neck import FPNNeck
    from head import FCOSHead


class FCOS(nn.Module):
    """ResNet-backbone FCOS for PASCAL VOC.

    Args:
        num_classes (int): number of object classes (VOC = 20).
        pretrained (bool): load ImageNet-pretrained backbone weights.
        backbone (str): backbone arch, "resnet18" or "resnet34".
        fpn_channels (int): the single width shared by all pyramid levels.
        num_head_convs (int): conv blocks per head tower (FCOS default 4).
        cls_prior (float): classification bias-init prior (default 0.01).
    """

    def __init__(self, num_classes: int = 20, pretrained: bool = True,
                 backbone: str = "resnet18", fpn_channels: int = 256,
                 num_head_convs: int = 4, cls_prior: float = 0.01):
        super().__init__()
        self.num_classes = num_classes

        # Backbone -> 3 feature maps (strides 8/16/32). resnet18/34 share the
        # same tap channels (128/256/512), so neck/head are unaffected by arch.
        self.backbone = ResNetBackbone(arch=backbone, pretrained=pretrained)
        # Neck fuses them top-down into a uniform-width pyramid.
        self.neck = FPNNeck(in_channels=self.backbone.out_channels,
                            out_channels=fpn_channels)
        # ONE shared head, run over every level (num_levels sets how many
        # per-level regression Scale scalars it holds).
        self.head = FCOSHead(
            in_ch=self.neck.out_channels,
            num_classes=num_classes,
            num_convs=num_head_convs,
            num_levels=len(self.backbone.strides),
            prior=cls_prior,
        )

    def forward(self, x: torch.Tensor):
        """Run the detector.

        Input:
            x: image batch [B, 3, H, W] (H, W multiples of 32).

        Output:
            list of 3 raw prediction tensors (ordered by stride 8, 16, 32):
                [B, H/8,  W/8,  5 + num_classes]
                [B, H/16, W/16, 5 + num_classes]
                [B, H/32, W/32, 5 + num_classes]
        """
        feats = self.backbone(x)   # (c3, c4, c5)
        feats = self.neck(feats)   # (p3, p4, p5)
        preds = self.head(feats)   # [out_p3, out_p4, out_p5]
        return preds

    # ---- Two-stage finetuning helpers ---------------------------------------
    # (Identical protocol to the YOLO3 project so training code carries over.)
    def freeze_backbone(self):
        """Stage 1: freeze the entire backbone (train only neck + head)."""
        self.backbone.freeze()

    def unfreeze_backbone_high(self, layers=("layer3", "layer4")):
        """Stage 2: unfreeze the high backbone stages (e.g. layer3/layer4)."""
        self.backbone.unfreeze_high_layers(layers)

    def unfreeze_backbone_all(self):
        """Stage 2 (full finetune): unfreeze the ENTIRE backbone (stem + all
        layers) so the pretrained features can fully adapt to detection. The
        YOLO3 experiments found the frozen ImageNet backbone was the accuracy
        bottleneck, so this is the default via config.STAGE2_UNFREEZE."""
        self.backbone.unfreeze()

    def set_bn_eval_on_frozen(self):
        """Keep BatchNorm layers whose params are frozen in eval mode.

        Call this AFTER model.train() each epoch. model.train() flips every
        BN back to training mode, which would let frozen layers keep updating
        their running mean/var -- undesirable. This restores eval mode for any
        BN whose affine weights are frozen, preserving the pretrained stats.
        (Only the backbone has BatchNorm; the FCOS head uses GroupNorm, which
        keeps no running stats and needs no such handling.)
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
# python models/fcos.py
if __name__ == "__main__":
    # pretrained=False avoids a network download for this shape check.
    model = FCOS(num_classes=20, pretrained=False)
    dummy = torch.randn(2, 3, 416, 416)
    outs = model(dummy)
    for i, o in enumerate(outs):
        print(f"level {i}: {tuple(o.shape)}")
    print("expected: (2,52,52,25), (2,26,26,25), (2,13,13,25)")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"params: total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
