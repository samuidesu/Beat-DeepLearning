"""Full YOLOv3 model: backbone -> neck -> heads.

Forward pass:
    image [B, 3, H, W]
      --backbone-->  (c3, c4, c5)
      --neck (FPN)-> (p3, p4, p5)
      --heads-->     3 raw prediction tensors, each [B, A, H_s, W_s, 5 + C]

The model returns the RAW predictions (no sigmoid/exp). Decoding into absolute
boxes happens later (in the loss for training, in detect/eval for inference).
"""

import torch
import torch.nn as nn

# Package-relative imports when used as `models.yolov3`; plain imports when this
# file is run directly as a script.
try:
    from .backbone import ResNet18Backbone
    from .neck import FPNNeck
    from .head import DetectionHead
except ImportError:
    from backbone import ResNet18Backbone
    from neck import FPNNeck
    from head import DetectionHead


class YOLOv3(nn.Module):
    """ResNet18-backbone YOLOv3 for PASCAL VOC.

    Args:
        num_classes (int): number of object classes (VOC = 20).
        num_anchors (int): anchors per scale (default 3).
        pretrained (bool): load ImageNet-pretrained backbone weights.
    """

    def __init__(self, num_classes: int = 20, num_anchors: int = 3, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone -> 3 feature maps (strides 8/16/32, channels 128/256/512).
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        # Neck fuses them top-down -> channels (64, 128, 256).
        self.neck = FPNNeck(in_channels=self.backbone.out_channels)
        # Heads -> raw predictions per scale.
        self.head = DetectionHead(
            in_channels=self.neck.out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor):
        """Run the detector.

        Input:
            x: image batch [B, 3, H, W] (H, W multiples of 32).

        Output:
            list of 3 raw prediction tensors (ordered by stride 8, 16, 32):
                [B, num_anchors, H/8,  W/8,  5 + num_classes]
                [B, num_anchors, H/16, W/16, 5 + num_classes]
                [B, num_anchors, H/32, W/32, 5 + num_classes]
        """
        feats = self.backbone(x)   # (c3, c4, c5)
        feats = self.neck(feats)   # (p3, p4, p5)
        preds = self.head(feats)   # [out_p3, out_p4, out_p5]
        return preds

    # ---- Two-stage finetuning helpers ---------------------------------------
    def freeze_backbone(self):
        """Stage 1: freeze the entire backbone (train only neck + head)."""
        self.backbone.freeze()

    def unfreeze_backbone_high(self, layers=("layer3", "layer4")):
        """Stage 2: unfreeze the high backbone stages (e.g. layer3/layer4)."""
        self.backbone.unfreeze_high_layers(layers)

    def set_bn_eval_on_frozen(self):
        """Keep BatchNorm layers whose params are frozen in eval mode.

        Call this AFTER model.train() each epoch. model.train() flips every
        BN back to training mode, which would let frozen layers keep updating
        their running mean/var -- undesirable. This restores eval mode for any
        BN whose affine weights are frozen, preserving the pretrained stats.
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
# python models/yolov3.py
if __name__ == "__main__":
    # pretrained=False avoids a network download for this shape check.
    model = YOLOv3(num_classes=20, num_anchors=3, pretrained=False)
    dummy = torch.randn(2, 3, 416, 416)
    outs = model(dummy)
    for i, o in enumerate(outs):
        print(f"scale {i}: {tuple(o.shape)}")
    print("expected: (2,3,52,52,25), (2,3,26,26,25), (2,3,13,13,25)")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"params: total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
