"""Backbone: feature extractor for FCOS.

This is the SAME ResNet backbone as the YOLO3 project (a torchvision ResNet
pretrained on ImageNet). FCOS, like YOLOv3, detects objects at 3 different
scales, so the backbone must expose 3 feature maps at strides 8, 16 and 32.

ResNet layout (for a 416x416 RGB input) and where we tap features:

    input            [B, 3, 416, 416]
    conv1 (s2)       [B, 64, 208, 208]   stride 2
    maxpool (s2)     [B, 64, 104, 104]   stride 4
    layer1           [B, 64, 104, 104]   stride 4
    layer2  -> C3    [B, 128, 52, 52]    stride 8   <- tap
    layer3  -> C4    [B, 256, 26, 26]    stride 16  <- tap
    layer4  -> C5    [B, 512, 13, 13]    stride 32  <- tap

C3/C4/C5 are the three feature maps handed to the neck (FPN).
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
)

# Supported backbones -> (constructor, weights enum, (C3, C4, C5) tap channels).
# ResNet-18 and -34 both use BasicBlock, so their tap channels are identical
# (128/256/512); only the number of blocks per stage differs. (ResNet-50+ use
# Bottleneck -> 512/1024/2048; add here with the right channels to support them.)
_RESNET_ARCHS = {
    "resnet18": (resnet18, ResNet18_Weights, (128, 256, 512)),
    "resnet34": (resnet34, ResNet34_Weights, (128, 256, 512)),
}


class ResNetBackbone(nn.Module):
    """ResNet backbone (resnet18 / resnet34) -> 3 multi-scale feature maps.

    Args:
        arch (str): which ResNet, "resnet18" or "resnet34". Both share the same
            tap channels, so the neck/head need no change when switching.
        pretrained (bool): if True, load ImageNet-pretrained weights.
        freeze (bool): if True, freeze all backbone parameters (stage-1 of a
            two-stage finetune: train only neck/heads first, unfreeze later).

    Attributes:
        out_channels (tuple[int, int, int]): channel counts of (C3, C4, C5),
            e.g. (128, 256, 512). The neck reads this to size its own convs.
        strides (tuple[int, int, int]): the strides (8, 16, 32) of (C3, C4, C5).
    """

    # Strides of the 3 taps (fixed by the ResNet layout). out_channels depends
    # on the arch and is set per-instance in __init__.
    strides = (8, 16, 32)

    def __init__(self, arch: str = "resnet18", pretrained: bool = True, freeze: bool = False):
        super().__init__()
        if arch not in _RESNET_ARCHS:
            raise ValueError(
                f"unsupported backbone {arch!r}; choose from {list(_RESNET_ARCHS)}")
        ctor, weights_enum, self.out_channels = _RESNET_ARCHS[arch]
        self.arch = arch

        # ---- Load the torchvision ResNet ------------------------------------
        # <Weights>.DEFAULT == the best available ImageNet weights.
        # weights=None gives a randomly-initialized network.
        weights = weights_enum.DEFAULT if pretrained else None
        net = ctor(weights=weights)

        # ---- Split the ResNet into the pieces we need ------------------------
        # "stem" = everything before the residual stages: conv1 -> bn1 -> relu
        # -> maxpool. After the stem the feature map is at stride 4.
        # We deliberately DROP net.avgpool and net.fc (those are for
        # classification; a detector does not use them).
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # stride 4,   64 channels (not tapped)
        self.layer2 = net.layer2  # stride 8,  128 channels -> C3
        self.layer3 = net.layer3  # stride 16, 256 channels -> C4
        self.layer4 = net.layer4  # stride 32, 512 channels -> C5

        # Optionally freeze the whole backbone for two-stage finetuning.
        if freeze:
            self.freeze()

    def forward(self, x: torch.Tensor):
        """Run the backbone.

        Input:
            x: image batch, shape [B, 3, H, W] (H, W ideally multiples of 32).

        Output:
            (c3, c4, c5): tuple of 3 feature maps
                c3: [B, 128, H/8,  W/8]   (stride 8,  fine   -> small objects)
                c4: [B, 256, H/16, W/16]  (stride 16, medium)
                c5: [B, 512, H/32, W/32]  (stride 32, coarse -> large objects)
        """
        # Stem: stride 2 conv + stride 2 maxpool -> overall stride 4.
        x = self.stem(x)        # [B, 64, H/4, W/4]
        # layer1 keeps the resolution (stride still 4); not used by the neck.
        x = self.layer1(x)      # [B, 64, H/4, W/4]
        # Each subsequent stage halves the spatial size and we tap its output.
        c3 = self.layer2(x)     # [B, 128, H/8,  W/8]
        c4 = self.layer3(c3)    # [B, 256, H/16, W/16]
        c5 = self.layer4(c4)    # [B, 512, H/32, W/32]
        return c3, c4, c5

    # ---- Helpers for two-stage finetuning -----------------------------------
    def freeze(self):
        """Freeze all backbone weights and put BatchNorm layers in eval mode.

        Freezing the parameters stops gradients; setting BN to eval() also
        stops its running mean/var from being updated, so the pretrained
        statistics are preserved while the neck/heads warm up.
        """
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfreeze(self):
        """Re-enable training of all backbone weights (stage-2 finetune)."""
        for p in self.parameters():
            p.requires_grad = True
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    def unfreeze_high_layers(self, layers=("layer3", "layer4")):
        """Unfreeze only the named high-level stages (for two-stage finetune).

        Lower stages (stem, layer1, layer2) keep their pretrained low-level
        features frozen; the deeper, more task-specific stages get finetuned.

        Args:
            layers: which stages to unfreeze, e.g. ("layer3", "layer4").
        """
        name_to_module = {
            "layer1": self.layer1, "layer2": self.layer2,
            "layer3": self.layer3, "layer4": self.layer4,
        }
        for name in layers:
            module = name_to_module[name]
            for p in module.parameters():
                p.requires_grad = True
            # Re-enable BN running-stat updates for the unfrozen stages.
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()


# ---- Quick self-test: run this file directly to verify output shapes --------
# python models/backbone.py
if __name__ == "__main__":
    model = ResNetBackbone(arch="resnet34", pretrained=False)  # no download for a shape check
    dummy = torch.randn(2, 3, 416, 416)         # a fake batch of 2 images
    c3, c4, c5 = model(dummy)
    print("c3:", tuple(c3.shape), "(expected (2, 128, 52, 52))")
    print("c4:", tuple(c4.shape), "(expected (2, 256, 26, 26))")
    print("c5:", tuple(c5.shape), "(expected (2, 512, 13, 13))")
    print("out_channels:", model.out_channels, "strides:", model.strides)
