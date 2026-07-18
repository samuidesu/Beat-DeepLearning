"""FCOS detection head (PASCAL VOC, 20 classes).

ONE head is SHARED by all pyramid levels (vs. YOLOv3's one-head-per-scale):
the same convs slide over p3, p4 and p5. Scale specialization comes from the
label assignment (each level only trains on objects in its size range, see
config.REGRESSION_RANGES) plus a tiny per-level learnable scalar on the
regression output (the `Scale` module below).

For every location on every level the head predicts, ANCHOR-FREE:

    [l, t, r, b, centerness, class_0, ..., class_{C-1}]

      l, t, r, b : raw regression values; decoded later via exp() into the
                   POSITIVE pixel distances from this location to the box's
                   left/top/right/bottom sides. exp maps R -> R+, which is why
                   no activation is needed here (compare YOLOv3's anchor*exp(t)
                   for sizes -- same idea, but there is NO anchor to multiply).
      centerness : raw logit; sigmoid gives how close this location is to the
                   center of the object it predicts (1 = dead center, ->0 near
                   the border). At inference it multiplies the class score to
                   suppress the low-quality boxes predicted from border
                   locations. This is FCOS's replacement for objectness.
      class_0..  : raw per-class logits (sigmoid, multi-label; trained with
                   focal loss against the overwhelming background majority).

These are RAW values: no sigmoid/exp is applied here. Decoding happens in the
loss / postprocess so training can use numerically-stable losses on logits.

Output layout per level is [B, H, W, 5 + num_classes] -- deliberately the same
"last dim is the prediction vector" contract as the YOLO3 project's
[B, A, H, W, 5 + C], just WITHOUT the anchor axis A (anchor-free!). Index 4 is
the "quality" logit in both projects (objectness there, centerness here).

Two towers (classification / regression) of conv+GroupNorm+ReLU precede the
prediction convs. GroupNorm instead of BatchNorm because the head is shared
across levels and GN is independent of batch size (FCOS convention).
"""

import math

import torch
import torch.nn as nn


class Scale(nn.Module):
    """A single learnable scalar multiplier (one per pyramid level).

    The regression branch is SHARED across levels, but the magnitude of its
    targets differs wildly per level (P3 regresses <=64 px, P5 can regress
    400+ px). Multiplying the shared branch's raw output by a per-level
    learnable scalar (init 1.0) lets each level adapt the output range without
    needing its own branch. Costs 1 parameter per level.
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input/Output: any shape; every element is multiplied by the scalar."""
        return x * self.scale


def conv_gn_relu(in_ch: int, out_ch: int, num_groups: int = 32) -> nn.Sequential:
    """A single Conv3x3 -> GroupNorm -> ReLU block (the FCOS tower unit).

    Input:
        in_ch, out_ch: input / output channel counts (out_ch must be divisible
            by num_groups).
        num_groups: GroupNorm group count (FCOS default 32).
    Output:
        nn.Sequential preserving spatial size: [B,in_ch,H,W] -> [B,out_ch,H,W].
    """
    return nn.Sequential(
        # bias=False because the following GroupNorm has its own shift term.
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(num_groups, out_ch),
        nn.ReLU(inplace=True),
    )


class FCOSHead(nn.Module):
    """The shared FCOS head, run once per pyramid level.

    Args:
        in_ch (int): channels of every neck output (all levels equal, e.g. 256).
        num_classes (int): number of object classes (VOC = 20).
        num_convs (int): conv+GN+ReLU blocks per tower (FCOS default 4).
        num_levels (int): number of pyramid levels (3 here: p3/p4/p5); sets how
            many per-level `Scale` scalars to create.
        prior (float): classification bias-init prior (RetinaNet trick): start
            with P(class)=prior everywhere so the background flood doesn't
            drown training in early false positives. Same trick the YOLO3 head
            applies to its objectness logit.
    """

    def __init__(self, in_ch: int = 256, num_classes: int = 20,
                 num_convs: int = 4, num_levels: int = 3, prior: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        # Length of the per-location prediction vector: 4 box + 1 ctr + C classes.
        self.num_outputs = 5 + num_classes  # = 25 for VOC

        # ---- Two towers: classification and regression ----------------------
        # Separate stacks because the two tasks want different features
        # (semantics vs. geometry). Both are SHARED across pyramid levels.
        self.cls_tower = nn.Sequential(
            *[conv_gn_relu(in_ch, in_ch) for _ in range(num_convs)])
        self.reg_tower = nn.Sequential(
            *[conv_gn_relu(in_ch, in_ch) for _ in range(num_convs)])

        # ---- Prediction convs (plain 3x3, bias=True, no norm/activation) ----
        self.cls_pred = nn.Conv2d(in_ch, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_ch, 4, kernel_size=3, padding=1)
        # Centerness rides on the REGRESSION tower (the FCOS paper's follow-up
        # ablation found this slightly better than the cls tower: centerness is
        # a geometric quantity, like the regression).
        self.ctr_pred = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

        # One learnable Scale per level for the regression output.
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        self._init_weights(prior)

    def _init_weights(self, prior: float):
        """RetinaNet/FCOS-style init.

        All head convs: weights ~ N(0, 0.01), bias 0 -- small random weights so
        the towers start near-identity-ish and predictions start near 0
        (=> decoded distances exp(0)=1 px: tiny boxes, harmless).

        cls_pred bias: log(prior / (1 - prior)) ~= -4.6, so every class starts
        at sigmoid(-4.6) = prior = 0.01 ("background everywhere" assumption).
        Without this, ~10k locations/image start at P=0.5 for every class and
        the focal loss spends ages beating down the false-positive flood.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.cls_pred.bias.fill_(math.log(prior / (1.0 - prior)))

    def forward_single(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Run the shared head on ONE pyramid level.

        Input:
            x: neck feature for this level, [B, in_ch, H, W].
            level: pyramid level index (0=p3, 1=p4, 2=p5) -- selects which
                per-level Scale to apply to the regression output.

        Output:
            raw predictions [B, H, W, 5 + num_classes], last-dim layout
            [l, t, r, b, centerness, class_0..class_{C-1}].
        """
        cls_feat = self.cls_tower(x)          # [B, in_ch, H, W]
        reg_feat = self.reg_tower(x)          # [B, in_ch, H, W]

        cls_out = self.cls_pred(cls_feat)                  # [B, C, H, W]
        # Per-level Scale on the raw regression values; exp() happens in the
        # loss / postprocess, so what we emit here is scale * raw.
        reg_out = self.scales[level](self.reg_pred(reg_feat))  # [B, 4, H, W]
        ctr_out = self.ctr_pred(reg_feat)                  # [B, 1, H, W]

        # Pack into one tensor: channels [l,t,r,b, ctr, cls...] -> last dim.
        out = torch.cat([reg_out, ctr_out, cls_out], dim=1)   # [B, 5+C, H, W]
        # [B, 5+C, H, W] -> [B, H, W, 5+C]; .contiguous() because the loss
        # will reshape/index this tensor.
        return out.permute(0, 2, 3, 1).contiguous()

    def forward(self, feats):
        """Run the shared head on all pyramid levels.

        Input:
            feats: tuple (p3, p4, p5) from the neck, each [B, in_ch, H_l, W_l].

        Output:
            list of 3 raw prediction tensors (ordered by stride 8, 16, 32):
                out[0]: [B, H/8,  W/8,  25]   (small objects)
                out[1]: [B, H/16, W/16, 25]   (medium objects)
                out[2]: [B, H/32, W/32, 25]   (large objects)
        """
        return [self.forward_single(f, level) for level, f in enumerate(feats)]


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python models/head.py
if __name__ == "__main__":
    # Fake neck outputs for a 416x416 input (batch of 2), all 256 channels.
    p3 = torch.randn(2, 256, 52, 52)
    p4 = torch.randn(2, 256, 26, 26)
    p5 = torch.randn(2, 256, 13, 13)

    head = FCOSHead(in_ch=256, num_classes=20, num_convs=4, num_levels=3)
    outs = head((p3, p4, p5))

    for i, o in enumerate(outs):
        print(f"level {i}: {tuple(o.shape)}")
    print("expected: (2,52,52,25), (2,26,26,25), (2,13,13,25)")
    # The classification bias init should make initial class probs ~= 0.01.
    with torch.no_grad():
        print("initial mean class prob:",
              float(outs[0][..., 5:].sigmoid().mean()), "(expected ~0.01)")
