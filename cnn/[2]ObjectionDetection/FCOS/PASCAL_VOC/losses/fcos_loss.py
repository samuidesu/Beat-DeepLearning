"""Multi-level FCOS loss: target assignment + the 3 loss terms.

This file is where FCOS actually differs from YOLOv3. Everything upstream
(data, backbone, FPN idea) and downstream (NMS, mAP) is nearly identical
between the two projects; the label-assignment rule and the loss are the model.

The loss has three parts, computed over all pyramid levels at once:

  cls loss : sigmoid focal loss over ALL locations (positives + background).
             FCOS has NO objectness output -- the per-class logits directly
             face the ~10k-background-per-object imbalance, so focal loss is
             part of the core design (in YOLOv3 it was an add-on for the
             objectness term). Normalized by the number of positives.
  reg loss : (1 - GIoU) between the decoded predicted box and its GT box,
             POSITIVE locations only. Each positive is weighted by its
             centerness TARGET and the sum is normalized by the total
             centerness weight: well-centered locations (which can regress
             accurately) dominate; border locations contribute little.
  ctr loss : BCE-with-logits between the centerness prediction and the
             centerness target, positives only. Teaches the head to KNOW how
             off-center it is, so inference can down-rank border predictions.

Target assignment (which locations are "positive") -- the anchor-free rule
that replaces YOLOv3's anchor matching. A location (x, y) on level L is a
positive for GT box (x1, y1, x2, y2) iff ALL of:

  1. INSIDE:       (x, y) falls inside the box -- its four side-distances
                       l = x-x1,  t = y-y1,  r = x2-x,  b = y2-y
                   are all > 0. (These distances ARE the regression target.)
  2. CENTER SAMPLING (optional refinement, config.CENTER_SAMPLING): (x, y)
                   also falls within radius*stride of the box CENTER (clipped
                   to the box). Border locations regress poorly; excluding
                   them gives cleaner positives.
  3. SCALE RANGE:  max(l, t, r, b) lies in level L's regression range
                   (config.REGRESSION_RANGES) -- small boxes are learned by
                   the fine stride-8 level, big boxes by stride-32. This is
                   the anchor-free replacement for "which anchor shape fits".

  Ambiguity rule: if a location satisfies all three for SEVERAL boxes
  (overlapping GTs), it takes the box with the SMALLEST area -- the small
  object is the one only representable at this location; the big one has
  plenty of other locations.

Centerness target (for positives), from the GT distances:

    centerness = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )  in (0, 1]

  = 1 exactly at the box center, -> 0 toward any border.

All box math here is in PIXELS at the network input scale (img_size=416):
targets arrive normalized [0,1] and are scaled up once. (The YOLO3 loss kept
normalized coords because its anchors were normalized; FCOS's regression
ranges are defined in pixels, so pixels are the natural unit here.)
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the project root importable (for `import config` and `utils.*`)
# whether this file is imported as a package or run directly as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.bbox import xywh_to_xyxy, bbox_giou  # noqa: E402
from utils.locations import make_locations  # noqa: E402

_INF = float("inf")
# Overflow guard for exp() -- MUST match utils/nms.py. Caps decoded distances
# at ~10,000 px; purely numerical protection, never binding for real boxes.
_LOG_MAX = math.log(1e4)


class FCOSLoss(nn.Module):
    """FCOS loss (focal classification + GIoU box + centerness).

    Args:
        strides: per-level strides, e.g. [8, 16, 32]. Order must match the
            model outputs.
        regression_ranges: per-level (low, high) bounds on max(l,t,r,b) in
            input pixels; decides which level owns each GT box.
        num_classes: number of classes (VOC = 20).
        img_size: network input size in pixels (e.g. 416).
        center_sampling: if True, positives must also be near the GT center.
        center_radius: center-sampling radius in units of the level's stride.
        focal_gamma / focal_alpha: focal loss parameters for classification.
        lambda_*: loss-term weights.
    """

    COMPONENTS = ("cls", "reg", "ctr")

    def __init__(self, strides=config.STRIDES,
                 regression_ranges=config.REGRESSION_RANGES,
                 num_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE,
                 center_sampling=config.CENTER_SAMPLING,
                 center_radius=config.CENTER_SAMPLING_RADIUS,
                 focal_gamma=config.FOCAL_GAMMA, focal_alpha=config.FOCAL_ALPHA,
                 lambda_cls=config.LAMBDA_CLS, lambda_reg=config.LAMBDA_REG,
                 lambda_ctr=config.LAMBDA_CTR):
        super().__init__()
        self.strides = list(strides)
        self.num_classes = num_classes
        self.img_size = img_size
        self.center_sampling = center_sampling
        self.center_radius = center_radius
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_ctr = lambda_ctr

        # Per-level (low, high) as a [num_levels, 2] buffer so it moves to the
        # right device with the module (inf survives the tensor round-trip).
        ranges = torch.tensor([[lo, hi] for lo, hi in regression_ranges],
                              dtype=torch.float32)
        self.register_buffer("regression_ranges", ranges)

    # ------------------------------------------------------------------ #
    # Focal loss (elementwise)                                           #
    # ------------------------------------------------------------------ #
    def _focal_loss_map(self, logits, targets):
        """Per-element sigmoid focal loss. No reduction.

        Focal = alpha_t * (1 - p_t)^gamma * BCE, where p_t is the predicted
        probability of the TRUE label and alpha_t = alpha for positives,
        (1 - alpha) for negatives. The (1 - p_t)^gamma factor down-weights easy
        (already-confident) locations so the gradient concentrates on hard
        ones; alpha balances positives vs the background flood.
        gamma=0, alpha=0.5 recovers (scaled) plain BCE.

        Input:  logits, targets of identical shape [..., C] (targets one-hot).
        Output: loss map of the same shape.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)        # prob of true label
        focal = (1.0 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return alpha_t * focal * bce

    # ------------------------------------------------------------------ #
    # Target assignment for ONE image                                    #
    # ------------------------------------------------------------------ #
    def _assign(self, gt_xyxy, gt_labels, locations, loc_strides, loc_ranges):
        """Assign every location of every level to a GT box or to background.

        Input:
            gt_xyxy:    [G, 4] GT boxes, xyxy in input pixels (G >= 1).
            gt_labels:  [G] long, class id of each GT.
            locations:  [L, 2] all levels' (x, y) locations, concatenated.
            loc_strides:[L] the stride each location belongs to.
            loc_ranges: [L, 2] the (low, high) regression range per location.

        Output:
            labels:      [L] long; class id for positives, -1 for background.
            reg_targets: [L, 4] the (l, t, r, b) distances to the assigned box
                         (only meaningful where labels >= 0).
        """
        L = locations.shape[0]
        G = gt_xyxy.shape[0]
        xs = locations[:, 0].unsqueeze(1)     # [L, 1] broadcast against [G]
        ys = locations[:, 1].unsqueeze(1)

        # ---- The regression target of every (location, GT) pair ----------
        # ltrb[i, j] = the 4 side-distances from location i to box j.
        l = xs - gt_xyxy[:, 0]                # [L, G]
        t = ys - gt_xyxy[:, 1]
        r = gt_xyxy[:, 2] - xs
        b = gt_xyxy[:, 3] - ys
        ltrb = torch.stack([l, t, r, b], dim=2)          # [L, G, 4]

        # ---- Condition 1: strictly inside the box -------------------------
        inside_box = ltrb.min(dim=2).values > 0          # [L, G]

        # ---- Condition 2: center sampling (optional) ----------------------
        if self.center_sampling and G > 0:
            cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2.0   # [G]
            cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2.0
            # Radius grows with the location's OWN stride: coarse levels get a
            # proportionally larger central zone (in pixels).
            radius = loc_strides.unsqueeze(1) * self.center_radius   # [L, 1]
            # The sampling sub-box = (center +/- radius) CLIPPED to the GT box
            # (so a tiny box never gets positives outside itself).
            sx1 = torch.maximum(cx - radius, gt_xyxy[:, 0])          # [L, G]
            sy1 = torch.maximum(cy - radius, gt_xyxy[:, 1])
            sx2 = torch.minimum(cx + radius, gt_xyxy[:, 2])
            sy2 = torch.minimum(cy + radius, gt_xyxy[:, 3])
            inside_center = (xs > sx1) & (xs < sx2) & (ys > sy1) & (ys < sy2)
            inside = inside_box & inside_center
        else:
            inside = inside_box

        # ---- Condition 3: the box's scale fits this location's level ------
        max_ltrb = ltrb.max(dim=2).values                # [L, G]
        in_range = ((max_ltrb >= loc_ranges[:, 0:1])
                    & (max_ltrb <= loc_ranges[:, 1:2]))  # [L, G]

        candidate = inside & in_range                    # [L, G]

        # ---- Ambiguity: choose the smallest-area GT among candidates ------
        areas = ((gt_xyxy[:, 2] - gt_xyxy[:, 0])
                 * (gt_xyxy[:, 3] - gt_xyxy[:, 1]))      # [G]
        # Non-candidates get area = inf, so min() only ever picks candidates.
        area_mat = areas.unsqueeze(0).expand(L, G).clone()
        area_mat[~candidate] = _INF
        min_area, matched_gt = area_mat.min(dim=1)       # [L] each
        pos = min_area < _INF                            # [L] bool

        # ---- Gather the per-location targets ------------------------------
        labels = gt_labels[matched_gt]                   # [L] (junk where ~pos)
        labels[~pos] = -1                                # -1 = background
        # Row i takes ltrb[i, matched_gt[i]] -- its own distances to its box.
        reg_targets = ltrb[torch.arange(L, device=ltrb.device), matched_gt]
        return labels, reg_targets

    # ------------------------------------------------------------------ #
    # Centerness target                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _centerness_target(reg_targets):
        """Compute the centerness target from GT (l, t, r, b) distances.

        Input:  reg_targets [P, 4] with all entries > 0 (positives only).
        Output: centerness [P] in (0, 1]; 1 at the box center, ->0 at borders.
        """
        lr = reg_targets[:, [0, 2]]                       # [P, 2] = (l, r)
        tb = reg_targets[:, [1, 3]]                       # [P, 2] = (t, b)
        # min/max ratio per axis; multiply the two axes; sqrt to soften.
        ratio = ((lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-8))
                 * (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-8)))
        return ratio.clamp(min=0).sqrt()

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, predictions, targets):
        """Compute the total loss and a per-component breakdown.

        Input:
            predictions: list of per-level raw model outputs, each
                [B, H_l, W_l, 5 + C] with last dim [l, t, r, b, ctr, cls...]
                (the reg values already include the head's per-level Scale).
            targets: [M, 6] = [batch_idx, class, cx, cy, w, h] (NORMALIZED),
                pooled over the whole batch (dataset/voc.py's collate format).

        Output:
            total: scalar tensor (differentiable).
            items: dict {"total","cls","reg","ctr"} of floats (the weighted
                   contributions, so the three parts sum to total).
        """
        device = predictions[0].device
        B = predictions[0].shape[0]
        C = self.num_classes

        # ---- 1) Locations for every level, concatenated -------------------
        # Per level: [H_l*W_l, 2]; also record each location's stride and
        # regression range, aligned with the same concatenation order.
        locs_list, stride_list, range_list = [], [], []
        for lvl, (pred, stride) in enumerate(zip(predictions, self.strides)):
            H, W = pred.shape[1], pred.shape[2]
            locs = make_locations(H, W, stride, device)              # [HW, 2]
            locs_list.append(locs)
            stride_list.append(torch.full((locs.shape[0],), float(stride),
                                          device=device))
            range_list.append(
                self.regression_ranges[lvl].unsqueeze(0).expand(locs.shape[0], 2))
        locations = torch.cat(locs_list, dim=0)          # [L, 2]
        loc_strides = torch.cat(stride_list, dim=0)      # [L]
        loc_ranges = torch.cat(range_list, dim=0)        # [L, 2]
        L = locations.shape[0]

        # ---- 2) Flatten predictions the same way ---------------------------
        # reshape(B, H*W, ...) is row-major, matching make_locations' order.
        reg_raw = torch.cat([p[..., 0:4].reshape(B, -1, 4) for p in predictions], dim=1)   # [B, L, 4]
        ctr_logits = torch.cat([p[..., 4].reshape(B, -1) for p in predictions], dim=1)     # [B, L]
        cls_logits = torch.cat([p[..., 5:].reshape(B, -1, C) for p in predictions], dim=1)  # [B, L, C]

        # ---- 3) Assign targets image by image ------------------------------
        cls_onehot = torch.zeros((B, L, C), device=device)           # focal target
        pos_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        reg_targets = torch.zeros((B, L, 4), device=device)
        for b in range(B):
            t = targets[targets[:, 0] == b]              # this image's GT rows
            if t.numel() == 0:
                continue                                 # no objects -> all background
            gt_labels = t[:, 1].long()                   # [G]
            # Normalized cxcywh -> pixel cxcywh -> pixel xyxy.
            gt_xyxy = xywh_to_xyxy(t[:, 2:6] * self.img_size)        # [G, 4]
            labels, regs = self._assign(gt_xyxy, gt_labels,
                                        locations, loc_strides, loc_ranges)
            pos = labels >= 0
            pos_mask[b] = pos
            reg_targets[b] = regs
            # One-hot classification target at the positives.
            cls_onehot[b][pos, labels[pos]] = 1.0

        num_pos = int(pos_mask.sum().item())
        n_pos_safe = max(num_pos, 1)

        # ---- 4) Classification: focal over ALL locations -------------------
        # Background locations have an all-zero one-hot row (there is no
        # explicit "background class" -- every class logit is pushed down).
        cls_loss = self._focal_loss_map(cls_logits, cls_onehot).sum() / n_pos_safe

        # ---- 5) Regression + centerness: positives only --------------------
        if num_pos > 0:
            # Gather positives across the whole batch at once. locations is
            # shared by every image, so expand it to [B, L, 2] before masking.
            pos_locs = locations.unsqueeze(0).expand(B, L, 2)[pos_mask]  # [P, 2]
            pos_reg_raw = reg_raw[pos_mask]                              # [P, 4]
            pos_reg_tgt = reg_targets[pos_mask]                          # [P, 4]

            # Centerness target from the GT distances (constant, no grad).
            ctr_tgt = self._centerness_target(pos_reg_tgt)               # [P]

            # Decode predictions: exp -> positive distances -> corner box.
            # MUST match utils/nms.py's decode.
            d = pos_reg_raw.clamp(max=_LOG_MAX).exp()                    # [P, 4]
            px, py = pos_locs[:, 0], pos_locs[:, 1]
            pred_boxes = torch.stack(
                [px - d[:, 0], py - d[:, 1], px + d[:, 2], py + d[:, 3]], dim=1)
            tgt_boxes = torch.stack(
                [px - pos_reg_tgt[:, 0], py - pos_reg_tgt[:, 1],
                 px + pos_reg_tgt[:, 2], py + pos_reg_tgt[:, 3]], dim=1)

            # GIoU loss, weighted by the centerness target and normalized by
            # the total centerness weight (the official-FCOS refinement):
            # well-centered positives -- the ones whose regression is actually
            # usable at inference -- dominate the box gradient.
            giou = bbox_giou(pred_boxes, tgt_boxes)                      # [P]
            reg_loss = ((1.0 - giou) * ctr_tgt).sum() / ctr_tgt.sum().clamp(min=1e-6)

            # Centerness: plain BCE against the (0, 1] target.
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_logits[pos_mask], ctr_tgt, reduction="sum") / n_pos_safe
        else:
            # No positives in the batch (rare): keep the graph connected with
            # zero-valued, differentiable placeholders.
            reg_loss = reg_raw.sum() * 0.0
            ctr_loss = ctr_logits.sum() * 0.0

        # ---- 6) Weighted total ---------------------------------------------
        total = (self.lambda_cls * cls_loss
                 + self.lambda_reg * reg_loss
                 + self.lambda_ctr * ctr_loss)

        items = {
            "cls": (self.lambda_cls * cls_loss).item(),
            "reg": (self.lambda_reg * reg_loss).item(),
            "ctr": (self.lambda_ctr * ctr_loss).item(),
            "total": total.item(),
        }
        return total, items


# ---- Quick self-test: run this file directly to sanity-check the loss -------
# python losses/fcos_loss.py
if __name__ == "__main__":
    import math
    torch.manual_seed(0)
    B, C = 2, config.NUM_CLASSES
    # Fake raw predictions for a 416 input (grids 52/26/13), requires_grad.
    # IMPORTANT: seed the CLASS logits (indices 5:) with the same bias prior the
    # real head uses (config.CLS_PRIOR), i.e. sigmoid(logit)=0.01. Without this,
    # random-N(0,1) class logits start every background location at p~=0.5 and
    # the focal loss sums to ~1400 -- a scary but ARTIFICIAL number that only
    # reflects an un-initialized head. With the prior, cls starts O(1), matching
    # the real model (see the integration numbers in the README): this is
    # exactly what the head's _init_weights does for you at construction.
    cls_bias = math.log(config.CLS_PRIOR / (1.0 - config.CLS_PRIOR))
    preds = []
    for (h, w) in [(52, 52), (26, 26), (13, 13)]:
        p = torch.randn(B, h, w, 5 + C)
        p[..., 5:] = cls_bias            # class logits -> P(class)=0.01 prior
        preds.append(p.requires_grad_(True))
    # Fake targets: [batch_idx, class, cx, cy, w, h] (normalized).
    # Which level a box lands on is decided by max(l,t,r,b); with center
    # sampling the positives sit near the box CENTER, where that max is about
    # HALF the larger side. E.g. the 0.30x0.40 box (125x166 px) gives ~83px
    # -> P4 (64..128); the 0.10x0.25 box (42x104 px) gives ~52px -> P3 (0..64).
    targets = torch.tensor([
        [0, 11, 0.50, 0.50, 0.30, 0.40],
        [0, 14, 0.20, 0.30, 0.10, 0.25],
        [1, 2, 0.70, 0.60, 0.20, 0.20],
    ], dtype=torch.float32)

    criterion = FCOSLoss()
    loss, items = criterion(preds, targets)
    print("loss items:", items)
    loss.backward()
    print("grad on preds[0]:", preds[0].grad is not None)

    # Empty-target edge case: everything should be background, loss finite.
    loss2, items2 = criterion([p.detach() for p in preds],
                              torch.zeros((0, 6)))
    print("empty-target loss items:", items2)
