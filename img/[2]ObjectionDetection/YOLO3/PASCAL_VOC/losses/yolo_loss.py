"""Multi-scale YOLOv3 loss.

The loss has three parts, summed over the 3 detection scales:

  box  loss : CIoU loss, computed ONLY on positive anchors (the anchor/cell
              assigned to a ground-truth box).
  obj  loss : sigmoid focal loss (down-weights easy cells) or plain BCE, per
              config.FOCAL_OBJ, on the objectness logit -- positives (target 1)
              and negatives (target 0). Negative anchors whose decoded box has
              high IoU (> ignore_thresh) with some GT are IGNORED -- neither
              positive nor counted as negative.
  cls  loss : BCEWithLogits per class, computed ONLY on positive anchors.

Target assignment (which anchors are "positive"):
  Each GT box is matched to EVERY one of the 9 anchors whose shape IoU (wh_iou)
  exceeds anchor_match_thresh, plus its single best anchor as a fallback so each
  GT always gets >= 1 positive. Each matched anchor's scale + the grid cell
  containing the GT center becomes a positive sample. This "multi-anchor" rule
  densifies positives vs. the original single-best-anchor YOLOv3 rule (more
  supervision -> better recall / localization). Setting the threshold to 1.0
  recovers the original single-best behavior.

All box math is done in NORMALIZED [0,1] image coordinates so predictions and
targets live in the same space.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the project root importable (for `import config` and `utils.bbox`)
# whether this file is imported as a package or run directly as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.bbox import xywh_to_xyxy, box_iou, bbox_ciou, wh_iou  # noqa: E402


class YOLOLoss(nn.Module):
    """Multi-scale YOLOv3 loss (CIoU box + objectness + classification).

    Args:
        anchors: list of `num_scales` lists, each of (w, h) anchor sizes in
                 PIXELS at `img_size`. Order must match the model outputs
                 (stride 8, 16, 32).
        strides: per-scale strides, e.g. [8, 16, 32].
        num_classes: number of classes (VOC = 20).
        img_size: network input size in pixels (e.g. 416).
        lambda_*: loss-term weights (default from config).
        ignore_thresh: IoU above which a negative anchor is ignored.
    """

    COMPONENTS = ("box", "obj", "noobj", "cls")

    def __init__(self, anchors, strides, num_classes=config.NUM_CLASSES,
                 img_size=config.IMG_SIZE,
                 lambda_box=config.LAMBDA_BOX, lambda_obj=config.LAMBDA_OBJ,
                 lambda_noobj=config.LAMBDA_NOOBJ, lambda_cls=config.LAMBDA_CLS,
                 ignore_thresh=config.IGNORE_THRESH,
                 anchor_match_thresh=config.ANCHOR_MATCH_THRESH,
                 focal_obj=config.FOCAL_OBJ, focal_gamma=config.FOCAL_GAMMA,
                 focal_alpha=config.FOCAL_ALPHA):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.strides = strides
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.ignore_thresh = ignore_thresh
        self.anchor_match_thresh = anchor_match_thresh
        # Objectness focal loss (down-weights easy examples; see config).
        self.focal_obj = focal_obj
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Anchors normalized to [0,1] (divide pixel sizes by img_size), shaped
        # [num_scales, num_anchors_per_scale, 2]. Registered as a buffer so it
        # moves to the right device with the module.
        anchors_t = torch.tensor(anchors, dtype=torch.float32) / img_size
        self.register_buffer("anchors_norm", anchors_t)          # [S, A, 2]
        self.register_buffer("all_anchors_norm", anchors_t.view(-1, 2))  # [S*A, 2]
        self.num_anchors_per_scale = anchors_t.shape[1]

    def _obj_loss_map(self, logits, targets):
        """Per-element objectness loss: sigmoid focal loss (if self.focal_obj)
        or plain BCE-with-logits. No reduction -- returns a [B, A, H, W] map.

        Focal = alpha_t * (1 - p_t)^gamma * BCE, where p_t is the predicted
        probability of the TRUE label and alpha_t = alpha for positives,
        (1 - alpha) for negatives. The (1 - p_t)^gamma factor down-weights easy
        (already-confident) cells so the gradient concentrates on hard ones;
        alpha balances positives vs negatives. gamma=0, alpha=0.5 -> plain BCE.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if not self.focal_obj:
            return bce
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)        # prob of true label
        focal = (1.0 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return alpha_t * focal * bce

    def forward(self, predictions, targets):
        """Compute the total loss and a per-component breakdown.

        Input:
            predictions: list of 3 raw model outputs, each
                [B, A, H_s, W_s, 5 + num_classes] (A = anchors per scale).
            targets: [M, 6] = [batch_idx, class, cx, cy, w, h] (normalized),
                pooled over the whole batch.

        Output:
            total: scalar tensor (differentiable).
            items: dict {"total","box","obj","noobj","cls"} of floats (the
                   weighted contributions, so the four parts sum to total).
        """
        device = predictions[0].device
        A = self.num_anchors_per_scale

        # ---- Match every GT to all anchors above the shape-IoU threshold ----
        M = targets.shape[0]
        if M > 0:
            gt_img = targets[:, 0].long()       # [M] which image in the batch
            gt_cls = targets[:, 1].long()       # [M] class id
            gt_xywh = targets[:, 2:6]           # [M, 4] normalized cx,cy,w,h
            # Shape IoU between each GT and all 9 anchors.
            ious = wh_iou(gt_xywh[:, 2:4], self.all_anchors_norm)  # [M, 9]
            # Multi-anchor matching: a GT is positive on EVERY anchor whose shape
            # IoU exceeds the threshold (not just the single best), which
            # densifies positives. The best anchor is forced on as a fallback so
            # every GT keeps >= 1 positive even if all its IoUs are below it.
            best = ious.argmax(dim=1)                          # [M] index 0..8
            anchor_match = ious > self.anchor_match_thresh     # [M, 9] bool
            anchor_match[torch.arange(M, device=device), best] = True

        # Accumulators (tensors so gradients flow through the sums).
        box_sum = torch.zeros((), device=device)
        cls_sum = torch.zeros((), device=device)
        obj_pos_sum = torch.zeros((), device=device)
        obj_neg_sum = torch.zeros((), device=device)
        n_pos = 0       # total positive anchors over all scales
        n_neg = 0       # total counted (non-ignored) negative anchors

        # ---- Loop over the 3 scales -----------------------------------------
        for s, pred in enumerate(predictions):
            B, _, H, W, _ = pred.shape
            anchors_s = self.anchors_norm[s]            # [A, 2] normalized (w,h)

            # Split the raw prediction into its parts.
            pred_xy = pred[..., 0:2]      # raw tx, ty            [B,A,H,W,2]
            pred_wh = pred[..., 2:4]      # raw tw, th            [B,A,H,W,2]
            pred_obj = pred[..., 4]       # raw objectness logit  [B,A,H,W]
            pred_cls = pred[..., 5:]      # raw class logits      [B,A,H,W,C]

            # --- Decode predictions to normalized xywh boxes ---
            # Grid cell indices: gx is the column (x), gy is the row (y).
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device), torch.arange(W, device=device),
                indexing="ij",
            )  # each [H, W]
            # bx,by: sigmoid(offset) + cell index, scaled to [0,1] by the grid.
            bx = (pred_xy[..., 0].sigmoid() + gx) / W
            by = (pred_xy[..., 1].sigmoid() + gy) / H
            # bw,bh: anchor size * exp(raw). anchors broadcast over B,H,W.
            bw = anchors_s[:, 0].view(1, A, 1, 1) * pred_wh[..., 0].exp()
            bh = anchors_s[:, 1].view(1, A, 1, 1) * pred_wh[..., 1].exp()
            pred_boxes = torch.stack([bx, by, bw, bh], dim=-1)  # [B,A,H,W,4]

            # --- Build dense target tensors for this scale ---
            obj_mask = torch.zeros((B, A, H, W), dtype=torch.bool, device=device)
            tbox = torch.zeros((B, A, H, W, 4), device=device)
            tcls = torch.zeros((B, A, H, W), dtype=torch.long, device=device)

            if M > 0:
                # (gt, anchor) pairs matched to THIS scale's block of anchors.
                match_s = anchor_match[:, s * A:(s + 1) * A]      # [M, A]
                gt_idx, a_idx = match_s.nonzero(as_tuple=True)    # each [K]
                for i, a in zip(gt_idx.tolist(), a_idx.tolist()):
                    b = int(gt_img[i])
                    cx, cy = gt_xywh[i, 0], gt_xywh[i, 1]
                    # Grid cell that contains the GT center (clamp to be safe).
                    gi = min(int(cx.item() * W), W - 1)
                    gj = min(int(cy.item() * H), H - 1)
                    obj_mask[b, a, gj, gi] = True
                    tbox[b, a, gj, gi] = gt_xywh[i]
                    tcls[b, a, gj, gi] = int(gt_cls[i])

            # --- Ignore mask: negatives that overlap a GT too much ---
            ignore_mask = torch.zeros((B, A, H, W), dtype=torch.bool, device=device)
            if M > 0:
                with torch.no_grad():
                    # Decoded boxes as corners for IoU (detached: no grad here).
                    pred_xyxy = xywh_to_xyxy(pred_boxes.detach())  # [B,A,H,W,4]
                    for b in range(B):
                        gt_b = gt_xywh[gt_img == b]                # [Gb, 4]
                        if gt_b.numel() == 0:
                            continue
                        pb = pred_xyxy[b].reshape(-1, 4)           # [A*H*W, 4]
                        iou = box_iou(pb, xywh_to_xyxy(gt_b))      # [A*H*W, Gb]
                        max_iou, _ = iou.max(dim=1)
                        ignore_mask[b] = (max_iou > self.ignore_thresh).view(A, H, W)
                # Positives are never ignored.
                ignore_mask &= ~obj_mask

            # --- Box loss + class loss (positives only) ---
            if obj_mask.any():
                pos_pred_box = pred_boxes[obj_mask]   # [P, 4]
                pos_tbox = tbox[obj_mask]             # [P, 4]
                ciou = bbox_ciou(pos_pred_box, pos_tbox)  # [P]
                box_sum = box_sum + (1.0 - ciou).sum()

                pos_cls_logits = pred_cls[obj_mask]                   # [P, C]
                pos_cls_target = F.one_hot(tcls[obj_mask], self.num_classes).float()
                cls_sum = cls_sum + F.binary_cross_entropy_with_logits(
                    pos_cls_logits, pos_cls_target, reduction="sum")

            n_pos += int(obj_mask.sum().item())

            # --- Objectness loss (positives + non-ignored negatives) ---
            # Sigmoid focal loss (down-weights easy cells) or plain BCE per config.
            obj_loss_map = self._obj_loss_map(pred_obj, obj_mask.float())  # [B,A,H,W]
            neg_mask = (~obj_mask) & (~ignore_mask)
            obj_pos_sum = obj_pos_sum + obj_loss_map[obj_mask].sum()
            obj_neg_sum = obj_neg_sum + obj_loss_map[neg_mask].sum()
            n_neg += int(neg_mask.sum().item())

        # ---- Normalize each term to a per-sample mean and weight it ---------
        n_pos_safe = max(n_pos, 1)
        box = box_sum / n_pos_safe                       # mean (1 - CIoU)
        cls = cls_sum / (n_pos_safe * self.num_classes)  # mean per-class BCE
        obj_pos = obj_pos_sum / n_pos_safe               # mean obj loss on positives
        # Focal loss normalizes by #positives (RetinaNet convention): the
        # (1-p_t)^gamma factor already silences easy negatives, so dividing by
        # n_neg would re-dilute the hard ones we just emphasized. Plain BCE keeps
        # the classic per-negative mean.
        obj_neg = obj_neg_sum / (n_pos_safe if self.focal_obj else max(n_neg, 1))

        total = (self.lambda_box * box
                 + self.lambda_obj * obj_pos
                 + self.lambda_noobj * obj_neg
                 + self.lambda_cls * cls)

        items = {
            "box": (self.lambda_box * box).item(),
            "obj": (self.lambda_obj * obj_pos).item(),
            "noobj": (self.lambda_noobj * obj_neg).item(),
            "cls": (self.lambda_cls * cls).item(),
            "total": total.item(),
        }
        return total, items


# ---- Quick self-test: run this file directly to sanity-check the loss -------
# python losses/yolo_loss.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, A, C = 2, config.NUM_ANCHORS_PER_SCALE, config.NUM_CLASSES
    # Fake raw predictions for a 416 input (grids 52/26/13), requires_grad.
    preds = [
        torch.randn(B, A, 52, 52, 5 + C, requires_grad=True),
        torch.randn(B, A, 26, 26, 5 + C, requires_grad=True),
        torch.randn(B, A, 13, 13, 5 + C, requires_grad=True),
    ]
    # Fake targets: [batch_idx, class, cx, cy, w, h] (normalized).
    targets = torch.tensor([
        [0, 11, 0.50, 0.50, 0.30, 0.40],
        [0, 14, 0.20, 0.30, 0.10, 0.25],
        [1, 2, 0.70, 0.60, 0.20, 0.20],
    ], dtype=torch.float32)

    criterion = YOLOLoss(anchors=config.ANCHORS, strides=config.STRIDES)
    loss, items = criterion(preds, targets)
    print("loss items:", items)
    loss.backward()
    print("grad on preds[0]:", preds[0].grad is not None)
