"""Evaluation metrics: accuracy, per-class precision/recall/F1, macro-F1.

IMDB's standard metric is plain ACCURACY, and here that is unusually safe to
trust: the corpus is EXACTLY balanced (50/50 in train, val and test alike),
so accuracy cannot be inflated by a majority-class shortcut and the chance
baseline is a clean 0.50.

Everything else in this file comes from the same confusion matrix the
segmentation projects built for mIoU -- one K x K table, M[g, p] = number of
samples whose true class is g and predicted class is p. Per class c:

    TP = M[c, c]                      FP = column_sum(c) - TP
    FN = row_sum(c) - TP

    precision = TP / (TP + FP)        "when it says c, how often is it right"
    recall    = TP / (TP + FN)        "of the real c's, how many did it find"
    F1        = harmonic mean of the two
    accuracy  = trace(M) / sum(M)
    macro-F1  = unweighted mean of the per-class F1s

With two balanced classes, accuracy and macro-F1 track each other very
closely, and the 2x2 matrix has only one degree of freedom beyond accuracy:
whether the model errs towards calling things positive or negative. That is
still worth printing. A model whose negative recall is five points below its
positive recall has learned an asymmetric decision boundary -- usually because
positive reviews are more formulaic while negative ones range from sarcasm to
plot-summary-with-a-sigh -- and no single accuracy number would say so.
"""

import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402


class ConfusionMatrix:
    """Streaming K x K sample confusion matrix.

    Usage: create once, .update() per batch, .compute() at the end.
    Kept on CPU: it only ever receives two small id tensors per batch, and CPU
    bincount sidesteps device quirks.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # mat[g, p] = number of samples with true class g, predicted class p.
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self):
        """Zero the matrix (start a fresh evaluation)."""
        self.mat.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Accumulate one batch.

        Input:
            pred: predicted class ids [B] (i.e. logits.argmax(dim=1), NOT raw
                logits).
            gt:   true class ids [B]; values outside [0, K) are dropped.
        """
        pred = pred.flatten().cpu()
        gt = gt.flatten().cpu()
        keep = (gt >= 0) & (gt < self.num_classes)
        # Encode each (gt, pred) pair as one integer gt*K + pred, histogram
        # them all at once with bincount, then reshape back to K x K.
        idx = gt[keep] * self.num_classes + pred[keep]
        counts = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Reduce the matrix to metrics.

        Output dict:
            accuracy:  float, fraction of correctly classified samples.
            macro_f1:  float, unweighted mean of the per-class F1 scores.
            per_class: list of {"precision", "recall", "f1", "support"} dicts,
                       indexed by class id.
            matrix:    the K x K counts as a nested list (for the log / plot).
        """
        mat = self.mat.double()
        tp = mat.diag()
        support = mat.sum(dim=1)              # true samples per class
        predicted = mat.sum(dim=0)            # predicted samples per class

        # clamp(min=1) only guards the DIVISION; classes with no samples get a
        # 0/1 = 0 score rather than a NaN, and their support says why.
        precision = tp / predicted.clamp(min=1)
        recall = tp / support.clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

        total = mat.sum().clamp(min=1)
        return {
            "accuracy": float(tp.sum() / total),
            "macro_f1": float(f1.mean()),
            "per_class": [
                {"precision": float(precision[c]), "recall": float(recall[c]),
                 "f1": float(f1[c]), "support": int(support[c])}
                for c in range(self.num_classes)
            ],
            "matrix": self.mat.tolist(),
        }


def print_report(result: dict, class_names=None):
    """Print the per-class table and the confusion matrix.

    The classification counterpart of the segmentation projects'
    print_per_class -- same "look at where it fails, not just the headline"
    idea, with precision/recall instead of IoU.
    """
    names = class_names or config.CLASS_NAMES
    print(f"\n{'class':<12} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}")
    print("-" * 45)
    for name, row in zip(names, result["per_class"]):
        print(f"{name:<12} {row['precision']:7.4f} {row['recall']:7.4f} "
              f"{row['f1']:7.4f} {row['support']:8d}")
    print("-" * 45)
    print(f"{'accuracy':<12} {result['accuracy']:7.4f}")
    print(f"{'macro_f1':<12} {result['macro_f1']:7.4f}")

    # Confusion matrix, rows = truth, cols = prediction.
    print("\nconfusion matrix (rows = true, cols = predicted)")
    header = " " * 12 + "".join(f"{n[:9]:>10}" for n in names)
    print(header)
    for name, row in zip(names, result["matrix"]):
        print(f"{name:<12}" + "".join(f"{v:>10d}" for v in row))


@torch.no_grad()
def compute_accuracy(model, loader, device, num_classes: int = None,
                     max_batches=None, verbose: bool = True) -> dict:
    """Run the model over `loader` and compute accuracy (& friends).

    Input:
        model:  classifier returning logits [B, C] from (ids, lengths).
        loader: yields (ids, lengths, labels) from collate_batch.
        device: torch device.
        num_classes: defaults to config.NUM_CLASSES (4).
        max_batches: if set, stop after this many batches (quick check).
        verbose: print the per-class report.

    Output:
        the ConfusionMatrix.compute() dict.
    """
    num_classes = num_classes or config.NUM_CLASSES
    model.eval()
    cm = ConfusionMatrix(num_classes)

    for i, (ids, lengths, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        logits = model(ids, lengths)
        cm.update(logits.argmax(dim=1), labels)

    result = cm.compute()
    if verbose:
        print_report(result)
    return result


# ---- Quick self-test: run this file directly (hand-checkable numbers) --------
# python utils/metrics.py
if __name__ == "__main__":
    # 8 samples, 2 classes, 4 of each, with a deliberately ASYMMETRIC error
    # pattern: two negatives are called positive, no positive is called
    # negative. That is the failure mode this file exists to make visible.
    gt = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    pred = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])

    cm = ConfusionMatrix(num_classes=2)
    cm.update(pred, gt)
    print_report(cm.compute(), class_names=["negative", "positive"])

    # By hand: 6 of 8 correct -> accuracy 0.75
    # negative: TP=2 FP=0 FN=2 -> P=1.0   R=0.5  F1=0.6667
    # positive: TP=4 FP=2 FN=0 -> P=0.667 R=1.0  F1=0.8000
    # macro F1 = (0.6667 + 0.8000) / 2 = 0.7333
    print("\nexpected accuracy 0.7500, macro_f1 0.7333")
    print("expected negative P/R = 1.0000 / 0.5000, positive = 0.6667 / 1.0000")
