"""Classification head: token features -> one document vector -> 2 logits.

    outputs [B, L, F] (+ final [B, F])  --pool-->  [B, F]  --linear-->  [B, 4]

The segmentation heads had to UPSAMPLE back to per-pixel predictions; a
document classifier does the opposite -- it must COLLAPSE a variable number of
token features into one fixed vector. Three ways to do it, selectable via
config.POOLING:

    "last"  the encoder's document vector: forward end + backward end for
            RNNs, or the last real-token feature for the Transformer.
    "max"   element-wise max over time. Each feature dimension reports its
            strongest activation anywhere in the review.
    "mean"  masked average over time -- smoother, but a 400-token review can
            dilute the one clause that mattered.

The right choice is LESS obvious here than on AG News, and for an interesting
reason. Topic classification rewards max pooling: one decisive keyword
("midfielder") settles the label wherever it appears. Sentiment does not work
that way. A negative review is full of strongly positive words -- describing
what the film attempted, quoting the trailer, praising one actor before
condemning everything else -- so "the strongest positive activation anywhere"
is a feature that fires on both classes. What decides the label is the
COMPOSITION, which is the encoder's job, not the pooling's. Note that this is
an argument about the ENCODER's features, not raw words: a good encoder can
make position 200 mean "praise, negated", at which point max pooling is fine
again.

MASKING is the part that must be right, and it bites harder here than on
either sibling: IMDB review lengths span two orders of magnitude, so a batch
holding a 30-token review and a 400-token one is 92% padding on that row.
Padded positions are not data: for "max" they are set to -inf before the
reduction (a plain max would happily return a padding activation whenever it
is the largest), and for "mean" the sum is divided by the TRUE length, not by
L. Get this wrong and accuracy quietly drops with batch composition -- the
same review scores differently depending on which others shared its batch.

WHY 2 LOGITS AND NOT 1 SIGMOID: with two mutually exclusive classes, softmax
over 2 logits is over-parameterized by exactly one direction (adding a
constant to both changes nothing), so a single sigmoid output would suffice
and is mathematically equivalent. Two logits are used anyway, because the
symmetric form keeps both classes' weight vectors interpretable, lets
nn.CrossEntropyLoss take the argmax directly, and makes this file identical to
the 4-class AG-News version -- which is the point of a repo that compares
tasks.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Masked pooling + dropout + linear classifier.

    Args:
        in_features: encoder output width (RNN hidden_size * directions, or dim).
        num_classes: 2 for IMDB.
        pooling: "last" / "max" / "mean".
        dropout: applied to the pooled document vector.
    """

    def __init__(self, in_features: int, num_classes: int = 2,
                 pooling: str = "last", dropout: float = 0.5):
        super().__init__()
        if pooling not in ("last", "max", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        # A single linear layer on purpose: with a 512-dim BiLSTM feature, an
        # extra hidden layer adds parameters and overfitting, not accuracy.
        # The capacity belongs in the encoder.
        self.fc = nn.Linear(in_features, num_classes)

    @staticmethod
    def _mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Build a [B, L] bool mask that is True at REAL token positions.

        arange(L) < length, broadcast over the batch -- the standard trick.
        """
        ar = torch.arange(max_len, device=lengths.device)        # [L]
        return ar[None, :] < lengths[:, None]                    # [B, L]

    def pool(self, outputs: torch.Tensor, final: torch.Tensor,
             lengths: torch.Tensor) -> torch.Tensor:
        """Collapse [B, L, F] token features into [B, F]. See module docstring."""
        if self.pooling == "last":
            # The encoder already selected the appropriate non-PAD features.
            return final

        mask = self._mask_from_lengths(lengths, outputs.size(1))[..., None]  # [B, L, 1]
        if self.pooling == "max":
            # -inf on padding so it can never win the max. (Padding rows come
            # back as zeros from the encoder, and 0 > a negative activation --
            # this is exactly the silent bug the masking prevents.)
            masked = outputs.masked_fill(~mask, float("-inf"))
            return masked.max(dim=1).values
        # mean: sum the real positions, divide by the true length.
        summed = (outputs * mask).sum(dim=1)                     # [B, F]
        return summed / lengths.clamp(min=1)[:, None].to(summed.dtype)

    def forward(self, outputs: torch.Tensor, final: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Input: encoder outputs/final + lengths. Output: logits [B, C]."""
        pooled = self.pool(outputs, final, lengths)
        return self.fc(self.dropout(pooled))


# ---- Quick self-test: run this file directly --------------------------------
# python model/head.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, F = 2, 4, 3
    # Row 1 is 2 tokens long; its padded tail holds LARGE values that masked
    # pooling must ignore (an unmasked max/mean would be fooled by them).
    outputs = torch.tensor([
        [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [0., 0., 0.]],
        [[1., 1., 1.], [3., 3., 3.], [99., 99., 99.], [99., 99., 99.]],
    ])
    final = torch.zeros(B, F)
    lengths = torch.tensor([3, 2])

    for pooling in ("max", "mean"):
        head = ClassifierHead(F, num_classes=2, pooling=pooling, dropout=0.0)
        pooled = head.pool(outputs, final, lengths)
        print(f"[{pooling}] pooled row0={pooled[0].tolist()} row1={pooled[1].tolist()}")
    print("expected max:  row0=[1,2,3]  row1=[3,3,3]  (99s masked out)")
    print("expected mean: row0=[.33,.67,1]  row1=[2,2,2]")

    head = ClassifierHead(F, num_classes=2, pooling="last", dropout=0.0)
    logits = head(outputs, final, lengths)
    print("logits:", tuple(logits.shape), "(expected (2, 2))")
