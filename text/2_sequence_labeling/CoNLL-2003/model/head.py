"""Tagging head: per-token features -> one 9-way tag distribution PER TOKEN.

    outputs [B, L, F]  --dropout-->  --linear-->  logits [B, L, 9]

THE SHORTEST FILE IN THE PROJECT, AND THAT IS THE POINT. Its IMDB counterpart
was 130 lines, almost all of them about POOLING: three strategies, a mask
built from lengths, -inf fills so padding could not win a max, a division by
true length rather than L, and a long argument about which strategy suits
sentiment. Every line of that existed to collapse [B, L, F] into [B, F].

A tagger does not collapse anything. Token i's features predict token i's tag,
so the head is `nn.Linear` applied at every position and there is nothing to
choose. This is the sequence-labeling counterpart of the segmentation
projects' per-pixel classifier, and the same observation holds there: once the
output is dense, the head gets simpler, not harder.

WHY THERE IS NO MASKING HERE. Padded positions do produce logits -- the linear
layer has a bias, so even the zero features the encoder returns for padding
map to something nonzero. That is deliberately left alone, because every
consumer downstream already ignores those positions:

    loss     labels are config.IGNORE_INDEX at padding, and
             nn.CrossEntropyLoss(ignore_index=...) drops them
    metrics  utils/metrics.py slices each row to its true length
    decoding predictions are read back per sentence, also by length

Masking here as well would be a fourth place that has to agree with the other
three about what "real" means -- which is how the four drift apart. The rule
this project follows is that padding is handled where lengths are already
known, and the head is not such a place.

WHY ONE LINEAR LAYER. With a 512-dim BiLSTM feature and 14,987 training
sentences, an extra hidden layer adds parameters and overfitting, not
accuracy. The capacity belongs in the encoder.

WHAT IS DELIBERATELY MISSING: a CRF. A softmax head scores each token
INDEPENDENTLY, so nothing stops it emitting "O, I-PER" -- a continuation with
no beginning, which is not a valid BIO2 sequence at all. A linear-chain CRF
adds a learned transition matrix and Viterbi decoding, and makes such
sequences structurally impossible. It is the single best-known upgrade to a
neural tagger on this corpus -- Lample et al. (2016) is the standard reference
-- and the published gap between word-level taggers with and without one runs
to several F1 points. It is not implemented here because this project's
variable is the ENCODER, and a CRF would change both models at once.
Instead, utils/metrics.py COUNTS the invalid transitions the softmax head
emits, so the size of the gap the CRF would close is measured rather than
assumed.
"""

import torch
import torch.nn as nn


class TaggerHead(nn.Module):
    """Dropout + linear classifier applied independently at every position.

    Args:
        in_features: encoder output width (hidden_size * directions, or dim).
        num_tags: 9 for CoNLL-2003 in BIO2.
        dropout: applied to the token features before the linear layer.
    """

    def __init__(self, in_features: int, num_tags: int = 9, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_tags)

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        """Input: encoder outputs [B, L, F]. Output: logits [B, L, num_tags].

        nn.Linear applies to the LAST dimension and broadcasts over the rest,
        so no reshape is needed -- [B, L, F] goes in and [B, L, K] comes out,
        with the same weights used at every position. That weight sharing is
        what makes this a sequence labeler rather than L separate classifiers.
        """
        return self.fc(self.dropout(outputs))


# ---- Quick self-test: run this file directly --------------------------------
# python model/head.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, F, K = 2, 4, 6, 9
    head = TaggerHead(F, num_tags=K, dropout=0.0).eval()

    outputs = torch.randn(B, L, F)
    with torch.no_grad():
        logits = head(outputs)
    print("logits:", tuple(logits.shape), f"(expected ({B}, {L}, {K}))")

    # Position independence + weight sharing: feeding one position's features
    # alone must give exactly that position's logits.
    with torch.no_grad():
        alone = head(outputs[0, 2][None, None, :])
    print("position 2 scored alone matches its slice:",
          bool(torch.allclose(alone[0, 0], logits[0, 2], atol=1e-6)),
          "(expected True -- same weights at every step, no cross-talk)")

    # The loss contract this head is built against: padded label slots carry
    # IGNORE_INDEX and must contribute exactly nothing.
    labels = torch.tensor([[3, 4, 0, -100], [1, -100, -100, -100]])
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    # CrossEntropyLoss wants [N, K] vs [N]; flatten batch and time together.
    loss = crit(logits.reshape(-1, K), labels.reshape(-1))
    print(f"\nloss over {int((labels != -100).sum())} real positions: {loss:.4f}")

    # Change ONLY the logits under ignored positions -- the loss must not move.
    perturbed = logits.clone()
    perturbed[0, 3] += 100.0
    perturbed[1, 1:] -= 100.0
    loss2 = crit(perturbed.reshape(-1, K), labels.reshape(-1))
    print(f"loss after wrecking the IGNORED positions: {loss2:.4f}")
    print("ignore_index really excludes them:",
          bool(torch.allclose(loss, loss2)), "(expected True)")
