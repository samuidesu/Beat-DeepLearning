"""Full model: GloVe + case embedding -> BiLSTM -> per-token linear tagger.

Forward pass:
    ids [B, L] + cases [B, L] + lengths [B]
      --embedding-->  vectors [B, L, 116]         (GloVe 100 + case 16)
      --encoder---->  outputs [B, L, 512]         (BiLSTM 2 x 256, both dirs)
      --head------->  logits  [B, L, 9]           (dropout + linear per step)

The three-part split is the same one the segmentation projects used (backbone
/ neck / head), because the two-stage finetuning protocol is defined on it:

    Stage 1: freeze the pretrained WORD table; train the case table, encoder
             and head, which all start from random weights.
    Stage 2: unfreeze everything, three LR tiers -- embedding slowest,
             encoder middle, head fastest.

Compared with the classification projects' RNNClassifier, exactly three things
changed, and all three are consequences of one label per TOKEN:

  1. forward() takes a `cases` stream alongside `ids`.
  2. The encoder returns only per-token features; there is no document vector.
  3. The output is [B, L, 9] instead of [B, C], so the loss flattens batch and
     time together and masks padding with ignore_index.

parameter_groups() returns the same three tiers train.py has used since the
DeepLab/HRNet projects, so build_layered_optimizer is unchanged.

NOTE ON THE CASE TABLE'S TIER. It is grouped with the "embedding" tier for
naming reasons, but it is NOT pretrained -- and stage 1 freezes only the word
table (see embedding.py). The consequence to be aware of: in stage 2 the case
table finetunes at STAGE2_LR_EMBEDDING (5e-5), which is the gentle rate meant
for vectors that already know English. For 112 randomly-initialized parameters
that is conservative. It is left that way because those parameters get ten
full epochs at the head/encoder rate during stage 1, which is where they
actually learn.
"""

from __future__ import annotations

from typing import Dict, Iterator, List

import torch
import torch.nn as nn

try:
    from .embedding import TokenEmbedding
    from .encoder import LSTMEncoder
    from .head import TaggerHead
except ImportError:  # running this file directly
    from embedding import TokenEmbedding
    from encoder import LSTMEncoder
    from head import TaggerHead


class LSTMTagger(nn.Module):
    """BiLSTM sequence labeler for CoNLL-2003 NER.

    Args:
        vocab_size: len(vocab).
        num_tags: 9 (BIO2 over 4 entity types).
        embed_dim: word-vector width (100 for glove.6B.100d).
        hidden_size: LSTM hidden width per direction.
        num_layers / bidirectional: encoder shape.
        dropout: shared by embedding, between encoder layers, and before fc.
        pad_idx: padding id (0), in both embedding tables.
        use_case: concatenate the capitalization embedding (config default).
        case_dim: case-vector width.
        pretrained_vectors: [vocab_size, embed_dim] GloVe matrix, or None to
            train the word vectors from scratch.
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int = 9,
        embed_dim: int = 100,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pad_idx: int = 0,
        use_case: bool = True,
        case_dim: int = None,
        pretrained_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.use_case = use_case

        self.embedding = TokenEmbedding(
            vocab_size, embed_dim, pad_idx=pad_idx,
            pretrained=pretrained_vectors, dropout=dropout,
            use_case=use_case, case_dim=case_dim)
        self.encoder = LSTMEncoder(
            input_size=self.embedding.out_dim, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.head = TaggerHead(
            in_features=self.encoder.out_dim, num_tags=num_tags, dropout=dropout)

    def forward(self, ids: torch.Tensor, cases: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Input: ids [B, L], cases [B, L], lengths [B]. Output: [B, L, num_tags]."""
        vectors = self.embedding(ids, cases if self.use_case else None)
        outputs = self.encoder(vectors, lengths)
        return self.head(outputs)

    # ---- Two-stage finetuning helpers (same interface as the CNN models) ----
    def freeze_embedding(self):
        """Stage 1: freeze the pretrained word vectors (not the case table)."""
        self.embedding.freeze()

    def unfreeze_all(self):
        """Stage 2: unfreeze everything for the layered-LR finetune."""
        self.embedding.unfreeze()

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Split parameters into the three layered-LR groups.

        Output (dict of lists, ready for torch.optim param_groups):
            "embedding" -> word table + case table   (slowest LR)
            "encoder"   -> the BiLSTM                (middle LR)
            "head"      -> the per-token classifier  (fastest LR)

        Frozen parameters are EXCLUDED, so the same call serves both stages:
        in stage 1 "embedding" contains only the case table, and train.py
        gives it the encoder/head tier by way of the stage-1 LR settings.
        """
        return {
            "embedding": [p for p in self.embedding.parameters() if p.requires_grad],
            "encoder": [p for p in self.encoder.parameters() if p.requires_grad],
            "head": [p for p in self.head.parameters() if p.requires_grad],
        }

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the parameters that currently require gradients."""
        return (p for p in self.parameters() if p.requires_grad)


# ---- Quick self-test: run this file directly --------------------------------
# python model/lstm_tagger.py
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    torch.manual_seed(0)
    V, B, L = 1000, 4, 12
    ids = torch.randint(2, V, (B, L))
    cases = torch.randint(1, config.NUM_CASE_CLASSES, (B, L))
    lengths = torch.tensor([12, 9, 5, 1])
    for i, n in enumerate(lengths):          # zero out the padded tails
        ids[i, n:] = 0
        cases[i, n:] = 0

    model = LSTMTagger(V, num_tags=config.NUM_TAGS, hidden_size=64, num_layers=2)
    model.eval()
    with torch.no_grad():
        logits = model(ids, cases, lengths)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"logits {tuple(logits.shape)} (expected (4, 12, 9))  params={n_par:.2f}M")

    # One prediction per token, not per sentence -- the whole difference from
    # the classification projects, visible in one line.
    print("predicted tag ids, row 0:",
          logits[0].argmax(-1).tolist(), f"({L} of them, one per token)")

    # The case stream must reach the output.
    with torch.no_grad():
        alt = model(ids, torch.full_like(cases, config.CASE2ID["lower"]), lengths)
    print("changing only the CASE ids changes the logits:",
          not bool(torch.allclose(logits, alt)), "(expected True)")

    # Stage-1 freeze: the word table leaves the embedding tier, the case
    # table stays (it is random, not pretrained).
    print()
    model.freeze_embedding()
    for name, params in model.parameter_groups().items():
        n = sum(p.numel() for p in params)
        print(f"stage-1 group {name:<9}: {n:>9,} trainable")
    print("  (embedding tier = the 7x16 case table only)")
    model.unfreeze_all()
    for name, params in model.parameter_groups().items():
        n = sum(p.numel() for p in params)
        print(f"stage-2 group {name:<9}: {n:>9,} trainable")

    # And the ablation path: no case stream at all.
    plain = LSTMTagger(V, num_tags=config.NUM_TAGS, hidden_size=64,
                       num_layers=2, use_case=False)
    plain.eval()
    with torch.no_grad():
        print("\n--no-case logits:", tuple(plain(ids, cases, lengths).shape),
              "(expected (4, 12, 9); case ids accepted and ignored)")
