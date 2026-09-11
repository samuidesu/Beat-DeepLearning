"""Embedding -> Transformer encoder -> per-token tagger. The second model.

Forward pass:
    ids [B, L] + cases [B, L] + lengths [B]
      --embedding-->  vectors [B, L, 116]     (GloVe 100 + case 16)
      --projection->  [B, L, 128] + sinusoidal positions
      --encoder---->  outputs [B, L, 128]     (nn.TransformerEncoderLayer)
      --head------->  logits  [B, L, 9]

Identical in interface to LSTMTagger -- same constructor keywords where they
overlap, same forward signature, same parameter_groups() -- so train.py builds
either from one code path and the comparison stays single-variable: the
ENCODER changes, nothing else does.

WHAT THIS COMPARISON IS ACTUALLY TESTING, on this corpus. The two encoders
differ in how a token reaches its context:

    BiLSTM       information travels step by step, so distance costs
                 something, but ADJACENCY IS BUILT IN -- "the previous token"
                 is what the recurrence hands you for free.
    Transformer  every token attends to every other in one hop, so distance
                 is free, but adjacency has to be RECONSTRUCTED from the
                 positional encoding.

BIO2 tagging is mostly a local, adjacency-shaped problem: deciding I-PER
versus B-PER is a statement about the immediately preceding token. That is the
one thing the recurrence gets for free and attention has to learn, which is
why the classic result on this corpus is a BiLSTM and why a small Transformer
trained from scratch on 15k sentences is not obviously going to beat it. The
long-range advantage that made Transformers win everywhere else has little to
bite on in a 14-token sentence.

Note also what is NOT here: pooling. The classification version of this file
had a `pooling` argument threaded through to ClassifierHead. There is nothing
to pool in a tagger, so the argument is gone rather than defaulted.
"""

import torch.nn as nn

from model.embedding import TokenEmbedding
from model.head import TaggerHead
from .encoder import TransformerEncoder


class TransformerTagger(nn.Module):
    """Transformer sequence labeler for CoNLL-2003 NER.

    Args:
        vocab_size: len(vocab).
        num_tags: 9 (BIO2 over 4 entity types).
        embed_dim: word-vector width (100 for glove.6B.100d).
        dim: model width after the projection; must be divisible by `group`.
        group: number of attention heads.
        num_layers: stacked encoder blocks.
        dropout: embedding, attention, residual and head dropout.
        pad_idx: padding id (0), in both embedding tables.
        max_len: positional-encoding limit (config.MAX_LEN).
        norm: "pre" or "post" LayerNorm placement (see encoder.py).
        use_case / case_dim: the capitalization feature.
        pretrained_vectors: [vocab_size, embed_dim] GloVe matrix, or None.
    """

    def __init__(self, vocab_size, num_tags=9, embed_dim=100, dim=128,
                 group=4, num_layers=2, dropout=0.1, pad_idx=0, max_len=128,
                 norm="pre", use_case=True, case_dim=None,
                 pretrained_vectors=None):
        super().__init__()
        self.use_case = use_case
        self.max_len = max_len
        # Input dropout is applied once, after projection + positional
        # encoding, so the embedding table's own dropout is disabled here.
        self.embedding = TokenEmbedding(
            vocab_size, embed_dim, pad_idx=pad_idx,
            pretrained=pretrained_vectors, dropout=0.0,
            use_case=use_case, case_dim=case_dim,
        )
        self.encoder = TransformerEncoder(
            input_dim=self.embedding.out_dim, dim=dim, group=group,
            num_layers=num_layers, dropout=dropout, max_len=max_len, norm=norm,
        )
        self.head = TaggerHead(
            in_features=self.encoder.out_dim, num_tags=num_tags, dropout=dropout,
        )

    def forward(self, ids, cases, lengths):
        """Input: ids [B, L], cases [B, L], lengths [B]. Output: [B, L, num_tags]."""
        lengths = lengths.to(ids.device)
        vectors = self.embedding(ids, cases if self.use_case else None)
        outputs = self.encoder(vectors, lengths)
        return self.head(outputs)

    def freeze_embedding(self):
        """Stage 1: freeze the pretrained word vectors (not the case table)."""
        self.embedding.freeze()

    def unfreeze_all(self):
        """Stage 2: unfreeze everything for the layered-LR finetune."""
        for p in self.parameters():
            p.requires_grad_(True)

    def parameter_groups(self):
        """Keep the same three learning-rate groups as LSTMTagger."""
        return {
            name: [p for p in module.parameters() if p.requires_grad]
            for name, module in (("embedding", self.embedding),
                                 ("encoder", self.encoder),
                                 ("head", self.head))
        }


# ---- Quick self-test: run this file directly --------------------------------
# python -m model_transformer.transformer_tagger
if __name__ == "__main__":
    import os
    import sys

    import torch

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    torch.manual_seed(0)
    V, B, L = 1000, 4, 12
    ids = torch.randint(2, V, (B, L))
    cases = torch.randint(1, config.NUM_CASE_CLASSES, (B, L))
    lengths = torch.tensor([12, 9, 5, 1])
    for i, n in enumerate(lengths):
        ids[i, n:] = 0
        cases[i, n:] = 0

    for norm in ("pre", "post"):
        model = TransformerTagger(V, num_tags=config.NUM_TAGS, dim=64, group=4,
                                  num_layers=2, max_len=config.MAX_LEN, norm=norm)
        model.eval()
        with torch.no_grad():
            logits = model(ids, cases, lengths)
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[{norm}-LN] logits {tuple(logits.shape)} (expected (4, 12, 9))  "
              f"params={n_par:.2f}M")

    print("\npredicted tag ids, row 0:", logits[0].argmax(-1).tolist(),
          "(one per token)")
    with torch.no_grad():
        alt = model(ids, torch.full_like(cases, config.CASE2ID["lower"]), lengths)
    print("changing only the CASE ids changes the logits:",
          not bool(torch.allclose(logits, alt)), "(expected True)")

    model.freeze_embedding()
    for name, params in model.parameter_groups().items():
        print(f"stage-1 group {name:<9}: {sum(p.numel() for p in params):>9,} trainable")
