"""Embedding layer: token ids (+ case ids) -> vectors. This project's "backbone".

    ids   [B, L]  --lookup-->  word vectors [B, L, 100]
    cases [B, L]  --lookup-->  case vectors [B, L, 16]
                  --concat-->  [B, L, 116]

TWO TABLES, and the second one is the difference between this file and its
IMDB counterpart. The word table is the pretrained GloVe backbone, exactly as
before. The case table is 7 rows learned from scratch, and it exists because
GloVe 6B is uncased: the word path has to lowercase to find anything, which
deletes capitalization -- the strongest surface cue for a named entity. See
config.CASE_CLASSES and dataset/vocab.py.

The sizes are worth looking at side by side, because they explain why this is
cheap enough to be uncontroversial:

    word table   len(vocab) x 100    ~2.4M parameters
    case table   7 x 16              112 parameters

112 parameters, and they carry the feature the task is most sensitive to. The
`--no-case` ablation exists to measure exactly that.

The two-stage protocol transfers from the CNN projects unchanged:

    stage 1  freeze()   -> the WORD table is a fixed feature extractor,
                           the encoder learns to read it
    stage 2  unfreeze() -> the word table finetunes at a much smaller LR, so
                           task-specific nuance can be learned without
                           destroying the general geometry

Note that freeze()/unfreeze() deliberately act on the WORD table only. The
case table is randomly initialized, not pretrained, so freezing it in stage 1
would mean training an encoder to read 7 rows of noise -- the same mistake the
`--no-glove` warning in train.py guards against for the word table.

Two details that are easy to get wrong and cost accuracy silently:

  * padding_idx=0 pins the <pad> row of BOTH tables to the zero vector AND
    blocks gradient for it, so padding stays inert even after unfreezing.
    config.CASE_CLASSES reserves index 0 for exactly this.
  * dropout is applied AFTER the concatenation, to the joined vector, so a
    dropped unit can be a case unit. Dropping the case feature sometimes is
    the intent -- it stops the model leaning on capitalization alone, which
    matters on a corpus with all-caps datelines.
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402


class TokenEmbedding(nn.Module):
    """Word-vector table (+ optional case table) with freeze control.

    Args:
        vocab_size: number of word rows (len(vocab)).
        embed_dim: word-vector width (100 for glove.6B.100d).
        pad_idx: row pinned to zero and excluded from gradients, in BOTH tables.
        pretrained: [vocab_size, embed_dim] float tensor from
            dataset.glove.build_embedding_matrix, or None for random init.
        dropout: dropout probability on the concatenated output vectors.
        use_case: build the case table and concatenate it (config.USE_CASE_FEATURE).
        case_dim: case-vector width (config.CASE_DIM).

    Attributes:
        out_dim: what the encoder above receives -- embed_dim, or
            embed_dim + case_dim when use_case.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_idx: int = 0,
        pretrained: torch.Tensor = None,
        dropout: float = 0.5,
        use_case: bool = True,
        case_dim: int = None,
    ):
        super().__init__()
        case_dim = config.CASE_DIM if case_dim is None else case_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        self.use_case = use_case
        # Recorded so train.py can warn when stage 1 would freeze RANDOM
        # weights -- the same guard the HRNet project has for its backbone.
        self.pretrained_loaded = False

        self.case_embedding = (
            nn.Embedding(config.NUM_CASE_CLASSES, case_dim, padding_idx=pad_idx)
            if use_case else None)
        self.out_dim = embed_dim + (case_dim if use_case else 0)

        if pretrained is not None:
            if pretrained.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"pretrained matrix {tuple(pretrained.shape)} does not match "
                    f"({vocab_size}, {embed_dim}) -- vocab/dim mismatch"
                )
            # copy_ under no_grad: overwrite the values, keep the Parameter.
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained)
                self.embedding.weight[pad_idx].zero_()
            self.pretrained_loaded = True

    def forward(self, ids: torch.Tensor, cases: torch.Tensor = None) -> torch.Tensor:
        """Input: ids [B, L], cases [B, L] (or None). Output: [B, L, out_dim]."""
        vectors = self.embedding(ids)
        if self.case_embedding is not None:
            if cases is None:
                raise ValueError("case ids are required when use_case=True")
            vectors = torch.cat([vectors, self.case_embedding(cases)], dim=-1)
        return self.dropout(vectors)

    # ---- two-stage finetuning helpers ----
    def freeze(self):
        """Stage 1: stop training the PRETRAINED word vectors.

        The case table keeps training: it is random at init, and freezing
        random weights is the mistake this whole protocol exists to avoid.
        """
        self.embedding.weight.requires_grad_(False)

    def unfreeze(self):
        """Stage 2: let the word vectors finetune (at a small LR)."""
        self.embedding.weight.requires_grad_(True)


# ---- Quick self-test: run this file directly --------------------------------
# python model/embedding.py
if __name__ == "__main__":
    torch.manual_seed(0)
    V, D = 50, 8
    pre = torch.randn(V, D)
    emb = TokenEmbedding(V, D, pad_idx=0, pretrained=pre, dropout=0.0,
                         use_case=True, case_dim=4)

    ids = torch.tensor([[5, 7, 0, 0], [3, 4, 9, 0]])      # 0 = <pad>
    cases = torch.tensor([[1, 2, 0, 0], [3, 1, 1, 0]])    # 0 = <pad>
    out = emb(ids, cases)
    print("out:", tuple(out.shape), "(expected (2, 4, 12) = 8 word + 4 case)")
    print("out_dim attribute:", emb.out_dim, "(expected 12)")
    print("pad position is all-zero:", bool(out[0, 2].abs().sum() == 0),
          "(both tables pinned at index 0)")
    print("word half of row 5 matches pretrained:",
          bool(torch.allclose(out[0, 0, :D], pre[5])))

    # The case feature must actually change the output. Same word ids, two
    # different case ids -> two different vectors. This is the whole point.
    a = emb(torch.tensor([[5]]), torch.tensor([[1]]))
    b = emb(torch.tensor([[5]]), torch.tensor([[2]]))
    print("same word, different case -> different vector:",
          not bool(torch.allclose(a, b)), "(expected True)")

    # Freeze / unfreeze must flip requires_grad on the WORD table only.
    emb.freeze()
    print("\nfrozen word table requires_grad:", emb.embedding.weight.requires_grad,
          "(expected False)")
    print("case table still requires_grad:", emb.case_embedding.weight.requires_grad,
          "(expected True -- it is not pretrained)")
    emb.unfreeze()
    emb(ids, cases).sum().backward()
    print("pad row grad after backward -- word:",
          float(emb.embedding.weight.grad[0].abs().sum()),
          " case:", float(emb.case_embedding.weight.grad[0].abs().sum()),
          "(both expected 0.0 -- padding_idx blocks them)")
    print("word row 5 grad:", float(emb.embedding.weight.grad[5].abs().sum()),
          "(expected > 0)")

    # And the ablation path: no case table at all.
    plain = TokenEmbedding(V, D, pretrained=pre, dropout=0.0, use_case=False)
    print("\n--no-case out_dim:", plain.out_dim, "(expected 8)")
    print("--no-case forward works without case ids:",
          tuple(plain(ids).shape), "(expected (2, 4, 8))")
