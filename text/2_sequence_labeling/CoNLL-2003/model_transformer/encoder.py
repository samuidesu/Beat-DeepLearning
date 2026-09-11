"""Projection, positional encoding and a stack of Transformer encoder layers.

The blocks are torch's own nn.TransformerEncoderLayer. The IMDB and AG-News
projects hand-wrote theirs (model_transformer/transformer_naive.py there);
this project does not, because the block itself is no longer the thing being
studied -- the variable here is the ENCODER FAMILY, LSTM against attention,
and a hand-written block would only add a second, unrelated way for that
comparison to be wrong. The two are the same function: transferring the
hand-written weights into nn.TransformerEncoderLayer(norm_first=True)
reproduced its output to 2.4e-07, at identical parameter count.

Three arguments below are load-bearing and easy to get wrong:

    norm_first=(norm == "pre")   NOT a hardcoded True -- otherwise --norm post
                                 silently keeps training a Pre-LN stack while
                                 final_norm below switches to Identity, which
                                 is a model that matches neither arrangement.
    activation="gelu"            torch defaults to ReLU.
    dim_feedforward=4 * dim      torch defaults to 2048 regardless of dim.

And in forward(), the padding mask MUST be passed by keyword. See the note
there.

Two further differences from the classification projects, both consequences of
the task rather than of the architecture:

  1. FORWARD RETURNS ONE THING, NOT TWO. The classification version returned
     (outputs, final), where `final` was the last real token's feature -- the
     document vector the head pooled. A tagger has no document vector, so
     `final` is gone. Everything the model produces is per-token.

  2. POSITIONAL ENCODING IS DOING REAL WORK NOW, and it is worth being
     explicit about why. Attention is permutation-invariant: without positions
     a Transformer sees a BAG of tokens. A sentiment classifier can survive
     that surprisingly well (the presence of "dreadful" carries the label
     wherever it sits). A tagger cannot survive it at all -- "B-PER" vs
     "I-PER" is a statement about ORDER, so the positional signal is not an
     enhancement here, it is load-bearing for the tag scheme itself.

Compute is a non-issue for the first time in this repo. Attention is O(L^2)
and CoNLL sentences average 14 tokens against IMDB's 400, so the attention
matrix is ~800x smaller. config.MAX_LEN = 128 only has to cover the corpus's
longest sentence (~113 tokens).
"""

import math

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(
        self, input_dim, dim=128, group=4, num_layers=2, dropout=0.1, max_len=128, norm="pre"
    ):
        super().__init__()
        if dim <= 0 or num_layers <= 0 or max_len <= 0:
            raise ValueError("dim, num_layers and max_len must be positive")

        self.out_dim = dim
        self.norm = norm
        self.projection = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim,
                    nhead=group,
                    dim_feedforward=4 * dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=(norm == "pre"),
                )
                for _ in range(num_layers)
            ]
        )
        # THIS IS NOT REDUNDANT WITH norm_first=True, which is the natural
        # thing to assume. A norm_first block ends on `x = x + ff(norm2(x))`:
        # the residual it returns was never normalized, so its magnitude grows
        # with depth. Measured on this exact stack at dim=128 / 4 layers, the
        # residual std after each block:
        #
        #     norm_first=True    1.02 -> 1.05 -> 1.07 -> 1.09   (drifts)
        #     norm_first=False   1.00 -> 1.00 -> 1.00 -> 1.00   (pinned)
        #
        # So Pre-LN needs one LayerNorm after the stack -- it is part of the
        # recipe, not an extra, and omitting it is a common way to make Pre-LN
        # look worse than it is. torch agrees: nn.TransformerEncoder takes a
        # `norm=` argument for exactly this. Post-LN needs nothing, because its
        # block already ends on `norm2(x + ff(x))` -- hence Identity.
        self.final_norm = nn.LayerNorm(dim) if norm == "pre" else nn.Identity()

        # Fixed sinusoidal positions; saved with the model and moved by .to().
        positions = torch.arange(max_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        angles = positions * frequencies
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : dim // 2])
        self.register_buffer("position_encoding", pe[None, :, :])

    def forward(self, X, lengths):
        """Return per-token features [B, L, dim].

        X is [B, L, input_dim]; lengths is [B] on X.device. Inputs are
        right-padded and every sequence has at least one real token.
        """
        B, L, _ = X.shape
        if L > self.position_encoding.size(1):
            raise ValueError("sequence length exceeds the positional encoding limit")

        padding_mask = torch.arange(L, device=X.device)[None, :] >= lengths[:, None]
        X = self.projection(X)
        X = self.dropout(X + self.position_encoding[:, :L].to(dtype=X.dtype))
        for layer in self.layers:
            # KEYWORD, not positional: forward(src, src_mask, src_key_padding_mask).
            # The 2nd positional arg is src_mask, an [L, L] attention mask -- a
            # [B, L] padding mask handed to it is a different thing entirely.
            X = layer(X, src_key_padding_mask=padding_mask)
        X = self.final_norm(X)

        # Padded QUERIES still produce features (only padded KEYS are masked
        # inside attention), so zero them here. Nothing downstream reads them
        # -- the loss ignores those label slots and the metrics slice by
        # length -- but leaving them as arbitrary activations would make any
        # debugging print of `outputs` misleading.
        return X.masked_fill(padding_mask[:, :, None], 0.0)


# ---- Quick self-test: run this file directly --------------------------------
# python model_transformer/encoder.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, dim = 3, 7, 16, 32
    x = torch.randn(B, L, D)
    lengths = torch.tensor([7, 4, 1])

    for norm in ("pre", "post"):
        enc = TransformerEncoder(
            D, dim=dim, group=4, num_layers=2, dropout=0.0, max_len=64, norm=norm
        ).eval()
        with torch.no_grad():
            out = enc(x, lengths)
        n_ln = sum(1 for m in enc.modules() if isinstance(m, nn.LayerNorm))
        print(
            f"[{norm}-LN] outputs {tuple(out.shape)} (expected (3, 7, 32))  "
            f"LayerNorms={n_ln}  "
            f"padded zeroed={bool(out[1, 4:].abs().sum() == 0)}"
        )
    print("(pre has one more LayerNorm than post: the final_norm after the stack)")

    # Padding-invariance: a row's real positions must not depend on how much
    # padding follows it. Padded KEYS are masked out of attention, which is
    # what makes this hold.
    enc = TransformerEncoder(
        D, dim=dim, group=4, num_layers=2, dropout=0.0, max_len=64, norm="pre"
    ).eval()
    with torch.no_grad():
        out = enc(x, lengths)
        out2 = enc(torch.cat([x, torch.randn(B, 5, D)], dim=1), lengths)
    same = all(
        torch.allclose(out[i, :n], out2[i, :n], atol=1e-5) for i, n in enumerate(lengths.tolist())
    )
    print(f"\ninvariant to extra padding (even NON-zero padding): {same}")

    # Positions must matter -- see the module docstring. Shuffle the tokens of
    # row 0 and the features must not simply follow them around.
    with torch.no_grad():
        shuffled = x.clone()
        shuffled[0] = x[0].flip(0)
        out3 = enc(shuffled, lengths)
    print(
        "reversing a sentence changes its features:",
        not bool(torch.allclose(out[0].flip(0), out3[0], atol=1e-5)),
        "(expected True -- attention alone would be permutation-invariant)",
    )
