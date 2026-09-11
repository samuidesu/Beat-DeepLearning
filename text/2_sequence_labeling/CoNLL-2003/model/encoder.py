"""BiLSTM encoder: word/case vectors -> contextual per-token features. The "neck".

    vectors [B, L, D]  --BiLSTM-->  outputs [B, L, H*2]

ONE CELL, NOT THREE. The classification projects made the recurrent cell
selectable (rnn / lstm / gru) because the comparison was the experiment. That
question is answered and does not reopen here: IMDB measured a 20.4-point gap
between a vanilla RNN and a GRU at 400 timesteps, against 3.7 points at AG
News' 128. CoNLL sentences average 14 tokens, so the memory-horizon pressure
that produced those numbers is simply absent, and a three-cell sweep would
measure noise. The LSTM stays because it is what the NER literature uses on
this corpus.

WHAT IS RETURNED IS ALSO DIFFERENT, and it is the clearest single sign that
the task changed. The classification encoders returned two things:

    outputs [B, L, H*2]   per-token features
    final   [B, H*2]      the document vector the head pooled down to

A tagger has no document vector. Every one of the L positions is an answer, so
`final` is not returned at all -- there is nothing left to pool. The head above
this is a linear layer applied at every step (see head.py).

Two mechanics implemented here that are specific to variable-length text:

  PACKING. A padded batch is a rectangle, but the sentences inside are not.
  Feeding <pad> into the recurrence would keep updating the hidden state after
  the sentence ended, and -- because this is bidirectional -- the BACKWARD
  pass would start by reading k padding steps before reaching any real token,
  corrupting the representation of every position in the row.
  pack_padded_sequence reorganizes the batch into the per-timestep groups of
  still-active rows, so cuDNN steps each row exactly `length` times.

  Note how much more this matters for tagging than it did for classification.
  There, unpacked padding would have corrupted the final state -- one vector.
  Here the backward direction's contamination reaches EVERY position, so a
  missing pack is not a small bias, it is a broken model.

  BIDIRECTIONALITY. Non-negotiable for this task. The evidence that "Mark
  Jones" is a person routinely sits after it ("..., a spokesman"), and the
  evidence that "Germany" is an ORG rather than a LOC is the verb that follows
  it ("Germany beat Argentina"). A forward-only tagger cannot see either.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """Multi-layer bidirectional LSTM producing one feature vector per token.

    Args:
        input_size: embedding width D (word dim + case dim).
        hidden_size: hidden width PER DIRECTION.
        num_layers: stacked recurrent layers (each reads the one below).
        bidirectional: run a second pass right-to-left and concatenate.
        dropout: applied BETWEEN stacked layers by torch (ignored when
            num_layers == 1, which torch warns about -- hence the guard).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                       # [B, L, *] everywhere
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def out_dim(self) -> int:
        """Feature width handed to the head (hidden_size * directions)."""
        return self.hidden_size * self.num_directions

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode one padded batch.

        Input:
            x: embedded tokens [B, L, D].
            lengths: true lengths [B] (long). Must be on CPU for packing --
                this method moves them, so callers can pass a CUDA tensor.

        Output:
            outputs: per-token features [B, L, H*dirs], padded rows zeroed.
        """
        total_length = x.size(1)

        # enforce_sorted=False lets torch sort/unsort internally, so the batch
        # can stay in dataset order (sorting by length ourselves would also
        # correlate batches with sentence length, which we do not want).
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        # total_length pins the output width back to L even when the longest
        # row was shorter (defensive: our collate makes them equal). Getting
        # this wrong on a TAGGER would misalign features against labels, which
        # is far worse than the classification case where only a final state
        # was read.
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=total_length)
        return outputs                                  # [B, L, H*dirs]


# ---- Quick self-test: run this file directly --------------------------------
# python model/encoder.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, H = 3, 6, 8, 5
    x = torch.randn(B, L, D)
    lengths = torch.tensor([6, 4, 1])
    x[1, 4:] = 0                                        # zero the padded tails
    x[2, 1:] = 0

    enc = LSTMEncoder(D, H, num_layers=2, bidirectional=True, dropout=0.0).eval()
    with torch.no_grad():
        out = enc(x, lengths)
    print("outputs:", tuple(out.shape), "(expected (3, 6, 10))")
    print("out_dim:", enc.out_dim, "(expected 10)")
    print("padded positions zero:", bool(out[1, 4:].abs().sum() == 0))

    # Padding-invariance: re-encoding with MORE padding must not change the
    # features at the REAL positions. This is the bug packing exists to
    # prevent, and on a tagger it would corrupt every position, not just one.
    x_pad = torch.cat([x, torch.zeros(B, 4, D)], dim=1)
    with torch.no_grad():
        out2 = enc(x_pad, lengths)
    same = all(torch.allclose(out[i, :n], out2[i, :n], atol=1e-6)
               for i, n in enumerate(lengths.tolist()))
    print("invariant to extra padding at every real position:", same)

    # And the reason bidirectionality is listed as non-negotiable above:
    # position 0's feature must depend on tokens that come AFTER it.
    x_alt = x.clone()
    x_alt[0, 3] += 5.0                                  # change a LATER token
    with torch.no_grad():
        out_alt = enc(x_alt, lengths)
    print("feature at position 0 sees later tokens:",
          not bool(torch.allclose(out[0, 0], out_alt[0, 0], atol=1e-6)),
          "(expected True -- that is the backward direction)")
