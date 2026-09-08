"""Projection, positional encoding and a stack of hand-written Transformer layers.

Copied unchanged from the AG-News project. One consequence of the longer
sequences here is worth stating explicitly: attention is O(L^2) in both time
and memory, so moving from AG News' L=128 to IMDB's L=400 multiplies the
attention matrix by ~10x per layer per head. That -- not the parameter count,
which is identical -- is why config.BATCH_SIZE drops to 64.
"""

import math

import torch
import torch.nn as nn

from .transformer_naive import Transformer


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, dim=128, group=4, num_layers=2,
                 dropout=0.1, max_len=128, norm="post"):
        super().__init__()
        if dim <= 0 or num_layers <= 0 or max_len <= 0:
            raise ValueError("dim, num_layers and max_len must be positive")

        self.out_dim = dim
        self.norm = norm
        self.projection = (nn.Linear(input_dim, dim) if input_dim != dim
                           else nn.Identity())
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            Transformer(dim, group, dropout, norm=norm) for _ in range(num_layers)
        ])
        # Pre-LN never normalizes the residual stream, so its magnitude grows
        # with depth and the last block's output would reach the pooling
        # unnormalized. One final LayerNorm is part of the Pre-LN recipe, not
        # an extra -- omitting it is a common way to make Pre-LN look worse
        # than it is. Identity for Post-LN, whose FFN_post already normalized
        # the output, and which therefore keeps its state_dict unchanged.
        self.final_norm = nn.LayerNorm(dim) if norm == "pre" else nn.Identity()

        # Fixed sinusoidal positions; saved with the model and moved by .to().
        positions = torch.arange(max_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        angles = positions * frequencies
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, :dim // 2])
        self.register_buffer("position_encoding", pe[None, :, :])

    def forward(self, X, lengths):
        """Return token features [B, L, dim] and last real-token features [B, dim].

        X is [B, L, input_dim]; lengths is [B] on X.device.
        Inputs are right-padded and every sequence has at least one real token.
        """
        B, L, _ = X.shape
        if L > self.position_encoding.size(1):
            raise ValueError("sequence length exceeds the positional encoding limit")

        padding_mask = torch.arange(L, device=X.device)[None, :] >= lengths[:, None]
        X = self.projection(X)
        X = self.dropout(X + self.position_encoding[:, :L].to(dtype=X.dtype))
        for layer in self.layers:
            X = layer(X, padding_mask)
        X = self.final_norm(X)

        # Padded queries can produce features; exclude them from the output.
        outputs = X.masked_fill(padding_mask[:, :, None], 0.0)
        final = outputs[torch.arange(B, device=X.device), lengths - 1]
        return outputs, final
