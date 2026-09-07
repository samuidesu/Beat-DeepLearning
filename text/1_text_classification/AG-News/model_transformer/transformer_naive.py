import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, dim, group=1, dropout=0.1):
        super().__init__()
        if group <= 0 or dim % group != 0:
            raise ValueError("group must be positive and divide dim")

        self.dim = dim
        self.group = group
        self.X_Q = nn.Linear(dim, dim, bias=True)
        self.X_K = nn.Linear(dim, dim, bias=True)
        self.X_V = nn.Linear(dim, dim, bias=True)
        self.W_O = nn.Linear(dim, dim, bias=True)
        self.MHA_post = nn.LayerNorm(dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.FFN = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

        self.FFN_post = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, padding_mask):
        # X: (batch, L, dim)
        # padding_mask: [B, L] bool on X.device; True = PAD.
        # Each sequence must contain at least one non-PAD token.
        B, L, _ = X.shape

        # MHA
        Q = self.X_Q(X)
        K = self.X_K(X)
        V = self.X_V(X)

        head_dim = self.dim // self.group

        Q = Q.view(B, L, self.group, head_dim)
        K = K.view(B, L, self.group, head_dim)
        V = V.view(B, L, self.group, head_dim)

        Q = Q.transpose(1, 2)  # [B,num_heads,L,head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores.masked_fill(
            padding_mask[:, None, None, :],
            float("-inf"),
        )
        scores = torch.softmax(scores / (head_dim**0.5), dim=-1)
        scores = self.attention_dropout(scores)
        heads = scores @ V
        heads = heads.transpose(1, 2)
        concat = heads.reshape(B, L, self.dim)
        MHA = self.W_O(concat)

        X = X + self.dropout1(MHA)

        # FFN
        X = self.MHA_post(X)

        X = X + self.dropout2(self.FFN(X))
        X = self.FFN_post(X)
        return X
        # return output


# class TransformerLayer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         group,
#     ):
#         super().__init__()
