import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, dim, group=1, dropout=0.1, norm="pre"):
        super().__init__()
        if group <= 0 or dim % group != 0:
            raise ValueError("group must be positive and divide dim")
        if norm not in ("pre", "post"):
            raise ValueError("norm must be 'pre' or 'post'")

        self.dim = dim
        self.group = group
        self.norm = norm
        self.X_Q = nn.Linear(dim, dim, bias=True)
        self.X_K = nn.Linear(dim, dim, bias=True)
        self.X_V = nn.Linear(dim, dim, bias=True)
        self.W_O = nn.Linear(dim, dim, bias=True)
        self.MHA_post = nn.LayerNorm(dim)
        self.attention_dropout_1 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

        # CA parameters
        self.CA_norm = nn.LayerNorm(dim)
        self.X_Q_CA = nn.Linear(dim, dim, bias=True)
        self.X_K_CA = nn.Linear(dim, dim, bias=True)
        self.X_V_CA = nn.Linear(dim, dim, bias=True)
        self.W_O_CA = nn.Linear(dim, dim, bias=True)
        self.attention_dropout_2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        ## FFN
        self.FFN = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

        self.FFN_post = nn.LayerNorm(dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, X, padding_mask, cross_X, cross_padding):
        # X: (batch, L, dim)
        # padding_mask: [B, L] bool on X.device; True = PAD.
        # Each sequence must contain at least one non-PAD token.
        B, L, _ = X.shape

        # Hold the residual aside so one forward can serve both arrangements:
        # pre-norm normalizes the SUBLAYER INPUT, post-norm normalizes the SUM.

        # 1. self attention
        residual = X
        if self.norm == "pre":
            X = self.MHA_post(X)

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

        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=X.device),
            diagonal=1,
        )

        mask = (
            padding_mask[:, None, None, :]  # [B, 1, 1, L]
            | causal_mask[None, None, :, :]  # [1, 1, L, L]
        )
        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )
        scores = torch.softmax(scores / (head_dim**0.5), dim=-1)
        scores = self.attention_dropout_1(scores)
        heads = scores @ V
        heads = heads.transpose(1, 2)
        concat = heads.reshape(B, L, self.dim)
        MHA = self.W_O(concat)

        X = residual + self.dropout1(MHA)
        if self.norm == "post":
            X = self.MHA_post(X)

        # 2. cross attention
        souce_L = cross_X.shape[1]
        residual = X
        if self.norm == "pre":
            X = self.CA_norm(X)

        Q = self.X_Q_CA(X)
        K = self.X_K_CA(cross_X)
        V = self.X_V_CA(cross_X)

        Q = Q.view(B, L, self.group, head_dim)
        K = K.view(B, souce_L, self.group, head_dim)
        V = V.view(B, souce_L, self.group, head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores.masked_fill(
            cross_padding[:, None, None, :],
            float("-inf"),
        )
        scores = torch.softmax(scores / (head_dim**0.5), dim=-1)
        scores = self.attention_dropout_2(scores)
        heads = scores @ V
        heads = heads.transpose(1, 2)
        concat = heads.reshape(B, L, self.dim)
        MHA = self.W_O_CA(concat)
        X = residual + self.dropout2(MHA)
        if self.norm == "post":
            X = self.CA_norm(X)

        # 3. FFN
        residual = X
        if self.norm == "pre":
            X = self.FFN_post(X)
        X = residual + self.dropout3(self.FFN(X))
        if self.norm == "post":
            X = self.FFN_post(X)
        return X
