import torch
import torch.nn as nn


class Transformer(nn.Module):
    """One encoder block: multi-head self-attention + FFN, both residual.

    `norm` decides WHERE the LayerNorm sits, which is the only difference
    between the 2017 original and every Transformer trained since:

      "post"  x = LN(x + Sublayer(x))     -- Vaswani et al. 2017.
              A LayerNorm sits ON the residual path, so the gradient reaching
              layer 1 has been rescaled once per layer above it. Early in
              training that makes the effective step size at the bottom
              layers depend on depth, which is why Post-LN needs learning-rate
              WARMUP -- and needs it more the deeper the stack. Skipping
              warmup on a deep Post-LN stack does not train slowly, it fails
              to leave chance level at all.

      "pre"   x = x + Sublayer(LN(x))     -- Xiong et al. 2020, and what GPT,
              LLaMA and friends use. The residual path is now an unbroken
              identity from input to output, so gradients reach layer 1
              unscaled and no warmup is required. The cost is that the
              residual stream is never normalized, so its magnitude grows
              with depth -- which is why a Pre-LN stack needs one final
              LayerNorm after the last block (see TransformerEncoder).

    The two LayerNorm modules keep the names MHA_post / FFN_post under both
    settings. The suffix is historical and now slightly wrong for "pre", but
    renaming them would invalidate every Transformer checkpoint already in
    outputs_*/ -- a worse trade than an inaccurate attribute name.
    """

    def __init__(self, dim, group=1, dropout=0.1, norm="post"):
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

        # Hold the residual aside so one forward can serve both arrangements:
        # pre-norm normalizes the SUBLAYER INPUT, post-norm normalizes the SUM.
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

        X = residual + self.dropout1(MHA)
        if self.norm == "post":
            X = self.MHA_post(X)

        # FFN
        residual = X
        if self.norm == "pre":
            X = self.FFN_post(X)
        X = residual + self.dropout2(self.FFN(X))
        if self.norm == "post":
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
