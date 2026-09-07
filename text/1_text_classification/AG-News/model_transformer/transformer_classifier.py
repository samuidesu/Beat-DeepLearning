"""Embedding -> Transformer encoder -> pooling -> topic logits."""

import torch.nn as nn

from model.embedding import TokenEmbedding
from model.head import ClassifierHead
from .encoder import TransformerEncoder


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=4, embed_dim=100, dim=128,
                 group=4, num_layers=2, pooling="mean", dropout=0.1,
                 pad_idx=0, max_len=128, pretrained_vectors=None):
        super().__init__()
        self.pooling = pooling
        self.max_len = max_len
        # Input dropout is applied once, after projection + positional encoding.
        self.embedding = TokenEmbedding(
            vocab_size, embed_dim, pad_idx=pad_idx,
            pretrained=pretrained_vectors, dropout=0.0,
        )
        self.encoder = TransformerEncoder(
            input_dim=embed_dim, dim=dim, group=group, num_layers=num_layers,
            dropout=dropout, max_len=max_len,
        )
        self.head = ClassifierHead(
            in_features=self.encoder.out_dim, num_classes=num_classes,
            pooling=pooling, dropout=dropout,
        )

    def forward(self, ids, lengths):
        """Map token ids [B, L] and true lengths [B] to raw logits [B, C]."""
        lengths = lengths.to(ids.device)
        vectors = self.embedding(ids)
        outputs, final = self.encoder(vectors, lengths)
        return self.head(outputs, final, lengths)

    def freeze_embedding(self):
        self.embedding.freeze()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def parameter_groups(self):
        """Keep the same three learning-rate groups as RNNClassifier."""
        return {
            name: [p for p in module.parameters() if p.requires_grad]
            for name, module in (("embedding", self.embedding),
                                 ("encoder", self.encoder),
                                 ("head", self.head))
        }
