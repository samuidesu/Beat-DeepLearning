import math

import torch
import torch.nn as nn
from .bert_encoder import BertEncoder
from .transformer_naive import TransformerDecoder


class Translater(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        dim=768,
        group=1,
        dropout=0.1,
        decoder_layers=4,
        max_target_positions=512,
        norm="pre",
        source_model_name="google-bert/bert-base-cased",
    ):
        super().__init__()
        self.bert = BertEncoder(source_model_name)
        self._encoder_frozen = False

        self.target_embedding = nn.Embedding(
            num_embeddings=target_vocab_size,
            embedding_dim=dim,
            padding_idx=0,
        )

        # Fixed sinusoidal positions: [1, max_target_positions, dim].
        # Buffers are saved with the model and follow model.to(device).
        positions = torch.arange(max_target_positions, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        angles = positions * frequencies
        pe = torch.zeros(max_target_positions, dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : dim // 2])
        self.register_buffer("position_encoding", pe[None, :, :])

        self.decoder = nn.ModuleList(
            [
                TransformerDecoder(dim, group=group, dropout=dropout, norm=norm)
                for _ in range(decoder_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(dim) if norm == "pre" else nn.Identity()
        self.prediction = nn.Linear(dim, target_vocab_size)
        self.prediction.weight = self.target_embedding.weight

    def freeze_encoder(self):
        """Stage 1: freeze BERT weights and disable its dropout."""
        self.bert.requires_grad_(False)
        self._encoder_frozen = True
        self.bert.eval()

    def unfreeze_all(self):
        """Stage 2: fine-tune BERT together with the complete target side."""
        self.requires_grad_(True)
        self._encoder_frozen = False
        self.bert.train(self.training)

    def train(self, mode=True):
        """Keep a frozen BERT in evaluation mode during decoder training."""
        super().train(mode)
        if self._encoder_frozen:
            self.bert.eval()
        return self

    def forward(
        self,
        source_ids,
        source_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        """Source IDs [B, S] and target IDs [B, T] -> logits [B, T, V]."""
        source = self.encode(source_ids, source_attention_mask)
        return self.decode(source, source_attention_mask,
                           decoder_input_ids, decoder_attention_mask)

    def encode(self, source_ids, source_attention_mask):
        """Source IDs [B, S] -> memory [B, S, dim], reused during generation."""
        return self.bert(source_ids, source_attention_mask)

    def decode(self, source, source_attention_mask,
               decoder_input_ids, decoder_attention_mask, last_token_only=False):
        """Memory and prefix -> logits [B, T, V], or [B, 1, V] for generation."""
        target_length = decoder_input_ids.shape[1]
        x = self.target_embedding(decoder_input_ids)  # [B, T, dim]
        x = x + self.position_encoding[:, :target_length].to(dtype=x.dtype)
        target_padding = decoder_attention_mask == 0
        source_padding = source_attention_mask == 0

        for layer in self.decoder:
            x = layer(x, target_padding, source, source_padding)
        x = self.final_norm(x)
        if last_token_only:
            x = x[:, -1:, :]
        # Scale the tied-weight dot products; the output bias stays unscaled.
        x = x * (self.target_embedding.embedding_dim ** -0.5)
        return self.prediction(x)
