"""Pretrained English BERT encoder for the translation model."""

import torch
import torch.nn as nn
from transformers import BertModel


class BertEncoder(nn.Module):
    """Return one contextual representation per source token.

    Use the same checkpoint name as the source tokenizer. The default is
    google-bert/bert-base-cased, whose hidden size is 768. Parameters remain
    trainable; the training stage decides when to freeze or unfreeze them.
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-cased",
        *,
        local_files_only: bool = False,
    ):
        super().__init__()
        # Cross-attention needs all token states, so the CLS pooler is unused.
        # Set local_files_only=True to require already-cached model files.
        self.bert = BertModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            local_files_only=local_files_only,
        )
        self.hidden_size = self.bert.config.hidden_size
        self.max_source_positions = self.bert.config.max_position_embeddings

    def forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode source_ids [B, S] into hidden states [B, S, hidden_size].

        source_attention_mask: [B, S], 1 = real token, 0 = PAD. Both tensors
        must be on the same device as the encoder. Inputs use right padding.
        PAD output states are not necessarily zero; the decoder must still
        mask them using cross_padding = (source_attention_mask == 0).
        """
        if source_ids.ndim != 2:
            raise ValueError("source_ids must have shape [B, S]")
        if source_attention_mask.shape != source_ids.shape:
            raise ValueError("source_attention_mask must match source_ids [B, S]")
        if source_ids.shape[1] > self.max_source_positions:
            raise ValueError(
                f"Source length {source_ids.shape[1]} exceeds BERT's position "
                f"limit {self.max_source_positions}; inputs are not truncated."
            )

        # BERT supplies position IDs and single-sentence token type IDs.
        outputs = self.bert(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state
