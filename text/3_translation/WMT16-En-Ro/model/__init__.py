"""BERT encoder and handwritten Transformer translation model."""

from typing import Any, Dict

from .bert_encoder import BertEncoder
from .translation import Translater

__all__ = ["BertEncoder", "Translater", "build_model"]


def build_model(cfg: Dict[str, Any], target_vocab_size: int) -> Translater:
    """Build the model used by training and checkpoint evaluation."""
    return Translater(
        target_vocab_size=target_vocab_size,
        source_model_name=cfg["source_tokenizer"]["name"],
        **cfg["model"],
    )
