"""The English (source) tokenizer: cased BERT WordPiece.

There is no English vocabulary to build in this project. The source encoder
will be a pretrained checkpoint, so the source vocabulary is fixed by that
checkpoint -- using any other tokenization would feed the encoder ids that
mean nothing to it.

CASED, and that is the entire decision in this file. bert-base-uncased
lowercases and strips accents inside its own normalizer, before WordPiece
runs. For classification that is usually harmless. For translation it deletes
information the target side needs:

    Apple  ->  apple      (a company becomes a fruit)
    US     ->  us         (a country becomes a pronoun)
    May    ->  may        (a month becomes a modal verb)

Romanian keeps those distinctions, so a model trained on folded English is
being asked to restore capitalization it never received.

Special tokens are BERT's own: [CLS] ... [SEP], added by the tokenizer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_source_tokenizer(name: str) -> PreTrainedTokenizerBase:
    """Load the cased BERT tokenizer named in the config.

    Raises on an uncased checkpoint rather than accepting it: the failure mode
    is silent (training simply gets worse) and is worth catching at load time.
    """
    if "uncased" in name.lower():
        raise ValueError(
            f"{name!r} is an uncased checkpoint. This project needs a CASED "
            "source tokenizer -- see the module docstring."
        )
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        raise ValueError(f"{name!r} has no pad token; batching needs one.")
    return tokenizer


def encode_source_batch(texts: List[str],
                        tokenizer: PreTrainedTokenizerBase) -> List[List[int]]:
    """Encode a batch of English strings to id lists, WITHOUT truncation.

    truncation=False is deliberate. Length is handled once, explicitly, by the
    filtering policy in dataset.py -- if the tokenizer truncated here, an
    over-long pair would quietly become a corrupt training example instead of
    being dropped.

    verbose=False silences the per-call "sequence longer than model_max_length"
    warning: over-length sources are expected here and are counted properly
    downstream.
    """
    return tokenizer(
        texts,
        add_special_tokens=True,   # [CLS] ... [SEP]
        truncation=False,
        padding=False,             # dynamic padding happens in collate.py
        verbose=False,
    )["input_ids"]


def source_special_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """The ids the rest of the pipeline needs from the source tokenizer."""
    return {
        "pad_id": tokenizer.pad_token_id,
        "cls_id": tokenizer.cls_token_id,
        "sep_id": tokenizer.sep_token_id,
        "unk_id": tokenizer.unk_token_id,
        "vocab_size": tokenizer.vocab_size,
    }
