"""Dynamic padding: a list of encoded examples -> one batch of tensors.

DYNAMIC PADDING means every batch is padded to its own longest member, not to
a global maximum. WMT16 sentences are mostly short with a long tail, so a
fixed 128-wide batch would be mostly padding; padding to the batch maximum
moves the same data in a fraction of the memory and time. It changes nothing
about the result -- padded positions are masked out of attention and out of
the loss either way.

WHAT THE DECODER SIDE LOOKS LIKE. For one example with target pieces
y1..yN, dataset.py stored the full sequence

    target_ids        = <bos> y1 y2 ... yN <eos>

and this module splits it by one position:

    decoder_input_ids = <bos> y1 y2 ... yN            (what the decoder reads)
    labels            =       y1 y2 ... yN <eos>      (what it must predict)

so that position t of the input is scored against position t of the labels:
<bos> -> y1, y1 -> y2, ..., yN -> <eos>. That is teacher forcing, and it is
why the model learns to stop: predicting <eos> after yN is a scored decision.

THREE PAD VALUES, THREE JOBS:
    source_ids        BERT's own pad id      masked by source_attention_mask
    decoder_input_ids <pad> (target id 0)    masked by decoder_attention_mask
    labels            -100                   ignored by CrossEntropyLoss

Labels must NOT use the target <pad> id. Padding is id 0 in the target
vocabulary, which is a real class the LM head can emit; training on it would
teach the model to produce padding. -100 is torch's ignore_index sentinel and
never reaches the loss.

NO CAUSAL MASK IS BUILT HERE. This module produces padding masks only. The
triangular causal mask belongs to the decoder's self-attention, which does not
exist yet -- and combining the two is part of implementing it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

# CrossEntropyLoss(ignore_index=-100) drops these positions.
LABEL_PAD_ID = -100


def build_decoder_io(target_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Split a full target sequence into (decoder_input_ids, labels).

    >>> build_decoder_io([2, 10, 11, 3])          # <bos> y1 y2 <eos>
    ([2, 10, 11], [10, 11, 3])
    """
    if len(target_ids) < 2:
        raise ValueError(
            f"target sequence {list(target_ids)} is too short; it must contain "
            "at least <bos> and <eos>."
        )
    return list(target_ids[:-1]), list(target_ids[1:])


def _pad(sequences: List[List[int]], pad_value: int) -> torch.Tensor:
    """Right-pad a list of id lists to the longest one -> [B, L] int64."""
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for row, seq in enumerate(sequences):
        out[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def collate_translation_batch(examples: List[Dict[str, Any]], source_pad_id: int,
                              target_pad_id: int) -> Dict[str, Any]:
    """Collate encoded examples into the batch the future model will consume.

    Returns, with S = longest source and T = longest decoder sequence IN THIS
    BATCH:

        source_ids             [B, S]  int64   [CLS] ... [SEP] + padding
        source_attention_mask  [B, S]  int64   1 = real token, 0 = padding
        decoder_input_ids      [B, T]  int64   <bos> y1 ... yN + padding
        decoder_attention_mask [B, T]  int64   1 = real token, 0 = padding
        labels                 [B, T]  int64   y1 ... yN <eos> + (-100)

    plus the raw strings under "source_text" / "target_text" when the split
    still carries them -- evaluation scores against the original reference
    text, not against a re-decoded copy of it.
    """
    source_ids = [ex["source_ids"] for ex in examples]
    decoder_inputs, labels = zip(*(build_decoder_io(ex["target_ids"]) for ex in examples))

    source = _pad(source_ids, source_pad_id)
    decoder_input_ids = _pad(list(decoder_inputs), target_pad_id)
    # Masks are built from the true lengths, not by comparing against the pad
    # id: a real token that happens to equal the pad id would break that.
    source_attention_mask = torch.zeros_like(source)
    for row, seq in enumerate(source_ids):
        source_attention_mask[row, :len(seq)] = 1
    decoder_attention_mask = torch.zeros_like(decoder_input_ids)
    for row, seq in enumerate(decoder_inputs):
        decoder_attention_mask[row, :len(seq)] = 1

    batch: Dict[str, Any] = {
        "source_ids": source,
        "source_attention_mask": source_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": _pad(list(labels), LABEL_PAD_ID),
    }
    if "source" in examples[0]:
        batch["source_text"] = [ex["source"] for ex in examples]
    if "target" in examples[0]:
        batch["target_text"] = [ex["target"] for ex in examples]
    return batch


def move_to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    """Move every tensor in a batch to `device`; leave the text lists alone."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}
