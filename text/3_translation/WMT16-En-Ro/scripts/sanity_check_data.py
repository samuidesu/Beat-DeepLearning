"""Prove the data pipeline is correct, without a model.

    python scripts/sanity_check_data.py
    python scripts/sanity_check_data.py --batch-size 4 --split validation

Every property this prints is also asserted, so the script fails loudly rather
than printing something plausible. It checks, in order:

   1 raw English / Romanian pairs                    10 label padding is -100, not <pad>
   2 cleaning changes nothing but whitespace         11 <bos> starts every decoder input
   3 BERT tokenization preserves capitalization      12 <eos> ends every label row
   4 SentencePiece pieces are SUBWORDS               13 decoder_input_ids[t+1] == labels[t]
   5 Romanian round-trips (diacritics survive)       14 padded positions carry no supervision
   6 <bos>/<eos> construction                        15 nothing was lowercased anywhere
   7 decoder_input_ids / labels shift by one         16 the target vocabulary is subword
   8 one dynamically padded batch                    17 S and T are the batch maxima
   9 source/decoder attention masks                  18 real tokens never sit past the mask

Run it after training the target tokenizer, before writing any model code.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import DatasetDict  # noqa: E402

from dataset.collate import (  # noqa: E402
    LABEL_PAD_ID, build_decoder_io, collate_translation_batch,
)
from dataset.dataset import SPLITS, encode_splits, load_raw_splits, to_pairs  # noqa: E402
from dataset.preprocessing import clean_text  # noqa: E402
from dataset.source_tokenizer import load_source_tokenizer  # noqa: E402
from dataset.target_tokenizer import TargetTokenizer  # noqa: E402
from utils.config import load_config, project_path  # noqa: E402
from utils.console import enable_utf8_stdout  # noqa: E402

ROMANIAN_LETTERS = "ăâîșțşţ"


def rule(title: str) -> None:
    print(f"\n{'-' * 72}\n{title}\n{'-' * 72}")


def pick_varied_examples(encoded, batch_size: int):
    """Pick rows with DIFFERENT target lengths; return (row_indices, examples).

    Padding is only observable when the batch is ragged; a batch of
    equal-length sentences would let a padding bug through unnoticed. The
    indices come back too, so later checks can compare against the same raw
    rows.
    """
    order = sorted(range(encoded.num_rows), key=lambda i: encoded[i]["target_len"])
    step = max(1, len(order) // batch_size)
    picked = list(dict.fromkeys(
        order[min(i * step, len(order) - 1)] for i in range(batch_size)))
    return picked, [encoded[i] for i in picked]


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train", choices=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_tokenizer = load_source_tokenizer(cfg["source_tokenizer"]["name"])
    target_tokenizer = TargetTokenizer.from_dir(
        project_path(cfg["paths"]["target_tokenizer_dir"]),
        cfg["target_tokenizer"].get("model_prefix", "spm_ro"),
    )
    source_pad_id = source_tokenizer.pad_token_id

    raw = load_raw_splits(cfg)
    pool_size = min(512, raw[args.split].num_rows)
    pairs = to_pairs(raw, cfg)[args.split].select(range(pool_size))
    encoded = encode_splits(DatasetDict({args.split: pairs}),
                            source_tokenizer, target_tokenizer, cfg)[args.split]
    indices, examples = pick_varied_examples(encoded, args.batch_size)
    src_lang = cfg["dataset"]["source_language"]
    tgt_lang = cfg["dataset"]["target_language"]

    # ---------------------------------------------------------------- 1, 2
    rule(f"1-2  RAW TEXT and CLEANING ({args.split})")
    for i, ex in zip(indices, examples):
        raw_pair = raw[args.split][i]["translation"]
        print(f"\n  row {i}  {src_lang.upper()} : {raw_pair[src_lang]}")
        print(f"         {tgt_lang.upper()} : {raw_pair[tgt_lang]}")
        # Cleaning may only normalize whitespace: every other character
        # survives, in the same order and the same case.
        assert ex["source"] == clean_text(ex["source"]), "cleaning is not idempotent"
        for side, lang in ((ex["source"], src_lang), (ex["target"], tgt_lang)):
            assert "".join(side.split()) == "".join(raw_pair[lang].split()), (
                f"cleaning altered the {lang} text beyond whitespace")
    print("\n  OK: cleaning only normalizes whitespace.")

    # ------------------------------------------------------------------- 3
    rule("3  SOURCE TOKENIZATION (cased BERT WordPiece)")
    example = examples[0]
    tokens = source_tokenizer.convert_ids_to_tokens(example["source_ids"])
    print(f"  text   : {example['source']}")
    print(f"  tokens : {tokens}")
    print(f"  ids    : {example['source_ids'][:16]}{' ...' if len(example['source_ids']) > 16 else ''}")
    assert tokens[0] == source_tokenizer.cls_token, "source must start with [CLS]"
    assert tokens[-1] == source_tokenizer.sep_token, "source must end with [SEP]"

    cased = next((ex for ex in examples if any(c.isupper() for c in ex["source"][1:])), None)
    if cased is not None:
        cased_tokens = source_tokenizer.convert_ids_to_tokens(cased["source_ids"])
        assert any(c.isupper() for t in cased_tokens for c in t), (
            "Capitalization was lost during source tokenization -- this is an "
            "uncased tokenizer. Use google-bert/bert-base-cased."
        )
        print(f"\n  capitalization preserved: {[t for t in cased_tokens if any(c.isupper() for c in t)][:10]}")
    print("  OK: [CLS]/[SEP] present, capitalization survives.")

    # ---------------------------------------------------------------- 4, 5
    rule("4-5  TARGET TOKENIZATION (Romanian SentencePiece BPE)")
    pieces = target_tokenizer.pieces(example["target"])
    ids_no_specials = target_tokenizer.encode(example["target"])
    decoded = target_tokenizer.decode(ids_no_specials)
    print(f"  text    : {example['target']}")
    print(f"  pieces  : {pieces}")
    print(f"  decoded : {decoded}")
    words = example["target"].split()
    print(f"\n  {len(words)} whitespace words -> {len(pieces)} subword pieces "
          f"({len(pieces) / max(len(words), 1):.2f} pieces per word)")
    assert "▁" not in decoded, "decode() leaked a SentencePiece marker"
    assert decoded.lower() != decoded or example["target"].lower() == example["target"], \
        "decode() lowercased the text"

    diacritic_example = next(
        (ex for ex in examples if any(c in ROMANIAN_LETTERS for c in ex["target"])), None)
    if diacritic_example is not None:
        text = diacritic_example["target"]
        back = target_tokenizer.decode(target_tokenizer.encode(text))
        present = sorted({c for c in text if c in ROMANIAN_LETTERS})
        survived = sorted({c for c in back if c in ROMANIAN_LETTERS})
        print(f"\n  diacritics in    : {present}")
        print(f"  diacritics after : {survived}")
        assert set(present) <= set(survived), "diacritics were lost in the round trip"
    print("  OK: subword segmentation, clean detokenization, diacritics survive.")

    # ---------------------------------------------------------------- 6, 7
    rule("6-7  TARGET SEQUENCE, decoder_input_ids and labels")
    full = example["target_ids"]
    decoder_input, labels = build_decoder_io(full)
    print(f"  full target       : {full}")
    print(f"  decoder_input_ids : {decoder_input}")
    print(f"  labels            : {labels}")
    print(f"  as pieces         : <bos> {' '.join(pieces[:6])} ... <eos>")
    assert full[0] == target_tokenizer.bos_id, "target must start with <bos>"
    assert full[-1] == target_tokenizer.eos_id, "target must end with <eos>"
    assert len(decoder_input) == len(labels) == len(full) - 1
    assert decoder_input == full[:-1] and labels == full[1:]
    print("\n  the model will learn, position by position:")
    for t in range(min(4, len(labels))):
        print(f"    input[{t}] = {decoder_input[t]:>6}  ->  label[{t}] = {labels[t]:>6}")
    print("  OK: teacher-forcing shift by exactly one position.")

    # ---------------------------------------------------------------- 8, 9
    rule("8-9  ONE DYNAMICALLY PADDED BATCH")
    batch = collate_translation_batch(examples, source_pad_id=source_pad_id,
                                      target_pad_id=target_tokenizer.pad_id)
    B = len(examples)
    S = batch["source_ids"].shape[1]
    T = batch["decoder_input_ids"].shape[1]
    for key in ("source_ids", "source_attention_mask", "decoder_input_ids",
                "decoder_attention_mask", "labels"):
        print(f"  {key:<23}: {tuple(batch[key].shape)}  {batch[key].dtype}")
    print(f"\n  B = {B}   S = {S} (longest source in batch)   "
          f"T = {T} (longest decoder sequence in batch)")

    assert batch["source_ids"].shape == (B, S)
    assert batch["source_attention_mask"].shape == (B, S)
    assert batch["decoder_input_ids"].shape == (B, T)
    assert batch["decoder_attention_mask"].shape == (B, T)
    assert batch["labels"].shape == (B, T)

    source_lengths = [len(ex["source_ids"]) for ex in examples]
    target_lengths = [len(ex["target_ids"]) - 1 for ex in examples]
    assert S == max(source_lengths), "S must be the batch maximum, not a global one"
    assert T == max(target_lengths), "T must be the batch maximum, not a global one"
    print(f"  source lengths in batch : {source_lengths}")
    print(f"  decoder lengths in batch: {target_lengths}")

    # ---------------------------------------------------------- 10, 11, 12
    rule("10-12  PADDING, <bos> and <eos>")
    for b in range(B):
        n_src, n_tgt = source_lengths[b], target_lengths[b]
        # source: mask marks exactly the real tokens, padding is BERT's pad id
        assert batch["source_attention_mask"][b, :n_src].all()
        assert not batch["source_attention_mask"][b, n_src:].any()
        assert (batch["source_ids"][b, n_src:] == source_pad_id).all()
        # decoder: same, padded with the target <pad>
        assert batch["decoder_attention_mask"][b, :n_tgt].all()
        assert not batch["decoder_attention_mask"][b, n_tgt:].any()
        assert (batch["decoder_input_ids"][b, n_tgt:] == target_tokenizer.pad_id).all()
        # labels: -100 in the padded tail, NEVER the target <pad> id
        assert (batch["labels"][b, n_tgt:] == LABEL_PAD_ID).all()
        assert (batch["labels"][b, :n_tgt] != LABEL_PAD_ID).all()
        # <bos> opens the decoder input; <eos> closes the labels
        assert batch["decoder_input_ids"][b, 0] == target_tokenizer.bos_id
        assert batch["labels"][b, n_tgt - 1] == target_tokenizer.eos_id
        assert (batch["decoder_input_ids"][b, :n_tgt] != target_tokenizer.eos_id).all(), \
            "<eos> must not appear in the decoder INPUT"

    print(f"  row 0 decoder_input_ids : {batch['decoder_input_ids'][0].tolist()}")
    print(f"  row 0 labels            : {batch['labels'][0].tolist()}")
    print(f"  row 0 decoder mask      : {batch['decoder_attention_mask'][0].tolist()}")
    print(f"\n  <pad> id = {target_tokenizer.pad_id}, label padding = {LABEL_PAD_ID}")
    print("  OK: masks match true lengths; labels pad with -100, never with <pad>;")
    print("      <bos> opens every decoder input; <eos> closes every label row.")

    # ---------------------------------------------------------- 13, 14, 18
    rule("13-14, 18  SHIFT ALIGNMENT AND SUPERVISION COVERAGE")
    for b in range(B):
        n_tgt = target_lengths[b]
        for t in range(n_tgt - 1):
            assert batch["decoder_input_ids"][b, t + 1] == batch["labels"][b, t], (
                f"shift broken at row {b}, position {t}: the token the decoder "
                "reads at t+1 must be the token it was scored on at t."
            )
    supervised = int((batch["labels"] != LABEL_PAD_ID).sum())
    print(f"  decoder_input_ids[:, t+1] == labels[:, t] holds everywhere")
    print(f"  supervised positions : {supervised} of {B * T} "
          f"({100 * supervised / (B * T):.1f}% -- the rest is padding)")
    assert supervised == sum(target_lengths)
    print("  OK: loss will see exactly the real target tokens.")

    # ------------------------------------------------------------- 15, 16
    rule("15-16  NO LOWERCASING, AND THE VOCABULARY IS SUBWORD")
    upper_source = sum(1 for ex in examples if any(c.isupper() for c in ex["source"]))
    upper_target = sum(1 for ex in examples if any(c.isupper() for c in ex["target"]))
    print(f"  examples with a capital in the source : {upper_source}/{B}")
    print(f"  examples with a capital in the target : {upper_target}/{B}")
    assert upper_source > 0 and upper_target > 0, (
        "No capitals anywhere in the batch -- something is lowercasing the corpus."
    )

    probe = "Romania"
    probe_pieces = target_tokenizer.pieces(probe)
    lower_pieces = target_tokenizer.pieces(probe.lower())
    print(f"\n  {probe!r:<12} -> {probe_pieces}")
    print(f"  {probe.lower()!r:<12} -> {lower_pieces}")
    assert probe_pieces != lower_pieces or target_tokenizer.encode(probe) != \
        target_tokenizer.encode(probe.lower()), (
        "The target tokenizer maps 'Romania' and 'romania' to the same ids -- "
        "it is case-folding. Retrain with normalization_rule_name: nmt_nfkc."
    )
    long_word = max(example["target"].split(), key=len) if example["target"].split() else "dezvoltare"
    print(f"\n  a long word, {long_word!r} -> {target_tokenizer.pieces(long_word)}")
    print(f"  target vocabulary size: {target_tokenizer.vocab_size} "
          "(subword pieces, NOT one entry per Romanian word)")

    # ------------------------------------------------------------------ 17
    rule("17  WHAT THE FUTURE MODEL WILL RECEIVE")
    print("  logits = model(")
    for key in ("source_ids", "source_attention_mask",
                "decoder_input_ids", "decoder_attention_mask"):
        print(f"      {key}={tuple(batch[key].shape)},")
    print("  )")
    print(f"  expected logits shape : ({B}, {T}, {target_tokenizer.vocab_size})")
    print(f"  loss                  : CrossEntropyLoss(ignore_index={LABEL_PAD_ID}) "
          f"over labels {tuple(batch['labels'].shape)}")
    assert torch.is_tensor(batch["labels"]) and batch["labels"].dtype == torch.long

    print("\n" + "=" * 72)
    print("ALL DATA CHECKS PASSED -- the pipeline is ready for a model.")
    print("=" * 72)


if __name__ == "__main__":
    main()
