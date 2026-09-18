"""Look at the raw corpus before trusting any pipeline built on it.

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --split validation --num-examples 10

Prints the official split sizes, the raw record structure, a handful of real
English/Romanian pairs before and after cleaning, character-length statistics,
and a count of empty or suspicious pairs.

No tokenizers are needed, so this runs before
scripts/train_target_tokenizer.py. It reads all three splits but only to
count and display them -- nothing here trains on anything.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset.dataset import SPLITS, load_raw_splits, to_pairs  # noqa: E402
from dataset.preprocessing import clean_text  # noqa: E402
from utils.config import describe_config, load_config  # noqa: E402
from utils.console import enable_utf8_stdout  # noqa: E402


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train", choices=list(SPLITS))
    parser.add_argument("--num-examples", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Configuration")
    print(describe_config(cfg))

    raw = load_raw_splits(cfg)
    src_lang = cfg["dataset"]["source_language"]
    tgt_lang = cfg["dataset"]["target_language"]

    print("\nOfficial splits (used exactly as shipped -- never re-split)")
    for split in SPLITS:
        print(f"  {split:<11}: {raw[split].num_rows:>9,} pairs")

    print(f"\nRaw record structure ({args.split})")
    print(f"  features: {raw[args.split].features}")
    first = raw[args.split][0]
    print(f"  example : {first}")

    print(f"\n{args.num_examples} raw pairs from {args.split}, "
          "exactly as they arrive")
    for i in range(min(args.num_examples, raw[args.split].num_rows)):
        pair = raw[args.split][i]["translation"]
        source, target = pair[src_lang], pair[tgt_lang]
        print(f"\n  [{i}] {src_lang.upper()}  : {source}")
        print(f"      {tgt_lang.upper()}  : {target}")
        cleaned_source, cleaned_target = clean_text(source), clean_text(target)
        if cleaned_source != source or cleaned_target != target:
            print(f"      cleaned {src_lang}: {cleaned_source}")
            print(f"      cleaned {tgt_lang}: {cleaned_target}")
        else:
            print("      (cleaning changed nothing)")

    # --- cleaned-corpus health ------------------------------------------
    pairs = to_pairs(raw, cfg)
    split = pairs[args.split]
    print(f"\nCleaned {args.split} split: {split.num_rows:,} pairs")

    empty_source = split.filter(lambda ex: not ex["source"]).num_rows
    empty_target = split.filter(lambda ex: not ex["target"]).num_rows
    print(f"  empty after cleaning : source {empty_source:,}, target {empty_target:,}")

    # Character lengths are a cheap, tokenizer-free sanity signal. Token
    # lengths -- the ones that decide the length thresholds -- come from
    # scripts/analyze_lengths.py.
    sample = split.select(range(min(20000, split.num_rows)))
    src_chars = [len(s) for s in sample["source"]]
    tgt_chars = [len(s) for s in sample["target"]]
    print(f"  character length (first {len(src_chars):,} pairs)")
    print(f"    source: mean {sum(src_chars) / len(src_chars):6.1f}  max {max(src_chars)}")
    print(f"    target: mean {sum(tgt_chars) / len(tgt_chars):6.1f}  max {max(tgt_chars)}")

    # Capitalization and diacritics must both be visible in the cleaned text;
    # if either has vanished, something upstream is folding the corpus.
    has_upper = sum(1 for s in sample["target"] if any(c.isupper() for c in s))
    romanian_letters = "ăâîșțşţ"
    has_diacritics = sum(1 for s in sample["target"]
                         if any(c in romanian_letters for c in s))
    print(f"  target sentences containing a capital letter : {has_upper:,} / {len(tgt_chars):,}")
    print(f"  target sentences containing a diacritic      : {has_diacritics:,} / {len(tgt_chars):,}")

    print("\nNext: python scripts/train_target_tokenizer.py, "
          "then python scripts/analyze_lengths.py")


if __name__ == "__main__":
    main()
