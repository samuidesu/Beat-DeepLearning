"""Measure the real token-length distribution before choosing any threshold.

    python scripts/analyze_lengths.py
    python scripts/analyze_lengths.py --split validation
    python scripts/analyze_lengths.py --max-examples 100000   # quick estimate

Source lengths are counted with google-bert/bert-base-cased (including [CLS]
and [SEP]); target lengths with the trained Romanian SentencePiece model
(including <bos> and <eos>), because those are the sequences the model
actually sees.

The output is the input to a DECISION, and the decision is yours: this script
writes statistics, never config. It also reports, for each candidate
threshold, how many PAIRS survive filtering both sides -- which is the number
that actually matters, since an over-long pair is dropped whole.

Requires the target tokenizer: run scripts/train_target_tokenizer.py first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset.dataset import (  # noqa: E402
    LENGTH_BUCKETS, SPLITS, encode_splits, load_raw_splits, to_pairs,
)
from dataset.source_tokenizer import load_source_tokenizer  # noqa: E402
from dataset.target_tokenizer import TargetTokenizer  # noqa: E402
from utils.config import load_config, project_path  # noqa: E402
from utils.console import enable_utf8_stdout  # noqa: E402

TRUNCATION_WARNING = """
------------------------------------------------------------------------
DO NOT TURN THESE NUMBERS INTO A TRUNCATION SETTING.

In classification, truncating a long input costs some context. In
translation it corrupts the supervision: a source cut at 128 tokens
paired with its complete target teaches the model to invent the part of
the sentence it was never shown, and it teaches that on exactly the
hardest examples in the corpus.

The policy in this project is therefore to DROP the whole pair on the
TRAIN split (lengths.long_train_pair_policy: filter), and to refuse to
quietly alter validation/test at all
(lengths.evaluation_overlength_policy: error).

Choose max_train_source_length / max_train_target_length by looking at
p95 / p99 / max below and at the pair-retention table: a threshold that
keeps ~99% of pairs costs almost no data and bounds the memory an
attention batch needs.
------------------------------------------------------------------------
"""


def percentile_stats(lengths: np.ndarray) -> Dict[str, float]:
    """mean / p50 / p90 / p95 / p99 / max, plus the bucket counts."""
    stats = {
        "count": int(lengths.size),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }
    stats["over"] = {
        str(bucket): {
            "count": int((lengths > bucket).sum()),
            "fraction": float((lengths > bucket).mean()),
        }
        for bucket in LENGTH_BUCKETS
    }
    return stats


def print_stats(title: str, stats: Dict[str, float]) -> None:
    print(f"\n{title}  ({stats['count']:,} examples)")
    print(f"  mean {stats['mean']:8.2f}   std {stats['std']:7.2f}   "
          f"min {stats['min']:>4}   max {stats['max']:>5}")
    print(f"  p50  {stats['p50']:8.1f}   p90 {stats['p90']:7.1f}   "
          f"p95 {stats['p95']:6.1f}   p99 {stats['p99']:6.1f}")
    print("  over threshold:")
    for bucket, entry in stats["over"].items():
        print(f"    > {bucket:>3} tokens : {entry['count']:>9,}  "
              f"({100 * entry['fraction']:6.3f}%)")


def pair_retention(src: np.ndarray, tgt: np.ndarray,
                   thresholds: Sequence[int]) -> List[Dict[str, float]]:
    """How many pairs survive filtering BOTH sides at each threshold."""
    rows = []
    total = src.size
    for threshold in thresholds:
        keep = int(((src <= threshold) & (tgt <= threshold)).sum())
        rows.append({
            "threshold": int(threshold),
            "pairs_kept": keep,
            "pairs_dropped": total - keep,
            "fraction_kept": keep / total,
        })
    return rows


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train", choices=list(SPLITS))
    parser.add_argument("--max-examples", type=int, default=None,
                        help="analyze only the first N pairs (a faster estimate)")
    parser.add_argument("--save", default=None,
                        help="JSON output path (default: paths.dataset_stats_path)")
    parser.add_argument("--no-save", action="store_true", help="print only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_tokenizer = load_source_tokenizer(cfg["source_tokenizer"]["name"])
    target_tokenizer = TargetTokenizer.from_dir(
        project_path(cfg["paths"]["target_tokenizer_dir"]),
        cfg["target_tokenizer"].get("model_prefix", "spm_ro"),
    )

    pairs = to_pairs(load_raw_splits(cfg), cfg)
    split = pairs[args.split]
    if args.max_examples:
        split = split.select(range(min(args.max_examples, split.num_rows)))
    from datasets import DatasetDict
    encoded = encode_splits(DatasetDict({args.split: split}),
                            source_tokenizer, target_tokenizer, cfg)[args.split]

    src = np.asarray(encoded["source_len"])
    tgt = np.asarray(encoded["target_len"])

    print(f"\n=== WMT16 {cfg['dataset']['config']} :: {args.split} split ===")
    print(f"source tokenizer : {cfg['source_tokenizer']['name']} "
          "(lengths include [CLS] and [SEP])")
    print(f"target tokenizer : {target_tokenizer.model_path} "
          f"(vocab {target_tokenizer.vocab_size}; lengths include <bos> and <eos>)")

    source_stats = percentile_stats(src)
    target_stats = percentile_stats(tgt)
    print_stats("SOURCE (English, WordPiece)", source_stats)
    print_stats("TARGET (Romanian, SentencePiece BPE)", target_stats)

    # The ratio is worth a glance: a target/source ratio far from ~1 means the
    # two vocabularies segment at very different granularities, which changes
    # what a shared length limit costs on each side.
    ratio = float(np.mean(tgt / np.maximum(src, 1)))
    print(f"\n  mean target/source length ratio: {ratio:.3f}")

    retention = pair_retention(src, tgt, LENGTH_BUCKETS)
    print("\nPAIR RETENTION -- filtering BOTH sides at one threshold")
    print("  threshold   pairs kept      dropped    kept %")
    for row in retention:
        print(f"  {row['threshold']:>9}   {row['pairs_kept']:>10,}   "
              f"{row['pairs_dropped']:>10,}   {100 * row['fraction_kept']:6.3f}")

    lengths_cfg = cfg["lengths"]
    print(f"\nCurrent config (PROVISIONAL, not derived from these numbers):")
    print(f"  max_train_source_length : {lengths_cfg['max_train_source_length']}")
    print(f"  max_train_target_length : {lengths_cfg['max_train_target_length']}")
    print(f"  max_source_positions    : {lengths_cfg['max_source_positions']} "
          "(bert-base hard limit)")
    print(TRUNCATION_WARNING)

    if not args.no_save:
        out_path = args.save or project_path(cfg["paths"]["dataset_stats_path"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "dataset": f"{cfg['dataset']['name']}/{cfg['dataset']['config']}",
            "split": args.split,
            "analyzed_examples": int(src.size),
            "max_examples_flag": args.max_examples,
            "source_tokenizer": cfg["source_tokenizer"]["name"],
            "target_tokenizer": {
                "path": target_tokenizer.model_path,
                "vocab_size": target_tokenizer.vocab_size,
            },
            "source_includes_special_tokens": ["[CLS]", "[SEP]"],
            "target_includes_special_tokens": ["<bos>", "<eos>"],
            "source_lengths": source_stats,
            "target_lengths": target_stats,
            "mean_target_over_source_ratio": ratio,
            "pair_retention": retention,
            "config_lengths_at_analysis_time": lengths_cfg,
        }
        # One file per split, so a validation run never overwrites the train
        # statistics the thresholds were chosen from.
        if args.split != "train":
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}_{args.split}{ext}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Statistics written to {out_path}")


if __name__ == "__main__":
    main()
