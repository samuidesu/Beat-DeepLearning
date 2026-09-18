"""Train the Romanian SentencePiece BPE tokenizer. TRAIN SPLIT ONLY.

    python scripts/train_target_tokenizer.py
    python scripts/train_target_tokenizer.py --vocab-size 32000 --force

What it does:
  1. loads the official WMT16 ro-en TRAIN split,
  2. keeps only the Romanian side,
  3. applies the same whitespace cleanup the training pipeline applies,
  4. trains SentencePiece BPE with case-preserving nmt_nfkc normalization,
  5. reserves <pad>=0, <unk>=1, <bos>=2, <eos>=3,
  6. saves to artifacts/tokenizer_ro/ and round-trips a diacritics sample.

It never reads the validation or test split. The merges learned here are
model parameters, and learning them from evaluation text leaks the benchmark
into the model.

An existing tokenizer is left alone unless --force is given: retraining shifts
every id, which silently invalidates every checkpoint trained against it.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset.dataset import load_raw_splits, to_pairs  # noqa: E402
from dataset.target_tokenizer import (  # noqa: E402
    TargetTokenizer, tokenizer_exists, tokenizer_paths, train_target_tokenizer,
)
from utils.config import load_config, project_path  # noqa: E402
from utils.console import enable_utf8_stdout  # noqa: E402

# Romanian sentence carrying all five diacritics; used as the round-trip check.
DIACRITICS_SAMPLE = (
    "In Romania, tara mea, stiinta si increderea au fost intotdeauna importante."
)
DIACRITICS_REAL = (
    "În România, țara mea, știința și "
    "încrederea au fost întotdeauna importante."
)


def iter_target_sentences(train_split):
    """Stream the cleaned Romanian side, one sentence at a time.

    A generator rather than a list or a dumped text file: the train split is
    ~600k sentences, and SentencePiece accepts an iterator directly.
    """
    kept = 0
    for row in train_split:
        text = row["target"]
        if text:
            kept += 1
            yield text
    print(f"  fed {kept:,} non-empty Romanian TRAIN sentences to SentencePiece")


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="override target_tokenizer.vocab_size")
    parser.add_argument("--output-dir", default=None,
                        help="override paths.target_tokenizer_dir")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if a tokenizer already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tok_cfg = cfg["target_tokenizer"]
    vocab_size = args.vocab_size or int(tok_cfg["vocab_size"])
    model_dir = args.output_dir or project_path(cfg["paths"]["target_tokenizer_dir"])
    prefix = tok_cfg.get("model_prefix", "spm_ro")

    if tokenizer_exists(model_dir, prefix) and not args.force:
        print(f"A tokenizer already exists at {tokenizer_paths(model_dir, prefix)['model']}")
        print("Nothing to do. Pass --force to retrain -- note that retraining "
              "changes every token id and invalidates existing checkpoints.")
        return

    print("Loading WMT16 TRAIN split (validation/test are NOT read here)...")
    pairs = to_pairs(load_raw_splits(cfg), cfg)
    train_split = pairs["train"]
    print(f"  train pairs: {train_split.num_rows:,}")

    print(f"\nTraining SentencePiece {tok_cfg['model_type']} "
          f"(vocab {vocab_size}, norm {tok_cfg['normalization_rule_name']}, "
          f"character_coverage {tok_cfg['character_coverage']})")
    model_path = train_target_tokenizer(
        sentences=iter_target_sentences(train_split),
        model_dir=model_dir,
        vocab_size=vocab_size,
        model_type=tok_cfg["model_type"],
        character_coverage=float(tok_cfg["character_coverage"]),
        normalization_rule_name=tok_cfg["normalization_rule_name"],
        input_sentence_size=int(tok_cfg.get("input_sentence_size", 0)),
        shuffle_input_sentence=bool(tok_cfg.get("shuffle_input_sentence", True)),
        model_prefix=prefix,
        seed=int(cfg["training"]["seed"]),
        metadata={
            "dataset": f"{cfg['dataset']['name']}/{cfg['dataset']['config']}",
            "trained_on_split": "train",
            "language": cfg["dataset"]["target_language"],
            "train_pairs": train_split.num_rows,
        },
    )
    print(f"\nSaved: {model_path}")

    # --- verification -----------------------------------------------------
    tokenizer = TargetTokenizer(model_path)
    print(f"\nVocabulary size : {tokenizer.vocab_size}")
    print(f"Special ids     : <pad>={tokenizer.pad_id} <unk>={tokenizer.unk_id} "
          f"<bos>={tokenizer.bos_id} <eos>={tokenizer.eos_id}")

    for sample in (DIACRITICS_REAL, DIACRITICS_SAMPLE):
        ids = tokenizer.encode(sample)
        back = tokenizer.decode(ids)
        print(f"\n  text    : {sample}")
        print(f"  pieces  : {tokenizer.pieces(sample)}")
        print(f"  ids     : {ids}")
        print(f"  decoded : {back}")
        if back != sample:
            # nmt_nfkc is a normalizer, so a round trip is allowed to differ in
            # normalization -- but never in case, and never by losing a letter.
            print("  NOTE: decoded text differs from the input (normalization).")
        assert back.lower() != back, (
            "Round-trip lost capitalization -- the tokenizer is case-folding.\n"
            f"  in : {sample!r}\n  out: {back!r}\n"
            "normalization_rule_name must be nmt_nfkc, not nmt_nfkc_cf."
        )
        lost = {c for c in sample if c in "ăâîșț"
                and c not in back}
        assert not lost, (
            f"Round-trip lost Romanian characters {sorted(lost)}.\n"
            "Set target_tokenizer.character_coverage to 1.0 and retrain."
        )
    print("\nRound-trip OK: diacritics and capitalization survive encode/decode.")


if __name__ == "__main__":
    main()
