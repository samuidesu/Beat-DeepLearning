"""Load, clean, encode and filter the three official WMT16 ro-en splits.

    load_raw_splits   ->  Hugging Face DatasetDict, untouched
    to_pairs          ->  {"source", "target"} cleaned strings
    encode_splits     ->  + source_ids, target_ids, source_len, target_len
    filter_train      ->  drops empty and over-long TRAIN pairs, with counts
    check_eval_split  ->  never drops anything; reports and enforces a policy
    prepare_splits    ->  the four above, in order, for train.py

THE SPLITS ARE THE OFFICIAL ONES. WMT16 ships train / validation / test;
nothing here re-splits, re-shuffles or mixes them. The test split is read by
evaluate.py only.

TRAIN AND EVAL ARE TREATED DIFFERENTLY, ON PURPOSE.

  train       may be filtered. Dropping a pair the model cannot learn from is
              a training decision and costs nothing but data.
  validation  may NOT be filtered silently. Dropping the hard (long) examples
  test        from a benchmark inflates the score against a corpus that is no
              longer the benchmark. Over-long evaluation pairs are counted and
              reported, and by default they stop the run.

WHAT A LENGTH MEANS HERE:
    source_len  includes [CLS] and [SEP]
    target_len  includes <bos> and <eos>  (the full teacher-forcing sequence;
                decoder_input_ids and labels are each one token shorter)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from dataset.preprocessing import clean_pair, is_non_empty_pair
from dataset.source_tokenizer import encode_source_batch

SPLITS = ("train", "validation", "test")

# Thresholds the reports count against, everywhere in the project.
LENGTH_BUCKETS = (64, 128, 256, 512)


# ---------------------------------------------------------------------------
# 1. loading
# ---------------------------------------------------------------------------
def load_raw_splits(cfg: Dict[str, Any]) -> DatasetDict:
    """load_dataset(name, config) for all three official splits.

    The first call downloads ~600k sentence pairs and caches them; later calls
    read the cache. The `wmt/<name>` retry covers newer `datasets` releases,
    where the canonical datasets moved under an organisation namespace.
    """
    name = cfg["dataset"]["name"]
    config_name = cfg["dataset"]["config"]
    cache_dir = cfg["dataset"].get("cache_dir")
    try:
        raw = load_dataset(name, config_name, cache_dir=cache_dir)
    except Exception as first_error:  # noqa: BLE001 - re-raised below
        try:
            raw = load_dataset(f"wmt/{name}", config_name, cache_dir=cache_dir)
        except Exception:
            raise first_error
    missing = [s for s in SPLITS if s not in raw]
    if missing:
        raise ValueError(
            f"{name}/{config_name} is missing official split(s) {missing}; "
            f"got {list(raw)}. This project does not invent its own splits."
        )
    return raw


def to_pairs(raw: DatasetDict, cfg: Dict[str, Any]) -> DatasetDict:
    """Flatten {"translation": {...}} into cleaned "source"/"target" columns."""
    src_lang = cfg["dataset"]["source_language"]
    tgt_lang = cfg["dataset"]["target_language"]
    num_proc = int(cfg["dataset"].get("num_proc", 1)) or None
    return raw.map(
        clean_pair,
        fn_kwargs={"source_language": src_lang, "target_language": tgt_lang},
        remove_columns=["translation"],
        num_proc=num_proc,
        desc="cleaning text",
    )


# ---------------------------------------------------------------------------
# 2. encoding
# ---------------------------------------------------------------------------
def _encode_batch(batch: Dict[str, List[str]], source_tokenizer,
                  target_tokenizer) -> Dict[str, List]:
    """Encode one map() batch. No truncation, no padding -- both happen later."""
    source_ids = encode_source_batch(batch["source"], source_tokenizer)
    # Full teacher-forcing sequence: <bos> y1 ... yN <eos>
    bos, eos = target_tokenizer.bos_id, target_tokenizer.eos_id
    target_ids = [[bos] + ids + [eos]
                  for ids in target_tokenizer.encode_batch(batch["target"])]
    return {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "source_len": [len(x) for x in source_ids],
        "target_len": [len(x) for x in target_ids],
    }


def encode_splits(pairs: DatasetDict, source_tokenizer, target_tokenizer,
                  cfg: Dict[str, Any]) -> DatasetDict:
    """Add id columns to every split. Cached by `datasets` across runs."""
    num_proc = int(cfg["dataset"].get("num_proc", 1)) or None
    return pairs.map(
        _encode_batch,
        batched=True,
        batch_size=1000,
        fn_kwargs={"source_tokenizer": source_tokenizer,
                   "target_tokenizer": target_tokenizer},
        num_proc=num_proc,
        desc="tokenizing",
    )


# ---------------------------------------------------------------------------
# 3. train filtering
# ---------------------------------------------------------------------------
def filter_train(train: Dataset, cfg: Dict[str, Any]) -> Tuple[Dataset, Dict[str, int]]:
    """Drop empty and over-long TRAIN pairs. Returns (dataset, statistics).

    Over-long pairs are dropped WHOLE. Truncating one side would pair half an
    English sentence with a complete Romanian one and supervise the model to
    hallucinate the missing half -- see the README.
    """
    lengths = cfg["lengths"]
    max_src = int(lengths["max_train_source_length"])
    max_tgt = int(lengths["max_train_target_length"])

    stats: Dict[str, int] = {"total": train.num_rows}

    # Empty after cleaning. Counted per side before the combined drop, so a
    # one-sided corpus defect is visible rather than averaged away.
    stats["removed_empty_source"] = train.filter(
        lambda ex: not ex["source"], desc="counting empty sources").num_rows
    stats["removed_empty_target"] = train.filter(
        lambda ex: not ex["target"], desc="counting empty targets").num_rows
    train = train.filter(is_non_empty_pair, desc="dropping empty pairs")

    # Over-long. numpy over the two length columns: one pass, exact per-reason
    # counts including the overlap.
    src_len = np.asarray(train["source_len"])
    tgt_len = np.asarray(train["target_len"])
    src_too_long = src_len > max_src
    tgt_too_long = tgt_len > max_tgt
    stats["removed_source_too_long"] = int(src_too_long.sum())
    stats["removed_target_too_long"] = int(tgt_too_long.sum())
    stats["removed_either_too_long"] = int((src_too_long | tgt_too_long).sum())

    keep = np.nonzero(~(src_too_long | tgt_too_long))[0]
    if keep.size != train.num_rows:
        train = train.select(keep)

    stats["final"] = train.num_rows
    stats["max_train_source_length"] = max_src
    stats["max_train_target_length"] = max_tgt
    if stats["final"] == 0:
        raise ValueError("Filtering removed every training example; check lengths.*")
    return train, stats


def print_train_filter_stats(stats: Dict[str, int]) -> None:
    """Print the filtering report. Nothing here is optional output."""
    total, final = stats["total"], stats["final"]
    print("\nTRAIN filtering")
    print(f"  total examples                : {total:,}")
    print(f"  removed, source empty         : {stats['removed_empty_source']:,}")
    print(f"  removed, target empty         : {stats['removed_empty_target']:,}")
    print(f"  removed, source > {stats['max_train_source_length']:<4d} tokens  : "
          f"{stats['removed_source_too_long']:,}")
    print(f"  removed, target > {stats['max_train_target_length']:<4d} tokens  : "
          f"{stats['removed_target_too_long']:,}")
    print(f"  removed, either side too long : {stats['removed_either_too_long']:,}")
    print(f"  final training examples       : {final:,} "
          f"({100.0 * final / max(total, 1):.2f}% of the split)")


# ---------------------------------------------------------------------------
# 4. evaluation splits: report, never silently modify
# ---------------------------------------------------------------------------
def check_eval_split(ds: Dataset, split: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Count evaluation pairs the current settings cannot handle.

    Nothing is dropped and nothing is truncated. Under
    `evaluation_overlength_policy: error` (the default) any violation raises;
    under `warn` the run continues with a loud report, and the score must then
    be reported as what it is -- a score on a modified benchmark.
    """
    lengths = cfg["lengths"]
    max_positions = int(lengths["max_source_positions"])
    max_tgt = int(lengths["max_eval_target_length"])
    policy = str(lengths["evaluation_overlength_policy"])

    src_len = np.asarray(ds["source_len"])
    tgt_len = np.asarray(ds["target_len"])
    report = {
        "split": split,
        "total": int(ds.num_rows),
        "empty_pairs": int(ds.filter(
            lambda ex: not (ex["source"] and ex["target"]),
            desc=f"checking {split} for empty pairs").num_rows),
        "source_over_max_positions": int((src_len > max_positions).sum()),
        "target_over_max_eval_length": int((tgt_len > max_tgt).sum()),
        "max_source_positions": max_positions,
        "max_eval_target_length": max_tgt,
        "source_max": int(src_len.max()),
        "target_max": int(tgt_len.max()),
    }
    report["violations"] = (report["empty_pairs"]
                            + report["source_over_max_positions"]
                            + report["target_over_max_eval_length"])

    print(f"\n{split.upper()} length check ({report['total']:,} official examples)")
    print(f"  longest source                 : {report['source_max']} tokens "
          f"(encoder limit {max_positions})")
    print(f"  longest target                 : {report['target_max']} tokens "
          f"(eval limit {max_tgt})")
    print(f"  empty pairs                    : {report['empty_pairs']:,}")
    print(f"  sources over encoder limit     : {report['source_over_max_positions']:,}")
    print(f"  targets over eval length limit : {report['target_over_max_eval_length']:,}")

    if report["violations"]:
        message = (
            f"{report['violations']:,} example(s) in the official {split} split "
            "exceed the configured limits.\n"
            "These examples are part of the benchmark. Dropping or truncating "
            "them silently would mean reporting a score on a corpus that is no "
            "longer WMT16.\n"
            "Choose explicitly:\n"
            "  - raise lengths.max_eval_target_length so the whole split fits, or\n"
            "  - set lengths.evaluation_overlength_policy: warn and state in the "
            "results that the split was modified, and how."
        )
        if policy == "error":
            raise ValueError(message)
        print("\n  WARNING: " + message.replace("\n", "\n  "))
    return report


# ---------------------------------------------------------------------------
# 5. the whole pipeline
# ---------------------------------------------------------------------------
def prepare_splits(cfg: Dict[str, Any], source_tokenizer, target_tokenizer,
                   splits: Sequence[str] = SPLITS,
                   filter_train_split: bool = True) -> Tuple[DatasetDict, Dict[str, Any]]:
    """load -> clean -> encode -> filter train -> check eval splits.

    Returns the prepared DatasetDict and a report dict holding the train
    filtering statistics and one length check per evaluation split.
    """
    raw = load_raw_splits(cfg)
    pairs = to_pairs(raw, cfg)
    pairs = DatasetDict({s: pairs[s] for s in splits})
    encoded = encode_splits(pairs, source_tokenizer, target_tokenizer, cfg)

    report: Dict[str, Any] = {"train_filtering": None, "eval_checks": {}}
    out = {}
    for split in splits:
        if split == "train" and filter_train_split:
            out[split], report["train_filtering"] = filter_train(encoded[split], cfg)
            print_train_filter_stats(report["train_filtering"])
        elif split == "train":
            out[split] = encoded[split]
        else:
            out[split] = encoded[split]
            report["eval_checks"][split] = check_eval_split(encoded[split], split, cfg)
    return DatasetDict(out), report


# ---------------------------------------------------------------------------
# 6. DataLoader
# ---------------------------------------------------------------------------
def make_dataloader(ds: Dataset, cfg: Dict[str, Any], source_pad_id: int,
                    target_pad_id: int, shuffle: bool, batch_size: int | None = None):
    """Wrap a prepared split in a DataLoader with dynamic-padding collate.

    A Hugging Face Dataset already satisfies the torch Dataset protocol
    (__len__ / __getitem__ -> dict), so there is no wrapper class here.
    """
    from functools import partial

    import torch
    from torch.utils.data import DataLoader

    from dataset.collate import collate_translation_batch
    from utils.seed import seed_worker

    training = cfg["training"]
    num_workers = int(training.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(training["seed"]))

    return DataLoader(
        ds.with_format(None),
        batch_size=batch_size or int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        collate_fn=partial(
            collate_translation_batch,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
        ),
    )
