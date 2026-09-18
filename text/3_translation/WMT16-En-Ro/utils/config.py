"""Load and validate configs/default.yaml.

The config is a plain nested dict -- no wrapper class, no attribute access.
`cfg["lengths"]["max_train_source_length"]` is longer to type than
`cfg.lengths.max_train_source_length` and says exactly where the value came
from, which is worth more here.

What this module adds on top of yaml.safe_load:

  1. FAIL LOUDLY on a config that is wrong, not merely on one that is
     incomplete. Every check below corresponds to a mistake that would
     otherwise corrupt results silently:
       - a case-folding target tokenizer  (Romania == romania),
       - an uncased source checkpoint     (Apple == apple),
       - long_train_pair_policy: truncate (half a source supervising a whole
         target -- see the README),
       - a training length limit above BERT's 512 positional embeddings.
  2. PATH RESOLUTION relative to the project root, so the scripts work from
     any working directory.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

# .../text/3_translation/WMT16-En-Ro -- the folder holding train.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

# Sections every config must define.
_REQUIRED_SECTIONS = (
    "dataset",
    "source_tokenizer",
    "model",
    "target_tokenizer",
    "lengths",
    "training",
    "stage1",
    "stage2",
    "evaluation",
    "paths",
)


def project_path(*parts: str) -> str:
    """Resolve a config-relative path against the project root.

    Absolute paths pass through unchanged, so a config may point the artifacts
    directory at another disk.
    """
    joined = os.path.join(*parts)
    if os.path.isabs(joined):
        return joined
    return os.path.join(PROJECT_ROOT, joined)


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Read a YAML config, validate it, and return it as a nested dict."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got {type(cfg).__name__}: {path}"
        )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Raise on any configuration that would silently produce wrong results."""
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Config is missing required section(s): {missing}")

    # --- the source side must stay cased ---------------------------------
    src_name = str(cfg["source_tokenizer"]["name"])
    if "uncased" in src_name.lower():
        raise ValueError(
            f"source_tokenizer.name = {src_name!r} is an UNCASED checkpoint.\n"
            "Machine translation is case-sensitive: an uncased tokenizer folds "
            "Apple -> apple, US -> us, May -> may before the encoder ever sees "
            "them, and the target side cannot recover what was deleted. "
            "Use google-bert/bert-base-cased."
        )

    # --- the target side must stay cased ---------------------------------
    tgt = cfg["target_tokenizer"]
    if str(tgt.get("type")) != "sentencepiece":
        raise ValueError(
            f"target_tokenizer.type must be 'sentencepiece', got {tgt.get('type')!r}"
        )
    if str(tgt.get("model_type")).lower() != "bpe":
        raise ValueError(
            f"target_tokenizer.model_type must be 'bpe', got {tgt.get('model_type')!r}"
        )
    if bool(tgt.get("lowercase", False)):
        raise ValueError(
            "target_tokenizer.lowercase must be false. Romanian output is "
            "case-sensitive and BLEU is scored against cased references."
        )
    norm = str(tgt.get("normalization_rule_name", ""))
    if norm.endswith("_cf"):
        raise ValueError(
            f"target_tokenizer.normalization_rule_name = {norm!r} is a "
            "case-FOLDING rule. Use nmt_nfkc, which normalizes without "
            "destroying capitalization."
        )
    if int(tgt.get("vocab_size", 0)) <= 0:
        raise ValueError("target_tokenizer.vocab_size must be a positive integer")

    # --- length policy ----------------------------------------------------
    lengths = cfg["lengths"]
    policy = str(lengths.get("long_train_pair_policy"))
    if policy != "filter":
        raise ValueError(
            f"lengths.long_train_pair_policy = {policy!r} is not supported; the "
            "only legal value is 'filter'.\n"
            "Truncating a translation pair is not a length policy, it is a "
            "labelling error: a truncated source paired with a complete target "
            "asks the model to generate content it was never shown."
        )
    eval_policy = str(lengths.get("evaluation_overlength_policy"))
    if eval_policy not in ("error", "warn"):
        raise ValueError(
            "lengths.evaluation_overlength_policy must be 'error' or 'warn', "
            f"got {eval_policy!r}. Silently dropping or truncating official "
            "validation/test examples changes the benchmark."
        )

    max_positions = int(lengths["max_source_positions"])
    if int(lengths["max_train_source_length"]) > max_positions:
        raise ValueError(
            f"lengths.max_train_source_length = {lengths['max_train_source_length']} "
            f"exceeds lengths.max_source_positions = {max_positions}. The planned "
            "encoder (bert-base) has a finite positional table; a longer source "
            "cannot be encoded at all."
        )
    for key in ("max_train_source_length", "max_train_target_length",
                "max_eval_target_length"):
        if int(lengths[key]) < 4:
            raise ValueError(f"lengths.{key} = {lengths[key]} is implausibly small")

    # --- metrics ----------------------------------------------------------
    unknown = [m for m in (cfg["evaluation"].get("metrics") or [])
               if m not in ("sacrebleu", "chrf")]
    if unknown:
        raise ValueError(f"evaluation.metrics contains unknown entries: {unknown}")


def describe_config(cfg: Dict[str, Any]) -> str:
    """One compact block summarising the settings that change results."""
    lengths, training = cfg["lengths"], cfg["training"]
    ds, tgt = cfg["dataset"], cfg["target_tokenizer"]
    return (
        f"  dataset            : {ds['name']} / {ds['config']} "
        f"({ds['source_language']} -> {ds['target_language']})\n"
        f"  source tokenizer   : {cfg['source_tokenizer']['name']}\n"
        f"  target tokenizer   : sentencepiece {tgt['model_type']}, "
        f"vocab {tgt['vocab_size']}, norm {tgt['normalization_rule_name']}\n"
        f"  train length limit : source <= {lengths['max_train_source_length']} "
        f"(incl. [CLS]/[SEP]), target <= {lengths['max_train_target_length']} "
        f"(incl. <bos>/<eos>)   [PROVISIONAL]\n"
        f"  long train pairs   : {lengths['long_train_pair_policy']}\n"
        f"  eval overlength    : {lengths['evaluation_overlength_policy']}\n"
        f"  batch size         : {training['batch_size']} "
        f"(eval {training['eval_batch_size']})\n"
        f"  seed               : {training['seed']}"
    )
