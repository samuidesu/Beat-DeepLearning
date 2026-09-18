"""Saving, loading and resuming -- state dicts only, no architecture.

A checkpoint here is a dict of state dicts plus the bookkeeping needed to
continue a run: epoch, global step, the best validation metric so far, the
config that produced it, stage, random states and tokenizer metadata.

The tokenizer matters: the LM head has one row
per target vocabulary entry, so a checkpoint is only meaningful with the
tokenizer it was trained against. Retrain the SentencePiece model with a
different vocab_size or a different corpus and every id shifts; the weights
would load without complaint and produce fluent nonsense. `save_checkpoint`
records the vocab size and model path. The size check catches incompatible
heads; keep the original tokenizer files, since equal sizes do not imply
equal token-to-ID mappings.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import torch
import numpy as np

BEST_NAME = "best.pt"
LAST_NAME = "last.pt"


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, global_step: int = 0,
                    best_metric: Optional[float] = None,
                    config: Optional[Dict[str, Any]] = None,
                    target_vocab_size: Optional[int] = None,
                    target_tokenizer_path: Optional[str] = None,
                    stage: int = 1,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """Write one checkpoint. Returns the path written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "stage": stage,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "target_vocab_size": target_vocab_size,
        "target_tokenizer_path": target_tokenizer_path,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    payload.update(extra or {})
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Read a checkpoint file into a plain dict.

    weights_only=False because the payload carries the config dict alongside
    the tensors. Only load checkpoints you produced.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_into(checkpoint: Dict[str, Any], model, optimizer=None, scheduler=None,
              scaler=None, strict: bool = True,
              restore_rng: bool = False) -> Dict[str, Any]:
    """Restore state into live objects. Returns the resume bookkeeping.

    Pass `optimizer`/`scheduler`/`scaler` to continue a run; omit them to load
    weights only, which is what evaluation wants.
    """
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and checkpoint.get("rng_state") is not None:
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_metric": checkpoint.get("best_metric"),
    }


def assert_tokenizer_matches(checkpoint: Dict[str, Any], target_vocab_size: int) -> None:
    """Refuse a checkpoint whose LM head does not match the loaded tokenizer."""
    saved = checkpoint.get("target_vocab_size")
    if saved is not None and int(saved) != int(target_vocab_size):
        raise ValueError(
            f"Checkpoint was trained with a target vocabulary of {saved} "
            f"tokens, but the loaded tokenizer has {target_vocab_size}.\n"
            f"Checkpoint tokenizer: {checkpoint.get('target_tokenizer_path')}\n"
            "Every id would mean something different. Use the tokenizer this "
            "checkpoint was trained with, or retrain."
        )
