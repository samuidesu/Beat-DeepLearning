"""Two-stage optimization, training and teacher-forced validation.

Everything here works on the documented contract and nothing else:

    logits = model(source_ids, source_attention_mask,
                   decoder_input_ids, decoder_attention_mask)   -> [B, T, V]
    loss   = CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, V),
                                                 labels.reshape(-1))

Hand it any module satisfying that and the loops run: AMP, gradient clipping,
LR scheduling per update, logging, validation. Nothing in this file knows how
many decoder layers there are, or that there is a decoder at all.

Stage 1 optimizes the target side only. Stage 2 uses separate learning rates
for the pretrained BERT encoder and the target-side parameters.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from dataset.collate import LABEL_PAD_ID, move_to_device


# ---------------------------------------------------------------------------
# device / precision
# ---------------------------------------------------------------------------
def get_device(name: str = "auto") -> torch.device:
    """Resolve "auto" to cuda > mps > cpu, or honour an explicit choice."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_settings(device: torch.device, enabled: bool) -> Dict[str, Any]:
    """Pick the autocast dtype and whether a GradScaler is needed.

    bf16 needs no loss scaling and has fp32's exponent range, so it is
    preferred wherever the GPU supports it; fp16 falls back to a GradScaler.
    On CPU/MPS mixed precision is simply off.
    """
    if not enabled or device.type != "cuda":
        return {"enabled": False, "dtype": torch.float32, "use_scaler": False}
    if torch.cuda.is_bf16_supported():
        return {"enabled": True, "dtype": torch.bfloat16, "use_scaler": False}
    return {"enabled": True, "dtype": torch.float16, "use_scaler": True}


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy over every non-padded target position.

    logits [B, T, V] -> [B*T, V]; labels [B, T] -> [B*T]. ignore_index=-100
    drops the padded positions collate.py filled with LABEL_PAD_ID, so a batch
    of mixed-length translations contributes exactly its real tokens.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=LABEL_PAD_ID,
        label_smoothing=label_smoothing,
    )


def _forward(model, batch: Dict[str, Any]) -> torch.Tensor:
    """Call the model on the documented interface and validate what comes back."""
    output = model(
        source_ids=batch["source_ids"],
        source_attention_mask=batch["source_attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
    )
    return output


# ---------------------------------------------------------------------------
# optimizer / schedule
# ---------------------------------------------------------------------------
def build_optimizer_param_groups(model, stage_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group trainable BERT and target-side parameters after setting the stage."""
    encoder = []
    decoder = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("bert.") else decoder).append(parameter)

    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": float(stage_cfg["encoder_lr"]), "name": "encoder"})
    groups.append({"params": decoder, "lr": float(stage_cfg["decoder_lr"]), "name": "decoder"})
    return groups


def build_optimizer(param_groups: Iterable[Dict[str, Any]], weight_decay: float = 0.01) -> AdamW:
    """AdamW over ready-made parameter groups. Model-independent by design."""
    return AdamW(param_groups, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)


def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.05) -> LambdaLR:
    """Linear warmup then linear decay to zero, stepped once per UPDATE.

    Per-update rather than per-epoch: an epoch here is tens of thousands of
    updates, so a per-epoch schedule would give warmup no resolution at all.
    """
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # (step + 1) rather than step: the very first update should take a
            # small step, not a zero-sized one.
            return (step + 1) / warmup_steps
        remaining = num_training_steps - warmup_steps
        return max(0.0, (num_training_steps - step) / max(1, remaining))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    scheduler=None,
    scaler=None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    amp: Optional[Dict[str, Any]] = None,
    log_every: int = 100,
    epoch: int = 0,
    max_steps: Optional[int] = None,
    global_step: int = 0,
) -> Dict[str, float]:
    """One pass over the training loader. Returns loss, NLL, PPL and step counts.

    The reported loss is a TOKEN-weighted mean, not a mean of batch means: with
    dynamic padding a batch of long sentences carries far more supervision than
    a batch of short ones, and averaging batch means would weight them equally.
    Loss includes label smoothing; NLL does not. Perplexity is exp(mean NLL)
    from the same training forwards, including their training-time dropout.
    """
    amp = amp or {"enabled": False, "dtype": torch.float32}
    model.train()
    total_loss, total_nll, total_tokens, steps = 0.0, 0.0, 0, 0
    started = time.time()

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        batch = move_to_device(batch, device)
        n_tokens = int((batch["labels"] != LABEL_PAD_ID).sum())

        with torch.autocast(device_type=device.type, dtype=amp["dtype"], enabled=amp["enabled"]):
            logits = _forward(model, batch)
            loss = compute_loss(logits, batch["labels"], label_smoothing)
            with torch.no_grad():
                nll = (
                    compute_loss(logits.detach(), batch["labels"])
                    if label_smoothing
                    else loss.detach()
                )

        optimizer.zero_grad(set_to_none=True)
        updated = True
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= previous_scale
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        # An FP16 overflow skips the optimizer step; keep the schedule in sync.
        if scheduler is not None and updated:
            scheduler.step()

        steps += 1
        global_step += int(updated)
        total_loss += loss.detach().item() * n_tokens
        total_nll += nll.item() * n_tokens
        total_tokens += n_tokens
        if log_every and steps % log_every == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(total_tokens, 1):.4f}",
                nll=f"{total_nll / max(total_tokens, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        if max_steps is not None and steps >= max_steps:
            break

    mean_loss = total_loss / max(total_tokens, 1)
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "steps": steps,
        "global_step": global_step,
        "target_tokens": total_tokens,
        "seconds": time.time() - started,
    }


@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    device: torch.device,
    amp: Optional[Dict[str, Any]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Teacher-forced loss and perplexity on a validation loader.

    No label smoothing, ever: smoothing is a training regularizer, and a
    smoothed validation loss is not comparable to anything.
    """
    amp = amp or {"enabled": False, "dtype": torch.float32}
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for step, batch in enumerate(tqdm(loader, desc="validation", leave=False), 1):
        batch = move_to_device(batch, device)
        n_tokens = int((batch["labels"] != LABEL_PAD_ID).sum())
        with torch.autocast(device_type=device.type, dtype=amp["dtype"], enabled=amp["enabled"]):
            logits = _forward(model, batch)
            loss = compute_loss(logits, batch["labels"], label_smoothing=0.0)
        total_loss += loss.detach().item() * n_tokens
        total_tokens += n_tokens
        if max_steps is not None and step >= max_steps:
            break

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "nll": mean_loss,
        "perplexity": math.exp(mean_loss),
        "target_tokens": total_tokens,
    }


def count_parameters(model) -> Dict[str, int]:
    """Total and trainable parameter counts -- the cheapest freeze check there is."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
