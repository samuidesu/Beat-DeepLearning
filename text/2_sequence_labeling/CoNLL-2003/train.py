"""Training entry point for BiLSTM and Transformer NER on CoNLL-2003.

Two models, each owning its output folder:
    --model lstm         -> BiLSTM tagger          -> outputs_lstm/
    --model transformer  -> Transformer encoder    -> outputs_transformer/
so the comparison is single-variable: same vocabulary, same GloVe init, same
case feature, same loss, same schedule, same eval protocol. Only the encoder
changes.

This is the repo's FIRST sequence-labeling project, and three things in this
file differ from its classification siblings. All three follow from having one
label per TOKEN rather than one per document:

  1. THE LOSS FLATTENS TIME INTO THE BATCH. Logits are [B, L, 9] and labels
     are [B, L], so both are reshaped to [B*L, 9] and [B*L] before
     nn.CrossEntropyLoss sees them. Padded label slots hold
     config.IGNORE_INDEX and are dropped -- the same mechanism the VOC
     segmentation projects used for unlabeled pixels.

  2. THE EPOCH LOSS IS AVERAGED OVER TOKENS, NOT SENTENCES. A 40-token
     sentence contributes 40 predictions and a 3-token sentence contributes 3.
     Weighting batches by sentence count would quietly over-weight batches of
     short sentences, and sentence length on this corpus ranges from 1 to 113.

  3. CHECKPOINT SELECTION IS ON ENTITY F1, NOT ACCURACY. Token accuracy on
     this corpus starts at 83% for a model that has found nothing at all (see
     utils/metrics.py), so selecting on it would be selecting on almost
     nothing. The best checkpoint is the one with the best MICRO entity F1 on
     the dev split.

WHAT IS EVALUATED: the corpus's own development split ("testa"), never the
test split. CoNLL-2003 ships a real dev set, so unlike IMDB there is nothing
to carve out -- but the same discipline applies, and it applies more sharply
here because the splits are TIME-SEPARATED rather than randomly drawn. test is
read only by eval.py, once, at the end.

Two-stage finetuning (same shape as the classification and segmentation
experiments, with the GloVe table playing the pretrained-backbone role):
    Stage 1 - FREEZE the word vectors; the from-scratch case table, encoder
              and head learn to read fixed GloVe features.
    Stage 2 - unfreeze the word table and finetune everything with LAYERED
              learning rates: embedding slowest, encoder middle, head fastest.

NOTE on pretrained vectors: with --no-glove the word vectors start random --
then skip stage 1 (--epochs-stage1 0), because freezing RANDOM embeddings
means training an encoder to read noise. That ablation bites much harder here
than on IMDB: this corpus has 204k training tokens against IMDB's 6M.

Usage:
    python train.py                             # BiLSTM, config.py defaults
    python train.py --model transformer         # Transformer -> outputs_transformer/
    python train.py --download                  # fetch CoNLL-2003 (+ GloVe) first
    python train.py --no-case --output-dir outputs_lstm_nocase   # the ablation
    python train.py --no-glove --epochs-stage1 0
"""

import os
import sys
import json
import time
import random
import ctypes
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from model.lstm_tagger import LSTMTagger
from model_transformer.transformer_tagger import TransformerTagger
from dataset.conll2003 import (
    CoNLLDataset,
    build_vocab_from_train,
    collate_batch,
    conll_present,
    download_conll,
)
from dataset.glove import build_embedding_matrix, download_glove, glove_present
from utils.metrics import TaggingMetrics, evaluate_tagger
from utils.viz import plot_confusion_matrix

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.

    def tqdm(iterable, **kwargs):
        return iterable


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Seed python / numpy / torch RNGs for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN's RNN kernels are the nondeterministic ones here (autotuned
    # algorithms, atomics in the backward pass). Forcing the deterministic
    # path costs some speed and buys reproducible curves.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    """Pick a device: explicit `pref`, else cuda > mps > cpu."""
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_exit(code: int = 0):
    """End the process, working around a cuDNN teardown crash on Windows.

    THE BUG, diagnosed in the IMDB project and unchanged here. On this machine
    (Windows 11, torch 2.11.0+cu128) a process that has used a cuDNN RNN with
    dropout in TRAIN mode dies during shutdown with

        STATUS_STACK_BUFFER_OVERRUN   0xC0000409   exit code -1073740791

    long after main() has returned. The trigger is cuDNN's dropout-state
    descriptor, which this build never releases cleanly. eval.py is unaffected
    because it calls model.eval(), so the descriptor is never created.

    It is NOT a bug in this project -- every checkpoint, log and PNG is already
    written and closed by the time it fires. But the nonzero exit code makes
    any script that chains runs abort after the first model.

    THE FIX. os._exit() is not enough: on Windows it reaches ExitProcess,
    which still runs DLL_PROCESS_DETACH -- and that is where the crash lives.
    TerminateProcess skips DLL detach entirely and is the only thing that
    produced a 0. Everything is flushed first; files are already closed by
    their context managers.

    Only the LSTM path needs this; main() returns the model name so
    __main__ can decide.
    """
    sys.stdout.flush()   # nothing below this point flushes anything
    sys.stderr.flush()
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p   # HANDLE, not int
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)       # non-Windows, or if TerminateProcess somehow returns


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(batch_size, num_workers, download, device, use_glove):
    """Build the train / dev dataloaders and the vocabulary.

    Train = 14,041 sentences, shuffled.
    Dev   =  3,250 sentences ("testa"), in order, larger batches.

    Both use collate_batch, which pads word ids and case ids with PAD_IDX and
    LABELS with IGNORE_INDEX -- see that function for why the two fill values
    must differ.

    Output:
        (train_loader, dev_loader, vocab).
    """
    # Fetch data when asked (--download) OR when anything is missing, so a
    # fresh machine needs no separate step. Both fetches are idempotent, and
    # the GloVe one normally does nothing at all: config.GLOVE_PATH finds the
    # copy the text classification projects already downloaded.
    if download or not conll_present():
        download_conll()
    if use_glove and (download or not glove_present()):
        download_glove()

    vocab = build_vocab_from_train()
    train_set = CoNLLDataset("train", vocab)
    dev_set = CoNLLDataset("valid", vocab)

    pin = device.type == "cuda"  # pinned memory only helps CUDA copies
    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin,
        drop_last=False,
        generator=g,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin,
    )
    return train_loader, dev_loader, vocab


# -----------------------------------------------------------------------------
# Train / evaluate one epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc="",
                    grad_clip=None):
    """Run one training epoch.

    Output:
        float: the TOKEN-weighted average training loss over the epoch.
    """
    model.train()
    total, seen = 0.0, 0
    for ids, cases, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)          # [B, L]
        cases = cases.to(device, non_blocking=True)      # [B, L]
        lengths = lengths.to(device, non_blocking=True)  # [B]
        labels = labels.to(device, non_blocking=True)    # [B, L]

        logits = model(ids, cases, lengths)              # [B, L, K]
        # Flatten time into the batch: CrossEntropyLoss wants [N, K] vs [N].
        # ignore_index drops the padded slots, so N is effectively the number
        # of REAL tokens in this batch.
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        # Rescale large gradients while preserving their direction. Not
        # optional for a recurrent model.
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Weight by REAL TOKENS, not by sentences: criterion already averaged
        # over the real tokens in this batch, and sentence lengths here range
        # from 1 to 113, so a per-batch mean would badly misweight the epoch.
        n_real = int((labels != config.IGNORE_INDEX).sum())
        total += float(loss.detach()) * n_real
        seen += n_real

    return total / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_desc="eval"):
    """One pass over the dev split -> loss AND the full tagging metrics.

    The whole dev split (3,250 sentences) is scored every epoch, so the
    per-epoch number is the real number, not a capped proxy.

    Output:
        dict {"loss", "f1", "macro_f1", "precision", "recall",
              "token_accuracy", "all_o_accuracy", "invalid_rate"}.
    """
    model.eval()
    metrics = TaggingMetrics()
    total, seen = 0.0, 0
    for ids, cases, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)
        cases = cases.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(ids, cases, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        n_real = int((labels != config.IGNORE_INDEX).sum())
        total += float(loss) * n_real
        seen += n_real
        metrics.update(logits.argmax(dim=-1), labels, lengths)

    res = metrics.compute()
    return {"loss": total / max(seen, 1),
            "f1": res["f1"],
            "macro_f1": res["macro_f1"],
            "precision": res["precision"],
            "recall": res["recall"],
            "token_accuracy": res["token_accuracy"],
            "all_o_accuracy": res["all_o_accuracy"],
            "invalid_rate": res["invalid_rate"]}


# -----------------------------------------------------------------------------
# Stage runner
# -----------------------------------------------------------------------------
def run_stage(stage_id, model, train_loader, dev_loader, criterion, optimizer,
              scheduler, epochs, device, history, best, ckpt_dir):
    """Train for `epochs` epochs, logging + checkpointing each one.

    Input:
        stage_id: 1 or 2 (recorded in the history for plotting).
        history: list of per-epoch dict records, appended in place.
        best: dict {"f1": float, "epoch": int} tracking the best model.
        ckpt_dir: directory to save best.pt.

    Output:
        the (possibly updated) `best` dict.
    """
    for e in range(1, epochs + 1):
        # Global epoch number = epochs already recorded + 1.
        global_epoch = len(history) + 1
        t0 = time.time()

        desc = f"[stage {stage_id}] epoch {e}/{epochs}"
        # Read the LRs BEFORE training/stepping so they reflect the LRs
        # actually used this epoch. With layered LRs there are 2-3 param
        # groups, each on its own cosine schedule; log EVERY group.
        group_lrs = {g.get("name", f"group{i}"): g["lr"]
                     for i, g in enumerate(optimizer.param_groups)}

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, desc, grad_clip=config.GRAD_CLIP)
        dev_metrics = evaluate(model, dev_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": global_epoch,
            "stage": stage_id,
            # One field per param group: lr_head / lr_encoder / lr_embedding.
            **{f"lr_{name}": v for name, v in group_lrs.items()},
            "time_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%m-%d %H:%M:%S"),  # wall clock at epoch END
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "f1": dev_metrics["f1"],
            "macro_f1": dev_metrics["macro_f1"],
            "precision": dev_metrics["precision"],
            "recall": dev_metrics["recall"],
            "token_accuracy": dev_metrics["token_accuracy"],
            # Constant across epochs (it is a property of the split, not the
            # model), but recorded per epoch so plot_curves can draw the
            # baseline without reloading the dataset.
            "all_o_accuracy": dev_metrics["all_o_accuracy"],
            "invalid_rate": dev_metrics["invalid_rate"],
        }
        history.append(record)

        lr_str = " ".join(f"{name}={v:.2e}" for name, v in group_lrs.items())
        print(f"[{record['timestamp']}] {desc}  lr[{lr_str}]  "
              f"train_loss={train_loss:.4f}  "
              f"dev_loss={dev_metrics['loss']:.4f}  "
              f"F1={dev_metrics['f1']:.4f}  "
              f"P={dev_metrics['precision']:.4f}  "
              f"R={dev_metrics['recall']:.4f}  "
              f"tokAcc={dev_metrics['token_accuracy']:.4f}  "
              f"({record['time_sec']}s)")

        # Checkpoint on ENTITY F1, not token accuracy. See the module
        # docstring: token accuracy starts at 0.83 for a model that has found
        # nothing, so selecting on it would barely be selecting at all.
        if dev_metrics["f1"] > best["f1"]:
            best["f1"] = dev_metrics["f1"]
            best["epoch"] = global_epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    return best


# -----------------------------------------------------------------------------
# Logging / plotting
# -----------------------------------------------------------------------------
def save_log(history, output_dir, meta=None):
    """Write the run to outputs_*/training_log.json.

    With `meta` (the config + training-param snapshot from collect_run_meta),
    the file is {"meta": {...}, "history": [...per-epoch...]} so a run is
    fully reproducible from its log alone.
    """
    payload = {"meta": meta, "history": history} if meta is not None else history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(payload, f, indent=2)


def collect_run_meta(args, device, train_loader, dev_loader, vocab, *,
                     model_name, model_cfg, extra=None):
    """Snapshot the run's config + training params for training_log.json.

    Records enough to reproduce the run from the log alone: corpus/vocab
    stats, model hyperparams, and the two-stage layered-LR schedule.
    """
    train_set = train_loader.dataset
    dev_set = dev_loader.dataset
    meta = {
        "model": model_name,
        "device": str(device),
        "seed": args.seed,
        "data": {
            "data_root": config.DATA_ROOT,
            "train_sentences": len(train_set),
            "train_tokens": sum(train_set.full_lengths),
            "train_batches": len(train_loader),
            "dev_sentences": len(dev_set),
            "batch_size": args.batch_size,
            "eval_batch_size": config.EVAL_BATCH_SIZE,
            "num_workers": args.num_workers,
            "vocab_size": len(vocab),
            "min_freq": config.MIN_FREQ,
            "max_len": config.MAX_LEN,
            "tags": config.TAGS,
            "train_entities": train_set.entity_counts(),
            "dev_entities": dev_set.entity_counts(),
            "dev_unk_rate": round(dev_set.unk_rate(), 4),
            "train_o_rate": round(train_set.o_rate(), 4),
        },
        "model_cfg": {"model_type": args.model, **model_cfg},
        "optim": {
            "optimizer": "Adam",
            "weight_decay": config.WEIGHT_DECAY,
            "grad_clip": config.GRAD_CLIP,
            "label_smoothing": config.LABEL_SMOOTHING,
            "ignore_index": config.IGNORE_INDEX,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "stage1_lr_head": config.STAGE1_LR_HEAD,
            "stage1_lr_encoder": config.STAGE1_LR_ENCODER,
            "stage2_lr_head": config.STAGE2_LR_HEAD,
            "stage2_lr_encoder": config.STAGE2_LR_ENCODER,
            "stage2_lr_embedding": config.STAGE2_LR_EMBEDDING,
        },
    }
    if extra:
        meta["model_cfg"].update(extra)
    return meta


def summarize_result(result):
    """Compact an evaluate_tagger() dict for the log."""
    return {
        "f1": round(result["f1"], 4),
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "token_accuracy": round(result["token_accuracy"], 4),
        "all_o_accuracy": round(result["all_o_accuracy"], 4),
        "n_gold": result["n_gold"],
        "n_pred": result["n_pred"],
        "invalid_transitions": result["invalid_transitions"],
        "invalid_rate": round(result["invalid_rate"], 6),
        "per_type": {
            t: {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items()}
            for t, row in result["per_type"].items()
        },
        "matrix": result["matrix"],
    }


def plot_curves(history, output_dir, title_prefix="BiLSTM"):
    """Plot training curves from the history and save PNGs to output_dir.

    Produces:
        loss_curve.png - train vs. dev cross-entropy loss per epoch.
        f1_curve.png   - dev entity F1 (micro + macro) and token accuracy,
                         with the all-O baseline drawn in.
    A dashed vertical line marks the stage-1 -> stage-2 boundary.
    """
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    stage2_start = next((r["epoch"] for r in history if r["stage"] == 2), None)

    # ---- Figure 1: cross-entropy loss, train vs dev ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in history], label="train")
    plt.plot(epochs, [r["dev_loss"] for r in history], label="dev")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss (per token)")
    plt.title(f"{title_prefix} loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ---- Figure 2: entity F1 vs the token-accuracy mirage ----
    # Both are plotted on one axis on purpose. The gap between the token
    # accuracy line and the all-O baseline is tiny while the F1 line moves
    # across most of the chart -- which is the argument for the metric,
    # drawn rather than asserted.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["f1"] for r in history], label="dev entity F1 (micro)")
    plt.plot(epochs, [r["macro_f1"] for r in history], label="dev entity F1 (macro)")
    plt.plot(epochs, [r["token_accuracy"] for r in history],
             label="dev token accuracy", ls=":", color="gray")
    all_o = history[0].get("all_o_accuracy")
    if all_o:
        plt.axhline(all_o, color="red", ls="--", lw=1,
                    label=f"all-O token accuracy ({all_o:.3f})")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.ylim(0.0, 1.0)
    plt.title(f"{title_prefix} dev entity F1 vs token accuracy")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Optimizer builder (layered learning rates)
# -----------------------------------------------------------------------------
def build_layered_optimizer(model, lr_head, lr_encoder, lr_embedding, weight_decay):
    """Build an Adam optimizer with up to three layered param groups.

    The split comes from model.parameter_groups() (frozen params excluded):

        head      -> lr_head      (from-scratch classifier, fastest)
        encoder   -> lr_encoder   (from-scratch encoder, same tier in stage 1)
        embedding -> lr_embedding (pretrained GloVe + the case table, slowest)

    Input:
        lr_*: learning rate per tier. Pass None to skip a tier.
        weight_decay: shared across all groups.

    Output:
        torch.optim.Adam with one param_group per ACTIVE tier, head first.

    NOTE for stage 1: unlike the classification projects, the "embedding" tier
    is NOT empty when the word table is frozen -- the randomly-initialized
    case table lives there and still needs training. Stage 1 therefore passes
    the ENCODER learning rate for that tier rather than None, because those
    parameters are from scratch exactly like the encoder's.
    """
    groups = model.parameter_groups()
    param_groups = []
    for name, lr in (("head", lr_head),
                     ("encoder", lr_encoder),
                     ("embedding", lr_embedding)):
        params = groups[name]
        if lr is None or not params:
            continue
        param_groups.append({"params": params, "lr": lr, "name": name})
    return optim.Adam(param_groups, weight_decay=weight_decay)


def count_trainable(model):
    """Return the number of trainable parameters (in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train an NER tagger on CoNLL-2003")
    p.add_argument("--model", choices=["lstm", "transformer"], default=config.MODEL,
                   help="encoder family. Each writes to its own outputs_<model>/")
    p.add_argument("--download", action="store_true",
                   help="download CoNLL-2003 (and GloVe) before training")
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="training seed. The train/dev/test SPLITS are the "
                        "corpus's own and do not move, so different seeds are "
                        "scored on identical data")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS,
                   help="stage-1 epochs (frozen word vectors); 0 skips stage 1 "
                        "(use 0 together with --no-glove)")
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS,
                   help="stage-2 epochs (word vectors unfrozen); 0 skips stage 2")
    p.add_argument("--hidden-size", type=int, default=config.HIDDEN_SIZE,
                   help="BiLSTM hidden width PER DIRECTION. 100 with --layers 1 "
                        "reproduces the Lample et al. (2016) shape")
    p.add_argument("--dim", type=int, default=config.TRANSFORMER_DIM,
                   help="Transformer width; must be divisible by --group")
    p.add_argument("--group", type=int, default=config.TRANSFORMER_GROUP,
                   help="number of Transformer attention heads")
    p.add_argument("--norm", choices=["pre", "post"],
                   default=config.TRANSFORMER_NORM,
                   help="Transformer LayerNorm placement. pre = x+f(LN(x)), "
                        "the default here; post = LN(x+f(x)), the 2017 "
                        "original, which needs warmup once it gets deep. "
                        "Ignored by --model lstm")
    p.add_argument("--layers", type=int, default=None,
                   help="encoder layers (default: model-specific config.py value)")
    p.add_argument("--dropout", type=float, default=None,
                   help="dropout probability (default: model-specific config.py value)")
    p.add_argument("--no-glove", action="store_true",
                   help="train word vectors from scratch (no GloVe init)")
    p.add_argument("--no-case", action="store_true",
                   help="drop the capitalization feature -- the ablation that "
                        "measures what lowercasing costs on NER")
    p.add_argument("--output-dir", default=None,
                   help="output folder name/path. Default: outputs_<model>/. A "
                        "bare name is placed under the project root; an "
                        "absolute path is used as-is.")
    args = p.parse_args()
    is_transformer = args.model == "transformer"
    if args.layers is None:
        args.layers = config.TRANSFORMER_LAYERS if is_transformer else config.NUM_LAYERS
    if args.dropout is None:
        args.dropout = config.TRANSFORMER_DROPOUT if is_transformer else config.DROPOUT
    if args.layers <= 0 or not 0 <= args.dropout <= 1:
        p.error("--layers must be positive and --dropout must be in [0, 1]")
    if args.hidden_size <= 0:
        p.error("--hidden-size must be positive")
    if is_transformer and (args.dim <= 0 or args.group <= 0 or args.dim % args.group):
        p.error("--dim and --group must be positive, and --group must divide --dim")
    if min(args.epochs_stage1, args.epochs_stage2) < 0 or not (
            args.epochs_stage1 + args.epochs_stage2):
        p.error("stage epochs must be nonnegative, with at least one epoch in total")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    use_glove = config.USE_GLOVE and not args.no_glove
    use_case = config.USE_CASE_FEATURE and not args.no_case
    is_transformer = args.model == "transformer"
    model_name = "Transformer" if is_transformer else "BiLSTM"

    # Separate model outputs; --output-dir overrides for additional experiments.
    if args.output_dir:
        output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                      else os.path.join(config.PROJECT_ROOT, args.output_dir))
    else:
        output_dir = config.output_dir_for_model(args.model)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}  seed: {args.seed}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Output dir: {output_dir}  ({model_name}, "
          f"{'GloVe' if use_glove else 'scratch'} embeddings, "
          f"{'with' if use_case else 'NO'} case feature)")

    # ---- Data ----
    train_loader, dev_loader, vocab = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device, use_glove)
    train_set, dev_set = train_loader.dataset, dev_loader.dataset
    print(f"Train sentences: {len(train_set)} ({len(train_loader)} batches, "
          f"{sum(train_set.full_lengths)} tokens)  "
          f"Dev sentences: {len(dev_set)}  Vocab: {len(vocab)}")
    print(f"Train entities: {train_set.entity_counts()}")
    print(f"Class balance: {train_set.o_rate():.2%} of training tokens are 'O' "
          f"-- that is what an all-O model would score on token accuracy")
    # Save the vocabulary NEXT TO THE CHECKPOINT. The build is deterministic,
    # so eval.py could rebuild it -- but a checkpoint whose id mapping lives
    # only in a rebuild step is a checkpoint waiting to be silently misread.
    vocab.save(os.path.join(output_dir, "vocab.json"))

    # ---- Model + loss ----
    vectors = None
    if use_glove:
        vectors, n_found = build_embedding_matrix(vocab)
    model_cfg = dict(
        num_tags=config.NUM_TAGS,
        embed_dim=config.EMBED_DIM,
        num_layers=args.layers,
        dropout=args.dropout,
        pad_idx=config.PAD_IDX,
        use_case=use_case,
        case_dim=config.CASE_DIM,
    )
    if is_transformer:
        model_cfg.update(dim=args.dim, group=args.group, max_len=config.MAX_LEN,
                         norm=args.norm)
        model_class = TransformerTagger
    else:
        model_cfg.update(hidden_size=args.hidden_size,
                         bidirectional=config.BIDIRECTIONAL)
        model_class = LSTMTagger
    model = model_class(vocab_size=len(vocab), pretrained_vectors=vectors,
                        **model_cfg).to(device)
    print(f"Model config: {model_cfg}")

    # Plain cross-entropy over flattened (batch x time) positions.
    # ignore_index drops the padded label slots -- the same mechanism the VOC
    # segmentation projects used for unlabeled pixels, and the reason
    # collate_batch fills labels with -100 rather than 0.
    #
    # No class weighting, despite 83% of tokens being "O". Weighting would
    # trade precision for recall on the entity classes, which moves F1 in an
    # unpredictable direction and adds a hyperparameter this project does not
    # need; the encoder handles the prior perfectly well. Label smoothing is
    # off for a related reason -- see config.LABEL_SMOOTHING.
    criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX,
                                    label_smoothing=config.LABEL_SMOOTHING)

    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"(embedding {len(vocab) * config.EMBED_DIM / 1e6:.2f}M)")

    # From-scratch guard: stage 1 freezes the word table, which only makes
    # sense when it is pretrained. Warn (don't override) so the run stays
    # reproducible from the command line alone.
    if not model.embedding.pretrained_loaded and args.epochs_stage1 > 0:
        print("[conll] WARNING: no GloVe vectors were loaded but stage 1 will "
              "freeze the (random) word table. Consider --epochs-stage1 0 for "
              "from-scratch training.")

    history = []
    best = {"f1": -1.0, "epoch": -1}
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    extra = {"glove": use_glove,
             "glove_path": config.GLOVE_PATH if use_glove else None}
    if use_glove:
        extra["glove_coverage"] = round(n_found / len(vocab), 4)
    meta = collect_run_meta(args, device, train_loader, dev_loader, vocab,
                            model_name=model_name, model_cfg=model_cfg, extra=extra)
    meta["started"] = started
    # Save the architecture before training so interrupted runs remain loadable.
    save_log(history, output_dir, meta)

    # ---- Stage 1: freeze the word table, train case table + encoder + head ----
    if args.epochs_stage1 > 0:
        print("\n=== Stage 1: freeze word vectors, train case+encoder+head ===")
        model.freeze_embedding()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE1_LR_HEAD,
            lr_encoder=config.STAGE1_LR_ENCODER,
            # NOT None: the case table is in this tier and is from scratch, so
            # it gets the same rate as the encoder. The word table is frozen
            # and excluded by parameter_groups() regardless.
            lr_embedding=config.STAGE1_LR_ENCODER,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
        best = run_stage(1, model, train_loader, dev_loader, criterion, optimizer,
                         scheduler, args.epochs_stage1, device, history, best, output_dir)

    # ---- Stage 2: unfreeze the word table, layered-LR finetune ----
    if args.epochs_stage2 <= 0:
        print("\n=== Stage 2 skipped (--epochs-stage2 0) ===")
    else:
        print("\n=== Stage 2: unfreeze word vectors, layered-LR finetune ===")
        model.unfreeze_all()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE2_LR_HEAD,
            lr_encoder=config.STAGE2_LR_ENCODER,
            lr_embedding=config.STAGE2_LR_EMBEDDING,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
        best = run_stage(2, model, train_loader, dev_loader, criterion, optimizer,
                         scheduler, args.epochs_stage2, device, history, best, output_dir)

    # ---- Save logs + curves ----
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["best_dev"] = {"f1": round(best["f1"], 4), "epoch": best["epoch"]}
    save_log(history, output_dir, meta)
    plot_curves(history, output_dir, title_prefix=model_name)
    print(f"\nDone. Best DEV entity F1={best['f1']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {output_dir}")

    # ---- Final report on the best checkpoint (per-type + confusion matrix) --
    # Still on DEV, not test: this run must not look at the test split at all.
    # Run `python eval.py --split test` afterwards for the reported number.
    best_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("\nBest checkpoint on the CoNLL-2003 dev split:")
        result = evaluate_tagger(model, dev_loader, device)  # verbose report
        plot_confusion_matrix(
            result["matrix"], config.TAGS,
            os.path.join(output_dir, "confusion_matrix.png"),
            title=f"{model_name} dev token confusion matrix")
        meta["final_dev"] = summarize_result(result)
        save_log(history, output_dir, meta)
        print("\nNow score the held-out test split:")
        print(f'  python eval.py --weights "{best_path}" '
              f"--split test --save-cm")
    # Notify only after training and the final report finish successfully.
    try:
        if os.name == "nt":
            import winsound

            winsound.Beep(1000, 500)
        else:
            print("\a", end="", flush=True)
    except RuntimeError:
        pass  # An unavailable audio device must not fail a completed run.
    return args.model


if __name__ == "__main__":
    if main() == "lstm":
        # Only recurrent training needs the cuDNN teardown workaround.
        clean_exit()
