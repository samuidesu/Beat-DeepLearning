"""Finetune pretrained BERT for IMDB sentiment classification.

    python train_bert.py                                   # bert-base-uncased -> outputs_bert/
    python train_bert.py --truncation head --output-dir outputs_bert_head
    python train_bert.py --pooling mean --output-dir outputs_bert_mean

WHY NOT `train.py --model bert`. train.py can call rnn / lstm / gru /
transformer a controlled comparison because those models share everything but
the encoder: vocabulary, GloVe table, the MAX_LEN window, the two-stage
schedule. BERT shares none of that, and folding it in would put a branch at
almost every step. What it does share is IMPORTED here rather than copied, and
the imported half is the half that decides whether the numbers are comparable:

    imported from the GloVe pipeline            specific to BERT (this file)
    ------------------------------------------  --------------------------------
    read_split: same 22,500 / 2,500 reviews     WordPiece encoding (model_bert/)
    truncate(): same head / head_tail cut       510-piece window, 128 + 382
    collate_batch: same padded batches          AdamW, decay on weight matrices only
    evaluate(): same loss, same val metrics     warmup + linear decay, PER UPDATE
    best.pt selected on val accuracy            bf16 autocast
    training_log.json and curve format          one stage, everything trainable

THE PROTOCOL is Devlin et al.'s finetuning recipe, plus Sun et al.'s head+tail
cut for reviews longer than the window; every number lives in config.py:
  - every parameter trains from the first update, at ONE learning rate -- only
    the 1,538-parameter classifier is random, so there is no frozen stage;
  - the rate ramps up from 0 over the first 10% of updates, then decays
    linearly to 0 at the last one;
  - the schedule steps after every UPDATE, not every epoch: 4 epochs are 2,816
    updates, and a per-epoch step would leave warmup no resolution at all;
  - gradients are clipped at global norm 1.0.

The loss keeps config.LABEL_SMOOTHING, the GloVe models' value, so this run's
loss curves sit on the same scale as theirs.

WHAT IS EVALUATED: the 2,500-review validation split, every epoch, through
train.py's evaluate() itself. test.csv is not read here.

REQUIREMENTS: `pip install transformers`. The first run downloads the
checkpoint (about 440 MB for bert-base) into the Hugging Face cache; later runs
read it from there. Set HF_ENDPOINT to download through a mirror.
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from dataset.imdb import collate_batch, download_imdb, imdb_present
from model_bert.bert_classifier import BertClassifier
from model_bert.wordpiece import BertIMDBDataset
# The shared half (see the module docstring): val evaluation, log format and
# curves are train.py's own functions, not copies of them.
from train import (evaluate, get_device, plot_curves, save_log, set_seed,
                   summarize_result, tqdm)
from utils.metrics import compute_accuracy
from utils.viz import plot_confusion_matrix


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(tokenizer, batch_size, num_workers, device, truncation,
                      head_len):
    """Train (shuffled) and val (in order) loaders over WordPiece-encoded IMDB.

    Both use dataset/imdb.py's collate_batch unchanged. Val runs at the
    TRAINING batch size rather than EVAL_BATCH_SIZE: that 128 was sized for
    the GloVe models, and a 512-position BERT row is a different budget even
    without a backward pass.

    Output:
        (train_loader, val_loader).
    """
    if not imdb_present():
        download_imdb()
    train_set = BertIMDBDataset("train", tokenizer, truncation=truncation,
                                head_len=head_len)
    val_set = BertIMDBDataset("val", tokenizer, truncation=truncation,
                              head_len=head_len)

    pin = device.type == "cuda"
    g = torch.Generator()
    g.manual_seed(config.SEED)  # same shuffle seeding as train.py

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_batch,
                              pin_memory=pin, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_batch,
                            pin_memory=pin)
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Optimizer + schedule
# -----------------------------------------------------------------------------
def build_optimizer(model, lr, weight_decay):
    """AdamW, with weight decay on weight MATRICES only.

    Every 1-D parameter in this model is a bias or a LayerNorm gain/shift --
    exactly the set the original BERT optimizer exempted from decay by name --
    so `ndim < 2` reproduces that exemption without matching names.
    """
    params = list(model.parameters())
    return optim.AdamW(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=lr)


def linear_warmup_decay(optimizer, warmup_steps, total_steps):
    """LR multiplier 0 -> 1 over `warmup_steps` updates, then 1 -> 0 at `total_steps`.

    The same curve as transformers.get_linear_schedule_with_warmup, written
    out so the schedule can be read here rather than imported.
    """
    def factor(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return optim.lr_scheduler.LambdaLR(optimizer, factor)


# -----------------------------------------------------------------------------
# Train one epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    use_amp, epoch_desc=""):
    """Run one finetuning epoch.

    Same loss and sample weighting as train.py's train_one_epoch. What differs
    is finetuning convention: the scheduler steps every UPDATE, the forward
    pass runs under bf16 autocast, and clipping is always on.

    Output:
        float: the sample-weighted average training loss over the epoch.
    """
    model.train()
    total, seen = 0.0, 0
    for ids, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)          # [B, T] pieces
        lengths = lengths.to(device, non_blocking=True)  # [B]
        labels = labels.to(device, non_blocking=True)    # [B]

        # bf16 needs no GradScaler: unlike fp16 it keeps float32's exponent
        # range, so small gradients do not underflow. The loss sits inside the
        # block so autocast computes cross-entropy in float32.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids, lengths)                 # [B, 2]
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.BERT_GRAD_CLIP)
        optimizer.step()
        scheduler.step()  # per UPDATE -- see the module docstring

        bs = ids.size(0)
        total += float(loss.detach()) * bs
        seen += bs

    return total / max(seen, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Finetune BERT for IMDB sentiment")
    p.add_argument("--bert", default=config.BERT_NAME,
                   help="checkpoint: hub id or local directory. The tokenizer "
                        "is loaded from the same name")
    p.add_argument("--epochs", type=int, default=config.BERT_EPOCHS)
    p.add_argument("--lr", type=float, default=config.BERT_LR,
                   help="peak learning rate, reached at the end of warmup")
    p.add_argument("--batch-size", type=int, default=config.BERT_BATCH_SIZE)
    p.add_argument("--truncation", choices=["head", "head_tail"],
                   default=config.TRUNCATION,
                   help="which end(s) of a review longer than the window to keep")
    p.add_argument("--head-len", type=int, default=config.BERT_HEAD_LEN,
                   help="pieces kept from the front under --truncation "
                        "head_tail; the remaining BERT_MAX_LEN - 2 - head_len "
                        "come from the end")
    p.add_argument("--pooling", choices=["last", "max", "mean"],
                   default=config.BERT_POOLING,
                   help="last = BERT's pooled [CLS]; mean / max = masked "
                        "pooling over the token outputs")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--no-amp", action="store_true",
                   help="train in float32 instead of bf16 autocast")
    p.add_argument("--output-dir", default=None,
                   help="output folder name/path. Default: outputs_bert/. A bare "
                        "name is placed under the project root; an absolute "
                        "path is used as-is.")
    args = p.parse_args()
    if args.epochs <= 0 or args.lr <= 0 or args.batch_size <= 0:
        p.error("--epochs, --lr and --batch-size must be positive")
    if not 0 <= args.head_len <= config.BERT_MAX_LEN - 2:
        p.error(f"--head-len must be in [0, BERT_MAX_LEN - 2 = {config.BERT_MAX_LEN - 2}]")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = (not args.no_amp and device.type == "cuda"
               and torch.cuda.is_bf16_supported())

    if args.output_dir:
        output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                      else os.path.join(config.PROJECT_ROOT, args.output_dir))
    else:
        output_dir = config.OUTPUT_DIR_BERT
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}  seed: {args.seed}  "
          f"precision: {'bf16 autocast' if use_amp else 'float32'}")
    print(f"Checkpoint: {args.bert}  ({args.pooling} pooling)")
    print(f"Output dir: {output_dir}")

    # ---- Data ----
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    train_loader, val_loader = build_dataloaders(
        tokenizer, args.batch_size, args.num_workers, device,
        args.truncation, args.head_len)
    train_set, val_set = train_loader.dataset, val_loader.dataset
    window = config.BERT_MAX_LEN - 2
    cut = (f"head {args.head_len} + tail {window - args.head_len}"
           if args.truncation == "head_tail" else f"first {window}")
    val_unk = val_set.unk_rate()
    print(f"Train docs: {len(train_set)}  ({len(train_loader)} batches)  "
          f"Val docs: {len(val_set)}")
    print(f"Cut at {window} pieces + [CLS]/[SEP] ({cut}): "
          f"{train_set.truncated_rate():.2%} of training reviews  "
          f"Val [UNK] rate: {val_unk:.4f}")

    # ---- Model, loss, optimizer, schedule ----
    model = BertClassifier(args.bert, num_classes=config.NUM_CLASSES,
                           pooling=args.pooling,
                           dropout=config.BERT_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M, all trainable "
          f"(classifier {sum(p.numel() for p in model.head.parameters()):,})")

    # The same criterion as train.py, label smoothing included.
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = build_optimizer(model, args.lr, config.BERT_WEIGHT_DECAY)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(config.BERT_WARMUP_RATIO * total_steps)
    scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)
    print(f"Schedule: {total_steps} updates, {warmup_steps} of them warmup, "
          f"peak lr {args.lr:.1e}")

    history = []
    best = {"accuracy": -1.0, "epoch": -1}
    best_path = os.path.join(output_dir, "best.pt")
    meta = {
        "model": f"BERT ({args.pooling} pooling)",
        "device": str(device),
        "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "data_root": config.DATA_ROOT,
            "train_docs": len(train_set),
            "train_batches": len(train_loader),
            "val_docs": len(val_set),
            "val_ratio": config.VAL_RATIO,
            "split_seed": config.SPLIT_SEED,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_len": config.BERT_MAX_LEN,
            "truncation": args.truncation,
            "head_len": args.head_len,
            "train_truncated_rate": round(train_set.truncated_rate(), 4),
            "val_unk_rate": round(val_unk, 4),
        },
        "model_cfg": {
            "model_type": "bert",
            "bert": args.bert,
            "num_classes": config.NUM_CLASSES,
            "pooling": args.pooling,
            "dropout": config.BERT_DROPOUT,
            "params": n_params,
        },
        "optim": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": config.BERT_WEIGHT_DECAY,
            "schedule": "linear warmup + linear decay, stepped per update",
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "epochs": args.epochs,
            "grad_clip": config.BERT_GRAD_CLIP,
            "label_smoothing": config.LABEL_SMOOTHING,
            "precision": "bf16" if use_amp else "float32",
        },
    }
    save_log(history, output_dir, meta)

    # ---- Finetune ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        desc = f"epoch {epoch}/{args.epochs}"
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler, device, use_amp, desc)
        # train.py's evaluate(), in float32: the val number does not depend on
        # the training precision.
        val = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "stage": 1,  # a single stage; plot_curves reads this field
            "lr": optimizer.param_groups[0]["lr"],  # at the END of the epoch
            "time_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%m-%d %H:%M:%S"),
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "accuracy": val["accuracy"],
            "macro_f1": val["macro_f1"],
        }
        history.append(record)
        print(f"[{record['timestamp']}] {desc}  lr_end={record['lr']:.2e}  "
              f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"acc={val['accuracy']:.4f}  macroF1={val['macro_f1']:.4f}  "
              f"({record['time_sec']}s)")

        if val["accuracy"] > best["accuracy"]:
            best = {"accuracy": val["accuracy"], "epoch": epoch}
            torch.save(model.state_dict(), best_path)

    # ---- Save logs + curves ----
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["best_val"] = {"accuracy": round(best["accuracy"], 4), "epoch": best["epoch"]}
    save_log(history, output_dir, meta)
    plot_curves(history, output_dir, title_prefix="BERT")
    print(f"\nDone. Best VAL accuracy={best['accuracy']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {output_dir}")

    # ---- Final report on the best checkpoint, still on VAL ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    print("\nBest checkpoint on the IMDB validation split:")
    result = compute_accuracy(model, val_loader, device)  # verbose report
    plot_confusion_matrix(
        result["matrix"], config.CLASS_NAMES,
        os.path.join(output_dir, "confusion_matrix.png"),
        title="BERT val confusion matrix")
    meta["final_val"] = summarize_result(result)
    save_log(history, output_dir, meta)

    # Notify only after training and the final report finish successfully.
    try:
        if os.name == "nt":
            import winsound

            winsound.Beep(1000, 500)
        else:
            print("\a", end="", flush=True)
    except RuntimeError:
        pass  # An unavailable audio device must not fail a completed run.


if __name__ == "__main__":
    main()
