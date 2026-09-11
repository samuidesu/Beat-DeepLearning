"""Finetune pretrained BERT for NER on CoNLL-2003 -- the third model.

    python train_bert.py                        # bert-base-cased -> outputs_bert/
    python train_bert.py --lr 5e-5 --output-dir outputs_bert_lr5e-5
    python train_bert.py --bert google-bert/bert-base-uncased --output-dir outputs_bert_uncased

WHY NOT `train.py --model bert`. train.py can call its comparison
single-variable because its two models share everything except the encoder:
vocabulary, GloVe table, case feature, the two-stage schedule. BERT shares none
of that, and folding it in would put a branch at almost every step. What it
does share is IMPORTED here rather than copied, and the imported half is
exactly the half that decides whether the numbers are comparable:

    imported from the GloVe pipeline           specific to BERT (this file)
    ---------------------------------------    ---------------------------------
    read_split: same sentences, same BIO2      WordPiece encoding (model_bert/)
    evaluate(): same loss, same dev metrics    AdamW, decay on weight matrices only
    TaggingMetrics: same entity F1             warmup + linear decay, PER UPDATE
    best.pt selected on dev entity F1          bf16 autocast
    training_log.json and curve format         one stage, everything trainable

THE PROTOCOL is the finetuning recipe of Devlin et al. (2019); every number
lives in config.py:
  - every parameter trains from the first update, at ONE learning rate -- only
    the 6,921-parameter head is random, so there is no frozen stage to protect;
  - the rate ramps up from 0 over the first 10% of updates, then decays
    linearly to 0 at the last one;
  - the schedule steps after every UPDATE, not every epoch: 4 epochs are 1,756
    updates, and a per-epoch step would leave warmup no resolution at all;
  - gradients are clipped at global norm 1.0.

WHAT IS EVALUATED: the dev split, every epoch, through train.py's evaluate()
itself. The test split is not read here.

REQUIREMENTS: `pip install transformers`. The first run downloads the
checkpoint (about 430 MB for bert-base) into the Hugging Face cache; later runs
read it from there. Set HF_ENDPOINT to download through a mirror.
"""

import os
import time
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset.conll2003 import conll_present, download_conll
from model_bert.bert_tagger import BertTagger
from model_bert.wordpiece import BertCoNLLDataset, collate_bert_batch, load_tokenizer
# The shared half (see the module docstring): dev evaluation, log format and
# curves are train.py's own functions, not copies of them.
from train import (evaluate, get_device, plot_curves, save_log, set_seed,
                   summarize_result, tqdm)
from utils.metrics import evaluate_tagger
from utils.viz import plot_confusion_matrix


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(tokenizer, batch_size, num_workers, device):
    """Train (shuffled) and dev (in order) loaders over WordPiece-encoded CoNLL.

    Output:
        (train_loader, dev_loader).
    """
    if not conll_present():
        download_conll()
    train_set = BertCoNLLDataset("train", tokenizer)
    dev_set = BertCoNLLDataset("valid", tokenizer)

    # The pad id comes from the tokenizer, and BertTagger is given the same
    # one: its attention mask is derived from exactly these fill positions.
    collate = partial(collate_bert_batch, pad_id=tokenizer.pad_token_id)
    pin = device.type == "cuda"
    g = torch.Generator()
    g.manual_seed(config.SEED)  # same shuffle seeding as train.py

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate,
                              pin_memory=pin, generator=g)
    dev_loader = DataLoader(dev_set, batch_size=config.EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=collate, pin_memory=pin)
    return train_loader, dev_loader


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

    Same loss and token weighting as train.py's train_one_epoch. What differs
    is finetuning convention: the scheduler steps every UPDATE, the forward
    pass runs under bf16 autocast, and clipping is always on.

    Output:
        float: the TOKEN-weighted average training loss over the epoch.
    """
    model.train()
    total, seen = 0.0, 0
    for input_ids, word_index, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        input_ids = input_ids.to(device, non_blocking=True)    # [B, T] pieces
        word_index = word_index.to(device, non_blocking=True)  # [B, W] words
        labels = labels.to(device, non_blocking=True)          # [B, W]

        # bf16 needs no GradScaler: unlike fp16 it keeps float32's exponent
        # range, so small gradients do not underflow. The loss sits inside the
        # block so autocast computes cross-entropy in float32.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids, word_index, lengths)     # [B, W, K]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.BERT_GRAD_CLIP)
        optimizer.step()
        scheduler.step()  # per UPDATE -- see the module docstring

        n_real = int((labels != config.IGNORE_INDEX).sum())
        total += float(loss.detach()) * n_real
        seen += n_real

    return total / max(seen, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Finetune BERT for NER on CoNLL-2003")
    p.add_argument("--bert", default=config.BERT_NAME,
                   help="checkpoint: hub id or local directory. The tokenizer "
                        "is loaded from the same name")
    p.add_argument("--epochs", type=int, default=config.BERT_EPOCHS)
    p.add_argument("--lr", type=float, default=config.BERT_LR,
                   help="peak learning rate, reached at the end of warmup")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
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
    print(f"Checkpoint: {args.bert}")
    print(f"Output dir: {output_dir}")

    # ---- Data ----
    tokenizer = load_tokenizer(args.bert)
    train_loader, dev_loader = build_dataloaders(
        tokenizer, args.batch_size, args.num_workers, device)
    train_set, dev_set = train_loader.dataset, dev_loader.dataset
    n_words = sum(len(index) for index in train_set.word_index)
    n_pieces = sum(len(ids) - 2 for ids in train_set.input_ids)
    longest = max(len(ids) for d in (train_set, dev_set) for ids in d.input_ids)
    dev_unk = dev_set.unk_rate()
    print(f"Train sentences: {len(train_set)} ({len(train_loader)} batches, "
          f"{n_words} words -> {n_pieces} WordPiece tokens, "
          f"{n_pieces / n_words:.2f} per word)  Dev sentences: {len(dev_set)}")
    print(f"Longest sentence: {longest} pieces incl. [CLS]/[SEP] "
          f"(limit {config.BERT_MAX_LEN})  Dev [UNK] word rate: {dev_unk:.4f}")

    # ---- Model, loss, optimizer, schedule ----
    model = BertTagger(args.bert, num_tags=config.NUM_TAGS,
                       dropout=config.BERT_DROPOUT,
                       pad_id=tokenizer.pad_token_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M, all trainable "
          f"(head {sum(p.numel() for p in model.head.parameters()):,})")

    # The same criterion as train.py: plain cross-entropy over flattened word
    # positions, padded word slots dropped by ignore_index.
    criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX,
                                    label_smoothing=config.LABEL_SMOOTHING)
    optimizer = build_optimizer(model, args.lr, config.BERT_WEIGHT_DECAY)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(config.BERT_WARMUP_RATIO * total_steps)
    scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)
    print(f"Schedule: {total_steps} updates, {warmup_steps} of them warmup, "
          f"peak lr {args.lr:.1e}")

    history = []
    best = {"f1": -1.0, "epoch": -1}
    best_path = os.path.join(output_dir, "best.pt")
    meta = {
        "model": "BERT",
        "device": str(device),
        "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "data_root": config.DATA_ROOT,
            "train_sentences": len(train_set),
            "train_tokens": n_words,
            "train_pieces": n_pieces,
            "train_batches": len(train_loader),
            "dev_sentences": len(dev_set),
            "batch_size": args.batch_size,
            "eval_batch_size": config.EVAL_BATCH_SIZE,
            "num_workers": args.num_workers,
            "max_len": config.BERT_MAX_LEN,
            "longest_pieces": longest,
            "tags": config.TAGS,
            "dev_unk_rate": round(dev_unk, 4),
        },
        "model_cfg": {
            "model_type": "bert",
            "bert": args.bert,
            "num_tags": config.NUM_TAGS,
            "dropout": config.BERT_DROPOUT,
            "pad_id": tokenizer.pad_token_id,
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
            "ignore_index": config.IGNORE_INDEX,
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
        # train.py's evaluate(), in float32: the dev number does not depend on
        # the training precision.
        dev = evaluate(model, dev_loader, criterion, device)

        record = {
            "epoch": epoch,
            "stage": 1,  # a single stage; plot_curves reads this field
            "lr": optimizer.param_groups[0]["lr"],  # at the END of the epoch
            "time_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%m-%d %H:%M:%S"),
            "train_loss": train_loss,
            "dev_loss": dev["loss"],
            **{k: dev[k] for k in ("f1", "macro_f1", "precision", "recall",
                                   "token_accuracy", "all_o_accuracy",
                                   "invalid_rate")},
        }
        history.append(record)
        print(f"[{record['timestamp']}] {desc}  lr_end={record['lr']:.2e}  "
              f"train_loss={train_loss:.4f}  dev_loss={dev['loss']:.4f}  "
              f"F1={dev['f1']:.4f}  P={dev['precision']:.4f}  "
              f"R={dev['recall']:.4f}  tokAcc={dev['token_accuracy']:.4f}  "
              f"({record['time_sec']}s)")

        if dev["f1"] > best["f1"]:
            best = {"f1": dev["f1"], "epoch": epoch}
            torch.save(model.state_dict(), best_path)

    # ---- Save logs + curves ----
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["best_dev"] = {"f1": round(best["f1"], 4), "epoch": best["epoch"]}
    save_log(history, output_dir, meta)
    plot_curves(history, output_dir, title_prefix="BERT")
    print(f"\nDone. Best DEV entity F1={best['f1']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {output_dir}")

    # ---- Final report on the best checkpoint, still on DEV ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    print("\nBest checkpoint on the CoNLL-2003 dev split:")
    result = evaluate_tagger(model, dev_loader, device)  # verbose report
    plot_confusion_matrix(
        result["matrix"], config.TAGS,
        os.path.join(output_dir, "confusion_matrix.png"),
        title="BERT dev token confusion matrix")
    meta["final_dev"] = summarize_result(result)
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
