"""Training entry point for FCN semantic segmentation on PASCAL VOC 2012.

Two-stage finetuning (identical protocol to the YOLO3 / FCOS projects, so the
"unfreeze the backbone" lesson transfers directly):
    Stage 1 - freeze the whole backbone, train only the neck + head.
    Stage 2 - unfreeze the backbone (config.STAGE2_UNFREEZE, default "all")
              and finetune it with a small LR while continuing to train the
              neck + head.

This file wires up the full pipeline: data -> model -> loss -> optimize, plus
evaluation (val loss + mIoU), checkpointing, logging (training_log.json) and
curve plotting. The skeleton is deliberately line-for-line parallel to the
FCOS train.py -- diff the two files to see exactly what changes when the task
switches from detection to segmentation (spoiler: the loss call and the
metric; everything else is identical machinery).

Usage:
    python train.py                 # train with config.py defaults
    python train.py --download      # download VOC2012 first, then train
    python train.py --batch-size 8 --device cpu
"""

import os
import json
import time
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from model.unet import UNet
from losses.fcn_loss import FCNLoss
from dataset.voc import (VOCSegDataset, build_train_dataset, download_voc,
                         voc_seg_present, sbd_present)
from utils.metrics import ConfusionMatrix, compute_miou

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
    # Seeding alone doesn't make CUDA runs repeatable: benchmark mode lets
    # cuDNN autotune conv algorithms (picking different, sometimes
    # nondeterministic kernels per run). Force the deterministic ones, at
    # some training-speed cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed each DataLoader worker so augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(pref: str = "auto") -> torch.device:
    """Pick a device: explicit `pref`, else cuda > mps > cpu."""
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(batch_size: int, num_workers: int, download: bool, device: torch.device):
    """Build the train / val dataloaders.

    Train = VOC2012 seg train (1464 images; + SBD if config.USE_SBD), random
            480x480 crops -> default collate stacks them (NO custom
            collate_fn, unlike detection: fixed-size tensors batch natively).
    Val   = VOC2012 seg val (1449 images) at ORIGINAL resolution, padded to
            /32 -- sizes differ per image, hence batch_size=1.

    Output:
        (train_loader, val_loader).
    """
    # Fetch data when asked (--download) OR when something is missing. VOC and
    # SBD are checked SEPARATELY: this machine may already have VOC (reused
    # from the FCN / detection projects) while SBD was never downloaded --
    # download_voc() skips whatever is present and fetches only the rest.
    if download or not voc_seg_present() or (config.USE_SBD and not sbd_present()):
        download_voc()

    train_set = build_train_dataset()
    val_set = VOCSegDataset(image_set="val", train=False)

    # pin_memory only helps when copying to a CUDA device.
    pin = device.type == "cuda"
    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Train / evaluate one epoch
# -----------------------------------------------------------------------------
def _accumulate(running: dict, items: dict, n: int):
    """Add this batch's loss components (weighted by batch size) into `running`."""
    for k, v in items.items():
        running[k] = running.get(k, 0.0) + v * n


def _average(running: dict, total: int) -> dict:
    """Turn summed loss components into per-sample averages."""
    return {k: (v / max(total, 1)) for k, v in running.items()}


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc=""):
    """Run one training epoch.

    Input:
        model, loader, criterion, optimizer, device as usual.
        epoch_desc: string shown on the progress bar.

    Output:
        dict of average loss components over the epoch ({"total"} here --
        the FCN loss is a single cross-entropy term).
    """
    model.train()
    # Keep frozen BatchNorm layers in eval mode (model.train() just re-enabled
    # them). Here BOTH the backbone and the neck/head ConvSets contain BN, but
    # only backbone BNs are ever frozen, so only those get flipped back.
    model.set_bn_eval_on_frozen()

    running, seen = {}, 0
    for images, masks in tqdm(loader, desc=epoch_desc, leave=False):
        images = images.to(device, non_blocking=True)   # [B, 3, S, S]
        masks = masks.to(device, non_blocking=True)     # [B, S, S] long
        bs = images.size(0)

        # Forward -> per-pixel logits [B, 21, S, S]; CE loss vs the id mask.
        logits = model(images)
        loss, items = criterion(logits, masks)

        # Standard optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _accumulate(running, items, bs)
        seen += bs

    return _average(running, seen)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None, epoch_desc="eval"):
    """One pass over the val set -> average loss AND mIoU / pixel accuracy.

    Detection needed TWO val passes per epoch (cheap loss pass + slow
    decode/NMS mAP pass). Segmentation's metric is just argmax + a histogram,
    so a single pass yields both -- that's why there is no separate
    compute-metric call here, unlike the FCOS train loop.

    Input:
        max_batches: cap on val batches (= images, since batch_size=1); None
            evaluates all 1449. Per-epoch we pass config.EVAL_MAX_BATCHES as
            a fast biased-but-consistent proxy (same trick as FCOS's
            MAP_EVAL_MAX_BATCHES); the final number is computed in full.

    Output:
        dict {"total": val_loss, "miou": ..., "pixel_acc": ...}.
    """
    model.eval()
    cm = ConfusionMatrix(config.NUM_CLASSES)
    running, seen = {}, 0
    for i, (images, masks) in enumerate(tqdm(loader, desc=epoch_desc, leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)   # [1, 3, H', W']
        masks = masks.to(device, non_blocking=True)     # [1, H', W']
        bs = images.size(0)

        logits = model(images)
        _, items = criterion(logits, masks)
        # Hard prediction -> confusion matrix (padding/void are 255 in the
        # mask, so cm.update drops them automatically).
        cm.update(logits.argmax(dim=1), masks)

        _accumulate(running, items, bs)
        seen += bs

    out = _average(running, seen)
    res = cm.compute()
    out["miou"] = res["miou"]
    out["pixel_acc"] = res["pixel_acc"]
    return out


# -----------------------------------------------------------------------------
# Stage runner
# -----------------------------------------------------------------------------
def run_stage(stage_id, model, train_loader, val_loader, criterion, optimizer,
              scheduler, epochs, device, history, best, ckpt_dir):
    """Train for `epochs` epochs, logging + checkpointing each one.

    Input:
        stage_id: 1 or 2 (recorded in the history for plotting).
        history: list of per-epoch dict records, appended in place.
        best: dict {"miou": float, "epoch": int} tracking the best model.
        ckpt_dir: directory to save best.pt / last.pt.

    Output:
        the (possibly updated) `best` dict.
    """
    for e in range(1, epochs + 1):
        # Global epoch number = epochs already recorded + 1.
        global_epoch = len(history) + 1
        t0 = time.time()

        desc = f"[stage {stage_id}] epoch {e}/{epochs}"
        # Read the LR BEFORE training/stepping so it reflects the LR actually
        # used this epoch (scheduler.step() below changes it for the next one).
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, desc)
        # Val loss + mIoU in one pass (proxy over EVAL_MAX_BATCHES images).
        val_metrics = evaluate(model, val_loader, criterion, device,
                               max_batches=config.EVAL_MAX_BATCHES)
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": global_epoch,
            "stage": stage_id,
            "lr": lr,
            "time_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%m-%d %H:%M:%S"),  # wall-clock at epoch end
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_total": val_metrics.get("total", 0.0),
            "miou": val_metrics["miou"],
            "pixel_acc": val_metrics["pixel_acc"],
        }
        history.append(record)

        print(
            f"[{record['timestamp']}] {desc}  lr={lr:.2e}  "
            f"train_total={train_metrics.get('total', 0):.4f}  "
            f"val_total={val_metrics.get('total', 0):.4f}  "
            f"mIoU={val_metrics['miou']:.4f}  "
            f"({record['time_sec']}s)"
        )

        # Checkpoint: always save 'last', save 'best' on mIoU improvement.
        # We select on mIoU (the metric we care about) rather than val loss --
        # like mAP in detection, the two can disagree late in training.
        # NOTE: this is the biased EVAL_MAX_BATCHES proxy, but it is measured
        # on the same val images every epoch, so ranking epochs by it is
        # consistent; the final best.pt mIoU is still computed in full.
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        cur = val_metrics["miou"]
        if cur > best["miou"]:
            best["miou"] = cur
            best["epoch"] = global_epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    return best


# -----------------------------------------------------------------------------
# Logging / plotting
# -----------------------------------------------------------------------------
def save_log(history, output_dir):
    """Write the full per-epoch history to outputs/training_log.json."""
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=2)


def plot_curves(history, output_dir):
    """Plot training curves from the history and save PNGs to output_dir.

    Produces:
        loss_curve.png  - train vs. val cross-entropy loss per epoch.
        miou_curve.png  - val mIoU and pixel accuracy per epoch.
    A dashed vertical line marks the stage-1 -> stage-2 boundary.
    (No loss-components figure, unlike detection: FCN's loss is one term.)

    Note: per-epoch mIoU is the (biased) proxy over the first
    config.EVAL_MAX_BATCHES val images, not the full-set number.
    """
    if not history:
        return
    epochs = [r["epoch"] for r in history]

    # Find where stage 2 starts (for a divider line), if at all.
    stage2_start = next((r["epoch"] for r in history if r["stage"] == 2), None)

    # ---- Figure 1: cross-entropy loss, train vs val ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r.get("train_total", 0) for r in history], label="train")
    plt.plot(epochs, [r.get("val_total", 0) for r in history], label="val")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("U-Net loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ---- Figure 2: val mIoU + pixel accuracy ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r.get("miou", 0) for r in history], label="mIoU")
    plt.plot(epochs, [r.get("pixel_acc", 0) for r in history], label="pixel acc")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.ylim(0, 1)
    plt.title("U-Net val mIoU / pixel accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "miou_curve.png"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Optimizer builders
# -----------------------------------------------------------------------------
def build_stage1_optimizer(model, lr, weight_decay):
    """Stage 1: optimize only the trainable params (neck + head)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_stage2_optimizer(model, lr_head, lr_backbone, weight_decay):
    """Stage 2: two param groups -- unfrozen backbone (small LR) and neck+head.

    Splitting by the "backbone." name prefix lets the pretrained backbone
    layers move slowly while the from-scratch neck/head learn faster.
    """
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            other_params.append(p)
    return optim.Adam(
        [
            {"params": other_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ],
        weight_decay=weight_decay,
    )


def count_trainable(model):
    """Return the number of trainable parameters (in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net on PASCAL VOC 2012 segmentation")
    p.add_argument("--download", action="store_true", help="download VOC before training")
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS)
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"Device: {device}")
    print(f"Data root: {config.DATA_ROOT}")

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device)
    print(f"Train batches: {len(train_loader)}  Val images: {len(val_loader)}")

    # ---- Model + loss ----
    # (No fpn_channels knob anymore: the U-Net decoder's widths are read off
    # the encoder's tap channels instead of being a free hyperparameter.)
    model = UNet(num_classes=config.NUM_CLASSES,
                 pretrained=True, backbone=config.BACKBONE).to(device)
    criterion = FCNLoss(ignore_index=config.IGNORE_INDEX).to(device)

    history = []
    best = {"miou": -1.0, "epoch": -1}

    # ---- Stage 1: freeze encoder, train decoder + head ----
    if args.epochs_stage1 > 0:
        print("\n=== Stage 1: freeze encoder, train decoder + head ===")
        model.freeze_backbone()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_stage1_optimizer(model, config.STAGE1_LR, config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
        best = run_stage(1, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage1, device, history, best, config.OUTPUT_DIR)

    # ---- Stage 2: unfreeze backbone, finetune ----
    if args.epochs_stage2 > 0:
        print(f"\n=== Stage 2: unfreeze {config.STAGE2_UNFREEZE}, finetune ===")
        if config.STAGE2_UNFREEZE == "all":
            model.unfreeze_backbone_all()
        else:
            model.unfreeze_backbone_high(config.STAGE2_UNFREEZE)
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_stage2_optimizer(
            model, config.STAGE2_LR_HEAD, config.STAGE2_LR_BACKBONE, config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
        best = run_stage(2, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage2, device, history, best, config.OUTPUT_DIR)

    # ---- Save logs + curves ----
    save_log(history, config.OUTPUT_DIR)
    plot_curves(history, config.OUTPUT_DIR)
    print(f"\nDone. Best mIoU={best['miou']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {config.OUTPUT_DIR}")

    # ---- Final FULL mIoU on the best checkpoint (all 1449 val images) ----
    best_path = os.path.join(config.OUTPUT_DIR, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("\nComputing full mIoU on VOC2012 val (best checkpoint)...")
        compute_miou(model, val_loader, device)  # verbose: per-class table

if __name__ == "__main__":
    main()
