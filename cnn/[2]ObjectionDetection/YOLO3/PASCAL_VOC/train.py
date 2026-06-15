"""Training entry point for YOLOv3 on PASCAL VOC.

Two-stage finetuning:
    Stage 1 - freeze the whole backbone, train only the neck + head.
    Stage 2 - unfreeze the high backbone stages (layer3/layer4) and finetune
              them with a small LR while continuing to train the neck + head.

This file wires up the full pipeline: data -> model -> loss -> optimize, plus
evaluation, checkpointing, logging (training_log.json) and curve plotting.

NOTE: the loss (losses/yolo_loss.py) is currently a zero-placeholder, so the
loss curves will be flat until the real loss is implemented.

Usage:
    python train.py                 # train with config.py defaults
    python train.py --download      # download VOC first, then train
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
from torch.utils.data import DataLoader, ConcatDataset

import config
from models.yolov3 import YOLOv3
from losses.yolo_loss import YOLOLoss
from dataset.voc import VOCDataset, voc_collate_fn, download_voc
from utils.metrics import compute_map

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

    Train = VOC2007 trainval + VOC2012 trainval (with augmentation).
    Val   = VOC2007 test (no augmentation), the standard VOC protocol.

    Output:
        (train_loader, val_loader).
    """
    if download:
        download_voc()

    # Concatenate the two trainval sets into one training set.
    train_set = ConcatDataset([
        VOCDataset(year="2007", image_set="trainval", train=True),
        VOCDataset(year="2012", image_set="trainval", train=True),
    ])
    val_set = VOCDataset(year="2007", image_set="test", train=False)

    # pin_memory only helps when copying to a CUDA device.
    pin = device.type == "cuda"
    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=voc_collate_fn, pin_memory=pin, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=voc_collate_fn, pin_memory=pin,
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
        dict of average loss components over the epoch, e.g.
        {"total","box","obj","noobj","cls"}.
    """
    model.train()
    # Keep frozen BatchNorm layers in eval mode (model.train() just re-enabled them).
    model.set_bn_eval_on_frozen()

    running, seen = {}, 0
    for images, targets in tqdm(loader, desc=epoch_desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)

        # Forward -> 3 raw prediction tensors; loss reduces them against targets.
        preds = model(images)
        loss, items = criterion(preds, targets)

        # Standard optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _accumulate(running, items, bs)
        seen += bs

    return _average(running, seen)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_desc="eval"):
    """Compute the average validation loss (no gradient updates).

    Output:
        dict of average loss components over the val set.

    TODO: also compute mAP@0.5 here once decode (utils) + NMS (utils/nms.py)
    + metrics (utils/metrics.py) are implemented.
    """
    # NOTE: mAP is computed separately (it needs decode + NMS and is slow to run
    # every epoch). See utils.metrics.compute_map / eval.py; this function only
    # tracks the (fast) validation loss for monitoring + checkpoint selection.
    model.eval()
    running, seen = {}, 0
    for images, targets in tqdm(loader, desc=epoch_desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)

        preds = model(images)
        _, items = criterion(preds, targets)

        _accumulate(running, items, bs)
        seen += bs

    return _average(running, seen)


# -----------------------------------------------------------------------------
# Stage runner
# -----------------------------------------------------------------------------
def run_stage(stage_id, model, train_loader, val_loader, criterion, optimizer,
              scheduler, epochs, device, history, best, ckpt_dir):
    """Train for `epochs` epochs, logging + checkpointing each one.

    Input:
        stage_id: 1 or 2 (recorded in the history for plotting).
        history: list of per-epoch dict records, appended in place.
        best: dict {"map_50": float, "epoch": int} tracking the best model.
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
        val_metrics = evaluate(model, val_loader, criterion, device)
        # mAP on the val set (decode + NMS + COCO-style AP). This is the metric
        # we actually care about; val loss alone can disagree with it.
        map_metrics = compute_map(model, val_loader, device,
                                  max_batches=config.MAP_EVAL_MAX_BATCHES, verbose=False)
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": global_epoch,
            "stage": stage_id,
            "lr": lr,
            "time_sec": round(time.time() - t0, 1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "map": map_metrics["map"],
            "map_50": map_metrics["map_50"],
            "map_75": map_metrics["map_75"],
        }
        history.append(record)

        print(
            f"{desc}  lr={lr:.2e}  "
            f"train_total={train_metrics.get('total', 0):.4f}  "
            f"val_total={val_metrics.get('total', 0):.4f}  "
            f"mAP@0.5={map_metrics['map_50']:.4f}  "
            f"({record['time_sec']}s)"
        )

        # Checkpoint: always save 'last', save 'best' on mAP@0.5 improvement.
        # We select on mAP (the metric we care about) rather than val loss --
        # the two can disagree, and a lower val loss doesn't guarantee better
        # detections. NOTE: this is the biased MAP_EVAL_MAX_BATCHES proxy, but
        # it's measured on the same val images every epoch, so ranking epochs by
        # it is consistent; the final best.pt mAP is still computed in full.
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        cur_map = map_metrics["map_50"]
        if cur_map > best["map_50"]:
            best["map_50"] = cur_map
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
    """Plot loss curves from the history and save PNGs to output_dir.

    Produces:
        loss_curve.png       - train vs. val TOTAL loss per epoch.
        loss_components.png  - train box/obj/noobj/cls losses per epoch.
        map_curve.png        - val mAP / mAP@0.5 / mAP@0.75 per epoch.
    A dashed vertical line marks the stage-1 -> stage-2 boundary.

    Note: the per-epoch mAP is the (biased) proxy over the first
    config.MAP_EVAL_MAX_BATCHES val batches, not the full-set number.
    """
    if not history:
        return
    epochs = [r["epoch"] for r in history]

    # Find where stage 2 starts (for a divider line), if at all.
    stage2_start = next((r["epoch"] for r in history if r["stage"] == 2), None)

    # ---- Figure 1: total loss, train vs val ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r.get("train_total", 0) for r in history], label="train")
    plt.plot(epochs, [r.get("val_total", 0) for r in history], label="val")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.title("YOLOv3 total loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ---- Figure 2: per-component train losses ----
    plt.figure(figsize=(8, 5))
    for comp in ("box", "obj", "noobj", "cls"):
        key = f"train_{comp}"
        if any(key in r for r in history):
            plt.plot(epochs, [r.get(key, 0) for r in history], label=comp)
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("YOLOv3 loss components (train)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_components.png"), dpi=150)
    plt.close()

    # ---- Figure 3: val mAP curves ----
    # Only plot if mAP was actually logged (older runs may not have it).
    if any("map_50" in r for r in history):
        plt.figure(figsize=(8, 5))
        for key, label in (("map_50", "mAP@0.5"),
                           ("map", "mAP@[.5:.95]"),
                           ("map_75", "mAP@0.75")):
            if any(key in r for r in history):
                plt.plot(epochs, [r.get(key, 0) for r in history], label=label)
        if stage2_start is not None:
            plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
        plt.xlabel("epoch")
        plt.ylabel("mAP")
        plt.ylim(bottom=0)
        plt.title("YOLOv3 val mAP")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "map_curve.png"), dpi=150)
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
    p = argparse.ArgumentParser(description="Train YOLOv3 on PASCAL VOC")
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

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ---- Model + loss ----
    model = YOLOv3(num_classes=config.NUM_CLASSES,
                   num_anchors=config.NUM_ANCHORS_PER_SCALE,
                   pretrained=True).to(device)
    # .to(device) moves the loss's anchor buffers onto the GPU too; without this
    # they stay on CPU and cause a device-mismatch on CUDA/MPS.
    criterion = YOLOLoss(
        anchors=config.ANCHORS, strides=config.STRIDES,
        num_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE,
    ).to(device)

    history = []
    best = {"map_50": -1.0, "epoch": -1}

    # ---- Stage 1: freeze backbone, train neck + head ----
    if args.epochs_stage1 > 0:
        print("\n=== Stage 1: freeze backbone, train neck + head ===")
        model.freeze_backbone()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_stage1_optimizer(model, config.STAGE1_LR, config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
        best = run_stage(1, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage1, device, history, best, config.OUTPUT_DIR)

    # ---- Stage 2: unfreeze high backbone layers, finetune ----
    if args.epochs_stage2 > 0:
        print(f"\n=== Stage 2: unfreeze {config.STAGE2_UNFREEZE}, finetune ===")
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
    print(f"\nDone. Best mAP@0.5={best['map_50']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {config.OUTPUT_DIR}")

    # ---- Final mAP on the best checkpoint ----
    best_path = os.path.join(config.OUTPUT_DIR, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("\nComputing mAP on VOC2007 test (best checkpoint)...")
        compute_map(model, val_loader, device)


if __name__ == "__main__":
    main()
