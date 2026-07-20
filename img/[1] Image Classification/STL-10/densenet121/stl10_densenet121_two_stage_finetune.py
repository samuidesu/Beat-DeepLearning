import os
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# Data
# -----------------------------
def get_transforms():
    # ImageNet pretrained DenseNet121 expects ImageNet-style normalization.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(args, device):
    train_transform, eval_transform = get_transforms()

    full_train_set = datasets.STL10(
        root=args.data_dir,
        split="train",
        download=True,
        transform=train_transform,
    )

    val_size = args.val_size
    train_size = len(full_train_set) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)

    train_set, val_set_random_aug = random_split(
        full_train_set,
        [train_size, val_size],
        generator=split_generator,
    )

    # Use eval transform for validation, but keep the exact same val indices.
    full_train_set_eval = datasets.STL10(
        root=args.data_dir,
        split="train",
        download=False,
        transform=eval_transform,
    )
    val_set = torch.utils.data.Subset(full_train_set_eval, val_set_random_aug.indices)

    test_set = datasets.STL10(
        root=args.data_dir,
        split="test",
        download=True,
        transform=eval_transform,
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Model and freezing utilities
# -----------------------------
def build_model(args, device):
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, args.mlp_hidden),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.mlp_hidden, 10),
    )

    freeze_all(model)
    unfreeze_fc(model)

    model = model.to(device)
    return model


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_fc(model):
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_block_and_fc(model):
    # DenseNet121 features: conv0, norm0, relu0, pool0,
    # denseblock1, transition1, denseblock2, transition2,
    # denseblock3, transition3, denseblock4, norm5.
    # denseblock4 (+ norm5) is the last dense block group, closest to the classifier head.
    freeze_all(model)
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True
    unfreeze_fc(model)


def set_frozen_bn_eval(model):
    """
    model.train() will put all BatchNorm layers into training mode.
    For frozen parts, we usually do not want BatchNorm running_mean/running_var to update.
    This function keeps BatchNorm modules in eval mode if their own parameters are frozen.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            has_trainable_param = any(p.requires_grad for p in module.parameters(recurse=False))
            if not has_trainable_param:
                module.eval()


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Train / eval
# -----------------------------
def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    set_frozen_bn_eval(model)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct, total = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_samples += total

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct, total = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_samples += total

    return total_loss / total_samples, total_correct / total_samples


def save_json_log(log_path, payload):
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_lr_dict(optimizer):
    return {
        group.get("name", f"group_{i}"): group["lr"]
        for i, group in enumerate(optimizer.param_groups)
    }


def plot_curves(history, output_dir):
    epochs = [item["global_epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    train_accs = [item["train_acc"] for item in history]
    val_accs = [item["val_acc"] for item in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Global Epoch")
    plt.ylabel("Loss")
    plt.title("STL-10 DenseNet121 Two-Stage Fine-tuning Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="train acc")
    plt.plot(epochs, val_accs, label="val acc")
    plt.xlabel("Global Epoch")
    plt.ylabel("Accuracy")
    plt.title("STL-10 DenseNet121 Two-Stage Fine-tuning Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()

    return loss_path, acc_path


def run_stage(
    *,
    stage_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    start_global_epoch,
    history,
    config,
    log_path,
    best_model_path,
    stage_best_model_path,
    best_val_acc,
):
    stage_best_val_acc = -1.0
    total_start_time = time.time()

    print(f"\n========== {stage_name} ==========")
    print(f"Trainable parameters: {count_trainable_params(model):,}")
    print(f"Initial LRs: {get_lr_dict(optimizer)}")
    print("==================================")

    for epoch_in_stage in range(1, epochs + 1):
        global_epoch = start_global_epoch + epoch_in_stage
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        lr_dict = get_lr_dict(optimizer)
        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - total_start_time

        epoch_log = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage_name,
            "global_epoch": global_epoch,
            "epoch_in_stage": epoch_in_stage,
            "stage_epochs": epochs,
            "lr": lr_dict,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "stage_elapsed_time_sec": elapsed_time,
            "trainable_params": count_trainable_params(model),
        }
        history.append(epoch_log)

        print(
            f"[{stage_name} | global {global_epoch:03d} | "
            f"stage {epoch_in_stage:03d}/{epochs:03d}] "
            f"lr={lr_dict} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        save_json_log(log_path, {"config": config, "history": history})

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "stage": stage_name,
            "global_epoch": global_epoch,
            "epoch_in_stage": epoch_in_stage,
            "val_acc": val_acc,
            "config": config,
        }

        if val_acc > stage_best_val_acc:
            stage_best_val_acc = val_acc
            torch.save(checkpoint, stage_best_model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_model_path)

    return best_val_acc, start_global_epoch + epochs


# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(args, device)
    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss()

    config = {
        "dataset": "STL-10",
        "model": "DenseNet121",
        "pretrained": True,
        "num_classes": 10,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR per stage",
        "stage1_head_epochs": args.head_epochs,
        "stage1_head_lr": args.head_lr,
        "stage2_finetune_epochs": args.finetune_epochs,
        "stage2_last_block_lr": args.last_block_lr,
        "stage2_head_lr": args.head_ft_lr,
        "mlp_hidden": args.mlp_hidden,
        "dropout": args.dropout,
        "val_size": args.val_size,
        "device": str(device),
        "num_workers": args.num_workers,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history = []
    log_path = os.path.join(args.output_dir, "training_log.json")
    best_model_path = os.path.join(args.output_dir, "best_densenet121_two_stage.pth")
    stage1_best_path = os.path.join(args.output_dir, "stage1_best_head_only.pth")
    stage2_best_path = os.path.join(args.output_dir, "stage2_best_last_block_fc.pth")
    last_model_path = os.path.join(args.output_dir, "last_densenet121_two_stage.pth")

    print("========== Training Config ==========")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=====================================")

    best_val_acc = -1.0
    global_epoch = 0

    # Stage 1: freeze backbone, train only the new MLP head.
    freeze_all(model)
    unfreeze_fc(model)
    optimizer_stage1 = optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": args.head_lr, "name": "classifier"},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1,
        T_max=args.head_epochs,
    )

    best_val_acc, global_epoch = run_stage(
        stage_name="stage1_head_only",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        device=device,
        epochs=args.head_epochs,
        start_global_epoch=global_epoch,
        history=history,
        config=config,
        log_path=log_path,
        best_model_path=best_model_path,
        stage_best_model_path=stage1_best_path,
        best_val_acc=best_val_acc,
    )

    # Start stage 2 from the best stage-1 head, not necessarily the last stage-1 epoch.
    stage1_ckpt = torch.load(stage1_best_path, map_location=device)
    model.load_state_dict(stage1_ckpt["model_state_dict"])

    # Stage 2: unfreeze the last DenseNet block group (denseblock4 + norm5) + MLP head, use smaller LRs.
    unfreeze_last_block_and_fc(model)
    last_block_params = (
        list(model.features.denseblock4.parameters())
        + list(model.features.norm5.parameters())
    )
    optimizer_stage2 = optim.AdamW(
        [
            {"params": last_block_params, "lr": args.last_block_lr, "name": "last_block"},
            {"params": model.classifier.parameters(), "lr": args.head_ft_lr, "name": "classifier"},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage2,
        T_max=args.finetune_epochs,
    )

    best_val_acc, global_epoch = run_stage(
        stage_name="stage2_last_block_plus_head",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        device=device,
        epochs=args.finetune_epochs,
        start_global_epoch=global_epoch,
        history=history,
        config=config,
        log_path=log_path,
        best_model_path=best_model_path,
        stage_best_model_path=stage2_best_path,
        best_val_acc=best_val_acc,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "global_epoch": global_epoch,
            "config": config,
        },
        last_model_path,
    )

    # Final test uses the best validation checkpoint across both stages.
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    config["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["best_val_acc"] = best_val_acc
    config["best_checkpoint_stage"] = best_ckpt.get("stage")
    config["best_checkpoint_global_epoch"] = best_ckpt.get("global_epoch")
    config["test_loss"] = test_loss
    config["test_acc"] = test_acc

    loss_fig, acc_fig = plot_curves(history, args.output_dir)

    save_json_log(
        log_path,
        {
            "config": config,
            "history": history,
            "final_test": {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_model_path": best_model_path,
                "stage1_best_path": stage1_best_path,
                "stage2_best_path": stage2_best_path,
                "last_model_path": last_model_path,
                "loss_curve_path": loss_fig,
                "accuracy_curve_path": acc_fig,
            },
        },
    )

    print("========== Final Result ==========")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Best stage:   {config['best_checkpoint_stage']}")
    print(f"Best epoch:   {config['best_checkpoint_global_epoch']}")
    print(f"Test loss:    {test_loss:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
    print(f"Best model:   {best_model_path}")
    print(f"Stage1 best:  {stage1_best_path}")
    print(f"Stage2 best:  {stage2_best_path}")
    print(f"Last model:   {last_model_path}")
    print(f"JSON log:     {log_path}")
    print(f"Loss curve:   {loss_fig}")
    print(f"Acc curve:    {acc_fig}")
    print("==================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./densenet121")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Stage 1: train the new MLP head only.
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-3)

    # Stage 2: unfreeze last dense block (denseblock4 + norm5) + head, then fine-tune with smaller LRs.
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--last_block_lr", type=float, default=1e-5)
    parser.add_argument("--head_ft_lr", type=float, default=5e-5)

    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)
