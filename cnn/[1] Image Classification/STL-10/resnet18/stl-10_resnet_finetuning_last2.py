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
from torchvision.models import resnet18, ResNet18_Weights


def set_seed(seed: int = 42):
    """
    固定随机种子，尽量保证实验可复现。
    注意：GPU 上完全 bit-level deterministic 有时仍然受 CUDA/cuDNN 版本影响。
    """
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
    """
    DataLoader 多进程 worker 的 seed。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms():
    """
    STL-10 原图是 96x96。
    但 ResNet18 pretrained on ImageNet 通常用 224x224 输入，
    所以这里把图像 resize/crop 到 224，并使用 ImageNet mean/std。
    """
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


def build_dataloaders(args):
    train_transform, eval_transform = get_transforms()

    full_train_set = datasets.STL10(
        root=args.data_dir, split="train", download=True, transform=train_transform
    )

    # 为了每个 epoch 有 validation 曲线，这里从官方 train 里切出一部分做 val。
    # STL-10 train 一共 5000 张，默认 4500 train / 500 val。
    val_size = args.val_size
    train_size = len(full_train_set) - val_size

    generator = torch.Generator().manual_seed(args.seed)

    train_set, val_set = random_split(full_train_set, [train_size, val_size], generator=generator)

    # 注意：random_split 后 val_set 仍然使用 full_train_set 的 train_transform。
    # 为了让 val 不使用随机增强，重新构造一个 eval_transform 的 full_train_set_eval，
    # 然后用相同 indices 创建 val_set。
    full_train_set_eval = datasets.STL10(
        root=args.data_dir, split="train", download=False, transform=eval_transform
    )
    val_set = torch.utils.data.Subset(full_train_set_eval, val_set.indices)

    test_set = datasets.STL10(
        root=args.data_dir, split="test", download=True, transform=eval_transform
    )

    loader_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    return train_loader, val_loader, test_loader


def build_model(args, device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10)
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def save_json_log(log_path, payload):
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_curves(history, output_dir):
    epochs = [item["epoch"] for item in history]

    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    train_accs = [item["train_acc"] for item in history]
    val_accs = [item["val_acc"] for item in history]

    # Loss figure
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("STL-10 ResNet18 Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Accuracy figure
    plt.figure()
    plt.plot(epochs, train_accs, label="train acc")
    plt.plot(epochs, val_accs, label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("STL-10 ResNet18 Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()

    return loss_path, acc_path


def main(args):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(args)
    model = build_model(args, device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        "dataset": "STL-10",
        "model": "ResNet18",
        "pretrained": True,
        "num_classes": 10,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "freeze_backbone": args.freeze_backbone,
        "val_size": args.val_size,
        "device": str(device),
        "num_workers": args.num_workers,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history = []
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_resnet18_stl10.pth")
    last_model_path = os.path.join(args.output_dir, "last_resnet18_stl10.pth")
    log_path = os.path.join(args.output_dir, "training_log.json")

    total_start_time = time.time()

    print("========== Training Config ==========")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=====================================")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - total_start_time

        epoch_log = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "epochs": args.epochs,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "elapsed_time_sec": elapsed_time,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "weight_decay": args.weight_decay,
            "freeze_backbone": args.freeze_backbone,
            "device": str(device),
        }

        history.append(epoch_log)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={current_lr:.6g} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # 每个 epoch 都保存 JSON，防止中途停止后 log 丢失。
        save_json_log(log_path, {"config": config, "history": history})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
                best_model_path,
            )

    # 保存最后一个 epoch 的模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": config,
        },
        last_model_path,
    )

    # 用 best model 在 test set 上评估
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model, loader=test_loader, criterion=criterion, device=device
    )

    config["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["best_val_acc"] = best_val_acc
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
                "last_model_path": last_model_path,
                "loss_curve_path": loss_fig,
                "accuracy_curve_path": acc_fig,
            },
        },
    )

    print("========== Final Result ==========")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test loss:    {test_loss:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
    print(f"Best model:   {best_model_path}")
    print(f"Last model:   {last_model_path}")
    print(f"JSON log:     {log_path}")
    print(f"Loss curve:   {loss_fig}")
    print(f"Acc curve:    {acc_fig}")
    print("==================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./esnet18_finetuning_last2")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)
