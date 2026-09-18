"""Two-stage training entry point for WMT16 En->Ro.

    python train.py --stage 1                      # frozen encoder
    python train.py --stage 2 --resume outputs_stage1/best.pt

Stage 2 loads stage 1's best weights and starts a new optimizer and schedule.
Resuming a checkpoint from the same stage restores the optimizer, schedule,
random states, history and next epoch. --epochs is the total for that stage.

THE TWO STAGES
    stage 1   encoder frozen. Only the randomly initialized parts learn:
              target embedding, decoder, cross-attention, LM head. Cross-
              attention starts as noise, and letting that noise backpropagate
              into a pretrained encoder at full learning rate is the fastest
              way to destroy it.
    stage 2   encoder unfrozen at a much smaller learning rate (2e-5 against
              1e-4). BERT already knows English; the decoder side is still
              learning Romanian, so they do not belong at the same rate.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch

from dataset.dataset import make_dataloader, prepare_splits
from dataset.source_tokenizer import load_source_tokenizer
from dataset.target_tokenizer import TargetTokenizer
from model import build_model
from training.checkpoint import (
    BEST_NAME, LAST_NAME, assert_tokenizer_matches, load_checkpoint, load_into,
    save_checkpoint,
)
from training.trainer import (
    amp_settings, build_optimizer, build_optimizer_param_groups, build_scheduler,
    count_parameters, evaluate_loss, get_device, train_one_epoch,
)
from utils.config import describe_config, load_config, project_path
from utils.console import enable_utf8_stdout
from utils.seed import set_seed

def preview_batch(loader, target_vocab_size: int) -> None:
    """Print the shapes of one real batch -- proof the data path works."""
    generator_state = loader.generator.get_state()
    batch = next(iter(loader))
    loader.generator.set_state(generator_state)
    print("\nFirst training batch")
    for key in ("source_ids", "source_attention_mask", "decoder_input_ids",
                "decoder_attention_mask", "labels"):
        print(f"  {key:<23}: {tuple(batch[key].shape)}  {batch[key].dtype}")
    B, T = batch["labels"].shape
    supervised = int((batch["labels"] != -100).sum())
    print(f"  supervised positions   : {supervised:,} of {B * T:,}")
    print(f"  expected model output  : ({B}, {T}, {target_vocab_size})")


def save_log(path: str, history: list) -> None:
    """Rewrite the JSON training log after every epoch, so a crash keeps it."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2),
                        help="1 = frozen encoder, 2 = joint fine-tuning")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None,
                        help="checkpoint to resume from (or to start stage 2 from)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop each epoch after N batches (smoke testing)")
    parser.add_argument("--max-val-steps", type=int, default=None,
                        help="limit validation batches for smoke testing only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage == 2 and args.resume is None:
        stage1_dir = cfg["training"]["output_dir"].format(stage=1)
        args.resume = project_path(stage1_dir, BEST_NAME)
    checkpoint = load_checkpoint(project_path(args.resume)) if args.resume else None
    same_stage = checkpoint is not None and checkpoint.get("stage", 1) == args.stage
    if checkpoint is not None and args.config is None:
        cfg = checkpoint["config"]

    stage_cfg: Dict[str, Any] = dict(cfg[f"stage{args.stage}"])
    stage_cfg["freeze_encoder"] = args.stage == 1
    training_cfg = dict(cfg["training"])
    if args.epochs is not None:
        stage_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
        cfg["training"]["batch_size"] = args.batch_size
    cfg[f"stage{args.stage}"] = stage_cfg

    if args.output_dir:
        output_dir = project_path(args.output_dir)
    elif same_stage:
        output_dir = os.path.dirname(project_path(args.resume))
    else:
        output_dir = project_path(training_cfg["output_dir"].format(stage=args.stage))
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== WMT16 En->Ro :: stage {args.stage} ===")
    print(describe_config(cfg))
    print(f"  stage settings     : {stage_cfg}")
    print(f"  output dir         : {output_dir}")
    if args.max_val_steps is not None:
        print(f"  SMOKE TEST: validation uses at most {args.max_val_steps} batches")

    set_seed(int(training_cfg["seed"]))
    device = get_device(training_cfg.get("device", "auto"))
    amp = amp_settings(device, bool(training_cfg["mixed_precision"]))
    print(f"  device             : {device}  "
          f"(amp {'off' if not amp['enabled'] else amp['dtype']})")

    # --- data -------------------------------------------------------------
    source_tokenizer = load_source_tokenizer(cfg["source_tokenizer"]["name"])
    tokenizer_dir = project_path(cfg["paths"]["target_tokenizer_dir"])
    target_tokenizer = TargetTokenizer.from_dir(
        tokenizer_dir, cfg["target_tokenizer"].get("model_prefix", "spm_ro"))
    print(f"  target vocabulary  : {target_tokenizer.vocab_size}")

    splits, report = prepare_splits(cfg, source_tokenizer, target_tokenizer,
                                    splits=("train", "validation"))
    train_loader = make_dataloader(
        splits["train"], cfg, source_tokenizer.pad_token_id,
        target_tokenizer.pad_id, shuffle=True)
    val_loader = make_dataloader(
        splits["validation"], cfg, source_tokenizer.pad_token_id,
        target_tokenizer.pad_id, shuffle=False,
        batch_size=int(training_cfg["eval_batch_size"]))
    print(f"\n  train batches/epoch: {len(train_loader):,}")
    print(f"  val batches        : {len(val_loader):,}")
    preview_batch(train_loader, target_tokenizer.vocab_size)

    with open(os.path.join(output_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --- model ------------------------------------------------------------
    model = build_model(cfg, target_tokenizer.vocab_size)
    model.to(device)
    if args.stage == 1:
        model.freeze_encoder()
    else:
        model.unfreeze_all()
    print(f"\n  parameters: {count_parameters(model)}")

    # --- optimization -----------------------------------------------------
    param_groups = build_optimizer_param_groups(model, stage_cfg)
    for group in param_groups:
        print(f"  {group['name']:<18}: lr={group['lr']:.2e}, "
              f"{sum(p.numel() for p in group['params']):,} parameters")
    optimizer = build_optimizer(param_groups, float(stage_cfg["weight_decay"]))
    epochs = int(stage_cfg["epochs"])
    steps_per_epoch = len(train_loader)
    if args.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    scheduler = build_scheduler(optimizer, steps_per_epoch * epochs,
                                float(stage_cfg["warmup_ratio"]))
    scaler = torch.amp.GradScaler(device.type) if amp["use_scaler"] else None

    start_epoch, global_step, best_loss = 0, 0, float("inf")
    history: list = []
    if checkpoint is not None:
        assert_tokenizer_matches(checkpoint, target_tokenizer.vocab_size)
        # Stage 2 continues from stage 1's WEIGHTS but starts a fresh schedule:
        # the learning rates and the frozen/unfrozen split both changed.
        state = load_into(checkpoint, model,
                          optimizer if same_stage else None,
                          scheduler if same_stage else None,
                          scaler if same_stage else None,
                          restore_rng=same_stage)
        if same_stage:
            start_epoch = state["epoch"] + 1
            global_step = state["global_step"]
            best_loss = state["best_metric"] if state["best_metric"] is not None else best_loss
            history = list(checkpoint.get("history", []))
            if checkpoint.get("train_generator_state") is not None:
                train_loader.generator.set_state(checkpoint["train_generator_state"])
        print(f"\n  resumed from {args.resume} (epoch {state['epoch']}, "
              f"step {state['global_step']}, "
              f"{'continuing this stage' if same_stage else 'weights only; new stage'})")
        del checkpoint

    # --- loop -------------------------------------------------------------
    log_path = os.path.join(output_dir, "training_log.json")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, scheduler=scheduler,
            scaler=scaler, grad_clip=float(training_cfg["grad_clip"]),
            label_smoothing=float(stage_cfg["label_smoothing"]), amp=amp,
            log_every=int(training_cfg["log_every"]), epoch=epoch,
            max_steps=args.max_steps, global_step=global_step)
        global_step = train_stats["global_step"]
        val_stats = evaluate_loss(model, val_loader, device, amp=amp,
                                  max_steps=args.max_val_steps)

        print(f"epoch {epoch}: train loss {train_stats['loss']:.4f} "
              f"/ NLL {train_stats['nll']:.4f} "
              f"(ppl {train_stats['perplexity']:.4g})  |  "
              f"val NLL {val_stats['nll']:.4f} (ppl {val_stats['perplexity']:.4g})  |  "
              f"{train_stats['seconds'] / 60:.1f} min")

        history.append({"epoch": epoch, "stage": args.stage,
                        "train": train_stats, "validation": val_stats})
        save_log(log_path, history)

        improved = val_stats["loss"] < best_loss
        if improved:
            best_loss = val_stats["loss"]
        checkpoint_args = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, global_step=global_step, config=cfg,
            target_vocab_size=target_tokenizer.vocab_size,
            target_tokenizer_path=tokenizer_dir, stage=args.stage,
            extra={"history": history,
                   "train_generator_state": train_loader.generator.get_state(),
                   "max_steps": args.max_steps, "max_val_steps": args.max_val_steps})
        save_checkpoint(os.path.join(output_dir, LAST_NAME),
                        best_metric=best_loss, **checkpoint_args)
        if improved:
            save_checkpoint(os.path.join(output_dir, BEST_NAME),
                            best_metric=best_loss, **checkpoint_args)
            print(f"  new best validation loss: {best_loss:.4f} -> {BEST_NAME}")

    # Model selection uses validation loss; no test examples enter the loop.
    print(f"\nBest validation loss: {best_loss:.4f}")
    if args.stage == 1:
        print(f'Next: python train.py --stage 2 --resume "{os.path.join(output_dir, BEST_NAME)}"')
    else:
        print(f"Stage 2 complete. Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
