"""Offline regression checks for stage switching and epoch-boundary resume.

Run from the experiment directory:
    python -m unittest discover -s tests -v

Only the BERT backbone and dataset loading are substituted. The handwritten
decoder, collate function, optimizer, training loop and checkpoints are real.
"""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn

import model.translation as translation
from model import build_model
import train
from dataset.dataset import make_dataloader
from training.checkpoint import load_checkpoint, load_into, save_checkpoint
from training.trainer import (
    build_optimizer, build_optimizer_param_groups, build_scheduler,
    evaluate_loss, train_one_epoch,
)
from utils.config import load_config
from utils.seed import set_seed


class TinyEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, source_ids, source_attention_mask):
        return self.dropout(self.embedding(source_ids))


class TwoStageTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        set_seed(42)
        self.encoder_patch = patch.object(translation, "BertEncoder", TinyEncoder)
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        self.cfg = load_config()
        self.cfg["model"].update(dim=8, group=2, decoder_layers=1,
                                 max_target_positions=32, dropout=0.2)
        self.cfg["training"].update(device="cpu", mixed_precision=False,
                                    batch_size=2, eval_batch_size=2, log_every=0)
        examples = [
            {"source_ids": [1, 4, 2], "target_ids": [2, 4, 5, 3]},
            {"source_ids": [1, 6, 7, 2], "target_ids": [2, 6, 3]},
            {"source_ids": [1, 8, 2], "target_ids": [2, 7, 8, 3]},
            {"source_ids": [1, 9, 10, 2], "target_ids": [2, 9, 3]},
        ]
        self.ds = Dataset.from_list(examples)

    def loader(self):
        return make_dataloader(self.ds, self.cfg, 0, 0, shuffle=True)

    def setup_stage(self, model, stage):
        if stage == 1:
            model.freeze_encoder()
        else:
            model.unfreeze_all()
        groups = build_optimizer_param_groups(model, self.cfg[f"stage{stage}"])
        optimizer = build_optimizer(groups)
        scheduler = build_scheduler(optimizer, 8)
        return optimizer, scheduler

    def test_freeze_then_unfreeze_and_learning_rates(self):
        model = build_model(self.cfg, 16)
        optimizer, scheduler = self.setup_stage(model, 1)
        model.train()
        self.assertFalse(model.bert.training)
        self.assertTrue(model.decoder.training)
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["decoder"])
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        stats = train_one_epoch(model, self.loader(), optimizer, torch.device("cpu"),
                                scheduler=scheduler, log_every=0)
        self.assertTrue(math.isfinite(stats["loss"]))
        for name, parameter in model.bert.named_parameters():
            self.assertTrue(torch.equal(before[f"bert.{name}"], parameter))
            self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(before["target_embedding.weight"], model.prediction.weight))

        optimizer, scheduler = self.setup_stage(model, 2)
        model.train()
        self.assertTrue(model.bert.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["encoder", "decoder"])
        self.assertEqual([g["initial_lr"] for g in optimizer.param_groups], [2e-5, 1e-4])
        grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        before_encoder = model.bert.embedding.weight.detach().clone()
        train_one_epoch(model, self.loader(), optimizer, torch.device("cpu"),
                        scheduler=scheduler, log_every=0)
        self.assertFalse(torch.equal(before_encoder, model.bert.embedding.weight))
        validation = evaluate_loss(model, self.loader(), torch.device("cpu"), max_steps=1)
        self.assertTrue(math.isfinite(validation["loss"]))

    def test_resume_reproduces_next_epoch_in_both_stages(self):
        for stage in (1, 2):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                set_seed(42)
                model = build_model(self.cfg, 16)
                optimizer, scheduler = self.setup_stage(model, stage)
                loader = self.loader()
                stats = train_one_epoch(model, loader, optimizer, torch.device("cpu"),
                                        scheduler=scheduler, log_every=0)
                path = str(Path(directory) / "last.pt")
                save_checkpoint(path, model, optimizer, scheduler, stage=stage,
                                global_step=stats["global_step"], config=self.cfg,
                                extra={"train_generator_state": loader.generator.get_state()})
                train_one_epoch(model, loader, optimizer, torch.device("cpu"),
                                scheduler=scheduler, log_every=0)
                expected = copy.deepcopy(model.state_dict())

                restored = build_model(self.cfg, 16)
                restored_optimizer, restored_scheduler = self.setup_stage(restored, stage)
                restored_loader = self.loader()
                checkpoint = load_checkpoint(path)
                load_into(checkpoint, restored, restored_optimizer,
                          restored_scheduler, restore_rng=True)
                restored_loader.generator.set_state(checkpoint["train_generator_state"])
                train_one_epoch(restored, restored_loader, restored_optimizer,
                                torch.device("cpu"), scheduler=restored_scheduler, log_every=0)
                for name, value in restored.state_dict().items():
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_scaler_skipped_update_does_not_advance_scheduler(self):
        model = build_model(self.cfg, 16)
        optimizer, scheduler = self.setup_stage(model, 1)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        before = copy.deepcopy(model.state_dict())
        before_step = scheduler.last_epoch
        with patch("training.trainer.compute_loss",
                   side_effect=lambda logits, *args: logits.sum() * float("inf")):
            stats = train_one_epoch(model, self.loader(), optimizer, torch.device("cpu"),
                                    scheduler=scheduler, scaler=scaler,
                                    max_steps=1, log_every=0)
        self.assertEqual(stats["global_step"], 0)
        self.assertEqual(scheduler.last_epoch, before_step)
        self.assertLess(scaler.get_scale(), 8.0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_main_stage_transition_resume_and_separate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = copy.deepcopy(self.cfg)
            cfg["training"]["output_dir"] = str(Path(directory) / "stage{stage}")
            splits = DatasetDict(train=self.ds, validation=self.ds)
            source = type("SourceTokenizer", (), {"pad_token_id": 0})()
            target = type("TargetTokenizer", (), {"vocab_size": 16, "pad_id": 0})()
            real_train_epoch = train.train_one_epoch

            def run(*args):
                with patch.object(sys, "argv", ["train.py", *args]), \
                     patch.object(train, "load_config", side_effect=lambda _: copy.deepcopy(cfg)), \
                     patch.object(train, "load_source_tokenizer", return_value=source), \
                     patch.object(train.TargetTokenizer, "from_dir", return_value=target), \
                     patch.object(train, "prepare_splits", return_value=(splits, {})), \
                     patch.object(train, "enable_utf8_stdout"), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            # Simulate interruption after an epoch checkpoint, keeping the
            # original two-epoch scheduler plan for an exact resume comparison.
            for stage in (1, 2):
                calls = []

                def interrupt_second_epoch(*args, **kwargs):
                    if calls:
                        raise InterruptedError("test interruption")
                    calls.append(True)
                    return real_train_epoch(*args, **kwargs)

                with patch.object(train, "train_one_epoch", side_effect=interrupt_second_epoch):
                    with self.assertRaises(InterruptedError):
                        run("--stage", str(stage), "--epochs", "2", "--max-steps", "1",
                            "--max-val-steps", "1")
                last = Path(directory) / f"stage{stage}" / "last.pt"
                first = load_checkpoint(str(last))
                self.assertEqual(first["stage"], stage)
                self.assertEqual(first["global_step"], 1)
                self.assertEqual(len(first["history"]), 1)
                self.assertEqual(len(first["optimizer"]["param_groups"]), stage)
                self.assertEqual(first["best_metric"], first["history"][0]["validation"]["loss"])
                run("--stage", str(stage), "--resume", str(last),
                    "--max-steps", "1", "--max-val-steps", "1")
                resumed = load_checkpoint(str(last))
                self.assertEqual(resumed["epoch"], 1)
                self.assertEqual(resumed["global_step"], 2)
                self.assertEqual(len(resumed["history"]), 2)
                log = json.loads(last.with_name("training_log.json").read_text())
                self.assertEqual([row["epoch"] for row in log], [0, 1])
                self.assertEqual(resumed["best_metric"], min(r["validation"]["loss"] for r in log))
            self.assertEqual(load_checkpoint(str(Path(directory) / "stage1" / "last.pt"))["stage"], 1)


if __name__ == "__main__":
    unittest.main()
