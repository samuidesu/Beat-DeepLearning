"""Offline checks for generation, corpus scoring and token-weighted perplexity."""

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

import evaluate
import model.translation as translation
from dataset.collate import collate_translation_batch
from evaluation.metrics import compute_metrics
from model import build_model
from training.checkpoint import save_checkpoint
from training.trainer import evaluate_loss, train_one_epoch
from utils.config import load_config


class ToyTokenizer:
    pad_id, unk_id, bos_id, eos_id, vocab_size = 0, 1, 2, 3, 8

    def decode_batch(self, rows):
        return [" ".join(str(token) for token in row if token > self.eos_id)
                for row in rows]


class TableModel(nn.Module):
    """Conditional probabilities indexed by source row and generated prefix."""

    def __init__(self, tables):
        super().__init__()
        self.tables = tables
        self.encode_calls = 0
        self.masks = []

    def encode(self, source_ids, source_attention_mask):
        self.encode_calls += 1
        return source_ids[:, :, None].float()

    def decode(self, source, source_attention_mask, decoder_input_ids,
               decoder_attention_mask, last_token_only=False):
        self.masks.append(decoder_attention_mask.clone())
        logits = torch.full((source.shape[0], 1, 8), -float("inf"))
        for row, prefix in enumerate(decoder_input_ids[:, 1:].tolist()):
            table = self.tables[int(source[row, 0, 0])]
            probabilities = table.get(tuple(prefix), {3: 1.0})
            for token, probability in probabilities.items():
                logits[row, 0, token] = math.log(probability)
        # These would win without special-token suppression.
        logits[:, :, 0] = 100.0
        logits[:, :, 2] = 100.0
        return logits


class TinyEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)

    def forward(self, source_ids, source_attention_mask):
        return self.embedding(source_ids)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.tokenizer = ToyTokenizer()

    def batch(self, size):
        # No reference IDs are supplied: generation must be autoregressive.
        return {"source_ids": torch.arange(size)[:, None],
                "source_attention_mask": torch.ones(size, 1, dtype=torch.long)}

    def test_greedy_eos_padding_and_one_encoder_call(self):
        model = TableModel([{(): {3: 1.0}},
                            {(): {4: 1.0}, (4,): {5: 1.0}, (4, 5): {3: 1.0}}])
        result = evaluate.greedy_decode(model, self.batch(2), self.tokenizer, 8)
        self.assertEqual(result, [[3], [4, 5, 3]])
        self.assertEqual(model.encode_calls, 1)
        self.assertFalse(model.training)
        self.assertEqual(model.masks[-1].tolist(), [[1, 1, 0], [1, 1, 1]])
        beam = evaluate.beam_search_decode(model, self.batch(2), self.tokenizer, 8, 2, 1.0)
        self.assertEqual(beam, result)
        self.assertEqual(model.encode_calls, 2)

    def test_no_eos_stops_at_exact_token_budget(self):
        model = TableModel([{(): {4: 1.0}, (4,): {5: 1.0}, (4, 5): {6: 1.0}}])
        for beams in (1, 2):
            with self.subTest(beams=beams):
                result = evaluate.beam_search_decode(
                    model, self.batch(1), self.tokenizer, 2, beams, 1.0)
                self.assertEqual(result, [[4, 5]])

    def test_beam_one_matches_greedy(self):
        model = TableModel([{(): {4: 0.6, 5: 0.4}, (4,): {3: 0.51, 6: 0.49}}])
        greedy = evaluate.greedy_decode(model, self.batch(1), self.tokenizer, 4)
        beam = evaluate.beam_search_decode(model, self.batch(1), self.tokenizer, 4, 1, 2.0)
        self.assertEqual(beam, greedy)

    def test_beam_finds_better_sequence_and_keeps_sources_separate(self):
        model = TableModel([
            {(): {4: 0.6, 5: 0.4}, (4,): {3: 0.51, 6: 0.49},
             (5,): {3: 0.99, 6: 0.01}},
            {(): {4: 0.4, 5: 0.6}, (4,): {3: 0.99, 6: 0.01},
             (5,): {3: 0.51, 6: 0.49}},
        ])
        result = evaluate.beam_search_decode(model, self.batch(2), self.tokenizer, 2, 2, 0.0)
        self.assertEqual(result, [[5, 3], [4, 3]])
        self.assertEqual(model.encode_calls, 1)

    def test_length_penalty_and_finished_beams_do_not_grow(self):
        model = TableModel([{(): {3: 0.55, 4: 0.45}, (4,): {3: 1.0}}])
        plain = evaluate.beam_search_decode(model, self.batch(1), self.tokenizer, 6, 2, 0.0)
        penalized = evaluate.beam_search_decode(model, self.batch(1), self.tokenizer, 6, 2, 2.0)
        self.assertEqual(plain, [[3]])
        self.assertEqual(penalized, [[4, 3]])

    def test_unfinished_prefix_competes_at_length_cap(self):
        model = TableModel([{(): {3: 0.4, 4: 0.6}, (4,): {5: 0.9, 3: 0.1}}])
        result = evaluate.beam_search_decode(model, self.batch(1), self.tokenizer, 2, 2, 0.0)
        self.assertEqual(result, [[4, 5]])

    def test_cached_encoder_decode_matches_forward(self):
        cfg = load_config()
        cfg["model"].update(dim=8, group=2, decoder_layers=2, dropout=0.0,
                             max_target_positions=8)
        with patch.object(translation, "BertEncoder", TinyEncoder):
            model = build_model(cfg, 8).eval()
        batch = collate_translation_batch([
            {"source_ids": [1, 4, 2], "target_ids": [2, 5, 3]},
            {"source_ids": [1, 2], "target_ids": [2, 4, 6, 3]},
        ], source_pad_id=0, target_pad_id=0)
        inputs = {key: batch[key] for key in ("source_ids", "source_attention_mask",
                                               "decoder_input_ids", "decoder_attention_mask")}
        with torch.no_grad():
            full = model(**inputs)
            source = model.encode(batch["source_ids"], batch["source_attention_mask"])
            last = model.decode(source, batch["source_attention_mask"], batch["decoder_input_ids"],
                                batch["decoder_attention_mask"], last_token_only=True)
        torch.testing.assert_close(last[:, 0], full[:, -1])
        greedy = evaluate.greedy_decode(model, batch, self.tokenizer, 4)
        beam = evaluate.beam_search_decode(model, batch, self.tokenizer, 4, 2, 1.0)
        self.assertEqual(len(greedy), 2)
        self.assertEqual(len(beam), 2)
        self.assertTrue(all(1 <= len(row) <= 4 for row in greedy + beam))

    def test_corpus_metric_wiring(self):
        references = ["România a semnat acest acord în anul 2007.",
                      "Știința și educația sunt priorități naționale."]
        metrics = compute_metrics(references, references)
        self.assertAlmostEqual(metrics["bleu"], 100.0)
        self.assertAlmostEqual(metrics["chrf"], 100.0)

    def test_cli_uses_checkpoint_config_and_saves_all_predictions(self):
        cfg = load_config()
        cfg["model"].update(dim=8, group=2, decoder_layers=1, dropout=0.0,
                             max_target_positions=8)
        cfg["training"].update(device="cpu", mixed_precision=False, eval_batch_size=128)
        cfg["generation"].update(max_new_tokens=2, num_beams=2)
        ds = Dataset.from_list([
            {"source_ids": [1, 4, 2], "target_ids": [2, 4, 5, 3],
             "source": "English text", "target": "4 5"} for _ in range(201)
        ])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(translation, "BertEncoder", TinyEncoder):
            weights = str(Path(directory) / "best.pt")
            save_checkpoint(weights, build_model(cfg, 8), config=cfg, target_vocab_size=8)
            for decoding, limit in (("greedy", None), ("beam", 2)):
                output = str(Path(directory) / "results" / f"{decoding}.json")
                argv = ["evaluate.py", "--weights", weights, "--decoding", decoding,
                        "--save", output]
                if limit:
                    argv += ["--limit", str(limit)]
                with patch.object(sys, "argv", argv), \
                     patch.object(evaluate, "load_config", side_effect=AssertionError("use saved config")), \
                     patch.object(evaluate, "load_source_tokenizer",
                                  return_value=type("SourceTokenizer", (), {"pad_token_id": 0})()), \
                     patch.object(evaluate.TargetTokenizer, "from_dir", return_value=self.tokenizer), \
                     patch.object(evaluate, "prepare_splits", return_value=(DatasetDict(validation=ds), {})), \
                     patch.object(evaluate, "enable_utf8_stdout"), \
                     contextlib.redirect_stdout(io.StringIO()):
                    evaluate.main()
                result = json.loads(Path(output).read_text(encoding="utf-8"))
                count = limit or len(ds)
                self.assertEqual(len(result["predictions"]), count)
                self.assertEqual(result["metrics"]["num_sentences"], count)
                self.assertEqual(result["num_examples"], count)
                self.assertEqual(result["full_split_size"], 201)
                self.assertEqual(result["is_subset"], limit is not None)
                self.assertEqual(result["num_beams"], 1 if decoding == "greedy" else 2)
                self.assertAlmostEqual(result["perplexity"], math.exp(result["nll"]))


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, source_ids, source_attention_mask, decoder_input_ids,
                decoder_attention_mask):
        return self.logits[source_ids[:, 0]]


class PerplexityTests(unittest.TestCase):
    def batches(self):
        # One real token then two: averaging the two batch means is incorrect.
        return [dict(source_ids=torch.tensor([[row]]),
                     source_attention_mask=torch.ones(1, 1, dtype=torch.long),
                     decoder_input_ids=torch.tensor([[2, 1]]),
                     decoder_attention_mask=torch.ones(1, 2, dtype=torch.long),
                     labels=torch.tensor([labels]))
                for row, labels in enumerate(([0, -100], [1, 0]))]

    def test_smoothed_loss_is_separate_from_token_weighted_nll(self):
        model = FixedLogits([[[math.log(0.8), math.log(0.2)], [0.0, 0.0]],
                             [[math.log(0.4), math.log(0.6)], [math.log(0.25), math.log(0.75)]]])
        # Independent probability calculation, with PAD excluded.
        expected_nll = -(math.log(0.8) + math.log(0.6) + math.log(0.25)) / 3
        uniform_loss = -sum(math.log(p) for p in (0.8, 0.2, 0.4, 0.6, 0.25, 0.75)) / 6
        before = copy.deepcopy(model.state_dict())
        for smoothing in (0.0, 0.2):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            train = train_one_epoch(model, self.batches(), optimizer, torch.device("cpu"),
                                    label_smoothing=smoothing, log_every=0)
            validation = evaluate_loss(model, self.batches(), torch.device("cpu"))
            self.assertAlmostEqual(train["loss"], (1 - smoothing) * expected_nll
                                   + smoothing * uniform_loss, places=6)
            for stats in (train, validation):
                self.assertEqual(stats["target_tokens"], 3)
                self.assertAlmostEqual(stats["nll"], expected_nll, places=6)
                self.assertAlmostEqual(stats["perplexity"], math.exp(expected_nll), places=6)
            self.assertAlmostEqual(validation["loss"], expected_nll, places=6)
        torch.testing.assert_close(model.logits, before["logits"], rtol=0, atol=0)

    def test_perplexity_above_exp_twenty_is_not_capped(self):
        model = FixedLogits([[[0.0, 25.0], [0.0, 0.0]],
                             [[25.0, 0.0], [0.0, 25.0]]])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        train = train_one_epoch(model, self.batches(), optimizer, torch.device("cpu"),
                                label_smoothing=0.1, log_every=0)
        validation = evaluate_loss(model, self.batches(), torch.device("cpu"))
        for stats in (train, validation):
            self.assertAlmostEqual(stats["nll"], 25.0)
            self.assertAlmostEqual(stats["perplexity"] / math.exp(25.0), 1.0)


if __name__ == "__main__":
    unittest.main()
