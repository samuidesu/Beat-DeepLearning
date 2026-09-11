"""Offline checks for the BERT evaluation guards; no model downloads."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import config
from eval_bert import check_dev, read_run


class EvaluationGuardsTest(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "meta": {
                "model_cfg": {"model_type": "bert", "num_tags": len(config.TAGS)},
                "data": {"tags": list(config.TAGS)},
                "optim": {"epochs": 2, "ignore_index": config.IGNORE_INDEX},
                "finished": "2026-09-10", "best_dev": {"epoch": 1},
            },
            "history": [{"epoch": 1, "f1": 0.91234567}, {"epoch": 2, "f1": 0.90}],
        }

    def read(self, payload):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "training_log.json").write_text(json.dumps(payload), encoding="utf-8")
            return read_run(path / "best.pt")

    def test_best_epoch_comes_from_dev_history(self):
        _, best = self.read(self.payload)
        self.assertEqual(best, self.payload["history"][0])

    def test_rejects_incompatible_or_incomplete_metadata(self):
        changes = [
            ("model_cfg", "model_type", "lstm"),
            ("model_cfg", "num_tags", 8),
            ("data", "tags", list(reversed(config.TAGS))),
            ("optim", "epochs", 3),
            ("optim", "ignore_index", 0),
            ("best_dev", "epoch", 2),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                payload = copy.deepcopy(self.payload)
                payload["meta"][section][key] = value
                with self.assertRaises(ValueError):
                    self.read(payload)

    def test_dev_check_uses_full_precision(self):
        best = self.payload["history"][0]
        check_dev({"f1": best["f1"]}, best)
        with self.assertRaises(ValueError):
            check_dev({"f1": round(best["f1"], 4)}, best)


if __name__ == "__main__":
    unittest.main()
