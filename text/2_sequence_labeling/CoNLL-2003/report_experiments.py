"""Evaluate existing best.pt files and write an auditable experiment report.

No training, downloads, or changes to the original run directories. Use a new
--output directory for each report. Model selection uses logged dev F1 only.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset.conll2003 import CoNLLDataset, collate_batch
from dataset.vocab import Vocab
from eval import get_device, load_model
from utils.metrics import TaggingMetrics, extract_entities


ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@torch.no_grad()
def score(model, dataset, device, batch_size, diagnose=False):
    """Reuse the training metric; diagnostics partition GOLD entities, not FP."""
    model.eval()
    metrics = TaggingMetrics()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    errors = dict(matched=0, wrong_type_same_boundary=0,
                  boundary_overlap=0, missed_no_overlap=0)
    coverage = {name: dict(gold=0, correct=0)
                for name in ("all_tokens_in_vocab", "contains_unk")}
    loss_sum, token_count = 0.0, 0
    for ids, cases, lengths, labels in loader:
        logits = model(ids.to(device), cases.to(device), lengths.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during evaluation")
        loss_sum += F.cross_entropy(
            logits.flatten(0, 1), labels.to(device).flatten(),
            ignore_index=config.IGNORE_INDEX, reduction="sum").item()
        token_count += int(lengths.sum())
        pred = logits.argmax(-1).cpu()
        metrics.update(pred, labels, lengths)
        if not diagnose:
            continue
        for i, length in enumerate(lengths.tolist()):
            predicted = set(extract_entities([config.TAGS[t] for t in pred[i, :length]]))
            gold = set(extract_entities([config.TAGS[t] for t in labels[i, :length]]))
            for entity in gold:
                typ, start, end = entity
                bucket = ("contains_unk" if (ids[i, start:end] == config.UNK_IDX).any()
                          else "all_tokens_in_vocab")
                coverage[bucket]["gold"] += 1
                coverage[bucket]["correct"] += int(entity in predicted)
                if entity in predicted:
                    errors["matched"] += 1
                elif any(s == start and e == end for _, s, e in predicted):
                    errors["wrong_type_same_boundary"] += 1
                elif any(s < end and start < e for _, s, e in predicted):
                    errors["boundary_overlap"] += 1
                else:
                    errors["missed_no_overlap"] += 1
    result = metrics.compute()
    result["loss_eval_mode"] = loss_sum / token_count
    result["n_tokens"] = token_count
    result["max_sentence_length"] = max(dataset.full_lengths)
    result["unk_rate"] = dataset.unk_rate()
    if diagnose:
        assert sum(errors.values()) == result["n_gold"]
        for bucket in coverage.values():
            bucket["recall"] = bucket["correct"] / max(bucket["gold"], 1)
        result["gold_entity_errors"] = errors
        result["gold_entity_vocab_recall"] = coverage
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; existing reports are preserved")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    logs = sorted(ROOT.glob("outputs_*/training_log.json"))
    if not logs:
        parser.error("No experiment logs found")

    # Lock the inventory and select by dev BEFORE reading test predictions.
    runs = []
    for path in logs:
        log = json.loads(path.read_text(encoding="utf-8"))
        meta, history = log["meta"], log["history"]
        if not meta.get("finished") or not history:
            raise ValueError(f"Incomplete run: {path.parent.name}")
        expected = meta["optim"]["epochs_stage1"] + meta["optim"]["epochs_stage2"]
        if len(history) != expected or meta["data"]["tags"] != config.TAGS:
            raise ValueError(f"Epoch count or tag mapping mismatch: {path}")
        files = [path.parent / name for name in ("best.pt", "vocab.json", "training_log.json")]
        hashes = {p.name: sha256(p) for p in files}
        best = max(history, key=lambda row: row["f1"])
        stage1 = [row["f1"] for row in history if row["stage"] == 1]
        runs.append(dict(run=path.parent.name, source_sha256=hashes,
                         meta=meta, logged_best=best, epochs=len(history),
                         stage1_best_f1=max(stage1) if stage1 else None,
                         mean_epoch_seconds=statistics.mean(r["time_sec"] for r in history)))

    device = get_device(args.device)
    report = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=torch.__version__, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        batch_size=args.batch_size,
        selected_by_logged_dev=max(runs, key=lambda r: r["logged_best"]["f1"])["run"],
        protocol="Existing dev-selected best.pt; full train/valid/test in eval mode; no tuning.",
        diagnostic_definitions={
            "gold_entity_errors": "Exclusive gold partition: exact match, same boundary wrong type, any overlap, no overlap. Not an FP partition.",
            "gold_entity_vocab_recall": "Exact entity recall grouped by whether at least one gold-span token maps to UNK. Not unseen-name F1.",
        },
        source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                       for p in sorted(ROOT.rglob("*.py"))},
        corpus_sha256={split: sha256(Path(config.CONLL_DIR) / f"{split}.txt")
                       for split in ("train", "valid", "test")},
        runs=[],
    )
    args.output.mkdir(parents=True)
    for run in runs:
        name = run["run"]
        folder = ROOT / name
        vocab = Vocab.load(str(folder / "vocab.json"))
        if len(vocab) != run["meta"]["data"]["vocab_size"]:
            raise ValueError(f"Vocabulary size mismatch: {name}")
        model, _ = load_model(str(folder / "best.pt"), vocab, device=device)
        run["parameters"] = sum(p.numel() for p in model.parameters())
        run["non_embedding_parameters"] = run["parameters"] - sum(
            p.numel() for p in model.embedding.parameters())
        run["evaluations"] = {}
        for split in ("valid", "train", "test"):
            ds = CoNLLDataset(split, vocab, max_len=run["meta"]["data"]["max_len"])
            if any(length > ds.max_len for length in ds.full_lengths):
                raise ValueError(f"Truncated gold labels: {name}/{split}")
            result = score(model, ds, device, args.batch_size, diagnose=split == "valid")
            if split == "valid" and not math.isclose(
                    result["f1"], run["logged_best"]["f1"], abs_tol=1e-8):
                raise ValueError(f"Checkpoint does not reproduce dev F1: {name}")
            run["evaluations"][split] = result
            print(f"{name:24s} {split:5s} F1={result['f1']:.6f} "
                  f"P={result['precision']:.6f} R={result['recall']:.6f}", flush=True)
        for filename, expected_hash in run["source_sha256"].items():
            if sha256(folder / filename) != expected_hash:
                raise RuntimeError(f"Run artifact changed during evaluation: {name}/{filename}")
        save_json(args.output / f"{name}.json", run)
        report["runs"].append(run)
        del model
    save_json(args.output / "summary.json", report)
    print(f"Saved {len(runs)} completed evaluations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
