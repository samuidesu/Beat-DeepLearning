"""Evaluate a dev-selected BERT checkpoint; never train or select on test.

    python eval_bert.py --weights outputs_bert/best.pt --split test --save-cm
    python eval_bert.py --weights outputs_bert_lr5e-5/best.pt --split valid

The adjacent training_log.json supplies the model and data settings. Test
evaluation first reproduces the logged best dev F1. Full-precision results
and provenance are saved separately; existing reports are never overwritten.
Set HF_HUB_OFFLINE=1 to use only the already-cached tokenizer and base model.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader

import config
from model_bert.bert_tagger import BertTagger
from model_bert.wordpiece import BertCoNLLDataset, collate_bert_batch, load_tokenizer
from train import get_device, set_seed
from utils.metrics import evaluate_tagger
from utils.viz import plot_confusion_matrix


ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_run(weights):
    """Require complete BERT metadata rather than guessing from current defaults."""
    path = weights.parent / "training_log.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta, history = payload["meta"], payload["history"]
    if meta["model_cfg"]["model_type"] != "bert":
        raise ValueError("This entry point accepts BERT checkpoints only")
    if not meta.get("finished") or len(history) != meta["optim"]["epochs"] or not history:
        raise ValueError("The training run is incomplete")
    if meta["data"]["tags"] != config.TAGS or meta["model_cfg"]["num_tags"] != len(config.TAGS):
        raise ValueError("The saved tag order/count differs from config.TAGS")
    if meta["optim"]["ignore_index"] != config.IGNORE_INDEX:
        raise ValueError("The saved label padding value differs from config.IGNORE_INDEX")
    best = max(history, key=lambda row: row["f1"])
    if best["epoch"] != meta["best_dev"]["epoch"]:
        raise ValueError("The recorded best epoch disagrees with dev history")
    return meta, best


def check_dev(result, best):
    # Compare to full-precision history, not the four-decimal summary.
    if not math.isclose(result["f1"], best["f1"], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"Checkpoint did not reproduce dev F1: {result['f1']:.10f} "
            f"!= logged {best['f1']:.10f}. Test has not been evaluated.")


def score(model, tokenizer, split, max_len, batch_size, num_workers, device):
    """Use the same word alignment and entity metric as BERT training."""
    dataset = BertCoNLLDataset(split, tokenizer, max_len=max_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=partial(collate_bert_batch, pad_id=tokenizer.pad_token_id),
        pin_memory=device.type == "cuda",
    )
    print(f"\nEvaluating full {split}: {len(dataset)} sentences", flush=True)
    result = evaluate_tagger(model, loader, device)
    result.update(
        n_tokens=sum(len(words) for words in dataset.tokens),
        n_pieces=sum(len(ids) - 2 for ids in dataset.input_ids),
        max_pieces=max(len(ids) for ids in dataset.input_ids),
        unk_word_rate=dataset.unk_rate(),
    )
    if result["n_sentences"] != len(dataset) or sum(map(sum, result["matrix"])) != result["n_tokens"]:
        raise RuntimeError("Evaluation did not cover every sentence and word")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path(config.OUTPUT_DIR_BERT) / "best.pt")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: eval_batch_size from the training log")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="new JSON path; default: eval_<split>.json beside weights")
    parser.add_argument("--save-cm", action="store_true",
                        help="save confusion_matrix_<split>.png beside the report")
    args = parser.parse_args()
    if (args.batch_size is not None and args.batch_size <= 0) or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    return args


def main():
    args = parse_args()
    weights = args.weights.resolve()
    output = (args.output or weights.parent / f"eval_{args.split}.json").resolve()
    matrix_path = output.parent / f"confusion_matrix_{args.split}.png"
    if output.exists() or (args.save_cm and matrix_path.exists()):
        raise FileExistsError("Report already exists; use --output in a new directory")
    meta, best = read_run(weights)
    cfg = meta["model_cfg"]
    batch_size = args.batch_size or meta["data"]["eval_batch_size"]
    device = get_device(args.device)
    set_seed(meta["seed"])

    splits = ["valid"] if args.split == "valid" else ["valid", "test"]
    files = [weights, weights.parent / "training_log.json"]
    files += [Path(config.CONLL_DIR) / f"{split}.txt" for split in splits]
    files += sorted(ROOT.rglob("*.py"))
    hashes = {str(path): sha256(path) for path in files}

    tokenizer = load_tokenizer(cfg["bert"])
    if tokenizer.pad_token_id != cfg["pad_id"]:
        raise ValueError("The tokenizer PAD id differs from the training log")
    model = BertTagger(cfg["bert"], num_tags=cfg["num_tags"],
                       dropout=cfg["dropout"], pad_id=cfg["pad_id"])
    # Load the WHOLE fine-tuned model, including the NER head, strictly.
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(device).eval()
    if meta["data"]["max_len"] > model.bert.config.max_position_embeddings:
        raise ValueError("The saved sequence limit exceeds BERT's position table")
    print(f"Device: {device}  precision: float32  batch_size: {batch_size}")
    print(f"Weights: {weights}\nLogged best dev epoch: {best['epoch']}", flush=True)

    evaluations = {}
    for split in splits:
        result = score(model, tokenizer, split, meta["data"]["max_len"],
                       batch_size, args.num_workers, device)
        if split == "valid":
            check_dev(result, best)
            print("Verified: best dev F1 reproduced before test.", flush=True)
        evaluations[split] = result

    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Input changed during evaluation: {path}")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "run": weights.parent.name, "weights": str(weights),
        "protocol": "Dev-selected checkpoint; verified dev F1; full split, eval mode, no-grad, FP32; no tuning.",
        "python_executable": sys.executable, "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "batch_size": batch_size, "num_workers": args.num_workers,
        "meta": meta, "logged_best": best,
        "bert_config": model.bert.config.to_dict(),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
        "source_sha256": hashes, "evaluations": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if args.save_cm:
        plot_confusion_matrix(evaluations[args.split]["matrix"], config.TAGS,
                              str(matrix_path), title=f"BERT {args.split} token confusion matrix")
    print(f"\nSaved report: {output}", flush=True)


if __name__ == "__main__":
    main()
