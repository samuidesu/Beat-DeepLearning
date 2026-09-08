"""Inference entry point: run a trained classifier on review text and show the
predicted sentiment with its confidence (the text counterpart of the detection
projects' detect.py and the segmentation projects' segment.py -- print a
probability instead of drawing boxes or painting pixels).

This script lives in IMDB/predict/. Results are written to
IMDB/predict/results/predictions.txt, which is wiped and rewritten on every
run.

Four ways to feed it text:
    --text "..."         one or more reviews straight from the command line
    --file reviews.txt   one review per line
    --test-random N      N random test reviews (gold labels shown)
    --test-mistakes N    the N test reviews the model gets WRONG, most
                         confidently-wrong first -- the single most useful
                         view for figuring out what the model cannot read.
                         On IMDB this is worth actually reading: the
                         confident errors are mostly sarcasm and plot
                         summaries of grim films given warm reviews. (Reviews
                         whose verdict falls outside the truncation window
                         used to be a third category; config.TRUNCATION =
                         "head_tail" is what removed most of it.)

Usage (run from the IMDB project root):
    python predict/predict.py --text "A dull, lifeless remake of a classic."
    python predict/predict.py --cell gru --test-random 20
    python predict/predict.py --test-mistakes 15
    python predict/predict.py --file my_reviews.txt
"""

import os
import sys
import random
import shutil
import argparse

import torch
import torch.nn.functional as F

# This file sits in IMDB/predict/, so the project root is its parent's
# parent. Put it on sys.path so `import config`, `model`, `utils`, `eval` ...
# resolve regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from dataset.imdb import read_split, _clean  # noqa: E402
from dataset.vocab import tokenize  # noqa: E402
from utils.viz import format_prediction  # noqa: E402
from train import clean_exit, get_device  # noqa: E402  device picker + exit fix
from eval import (load_model, load_vocab, model_description,  # noqa: E402
                  read_run_data)

# Default output folder: IMDB/predict/results/ (sits next to this file).
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    p = argparse.ArgumentParser(description="Classify review text with a trained IMDB model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+", help="one or more reviews to classify")
    src.add_argument("--file", help="a text file with one review per line")
    src.add_argument("--test-random", type=int, metavar="N",
                     help="randomly sample N reviews from the test split")
    src.add_argument("--test-mistakes", type=int, metavar="N",
                     help="show the N most confidently WRONG test predictions")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible --test-random sampling")
    p.add_argument("--cell", choices=["rnn", "lstm", "gru"], default=None,
                   help="cell the checkpoint was trained with (default: read "
                        "from training_log.json, else config.py)")
    p.add_argument("--pooling", choices=["last", "max", "mean"], default=None)
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in outputs_<cell>/)")
    p.add_argument("--vocab", default=None, help="vocab.json path")
    p.add_argument("--out", default=_RESULTS_DIR,
                   help="output folder (wiped and recreated fresh each run)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    return p.parse_args()


def collect_inputs(args):
    """Turn the CLI source flag into a list of (text, gold_label_or_None).

    Output:
        list of (str, int | None). Gold labels exist only for the test-split
        modes; free-form text has none, and format_prediction then prints no
        hit/miss marker.
    """
    if args.text:
        return [(t, None) for t in args.text]

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            return [(ln.strip(), None) for ln in f if ln.strip()]

    # Both test modes read the labeled test split.
    pairs = read_split("test")
    if args.test_random is not None:
        rng = random.Random(args.seed)
        return rng.sample(pairs, min(args.test_random, len(pairs)))
    return pairs  # --test-mistakes: score everything, filter after inference


@torch.no_grad()
def predict(model, vocab, texts, device, batch_size=256,
            max_len=None, truncation=None, head_len=None):
    """Classify a list of raw reviews.

    Batches them exactly like the training pipeline (clean -> tokenize ->
    encode -> pad -> lengths), because ANY difference here is train/serve
    skew: the same text must produce the same ids it would have produced
    during training. _clean() in particular is easy to forget -- read_split()
    applies it for the test modes, but text arriving via --text or --file has
    not been through it, so it is applied here for every path. (Pasting a
    review straight off the IMDB website really does bring <br /> tags with
    it, so this is not a hypothetical.)

    Input:
        model: an eval-mode RNNClassifier or TransformerClassifier.
        vocab: the Vocab the checkpoint was trained with.
        texts: list of raw strings.
        device / batch_size: as usual.
        max_len / truncation / head_len: how to cut a too-long review. Pass
            what the CHECKPOINT recorded (main() reads it out of
            training_log.json), not what config.py currently says.
    Output:
        probs [N, C] float tensor of per-class probabilities (on CPU).
    """
    model.eval()
    if max_len is None:
        max_len = getattr(model, "max_len", config.MAX_LEN)
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = [vocab.encode(tokenize(_clean(t)), max_len, truncation,
                                head_len) or [config.UNK_IDX]
                   for t in chunk]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        ids = torch.full((len(encoded), int(lengths.max())), config.PAD_IDX,
                         dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)

        logits = model(ids.to(device), lengths.to(device))
        # softmax turns raw logits into the probabilities we print. The model
        # never applies it internally -- cross-entropy wants raw logits.
        out.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(out) if out else torch.empty(0, config.NUM_CLASSES)


def main():
    args = parse_args()
    device = get_device(args.device)

    output_dir = config.output_dir_for_cell(args.cell or config.CELL)
    weights = args.weights or os.path.join(output_dir, "best.pt")
    if args.weights:
        output_dir = os.path.dirname(os.path.abspath(args.weights))
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"{weights} not found -- train a model first (python train.py)")

    vocab = load_vocab(args.vocab, output_dir)
    model, cfg = load_model(weights, vocab, args.cell, args.pooling, device, output_dir)
    print(f"Device: {device}")
    print(f"Loaded weights: {weights}")
    print(f"Model: {model_description(cfg)}  vocab={len(vocab)}\n")

    pairs = collect_inputs(args)
    texts = [t for t, _ in pairs]
    golds = [y for _, y in pairs]
    data_meta = read_run_data(output_dir)
    probs = predict(model, vocab, texts, device, args.batch_size,
                    max_len=data_meta.get("max_len"),
                    # No "truncation" key = a checkpoint from before the
                    # head_tail switch, trained head-only.
                    truncation=data_meta.get("truncation", "head"),
                    head_len=data_meta.get("head_len"))
    preds = probs.argmax(dim=1).tolist()

    if args.test_mistakes is not None:
        # Keep only the errors, hardest first: sort by the probability the
        # model gave its (wrong) answer, descending. A confidently wrong
        # prediction is a real modeling failure; a 0.51 miss on a mixed review
        # is noise -- and IMDB has plenty of genuinely mixed reviews.
        wrong = [i for i, (p, y) in enumerate(zip(preds, golds))
                 if y is not None and p != y]
        wrong.sort(key=lambda i: float(probs[i, preds[i]]), reverse=True)
        keep = wrong[:args.test_mistakes]
        print(f"{len(wrong)} wrong out of {len(preds)} test reviews "
              f"(accuracy {1 - len(wrong) / max(len(preds), 1):.4f}); "
              f"showing the {len(keep)} most confident errors\n")
        texts = [texts[i] for i in keep]
        golds = [golds[i] for i in keep]
        probs = probs[keep] if keep else probs[:0]

    # Fresh results folder every run (same convention as detect/segment).
    out_dir = args.out
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for text, gold, prob in zip(texts, golds, probs):
        line = format_prediction(text, prob, config.CLASS_NAMES, gold=gold)
        print(line)
        lines.append(line)

    # Report accuracy whenever gold labels were available.
    scored = [(int(p.argmax()), y) for p, y in zip(probs, golds) if y is not None]
    summary = ""
    if scored and args.test_mistakes is None:
        acc = sum(1 for p, y in scored if p == y) / len(scored)
        summary = f"\naccuracy on these {len(scored)} labeled reviews: {acc:.4f}"
        print(summary)

    out_path = os.path.join(out_dir, "predictions.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# weights: {weights}\n# model: {model_description(cfg)}\n")
        f.write("\n".join(lines) + summary + "\n")
    print(f"\nWrote {out_path}")
    return cfg["model_type"]


if __name__ == "__main__":
    if main() == "rnn":
        clean_exit()  # Keep the existing RNN exit behavior.
