"""Inference entry point: run a trained tagger on text and show the entities.

The sequence-labeling counterpart of the detection projects' detect.py and the
segmentation projects' segment.py -- and closer to those than to the
classification projects' predict.py, because the output is a STRUCTURE laid
over the input rather than a single label with a confidence.

This script lives in CoNLL-2003/predict/. Results are written to
CoNLL-2003/predict/results/predictions.txt, which is wiped and rewritten on
every run.

Four ways to feed it text:
    --text "..."         one or more sentences straight from the command line
    --file sents.txt     one sentence per line
    --test-random N      N random test sentences (gold tags shown)
    --test-errors N      the N test sentences whose predicted entity set
                         differs from gold, longest-diff first -- the single
                         most useful view for figuring out what the model
                         cannot read. On this corpus the recurring stories are
                         ORG-vs-LOC on country names in sports results, and
                         MISC on nationality adjectives.

ONE THING TO BE CAREFUL ABOUT, and it is specific to this task. The corpus is
PRE-TOKENIZED: the annotators decided that "1996-08-22" is one token and that
"U.S." splits a particular way, and the model has only ever seen text shaped
like that. Free-form text arriving through --text or --file has to be given
the same shape, which is what dataset.vocab.tokenize() is for. It preserves
case (unlike every tokenizer in the classification projects) because case is
a feature here, and it keeps hyphenated forms whole to match CoNLL. Any
mismatch is train/serve skew that shows up as mysteriously bad predictions on
text that looks fine.

Usage (run from the CoNLL-2003 project root):
    python predict/predict.py --text "Peter Blackburn reported from Brussels."
    python predict/predict.py --model transformer --test-random 20
    python predict/predict.py --test-errors 15
    python predict/predict.py --file my_sentences.txt
"""

import os
import sys
import random
import shutil
import argparse

import torch

# This file sits in CoNLL-2003/predict/, so the project root is its parent's
# parent. Put it on sys.path so `import config`, `model`, `utils`, `eval` ...
# resolve regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from dataset.conll2003 import read_split  # noqa: E402
from dataset.vocab import case_ids, tokenize  # noqa: E402
from utils.metrics import extract_entities  # noqa: E402
from utils.viz import format_diff, format_tagged  # noqa: E402
from train import clean_exit, get_device  # noqa: E402  device picker + exit fix
from eval import load_model, load_vocab, model_description  # noqa: E402

# Default output folder: CoNLL-2003/predict/results/ (sits next to this file).
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    p = argparse.ArgumentParser(
        description="Tag named entities with a trained CoNLL-2003 model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+", help="one or more sentences to tag")
    src.add_argument("--file", help="a text file with one sentence per line")
    src.add_argument("--test-random", type=int, metavar="N",
                     help="randomly sample N sentences from the test split")
    src.add_argument("--test-errors", type=int, metavar="N",
                     help="show the N test sentences the model gets most wrong")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible --test-random sampling")
    p.add_argument("--model", choices=["lstm", "transformer"], default=None,
                   help="model the checkpoint was trained with (default: read "
                        "from training_log.json, else config.py)")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in outputs_<model>/)")
    p.add_argument("--vocab", default=None, help="vocab.json path")
    p.add_argument("--out", default=_RESULTS_DIR,
                   help="output folder (wiped and recreated fresh each run)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    return p.parse_args()


def collect_inputs(args):
    """Turn the CLI source flag into a list of (tokens, gold_tags_or_None).

    Output:
        list of (list[str] tokens, list[str] tags | None). Gold tags exist
        only for the test-split modes; free-form text has none, and
        format_diff then prints only the prediction.
    """
    if args.text:
        return [(tokenize(t), None) for t in args.text]

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            return [(tokenize(ln.strip()), None) for ln in f if ln.strip()]

    # Both test modes read the labeled test split, which is already tokenized
    # by the annotators -- so tokenize() is deliberately NOT applied here.
    pairs = read_split("test")
    if args.test_random is not None:
        rng = random.Random(args.seed)
        return rng.sample(pairs, min(args.test_random, len(pairs)))
    return pairs  # --test-errors: tag everything, filter after inference


@torch.no_grad()
def predict(model, vocab, sentences, device, batch_size=128):
    """Tag a list of pre-tokenized sentences.

    Batches them exactly like the training pipeline (encode -> case ids -> pad
    -> lengths), because ANY difference here is train/serve skew: the same
    tokens must produce the same ids they would have produced during training.

    Input:
        model: an eval-mode LSTMTagger or TransformerTagger.
        vocab: the Vocab the checkpoint was trained with.
        sentences: list of token lists (case preserved).
        device / batch_size: as usual.
    Output:
        list of BIO2 tag-string lists, one per sentence, each the same length
        as its input.
    """
    model.eval()
    out = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]
        # An empty line would produce a zero-length sequence, which packing
        # rejects; give it a single <unk> so the batch still forms.
        encoded = [vocab.encode(toks) or [config.UNK_IDX] for toks in chunk]
        cased = [case_ids(toks) or [config.CASE2ID["other"]] for toks in chunk]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)

        ids = torch.full((len(encoded), int(lengths.max())), config.PAD_IDX,
                         dtype=torch.long)
        cases = torch.full_like(ids, config.PAD_IDX)
        for i, (e, c) in enumerate(zip(encoded, cased)):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            cases[i, :len(c)] = torch.tensor(c, dtype=torch.long)

        logits = model(ids.to(device), cases.to(device), lengths.to(device))
        preds = logits.argmax(dim=-1).cpu()
        # Slice each row back to its own length: everything past it is padding
        # and would otherwise become phantom tags hanging off the sentence.
        for i, n in enumerate(lengths.tolist()):
            out.append([config.TAGS[t] for t in preds[i, :n].tolist()])
    return out


def main():
    args = parse_args()
    device = get_device(args.device)

    output_dir = config.output_dir_for_model(args.model or config.MODEL)
    weights = args.weights or os.path.join(output_dir, "best.pt")
    if args.weights:
        output_dir = os.path.dirname(os.path.abspath(weights))
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"{weights} not found -- train a model first (python train.py)")

    vocab = load_vocab(args.vocab, output_dir)
    model, cfg = load_model(weights, vocab, args.model, device, output_dir)
    print(f"Device: {device}")
    print(f"Loaded weights: {weights}")
    print(f"Model: {model_description(cfg)}  vocab={len(vocab)}\n")

    pairs = collect_inputs(args)
    sentences = [toks for toks, _ in pairs]
    golds = [tags for _, tags in pairs]
    preds = predict(model, vocab, sentences, device, args.batch_size)

    if args.test_errors is not None:
        # Keep only the sentences whose ENTITY SET differs from gold, worst
        # first. "Worst" is measured in differing entities rather than
        # differing tokens: one boundary slip that shifts a long span would
        # otherwise dominate the list over a genuine type confusion.
        scored = []
        for i, (gold, pred) in enumerate(zip(golds, preds)):
            g, p = set(extract_entities(gold)), set(extract_entities(pred))
            n_diff = len(g ^ p)
            if n_diff:
                scored.append((n_diff, i))
        scored.sort(reverse=True)
        keep = [i for _, i in scored[:args.test_errors]]
        print(f"{len(scored)} of {len(preds)} test sentences have at least one "
              f"entity wrong; showing the {len(keep)} worst\n")
        sentences = [sentences[i] for i in keep]
        golds = [golds[i] for i in keep]
        preds = [preds[i] for i in keep]

    # Fresh results folder every run (same convention as detect/segment).
    out_dir = args.out
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for toks, gold, pred in zip(sentences, golds, preds):
        block = (format_diff(toks, gold, pred, max_chars=140) if gold is not None
                 else "  " + format_tagged(toks, pred, max_chars=140))
        print(block)
        print()
        lines.append(block)

    # Report entity-level P/R/F1 whenever gold tags were available. Skipped
    # for --test-errors, where the sample is selected to be wrong and a score
    # over it would be meaningless.
    summary = ""
    if any(g is not None for g in golds) and args.test_errors is None:
        tp = fp = fn = 0
        for gold, pred in zip(golds, preds):
            g, p = set(extract_entities(gold)), set(extract_entities(pred))
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        summary = (f"\nentity F1 on these {len(golds)} sentences: {f1:.4f} "
                   f"(P {prec:.4f} / R {rec:.4f}, {tp} correct of {tp + fn} gold)")
        print(summary)

    out_path = os.path.join(out_dir, "predictions.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# weights: {weights}\n# model: {model_description(cfg)}\n\n")
        f.write("\n\n".join(lines) + summary + "\n")
    print(f"\nWrote {out_path}")
    return cfg["model_type"]


if __name__ == "__main__":
    if main() == "lstm":
        clean_exit()  # Keep the existing recurrent exit behavior.
