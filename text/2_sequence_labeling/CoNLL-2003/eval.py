"""Evaluation entry point: score a trained CoNLL-2003 checkpoint on a split.

Prints the per-type precision/recall/F1 table, the micro-averaged headline
score, the token-level diagnostics and the confusion matrix, and optionally
saves the matrix as a PNG.

THIS is where the test split gets read. train.py deliberately never touches
it: the corpus ships a real development split ("testa"), so the honest
protocol is to select on --split valid and report --split test, once.

That discipline matters more here than on the classification projects, because
the CoNLL splits are TIME-SEPARATED rather than randomly drawn -- train is
August 1996 newswire, test is December 1996. The entities themselves turn
over: different people in the news, different companies, different tournaments.
Expect test F1 several points BELOW dev F1 for every model, and read that gap
as distribution shift rather than as overfitting. Every published number on
this benchmark has the same gap.

The 3,684-sentence test split holds 5,648 entities, so the 95% confidence
interval on an F1 near 0.85 is roughly +/-0.9 points -- wider than IMDB's
25,000-review test split gave, and worth remembering before treating a
half-point difference between two models as real.

Usage:
    python eval.py --split test                     # BiLSTM, outputs_lstm/best.pt
    python eval.py --model transformer --split test
    python eval.py --weights outputs_lstm_nocase/best.pt --split test
    python eval.py --split valid                    # what training selected on
    python eval.py --split train                    # sanity check: fit on train
    python eval.py --split test --save-cm           # write confusion_matrix_test.png
    python eval.py --split test --show-errors 20    # read the actual mistakes

The model shape MUST match how the checkpoint was trained or load_state_dict
fails loudly -- which is the good outcome, unlike a silent wrong-config
evaluation. To make that easy, the shape is read back from the run's own
training_log.json when one sits next to the checkpoint; explicit flags
override, config.py fills whatever is left.
"""

import json
import os
import argparse

import torch
from torch.utils.data import DataLoader

import config
from model.lstm_tagger import LSTMTagger
from model_transformer.transformer_tagger import TransformerTagger
from dataset.conll2003 import CoNLLDataset, build_vocab_from_train, collate_batch
from dataset.vocab import Vocab
from utils.metrics import evaluate_tagger
from utils.viz import format_diff, plot_confusion_matrix
from train import clean_exit, get_device  # reuse the device picker + exit fix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an NER tagger on CoNLL-2003")
    p.add_argument("--model", choices=["lstm", "transformer"], default=None,
                   help="model the checkpoint was trained with (MUST match; "
                        "default: read from training_log.json, else config.py)")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in outputs_<model>/)")
    p.add_argument("--vocab", default=None,
                   help="vocab.json path (default: next to the checkpoint; "
                        "falls back to rebuilding it from the train split)")
    p.add_argument("--split", choices=["test", "valid", "train"], default="test",
                   help="which split to score. test = the reported number; "
                        "valid = what training selected on; train = sanity check")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit the number of batches (quick spot-check)")
    p.add_argument("--save-cm", action="store_true",
                   help="save confusion_matrix_<split>.png next to the checkpoint")
    p.add_argument("--show-errors", type=int, default=0, metavar="N",
                   help="print N sentences whose predicted entities differ from "
                        "gold. Reading these is the fastest route to knowing "
                        "WHY the F1 is what it is")
    return p.parse_args()


def read_run_meta(output_dir: str, section: str = "model_cfg") -> dict:
    """Return one block of a run's training_log.json meta, or {}.

    Lets eval.py and predict.py rebuild the exact architecture a checkpoint was
    trained with instead of trusting that config.py has not been edited since
    -- the failure mode that would cause is a shape error at best and a wrong
    number at worst.
    """
    path = os.path.join(output_dir, "training_log.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(payload, dict):
        return (payload.get("meta") or {}).get(section, {}) or {}
    return {}


def read_run_data(output_dir: str) -> dict:
    """The DATA section of training_log.json (max_len, vocab size, counts).

    Separate from the model config because it answers a different question:
    not "what shape was the network" but "how was the text prepared before it
    got there".
    """
    return read_run_meta(output_dir, "data")


def load_model(weights: str, vocab: Vocab, model_type: str = None,
               device=None, output_dir: str = None):
    """Rebuild the architecture and load a checkpoint into it.

    Input:
        weights: path to best.pt.
        vocab: the Vocab whose size defines the embedding table.
        model_type: explicit override (None -> log -> config.py).
        device: torch device to place the model on.
        output_dir: where to look for training_log.json (default: the
            checkpoint's own folder).

    Output:
        the model in eval mode, plus the resolved cfg dict (for printing).
    """
    output_dir = output_dir or os.path.dirname(os.path.abspath(weights))
    meta = read_run_meta(output_dir)
    model_type = model_type or meta.get("model_type", config.MODEL)
    if model_type not in ("lstm", "transformer"):
        raise ValueError(f"unknown model_type: {model_type!r}")
    is_transformer = model_type == "transformer"

    def pick(name, explicit, fallback):
        """Precedence: explicit flag > training_log.json > config.py."""
        return explicit if explicit is not None else meta.get(name, fallback)

    cfg = {
        "embed_dim": pick("embed_dim", None, config.EMBED_DIM),
        "num_tags": pick("num_tags", None, config.NUM_TAGS),
        "pad_idx": pick("pad_idx", None, config.PAD_IDX),
        "num_layers": pick("num_layers", None, config.TRANSFORMER_LAYERS
                           if is_transformer else config.NUM_LAYERS),
        "dropout": pick("dropout", None, config.TRANSFORMER_DROPOUT
                        if is_transformer else config.DROPOUT),
        # The case feature changes the embedding's OUTPUT WIDTH, so getting it
        # wrong is a shape error rather than a silent one -- but read it from
        # the log anyway so `--no-case` runs evaluate without extra flags.
        "use_case": pick("use_case", None, config.USE_CASE_FEATURE),
        "case_dim": pick("case_dim", None, config.CASE_DIM),
    }
    if is_transformer:
        cfg.update(
            dim=pick("dim", None, config.TRANSFORMER_DIM),
            group=pick("group", None, config.TRANSFORMER_GROUP),
            max_len=pick("max_len", None, config.MAX_LEN),
            norm=pick("norm", None, config.TRANSFORMER_NORM),
        )
        model_class = TransformerTagger
    else:
        cfg.update(
            hidden_size=pick("hidden_size", None, config.HIDDEN_SIZE),
            bidirectional=pick("bidirectional", None, config.BIDIRECTIONAL),
        )
        model_class = LSTMTagger

    # pretrained_vectors=None: the checkpoint already holds trained word
    # vectors, so GloVe is not needed (or wanted) at evaluation time.
    model = model_class(vocab_size=len(vocab), pretrained_vectors=None, **cfg)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    cfg["model_type"] = model_type
    return (model.to(device) if device is not None else model), cfg


def model_description(cfg):
    """A shared model label for evaluation, plots and prediction reports."""
    if cfg.get("model_type") == "transformer":
        name = (f"Transformer dim={cfg['dim']} heads={cfg['group']} "
                f"{cfg.get('norm', 'pre')}-LN")
    else:
        prefix = "Bi" if cfg["bidirectional"] else ""
        name = f"{prefix}LSTM h={cfg['hidden_size']}"
    case = "case" if cfg.get("use_case", True) else "NO-case"
    return f"{name} layers={cfg['num_layers']} {case}"


def load_vocab(path: str = None, output_dir: str = None) -> Vocab:
    """Load vocab.json, falling back to rebuilding it from the train split.

    The rebuild is deterministic (same files + same MIN_FREQ -> same itos), so
    it recovers a lost vocab.json -- but only as long as config.py has not
    changed since training, hence the warning.
    """
    if path is None and output_dir is not None:
        cand = os.path.join(output_dir, "vocab.json")
        path = cand if os.path.isfile(cand) else None
    if path and os.path.isfile(path):
        return Vocab.load(path)
    print("[eval] vocab.json not found -- rebuilding from the train split (ids "
          "match only if config.MIN_FREQ / MAX_VOCAB_SIZE are unchanged)")
    return build_vocab_from_train()


@torch.no_grad()
def show_errors(model, dataset, device, n: int, batch_size: int):
    """Print up to `n` sentences whose predicted entity set differs from gold.

    Sentence-level, not token-level, and that is the useful granularity: a
    tagger's mistakes come in shapes (a boundary that drifted one token, a
    country read as ORG instead of LOC), and those shapes are only visible
    with the sentence around them.
    """
    from utils.metrics import extract_entities

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_batch)
    shown = 0
    index = 0
    for ids, cases, lengths, _ in loader:
        preds = model(ids.to(device), cases.to(device),
                      lengths.to(device)).argmax(dim=-1).cpu()
        for row in range(ids.size(0)):
            i, index = index, index + 1
            if shown >= n:
                return
            length = int(lengths[row])
            pred_tags = [config.TAGS[t] for t in preds[row, :length].tolist()]
            gold_tags = dataset.tags[i]
            if set(extract_entities(pred_tags)) == set(extract_entities(gold_tags)):
                continue
            shown += 1
            print(f"\n[{shown}] sentence {i}")
            print(format_diff(dataset.tokens[i], gold_tags, pred_tags,
                              max_chars=140))


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = config.output_dir_for_model(args.model or config.MODEL)
    if args.weights is None:
        args.weights = os.path.join(output_dir, "best.pt")
    else:
        output_dir = os.path.dirname(os.path.abspath(args.weights))

    vocab = load_vocab(args.vocab, output_dir)
    model, cfg = load_model(args.weights, vocab, args.model, device, output_dir)
    print(f"Loaded weights: {args.weights}")
    print(f"Model: {model_description(cfg)}  vocab={len(vocab)}")

    data_meta = read_run_data(output_dir)
    dataset = CoNLLDataset(args.split, vocab,
                           max_len=data_meta.get("max_len", config.MAX_LEN))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_batch)
    print(f"{args.split} sentences: {len(dataset)}  "
          f"tokens: {sum(dataset.full_lengths)}  "
          f"entities: {dataset.entity_counts()}")
    # The OOV rate is the number to watch on this corpus: the splits are
    # time-separated, so test-time unknown words are mostly the NAMES the
    # model is being asked to tag.
    print(f"<unk> rate: {dataset.unk_rate():.4f}  "
          f"all-O token accuracy: {dataset.o_rate():.4f}")

    result = evaluate_tagger(model, loader, device, max_batches=args.max_batches)

    if args.show_errors:
        print(f"\n{'=' * 70}\nsentences whose entity set differs from gold:")
        show_errors(model, dataset, device, args.show_errors, args.batch_size)

    if args.save_cm:
        path = os.path.join(output_dir, f"confusion_matrix_{args.split}.png")
        plot_confusion_matrix(
            result["matrix"], config.TAGS, path,
            title=f"{model_description(cfg)} {args.split} token confusion matrix")
        print(f"\nWrote {path}")
    return cfg["model_type"]


if __name__ == "__main__":
    if main() == "lstm":
        clean_exit()  # Keep the existing recurrent exit behavior.
