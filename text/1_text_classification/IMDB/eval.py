"""Evaluation entry point: score a trained IMDB checkpoint on a split.

Prints accuracy / macro-F1, the per-class precision-recall table and the
confusion matrix, and optionally saves the matrix as a PNG.

THIS is where test.csv gets read. train.py deliberately never touches it --
IMDB publishes its test labels, so evaluating every epoch on test and then
keeping the best epoch would turn the headline into a best-of-N number. The
honest protocol is: select on --split val, report --split test, once.

The 25,000-review test split is as large as train, which is unusually
generous: the 95% confidence interval on an accuracy near 0.88 is +/-0.4
points, tight enough that differences between the four models are worth
taking seriously (unlike the val split, which is ten times smaller).

Usage:
    python eval.py --split test             # LSTM, outputs_lstm/best.pt
    python eval.py --cell gru --split test  # GRU,  outputs_gru/best.pt
    python eval.py --weights outputs_transformer/best.pt --split test
    python eval.py --weights path/to.pt --vocab path/to/vocab.json
    python eval.py --split val              # what training selected on
    python eval.py --split train            # sanity check: fit on train data
    python eval.py --split test --save-cm   # write confusion_matrix_test.png

The model shape (cell / pooling / widths) MUST match how the checkpoint was
trained or load_state_dict fails loudly -- which is the good outcome, unlike
a silent wrong-config evaluation. To make that easy, the shape is read back
from the run's own training_log.json when one sits next to the checkpoint;
explicit flags override, config.py fills whatever is left.
"""

import json
import os
import argparse

import torch
from torch.utils.data import DataLoader

import config
from model.rnn_classifier import RNNClassifier
from model_transformer.transformer_classifier import TransformerClassifier
from dataset.imdb import IMDBDataset, build_vocab_from_train, collate_batch
from dataset.vocab import Vocab
from utils.metrics import compute_accuracy
from utils.viz import plot_confusion_matrix
from train import clean_exit, get_device  # reuse the device picker + exit fix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a sentiment classifier on IMDB")
    p.add_argument("--cell", choices=["rnn", "lstm", "gru"], default=None,
                   help="cell the checkpoint was trained with (MUST match; "
                        "default: read from training_log.json, else config.py)")
    p.add_argument("--pooling", choices=["last", "max", "mean"], default=None,
                   help="pooling the checkpoint was trained with (MUST match)")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in outputs_<cell>/)")
    p.add_argument("--vocab", default=None,
                   help="vocab.json path (default: next to the checkpoint; "
                        "falls back to rebuilding it from train.csv)")
    p.add_argument("--split", choices=["test", "val", "train"], default="test",
                   help="which split to score. test = the reported number; "
                        "val = what training selected on; train = sanity check")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit the number of batches (quick spot-check)")
    p.add_argument("--save-cm", action="store_true",
                   help="save confusion_matrix_<split>.png next to the checkpoint")
    return p.parse_args()


def read_run_meta(output_dir: str, section: str = "model_cfg") -> dict:
    """Return one block of a run's training_log.json meta, or {}.

    Lets eval.py and predict.py rebuild the exact architecture a checkpoint
    was trained with instead of trusting that config.py has not been edited
    since -- the failure mode that would cause is a shape error at best and a
    wrong number at worst.
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
    """The DATA section of training_log.json (max_len, truncation, head_len).

    Separate from the model config because it answers a different question:
    not "what shape was the network" but "how was the text cut before it got
    there". Evaluating with a different cut than training used is silent
    train/serve skew -- the model still runs, it just reads text it has never
    seen the shape of.
    """
    return read_run_meta(output_dir, "data")


def load_model(weights: str, vocab: Vocab, cell: str = None, pooling: str = None,
               device=None, output_dir: str = None):
    """Rebuild the architecture and load a checkpoint into it.

    Input:
        weights: path to best.pt.
        vocab: the Vocab whose size defines the embedding table.
        cell / pooling: explicit overrides (None -> log -> config.py).
        device: torch device to place the model on.
        output_dir: where to look for training_log.json (default: the
            checkpoint's own folder).

    Output:
        the model in eval mode, plus the resolved cfg dict (for printing).
    """
    output_dir = output_dir or os.path.dirname(os.path.abspath(weights))
    meta = read_run_meta(output_dir)
    # Older logs have no model_type and describe an RNN.
    model_type = meta.get("model_type", "rnn")
    if model_type not in ("rnn", "transformer"):
        raise ValueError(f"unknown model_type: {model_type!r}")
    is_transformer = model_type == "transformer"
    if is_transformer and cell is not None:
        raise ValueError("--cell only applies to RNN checkpoints")

    def pick(name, explicit, fallback):
        """Precedence: explicit flag > training_log.json > config.py."""
        return explicit if explicit is not None else meta.get(name, fallback)

    cfg = {
        "pooling": pick("pooling", pooling, config.TRANSFORMER_POOLING
                        if is_transformer else config.POOLING),
        "embed_dim": pick("embed_dim", None, config.EMBED_DIM),
        "num_classes": pick("num_classes", None, config.NUM_CLASSES),
        "pad_idx": pick("pad_idx", None, config.PAD_IDX),
        "num_layers": pick("num_layers", None, config.TRANSFORMER_LAYERS
                           if is_transformer else config.NUM_LAYERS),
        "dropout": pick("dropout", None, config.TRANSFORMER_DROPOUT
                        if is_transformer else config.DROPOUT),
    }
    if is_transformer:
        cfg.update(
            dim=pick("dim", None, config.TRANSFORMER_DIM),
            group=pick("group", None, config.TRANSFORMER_GROUP),
            max_len=pick("max_len", None, config.MAX_LEN),
            # Checkpoints from before the switch have no "norm" key and are
            # all Post-LN, which is also the config default -- so they keep
            # loading unchanged.
            norm=pick("norm", None, config.TRANSFORMER_NORM),
        )
        model_class = TransformerClassifier
    else:
        cfg.update(
            cell=pick("cell", cell, config.CELL),
            hidden_size=pick("hidden_size", None, config.HIDDEN_SIZE),
            bidirectional=pick("bidirectional", None, config.BIDIRECTIONAL),
        )
        model_class = RNNClassifier

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
                f"{cfg.get('norm', 'post')}-LN")
    else:
        prefix = "Bi" if cfg["bidirectional"] else ""
        name = f"{prefix}{cfg['cell'].upper()} h={cfg['hidden_size']}"
    return f"{name} layers={cfg['num_layers']} pooling={cfg['pooling']}"


def load_vocab(path: str = None, output_dir: str = None) -> Vocab:
    """Load vocab.json, falling back to rebuilding it from train.csv.

    The rebuild is deterministic (same csv + same config.SPLIT_SEED +
    MIN_FREQ -> same itos), so it recovers a lost vocab.json -- but only as
    long as config.py has not changed since training, hence the warning.
    """
    if path is None and output_dir is not None:
        cand = os.path.join(output_dir, "vocab.json")
        path = cand if os.path.isfile(cand) else None
    if path and os.path.isfile(path):
        return Vocab.load(path)
    print("[eval] vocab.json not found -- rebuilding from train.csv (ids match "
          "only if config.MIN_FREQ / MAX_VOCAB_SIZE / SPLIT_SEED are unchanged)")
    return build_vocab_from_train()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = config.output_dir_for_cell(args.cell or config.CELL)
    if args.weights is None:
        args.weights = os.path.join(output_dir, "best.pt")
    else:
        output_dir = os.path.dirname(os.path.abspath(args.weights))

    vocab = load_vocab(args.vocab, output_dir)
    model, cfg = load_model(args.weights, vocab, args.cell, args.pooling,
                            device, output_dir)
    print(f"Loaded weights: {args.weights}")
    print(f"Model: {model_description(cfg)}  vocab={len(vocab)}")

    data_meta = read_run_data(output_dir)
    dataset = IMDBDataset(
        args.split, vocab,
        max_len=data_meta.get("max_len", cfg.get("max_len", config.MAX_LEN)),
        # Pre-truncation checkpoints have no "truncation" key and were trained
        # head-only; do not silently re-cut them the new way.
        truncation=data_meta.get("truncation", "head"),
        head_len=data_meta.get("head_len"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_batch)
    cut = (f"head {dataset.head_len}+tail {dataset.max_len - dataset.head_len}"
           if dataset.truncation == "head_tail" else "head")
    print(f"{args.split} reviews: {len(dataset)}  "
          f"<unk> rate: {dataset.unk_rate():.4f}  "
          f"truncated: {dataset.truncated_rate():.2%} ({cut})")

    result = compute_accuracy(model, loader, device, max_batches=args.max_batches)

    if args.save_cm:
        path = os.path.join(output_dir, f"confusion_matrix_{args.split}.png")
        plot_confusion_matrix(
            result["matrix"], config.CLASS_NAMES, path,
            title=f"{model_description(cfg)} {args.split} confusion matrix")
        print(f"\nWrote {path}")
    return cfg["model_type"]


if __name__ == "__main__":
    if main() == "rnn":
        clean_exit()  # Keep the existing RNN exit behavior.
