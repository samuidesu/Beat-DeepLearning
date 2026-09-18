"""Score a checkpoint with corpus sacreBLEU and chrF++.

    python evaluate.py --self-test                       # metric wiring, no model
    python evaluate.py --weights outputs_stage1/best.pt --split validation
    python evaluate.py --weights outputs_stage2/best.pt --split test

Greedy and beam search encode each source once and recompute the target
prefix at every step. Decoder KV caching is not implemented.

METRICS ARE CORPUS-LEVEL. Hypotheses and references are collected across the
whole split and scored once -- never per batch and averaged. See
evaluation/metrics.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm

from dataset.collate import move_to_device
from dataset.dataset import make_dataloader, prepare_splits
from dataset.source_tokenizer import load_source_tokenizer
from dataset.target_tokenizer import TargetTokenizer
from evaluation.metrics import compute_metrics, format_metrics
from model import build_model
from training.checkpoint import assert_tokenizer_matches, load_checkpoint, load_into
from training.trainer import amp_settings, evaluate_loss, get_device
from utils.config import load_config, project_path
from utils.console import enable_utf8_stdout
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# decoding -- cached source memory, full target prefixes
# ---------------------------------------------------------------------------
def _next_token_logits(model, source, source_attention_mask, ids, tokenizer):
    """Score only the last position; PAD and BOS are not output tokens."""
    logits = model.decode(
        source, source_attention_mask, ids, ids.ne(tokenizer.pad_id).long(),
        last_token_only=True,
    )[:, -1, :].float()
    logits[:, [tokenizer.pad_id, tokenizer.bos_id]] = -float("inf")
    return logits


@torch.no_grad()
def greedy_decode(model, batch: Dict[str, Any], target_tokenizer,
                  max_new_tokens: int) -> List[List[int]]:
    """Generate from BOS, without references. Return IDs excluding BOS/PAD.

    EOS is included when generated and consumes one of max_new_tokens.
    Rows that finish early are padded while the other rows continue.
    """
    model.eval()
    source_mask = batch["source_attention_mask"]
    source = model.encode(batch["source_ids"], source_mask)
    batch_size = source.shape[0]
    ids = torch.full((batch_size, 1), target_tokenizer.bos_id,
                     dtype=torch.long, device=source.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
    for _ in range(max_new_tokens):
        logits = _next_token_logits(model, source, source_mask, ids, target_tokenizer)
        next_ids = logits.argmax(dim=-1)
        next_ids = next_ids.masked_fill(finished, target_tokenizer.pad_id)
        ids = torch.cat((ids, next_ids[:, None]), dim=1)
        finished |= next_ids.eq(target_tokenizer.eos_id)
        if finished.all():
            break
    return [[token for token in row if token != target_tokenizer.pad_id]
            for row in ids[:, 1:].tolist()]


@torch.no_grad()
def beam_search_decode(model, batch: Dict[str, Any], target_tokenizer,
                       max_new_tokens: int, num_beams: int,
                       length_penalty: float) -> List[List[int]]:
    """Keep K live prefixes per source and retain the best completed sequences.

    Final score = sum(log probabilities) / generated_length ** length_penalty.
    Length includes EOS, excludes BOS. At the length cap, unfinished prefixes
    compete with completed ones. Returned IDs exclude BOS/PAD and retain EOS.
    num_beams=1 uses exactly the same choices as greedy decoding.
    """
    if num_beams == 1:
        return greedy_decode(model, batch, target_tokenizer, max_new_tokens)

    model.eval()
    source_mask = batch["source_attention_mask"]
    source = model.encode(batch["source_ids"], source_mask)
    batch_size = source.shape[0]
    source = source.repeat_interleave(num_beams, dim=0)
    source_mask = source_mask.repeat_interleave(num_beams, dim=0)
    ids = torch.full((batch_size, num_beams, 1), target_tokenizer.bos_id,
                     dtype=torch.long, device=source.device)
    scores = torch.full((batch_size, num_beams), -float("inf"), device=source.device)
    scores[:, 0] = 0.0  # Start from one BOS, not K identical live candidates.
    completed = [[] for _ in range(batch_size)]
    done = [False] * batch_size

    for length in range(1, max_new_tokens + 1):
        logits = _next_token_logits(model, source, source_mask,
                                    ids.flatten(0, 1), target_tokenizer)
        vocab_size = logits.shape[-1]
        log_probs = logits.log_softmax(dim=-1).view(batch_size, num_beams, -1)
        candidates = (scores[:, :, None] + log_probs).flatten(1)
        # At most K candidates end in EOS, so 2K leaves K continuations.
        top_scores, top_indices = candidates.topk(2 * num_beams, dim=-1)
        next_parents, next_tokens, next_scores = [], [], []
        for row, (row_scores, row_indices) in enumerate(
                zip(top_scores.tolist(), top_indices.tolist())):
            live = []
            if not done[row]:
                for rank, (score, index) in enumerate(zip(row_scores, row_indices)):
                    if not math.isfinite(score):
                        continue
                    parent, token = divmod(index, vocab_size)
                    if token == target_tokenizer.eos_id:
                        if rank < num_beams:
                            sequence = ids[row, parent, 1:].tolist() + [token]
                            completed[row].append((score / length ** length_penalty,
                                                   sequence))
                    elif len(live) < num_beams:
                        live.append((score, parent, token))
                completed[row].sort(key=lambda item: item[0], reverse=True)
                del completed[row][num_beams:]

                if not live:
                    done[row] = True
                elif len(completed[row]) == num_beams:
                    # Future log probabilities cannot increase the raw score.
                    # This optimistic bound also accounts for length penalties.
                    best_length = max_new_tokens if length_penalty > 0 else length
                    upper_bound = live[0][0] / best_length ** length_penalty
                    done[row] = completed[row][-1][0] >= upper_bound
            if done[row]:
                live = []
            live += [(-float("inf"), 0, target_tokenizer.pad_id)] * (num_beams - len(live))
            next_scores.append([item[0] for item in live])
            next_parents.append([item[1] for item in live])
            next_tokens.append([item[2] for item in live])

        parents = torch.tensor(next_parents, device=source.device)
        ids = ids.gather(1, parents[:, :, None].expand(-1, -1, ids.shape[-1]))
        tokens = torch.tensor(next_tokens, device=source.device)
        ids = torch.cat((ids, tokens[:, :, None]), dim=-1)
        scores = torch.tensor(next_scores, device=source.device)
        if all(done):
            break

    predictions = []
    for row, row_scores in enumerate(scores.tolist()):
        for beam, score in enumerate(row_scores):
            if math.isfinite(score):
                sequence = ids[row, beam, 1:].tolist()
                completed[row].append((score / len(sequence) ** length_penalty, sequence))
        predictions.append(max(completed[row], key=lambda item: item[0])[1])
    return predictions


# ---------------------------------------------------------------------------
# metric self-test -- runs today, no model required
# ---------------------------------------------------------------------------
def self_test(cfg: Dict[str, Any]) -> None:
    """Score the references against themselves and against a degraded copy.

    A perfect hypothesis set must score BLEU 100 / chrF++ 100. A shuffled-word
    copy must score far lower. If either is off, the metric wiring is wrong and
    no model result computed through it would mean anything.
    """
    references = [
        "Comisia Europeană a aprobat propunerea săptămâna trecută.",
        "România a semnat acordul în 2007.",
        "Știința și educația sunt priorități naționale.",
    ]
    order = cfg["evaluation"].get("chrf_word_order", 2)

    print("\nPerfect hypotheses (hypothesis == reference)")
    print(format_metrics(compute_metrics(references, references, order)))

    degraded = [" ".join(reversed(r.split())) for r in references]
    print("\nDegraded hypotheses (words reversed)")
    print(format_metrics(compute_metrics(degraded, references, order)))

    perfect = compute_metrics(references, references, order)
    assert perfect["bleu"] > 99.9, "identical text must score BLEU 100"
    assert perfect["chrf"] > 99.9, "identical text must score chrF++ 100"
    print("\nMetric wiring OK: corpus sacreBLEU and chrF++ behave as expected.")


# ---------------------------------------------------------------------------
def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None, help="checkpoint to score")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--decoding", default="greedy", choices=("greedy", "beam"))
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N examples (debugging; a "
                             "partial score is NOT a benchmark result)")
    parser.add_argument("--save", default=None, help="write predictions + scores here")
    parser.add_argument("--self-test", action="store_true",
                        help="check the metric path without a model, then exit")
    args = parser.parse_args()

    if args.self_test:
        self_test(load_config(args.config))
        return
    if not args.weights:
        parser.error("--weights is required (or use --self-test)")

    # Reconstruct the architecture/tokenizers used to train these weights.
    # Keep optimizer state on CPU; evaluation never needs it on the GPU.
    checkpoint = load_checkpoint(project_path(args.weights))
    cfg = load_config(args.config) if args.config else checkpoint["config"]
    generation = cfg["generation"]
    max_new_tokens = (args.max_new_tokens if args.max_new_tokens is not None
                      else int(generation["max_new_tokens"]))
    num_beams = (args.num_beams if args.num_beams is not None
                 else int(generation["num_beams"])) if args.decoding == "beam" else 1
    length_penalty = (args.length_penalty if args.length_penalty is not None
                      else float(generation["length_penalty"]))
    if not 1 <= max_new_tokens <= int(cfg["model"]["max_target_positions"]):
        parser.error("--max-new-tokens must be between 1 and model.max_target_positions")
    if num_beams < 1:
        parser.error("--num-beams must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    set_seed(int(cfg["training"]["seed"]))
    device = get_device(cfg["training"].get("device", "auto"))
    amp = amp_settings(device, bool(cfg["training"]["mixed_precision"]))

    source_tokenizer = load_source_tokenizer(cfg["source_tokenizer"]["name"])
    target_tokenizer = TargetTokenizer.from_dir(
        project_path(cfg["paths"]["target_tokenizer_dir"]),
        cfg["target_tokenizer"].get("model_prefix", "spm_ro"))

    # The length check runs under lengths.evaluation_overlength_policy: with
    # the default "error" an over-long official example stops the run instead
    # of being silently dropped.
    splits, data_report = prepare_splits(cfg, source_tokenizer, target_tokenizer,
                                         splits=(args.split,), filter_train_split=False)
    split = splits[args.split]
    full_size = split.num_rows
    if args.limit:
        print(f"\nWARNING: --limit {args.limit} scores a SUBSET of {args.split}. "
              "The result is a debugging number, not a benchmark score.")
        split = split.select(range(min(args.limit, split.num_rows)))
    loader = make_dataloader(split, cfg, source_tokenizer.pad_token_id,
                             target_tokenizer.pad_id, shuffle=False,
                             batch_size=int(cfg["training"]["eval_batch_size"]))

    assert_tokenizer_matches(checkpoint, target_tokenizer.vocab_size)
    model = build_model(cfg, target_tokenizer.vocab_size)
    load_into(checkpoint, model)
    del checkpoint
    model.to(device).eval()

    loss_stats = evaluate_loss(model, loader, device, amp=amp)

    hypotheses: List[str] = []
    references: List[str] = []
    without_eos = 0
    for batch in tqdm(loader, desc=f"generating ({args.decoding})"):
        references.extend(batch["target_text"])
        batch = move_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp["dtype"],
                            enabled=amp["enabled"]):
            if args.decoding == "greedy":
                predicted = greedy_decode(model, batch, target_tokenizer, max_new_tokens)
            else:
                predicted = beam_search_decode(model, batch, target_tokenizer,
                                               max_new_tokens, num_beams, length_penalty)
        without_eos += sum(row[-1] != target_tokenizer.eos_id for row in predicted)
        # decode() drops <pad>/<bos>, stops at <eos>, and joins the subwords
        # back into normal Romanian text.
        hypotheses.extend(target_tokenizer.decode_batch(predicted))

    metrics = compute_metrics(hypotheses, references,
                              cfg["evaluation"].get("chrf_word_order", 2))
    print(f"\n=== {args.split} ===")
    print(format_metrics(metrics, loss=loss_stats["loss"]))
    print(f"  perplexity      : {loss_stats['perplexity']:.4g}")
    print(f"  length cap, no EOS: {without_eos:,}")

    out_path = args.save or os.path.join(
        os.path.dirname(project_path(args.weights)),
        f"eval_{args.split}_{args.decoding}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "weights": args.weights,
                   "decoding": args.decoding, "num_beams": num_beams,
                   "max_new_tokens": max_new_tokens,
                   "length_penalty": length_penalty if args.decoding == "beam" else None,
                   "num_examples": len(references), "full_split_size": full_size,
                   "is_subset": len(references) < full_size,
                   "generations_without_eos": without_eos,
                   "loss": loss_stats["loss"], "nll": loss_stats["nll"],
                   "perplexity": loss_stats["perplexity"], "metrics": metrics,
                   "config": cfg, "data_report": data_report,
                   "predictions": [{"reference": r, "hypothesis": h}
                                   for r, h in zip(references, hypotheses)]},
                  f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
