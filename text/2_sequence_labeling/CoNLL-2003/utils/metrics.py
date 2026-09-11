"""Entity-level evaluation: span extraction, micro/macro F1, BIO validity.

THE MOST IMPORTANT FILE IN THIS PROJECT, because it is where the task's
definition of "right" actually lives -- and that definition is not the one the
classification projects used.

WHY TOKEN ACCURACY IS NOT THE METRIC. 83% of CoNLL-2003 tokens are tagged "O".
A model that predicts "O" for every token -- one that has found no entities at
all, and is useless -- scores 83% token accuracy. Worse, it scores it
STABLY: a model that improves from finding no entities to finding half of them
moves token accuracy by a few points, so the number cannot distinguish
"learning the task" from "learning the prior". Token accuracy is still
computed below, precisely so that gap stays visible, but it is never the
headline.

WHAT THE METRIC IS. Entity-level, exact match, micro-averaged F1 -- the
CoNLL-2000/2003 shared task definition, which is what every published number
on this corpus means. A prediction counts as a true positive only when BOTH
the span boundaries AND the type match a gold entity exactly:

    gold                pred                  result
    [European Commission]ORG  [European Commission]ORG   1 TP
    [European Commission]ORG  [European]ORG              1 FP + 1 FN
    [European Commission]ORG  [European Commission]LOC   1 FP + 1 FN
    [European Commission]ORG  (nothing)                  1 FN

Note the second row: a boundary error is scored as BOTH a false positive and a
false negative, and earns no partial credit for the word it got right. That is
harsh, and it is the point -- a half-extracted entity is not half-useful to
anything downstream.

MICRO, NOT MACRO. The official score pools TP/FP/FN across all four types
before dividing, so a type with more entities counts for more. Macro-F1 (the
unweighted mean of the four per-type F1s) is also reported here because it
exposes something micro hides: MISC is both the rarest and the hardest type,
so a model can lose several macro points while barely moving micro. When this
project quotes "F1" with no qualifier, it means micro.

BIO VALIDITY, which is the diagnostic the missing CRF would have fixed. A
softmax head scores each token independently, so nothing prevents it emitting
"O, I-PER" -- a continuation with no beginning, which is not a legal BIO2
sequence. count_invalid_transitions() counts exactly those, so the size of the
problem a CRF would remove is measured rather than argued about. The scoring
itself follows conlleval and is LENIENT about them (a stray I-X is read as
starting an entity), which is why the count has to be reported separately:
without it, the leniency would hide the defect it is papering over.
"""

import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402


# -----------------------------------------------------------------------------
# Span extraction
# -----------------------------------------------------------------------------
def extract_entities(tags) -> list:
    """Turn a BIO2 tag sequence into a list of (type, start, end) spans.

    `end` is EXCLUSIVE, so a one-token entity at position 3 is (type, 3, 4)
    and spans compare with plain tuple equality -- which is what makes the
    exact-match scoring below a set intersection.

    Malformed input is handled the way conlleval does: a bare "I-X" that does
    not continue an X (because it is first, or follows "O", or follows a
    different type) is read as STARTING an entity. Anything stricter would
    score a model on tag hygiene rather than on the entities it found, and
    anything more lenient would let boundary errors disappear.
    Non-conforming positions are counted separately by
    count_invalid_transitions().

    Input:  list of BIO2 tag strings for ONE sentence.
    Output: list of (type str, start int, end int) tuples, in order.
    """
    spans = []
    typ, start = None, None
    for i, tag in enumerate(tags):
        if tag == "O":
            if typ is not None:
                spans.append((typ, start, i))
                typ = None
            continue
        prefix, _, this_type = tag.partition("-")
        if prefix == "B" or typ is None or this_type != typ:
            # A start: either an explicit B-, or an I- that cannot continue
            # anything (see the docstring).
            if typ is not None:
                spans.append((typ, start, i))
            typ, start = this_type, i
    if typ is not None:
        spans.append((typ, start, len(tags)))
    return spans


def count_invalid_transitions(tags) -> int:
    """Count positions whose tag cannot legally follow the previous one.

    Under BIO2 the only illegal move is an "I-X" that does not follow a "B-X"
    or an "I-X". Everything else -- O after anything, B-anything after
    anything -- is legal.

    This is precisely the constraint a linear-chain CRF makes structurally
    impossible. On a softmax head it is an empirical quantity, and reporting
    it turns "we should probably add a CRF" into a number.

    Input:  list of BIO2 tag strings for ONE sentence.
    Output: int, how many positions are illegal.
    """
    bad = 0
    prev = "O"
    for tag in tags:
        if tag.startswith("I-"):
            typ = tag[2:]
            if prev not in (f"B-{typ}", f"I-{typ}"):
                bad += 1
        prev = tag
    return bad


# -----------------------------------------------------------------------------
# Streaming accumulator
# -----------------------------------------------------------------------------
class TaggingMetrics:
    """Streaming entity-level + token-level metrics.

    Usage: create once, .update() per batch, .compute() at the end -- the same
    shape as the classification projects' ConfusionMatrix, and it keeps one of
    those internally for the token-level view.

    Kept on CPU: it receives two small id tensors per batch, and the entity
    bookkeeping is python-side anyway.
    """

    def __init__(self, tags=None):
        self.tags = list(tags or config.TAGS)
        self.num_tags = len(self.tags)
        self.reset()

    def reset(self):
        """Zero every counter (start a fresh evaluation)."""
        # mat[g, p] = tokens whose true tag is g and predicted tag is p.
        self.mat = torch.zeros((self.num_tags, self.num_tags), dtype=torch.int64)
        self.tp = {t: 0 for t in config.ENTITY_TYPES}
        self.fp = {t: 0 for t in config.ENTITY_TYPES}
        self.fn = {t: 0 for t in config.ENTITY_TYPES}
        self.n_invalid = 0
        self.n_tokens = 0
        self.n_sentences = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gold: torch.Tensor, lengths: torch.Tensor):
        """Accumulate one batch.

        Input:
            pred: predicted tag ids [B, L] (i.e. logits.argmax(dim=-1), NOT
                raw logits).
            gold: true tag ids [B, L]; padded slots hold config.IGNORE_INDEX.
            lengths: true sentence lengths [B].

        Every row is sliced to its true length FIRST. This is the one place
        where getting padding wrong would silently corrupt the score rather
        than raise: a padded position carries a real predicted tag, and
        letting it through would invent entities that run off the end of the
        sentence.
        """
        pred = pred.detach().cpu()
        gold = gold.detach().cpu()
        lengths = lengths.detach().cpu()

        for row in range(pred.size(0)):
            n = int(lengths[row])
            p_ids = pred[row, :n].tolist()
            g_ids = gold[row, :n].tolist()

            p_tags = [self.tags[i] for i in p_ids]
            g_tags = [self.tags[i] for i in g_ids]

            self.n_sentences += 1
            self.n_tokens += n
            self.n_invalid += count_invalid_transitions(p_tags)

            # Sets, not lists: an exact-match score is a set intersection, and
            # a span cannot legitimately occur twice in one sentence anyway.
            p_spans = set(extract_entities(p_tags))
            g_spans = set(extract_entities(g_tags))
            for span in p_spans & g_spans:
                self.tp[span[0]] += 1
            for span in p_spans - g_spans:
                self.fp[span[0]] += 1
            for span in g_spans - p_spans:
                self.fn[span[0]] += 1

        # Token-level confusion matrix, real positions only. Encode each
        # (gold, pred) pair as one integer and histogram them with bincount.
        flat_p = pred.flatten()
        flat_g = gold.flatten()
        keep = (flat_g >= 0) & (flat_g < self.num_tags)   # drops IGNORE_INDEX
        idx = flat_g[keep] * self.num_tags + flat_p[keep]
        counts = torch.bincount(idx, minlength=self.num_tags ** 2)
        self.mat += counts.reshape(self.num_tags, self.num_tags)

    def compute(self) -> dict:
        """Reduce the counters to metrics.

        Output dict:
            f1 / precision / recall: MICRO-averaged over entities -- the
                official CoNLL score, and what "F1" means everywhere else in
                this project.
            macro_f1: unweighted mean of the four per-type F1s.
            per_type: {type: {"precision","recall","f1","support","predicted"}}
            token_accuracy: fraction of real tokens tagged correctly.
            all_o_accuracy: what predicting "O" everywhere would have scored
                on this split -- the number token_accuracy has to be read
                against.
            n_gold / n_pred: entity counts. Their ratio is the fastest read on
                whether a model is being timid.
            invalid_transitions / invalid_rate: illegal I- positions emitted.
            matrix: the token-level K x K counts (rows = true).
        """
        def prf(tp, fp, fn):
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f = 2 * p * r / max(p + r, 1e-12)
            return p, r, f

        per_type = {}
        for t in config.ENTITY_TYPES:
            p, r, f = prf(self.tp[t], self.fp[t], self.fn[t])
            per_type[t] = {
                "precision": p, "recall": r, "f1": f,
                "support": self.tp[t] + self.fn[t],        # gold entities
                "predicted": self.tp[t] + self.fp[t],      # predicted entities
            }

        tp = sum(self.tp.values())
        fp = sum(self.fp.values())
        fn = sum(self.fn.values())
        micro_p, micro_r, micro_f = prf(tp, fp, fn)

        mat = self.mat
        total = int(mat.sum())
        correct = int(mat.diag().sum())
        # Row 0 of the matrix is the gold "O" row, so its sum is how many
        # tokens an all-O prediction would have got right.
        n_gold_o = int(mat[config.TAG2ID["O"]].sum())

        return {
            "f1": micro_f, "precision": micro_p, "recall": micro_r,
            "macro_f1": sum(v["f1"] for v in per_type.values()) / len(per_type),
            "per_type": per_type,
            "token_accuracy": correct / max(total, 1),
            "all_o_accuracy": n_gold_o / max(total, 1),
            "n_gold": tp + fn,
            "n_pred": tp + fp,
            "n_sentences": self.n_sentences,
            "invalid_transitions": self.n_invalid,
            "invalid_rate": self.n_invalid / max(self.n_tokens, 1),
            "matrix": mat.tolist(),
        }


def print_report(result: dict, tags=None):
    """Print the per-type table, the headline scores and the diagnostics."""
    names = list(tags or config.TAGS)

    print(f"\n{'type':<8} {'prec':>7} {'recall':>7} {'f1':>7} "
          f"{'gold':>7} {'pred':>7}")
    print("-" * 47)
    for t in config.ENTITY_TYPES:
        row = result["per_type"][t]
        print(f"{t:<8} {row['precision']:7.4f} {row['recall']:7.4f} "
              f"{row['f1']:7.4f} {row['support']:7d} {row['predicted']:7d}")
    print("-" * 47)
    print(f"{'MICRO':<8} {result['precision']:7.4f} {result['recall']:7.4f} "
          f"{result['f1']:7.4f} {result['n_gold']:7d} {result['n_pred']:7d}"
          f"   <- the reported score")
    print(f"{'macro':<8} {'':>7} {'':>7} {result['macro_f1']:7.4f}")

    # The two numbers that keep the headline honest.
    print(f"\ntoken accuracy      {result['token_accuracy']:.4f}   "
          f"(predicting O everywhere would score "
          f"{result['all_o_accuracy']:.4f} -- this is why F1 is the metric)")
    ratio = result["n_pred"] / max(result["n_gold"], 1)
    verdict = ("balanced" if result["n_pred"] == result["n_gold"]
               else "under-predicting" if ratio < 1 else "over-predicting")
    print(f"entities pred/gold  {result['n_pred']}/{result['n_gold']} "
          f"= {ratio:.3f}   ({verdict})")
    print(f"invalid I- tags     {result['invalid_transitions']} "
          f"({result['invalid_rate']:.4%} of tokens) -- a CRF would make "
          f"these impossible")

    # Token-level confusion matrix, rows = truth, cols = prediction.
    print("\ntoken confusion matrix (rows = true, cols = predicted)")
    print(" " * 8 + "".join(f"{n:>8}" for n in names))
    for name, row in zip(names, result["matrix"]):
        print(f"{name:<8}" + "".join(f"{v:>8d}" for v in row))


@torch.no_grad()
def evaluate_tagger(model, loader, device, max_batches=None,
                    verbose: bool = True) -> dict:
    """Run the model over `loader` and compute the tagging metrics.

    Input:
        model:  tagger returning logits [B, L, K] from (ids, cases, lengths).
        loader: yields (ids, cases, lengths, labels) from collate_batch.
        device: torch device.
        max_batches: if set, stop after this many batches (quick check).
        verbose: print the full report.

    Output:
        the TaggingMetrics.compute() dict.
    """
    model.eval()
    metrics = TaggingMetrics()

    for i, (ids, cases, lengths, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = ids.to(device, non_blocking=True)
        cases = cases.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        logits = model(ids, cases, lengths)
        metrics.update(logits.argmax(dim=-1), labels, lengths)

    result = metrics.compute()
    if verbose:
        print_report(result)
    return result


# ---- Quick self-test: run this file directly (hand-checkable numbers) --------
# python utils/metrics.py
if __name__ == "__main__":
    print("span extraction:")
    cases = [
        (["B-PER", "I-PER", "O", "B-LOC"], [("PER", 0, 2), ("LOC", 3, 4)],
         "two entities, one of them two tokens long"),
        (["B-PER", "B-PER"], [("PER", 0, 1), ("PER", 1, 2)],
         "adjacent same-type entities stay separate -- this is why BIO2 exists"),
        (["O", "I-PER"], [("PER", 1, 2)],
         "bare I- is read as a start (conlleval behaviour)"),
        (["B-PER", "I-LOC"], [("PER", 0, 1), ("LOC", 1, 2)],
         "a type change ends the span"),
        (["O", "O"], [], "nothing"),
    ]
    for tags, want, why in cases:
        got = extract_entities(tags)
        print(f"  {'OK  ' if got == want else 'FAIL'} {tags} -> {got}   ({why})")

    print("\ninvalid transitions:")
    for tags, want in [(["O", "I-PER"], 1), (["B-PER", "I-PER"], 0),
                       (["I-PER"], 1), (["B-PER", "I-LOC"], 1),
                       (["B-PER", "I-PER", "I-PER"], 0), (["O", "B-LOC"], 0)]:
        got = count_invalid_transitions(tags)
        print(f"  {'OK  ' if got == want else 'FAIL'} {tags} -> {got} "
              f"(expected {want})")

    # A hand-checkable end-to-end scoring example.
    #   sentence 1  gold: [Mark Jones]PER  ate  in [Berlin]LOC
    #               pred: [Mark]PER       ate  in [Berlin]LOC
    #   sentence 2  gold: [EU]ORG rejects
    #               pred: [EU]LOC rejects
    print("\nend-to-end scoring on 2 hand-built sentences:")
    T = config.TAG2ID
    gold = torch.tensor([
        [T["B-PER"], T["I-PER"], T["O"], T["O"], T["B-LOC"]],
        [T["B-ORG"], T["O"], config.IGNORE_INDEX, config.IGNORE_INDEX,
         config.IGNORE_INDEX],
    ])
    pred = torch.tensor([
        [T["B-PER"], T["O"], T["O"], T["O"], T["B-LOC"]],
        [T["B-LOC"], T["O"], T["O"], T["O"], T["O"]],   # junk in the pad slots
    ])
    lengths = torch.tensor([5, 2])

    m = TaggingMetrics()
    m.update(pred, gold, lengths)
    res = m.compute()
    print_report(res)
    print("\nby hand: gold entities = PER(0,2), LOC(4,5), ORG(0,1) -> 3")
    print("         pred entities = PER(0,1), LOC(4,5), LOC(0,1) -> 3")
    print("         exact matches = LOC(4,5) only                -> TP 1")
    print("         so micro P = R = F1 = 1/3 = 0.3333")
    print("         note the boundary miss on 'Mark Jones' scored FP *and* FN,")
    print("         and the padded junk in sentence 2 was correctly ignored")
