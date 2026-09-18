"""Corpus-level MT metrics on decoded Romanian text.

TWO RULES, and both are easy to get wrong.

1. CORPUS-LEVEL, NOT AVERAGED OVER BATCHES. BLEU is a ratio of n-gram counts
   summed over the whole corpus with a single brevity penalty; the mean of
   per-batch BLEUs is a different quantity that happens to look similar. It is
   systematically wrong on short sentences (where per-sentence n-gram counts
   are tiny and the smoothing dominates) and it is not comparable to any
   published number. So: collect every hypothesis and every reference first,
   score once.

2. SCORE TEXT, NOT IDS. Metrics run on detokenized strings. sacreBLEU applies
   its own tokenization (13a) internally, which is exactly the point -- it is
   what makes two papers' numbers comparable. Feeding it SentencePiece pieces
   would produce a large, meaningless number.

WHAT IS REPORTED:
    sacreBLEU   the standard, with its signature so the setup is reproducible
    chrF++      chrF with word_order=2: character n-grams plus word bigrams.
                For a morphologically rich target like Romanian this
                correlates with human judgement better than BLEU, because it
                gives partial credit for a nearly-right inflection instead of
                zero.

COMET (a neural metric, better correlated still) is a deliberate non-
dependency: it pulls in a ~2 GB model. To add it later:
`pip install unbabel-comet`, score with `Unbabel/wmt22-comet-da`, and report
it alongside -- never instead of -- BLEU and chrF++.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from sacrebleu.metrics import BLEU, CHRF


def _validate(hypotheses: Sequence[str], references: Sequence[str]) -> None:
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses but {len(references)} references; "
            "every evaluation example must be scored exactly once."
        )
    if not hypotheses:
        raise ValueError("Nothing to score: the hypothesis list is empty.")


def compute_metrics(hypotheses: Sequence[str], references: Sequence[str],
                    chrf_word_order: int = 2) -> Dict[str, object]:
    """Corpus sacreBLEU + chrF++ over aligned hypothesis/reference lists.

    Single-reference evaluation, which is what WMT16 ro-en provides: sacrebleu
    wants the references as a list of reference STREAMS, hence [references].
    """
    _validate(hypotheses, references)
    hyps: List[str] = [h.strip() for h in hypotheses]
    refs: List[str] = [r.strip() for r in references]

    bleu = BLEU()
    bleu_score = bleu.corpus_score(hyps, [refs])
    chrf = CHRF(word_order=chrf_word_order)
    chrf_score = chrf.corpus_score(hyps, [refs])

    return {
        "bleu": bleu_score.score,
        "bleu_signature": str(bleu.get_signature()),
        "bleu_detail": bleu_score.format(),
        "chrf": chrf_score.score,
        "chrf_word_order": chrf_word_order,
        "chrf_signature": str(chrf.get_signature()),
        "num_sentences": len(hyps),
        "empty_hypotheses": sum(1 for h in hyps if not h),
    }


def format_metrics(metrics: Dict[str, object], loss: float | None = None) -> str:
    """One printable block: loss, BLEU, chrF++, and the sacreBLEU signatures."""
    name = "chrF++" if metrics.get("chrf_word_order") else "chrF"
    lines = []
    if loss is not None:
        lines.append(f"  NLL (per token) : {loss:.4f}")
    lines += [
        f"  sacreBLEU       : {metrics['bleu']:.2f}",
        f"  {name:<16}: {metrics['chrf']:.2f}",
        f"  sentences       : {metrics['num_sentences']:,}"
        + (f"  ({metrics['empty_hypotheses']} empty hypotheses)"
           if metrics["empty_hypotheses"] else ""),
        f"  BLEU signature  : {metrics['bleu_signature']}",
        f"  chrF signature  : {metrics['chrf_signature']}",
    ]
    return "\n".join(lines)
