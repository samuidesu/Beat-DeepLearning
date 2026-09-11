"""Visualization for sequence labeling: bracketed spans + a 9x9 heatmap.

A classifier's prediction fits in one line with a probability bar, which is
what the classification projects' viz.py drew. A tagger's prediction is a
whole annotated sentence, and the thing worth looking at is not a confidence
number but WHERE the spans landed. So "visualization" here means:

  1. format_tagged() -- a sentence with its entities bracketed inline:
         the [European Commission]ORG said on Thursday
     This is the fastest way to see what a model is doing, and unlike an F1
     number it shows the failure MODE: boundary drift, type confusion, or
     silence.

  2. format_diff() -- gold and prediction side by side with the differing
     tokens marked. Reading twenty of these tells you more than the score
     does; on this corpus the recurring stories are ORG-vs-LOC on country
     names in sports results, and MISC on nationality adjectives.

  3. plot_confusion_matrix() -- the token-level 9x9 counts as a heatmap.
     Bigger than the classification projects' 2x2 or 4x4, and read
     differently: the interesting mass is OFF the diagonal but INSIDE a type's
     B-/I- pair (a boundary problem) versus ACROSS types (a type problem).
     Those are different bugs with different fixes, and the block structure
     separates them at a glance -- which is why config.TAGS orders each type's
     B- and I- adjacently.
"""

import os

import numpy as np


def format_tagged(tokens, tags, max_chars: int = 100) -> str:
    """Render a tagged sentence with its entities bracketed inline.

    Input:
        tokens: list of token strings (case preserved).
        tags: list of BIO2 tag strings, same length.
        max_chars: truncate the rendered line so console output stays aligned.
    Output:
        a string like "the [European Commission]ORG said on Thursday".

    Entities are closed on a B-, on an O, on a type change, and at the end of
    the sentence -- the same four conditions extract_entities() uses, so the
    brackets always agree with what the scorer counted.
    """
    out, open_type = [], None

    def close():
        if open_type is not None:
            out.append(f"]{open_type}")

    for token, tag in zip(tokens, tags):
        if tag == "O":
            close()
            open_type = None
            out.append(" " + token)
            continue
        prefix, _, typ = tag.partition("-")
        if prefix == "B" or open_type != typ:
            close()
            open_type = typ
            out.append(" [" + token)
        else:
            out.append(" " + token)
    close()

    line = "".join(out).strip()
    return line if len(line) <= max_chars else line[:max_chars - 3] + "..."


def format_diff(tokens, gold, pred, max_chars: int = 100) -> str:
    """Two-line gold-vs-prediction view for one sentence.

    Input:
        tokens: list of token strings.
        gold / pred: BIO2 tag lists, same length as tokens. `gold` may be None
            for free-form text, in which case only the prediction is shown.
    Output:
        a multi-line string.

    The token-level mismatch list at the end is deliberately separate from the
    bracketed lines: the brackets show what each side EXTRACTED, the list
    shows where the tag sequences literally disagree, and those two views
    disagree in an informative way whenever a boundary error shifts a span.
    """
    if gold is None:
        return f"  pred: {format_tagged(tokens, pred, max_chars)}"

    lines = [f"  gold: {format_tagged(tokens, gold, max_chars)}",
             f"  pred: {format_tagged(tokens, pred, max_chars)}"]
    bad = [f"{t}({g}->{p})" for t, g, p in zip(tokens, gold, pred) if g != p]
    if bad:
        lines.append(f"  diff: {' '.join(bad[:8])}"
                     + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""))
    return "\n".join(lines)


def plot_confusion_matrix(matrix, class_names, path: str, title: str = None,
                          normalize: bool = True):
    """Save the token-level confusion matrix as an annotated heatmap PNG.

    Input:
        matrix: K x K nested list / array of counts (rows = true tag).
        class_names: labels for both axes (config.TAGS).
        path: output .png path.
        title: figure title (defaults to "confusion matrix").
        normalize: color cells by ROW-normalized rate (per-tag recall) while
            still printing the raw counts. Not optional in practice on this
            corpus: the "O" row holds 83% of all tokens, so an un-normalized
            heatmap is one dark square and eight invisible ones.
    """
    import matplotlib
    matplotlib.use("Agg")               # headless: no display needed
    import matplotlib.pyplot as plt

    counts = np.asarray(matrix, dtype=np.float64)
    shown = (counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
             if normalize else counts)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=max(shown.max(), 1e-9))
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "confusion matrix")

    # Annotate every cell with the raw count (and the rate when normalizing),
    # flipping the text color on dark cells so it stays readable. 9x9 = 81
    # cells still fits; the font is small on purpose.
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            label = (f"{int(counts[i, j])}\n{shown[i, j]:.0%}"
                     if normalize else f"{int(counts[i, j])}")
            ax.text(j, i, label, ha="center", va="center", fontsize=6,
                    color="white" if shown[i, j] > shown.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- Quick self-test: run this file directly --------------------------------
# python utils/viz.py
if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    tokens = ["The", "European", "Commission", "said", "German", "lamb",
              "was", "safe", ",", "Peter", "Blackburn", "reported"]
    gold = ["O", "B-ORG", "I-ORG", "O", "B-MISC", "O",
            "O", "O", "O", "B-PER", "I-PER", "O"]
    # A realistic set of mistakes: a boundary miss on the ORG, a type error on
    # the MISC, and a correctly-found PER.
    pred = ["O", "B-ORG", "O", "O", "B-LOC", "O",
            "O", "O", "O", "B-PER", "I-PER", "O"]

    print("format_tagged (gold):")
    print("  " + format_tagged(tokens, gold))
    print("\nformat_diff:")
    print(format_diff(tokens, gold, pred))
    print("\nformat_diff with no gold (free-form text):")
    print(format_diff(tokens, None, pred))

    # Adjacent same-type entities must bracket separately -- the case BIO2
    # exists for, and the one a naive renderer merges into a single span.
    print("\nadjacent same-type entities:")
    print("  " + format_tagged(["France", "Germany", "drew"],
                               ["B-LOC", "B-LOC", "O"]))
    print("  (expected two separate [..]LOC spans, not one)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_cm_selftest.png")
    mat = np.zeros((config.NUM_TAGS, config.NUM_TAGS), dtype=int)
    np.fill_diagonal(mat, [38000, 1500, 1100, 1600, 250, 1300, 800, 700, 200])
    mat[config.TAG2ID["B-ORG"]][config.TAG2ID["B-LOC"]] = 260   # the ORG/LOC bleed
    mat[config.TAG2ID["B-MISC"]][config.TAG2ID["O"]] = 180
    plot_confusion_matrix(mat.tolist(), config.TAGS, out, title="self-test")
    print(f"\nwrote {out} ({os.path.getsize(out)} bytes) -- delete it when done")
