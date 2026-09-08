"""Visualization for text classification: readable predictions + a heatmap.

Segmentation could paint its output straight onto the image; a sentiment
prediction has no picture, so "visualization" here means two much smaller
things that are still the fastest way to see what the model is doing:

  1. format_prediction() -- one console line per review with a probability
     bar, the predicted class, and (when the gold label is known) a hit/miss
     marker. Reading 20 of these tells you more about a model than the
     accuracy number does: you see whether the mistakes are near 0.5
     (genuinely ambiguous) or confident and wrong (a real failure).
  2. plot_confusion_matrix() -- the 2x2 counts as a labeled heatmap, so the
     asymmetry between negative and positive recall is visible at a glance.

Taken from the SST-2 project rather than the AG-News one, because the
probability bar is the part that differs: with two classes there IS a single
decision boundary at 0.5, so the bar is drawn centered on it and its direction
carries the answer. AG News' four classes have no such boundary and its bar
just reports the winner's confidence. One addition for IMDB: reviews contain
newlines, so the text is whitespace-collapsed before it is truncated to one
console line.
"""

import os

import numpy as np


def prob_bar(p: float, width: int = 20) -> str:
    """Render a probability in [0, 1] as a fixed-width text bar.

    The bar is centered on 0.5 (the decision boundary): the left half fills
    towards "negative", the right half towards "positive", so a glance at the
    column tells you both the class AND how close the call was.
    """
    p = min(max(float(p), 0.0), 1.0)
    half = width // 2
    if p >= 0.5:
        filled = round((p - 0.5) * 2 * half)
        return " " * half + "|" + "#" * filled + " " * (half - filled)
    filled = round((0.5 - p) * 2 * half)
    return " " * (half - filled) + "#" * filled + "|" + " " * half


def format_prediction(sentence: str, probs, class_names, gold: int = None,
                      max_chars: int = 70) -> str:
    """One-line summary of a single prediction.

    Input:
        sentence: the raw text (a whole review here -- most of it will be cut).
        probs: per-class probabilities (list/tensor of length C).
        class_names: e.g. ["negative", "positive"].
        gold: true class id, or None when unknown (free-form user text).
        max_chars: truncate the text so the columns stay aligned.
    Output:
        a formatted string, e.g.
        "[positive 0.93]      |######### OK  a charming, funny film"
    """
    probs = [float(p) for p in probs]
    pred = max(range(len(probs)), key=probs.__getitem__)
    # The bar is drawn from the POSITIVE probability (index 1) for the binary
    # case; for more classes it just shows the winning class's confidence.
    p_pos = probs[1] if len(probs) == 2 else probs[pred]

    mark = "   "
    if gold is not None:
        mark = "OK " if pred == gold else "MISS"

    # Collapse whitespace FIRST: IMDB reviews are multi-paragraph, and a raw
    # newline would break the one-line-per-prediction layout entirely.
    sentence = " ".join(sentence.split())
    text = sentence if len(sentence) <= max_chars else sentence[:max_chars - 3] + "..."
    return (f"[{class_names[pred]:<8} {probs[pred]:.2f}] "
            f"{prob_bar(p_pos)} {mark} {text}")


def plot_confusion_matrix(matrix, class_names, path: str, title: str = None,
                          normalize: bool = True):
    """Save the confusion matrix as an annotated heatmap PNG.

    Input:
        matrix: K x K nested list / array of counts (rows = true class).
        class_names: labels for both axes.
        path: output .png path.
        title: figure title (defaults to "confusion matrix").
        normalize: color cells by ROW-normalized rate (per-class recall) while
            still printing the raw counts. Without it a class with more
            samples simply looks darker, which says nothing about quality.
    """
    import matplotlib
    matplotlib.use("Agg")               # headless: no display needed
    import matplotlib.pyplot as plt

    counts = np.asarray(matrix, dtype=np.float64)
    shown = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None) if normalize else counts

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max())
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "confusion matrix")

    # Annotate every cell with the raw count (and the rate when normalizing),
    # flipping the text color on dark cells so it stays readable.
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            label = (f"{int(counts[i, j])}\n{shown[i, j]:.1%}"
                     if normalize else f"{int(counts[i, j])}")
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if shown[i, j] > shown.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- Quick self-test: run this file directly --------------------------------
# python utils/viz.py
if __name__ == "__main__":
    names = ["negative", "positive"]
    print(format_prediction("A charming, funny film with real heart.",
                            [0.07, 0.93], names, gold=1))
    print(format_prediction("Dull, pointless, and forty minutes too long.",
                            [0.96, 0.04], names, gold=0))
    print(format_prediction("It has its moments, but never comes together.",
                            [0.52, 0.48], names, gold=1))
    print(format_prediction("Line one of a review.\n\nLine two, after a blank line.",
                            [0.30, 0.70], names))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cm_selftest.png")
    plot_confusion_matrix([[10850, 1650], [1150, 11350]], names, out,
                          title="self-test")
    print(f"\nwrote {out} ({os.path.getsize(out)} bytes) -- delete it when done")
