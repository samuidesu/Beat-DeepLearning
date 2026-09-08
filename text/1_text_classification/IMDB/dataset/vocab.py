"""Tokenizer + vocabulary: the text counterpart of the image transforms.

    review    --tokenize-->  ["this", "movie", "was", "a", "waste", ...]
    tokens    --Vocab----->  [42, 21, 16, 6, 1893, ...]

The ids then index the embedding table, which is where the actual numbers
(word vectors) live. Two properties matter and both are enforced here:

  1. The vocabulary is built from the TRAINING portion only. Building it over
     val/test would let evaluation words influence training-time decisions --
     the text equivalent of computing normalization stats over the val set.
  2. The mapping must be IDENTICAL at train and inference time, or every id
     points at the wrong vector. Hence save()/load(): train.py writes
     vocab.json next to the checkpoint, eval.py and predict.py read it back.

Special tokens are pinned at fixed ids (config.PAD_IDX=0, config.UNK_IDX=1):
  <pad> fills short reviews up to the batch length and is masked out
        everywhere (padding_idx in the embedding, packing in the encoder,
        masks in the pooling);
  <unk> catches words below min_freq or never seen in training.

Unchanged from the SST-2 and AG-News projects on purpose: the tokenizer is
what decides whether a word can be found in GloVe at all, so keeping it
identical is what makes the three projects' coverage numbers comparable.
"""

import json
import os
import re
import sys
from collections import Counter

# Make the project root importable so `import config` works whether this file
# is run directly (python dataset/vocab.py) or imported as dataset.vocab.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402

# One token = a run of letters/digits/apostrophes, OR a single other non-space
# character (so punctuation becomes its own token instead of sticking to the
# previous word).
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list:
    """Split a document into lowercase tokens.

IMDB is RAW user-written text, not pre-tokenized the way GLUE's SST-2 tsv
    was, so the regex is doing real work on the corpus itself rather than
    acting as a safety net for predict.py. Splitting punctuation off is what
    turns "didn't" into ["didn", "'", "t"] and "so-called" into
    ["so", "-", "called"], whose pieces GloVe knows, instead of leaving
    out-of-vocabulary strings.

    One IMDB-specific consequence worth knowing: 6.0% of reviews write an
    explicit score into the text ("I'd give it 8/10"), plus another 2.7% in
    words ("8 out of 10"). This tokenizer turns the first form into
    ["8", "/", "10"] -- three ordinary tokens GloVe has vectors for. Note that
    it is the DIGIT that carries the signal, not the pattern: reviews
    containing an x/10 score are 55% positive, i.e. barely above the 50% base
    rate, so a model that merely detected "there is a score here" would learn
    nothing.

    GloVe 6B is lowercase-only, so lowercasing here is not just normalization:
    it is what makes a word findable in the pretrained table at all.

    Input:  raw text string.
    Output: list of token strings.
    """
    return _TOKEN_RE.findall(text.lower())


def truncate(ids, max_len=None, mode=None, head_len=None) -> list:
    """Cut a sequence down to `max_len` ids. See config.TRUNCATION for why.

        "head"       ids[:max_len]
        "head_tail"  ids[:head_len] + ids[-(max_len - head_len):]

    A module-level function, not a Vocab method, because BOTH encoding paths
    have to make the identical cut and they reach it differently: IMDBDataset
    encodes first (it needs the pre-truncation length for its statistics) and
    truncates after, while predict.py truncates inside encode(). Two copies of
    this arithmetic is exactly how train/serve skew gets in.

    Input:  ids list, max_len (None/0 = no truncation), mode + head_len
            (None = take config's).
    Output: a list of at most max_len ids (the input itself when short enough).
    """
    if not max_len or len(ids) <= max_len:
        return ids
    mode = mode or config.TRUNCATION
    if mode == "head":
        return ids[:max_len]
    if mode != "head_tail":
        raise ValueError(f"unknown truncation mode {mode!r}")
    head = config.HEAD_LEN if head_len is None else head_len
    head = max(0, min(head, max_len))
    tail = max_len - head
    # ids[-0:] is the WHOLE list, not the empty one -- so head_len == max_len
    # has to be handled explicitly rather than falling out of the slice.
    return ids[:head] + (ids[-tail:] if tail else [])


class Vocab:
    """Bidirectional token <-> id mapping with frequency-based pruning.

    Attributes:
        itos: list of tokens, indexed by id (id -> string).
        stoi: dict token -> id (string -> id).
    """

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    # ---- construction --------------------------------------------------
    @classmethod
    def build(cls, token_lists, min_freq=1, max_size=None,
              specials=(config.PAD_TOKEN, config.UNK_TOKEN)):
        """Build a vocabulary from an iterable of token lists.

        Input:
            token_lists: iterable of lists of tokens (the TRAIN portion).
            min_freq: drop tokens appearing fewer than this many times. They
                become <unk> at lookup time, which is exactly how unseen test
                words behave -- so min_freq > 1 also TEACHES the model what
                <unk> looks like instead of leaving that embedding untrained.
            max_size: cap on the number of non-special tokens (None = no cap).
            specials: tokens forced to the front, in order, so <pad> gets id 0
                and <unk> id 1 (config.PAD_IDX / UNK_IDX).

        Output:
            a Vocab instance.
        """
        freqs = Counter()
        for tokens in token_lists:
            freqs.update(tokens)

        # Sort by frequency (descending), ties broken alphabetically so the
        # vocabulary -- and therefore every id -- is fully deterministic.
        ordered = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [tok for tok, n in ordered if n >= min_freq]
        if max_size is not None:
            kept = kept[:max_size]
        return cls(list(specials) + kept)

    # ---- lookup --------------------------------------------------------
    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def encode(self, tokens, max_len=None, mode=None, head_len=None) -> list:
        """Map tokens to ids, unknown -> UNK, then truncate to `max_len`.

        The cut defaults to config.TRUNCATION ("head_tail" on this project);
        see truncate() above. mode/head_len exist so eval.py and predict.py
        can pass what the CHECKPOINT was trained with instead of what config
        happens to say today.
        """
        ids = [self.stoi.get(tok, config.UNK_IDX) for tok in tokens]
        return truncate(ids, max_len, mode, head_len)

    def decode(self, ids) -> list:
        """Map ids back to token strings (for debugging / printing)."""
        return [self.itos[i] if 0 <= i < len(self.itos) else config.UNK_TOKEN
                for i in ids]

    # ---- persistence ---------------------------------------------------
    def save(self, path: str):
        """Write the vocabulary as json (itos is enough to rebuild stoi)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        """Read back a vocabulary written by save()."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["itos"])


# ---- Quick self-test: run this file directly --------------------------------
# python dataset/vocab.py
if __name__ == "__main__":
    raw = "I didn't hate it, but this so-called classic is a 4/10 at best."
    print("tokenize:", tokenize(raw))

    corpus = [tokenize(s) for s in [
        "This movie was a waste of time",
        "This movie was a waste of talent",
        "Truly wonderful performances",
    ]]
    v = Vocab.build(corpus, min_freq=1)
    print("size:", len(v), "(expected 2 specials + 11 types = 13)")
    print("itos[:6]:", v.itos[:6], "(pad, unk, then most frequent first)")
    print("pad id:", v.stoi[config.PAD_TOKEN], "unk id:", v.stoi[config.UNK_TOKEN],
          "(expected 0 and 1)")

    ids = v.encode(tokenize("this movie was dreadful"), max_len=10)
    print("encode:", ids)
    print("decode:", v.decode(ids), "(unseen words -> <unk>)")

    # Truncation, on a sequence short enough to read: keep 6 of 10.
    seq = list(range(10))
    print("\nhead      :", truncate(seq, 6, "head"), "(expected 0..5)")
    print("head_tail :", truncate(seq, 6, "head_tail", head_len=4),
          "(expected 0..3 then 8,9 -- the middle is dropped)")
    print("no cut    :", truncate(seq, 20, "head_tail", head_len=4),
          "(shorter than max_len, returned untouched)")
    print("head_len=max_len:", truncate(seq, 6, "head_tail", head_len=6),
          "(degenerates to head, must not return all 10)")

    # min_freq=2 prunes everything seen once: only the words shared by the
    # first two sentences survive (this/movie/was/a/waste/of).
    v2 = Vocab.build(corpus, min_freq=2)
    print("min_freq=2 size:", len(v2), "->", v2.itos)
