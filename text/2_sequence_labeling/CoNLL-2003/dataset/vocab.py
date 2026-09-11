"""Vocabulary, casing feature, and the tokenizer used only for free-form text.

    tokens  --Vocab-->      [42, 21, 16, 6, 1893, ...]     word ids
    tokens  --case_ids-->   [ 2,  0,  0, 0,    1, ...]     casing ids

Two id streams instead of the classification projects' one, and the second is
the whole reason this file differs from its IMDB counterpart.

WHY TWO STREAMS. GloVe 6B is lowercase-only, so a word is findable in the
pretrained table only after lowercasing. On sentiment that was free -- "Awful"
and "awful" mean the same thing. On NER it is not free at all: capitalization
is the single strongest surface cue for an entity, and lowercasing deletes it
before the model ever sees it. So the lowercasing stays (the word ids need
it), and the casing is recovered separately as a 6-way categorical feature
that gets its own embedding. See config.CASE_CLASSES.

A useful way to see the size of what would otherwise be thrown away: in
"Turkey said it would ban imports" versus "turkey said it would ban imports",
the word ids are IDENTICAL. Only the case stream separates a country from a
bird.

WHAT IS NOT HERE. There is no truncate() -- IMDB needed one because reviews
ran to 2,737 tokens and the head/tail choice was a real experiment. CoNLL
sentences top out around 113 tokens against config.MAX_LEN = 128, so nothing
is ever cut; and if it were, the cut would have to fall on the TOKENS AND THE
TAGS TOGETHER, which makes it the Dataset's job (see conll2003.py), not the
encoder's. Splitting that arithmetic across two files is exactly how a tagger
ends up trained against shifted labels.

The mapping must be IDENTICAL at train and inference time or every id points
at the wrong vector, hence save()/load(): train.py writes vocab.json next to
the checkpoint, eval.py and predict.py read it back.
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

# One token = a run of letters/digits/apostrophes/hyphens, OR a single other
# non-space character. NOTE THE ABSENCE OF re.IGNORECASE AND OF .lower():
# unlike the classification projects' tokenizer, this one preserves case,
# because case is a feature here rather than noise.
#
# The hyphen is inside the word class, which the classification tokenizers did
# not do. That is to match how CoNLL-2003 itself tokenizes: it keeps
# "1996-08-22" and "German-based" as single tokens, and predict.py must
# produce the same shape of token the model was trained on.
_TOKEN_RE = re.compile(r"[A-Za-z0-9'\-]+|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list:
    """Split raw text into CASE-PRESERVING tokens.

    Used ONLY by predict.py, for text a user types or pastes. The corpus
    itself arrives pre-tokenized -- CoNLL-2003 ships one token per line, with
    the tokenization already decided by the annotators -- so this regex never
    touches the training data. That is a genuine difference from IMDB, where
    the tokenizer ran over the corpus and its choices shaped the whole
    vocabulary.

    Input:  raw text string.
    Output: list of token strings, case preserved.
    """
    return _TOKEN_RE.findall(text)


def normalize(token: str) -> str:
    """Map a token to the form used for the WORD-ID lookup.

    Lowercasing only. Digit normalization ("1996" -> "0000") is the other
    classic NER step and is deliberately skipped -- see the note in config.py
    for the measurement behind that.
    """
    return token.lower()


def case_class(token: str) -> str:
    """Classify a token's capitalization into one of config.CASE_CLASSES.

    Order matters, and the first two tests are the non-obvious ones:

      * DIGIT FIRST. "1996-08-22" contains no letters to reason about, and
        "22nd" would otherwise be read as an ordinary lowercase word. Anything
        containing a digit is a digit token.
      * THEN NON-ALPHABETIC. Punctuation has no case, and letting "." fall
        through to islower() (which returns False for it) would file it as
        "mixed" alongside "eBay".

    A lone capital ("J" in "J. Smith") satisfies str.isupper() and is filed as
    "upper" rather than "title". That is deliberate: single-letter initials
    behave much more like the all-caps dateline tokens than like "Germany".

    Input:  one token string, case preserved.
    Output: one of config.CASE_CLASSES.
    """
    if not token:
        return "other"
    if any(c.isdigit() for c in token):
        return "digit"
    if not any(c.isalpha() for c in token):
        return "other"
    if token.islower():
        return "lower"
    if token.isupper():
        return "upper"
    if token[0].isupper() and token[1:].islower():
        return "title"
    return "mixed"


def case_ids(tokens) -> list:
    """Map a list of CASE-PRESERVING tokens to config.CASE2ID ids."""
    return [config.CASE2ID[case_class(t)] for t in tokens]


class Vocab:
    """Bidirectional token <-> id mapping with frequency-based pruning.

    Holds LOWERCASED types only -- the casing lives in the parallel stream
    from case_ids(), never in this table. Keeping them separate is what lets
    one 100-dim GloVe row serve "Turkey", "TURKEY" and "turkey" while the
    model still tells them apart.

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
        """Build a vocabulary from an iterable of token lists (the TRAIN split).

        Applies normalize() itself, so callers pass the raw cased tokens and
        cannot accidentally build a cased table that GloVe can never match.

        Input:
            token_lists: iterable of lists of cased tokens.
            min_freq: drop types appearing fewer than this many times.
                config.MIN_FREQ is 1 here -- see the note there for why NER
                wants the opposite setting from sentiment.
            max_size: cap on the number of non-special types (None = no cap).
            specials: tokens forced to the front, in order, so <pad> gets id 0
                and <unk> id 1 (config.PAD_IDX / UNK_IDX).

        Output:
            a Vocab instance.
        """
        freqs = Counter()
        for tokens in token_lists:
            freqs.update(normalize(t) for t in tokens)

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
        return normalize(token) in self.stoi

    def encode(self, tokens) -> list:
        """Map cased tokens to word ids (unknown -> UNK), normalizing first.

        No max_len parameter, on purpose: cutting a sequence-labeling input
        has to cut the TAGS in the same place, so the Dataset owns that.
        """
        return [self.stoi.get(normalize(t), config.UNK_IDX) for t in tokens]

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
    raw = "The German-based EU office in BRUSSELS closed on 1996-08-22, McDonald said."
    toks = tokenize(raw)
    print("tokenize:", toks)
    print("  (case preserved, and 'German-based' / '1996-08-22' stay whole)")

    print("\ncase classes:")
    probe = ["exports", "Germany", "EU", "BRUSSELS", "McDonald", "eBay",
             "1996", "1996-08-22", "22nd", ".", "J", "U.S"]
    for t in probe:
        print(f"  {t:<12} -> {case_class(t)}")
    print("expected: lower title upper upper mixed mixed digit digit digit "
          "other upper upper")
    print("  ('U.S' is upper, not mixed: str.isupper() ignores the '.', which "
          "is the answer we want)")

    # The point of the whole file, in two lines: identical word ids, different
    # case ids. A model without the case stream literally cannot tell these
    # apart.
    corpus = [tokenize("Turkey said it would ban imports"),
              tokenize("Rice and turkey are cheap"),
              tokenize("Condoleezza Rice said nothing")]
    v = Vocab.build(corpus, min_freq=1)
    a, b = tokenize("Turkey said"), tokenize("turkey said")
    print(f"\nword ids  'Turkey said' -> {v.encode(a)}")
    print(f"word ids  'turkey said' -> {v.encode(b)}  (IDENTICAL -- lowercased)")
    print(f"case ids  'Turkey said' -> {case_ids(a)}")
    print(f"case ids  'turkey said' -> {case_ids(b)}  (DIFFERENT -- the signal)")

    print(f"\nvocab size: {len(v)} (2 specials + the lowercased types)")
    print("itos[:6]:", v.itos[:6], "(pad, unk, then most frequent first)")
    print("pad id:", v.stoi[config.PAD_TOKEN], "unk id:", v.stoi[config.UNK_TOKEN],
          "(expected 0 and 1)")
    print("'RICE' in vocab:", "RICE" in v, "(expected True -- matched via lowercase)")

    ids = v.encode(tokenize("Turkey banned Klingon imports"))
    print("\nencode:", ids)
    print("decode:", v.decode(ids), "(unseen words -> <unk>)")
