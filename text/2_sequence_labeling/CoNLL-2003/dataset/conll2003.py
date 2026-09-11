"""CoNLL-2003 English NER: download, parsing, IOB1->BIO2, Dataset, collate_fn.

Five responsibilities, the same shape as the classification projects'
dataset modules, with one extra step that only sequence labeling needs:

  1. Download the three split files into config.CONLL_DIR.
  2. Parse the column format into sentences of (tokens, tags).
  3. CONVERT THE TAG ENCODING from IOB1 to BIO2.        <- the extra step
  4. Wrap a split in a Dataset yielding (ids, case_ids, length, labels).
  5. Provide the COLLATE FUNCTION, which now pads TWO id streams and a LABEL
     stream, with different fill values for each.

THE FILE FORMAT. One token per line, blank line between sentences, four
whitespace-separated columns -- token, POS tag, chunk tag, NER tag:

    EU       NNP  I-NP  I-ORG
    rejects  VBZ  I-VP  O
    German   JJ   I-NP  I-MISC
    call     NN   I-NP  O
    .        .    O     O

Only columns 0 and -1 are read. The POS and chunk columns are gold
annotations that a real system would not have at inference time, and using
them would make the numbers incomparable with everything published since
about 2011.

Some mirrors additionally carry `-DOCSTART- -X- O O` lines marking article
boundaries; others strip them. Both are handled: -DOCSTART- is treated as a
sentence separator and never becomes a training example. It is not a sentence
and tagging it would be free, misleading accuracy.

THE TAG ENCODING, which is the one trap in this corpus. The released files are
IOB1, NOT the BIO2 that every modern paper reports. In IOB1 an entity normally
STARTS with I-, and B- appears only to separate two adjacent entities of the
same type:

    IOB1 (the file)      BIO2 (this project)   what it means
    EU       I-ORG       B-ORG                 entity starts here
    German   I-MISC      B-MISC                entity starts here
    European I-ORG       B-ORG                 entity starts here
    Commission I-ORG     I-ORG                 ...and continues here

Training directly on IOB1 makes the first token of every entity ambiguous with
its own continuation -- the same symbol means both -- which is a real accuracy
cost and, worse, silently changes what the entity-F1 score means. iob1_to_bio2
below does the conversion, and the published entity counts are re-checked
afterwards, which is what proves the conversion did not invent or lose spans.

LICENSING. The NER annotations are freely redistributable and are what the
mirrors below serve. The underlying Reuters RCV1 newswire text is owned by
Reuters and its official distribution requires an agreement with NIST; the
mirrors bundle the words with the annotations, as essentially all public
reproductions of this benchmark do. That is fine for local study -- which is
all this repo does -- and worth knowing before redistributing anything.

How to download: python dataset/conll2003.py --download
(train.py also downloads automatically when the corpus is missing.)
"""

import argparse
import os
import sys

import torch
from torch.utils.data import Dataset

# Make the project root importable (works both as a script and as a package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.download import download_with_mirrors  # noqa: E402

try:
    from .vocab import Vocab, case_ids, tokenize
except ImportError:  # running this file directly
    from vocab import Vocab, case_ids, tokenize

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
# The corpus ships as three plain text files under their original shared-task
# names. Both mirrors were checked to serve the real files; they differ only
# in whether -DOCSTART- lines survived, which the parser handles.
_SPLIT_FILES = {"train": "eng.train", "valid": "eng.testa", "test": "eng.testb"}
_MIRRORS = [
    "https://raw.githubusercontent.com/glample/tagger/master/dataset/{name}",
    "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/{name}",
]

# Expected counts, used INSTEAD of an md5: the mirrors disagree byte-for-byte
# (-DOCSTART-, line endings) but must agree on every number below. A mismatch
# means a truncated download, a re-tokenized mirror, or a broken IOB1->BIO2
# conversion -- and the entity counts in particular are what make this a real
# check on the conversion rather than just on the bytes.
#
# TOKENS AND ENTITIES are Tjong Kim Sang & De Meulder (2003) Table 2 verbatim.
#
# SENTENCES ARE NOT, and the discrepancy is worth writing down because it costs
# everyone an afternoon exactly once. Table 2 reports 14,987 / 3,466 / 3,684,
# which count each `-DOCSTART-` line as a sentence of its own. Subtracting the
# paper's own article counts (946 / 216 / 231) gives the numbers below, which
# are the REAL sentence counts and the ones every modern source reports
# (HuggingFace's conll2003 card included):
#
#     split   Table 2   articles   real sentences
#     train    14,987       946        14,041
#     valid     3,466       216         3,250
#     test      3,684       231         3,453
#
# Note the paper is internally inconsistent here: its TOKEN counts do not
# include the -DOCSTART- tokens, while its sentence counts do include the
# -DOCSTART- pseudo-sentences. This loader excludes -DOCSTART- from both,
# which is why tokens match Table 2 exactly and sentences do not.
_EXPECTED = {
    "train": {"sentences": 14041, "tokens": 203621,
              "entities": {"LOC": 7140, "MISC": 3438, "ORG": 6321, "PER": 6600}},
    "valid": {"sentences": 3250, "tokens": 51362,
              "entities": {"LOC": 1837, "MISC": 922, "ORG": 1341, "PER": 1842}},
    "test": {"sentences": 3453, "tokens": 46435,
             "entities": {"LOC": 1668, "MISC": 702, "ORG": 1661, "PER": 1617}},
}


def conll_present() -> bool:
    """True when all three split files exist under config.CONLL_DIR."""
    return all(os.path.isfile(os.path.join(config.CONLL_DIR, f"{s}.txt"))
               for s in _SPLIT_FILES)


def download_conll():
    """Fetch the three CoNLL-2003 splits into config.CONLL_DIR.

    Idempotent: returns immediately when all three files are already there.
    Saved under this project's split names (train/valid/test.txt) rather than
    the shared task's eng.train/eng.testa/eng.testb, because "testa" being the
    DEV set is exactly the kind of naming that produces an accidental
    select-on-test.
    """
    if conll_present():
        print(f"CoNLL-2003 already present at {config.CONLL_DIR}")
        return
    os.makedirs(config.CONLL_DIR, exist_ok=True)
    print("Downloading CoNLL-2003 (~4.8 MB)...")
    for split, name in _SPLIT_FILES.items():
        dest = os.path.join(config.CONLL_DIR, f"{split}.txt")
        if os.path.isfile(dest):
            continue
        print(f"  {name} -> {split}.txt")
        # connections=4: these are 1-3 MB files, and raw.githubusercontent
        # serves them fast enough that 16 range requests is pure overhead.
        download_with_mirrors([m.format(name=name) for m in _MIRRORS], dest,
                              md5=None, connections=4)
    if not conll_present():
        raise RuntimeError(f"CoNLL-2003 files missing from {config.CONLL_DIR}")


# -----------------------------------------------------------------------------
# Tag encoding
# -----------------------------------------------------------------------------
def iob1_to_bio2(tags) -> list:
    """Convert one sentence's tags from IOB1 to BIO2.

    The rule is short: an "I-X" that does not CONTINUE an X becomes "B-X".
    It does not continue an X when it is the first tag of the sentence, when
    the previous tag is "O", or when the previous tag has a different type.
    A tag that is already "B-X" stays "B-X" (IOB1 emits those only at a
    genuine boundary, which is a boundary under BIO2 too), and "O" is "O".

    Applied PER SENTENCE, which is why it takes a sentence rather than the
    whole corpus: the previous-tag test must not reach across a sentence
    break, or the first entity of a sentence would be read as a continuation
    of the last entity of the one before it.

    Input:  list of IOB1 tag strings.
    Output: list of BIO2 tag strings, same length.
    """
    out = []
    for i, tag in enumerate(tags):
        if tag == "O":
            out.append("O")
            continue
        prefix, _, typ = tag.partition("-")
        if not typ:
            raise ValueError(f"malformed tag {tag!r}")
        if prefix == "B":
            out.append(f"B-{typ}")
            continue
        if prefix != "I":
            raise ValueError(f"unknown tag prefix in {tag!r}")
        prev = tags[i - 1] if i > 0 else "O"
        prev_type = prev.partition("-")[2]      # "" for "O"
        out.append(f"I-{typ}" if prev_type == typ else f"B-{typ}")
    return out


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
def read_split(split: str):
    """Read one split into a list of (tokens, BIO2 tags) sentence pairs.

    Input:
        split: "train" / "valid" / "test".
    Output:
        list of (list[str] cased tokens, list[str] BIO2 tags), equal lengths.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; choose one of {list(_SPLIT_FILES)}")
    path = os.path.join(config.CONLL_DIR, f"{split}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python dataset/conll2003.py --download`")

    sentences = []
    tokens, tags = [], []

    def flush():
        if tokens:
            sentences.append((list(tokens), iob1_to_bio2(tags)))
            tokens.clear()
            tags.clear()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                flush()
                continue
            parts = line.split()
            # An article boundary, present in some mirrors and stripped from
            # others. It separates sentences and is never an example itself.
            if parts[0] == "-DOCSTART-":
                flush()
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])          # column -1 = NER; POS/chunk unused
    flush()

    _check_counts(split, sentences)
    return sentences


def _check_counts(split: str, sentences):
    """Warn when a split disagrees with the published counts.

    A warning rather than an exception, matching the IMDB project's row-count
    guard: a non-standard copy should be loudly visible but should not make
    the project unusable for someone holding a slightly different mirror.
    """
    expected = _EXPECTED.get(split)
    if expected is None:
        return
    n_sent = len(sentences)
    n_tok = sum(len(t) for t, _ in sentences)
    got_ent = {t: 0 for t in config.ENTITY_TYPES}
    for _, tags in sentences:
        for tag in tags:
            if tag.startswith("B-"):
                got_ent[tag[2:]] = got_ent.get(tag[2:], 0) + 1

    problems = []
    if n_sent != expected["sentences"]:
        problems.append(f"sentences {n_sent} != {expected['sentences']}")
    if n_tok != expected["tokens"]:
        problems.append(f"tokens {n_tok} != {expected['tokens']}")
    for typ, want in expected["entities"].items():
        if got_ent.get(typ, 0) != want:
            problems.append(f"{typ} {got_ent.get(typ, 0)} != {want}")
    if problems:
        print(f"[conll] WARNING: {split} disagrees with the published counts "
              f"-- {'; '.join(problems)}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CoNLLDataset(Dataset):
    """One CoNLL-2003 split as (word ids, case ids, length, tag ids) samples.

    The whole split is encoded ONCE in __init__ -- 15k short sentences, well
    under a second -- which keeps __getitem__ free and makes num_workers=0 the
    right default on Windows.

    Note the THREE parallel streams per sentence, all of the same length and
    all cut in the same place if they are ever cut at all:
        ids     lowercased word ids, for the GloVe-backed embedding table
        cases   6-way capitalization ids, for the case embedding
        labels  BIO2 tag ids

    Args:
        split: "train" / "valid" / "test".
        vocab: the Vocab to encode with. Pass the TRAIN vocab for every split.
        max_len: cap on sentence length (config.MAX_LEN). Cutting here deletes
            gold entities, so it warns rather than doing it silently.
    """

    def __init__(self, split: str, vocab: Vocab, max_len: int = None):
        self.split = split
        self.vocab = vocab
        self.max_len = max_len or config.MAX_LEN

        sentences = read_split(split)
        self.tokens = [t for t, _ in sentences]
        self.tags = [g for _, g in sentences]
        self.full_lengths = [len(t) for t in self.tokens]

        # Truncation is a LOSS OF LABELS here, not just of context, so say so.
        n_cut = sum(1 for L in self.full_lengths if L > self.max_len)
        if n_cut:
            dropped = sum(max(0, L - self.max_len) for L in self.full_lengths)
            print(f"[conll] WARNING: {n_cut} {split} sentences exceed "
                  f"max_len={self.max_len}; {dropped} tokens AND THEIR GOLD "
                  f"TAGS are being dropped from both training and scoring. "
                  f"Raise config.MAX_LEN.")
        self.tokens = [t[:self.max_len] for t in self.tokens]
        self.tags = [g[:self.max_len] for g in self.tags]

        self.ids = [vocab.encode(t) for t in self.tokens]
        self.cases = [case_ids(t) for t in self.tokens]
        self.labels = [[config.TAG2ID[g] for g in tags] for tags in self.tags]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Output: (ids [L], cases [L], length int, labels [L]) -- all long."""
        return (torch.tensor(self.ids[idx], dtype=torch.long),
                torch.tensor(self.cases[idx], dtype=torch.long),
                len(self.ids[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))

    # ---- small helpers used by the report / logs ----
    def unk_rate(self) -> float:
        """Fraction of tokens that map to <unk> (vocabulary coverage check).

        Expect this to be far higher on test than the classification projects
        ever saw: the test split is from a different month's news, so the
        people and organizations in it are largely new words. That is the
        problem the case feature exists to survive.
        """
        total = sum(len(ids) for ids in self.ids)
        unks = sum(sum(1 for i in ids if i == config.UNK_IDX) for ids in self.ids)
        return unks / max(total, 1)

    def entity_counts(self) -> dict:
        """Entity spans per type, counted as B- tags (BIO2)."""
        out = {t: 0 for t in config.ENTITY_TYPES}
        for tags in self.tags:
            for tag in tags:
                if tag.startswith("B-"):
                    out[tag[2:]] += 1
        return out

    def tag_counts(self) -> dict:
        """Token-level histogram over all 9 tags (the class-imbalance view)."""
        out = {t: 0 for t in config.TAGS}
        for tags in self.tags:
            for tag in tags:
                out[tag] += 1
        return out

    def o_rate(self) -> float:
        """Fraction of tokens tagged "O" -- i.e. the all-O baseline's accuracy."""
        counts = self.tag_counts()
        return counts["O"] / max(sum(counts.values()), 1)


def collate_batch(batch):
    """Pad a list of samples into one rectangular batch.

    THE FILL VALUES ARE NOT ALL THE SAME, and that is the point of this
    function. Word ids and case ids pad with config.PAD_IDX (0), which the
    embedding tables pin to the zero vector. LABELS pad with
    config.IGNORE_INDEX (-100), which nn.CrossEntropyLoss drops.

    Padding labels with 0 instead would be a silent disaster rather than an
    error: tag id 0 is "O", so every padded position would become a training
    example teaching the model to predict "O" -- on a task where 83% of real
    tokens are already "O", and where the metric punishes exactly the
    resulting timidity. The model would still train, the loss would still
    fall, and the entity F1 would just be quietly worse.

    Pads DYNAMICALLY, to the longest sentence in THIS batch. Unlike IMDB --
    where 18% of reviews hit the cap so nearly every batch was 400 wide and
    dynamic padding saved almost nothing -- sentence lengths here are short
    and highly variable, so this genuinely cuts the work.

    Input:
        batch: list of (ids [L_i], cases [L_i], length_i, labels [L_i]).
    Output:
        ids:     [B, L_max] long, padded with config.PAD_IDX (0)
        cases:   [B, L_max] long, padded with config.PAD_IDX (0)
        lengths: [B] long, the TRUE length of each row
        labels:  [B, L_max] long, padded with config.IGNORE_INDEX (-100)
    """
    seqs, cases, lengths, labels = zip(*batch)
    max_len = max(lengths)
    B = len(seqs)

    out_ids = torch.full((B, max_len), config.PAD_IDX, dtype=torch.long)
    out_cases = torch.full((B, max_len), config.PAD_IDX, dtype=torch.long)
    out_labels = torch.full((B, max_len), config.IGNORE_INDEX, dtype=torch.long)
    for i, (seq, case, lab) in enumerate(zip(seqs, cases, labels)):
        n = len(seq)
        out_ids[i, :n] = seq
        out_cases[i, :n] = case
        out_labels[i, :n] = lab
    return out_ids, out_cases, torch.tensor(lengths, dtype=torch.long), out_labels


def build_vocab_from_train(min_freq=None, max_size=None) -> Vocab:
    """Build the vocabulary from the TRAIN split only (see vocab.py).

    Deterministic: same files + same min_freq/max_size always yield the same
    itos, so a rebuild in eval.py matches the ids the checkpoint was trained
    with even if vocab.json went missing.
    """
    sentences = read_split("train")
    return Vocab.build((toks for toks, _ in sentences),
                       min_freq=config.MIN_FREQ if min_freq is None else min_freq,
                       max_size=config.MAX_VOCAB_SIZE if max_size is None else max_size)


# ---- Quick self-test / downloader: run this file directly --------------------
# python dataset/conll2003.py --download
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download / inspect CoNLL-2003")
    p.add_argument("--download", action="store_true", help="download the corpus first")
    args = p.parse_args()

    # The conversion is checkable WITHOUT the corpus, so do it first: these
    # are the two cases that separate IOB1 from BIO2.
    print("IOB1 -> BIO2 conversion:")
    cases = [
        (["I-ORG", "O", "I-MISC"], ["B-ORG", "O", "B-MISC"],
         "bare I- starts an entity in IOB1"),
        (["I-ORG", "I-ORG", "O"], ["B-ORG", "I-ORG", "O"],
         "only the FIRST becomes B-"),
        (["I-PER", "B-PER"], ["B-PER", "B-PER"],
         "IOB1 B- = two adjacent same-type entities; both are starts"),
        (["I-LOC", "I-ORG"], ["B-LOC", "B-ORG"],
         "a type change is a boundary"),
        (["O", "I-PER", "I-PER", "I-LOC"], ["O", "B-PER", "I-PER", "B-LOC"],
         "combined"),
    ]
    for src, want, why in cases:
        got = iob1_to_bio2(src)
        print(f"  {'OK ' if got == want else 'FAIL'} {src} -> {got}   ({why})")

    if args.download or not conll_present():
        download_conll()

    vocab = build_vocab_from_train()
    print(f"\nvocab size: {len(vocab)} (min_freq={config.MIN_FREQ}, lowercased)")

    for split in ("train", "valid", "test"):
        ds = CoNLLDataset(split, vocab)
        lens = sorted(ds.full_lengths)
        n_tok = sum(lens)
        print(f"\n{split:5} sentences={len(ds):6}  tokens={n_tok:7}")
        print(f"      entities={ds.entity_counts()}")
        print(f"      <unk> rate={ds.unk_rate():.4f}   "
              f"all-O token accuracy={ds.o_rate():.4f}")
        print(f"      sentence len mean={n_tok / len(lens):.1f} "
              f"median={lens[len(lens) // 2]} p95={lens[int(len(lens) * 0.95)]} "
              f"max={lens[-1]}  (max_len={ds.max_len})")

    ds = CoNLLDataset("train", vocab)
    print("\nfirst 2 train sentences:")
    for i in range(2):
        pairs = " ".join(f"{t}/{g}" for t, g in zip(ds.tokens[i], ds.tags[i]))
        print(f"  {pairs}")

    ids, cases, lengths, labels = collate_batch([ds[0], ds[1], ds[2]])
    print(f"\ncollated ids {tuple(ids.shape)} cases {tuple(cases.shape)} "
          f"labels {tuple(labels.shape)} lengths={lengths.tolist()}")
    short = int(lengths.argmin())
    print(f"padded tail of the shortest row -- ids: "
          f"{ids[short, int(lengths.min()):].tolist()[:8]} (expected 0s)")
    print(f"                                  labels: "
          f"{labels[short, int(lengths.min()):].tolist()[:8]} (expected -100s)")
    print(f"real positions never hold the ignore value: "
          f"{bool((labels[labels != config.IGNORE_INDEX] >= 0).all())}")
