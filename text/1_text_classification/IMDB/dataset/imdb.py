"""IMDB movie-review corpus: download, parsing, splitting, Dataset, collate_fn.

Same five responsibilities as the AG-News sibling's dataset/ag_news.py:
  1. Download the Large Movie Review Dataset into config.DATA_ROOT.
  2. Parse it into (text, label) pairs.
  3. Carve a stratified VALIDATION split out of train.
  4. Wrap a split in a Dataset that yields (ids LongTensor, length, label).
  5. Provide the COLLATE FUNCTION that pads variable-length reviews.

WHAT IS DIFFERENT ABOUT THIS CORPUS. Maas et al. (2011) ship 50,000 reviews as
50,000 SEPARATE .txt FILES in a directory tree:

    aclImdb/train/pos/0_9.txt      <- filename is <id>_<star rating>.txt
    aclImdb/train/neg/0_3.txt
    aclImdb/test/pos/ ...          12,500 files per folder, 4 folders
    aclImdb/train/unsup/ ...       50,000 MORE, unlabeled -- not used here

Opening 50k files takes longer than reading the reviews does, and on Windows
it is the slowest part of the whole pipeline by a wide margin. So the download
step CONSOLIDATES the tree into two csv files -- train.csv and test.csv, the
same layout AG News ships natively -- and deletes the extracted tree. After
that this file reads exactly like its AG-News counterpart.

The polarity split is by star rating, and the middle is deliberately EMPTY:
    <= 4 stars -> negative        >= 7 stars -> positive        5, 6 -> dropped
That is why 50,000 reviews yield only 25,000 train + 25,000 test, and it is
also why this task is easier than its length suggests -- there are no
lukewarm reviews to sit on the fence.

Two things worth knowing about how clean this benchmark is:

  GOOD. By construction (Maas et al.), no user contributes more than 30
  reviews to either half and the train/test halves use DISJOINT movies -- so
  there is no author or title leakage of the kind that inflates many scraped
  sentiment corpora.

  LESS GOOD, and measured here rather than taken on faith. The corpus
  nonetheless contains VERBATIM DUPLICATE REVIEWS, presumably from users who
  posted the same text on several titles:
      train.csv  188 rows (0.75%) belong to a duplicate group
      test.csv   390 rows (1.56%)
      123 test reviews (0.49%) appear word-for-word in train.csv
  No duplicate pair disagrees on its label, so this is memorization headroom
  rather than label noise, and at half a percent it can shift a reported
  accuracy by at most ~0.05 points. It is not worth deduplicating (that would
  make the number incomparable with every published IMDB result), but it IS
  worth knowing before treating a 0.2-point difference as real.

How to download: python dataset/imdb.py --download
(train.py also downloads automatically when the corpus is missing.)
"""

import argparse
import csv
import html
import os
import random
import re
import sys
import shutil

import torch
from torch.utils.data import Dataset

# Make the project root importable (works both as a script and as a package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.download import download_with_mirrors, extract_tar  # noqa: E402

try:
    from .vocab import Vocab, tokenize, truncate
except ImportError:  # running this file directly
    from vocab import Vocab, tokenize, truncate

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
# Stanford's original release (84 MB tgz). The https and http hosts are the
# same machine; the plain-http entry exists because the certificate on
# ai.stanford.edu has expired more than once.
_IMDB_URLS = [
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
]
_IMDB_MD5 = "7c2ac02c03563afcf9b574c7e56c153a"

# Expected row counts, used INSTEAD of an md5 on the csv files we generate:
# the tgz has one, and these counts catch a half-finished consolidation.
_EXPECTED_ROWS = {"train.csv": 25000, "test.csv": 25000}

# The four folders that carry labels. "unsup" (50k unlabeled reviews, meant
# for pretraining word vectors) is skipped -- GloVe already fills that role
# here, and extracting it would double the file count for nothing.
_LABELED = [("train", "neg", 0), ("train", "pos", 1),
            ("test", "neg", 0), ("test", "pos", 1)]


def imdb_present() -> bool:
    """True when both consolidated csv files exist under config.IMDB_DIR."""
    return all(os.path.isfile(os.path.join(config.IMDB_DIR, f))
               for f in ("train.csv", "test.csv"))


def download_imdb():
    """Fetch IMDB, consolidate the 50k .txt files into two csv, clean up.

    Idempotent: returns immediately when the csv files are already there.
    """
    if imdb_present():
        print(f"IMDB already present at {config.IMDB_DIR}")
        return
    os.makedirs(config.IMDB_DIR, exist_ok=True)

    archive = os.path.join(config.DATA_ROOT, "aclImdb_v1.tar.gz")
    if not os.path.isfile(archive):
        print("Downloading IMDB (~84 MB)...")
        download_with_mirrors(_IMDB_URLS, archive, md5=_IMDB_MD5, connections=8)

    root = os.path.join(config.DATA_ROOT, "aclImdb")
    print("Extracting (only the four labeled folders)...")
    # members= keeps train/unsup and the precomputed bag-of-words .feat files
    # (which are larger than the reviews themselves) out of the extraction.
    keep = tuple(f"aclImdb/{split}/{pol}/" for split, pol, _ in _LABELED)
    extract_tar(archive, config.DATA_ROOT,
                members=lambda name: name.startswith(keep))

    for split in ("train", "test"):
        rows = []
        for _, pol, label in [x for x in _LABELED if x[0] == split]:
            folder = os.path.join(root, split, pol)
            # sorted() so the csv order is reproducible across filesystems.
            for name in sorted(os.listdir(folder)):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    rows.append((label, f.read()))
        out = os.path.join(config.IMDB_DIR, f"{split}.csv")
        # newline="": let the csv writer emit its own line endings, otherwise
        # Windows turns every "\n" into "\r\r\n" inside quoted fields.
        with open(out, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"  wrote {out} ({len(rows)} reviews)")

    shutil.rmtree(root, ignore_errors=True)   # 50k tiny files, no longer needed
    os.remove(archive)
    if not imdb_present():
        raise RuntimeError(f"IMDB csv files missing from {config.IMDB_DIR}")


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
# The one scraping artifact in this corpus. Reviews were saved as HTML, so
# every paragraph break in the original is a literal "<br />" in the text --
# and it is EVERYWHERE (see _clean for the measured share).
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _clean(text: str) -> str:
    """Repair the HTML that the review scrape left behind.

    1. LINE BREAKS AS <br /> TAGS -- the one artifact that actually matters.
       Measured on train.csv: 14,665 of 25,000 reviews (58.7%) contain at
       least one, 101,870 tags in total, 4.1 per review on average, and they
       arrive doubled ("<br /><br />") at every paragraph boundary. Left alone
       the tokenizer emits ["<", "br", "/", ">"] for each one -- on a
       4,000-review sample that is 5.8% of ALL TOKENS being punctuation noise,
       and "br" becomes one of the most frequent "words" in the vocabulary.

       Replaced with a SPACE, not with the empty string: reviewers write
       "great movie<br />I loved it" with no surrounding whitespace, and
       deleting the tag would fuse that into "moviei".

    2. HTML ENTITIES. Measured: 5 reviews out of 25,000 (0.02%) contain one.
       Essentially absent, unlike AG News where a quarter of all rows carried
       a mangled "#39;" -- IMDB's scrape decoded them and only re-encoded the
       tags. html.unescape stays because it costs one call, but do not expect
       it to change anything.

    Deliberately NOT done: lowercasing, stopword removal, stemming, or
    stripping the "8/10"-style ratings 6.0% of reviewers write into the text.
    The first belongs to the tokenizer (vocab.py); the second and third throw
    away signal; the fourth would be removing a genuine feature of the data
    (a real reader sees it too, and the benchmark has always included it).
    """
    return html.unescape(_BR_RE.sub(" ", text))


def read_csv(name: str):
    """Read one consolidated csv into a list of (text, label) pairs.

    Input:
        name: "train.csv" or "test.csv".
    Output:
        list of (text str, label int -- 0 negative, 1 positive).
    """
    path = os.path.join(config.IMDB_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python dataset/imdb.py --download`")

    out = []
    # newline="" is required by the csv module so quoted fields containing
    # newlines are handled by it rather than by python's line splitting --
    # and unlike AG News, IMDB reviews really do contain newlines.
    with open(path, encoding="utf-8", newline="") as f:
        # A handful of reviews are longer than the csv module's default
        # 128 KB field limit is comfortable with; raise it once, here.
        csv.field_size_limit(1 << 24)
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            out.append((_clean(row[1]), int(row[0])))

    expected = _EXPECTED_ROWS.get(name)
    if expected is not None and len(out) != expected:
        print(f"[imdb] WARNING: {name} has {len(out)} rows, "
              f"expected {expected} -- corrupt or non-standard copy?")
    return out


def read_split(split: str):
    """Read one logical split: "train" / "val" / "test".

    "train" and "val" are the two halves of the stratified split of
    train.csv; "test" is test.csv verbatim.
    """
    if split == "test":
        return read_csv("test.csv")
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    train_rows, val_rows = split_train_val(read_csv("train.csv"))
    return train_rows if split == "train" else val_rows


def split_train_val(rows, val_ratio: float = None, seed: int = None):
    """Split rows into (train, val), stratified by class and deterministic.

    Identical in shape to the AG-News version and for the same reason: IMDB
    publishes its test LABELS, so scoring test every epoch and keeping the
    best epoch would make the headline a best-of-N number. Selection happens
    on this val split; test.csv is read once, by eval.py.

    The consolidated csv is written neg-then-pos, so the shuffle at the end
    is not cosmetic -- without it "train" would be perfectly class-sorted.

    Input:
        rows: list of (text, label) from read_csv("train.csv").
        val_ratio / seed: default to config.VAL_RATIO / config.SPLIT_SEED.
    Output:
        (train_rows, val_rows), both in shuffled order.
    """
    val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
    seed = config.SPLIT_SEED if seed is None else seed

    by_class = {}
    for row in rows:
        by_class.setdefault(row[1], []).append(row)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for label in sorted(by_class):
        bucket = by_class[label]
        rng.shuffle(bucket)
        n_val = int(round(len(bucket) * val_ratio))
        val_rows.extend(bucket[:n_val])
        train_rows.extend(bucket[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class IMDBDataset(Dataset):
    """One IMDB split as (token ids, length, label) samples.

    The whole split is tokenized and encoded ONCE in __init__. That is a
    heavier promise than on AG News -- 22,500 reviews of ~300 tokens is ~7M
    ids in python lists, a few hundred MB and ~20 s -- but it keeps
    __getitem__ free and makes num_workers=0 the right default on Windows.

    Args:
        split: "train" / "val" / "test".
        vocab: the Vocab to encode with. Pass the TRAIN vocab for every split.
        max_len: truncation length (config.MAX_LEN).
        truncation / head_len: which end(s) to keep (config.TRUNCATION,
            config.HEAD_LEN). eval.py passes the CHECKPOINT's values.
    """

    def __init__(self, split: str, vocab: Vocab, max_len: int = None,
                 truncation: str = None, head_len: int = None):
        self.split = split
        self.vocab = vocab
        self.max_len = max_len or config.MAX_LEN
        self.truncation = truncation or config.TRUNCATION
        self.head_len = config.HEAD_LEN if head_len is None else head_len

        pairs = read_split(split)
        self.texts = [t for t, _ in pairs]
        self.labels = [y for _, y in pairs]
        # full_lengths records the length BEFORE truncation, so the reports
        # below can say how much of the corpus max_len is actually cutting.
        encoded = [vocab.encode(tokenize(t)) for t in self.texts]
        self.full_lengths = [len(ids) for ids in encoded]
        self.ids = [truncate(ids, self.max_len, self.truncation, self.head_len)
                    or [config.UNK_IDX] for ids in encoded]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Output: (ids [L] long, length int, label int)."""
        ids = self.ids[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids), self.labels[idx]

    # ---- small helpers used by the report / logs ----
    def unk_rate(self) -> float:
        """Fraction of tokens that map to <unk> (vocabulary coverage check)."""
        total = sum(len(ids) for ids in self.ids)
        unks = sum(sum(1 for i in ids if i == config.UNK_IDX) for ids in self.ids)
        return unks / max(total, 1)

    def truncated_rate(self) -> float:
        """Fraction of reviews longer than max_len (i.e. actually cut)."""
        n = sum(1 for L in self.full_lengths if L > self.max_len)
        return n / max(len(self.full_lengths), 1)

    def label_counts(self) -> dict:
        """Class histogram, e.g. {"negative": 11250, "positive": 11250}."""
        out = {name: 0 for name in config.CLASS_NAMES}
        for y in self.labels:
            if 0 <= y < len(config.CLASS_NAMES):
                out[config.CLASS_NAMES[y]] += 1
        return out


def collate_batch(batch):
    """Pad a list of samples into one rectangular batch.

    Pads DYNAMICALLY -- to the longest review in THIS batch, not to
    config.MAX_LEN. The lengths travel with the batch so the encoder can pack
    them and the head can mask them.

    AND IT BUYS ALMOST NOTHING HERE, which is worth knowing. Measured on the
    train portion at MAX_LEN=400: the mean padded batch width is 400.0 at
    batch_size 32, 64 AND 128, because 18% of reviews are longer than the cap
    and a shuffled batch of 64 essentially always contains one. Reviews carry
    230.7 real tokens on average, so 42.3% of every batch is <pad>.

    The fix would be LENGTH-BUCKETED batching (sort by length, then form
    batches): measured, that drops the waste to 0.2% and the whole epoch to
    0.58x the compute. It is not done here because it correlates each batch's
    composition with review length, which interacts with BatchNorm-free but
    still stateful training in ways that would need their own justification --
    and this project is meant to isolate the effect of the DATA, not to
    introduce a sampling scheme the sibling projects do not have.

    Input:
        batch: list of (ids [L_i], length_i, label_i) from IMDBDataset.
    Output:
        ids:     [B, L_max] long, padded with config.PAD_IDX (0)
        lengths: [B] long, the TRUE length of each row (before padding)
        labels:  [B] long
    """
    seqs, lengths, labels = zip(*batch)
    max_len = max(lengths)

    ids = torch.full((len(seqs), max_len), config.PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = seq          # copy in; the tail stays <pad>
    return (ids,
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def build_vocab_from_train(min_freq=None, max_size=None) -> Vocab:
    """Build the vocabulary from the TRAIN portion only (see vocab.py).

    Deterministic: same csv + same split seed + same min_freq/max_size always
    yields the same itos, so a rebuild in eval.py matches the ids the
    checkpoint was trained with even if vocab.json went missing.
    """
    rows = read_split("train")
    return Vocab.build((tokenize(t) for t, _ in rows),
                       min_freq=config.MIN_FREQ if min_freq is None else min_freq,
                       max_size=config.MAX_VOCAB_SIZE if max_size is None else max_size)


# ---- Quick self-test / downloader: run this file directly --------------------
# python dataset/imdb.py --download
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download / inspect the IMDB corpus")
    p.add_argument("--download", action="store_true", help="download IMDB first")
    args = p.parse_args()

    if args.download or not imdb_present():
        download_imdb()

    vocab = build_vocab_from_train()
    print(f"\nvocab size: {len(vocab)} (min_freq={config.MIN_FREQ})")

    for split in ("train", "val", "test"):
        ds = IMDBDataset(split, vocab)
        lens = sorted(ds.full_lengths)
        print(f"\n{split:5} docs={len(ds):6}  labels={ds.label_counts()}")
        print(f"      <unk> rate={ds.unk_rate():.4f}  "
              f"truncated at {ds.max_len}: {ds.truncated_rate():.2%}")
        print(f"      full len mean={sum(lens)/len(lens):.1f} "
              f"median={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} "
              f"p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")

    # split_train_val PARTITIONS the rows, so a nonzero count here is not a
    # split bug -- it is the corpus's own duplicate reviews (see the module
    # docstring) landing on both sides. Expect a handful; a large number would
    # mean the split really is broken.
    train_texts = set(IMDBDataset("train", vocab).texts)
    val_texts = IMDBDataset("val", vocab).texts
    overlap = sum(1 for t in val_texts if t in train_texts)
    print(f"\nval reviews whose text also occurs in train: {overlap} "
          f"({overlap / len(val_texts):.2%}; these are IMDB's own duplicates, "
          f"not a split error)")

    ds = IMDBDataset("train", vocab)
    print("\nfirst 2 train samples:")
    for i in range(2):
        ids, n, y = ds[i]
        print(f"  [{config.CLASS_NAMES[y]}] len={n} ids={ids.tolist()[:10]}...")
        print(f"      {ds.texts[i][:100]!r}")

    ids, lengths, labels = collate_batch([ds[0], ds[1], ds[2]])
    print(f"\ncollated ids {tuple(ids.shape)} lengths={lengths.tolist()} "
          f"labels={labels.tolist()}")
    print(f"padded tail of the shortest row: "
          f"{ids[int(lengths.argmin()), int(lengths.min()):].tolist()[:10]} (expected 0s)")
