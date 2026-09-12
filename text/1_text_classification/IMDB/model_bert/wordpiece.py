"""WordPiece encoding for BERT: review text -> [CLS] + pieces + [SEP], at most 512.

dataset/imdb.py turns a review into GloVe word ids; this file turns the SAME
cleaned review into BERT's WordPiece ids. The two pipelines share everything
that decides what a model gets to read:

    read_split()    same 22,500 / 2,500 / 25,000 reviews, same <br /> cleanup
    truncate()      same head / head_tail arithmetic (dataset/vocab.py),
                    applied to pieces instead of words
    collate_batch   same padding, same (ids, lengths, labels) batch

and differ only in the tokenizer and the window:

                     GloVe models                    BERT
    tokenizer        regex + lowercase, min_freq=5   WordPiece (uncased), fixed vocab
    unseen word      <unk>                           spelled from smaller pieces
    window           MAX_LEN = 400 tokens            510 pieces + [CLS] + [SEP]
    head_tail cut    HEAD_LEN 300 + 100              BERT_HEAD_LEN 128 + 382

CUT AFTER TOKENIZING. The limit is counted in pieces, and a word's piece count
is only known once WordPiece has run, so each review is tokenized whole and the
cut is made on the ids -- which is also what lets truncate() be reused as is.
While that happens the tokenizer logs a one-time warning that a sequence is
longer than 512; that is expected, since the cut comes right after.

THE SEAM is the same compromise as in the GloVe pipeline (config.py,
TRUNCATION): head and tail are joined with no separator, and BERT's position
embeddings number the joined sequence as if it were contiguous.
"""

import os
import sys

import torch
from torch.utils.data import Dataset

# Make the project root importable (works both as a script and as a package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from dataset.imdb import read_split  # noqa: E402
from dataset.vocab import truncate  # noqa: E402


class BertIMDBDataset(Dataset):
    """One IMDB split as (piece ids, length, label) samples.

    The SAME sample tuple as IMDBDataset, so dataset.imdb.collate_batch pads it
    and train.py's evaluate() reads it unchanged. `length` counts pieces
    including [CLS] and [SEP], and it is the only padding signal BertClassifier
    uses.

    Args:
        split: "train" / "val" / "test".
        tokenizer: loaded from the same name as the checkpoint.
        max_len: piece limit including [CLS]/[SEP] (config.BERT_MAX_LEN).
        truncation / head_len: which end(s) to keep (config.TRUNCATION,
            config.BERT_HEAD_LEN).
    """

    def __init__(self, split: str, tokenizer, max_len: int = None,
                 truncation: str = None, head_len: int = None):
        self.split = split
        self.max_len = max_len or config.BERT_MAX_LEN
        self.truncation = truncation or config.TRUNCATION
        self.head_len = config.BERT_HEAD_LEN if head_len is None else head_len
        self.unk_id = tokenizer.unk_token_id

        pairs = read_split(split)
        self.texts = [t for t, _ in pairs]
        self.labels = [y for _, y in pairs]

        self.ids, self.full_lengths = [], []
        # 1,000 reviews per call: batched, so the fast tokenizer can use every
        # core, without holding all 22,500 untruncated encodings at once.
        for start in range(0, len(self.texts), 1000):
            chunk = tokenizer(self.texts[start:start + 1000],
                              add_special_tokens=False)["input_ids"]
            for pieces in chunk:
                # Recorded BEFORE the cut, [CLS]/[SEP] included, so it compares
                # directly against max_len in truncated_rate().
                self.full_lengths.append(len(pieces) + 2)
                pieces = truncate(pieces, self.max_len - 2, self.truncation,
                                  self.head_len)
                self.ids.append([tokenizer.cls_token_id] + pieces
                                + [tokenizer.sep_token_id])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Output: (ids [T] long, length int, label int) -- IMDBDataset's format."""
        ids = self.ids[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids), self.labels[idx]

    def truncated_rate(self) -> float:
        """Fraction of reviews longer than max_len pieces (i.e. actually cut)."""
        n = sum(1 for L in self.full_lengths if L > self.max_len)
        return n / max(len(self.full_lengths), 1)

    def unk_rate(self) -> float:
        """Fraction of pieces that are [UNK] ([CLS]/[SEP] excluded).

        The counterpart of IMDBDataset.unk_rate(), where every word below
        min_freq became <unk>; WordPiece emits [UNK] only for a word it cannot
        spell from its pieces at all.
        """
        total = sum(len(ids) - 2 for ids in self.ids)
        unks = sum(ids.count(self.unk_id) for ids in self.ids)
        return unks / max(total, 1)


# ---- Quick self-test: run this file directly ---------------------------------
# python model_bert/wordpiece.py     (fetches the tokenizer files, not the weights)
if __name__ == "__main__":
    from transformers import AutoTokenizer

    from dataset.imdb import collate_batch

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_NAME)
    for split in ("train", "val", "test"):
        ds = BertIMDBDataset(split, tokenizer)
        lens = sorted(ds.full_lengths)
        print(f"{split:5} docs={len(ds):6}  pieces incl. [CLS]/[SEP]: "
              f"mean={sum(lens) / len(lens):.1f} median={lens[len(lens) // 2]} "
              f"p95={lens[int(len(lens) * 0.95)]} max={lens[-1]}  "
              f"cut at {ds.max_len}: {ds.truncated_rate():.2%}  "
              f"[UNK] rate={ds.unk_rate():.4f}")

    ds = BertIMDBDataset("train", tokenizer)
    if ds.truncation == "head_tail":
        i = next(k for k, L in enumerate(ds.full_lengths) if L > ds.max_len)
        pieces = tokenizer.convert_ids_to_tokens(ds.ids[i])
        h = ds.head_len   # head pieces sit at positions 1..h, the tail starts at h+1
        print(f"\ntrain review {i}: {ds.full_lengths[i]} pieces cut to "
              f"{len(ds.ids[i])}. Around the seam:")
        print(f"  ...{' '.join(pieces[h - 5:h + 1])}  ||  "
              f"{' '.join(pieces[h + 1:h + 7])}...")

    ids, lengths, labels = collate_batch([ds[0], ds[1], ds[2]])
    print(f"\ncollated ids {tuple(ids.shape)} lengths={lengths.tolist()} "
          f"labels={labels.tolist()}")
