"""WordPiece encoding for BERT: CoNLL words -> piece ids + first-piece index.

dataset/conll2003.py gives the GloVe models three streams of the SAME length --
word ids, case ids, labels -- because those models read one vector per word.
BERT reads WordPiece tokens, and one word can be several of them, so the
streams stop lining up. A sentence of three words, the middle one split into
three pieces (the pieces are schematic, not bert-base-cased's real output):

    words         w0          w1                w2
    position      0     1     2    3     4      5     6
    input_ids     [CLS] a     b    ##c   ##d    e     [SEP]     T = 7
    word_index          1     2                 5               W = 3
    labels              y0    y1                y2              W = 3

input_ids is what BERT reads. word_index says where each word's FIRST piece
sits, and labels stay one per WORD. That last point is the design: `lengths`
still counts words and the labels are the same BIO2 ids CoNLLDataset produces,
so everything downstream of the logits -- loss, TaggingMetrics, entity F1 --
sees this model exactly as it sees the other two.

WHY THE FIRST PIECE. Devlin et al. (2019) represent each word by its first
sub-token; the later pieces still feed context to every layer, they just never
carry a label. model_bert/bert_tagger.py reads the positions in word_index.

WHY TOKENIZE WORD BY WORD. CoNLL is already tokenized, and BERT's tokenizer
never merges across a word boundary, so tokenizing each word on its own gives
the same pieces as tokenizing the sentence -- and makes the alignment something
BUILT here rather than recovered afterwards: a piece belongs to word i because
it came out of word i.

NO <unk> IN THE GloVe SENSE. The GloVe models map every word missing from the
train vocabulary to one shared <unk> vector. WordPiece spells such a word out
of smaller pieces that ARE in its vocabulary, and emits [UNK] only for a word
it cannot cover at all -- in practice, one containing a character it has no
piece for. unk_rate() measures that per word, so it reads directly against
CoNLLDataset.unk_rate().
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Make the project root importable (works both as a script and as a package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from dataset.conll2003 import read_split  # noqa: E402


def load_tokenizer(name: str):
    """Load the WordPiece tokenizer that belongs to checkpoint `name`.

    The tokenizer is part of the CHECKPOINT, not of the data: embedding row k
    means whatever vocabulary entry k is, so tokenizer and weights must always
    be loaded from the same name.

    Warns (does not refuse) when the tokenizer lowercases. An uncased
    checkpoint is a legitimate run -- BERT's version of the --no-case
    ablation -- but on this task it should never happen by accident.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.tokenize("Paris") == tokenizer.tokenize("paris"):
        print(f"[bert] WARNING: {name} lowercases its input, so casing -- the "
              f"strongest NER feature (config.py, point 4) -- never reaches the "
              f"model. Use it only as the uncased ablation.")
    return tokenizer


def encode_words(words, tokenizer):
    """Encode one pre-tokenized sentence (see the diagram in the module docstring).

    Input:
        words: list[str], the cased CoNLL tokens of ONE sentence.
        tokenizer: a BERT WordPiece tokenizer.
    Output:
        input_ids: list[int], [CLS] + every word's pieces + [SEP]     length T
        word_index: list[int], position of each word's first piece    length len(words)
    """
    # One call per sentence, but every word is a separate input, so pieces[i]
    # is exactly word i's pieces.
    pieces = tokenizer(list(words), add_special_tokens=False)["input_ids"]
    input_ids = [tokenizer.cls_token_id]
    word_index = []
    for word_pieces in pieces:
        word_index.append(len(input_ids))
        # BERT's normalizer deletes control characters, so a word made only of
        # them would come back with no pieces -- and no position to read its
        # label from. [UNK] keeps the word, and its gold tag, in the sentence.
        input_ids.extend(word_pieces or [tokenizer.unk_token_id])
    input_ids.append(tokenizer.sep_token_id)
    return input_ids, word_index


class BertCoNLLDataset(Dataset):
    """One CoNLL-2003 split as (input_ids, word_index, length, labels) samples.

    The same 4-tuple shape as CoNLLDataset's (ids, cases, length, labels),
    with word_index in the slot the case ids used to hold. `length` still
    counts WORDS. That shape is what lets train.py's evaluate() and
    utils.metrics.evaluate_tagger() iterate this dataset's loader unchanged.

    Args:
        split: "train" / "valid" / "test".
        tokenizer: from load_tokenizer(), matching the checkpoint.
        max_len: piece limit including [CLS]/[SEP] (config.BERT_MAX_LEN).
    """

    def __init__(self, split: str, tokenizer, max_len: int = None):
        self.split = split
        self.max_len = max_len or config.BERT_MAX_LEN
        self.unk_id = tokenizer.unk_token_id

        sentences = read_split(split)
        self.tokens = [t for t, _ in sentences]
        self.tags = [g for _, g in sentences]
        self.labels = [[config.TAG2ID[g] for g in tags] for tags in self.tags]

        self.input_ids, self.word_index = [], []
        for words in self.tokens:
            ids, index = encode_words(words, tokenizer)
            if len(ids) > self.max_len:
                raise ValueError(
                    f"a {split} sentence of {len(words)} words is {len(ids)} "
                    f"WordPiece tokens, over the {self.max_len}-position limit. "
                    f"Truncating would drop gold labels; this input needs a "
                    f"sliding window.")
            self.input_ids.append(ids)
            self.word_index.append(index)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Output: (input_ids [T], word_index [W], W int, labels [W]) -- all long."""
        return (torch.tensor(self.input_ids[idx], dtype=torch.long),
                torch.tensor(self.word_index[idx], dtype=torch.long),
                len(self.word_index[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))

    def unk_rate(self) -> float:
        """Fraction of WORDS containing an [UNK] piece."""
        words = unks = 0
        for ids, index in zip(self.input_ids, self.word_index):
            ends = index[1:] + [len(ids) - 1]    # a word runs up to the next one
            unks += sum(self.unk_id in ids[s:e] for s, e in zip(index, ends))
            words += len(index)
        return unks / max(words, 1)


def collate_bert_batch(batch, pad_id):
    """Pad a list of samples into one batch. Three streams, three fill values:

        input_ids   pad_id ([PAD])   BertTagger derives its attention mask from it
        word_index  0 ([CLS])        any real position works: those label slots
                                     are ignored
        labels      IGNORE_INDEX     dropped by the loss (see collate_batch in
                                     dataset/conll2003.py for why not 0)

    The two widths differ: input_ids pads to the longest PIECE sequence in the
    batch, word_index and labels to the longest WORD sequence.

    Input:
        batch: list of (input_ids [T_i], word_index [W_i], W_i, labels [W_i]).
        pad_id: the tokenizer's [PAD] id.
    Output:
        input_ids [B, T_max], word_index [B, W_max], lengths [B], labels [B, W_max]
    """
    input_ids, word_index, lengths, labels = zip(*batch)
    B = len(batch)
    T = max(len(ids) for ids in input_ids)
    W = max(lengths)

    out_ids = torch.full((B, T), pad_id, dtype=torch.long)
    out_index = torch.zeros((B, W), dtype=torch.long)
    out_labels = torch.full((B, W), config.IGNORE_INDEX, dtype=torch.long)
    for i, (ids, index, lab) in enumerate(zip(input_ids, word_index, labels)):
        out_ids[i, :len(ids)] = ids
        out_index[i, :len(index)] = index
        out_labels[i, :len(lab)] = lab
    return out_ids, out_index, torch.tensor(lengths, dtype=torch.long), out_labels


# ---- Quick self-test: run this file directly ---------------------------------
# python model_bert/wordpiece.py     (fetches the tokenizer files, not the weights)
if __name__ == "__main__":
    tokenizer = load_tokenizer(config.BERT_NAME)
    ds = BertCoNLLDataset("train", tokenizer)

    ids, index = ds.input_ids[0], ds.word_index[0]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    ends = index[1:] + [len(ids) - 1]
    print("first train sentence -- each tag is read at its word's first piece:")
    for word, tag, start, end in zip(ds.tokens[0], ds.tags[0], index, ends):
        print(f"  {word:<12} {tag:<7} pos {start:>2}   {' '.join(pieces[start:end])}")

    for split in ("train", "valid", "test"):
        d = ds if split == "train" else BertCoNLLDataset(split, tokenizer)
        n_words = sum(len(i) for i in d.word_index)
        n_pieces = sum(len(i) - 2 for i in d.input_ids)
        print(f"{split:5} words={n_words:6}  pieces={n_pieces:6} "
              f"({n_pieces / n_words:.2f}/word)  "
              f"longest={max(len(i) for i in d.input_ids)}/{d.max_len}  "
              f"[UNK] word rate={d.unk_rate():.4f}")

    input_ids, word_index, lengths, labels = collate_bert_batch(
        [ds[0], ds[1], ds[2]], tokenizer.pad_token_id)
    print(f"\ncollated input_ids {tuple(input_ids.shape)} (pieces)   "
          f"word_index {tuple(word_index.shape)} + labels {tuple(labels.shape)} "
          f"(words)   lengths={lengths.tolist()}")
