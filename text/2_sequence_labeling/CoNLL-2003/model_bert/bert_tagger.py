"""Pretrained BERT -> first-piece gather -> per-word tagger. The third model.

Forward pass (default checkpoint, bert-base-cased):
    input_ids [B, T] + word_index [B, W]
      --BERT------->  hidden  [B, T, 768]   12 pretrained layers, all finetuned
      --gather----->  words   [B, W, 768]   each word's FIRST piece
      --head------->  logits  [B, W, 9]     model/head.py, unchanged

T counts WordPiece tokens, W counts CoNLL words (model_bert/wordpiece.py draws
the difference). The gather is the one line where the two become the same.

WHAT BERT REPLACES. Below the encoder, the other two models are a stack of
fixes for the limits of GloVe 6B. Each has a direct counterpart here, and none
of the fixes is needed:

    GloVe models                               BERT
    -----------------------------------------  ---------------------------------
    vocab from train; other words -> <unk>     fixed 28,996-piece vocabulary;
                                               an unseen word is spelled from pieces
    lowercased lookup (GloVe 6B is uncased)    cased pieces: "Apple" != "apple"
    6-way case embedding to put casing back    nothing was taken away
    static vectors; context from 2 layers      context in all 12 layers,
      trained on 14k sentences                   pretrained

That is also why this model has its own train script instead of a --model
value: almost nothing around its encoder is shared (see train_bert.py).

THE GATHER, and why it is not the usual formulation. Devlin et al. (2019)
label each word at its first sub-token; the usual implementation scores all T
positions and sets the labels of the non-first pieces to IGNORE_INDEX. Picking
out the first pieces BEFORE the head computes exactly the same loss -- the same
logits against the same labels -- but returns WORD-level logits [B, W, 9], the
shape LSTMTagger and TransformerTagger return. So train.py's evaluate() and
utils/metrics.py score this model through the same code as the other two, and
the three F1s are comparable by construction rather than by a careful copy.
The self-test below checks the loss equivalence numerically.

forward() takes (input_ids, word_index, lengths) for the same reason: it is the
(ids, cases, lengths) call the shared code makes. `lengths` itself is unused --
padding is already visible in input_ids and in the labels.

PADDING. The attention mask is derived from input_ids != pad_id, so padded
PIECES are invisible to every attention layer -- that is what keeps a
sentence's logits independent of what it was batched with. Padded WORD slots
gather position 0 ([CLS]) and produce real-looking logits; as in head.py, the
loss and the metrics already ignore them.

WHAT IS NOT HERE.
  - No CRF, for the reason given in model/head.py.
  - No document context. Devlin et al. fed BERT "the maximal document context
    provided by the data" and report 92.4 test F1 for BERT-base. This model
    reads one sentence at a time, like the other two, so that number is not
    its target; the other two models are its comparison.
  - No pooler. add_pooling_layer=False drops the [CLS] classification layer a
    tagger never reads, so loading reports some checkpoint weights as unused
    (the pooler, and the pretraining heads under cls.*). That is expected.
"""

import os
import sys

import torch.nn as nn
from transformers import AutoModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from model.head import TaggerHead  # noqa: E402


class BertTagger(nn.Module):
    """Pretrained BERT encoder + per-word tagging head for CoNLL-2003 NER.

    Args:
        name: hub id or local directory of the checkpoint (config.BERT_NAME).
            The tokenizer MUST be loaded from the same name.
        num_tags: 9 (BIO2 over 4 entity types).
        dropout: before the head (config.BERT_DROPOUT). Dropout inside BERT
            stays at the checkpoint's own setting.
        pad_id: the tokenizer's [PAD] id; positions holding it are masked.
    """

    def __init__(self, name, num_tags=9, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.bert = AutoModel.from_pretrained(name, add_pooling_layer=False)
        self.head = TaggerHead(in_features=self.bert.config.hidden_size,
                               num_tags=num_tags, dropout=dropout)

    def forward(self, input_ids, word_index, lengths=None):
        """Input: input_ids [B, T], word_index [B, W]. Output: [B, W, num_tags]."""
        attention_mask = (input_ids != self.pad_id).long()                 # [B, T]
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state  # [B, T, H]
        # Gather along T: word w of row b reads hidden[b, word_index[b, w]].
        index = word_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1))  # [B, W, H]
        return self.head(hidden.gather(1, index))                          # [B, W, K]


# ---- Quick self-test: run this file directly --------------------------------
# python model_bert/bert_tagger.py     (a tiny RANDOM BERT -- downloads nothing)
if __name__ == "__main__":
    import tempfile

    import torch
    from transformers import BertConfig, BertModel

    import config

    torch.manual_seed(0)
    # A 2-layer, 32-wide random BERT saved to a temp dir, so the real
    # from_pretrained path runs without fetching any weights.
    tiny = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=64)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        BertModel(tiny, add_pooling_layer=False).save_pretrained(tmp)
        model = BertTagger(tmp, num_tags=config.NUM_TAGS, pad_id=0).eval()

    # [PAD]=0 [CLS]=1 [SEP]=2. Row 0: 4 words in 6 pieces, word 1 is three
    # pieces (positions 2-4). Row 1: 2 words in 3 pieces, then padding.
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 2],
                              [1, 11, 12, 13, 2, 0, 0, 0]])
    word_index = torch.tensor([[1, 2, 5, 6],
                               [1, 2, 0, 0]])
    lengths = torch.tensor([4, 2])
    labels = torch.tensor([[1, 2, 0, 3],
                           [5, 6, config.IGNORE_INDEX, config.IGNORE_INDEX]])

    with torch.no_grad():
        logits = model(input_ids, word_index, lengths)
        hidden = model.bert(input_ids=input_ids,
                            attention_mask=(input_ids != 0).long()).last_hidden_state
        first_piece = model.head(hidden[0, 2])
        alone = model(input_ids[1:, :5], word_index[1:, :2], lengths[1:])
        piece_logits = model.head(hidden)                              # [B, T, K]
    print(f"logits {tuple(logits.shape)} (expected (2, 4, 9): one row per WORD, "
          f"from {input_ids.size(1)} pieces)")

    print("word 1 == head(its first piece, position 2):",
          bool(torch.allclose(logits[0, 1], first_piece, atol=1e-6)), "(expected True)")
    print("row 1 scored alone == row 1 scored padded next to row 0:",
          bool(torch.allclose(alone[0], logits[1, :2], atol=1e-5)), "(expected True)")

    # The equivalence the docstring claims: gathered word logits vs the usual
    # formulation, IGNORE_INDEX on every piece except each word's first.
    crit = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
    piece_labels = torch.full(input_ids.shape, config.IGNORE_INDEX)
    for b, n in enumerate(lengths.tolist()):
        piece_labels[b, word_index[b, :n]] = labels[b, :n]
    loss_words = crit(logits.reshape(-1, config.NUM_TAGS), labels.reshape(-1))
    loss_pieces = crit(piece_logits.reshape(-1, config.NUM_TAGS), piece_labels.reshape(-1))
    print(f"loss over gathered words {loss_words:.6f} vs over first-piece labels "
          f"{loss_pieces:.6f}:", bool(torch.allclose(loss_words, loss_pieces)),
          "(expected True)")
