"""Pretrained BERT -> pooling -> sentiment logits.

Forward pass (default checkpoint, bert-base-uncased):
    ids [B, T] + lengths [B]
      --BERT------>  outputs [B, T, 768]  +  pooled [CLS] [B, 768]
      --head------>  logits  [B, 2]          model/head.py, unchanged

T is at most 512 WordPiece positions, [CLS] and [SEP] included;
model_bert/wordpiece.py decides which 510 pieces of a longer review survive.

WHAT BERT REPLACES. Below the head, the GloVe models are built around what
22,500 labeled reviews can teach from scratch:

    GloVe models                                BERT
    ------------------------------------------  --------------------------------
    vocab from train, min_freq=5, rest <unk>    fixed 30,522-piece vocabulary;
                                                rare words spelled from pieces
    static word vectors; all context learned    12 layers of context, pretrained
      from 22.5k binary labels                    before seeing a single label
    MAX_LEN = 400 tokens                        510 pieces

That is also why it has its own train script rather than a --model value
(see train_bert.py).

THE SAME CALL SHAPE. forward(ids, lengths) -> [B, 2] is exactly
RNNClassifier's and TransformerClassifier's, and the batches come from
dataset/imdb.py's own collate_batch. So train.py's evaluate() and
utils.metrics.compute_accuracy() score this model through the same code as the
others.

LENGTHS ARE THE ONLY PADDING SIGNAL. collate_batch pads with config.PAD_IDX,
which happens to equal BERT's [PAD] id -- but nothing here relies on that. The
attention mask is built from `lengths` with the same arange-vs-length rule
ClassifierHead uses for its mean/max masks, so the two can never disagree
about which positions are real, and the pad VALUE is irrelevant (the self-test
fills the padding with a real token id to show it).

POOLING. ClassifierHead's "last" returns whatever document vector the encoder
hands it: the final state for the recurrent models, BERT's POOLED [CLS] here --
a dense + tanh layer over the [CLS] output, pretrained through next-sentence
prediction, and the standard BERT classification input. "mean" and "max" pool
the token outputs with the head's existing masks, so the pooling question in
model/head.py can be asked of a pretrained encoder too.

Loading reports the pretraining heads (cls.*) as unused checkpoint weights.
That is expected: this model keeps the encoder and the pooler, not the
masked-LM and next-sentence classifiers.
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from model.head import ClassifierHead  # noqa: E402


class BertClassifier(nn.Module):
    """Pretrained BERT encoder + ClassifierHead for IMDB sentiment.

    Args:
        name: hub id or local directory of the checkpoint (config.BERT_NAME).
            The tokenizer MUST be loaded from the same name.
        num_classes: 2 for IMDB.
        pooling: "last" (pooled [CLS]) / "mean" / "max" (config.BERT_POOLING).
        dropout: before the linear layer (config.BERT_DROPOUT). Dropout inside
            BERT stays at the checkpoint's own setting.
    """

    def __init__(self, name, num_classes=2, pooling="last", dropout=0.1):
        super().__init__()
        self.pooling = pooling
        self.bert = AutoModel.from_pretrained(name)
        self.head = ClassifierHead(in_features=self.bert.config.hidden_size,
                                   num_classes=num_classes, pooling=pooling,
                                   dropout=dropout)

    def forward(self, ids, lengths):
        """Map piece ids [B, T] and true lengths [B] to raw logits [B, C]."""
        lengths = lengths.to(ids.device)
        # True at real positions: [CLS], the pieces, [SEP].
        attention_mask = (torch.arange(ids.size(1), device=ids.device)[None, :]
                          < lengths[:, None]).long()                      # [B, T]
        out = self.bert(input_ids=ids, attention_mask=attention_mask)
        # Token outputs [B, T, H] feed mean/max; the pooled [CLS] [B, H] is "last".
        return self.head(out.last_hidden_state, out.pooler_output, lengths)


# ---- Quick self-test: run this file directly --------------------------------
# python model_bert/bert_classifier.py     (a tiny RANDOM BERT -- downloads nothing)
if __name__ == "__main__":
    import tempfile

    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    # A 2-layer, 32-wide random BERT saved to a temp dir, so the real
    # from_pretrained path runs without fetching any weights.
    tiny = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=64)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        BertModel(tiny).save_pretrained(tmp)
        model = BertClassifier(tmp, num_classes=2).eval()

    # [PAD]=0 [CLS]=1 [SEP]=2. Row 0 is 6 positions long, row 1 is 4 + padding.
    ids = torch.tensor([[1, 5, 6, 7, 8, 2],
                        [1, 9, 10, 2, 0, 0]])
    lengths = torch.tensor([6, 4])

    for pooling in ("last", "mean", "max"):
        model.head.pooling = pooling
        with torch.no_grad():
            logits = model(ids, lengths)
            alone = model(ids[1:, :4], lengths[1:])
        print(f"[{pooling:4}] logits {tuple(logits.shape)} (expected (2, 2))  "
              f"row 1 alone == row 1 padded:",
              bool(torch.allclose(alone[0], logits[1], atol=1e-5)), "(expected True)")

    # Only `lengths` decides what is real: fill the padding with a real token.
    junk = ids.clone()
    junk[1, 4:] = 7
    with torch.no_grad():
        same = torch.allclose(model(junk, lengths), model(ids, lengths), atol=1e-5)
    print("padding filled with a real token id instead of [PAD] changes nothing:",
          bool(same), "(expected True)")
