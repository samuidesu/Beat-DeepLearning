# model/ — translation model components

[`bert_encoder.py`](bert_encoder.py) provides the pretrained `BertEncoder`.
[`transformer_naive.py`](transformer_naive.py) contains the handwritten decoder
layer. [`translation.py`](translation.py) assembles them in `Translater`, and
`build_model()` constructs it from the `model` section of the YAML configuration.

The data and training code accesses the model through the interface below.

## Using the BERT encoder

```python
from model import BertEncoder

# Match the source tokenizer's checkpoint. Use cached weights only here.
encoder = BertEncoder(cfg["source_tokenizer"]["name"], local_files_only=True)
cross_X = encoder(source_ids, source_attention_mask)  # [B, S, 768]
cross_padding = source_attention_mask == 0           # [B, S], True = PAD
```

The encoder returns all final token states, including special tokens and PAD
positions. Cross-attention must use `cross_padding` to ignore PAD positions.
BERT handles its own positional embeddings; the target decoder still needs
its own position information. Set the current decoder's `dim` to
`encoder.hidden_size` so its K/V projections accept the encoder output.

Parameters are trainable by default. On the assembled model,
`model.freeze_encoder()` freezes `model.bert` and keeps it in evaluation mode
even after `model.train()`. `model.unfreeze_all()` restores gradients for all
parameters and re-enables BERT dropout when training.

## What must not be used here

Not as a shortcut, not as a "temporary" placeholder, not hidden behind a
wrapper:

- `nn.Transformer`, `nn.TransformerDecoder`, `nn.TransformerDecoderLayer`
- `nn.MultiheadAttention`
- `transformers.EncoderDecoderModel`
- BART, T5, mBART, MarianMT, or any other ready-made seq2seq checkpoint

`transformers.BertModel` **is** allowed, and only as the pretrained encoder —
that is the one component that is supposed to be pretrained.

## Components

| Component | Note |
|---|---|
| BERT encoder wrapper | implemented in `bert_encoder.py`; takes `source_ids`, `source_attention_mask`; returns `[B, S, H]` |
| Target embedding | random init, `target_vocab_size` rows, `padding_idx=0` (`<pad>`) |
| Positional encoding | sinusoidal, learned, or RoPE — your choice |
| Causal self-attention | must combine the causal mask with `decoder_attention_mask` |
| Cross-attention | queries from the decoder, keys/values from the encoder, masked by `source_attention_mask` |
| FFN | position-wise, inside each decoder layer |
| Decoder layer | implemented in `transformer_naive.py`: self-attention → cross-attention → FFN, with residuals and norm |
| Decoder stack | N layers |
| LM head | `[B, T, H]` → `[B, T, target_vocab_size]`; weight tying with the target embedding is optional |
| Seq2Seq module | assembles the above and satisfies the interface below |

## The interface the rest of the project assumes

```python
logits = model(
    source_ids=source_ids,                          # [B, S] int64, [CLS] ... [SEP] + pad
    source_attention_mask=source_attention_mask,    # [B, S] 1 = real token, 0 = pad
    decoder_input_ids=decoder_input_ids,            # [B, T] int64, <bos> y1 ... yN + pad
    decoder_attention_mask=decoder_attention_mask,  # [B, T] 1 = real token, 0 = pad
)
# logits: [B, T, target_vocab_size]
```

`training/trainer.py` then computes
`CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, V), labels.reshape(-1))`.
The loss code already exists; it is waiting for `logits`.

`build_model(cfg, target_vocab_size)` in [`__init__.py`](__init__.py) returns
the assembled model. `Translater` includes sinusoidal target positions and a
final LayerNorm for the pre-norm decoder stack.

For generation, `model.encode(source_ids, source_attention_mask)` returns
source memory `[B, S, dim]` once. `model.decode(memory, source_attention_mask,
decoder_input_ids, decoder_attention_mask, last_token_only=True)` reuses it
and returns `[B, 1, target_vocab_size]` for the next-token decision. The regular
`forward()` still returns all positions for training. Neither interface caches
decoder K/V tensors.

## Two things the data pipeline deliberately does not do for you

1. **No causal mask.** `collate.py` produces padding masks only. Building the
   triangular mask belongs to the decoder layer. Convert the collate mask with
   `padding_mask = (decoder_attention_mask == 0)` before passing it to that layer.
2. **No cross-attention mask.** You get `source_attention_mask` as `[B, S]`
   ones and zeros. Broadcasting it into whatever shape your attention
   implementation needs is a model decision.

## Also left for later

- Optional decoder KV caching. Greedy and beam decoding are implemented in
  `evaluate.py` using cached source memory and full target prefixes.
