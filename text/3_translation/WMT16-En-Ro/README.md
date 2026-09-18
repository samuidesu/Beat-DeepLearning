# WMT16 English → Romanian — BERT encoder + hand-written Transformer decoder

**Status: completed.** Both training stages and the final held-out test
evaluation are complete. Results below were finalized on **2026-09-18**.

The BERT encoder, handwritten decoder and two-stage training are implemented.
Stage 1 freezes BERT; stage 2 fine-tunes the whole model. Each epoch records
teacher-forced validation NLL/perplexity and saves checkpoints. `evaluate.py`
supports autoregressive greedy/beam decoding and corpus sacreBLEU/chrF++.

## Results and conclusions

### Recorded experiment

- Encoder: pretrained `google-bert/bert-base-cased`.
- Decoder: 6 handwritten pre-norm Transformer layers, hidden size 768,
  12 attention heads, dropout 0.1, sinusoidal positions, and tied target
  embedding/output weights. The target-side parameters start from random
  initialization; no decoder or joint encoder-decoder pretraining was used.
- Target vocabulary: 16,000 SentencePiece BPE pieces learned from train only.
- Training: seed 42, batch size 32, label smoothing 0.1, five epochs with the
  encoder frozen followed by five epochs with all parameters trainable.
- Training data: 609,828 retained pairs out of 610,320; both training length
  limits are 128 tokens including special tokens.
- Checkpoints: each stage's best validation NLL occurred at epoch index 4
  (the fifth epoch). Stage 2 started from stage 1's `best.pt`.

### Corpus scores

All rows use beam search with **4 beams**, length penalty **1.0**, and a
generation budget of **256 new tokens** including EOS, excluding BOS.
BLEU/chrF++ are case-sensitive scores on the original decoded output and
references, without Romanian cedilla/comma-below character unification.
NLL is unsmoothed, weighted by non-padding target tokens; PPL is `exp(NLL)`.

| Checkpoint | Split | Examples | sacreBLEU | chrF++ | NLL | PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 best | validation | 1,999 | 19.46 | 45.82 | 2.6200 | 13.74 |
| Stage 2 best | validation | 1,999 | 23.46 | 50.51 | 2.2144 | 9.16 |
| **Stage 2 best** | **test** | **1,999** | **22.59** | **49.90** | **2.2702** | **9.68** |

Final test evaluation has **0 empty hypotheses** and **0 outputs that reach
the generation limit without EOS**. No test examples were removed and no
reference was truncated. The test scores are slightly below validation
(-0.87 BLEU, -0.62 chrF++), with the same fixed model and decoding settings.

Metric signatures:

```text
BLEU: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF++: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

Validation reports:
[stage 1](outputs_stage1/eval_validation_beam.json),
[stage 2](outputs_stage2/eval_validation_beam.json).
Final test report with all reference/hypothesis pairs:
[eval_test_beam.json](outputs_stage2/eval_test_beam.json).
Training histories:
[stage 1](outputs_stage1/training_log.json),
[stage 2](outputs_stage2/training_log.json).

### Final test protocol

The final checkpoint is **`outputs_stage2/best.pt`**, selected by validation
NLL. The official test split has **1,999 pairs**. Its longest source has 175
tokens and its longest reference has 273 tokens, including special tokens.
One reference exceeds the original 256-token evaluation check. The saved
[test configuration](outputs_stage2/config_test.json) raises only that
reference-length check to 512, within the model's positional capacity;
generation keeps the same 256-token budget used for validation. Every source
and reference is kept complete.

The other configuration override redirects the dataset cache to
`outputs_stage2/hf_dataset_cache/`, so evaluation can run offline within the
workspace. The copied train/validation/test Arrow files were verified to have
the same SHA-256 hashes as the existing local cache. Model weights, tokenizers,
and decoding settings are unchanged; test scores are not used for tuning.

Exact evaluation command, run from this experiment directory:

```powershell
python evaluate.py --weights outputs_stage2/best.pt --config outputs_stage2/config_test.json --split test --decoding beam --num-beams 4 --length-penalty 1.0 --max-new-tokens 256 --save outputs_stage2/eval_test_beam.json
```

The [run record](outputs_stage2/eval_test_run.json) stores the command,
configuration overrides, checkpoint hash, library versions, and verification
results. Corpus scores were recomputed from the saved predictions; all 1,999
references were checked against the cleaned official test split in order.
The [console log](outputs_stage2/eval_test_beam.log) records the run. Generated
artifacts and checkpoints remain local under the ignored output directories.

### What the results establish

1. **The complete translation pipeline learns a useful baseline.** Stage 2
   improves validation BLEU by **4.00** and chrF++ by **4.70** over stage 1.
   Validation NLL decreases throughout both stages; the final stage-2
   improvement is small, from 2.2220 to 2.2144. The stage comparison includes
   both unfreezing and additional training, so it does not isolate the effect
   of encoder unfreezing alone.
2. **Translation fidelity remains a limitation.** Validation samples show
   improved sentence meaning but persistent name substitutions and omissions.
   At zero-based prediction index 191, `Bruce Billson` becomes `Larry Billson`;
   at index 1, `Petrobras` becomes `Petrobas`; at index 61, the `4G Super Voice`
   service becomes `Super Fox` and loses `4G`. Fluency alone is insufficient
   evidence that a translation preserves the source facts.
3. **Orthography contributes a measurable part of the score loss.** Stage-2
   validation output contains 4,394 cedilla-form characters (`ş/Ş/ţ/Ţ`), while
   the references contain none of those forms. A diagnostic that maps
   `ş→ș`, `Ş→Ș`, `ţ→ț`, and `Ţ→Ț` on both hypotheses and references changes
   stage-2 validation BLEU from **23.46 to 28.15** and chrF++ from **50.51 to
   53.54**. This is a different scoring policy, not an improved checkpoint;
   the main scores above retain the original strings. Full diagnostic results
   are in [validation_orthography_diagnostic.json](outputs_stage2/validation_orthography_diagnostic.json).
4. **Limited target-side pretraining is a plausible contributor to the
   remaining quality gap.** BERT is already pretrained, but the target
   embedding, decoder, cross-attention and output head learn only from the
   parallel training corpus. These experiments do not establish that missing
   decoder/joint pretraining is the principal cause: no controlled pretrained
   encoder-decoder comparison was run, and the observed orthography and
   factual errors have distinct effects on quality.

The project concludes as an educational implementation of translation data
preparation, a handwritten decoder, two-stage training, and generative
evaluation. The recorded results describe this single seed and setup;
published BLEU scores with other splits, preprocessing or tokenization are
not directly comparable.

---

The first **sequence-to-sequence** project in this repo. Everything before it
consumed a sequence and produced a label — one per document (SST-2, AG News,
IMDB) or one per token (CoNLL-2003). This one produces a **sequence of a
different length, in a different language, one token at a time, conditioned on
what it has already written**.

| project | output | what one prediction is |
| --- | --- | --- |
| SST-2 / IMDB | `[B, 2]` | is this document positive |
| AG News | `[B, 4]` | what is this article about |
| CoNLL-2003 | `[B, L, 9]` | what entity does this token belong to |
| **WMT16 En-Ro** | **`[B, T, V]`, T generated** | **which Romanian subword comes next** |

Three things follow from that change, and they shape this whole folder:

1. **There are two vocabularies**, not one. English goes in, Romanian comes
   out, and they are tokenized by different tokenizers with different ids.
2. **The output is autoregressive.** Training uses teacher forcing with a
   shifted target; inference has to generate, which is why decoding and beam
   search are separate concerns from the forward pass.
3. **Accuracy is meaningless.** A translation is scored against a reference by
   n-gram overlap over the whole corpus — sacreBLEU and chrF++.

---

## 1. Objective

English → Romanian neural machine translation, trained on parallel data.

```
English source text
    ↓
pretrained BERT encoder  (google-bert/bert-base-cased)
    ↓
contextual source representations            [B, S, H]
    ↓
hand-written Transformer decoder
    causal self-attention  +  encoder-decoder cross-attention  +  FFN
    ↓
Romanian LM head
    ↓
Romanian subword-token probabilities         [B, T, V]
```

Only the encoder is pretrained. The target embedding, the decoder, the
cross-attention and the LM head are randomly initialized and trained directly
on parallel data — **the decoder is never pretrained separately as a language
model.**

## 2. Dataset

WMT16 `ro-en`, loaded through Hugging Face `datasets`:

```python
load_dataset("wmt16", "ro-en")      # official train / validation / test
source = example["translation"]["en"]
target = example["translation"]["ro"]
```

The official train/validation/test boundaries are preserved. Training updates
use only the training split; validation selects checkpoints; the final model
is evaluated on test without using test scores to choose model or decoding
settings.

## 3–4. Why the **cased** encoder

`google-bert/bert-base-cased`. The uncased checkpoint lowercases and strips
accents inside its own normalizer, before WordPiece runs:

```
Apple → apple      a company becomes a fruit
US    → us         a country becomes a pronoun
May   → may        a month becomes a modal verb
```

Romanian keeps all three distinctions, so a model trained on folded English is
being asked to restore capitalization it never received. For classification
that loss is usually tolerable. For translation it is a direct accuracy cost on
the target side, and it is unrecoverable. `utils/config.py` refuses an uncased
name rather than accepting it quietly.

The same argument applies to the target: Romanian references are cased, BLEU is
computed against cased text, and the SentencePiece model uses `nmt_nfkc` — the
case-**preserving** normalization — not `nmt_nfkc_cf`.

## 5. Decoder

A Transformer decoder written by hand: causal self-attention, encoder-decoder
cross-attention, position-wise FFN, positional encoding, N stacked layers, an
LM head. `nn.Transformer`, `nn.TransformerDecoder`, `nn.MultiheadAttention`,
`EncoderDecoderModel`, BART, T5 and MarianMT are all off-limits — see
[`model/README.md`](model/README.md). `transformers.BertModel` is allowed for
the encoder, and only there.

## 6–7. Target tokenizer: Romanian SentencePiece BPE

Trained by `scripts/train_target_tokenizer.py` on the **Romanian side of the
TRAIN split only** — never on validation or test. The merges are model
parameters; learning them from evaluation text leaks the benchmark.

* 16,000 pieces by default (`target_tokenizer.vocab_size`)
* `model_type: bpe`, `character_coverage: 1.0`
* `normalization_rule_name: nmt_nfkc` — preserves case and the Romanian
  letters **ă â î ș ț**
* saved under `artifacts/tokenizer_ro/`, and never retrained automatically
  (retraining shifts every id and silently invalidates every checkpoint)

**The target vocabulary is a SUBWORD vocabulary.** It does not hold one entry
per Romanian word, and it could not: Romanian is morphologically rich and has
far more than 16,000 word forms. A frequent word survives whole; a rarer one is
split into the pieces the BPE merges produced:

```
"este"          →  ["▁este"]                     one token
"dezvoltării"   →  ["▁dezvolt", "ării"]          two pieces          (illustrative)
```

Nothing in this project assumes one token equals one word. Counting "words" in
a target sequence, or reading a length limit as a word count, would be wrong.

## 8. Special tokens

| token | id | meaning |
| --- | --- | --- |
| `<pad>` | 0 | padding; never contributes to the loss |
| `<unk>` | 1 | out-of-vocabulary fallback |
| `<bos>` | 2 | **B**eginning **O**f **S**equence — the decoder's first input |
| `<eos>` | 3 | **E**nd **O**f **S**equence — what the decoder emits to stop |

Ids are pinned, not discovered, and exposed as `tokenizer.pad_id`, `.unk_id`,
`.bos_id`, `.eos_id`. `<eos>` is not decoration: generation is only finite
because emitting `<eos>` is a scored decision the model is trained to make.

## 9. Teacher forcing — how the target is shifted

For Romanian pieces `y1 … yN`, the full target sequence is

```
full_target       = <bos> y1 y2 ... yN <eos>

decoder_input_ids = full_target[:-1]  =  <bos> y1 y2 ... yN
labels            = full_target[1:]   =        y1 y2 ... yN <eos>
```

Position `t` of the input is scored against position `t` of the labels, so the
model learns `<bos> → y1`, `y1 → y2`, …, `yN → <eos>`. Both tensors are `[B, T]`
and the shift is exactly one position — `scripts/sanity_check_data.py` asserts
`decoder_input_ids[:, t+1] == labels[:, t]`.

During training the decoder reads the *gold* prefix (that is what "teacher
forcing" means), which is why training is one parallel forward pass while
inference is T sequential ones.

**No causal mask is built in the data pipeline.** `collate.py` emits padding
masks only; the triangular mask belongs inside the decoder's self-attention,
and combining the two is part of implementing it.

## 10. Dynamic padding

Every batch is padded to *its own* longest member, not to a global maximum:

```
source_ids             [B, S]   S = longest source in this batch
source_attention_mask  [B, S]   1 = real token, 0 = padding
decoder_input_ids      [B, T]   T = longest decoder sequence in this batch
decoder_attention_mask [B, T]   1 = real token, 0 = padding
labels                 [B, T]   -100 in padded positions
```

WMT16 sentences are mostly short with a long tail, so fixed-width batches would
be mostly padding. Three different pad values, three different jobs:

| tensor | pad value | who handles it |
| --- | --- | --- |
| `source_ids` | BERT's `[PAD]` id | `source_attention_mask` → encoder attention |
| `decoder_input_ids` | `<pad>` = 0 | `decoder_attention_mask` → decoder attention |
| `labels` | **-100** | `CrossEntropyLoss(ignore_index=-100)` |

**Labels must not pad with `<pad>`.** Id 0 is a real class the LM head can
emit; training on it would teach the model to produce padding. `-100` is
torch's ignore sentinel and never reaches the loss.

## 11. Why the decoder needs padding at all

Autoregressive inference generates one sequence at a time and needs no physical
padding. Batched *training* does: sample A's target is 13 tokens, B's is 27,
C's is 18, and a tensor has to be rectangular. So the short rows are padded to
27, `decoder_attention_mask` marks which positions are real, and the padded
label positions are set to -100 so they contribute nothing. The decoder's
self-attention then has to mask **both** the future (causal) and the padding —
two different masks, combined inside the model.

## 12. Why naive truncation is dangerous in MT

This is the one data decision in this project that is not a matter of taste.

Truncating a classification input costs some context. Truncating a translation
pair **corrupts the supervision**:

```
source (truncated at 128):  "The Commission proposed a regulation that ..."   [cut]
target (complete):          "...care va intra în vigoare în ianuarie 2017."
```

The model is now being trained to produce content it was never shown — and on
exactly the hardest, longest examples in the corpus. It learns to hallucinate,
and it learns it from the training signal itself.

Therefore:

* **TRAIN** — an over-long pair is **dropped whole**
  (`long_train_pair_policy: filter`). Never one side, never independently.
  Dropping data the model cannot learn from is free; corrupting it is not.
* **VALIDATION / TEST** — nothing is dropped or truncated silently. Over-long
  official examples are detected, counted, reported, and by default they stop
  the run (`evaluation_overlength_policy: error`). Quietly removing the hard
  examples from a benchmark inflates the score against a corpus that is no
  longer WMT16. If you must proceed, set the policy to `warn` and say so when
  reporting the number.
* BERT has 512 positional embeddings — a hard architectural ceiling
  (`max_source_positions`). A longer source cannot be encoded at all, and that
  is checked separately from the training limits.

## 13. Length analysis

The completed run used source/target training limits of **128/128 tokens**.
The full training-corpus analysis in `artifacts/dataset_stats.json` confirms
that these limits retain **609,828 of 610,320 pairs (99.92%)** and remove 492
over-long pairs. Source/target p99 lengths are 80/84 tokens. Validation and
test are evaluated without filtering or truncation.

The analysis workflow is:

```
train the target tokenizer
        ↓
python scripts/analyze_lengths.py
        ↓
read mean / p50 / p90 / p95 / p99 / max for BOTH sides,
the counts over 64 / 128 / 256 / 512, and the pair-retention table
        ↓
choose max_train_source_length and max_train_target_length by hand
        ↓
edit configs/default.yaml, re-run train.py and read the filtering report
```

`analyze_lengths.py` writes `artifacts/dataset_stats.json` and changes no
configuration. The number to look at is **pair retention** — over-long pairs
are dropped whole, so what matters is how many *pairs* survive filtering both
sides at a given threshold. A limit that keeps ~99% of pairs costs almost no
data and bounds the memory an attention batch needs; a limit that keeps 80% is
throwing away a fifth of the corpus to save a little VRAM.

Source lengths include `[CLS]`/`[SEP]`; target lengths include `<bos>`/`<eos>`,
because those are the sequences the model actually sees.

## 14. Two-stage training

| | stage 1 | stage 2 |
| --- | --- | --- |
| BERT encoder | **frozen** | unfrozen, `encoder_lr: 2e-5` |
| target embedding, decoder, cross-attention, LM head | trained, `decoder_lr: 3e-4` | trained, `decoder_lr: 1e-4` |

Stage 1 exists because cross-attention starts as noise. Backpropagating that
noise into a pretrained encoder at 3e-4 is the fastest way to destroy what the
checkpoint knows. Once the decoder side can produce something, stage 2 unfreezes
BERT at a rate ~5× smaller than the decoder side: the encoder already knows
English, the decoder side is still learning Romanian, and they do not belong at
the same learning rate.

The completed experiment used these learning rates for five epochs per stage;
no learning-rate grid search was performed.

`Translater.freeze_encoder()` freezes BERT and keeps its dropout disabled,
including after `model.train()`. `unfreeze_all()` enables gradients for every
parameter. The optimizer excludes frozen parameters in stage 1, and has BERT
and decoder groups with separate learning rates in stage 2.

Stage 2 starts from stage 1's `best.pt` with a fresh optimizer and schedule.
Resuming a checkpoint from the same stage restores optimizer/scheduler/scaler,
the completed epoch, global step, best loss, training history and random states.
Checkpoints are saved at epoch boundaries. Without `--config`, resume uses
the saved configuration; `--epochs` specifies the total epochs for that stage.

## 15. Evaluation

* **sacreBLEU** — the standard, with its signature printed so the setup is
  reproducible
* **chrF++** — chrF with `word_order=2`. For a morphologically rich target
  like Romanian it correlates better with human judgement than BLEU, because a
  nearly-right inflection gets partial credit instead of zero.

Both are **corpus-level**: every hypothesis and reference is collected across
the whole split and scored once. Per-batch BLEU averaged over batches is a
different, systematically wrong quantity, and is not comparable to any
published number.

Predictions are decoded back into normal Romanian text first — `<pad>` and
`<bos>` dropped, everything after `<eos>` cut, subwords joined, no `▁` markers
left anywhere.

The reported metrics are sacreBLEU and chrF++; COMET was not evaluated.

Check the metric wiring today, without a model:

```powershell
python evaluate.py --self-test
```

## 16. Setup

```powershell
cd C:\code\beatDL\text\3_translation\WMT16-En-Ro
pip install -r requirements.txt
```

Then, in order:

```powershell
python scripts/inspect_dataset.py                 # look at the corpus first
python scripts/train_target_tokenizer.py          # -> artifacts/tokenizer_ro/
python scripts/analyze_lengths.py                 # -> artifacts/dataset_stats.json
python scripts/sanity_check_data.py               # 18 asserted checks, no model
python evaluate.py --self-test                    # metric wiring, no model
```

The first dataset call downloads WMT16 ro-en (~610k pairs) into the Hugging
Face cache; later runs read the cache. `scripts/train_target_tokenizer.py` is
the slow one — it reads the whole Romanian train side once.

Run the two stages in order:

```powershell
python train.py --stage 1
python train.py --stage 2 --resume outputs_stage1/best.pt
```

The default directories are `outputs_stage1` and `outputs_stage2`.
`python train.py --stage 2` also selects the default stage-1 `best.pt` automatically.
Use `--output-dir` for an independent experiment. To resume an interrupted run:

```powershell
python train.py --stage 1 --resume outputs_stage1/last.pt
python train.py --stage 2 --resume outputs_stage2/last.pt
```

Each output directory contains `best.pt`, `last.pt`, `training_log.json`,
`config.json` and `data_report.json`. Best weights are selected by validation
loss; the training loop never evaluates the test split.

Training logs distinguish `loss` (the optimization objective, including label
smoothing) from `nll` (unsmoothed negative log likelihood per target token).
Both exclude PAD positions and are weighted by the number of real target tokens.
`perplexity = exp(nll)`, with no cap at NLL 20. Validation uses no label
smoothing, so its `loss` and `nll` are equal. Training NLL uses the same forward
passes as training, with dropout enabled and weights changing between batches.

For a brief startup check (still prepares the data first):

```powershell
python train.py --stage 1 --epochs 1 --batch-size 2 --max-steps 2 --max-val-steps 2 --output-dir outputs_smoke_stage1
python train.py --stage 2 --resume outputs_smoke_stage1/best.pt --epochs 1 --max-steps 2 --max-val-steps 2 --output-dir outputs_smoke_stage2
```

These limited runs produce debugging losses, not benchmark results. The
default validation target limit is 256: the cached official validation split
has 1,999 pairs, a maximum target length of 176 including BOS/EOS, and four
targets longer than 128. Training still filters at 128; validation is kept whole.

Offline regression checks use a tiny substitute encoder and synthetic pairs:

```powershell
python -m unittest discover -s tests -v
```

They cover frozen weights, unfreezing, learning-rate groups, stage transitions,
epoch-boundary resume, decoding, corpus scoring and NLL/perplexity accounting.

Evaluate a saved checkpoint on validation data:

```powershell
python evaluate.py --weights outputs_stage2/best.pt --split validation --decoding greedy
python evaluate.py --weights outputs_stage2/best.pt --split validation --decoding beam --num-beams 4
```

Evaluation uses the checkpoint's saved configuration unless `--config` is
explicitly supplied. Choose decoding settings on validation, then run the
chosen setup once with `--split test`. `--limit 32` is available for a quick
debugging run; its output is marked as a subset, not a full benchmark score.

Generation starts from BOS and reads only the source and its own generated
prefix, never the reference target. Each source is encoded once; the decoder
recomputes the full target prefix at each step (no decoder KV cache).
PAD and BOS are suppressed as outputs, EOS ends a sequence, and the token
budget counts EOS but excludes BOS. Beam search ranks sequences by
`sum(log probabilities) / generated_length ** length_penalty`; at the length
cap, unfinished prefixes also compete. With `--num-beams 1`, beam matches greedy.

Scores, metric signatures, NLL/PPL, decoding settings and all reference/prediction
pairs are saved beside the checkpoint as `eval_validation_greedy.json` or
`eval_validation_beam.json` (use `--save` for another path). The report also counts
outputs that hit the generation limit without EOS. References are kept complete.

## 17. Layout

```
WMT16-En-Ro/
├── README.md                     this file
├── requirements.txt
├── configs/default.yaml          data, model and both training stages
├── dataset/
│   ├── preprocessing.py          whitespace cleanup, nothing destructive
│   ├── source_tokenizer.py       cased BERT WordPiece (English)
│   ├── target_tokenizer.py       Romanian SentencePiece BPE: train/load/encode/decode
│   ├── dataset.py                load → clean → encode → filter → check → DataLoader
│   └── collate.py                dynamic padding, target shift, -100 labels
├── scripts/
│   ├── inspect_dataset.py        look at raw pairs and split sizes
│   ├── train_target_tokenizer.py train the Romanian tokenizer (TRAIN split only)
│   ├── analyze_lengths.py        length distribution → artifacts/dataset_stats.json
│   └── sanity_check_data.py      18 asserted checks on one real batch
├── artifacts/                    tokenizer + statistics (generated, gitignored)
├── model/                        BERT, decoder layers and assembled Translater
├── training/
│   ├── trainer.py                loops, loss, AMP, schedule; model-independent
│   └── checkpoint.py             state, stage, RNG and tokenizer metadata
├── tests/                       offline training, decoding and metric checks
├── evaluation/metrics.py         corpus sacreBLEU + chrF++
├── utils/{config.py, seed.py}    config validation, seeding
├── train.py                      two-stage training entry point
└── evaluate.py                   greedy/beam generation + corpus scoring
```

The package is `dataset/` rather than `data/` because the repository
`.gitignore` excludes `data/` at any depth (it is where downloaded corpora go),
and because every sibling project in this repo uses the same name.
