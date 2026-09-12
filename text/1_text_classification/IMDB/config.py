"""Central configuration for RNN and Transformer sentiment classification on IMDB.

THIRD TEXT PROJECT of this repo. Structure inherited unchanged from the SST-2
and AG-News siblings -- same embedding/encoder/head split, same two-stage
layered-LR finetune, same per-epoch JSON log + curve plots, same hand-written
Transformer. What changes is the one variable this project exists to push:

    project   classes  train docs   median length   what it stresses
    -------   -------  -----------  --------------  ----------------------
    SST-2     2         67k            7 tokens     nothing; too short
    AG News   4        114k           44 tokens     moderate context
    IMDB      2         22.5k        201 tokens     LONG-RANGE memory

Read that table as two changes at once, because both matter:

  1. DOCUMENTS ARE 5x LONGER. With "last" pooling the prediction depends on a
     hidden state that survived a walk of several hundred steps, which is
     exactly the regime the gated cells were invented for. On SST-2 the three
     cells landed within 2 points of each other; here they should not.

  2. THERE IS 10x LESS SUPERVISION. 22,500 binary labels is 22.5 kbit of
     training signal, against AG News' 114,000 four-way labels (228 kbit).
     Every capacity decision below is really a decision about that ratio --
     see the MIN_FREQ and HIDDEN_SIZE notes, which point at the same
     conclusion from opposite ends.

And the task itself is different in kind. AG News is largely LEXICAL --
"midfielder" means Sports wherever it appears, which is why a bag of words
already scores ~89%. Sentiment is COMPOSITIONAL and frequently reversed late:
"the acting is superb, the cinematography gorgeous -- and none of it saves a
script this lazy" is a negative review made almost entirely of positive words.
That is why the pooling discussion in model/head.py is longer here than in
either sibling.

Everything else in the protocol is held fixed on purpose, so a difference in
the results is attributable to the data rather than to retuned knobs.

IMDB publishes labeled test data, exactly like AG News, so the same
methodological guard applies: 10% of train is held out (stratified, seeded)
for per-epoch monitoring and best-checkpoint selection, and test.csv is read
exactly once, at the end, by eval.py.

The recurrent cell is selectable (config.CELL or train.py --cell):

    "rnn"  -- vanilla Elman RNN: h_t = tanh(W x_t + U h_{t-1})
    "lstm" -- long short-term memory (input/forget/output gates + cell state)
    "gru"  -- gated recurrent unit (2 gates, no separate cell state)

plus train.py --model transformer for the hand-written encoder. Each writes to
its OWN output folder, so the experiments never overwrite each other.

A pretrained model sits outside both switches: BERT, finetuned by
train_bert.py (model_bert/ -> outputs_bert/). It is not a --model value
because almost nothing around its encoder carries over -- no vocabulary, no
GloVe, a 512-piece window instead of MAX_LEN, a different optimizer and
schedule -- while the parts that make the numbers comparable (the split, the
truncation arithmetic, the loss, the val metric) are imported from this
pipeline rather than copied.

Every number quoted in the comments below was measured on this machine's copy
of the corpus, not estimated. The README records how.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../IMDB).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Corpus + word-vector location. The consolidated IMDB csv files are 66 MB and
# live with this project; GloVe is 862 MB zipped, so it is worth REUSING across
# text projects: scan the sibling projects under text/1_text_classification/
# first (SST-2 already downloaded it) and only fall back to our own
# dataset/data/ when no sibling copy exists.
_LOCAL_DATA = os.path.join(PROJECT_ROOT, "dataset", "data")
_SIBLING_ROOT = os.path.dirname(PROJECT_ROOT)          # .../1_text_classification


def _sibling_data_dirs():
    """Every <sibling project>/dataset/data directory next to this project."""
    if not os.path.isdir(_SIBLING_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(_SIBLING_ROOT)):
        cand = os.path.join(_SIBLING_ROOT, name, "dataset", "data")
        if os.path.isdir(cand) and os.path.normpath(cand) != os.path.normpath(_LOCAL_DATA):
            out.append(cand)
    return out


DATA_ROOT = _LOCAL_DATA

# Consolidated corpus folder: <DATA_ROOT>/imdb/{train,test}.csv. The published
# release is 50,000 separate .txt files; dataset/imdb.py folds them into two
# csv at download time -- see that file for why.
IMDB_DIR = os.path.join(DATA_ROOT, "imdb")

# Word-vector width, and the GloVe filename DERIVED from it -- the two must
# agree, so build the name instead of writing it twice.
EMBED_DIM = 100
GLOVE_NAME = f"glove.6B.{EMBED_DIM}d.txt"

# Where glove.6B.<dim>d.txt lives. First sibling copy wins (normally
# ../SST-2/dataset/data/glove/), else our own data dir -> download.
GLOVE_DIR = os.path.join(DATA_ROOT, "glove")
for _cand in _sibling_data_dirs():
    if os.path.isfile(os.path.join(_cand, "glove", GLOVE_NAME)):
        GLOVE_DIR = os.path.join(_cand, "glove")
        break
GLOVE_PATH = os.path.join(GLOVE_DIR, GLOVE_NAME)

# Logs, curves, checkpoints, vocab -- one folder per model.
OUTPUT_DIR_RNN = os.path.join(PROJECT_ROOT, "outputs_rnn")
OUTPUT_DIR_LSTM = os.path.join(PROJECT_ROOT, "outputs_lstm")
OUTPUT_DIR_GRU = os.path.join(PROJECT_ROOT, "outputs_gru")
OUTPUT_DIR_TRANSFORMER = os.path.join(PROJECT_ROOT, "outputs_transformer")


def output_dir_for_cell(cell: str) -> str:
    """Map a cell name ("rnn" / "lstm" / "gru") to its output folder."""
    return {"rnn": OUTPUT_DIR_RNN,
            "lstm": OUTPUT_DIR_LSTM,
            "gru": OUTPUT_DIR_GRU}[cell]


# -----------------------------------------------------------------------------
# Dataset: IMDB (Maas et al. 2011, "Large Movie Review Dataset")
# -----------------------------------------------------------------------------
# Full-length movie reviews labeled by their star rating, with the middle of
# the scale deliberately removed (<=4 negative, >=7 positive, 5-6 dropped):
#     train.csv  25,000 rows (12,500 per class)
#     test.csv   25,000 rows (12,500 per class)
# Exactly balanced, so accuracy is the honest metric and chance is 0.50. The
# empty middle is also why this task is easier than its length suggests --
# there are no lukewarm reviews sitting on the fence.
CLASS_NAMES = ["negative", "positive"]
NUM_CLASSES = len(CLASS_NAMES)  # 2

# Validation split carved out of train.csv -> 22,500 train / 2,500 val.
# 10% rather than AG News' 5%, because train.csv is a fifth the size there:
# 5% would leave 1,250 reviews, whose +/-1.8-point 95% interval is too wide to
# select a checkpoint on. 2,500 narrows that to +/-1.3 points, and it still is
# not tight -- which is worth remembering when two epochs are half a point
# apart. The price is 10% of the training data, and that is the real cost of
# refusing to select on test.
VAL_RATIO = 0.10
SPLIT_SEED = 1234   # deliberately separate from SEED: the data split must not
                    # move when the training seed changes

# Special tokens. Ids are pinned: <pad>=0 so padding_idx=0 works everywhere,
# <unk>=1 for words missing from the vocabulary at inference time.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

# Vocabulary is built from the TRAIN portion only (never from val/test).
# min_freq=5, against AG News' 2 and SST-2's 1, and this is the single most
# important capacity decision in the file. Measured on the cleaned train
# portion (22,500 reviews, 6,067,532 tokens, 83,232 distinct types):
#
#     min_freq   types    token coverage   embedding params
#            1   83,232        100.00%           8.32M
#            2   49,559         99.44%           4.96M
#            3   38,982         99.10%           3.90M
#            5   29,107         98.54%           2.91M   <- chosen
#           10   19,529         97.50%           1.95M
#           20   12,634         95.95%           1.26M
#
# 40.5% of all types appear EXACTLY ONCE. Keeping them would nearly triple the
# embedding table for rows that receive one gradient update per epoch, against
# 22,500 training labels. min_freq=5 gives up 1.5% of tokens to <unk> -- which
# is not even a pure loss, since it TEACHES the model what <unk> looks like
# instead of leaving that vector at its initialization.
MIN_FREQ = 5
MAX_VOCAB_SIZE = None  # None = no cap

# Truncate long reviews to this many tokens. Measured on the cleaned train
# portion: mean 269.7, p25 146, median 201, p75 328, p90 527, p95 688,
# p99 1042, max 2737. The trade-off at batch_size=64:
#
#     MAX_LEN   reviews whole   tokens kept   RNN cost   attention cost
#         256          64.0%         71.8%       2.00x            4.00x
#         400          82.3%         85.5%       3.12x            9.77x   <-
#         512          89.4%         91.3%       4.00x           16.00x
#         800          96.8%         97.8%       6.16x           38.01x
#     (costs relative to MAX_LEN=128, which is what AG News used)
#
# 400 is where the curves cross: it keeps 82% of reviews WHOLE and 86% of all
# tokens, and going to 512 buys 5.8 more points of token coverage for 28% more
# recurrent compute and 64% more attention. It is also long enough that the
# vanilla RNN genuinely has to remember something.
#
MAX_LEN = 400

# WHICH 400 tokens to keep, for the 17.7% of reviews that are longer.
#
# "head"       the first MAX_LEN tokens. What SST-2 and AG News do, and what
#              is right THERE: an AG News row is "headline + lead paragraph",
#              so the topic is settled in the first dozen words.
# "head_tail"  the first HEAD_LEN tokens plus the last (MAX_LEN - HEAD_LEN),
#              dropping the middle. The default HERE, because an IMDB review
#              is an ARGUMENT and its verdict routinely lands in the closing
#              sentence -- "...but overall, a complete waste of two hours".
#              Head-only truncation throws away exactly the sentence a human
#              skimmer would read first.
#
# The cost is a SEAM: the token at HEAD_LEN-1 and the one after it are not
# really adjacent, but every downstream component treats them as if they were
# (the RNN carries its state straight across, and the Transformer's positional
# encoding numbers the joined sequence 0..399 with no gap). No separator token
# is inserted -- it would need a vocabulary entry with no GloVe vector and its
# own justification. In exchange the model sees both ends of every review.
#
# HEAD_LEN=300 is the smaller change from head-only: it keeps most of the
# opening and adds the closing ~100 tokens that were missing. The literature
# points the other way -- Sun et al., "How to Fine-Tune BERT for Text
# Classification", found head 128 + tail 382 (25/75) best on this exact
# corpus -- but that is BERT at 512, not a BiLSTM at 400, so it is a
# hypothesis rather than a setting to copy. --head-len 100 reproduces their
# ratio and is the first sweep worth running.
TRUNCATION = "head_tail"   # "head" | "head_tail"; train.py --truncation wins
HEAD_LEN = 300             # only used when TRUNCATION == "head_tail"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Recurrent cell: "rnn" (vanilla Elman) / "lstm" / "gru". train.py --cell wins.
CELL = "lstm"

# Hidden width PER DIRECTION (train.py --hidden-size overrides). Bidirectional
# doubles the feature the head sees. Measured parameter counts at vocab=29,109:
#
#     BiLSTM h=128 x2   3.54M total  =  2.91M embedding + 0.63M encoder
#     BiLSTM h=256 x2   5.22M total  =  2.91M embedding + 2.31M encoder  <-
#     BiLSTM h=384 x2   7.95M total  =  2.91M embedding + 5.04M encoder
#
# 256 is kept from both sibling projects, so that a difference in the results
# is attributable to the data. But note what the table actually says: the
# EMBEDDING TABLE is 56% of the model, and it is the same size whatever
# HIDDEN_SIZE does. The real capacity knob on this dataset is MIN_FREQ above,
# not this constant -- halving the hidden width removes 1.7M parameters,
# while raising min_freq from 5 to 20 removes 1.7M as well and costs less
# accuracy. Worth an ablation (--hidden-size 128) if the curves overfit.
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True

# Dropout on the embeddings, BETWEEN stacked RNN layers, and before the
# classifier. 0.5, unchanged from both siblings. This is the project in the
# group most likely to NEED it: 5.22M parameters against 22,500 binary labels
# is by far the worst parameter-per-bit ratio of the three.
DROPOUT = 0.5

# Transformer defaults (train.py --model transformer).
# The embedding width stays EMBED_DIM; a trainable projection maps it to dim.
# d=128 / 4 heads / 2 layers is the configuration that won on AG News, carried
# over unchanged. Measured: 3.32M total = 2.91M embedding + 0.41M encoder --
# the encoder is a SIXTH the size of the BiLSTM's, which is worth keeping in
# mind before reading anything into a head-to-head comparison.
#
# What changes on its own is compute, not parameters: attention is O(L^2), so
# L=400 instead of AG News' L=128 does ~10x the attention work per review.
TRANSFORMER_DIM = 128
TRANSFORMER_GROUP = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_POOLING = "mean"

# Where the LayerNorm sits: "post" = LN(x + Sublayer(x)), the 2017 original;
# "pre" = x + Sublayer(LN(x)), what everything since ~2020 uses. train.py
# --norm overrides.
#
# "post" stays the default so every existing outputs_transformer* run remains
# reproducible and comparable. It is also what produced the finding that made
# this switch worth having: at dim=256 / 4 layers with no warmup, the Post-LN
# stack sat at val_acc = 0.5000 (loss = ln 2, i.e. one class for everything)
# for FIVE epochs before escaping, and still finished below the 2-layer model.
# The 2-layer / dim=128 default never showed it -- the failure is a function of
# DEPTH, which is exactly what the Post-LN literature predicts.
TRANSFORMER_NORM = "post"

# How the variable-length token features become ONE document vector:
#     "last" -- RNN final state / Transformer last real-token feature
#     "max"  -- element-wise max over time (masked)
#     "mean" -- masked average over time
# "last" stays the RNN default so the cell comparison runs under the setting
# that stresses the recurrence hardest -- and on 400-token reviews that is a
# real stress test rather than the formality it was on SST-2. See head.py for
# why the pooling choice is less obvious on sentiment than on topic.
POOLING = "last"

# GloVe pretrained embeddings = this project's "ImageNet backbone". Set False
# (or pass --no-glove) to train the word vectors from scratch. Measured
# coverage of the min_freq=5 vocabulary: 27,002/29,109 types (92.76%) and
# 96.82% of tokens.
USE_GLOVE = True

# -----------------------------------------------------------------------------
# Training -- same two-stage layered-LR protocol as SST-2 and AG News
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"

# 64 rather than AG News' 128, and the reason is OPTIMIZATION, not memory:
# 22,500 reviews at batch 64 is only 352 steps per epoch (AG News had 891), so
# a larger batch would leave this model with very few updates in a 9-epoch
# budget. Memory is not the constraint -- at L=400 the attention scores are
# ~0.66 GB retained for backward, comfortable on any modern card.
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128

# 0 = load in the main process. The whole corpus is tokenized into python
# lists in __init__, so a worker would only pay Windows' process-spawn cost.
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4

# Gradient-norm clipping. Not optional for RNNs, and least optional here:
# backprop through time multiplies by the same recurrent Jacobian at every
# step, and these sequences run to 400 steps -- three times the longest in
# either sibling project.
GRAD_CLIP = 5.0

# Label smoothing (0 disables): softens the CE target, a mild regularizer.
LABEL_SMOOTHING = 0.05

# Stage 1: FREEZE the embedding table; the from-scratch encoder + head learn
#          to read fixed GloVe vectors.
# Stage 2: unfreeze everything, three LR tiers (embedding slowest, head
#          fastest) so the pretrained vectors drift instead of being wrecked.
# 6 + 3 rather than AG News' 5 + 3: an epoch here is 352 steps against AG
# News' 891, so the same number of epochs is far less training. Note that
# IMDB also overfits sooner than either sibling -- best-checkpoint selection
# on val is what protects the reported number, so a couple of wasted epochs at
# the end cost time, not accuracy.
STAGE1_EPOCHS = 6
STAGE1_LR_HEAD = 1e-3
STAGE1_LR_ENCODER = 1e-3        # also from scratch -> same tier as the head

STAGE2_EPOCHS = 3
STAGE2_LR_HEAD = 3e-4
STAGE2_LR_ENCODER = 3e-4
STAGE2_LR_EMBEDDING = 5e-5      # pretrained words: gentle updates only

# -----------------------------------------------------------------------------
# BERT finetuning (train_bert.py, model_bert/)
# -----------------------------------------------------------------------------
OUTPUT_DIR_BERT = os.path.join(PROJECT_ROOT, "outputs_bert")

# Hub id or local directory; the tokenizer is always loaded from the same name.
# UNCASED, because casing is not the variable under study: the GloVe models
# lowercase every token (GloVe 6B is uncased), so a cased BERT would change two
# things at once. Contrast the CoNLL-2003 project, where casing IS the task.
BERT_NAME = "google-bert/bert-base-uncased"

# The window, in WordPiece positions with [CLS] and [SEP] included. 512 is the
# checkpoint's hard limit (its position table has 512 rows), not a trade-off
# like MAX_LEN above, so a review gets at most 510 pieces. Longer ones are cut
# by the SAME truncate() the GloVe models use, in the TRUNCATION mode above.
BERT_MAX_LEN = 512

# Pieces kept from the front under head_tail: 128, plus 382 from the back --
# the split Sun et al. found best on this corpus. HEAD_LEN's note above calls
# that ratio a hypothesis for a BiLSTM at 400; here it is the setting it was
# measured with, BERT at 512.
BERT_HEAD_LEN = 128

# Dropout before the classifier, equal to BERT's own hidden_dropout_prob.
# Dropout inside the 12 layers stays at the checkpoint's setting (also 0.1).
BERT_DROPOUT = 0.1

# ClassifierHead pooling. "last" is the encoder's own document vector, which
# for BERT is the POOLED [CLS] -- the standard BERT classification input.
# "mean" / "max" pool the token outputs with the head's existing masks.
BERT_POOLING = "last"

# Optimization, from Devlin et al.'s finetuning grid (batch 16/32, lr
# 5e-5/3e-5/2e-5, epochs 2/3/4):
#   lr 2e-5    the bottom of the grid. Sun et al. found a rate this low
#              necessary for BERT to avoid catastrophic forgetting on IMDB; an
#              aggressive 4e-4 did not converge.
#   batch 32   the larger of the grid's two sizes, and deliberately not
#              BATCH_SIZE's 64: that was picked to give small from-scratch
#              models more updates, while a 512-position row through 12 layers
#              is a different memory budget. 22,500 / 32 = 704 updates per epoch.
#   4 epochs   the top of the grid; best.pt is selected on val anyway.
#
# One learning rate for every parameter and no frozen stage, unlike the
# two-stage protocol above. That protocol exists because the GloVe models put
# a large RANDOM encoder on pretrained vectors; here only the 1,538-parameter
# classifier is random. What protects the pretrained weights instead is
# WARMUP: the rate ramps up from 0 over the first 10% of updates, so the
# earliest steps -- taken while the classifier is still random and Adam's
# moment estimates are still noise -- are small ones.
BERT_BATCH_SIZE = 32
BERT_EPOCHS = 4
BERT_LR = 2e-5
BERT_WARMUP_RATIO = 0.1

# AdamW's decoupled decay, on weight matrices only: biases and LayerNorm
# parameters are exempt, as in the original BERT optimizer.
BERT_WEIGHT_DECAY = 0.01

# The original BERT value, rather than the GloVe models' GRAD_CLIP = 5.0.
BERT_GRAD_CLIP = 1.0
