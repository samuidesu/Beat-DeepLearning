"""Central configuration for BiLSTM and Transformer NER on CoNLL-2003.

FIRST SEQUENCE-LABELING PROJECT of this repo. Everything before it -- SST-2,
AG News, IMDB -- produced ONE label per document. This one produces one label
PER TOKEN, and almost every difference in this folder follows from that single
change:

    project    output shape        what one prediction is
    --------   -----------------   --------------------------------------
    SST-2      [B, 2]              is this sentence positive
    AG News    [B, 4]              what is this article about
    IMDB       [B, 2]              is this review positive
    CoNLL-03   [B, L, 9]           what entity does THIS TOKEN belong to

The architecture barely moves: embedding -> encoder -> linear. What moves is
everything around it, and the four changes below are the whole project.

  1. THE HEAD STOPS POOLING. A classifier collapses [B, L, F] into [B, F], and
     that pooling choice was a real decision -- three separate IMDB
     experiments went into it. A tagger keeps all L positions, so head.py has
     no pooling at all: it is a linear layer applied at every step. This is
     the sequence-labeling counterpart of the segmentation projects' per-pixel
     head.

  2. PADDING BECOMES A LOSS PROBLEM, NOT JUST A POOLING PROBLEM. With one
     label per document, padded positions never reached the loss. Here they
     would -- a padded batch has a label slot at every position. Padding
     labels are set to IGNORE_INDEX and the loss skips them, which is exactly
     the ignore_index=255 mechanism the VOC segmentation projects used for
     unlabeled pixels.

  3. THE METRIC STOPS BEING ACCURACY. Token accuracy is a broken metric here,
     and it is worth seeing why before reading any result: 83% of tokens are
     tagged "O", so a model that predicts "O" everywhere -- one that finds no
     entities whatsoever -- scores 83% token accuracy. The benchmark's metric
     is ENTITY-LEVEL F1 with exact boundary AND type match: tagging
     "European Commission" as ORG when the gold span is "European Commission"
     is one true positive, while getting the type right and the boundary
     wrong ("European" alone) scores zero, not partial credit. See
     utils/metrics.py.

  4. CAPITALIZATION STOPS BEING NOISE AND BECOMES THE STRONGEST FEATURE. The
     three classification projects lowercase every token, because GloVe 6B is
     uncased and lowercasing is what makes a word findable at all. On NER that
     same step deletes the most informative signal in the input: the
     difference between "apple" and "Apple" IS the task. The fix here keeps
     lowercasing for the GloVe lookup AND feeds the casing back in separately
     as its own small embedding -- see CASE_CLASSES below.

Two models, selected with train.py --model:

    "lstm"         BiLSTM tagger                    -> outputs_lstm/
    "transformer"  Transformer encoder              -> outputs_transformer/

Deliberately only two. The rnn/gru cells from the classification projects are
dropped: IMDB already answered what a vanilla RNN does on long inputs, and
CoNLL sentences average 14 tokens, so the memory-horizon question that made
that comparison interesting does not exist here.

A third model sits outside that switch: pretrained BERT, finetuned by
train_bert.py (model_bert/ -> outputs_bert/). It is not a --model value
because almost nothing around its encoder carries over -- no vocabulary, no
GloVe, no case feature, a different optimizer and schedule -- while the parts
that make the numbers comparable (the BIO2 labels, the loss, the dev metric)
are imported from this pipeline rather than copied.

Numbers quoted below are either (a) published in Tjong Kim Sang & De Meulder
(2003) and re-checked by the loader at download time, or (b) measured on this
machine. Each is labeled. Nothing here is an estimate.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../2_sequence_labeling/CoNLL-2003).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Corpus + word-vector location. CoNLL-2003 is 4.8 MB and lives with this
# project; GloVe is 347 MB unpacked, so it is worth REUSING.
#
# The search is WIDER than the classification projects', because this project
# sits one directory level further out. IMDB scanned its siblings inside
# text/1_text_classification/; here the existing copy lives in a DIFFERENT
# category (text/1_text_classification/SST-2/), so the scan has to walk
# text/<category>/<project>/dataset/data instead.
_LOCAL_DATA = os.path.join(PROJECT_ROOT, "dataset", "data")
_CATEGORY_ROOT = os.path.dirname(PROJECT_ROOT)          # .../2_sequence_labeling
_TEXT_ROOT = os.path.dirname(_CATEGORY_ROOT)            # .../text


def _peer_data_dirs():
    """Every <category>/<project>/dataset/data directory under text/.

    Sorted so the pick is deterministic, and the local dir is excluded so a
    half-created dataset/data/glove/ here can never shadow a complete copy
    elsewhere.
    """
    out = []
    if not os.path.isdir(_TEXT_ROOT):
        return out
    for category in sorted(os.listdir(_TEXT_ROOT)):
        cat_dir = os.path.join(_TEXT_ROOT, category)
        if not os.path.isdir(cat_dir):
            continue
        for project in sorted(os.listdir(cat_dir)):
            cand = os.path.join(cat_dir, project, "dataset", "data")
            if (os.path.isdir(cand)
                    and os.path.normpath(cand) != os.path.normpath(_LOCAL_DATA)):
                out.append(cand)
    return out


DATA_ROOT = _LOCAL_DATA

# Corpus folder: <DATA_ROOT>/conll2003/{train,valid,test}.txt.
CONLL_DIR = os.path.join(DATA_ROOT, "conll2003")

# Word-vector width, and the GloVe filename DERIVED from it -- the two must
# agree, so build the name instead of writing it twice.
EMBED_DIM = 100
GLOVE_NAME = f"glove.6B.{EMBED_DIM}d.txt"

# Where glove.6B.<dim>d.txt lives. First peer copy wins (normally
# ../../1_text_classification/SST-2/dataset/data/glove/), else our own data
# dir -> download.
GLOVE_DIR = os.path.join(DATA_ROOT, "glove")
for _cand in _peer_data_dirs():
    if os.path.isfile(os.path.join(_cand, "glove", GLOVE_NAME)):
        GLOVE_DIR = os.path.join(_cand, "glove")
        break
GLOVE_PATH = os.path.join(GLOVE_DIR, GLOVE_NAME)

# Logs, curves, checkpoints, vocab -- one folder per model.
OUTPUT_DIR_LSTM = os.path.join(PROJECT_ROOT, "outputs_lstm")
OUTPUT_DIR_TRANSFORMER = os.path.join(PROJECT_ROOT, "outputs_transformer")


def output_dir_for_model(model: str) -> str:
    """Map a model name ("lstm" / "transformer") to its default output folder."""
    return {"lstm": OUTPUT_DIR_LSTM,
            "transformer": OUTPUT_DIR_TRANSFORMER}[model]


# -----------------------------------------------------------------------------
# Dataset: CoNLL-2003 English (Tjong Kim Sang & De Meulder, 2003)
# -----------------------------------------------------------------------------
# Reuters newswire from August 1996 - August 1997, hand-annotated with four
# entity types. THE reference benchmark for sequence labeling: essentially
# every NER paper since 2003 reports on it, so the numbers this project
# produces are directly comparable to published ones (see the README).
#
# Published counts (Tjong Kim Sang & De Meulder 2003, Table 2). The loader
# re-checks all of them at load time and warns on a mismatch, which is how a
# truncated download or a re-encoded mirror gets caught:
#
#     split          articles  sentences   tokens    LOC   MISC    ORG    PER
#     train               946     14,041  203,621   7,140  3,438  6,321  6,600
#     valid (testa)       216      3,250   51,362   1,837    922  1,341  1,842
#     test  (testb)       231      3,453   46,435   1,668    702  1,661  1,617
#
# The sentence column is NOT Table 2's, which reports 14,987 / 3,466 / 3,684 by
# counting every `-DOCSTART-` line as a sentence. Subtract the article counts
# and you get the numbers above -- the real ones, and what every modern source
# reports. See _EXPECTED in dataset/conll2003.py for the full note.
#
# NOTE THE SPLIT NAMES. Unlike IMDB and AG News, this corpus ships its OWN
# development set, so there is no stratified split to carve and no SPLIT_SEED
# in this file. "testa" is the dev set everyone tunes on; "testb" is the test
# set everyone reports.
#
# They are TIME-SEPARATED, not randomly split -- testb is drawn from December
# 1996 while train is August 1996 -- so the usual train/test gap here includes
# genuine distribution shift: different news stories, different people in the
# news, different organizations. That is a property of the benchmark, and part
# of why test F1 sits several points BELOW dev F1 for every model ever
# published on it. A random split would not behave that way, and this is worth
# remembering before reading a dev/test gap here as overfitting.
#
# The entity types, and what makes each hard:
#   PER   people. Easiest: reliably capitalized, and first names are a
#         relatively closed set that GloVe already knows.
#   LOC   locations. Also easy in isolation, but collides with ORG constantly
#         -- a country name is LOC in "exports from Germany" and ORG in
#         "Germany beat Argentina" (the national team). Reuters ran a lot of
#         sports results in 1996, so this collision is not a corner case.
#   ORG   organizations. Hardest of the three, for the reason above.
#   MISC  everything else that is a named thing: nationalities ("German"),
#         events ("World Cup"), adjectival forms. Semantically incoherent by
#         construction, and consequently the lowest-F1 class in every
#         published result.
ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]

# The tag set, in BIO2. Order is pinned: "O" is id 0 so a zero-initialized
# tensor reads as "no entity", and each type's B-/I- pair is adjacent so the
# confusion matrix has a readable block structure.
TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
NUM_TAGS = len(TAGS)  # 9

# BIO2 ("IOB2") is NOT the encoding the files ship in. The released corpus is
# IOB1, where B- appears ONLY to split two adjacent entities of the same type,
# and an entity otherwise STARTS with I-:
#
#     IOB1 (what the file says)    IOB2 (what this project uses)
#     EU        I-ORG              EU        B-ORG
#     rejects   O                  rejects   O
#     German    I-MISC             German    B-MISC
#
# dataset/conll2003.py converts on load. This is not cosmetic: under IOB1 the
# model has to learn that I-ORG sometimes begins an entity and sometimes
# continues one, so the first token of every entity is ambiguous with its own
# continuation. Every modern result on this corpus is BIO2, so reporting IOB1
# numbers would also be silently incomparable with the literature.

# Padding a batch creates label slots that correspond to no token. They are
# filled with this value, and nn.CrossEntropyLoss(ignore_index=...) drops
# them. -100 is torch's own default sentinel, which is why it is used rather
# than the segmentation projects' 255 -- there it had to be a valid uint8
# pixel value; here it only has to fall outside [0, NUM_TAGS).
IGNORE_INDEX = -100

# Special tokens. Ids are pinned: <pad>=0 so padding_idx=0 works everywhere,
# <unk>=1 for words missing from the vocabulary at inference time.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

# Vocabulary is built from the TRAIN split only (never from valid/test).
#
# min_freq=1, against IMDB's 5 -- the opposite end of the range, for a reason
# specific to this task. On IMDB, types appearing once were mostly typos and
# 40% of the vocabulary; dropping them cost 1.5% of tokens and saved 5.4M
# parameters. Here the words that appear once ARE THE ANSWER: a person named
# in one article and nowhere else is exactly what the model has to tag, and
# mapping it to <unk> during training would teach the model that the
# interesting tokens are the unreadable ones. The whole corpus is 204k
# training tokens, so keeping every type costs a fraction of what it cost on
# IMDB.
#
# The corresponding risk is that test-time OOV is high -- names in December
# 1996 news were not in August 1996 news -- which is precisely what the case
# feature below exists to survive.
MIN_FREQ = 1
MAX_VOCAB_SIZE = None  # None = no cap

# Cap on sentence length, set ABOVE the longest sentence in the corpus so that
# it never actually fires. That is a different intent from the classification
# projects, where MAX_LEN was a real trade-off knob (IMDB cut 18% of its
# reviews on purpose), and the reason is that truncation means something worse
# here: a document classifier that loses its tail loses CONTEXT, while a
# tagger that loses its tail loses LABELS -- gold entities silently vanish
# from both the loss and the score, so the reported F1 is computed against a
# quietly different corpus.
#
# dataset/conll2003.py therefore treats any truncation as a defect and prints
# a loud warning naming the number of sentences and gold tags dropped, rather
# than doing it routinely. If that warning ever appears, raise this value; do
# not reason about the result until it stops.
MAX_LEN = 128

# -----------------------------------------------------------------------------
# Casing: the feature the classification pipeline threw away
# -----------------------------------------------------------------------------
# GloVe 6B is lowercase-only, so the word-id path MUST lowercase. That leaves
# the model unable to distinguish "Bush" from "bush", "US" from "us", or a
# sentence-initial capital from a name. Casing is fed back in as its own
# categorical feature with its own small embedding, concatenated to the word
# vector -- the standard pre-BERT recipe (Collobert et al. 2011 used the same
# idea; Lample et al. 2016 replaced it with a character BiLSTM, which is
# strictly better and strictly more code).
#
# The six classes, and why each earns a slot on THIS corpus:
#   "lower"  ordinary word.                        exports
#   "title"  first letter capitalized.             Germany     <- the main one
#   "upper"  all caps.                             EU, BRUSSELS, U.S, J
#            Newswire datelines are all-caps, so this class carries a strong
#            and slightly treacherous LOC prior. Lone capitals (the "J" of
#            "J. Smith") land here too, which is the behaviour we want --
#            initials pattern with datelines, not with "Germany".
#   "mixed"  internal capitals.                    McDonald, eBay
#   "digit"  contains a digit.                     1996-08-22, 22
#   "other"  punctuation and everything else.      . , (
#
# Index 0 is RESERVED FOR PADDING, exactly like PAD_IDX in the word table, so
# both id streams pad with 0 and both embeddings can use padding_idx=0. That
# reservation is not cosmetic: without it "lower" would sit at id 0, padded
# positions would read as real lowercase words, and pinning the padding row to
# zero would silently zero the most common case class instead.
CASE_CLASSES = [PAD_TOKEN, "lower", "title", "upper", "mixed", "digit", "other"]
CASE2ID = {c: i for i, c in enumerate(CASE_CLASSES)}
NUM_CASE_CLASSES = len(CASE_CLASSES)  # 7 = 6 real classes + <pad>

# Width of the case embedding. 16 is generous for 6 classes (112 parameters
# against a multi-million-row word table) and keeps the concatenated input at
# a round 116.
CASE_DIM = 16

# Set False (or pass --no-case) to run the ablation this feature exists for:
# the same model reading lowercased text with no casing signal at all, which
# is exactly what a classification-project pipeline would have produced.
USE_CASE_FEATURE = True

# Digit normalization ("1996" -> "0000") is the other classic NER
# preprocessing step, and it is NOT done here. Measured against this machine's
# glove.6B.100d.txt: "1996", "22" and even "0000" all have vectors, so
# collapsing them neither rescues an OOV token nor loses one -- it only throws
# away the distinction between a year and a score. The tokens genuinely
# missing from GloVe are full dates like "1996-08-22" (CoNLL keeps those as a
# SINGLE token), and those stay missing under either choice. The step is
# standard because Lample et al. trained their own vectors; against a fixed
# pretrained table it buys nothing here.

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Which encoder train.py builds: "lstm" or "transformer".
MODEL = "lstm"

# Hidden width PER DIRECTION for the BiLSTM (train.py --hidden-size overrides).
# 256 x 2 layers is carried over unchanged from all three classification
# projects, so a difference in behaviour is attributable to the TASK.
#
# Be aware that this is larger than the NER literature's default: Lample et
# al. (2016) used 100 per direction and a single layer on this exact corpus.
# With only 14,041 training sentences that is a real gap, and
# `--hidden-size 100 --layers 1` is the first capacity ablation worth running.
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True

# Bidirectionality is not the free choice it was on IMDB. A tagger decides
# token i, and the evidence for token i routinely sits AFTER it: in
# "... said Mark Jones , a spokesman" the appositive is what makes "Mark
# Jones" a PER, and in "Germany beat Argentina" the verb is what makes both
# ORG rather than LOC. A forward-only tagger cannot see either.

# Dropout on the embeddings, between stacked LSTM layers, and before the
# per-token classifier. 0.5 matches both this repo's siblings and Lample et
# al., who reported dropout worth ~1.5 F1 on this corpus.
DROPOUT = 0.5

# Transformer defaults (train.py --model transformer). Same shape as the
# classification projects: a trainable projection maps EMBED_DIM (+ CASE_DIM)
# to dim, then a stack of nn.TransformerEncoderLayer blocks.
#
# Attention cost is a non-issue here for the first time in this repo: the mean
# sentence is 14 tokens against IMDB's 400, so the L^2 term is ~800x smaller
# and batch size is limited by nothing.
TRANSFORMER_DIM = 128
TRANSFORMER_GROUP = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

# Where the LayerNorm sits: "post" = LN(x + Sublayer(x)), the 2017 original;
# "pre" = x + Sublayer(LN(x)), what everything since ~2020 uses.
#
# "pre" is the default HERE, unlike the IMDB project, which kept "post" only
# so its existing checkpoints stayed reproducible. That project measured the
# reason to switch: at dim=256 / 4 layers with no warmup, the Post-LN stack
# sat at exactly chance for five epochs before escaping, while the Pre-LN
# stack was training normally after one. There are no legacy checkpoints in
# this folder, so there is no reason to inherit the trap. --norm post
# reproduces the old arrangement.
TRANSFORMER_NORM = "pre"

# GloVe pretrained embeddings. Set False (or pass --no-glove) to train the
# word vectors from scratch -- which on a 204k-token corpus is a far harsher
# ablation than it was on IMDB's 6M tokens.
USE_GLOVE = True

# -----------------------------------------------------------------------------
# Training -- same two-stage layered-LR protocol as the classification projects
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"

# 32 rather than IMDB's 64: sentences are 14 tokens instead of 400, so an
# epoch is only 14,041 short sequences of work and the binding constraint is
# the NUMBER OF UPDATES, not memory. At 32 that is 439 steps per epoch.
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128

# 0 = load in the main process. The whole corpus is parsed into python lists
# in __init__ (it is 4.8 MB), so a worker would only pay Windows' spawn cost.
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

# Label smoothing is OFF here, against IMDB's 0.05. Smoothing moves target
# mass onto the wrong classes uniformly, and with 9 tags at an 83%/17%
# O-to-entity prior most of that mass lands on "O" -- i.e. it regularizes in
# the direction the class imbalance is already pushing. The metric that
# matters (entity F1) is recall-sensitive, so that is the one direction not to
# nudge.
LABEL_SMOOTHING = 0.0

# Stage 1: FREEZE the embedding table; the from-scratch encoder + head learn
#          to read fixed GloVe vectors.
# Stage 2: unfreeze everything, three LR tiers (embedding slowest, head
#          fastest) so the pretrained vectors drift instead of being wrecked.
#
# 10 + 10 = 20 epochs, more than double IMDB's 9, and that is a direct
# consequence of what the IMDB project measured: under a fixed epoch budget
# the GRU/LSTM ranking flipped SIGN between 9 and 18 epochs, because the
# budget was measuring convergence speed rather than final quality. An epoch
# here is 439 steps over 14-token sequences -- seconds, not minutes -- so
# there is no reason to repeat that mistake.
STAGE1_EPOCHS = 10
STAGE1_LR_HEAD = 1e-3
STAGE1_LR_ENCODER = 1e-3        # also from scratch -> same tier as the head

STAGE2_EPOCHS = 10
STAGE2_LR_HEAD = 3e-4
STAGE2_LR_ENCODER = 3e-4
STAGE2_LR_EMBEDDING = 5e-5      # pretrained words: gentle updates only

# -----------------------------------------------------------------------------
# BERT finetuning (train_bert.py, model_bert/) -- Devlin et al. (2019)'s recipe
# -----------------------------------------------------------------------------
OUTPUT_DIR_BERT = os.path.join(PROJECT_ROOT, "outputs_bert")

# Hub id or local directory; the tokenizer is always loaded from the same name.
#
# CASED, and on this task that is the whole choice: an uncased checkpoint
# lowercases before WordPiece, deleting the signal in point 4 of this file's
# docstring -- with no case embedding to put it back. That makes
# `--bert google-bert/bert-base-uncased` BERT's version of the --no-case
# ablation.
BERT_NAME = "google-bert/bert-base-cased"

# Length limit in WordPiece tokens, [CLS] and [SEP] included. This is the
# checkpoint's hard limit -- its position table has 512 rows -- not a
# trade-off knob like MAX_LEN. model_bert/wordpiece.py RAISES past it rather
# than truncating: truncation would drop gold labels (see MAX_LEN), and the
# real fix for longer input is a sliding window, not a larger number.
BERT_MAX_LEN = 512

# Dropout before the tagging head, equal to BERT's own hidden_dropout_prob.
# Dropout INSIDE the 12 layers is left at the checkpoint's setting (also 0.1).
BERT_DROPOUT = 0.1

# Optimization. Devlin et al. finetune with batch 16/32, lr 5e-5/3e-5/2e-5
# and 2/3/4 epochs; this picks the middle rate and the longest budget, since
# best.pt is selected on dev anyway. Batch size is BATCH_SIZE above, shared
# with the other two models, so an epoch is the same 439 updates.
#
# One learning rate for every parameter and no frozen stage -- the opposite
# of the two-stage protocol above. That protocol exists because the GloVe
# models stack a large RANDOM encoder on pretrained vectors; here only the
# 6,921-parameter head is random. What protects the pretrained weights instead
# is WARMUP: the rate ramps up from 0 over the first 10% of updates, so the
# earliest steps -- taken while the head is still random and Adam's moment
# estimates are still noise -- are small ones.
BERT_EPOCHS = 4
BERT_LR = 3e-5
BERT_WARMUP_RATIO = 0.1

# AdamW's decoupled decay, applied to weight matrices only: biases and
# LayerNorm parameters are exempt, as in the original BERT optimizer.
BERT_WEIGHT_DECAY = 0.01

# The original BERT value, rather than the GloVe models' GRAD_CLIP = 5.0.
BERT_GRAD_CLIP = 1.0
