"""The Romanian (target) tokenizer: a SentencePiece BPE model we train.

WHY NOT REUSE THE BERT TOKENIZER. bert-base-cased has an English WordPiece
vocabulary. Run it on Romanian and most words shatter into unrelated English
fragments, and the LM head then has to predict over 29k mostly-English pieces.
The target side gets its own vocabulary, trained on the language it generates.

WHAT A TARGET TOKEN IS. Subword, not word. A frequent word may survive whole
("este"); a rarer one is split into several pieces the BPE merges produced
("dezvoltarii" -> something like "_dezvolt" + "arii"). Nothing in this project
assumes one token == one Romanian word, and the vocabulary size (16k by
default) is far below the number of distinct Romanian word forms precisely
because it does not have to hold them.

TRAIN SPLIT ONLY. The merges learned here are model parameters like any other.
Learning them from validation or test text leaks the evaluation data into the
model, so scripts/train_target_tokenizer.py reads the train split and nothing
else.

CASE AND DIACRITICS SURVIVE. normalization_rule_name is nmt_nfkc, the
case-PRESERVING NFKC variant. Its case-folding sibling nmt_nfkc_cf would merge
Romania with romania, and the config validator rejects it.

The four special tokens are given fixed, explicit ids:

    0  <pad>   padding; never contributes to the loss
    1  <unk>   out-of-vocabulary fallback
    2  <bos>   Beginning Of Sequence: the decoder's first input
    3  <eos>   End Of Sequence: what the decoder must learn to emit to stop
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Sequence

import sentencepiece as spm

# Fixed special-token layout. Pinned rather than discovered so that a
# checkpoint and a tokenizer can never disagree about which id is padding.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
PAD_PIECE, UNK_PIECE, BOS_PIECE, EOS_PIECE = "<pad>", "<unk>", "<bos>", "<eos>"

# Written next to the model so a trained tokenizer records how it was made.
METADATA_NAME = "tokenizer_meta.json"


def tokenizer_paths(model_dir: str, model_prefix: str = "spm_ro") -> dict:
    """The three files that make up a trained tokenizer directory."""
    return {
        "model": os.path.join(model_dir, f"{model_prefix}.model"),
        "vocab": os.path.join(model_dir, f"{model_prefix}.vocab"),
        "meta": os.path.join(model_dir, METADATA_NAME),
    }


def tokenizer_exists(model_dir: str, model_prefix: str = "spm_ro") -> bool:
    """True when `model_dir` already holds a usable SentencePiece model."""
    return os.path.isfile(tokenizer_paths(model_dir, model_prefix)["model"])


def train_target_tokenizer(
    sentences: Iterable[str],
    model_dir: str,
    vocab_size: int,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    normalization_rule_name: str = "nmt_nfkc",
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = True,
    model_prefix: str = "spm_ro",
    seed: int = 42,
    metadata: dict | None = None,
) -> str:
    """Train a SentencePiece model on `sentences` and save it to `model_dir`.

    `sentences` is consumed as an iterator, so the caller can stream the train
    split straight out of Hugging Face without writing an 80 MB text file.

    Returns the path of the written .model file.
    """
    if normalization_rule_name.endswith("_cf"):
        raise ValueError(
            f"normalization_rule_name={normalization_rule_name!r} folds case. "
            "Romanian output is case-sensitive; use nmt_nfkc."
        )
    os.makedirs(model_dir, exist_ok=True)
    paths = tokenizer_paths(model_dir, model_prefix)
    prefix = os.path.join(model_dir, model_prefix)

    # Sentence sampling (input_sentence_size) shuffles; seed it so a retrain
    # with the same corpus produces the same merges.
    spm.set_random_generator_seed(seed)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        normalization_rule_name=normalization_rule_name,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        # Explicit, pinned special tokens.
        pad_id=PAD_ID, pad_piece=PAD_PIECE,
        unk_id=UNK_ID, unk_piece=UNK_PIECE,
        bos_id=BOS_ID, bos_piece=BOS_PIECE,
        eos_id=EOS_ID, eos_piece=EOS_PIECE,
    )

    meta = {
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "normalization_rule_name": normalization_rule_name,
        "input_sentence_size": input_sentence_size,
        "shuffle_input_sentence": shuffle_input_sentence,
        "seed": seed,
        "special_tokens": {
            PAD_PIECE: PAD_ID, UNK_PIECE: UNK_ID,
            BOS_PIECE: BOS_ID, EOS_PIECE: EOS_ID,
        },
    }
    meta.update(metadata or {})
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return paths["model"]


class TargetTokenizer:
    """Thin wrapper over a trained SentencePiece model.

    It exists for two things the raw processor does not give cleanly: named
    special-token ids, and a decode() that knows what a model prediction looks
    like (padding, a leading <bos>, junk after <eos>, -100 label padding).
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"No SentencePiece model at {model_path}.\n"
                "Train it first:  python scripts/train_target_tokenizer.py"
            )
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self._check_special_ids()

    @classmethod
    def from_dir(cls, model_dir: str, model_prefix: str = "spm_ro") -> "TargetTokenizer":
        return cls(tokenizer_paths(model_dir, model_prefix)["model"])

    def _check_special_ids(self) -> None:
        """Fail loudly if the loaded model does not use this project's ids."""
        actual = (self.sp.pad_id(), self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id())
        expected = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
        if actual != expected:
            raise ValueError(
                f"{self.model_path} has special ids {actual}, expected {expected} "
                "(<pad>, <unk>, <bos>, <eos>). Retrain with "
                "scripts/train_target_tokenizer.py."
            )

    # --- ids --------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def vocab_size(self) -> int:
        """Size of the target vocabulary -- the width of the future LM head."""
        return self.sp.get_piece_size()

    def __len__(self) -> int:
        return self.vocab_size

    # --- encoding ---------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Romanian text -> subword ids, WITHOUT <bos>/<eos>."""
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        """Batched encode; same output as calling encode() per string."""
        return self.sp.encode(list(texts), out_type=int)

    def encode_with_specials(self, text: str) -> List[int]:
        """Romanian text -> [<bos>] + ids + [<eos>], the full target sequence.

        This is what dataset.py stores; collate.py slices it into
        decoder_input_ids and labels.
        """
        return [BOS_ID] + self.encode(text) + [EOS_ID]

    def pieces(self, text: str) -> List[str]:
        """The subword pieces themselves -- for inspection, not for training."""
        return self.sp.encode(text, out_type=str)

    # --- decoding ---------------------------------------------------------
    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        """Subword ids -> normal Romanian text.

        Handles everything a model output or a label row can contain:
          - stops at the first <eos> (nothing after it is a prediction),
          - drops <pad> and <bos>,
          - drops -100, the label-padding sentinel,
          - joins the pieces back into text, so no SentencePiece marker is
            ever visible in a final translation.
        """
        clean: List[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if stop_at_eos and token_id == EOS_ID:
                break
            if token_id in (PAD_ID, BOS_ID) or token_id < 0:  # < 0 covers -100
                continue
            clean.append(token_id)
        return self.sp.decode(clean)

    def decode_batch(self, batch_ids, stop_at_eos: bool = True) -> List[str]:
        """decode() over a [B, T] tensor / nested list."""
        return [self.decode(row, stop_at_eos=stop_at_eos) for row in batch_ids]
