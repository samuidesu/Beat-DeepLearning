"""Minimal text cleanup for both languages.

The rule for a translation corpus is that preprocessing should be as close to
NOTHING as possible. Everything this module could do to "clean" the text is
something a translator would have had to reproduce on the other side:

  lowercasing            deletes the Apple/apple, US/us, May/may distinction
                         on the source, and makes the target unscoreable
                         against cased references
  stripping punctuation  deletes sentence boundaries and quotation
  stripping diacritics   turns Romanian into a different, wrong language
                         (ana -> ana, si -> si: `tara` and `tara` are
                         different words)
  pre-tokenizing English WordPiece already does this, and doing it first only
                         creates spacing WordPiece has to re-learn
  splitting Romanian     SentencePiece works on raw text by design

So the whole of preprocessing is: normalize whitespace, drop empty strings.
Unicode, punctuation and capitalization all survive untouched.
"""

from __future__ import annotations

from typing import Any, Dict

# Whitespace normalization is the one edit that is made. WMT16 lines can carry
# tabs, non-breaking spaces and stray double spaces from their SGML origin;
# collapsing runs of whitespace to a single space keeps SentencePiece from
# learning pieces that encode formatting noise. It never changes a word.
_WHITESPACE = str.maketrans({"\t": " ", "\n": " ", "\r": " ", " ": " "})


def clean_text(text: str) -> str:
    """Strip and collapse whitespace. Case, punctuation and Unicode survive.

    >>> clean_text("  Bucuresti  este   capitala .  ")
    'Bucuresti este capitala .'
    """
    if text is None:
        return ""
    return " ".join(text.translate(_WHITESPACE).split())


def clean_pair(example: Dict[str, Any], source_language: str,
               target_language: str) -> Dict[str, str]:
    """Map a raw WMT16 record to {"source", "target"} cleaned strings.

    A WMT16 record is {"translation": {"en": ..., "ro": ...}}; which side is
    source and which is target comes from the config, not from the column
    order.
    """
    pair = example["translation"]
    return {
        "source": clean_text(pair[source_language]),
        "target": clean_text(pair[target_language]),
    }


def is_non_empty_pair(example: Dict[str, str]) -> bool:
    """True when both sides still have content after cleaning."""
    return bool(example["source"]) and bool(example["target"])
