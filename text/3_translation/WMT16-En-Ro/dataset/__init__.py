"""Data pipeline for WMT16 En->Ro.

    preprocessing.py     minimal text cleanup, nothing destructive
    source_tokenizer.py  the cased BERT WordPiece tokenizer (English)
    target_tokenizer.py  the Romanian SentencePiece BPE tokenizer we train
    dataset.py           load / clean / encode / filter the three splits
    collate.py           dynamic padding into batch tensors

Named `dataset/` rather than `data/` for two reasons: every sibling project in
this repo uses that name, and the repository .gitignore excludes `data/` at any
depth (it is where downloaded corpora go), which would leave this package
untracked.
"""
