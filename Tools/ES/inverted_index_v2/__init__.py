"""
Inverted Index package initialization.

This package now provides two indexing pathways:
1) In-memory index (demo/learning, supports skip list, VB+gap, BM25).
2) File-system (FS) index (industrial): parallel segment build, segment merge,
   field-weighted BM25F, on-disk lexicon + postings, non-pickle format.

All modules follow Google-style docstrings and typing, with Chinese PEP8-style comments.
"""
__all__ = [
    "text_preprocess",
    "stemmer",
    "vb_codec",
    "postings",
    "bm25",
    "query",
    "index",
    "storage_fs",
]

