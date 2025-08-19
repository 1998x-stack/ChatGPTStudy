"""
Inverted Index package initialization.

This package provides:
- Text preprocessing (tokenization, stop words, stemming).
- VB (Variable-Byte) compression with gap encoding.
- Posting list structure with Skip List acceleration.
- Boolean query parser and phrase matching.
- BM25 ranking.
- InvertedIndex build/load/query APIs and a CLI entry point.

All modules use Google-style docstrings and typing, with Chinese PEP8-style inline comments.
"""
__all__ = [
    "text_preprocess",
    "stemmer",
    "vb_codec",
    "postings",
    "bm25",
    "query",
    "index",
]
