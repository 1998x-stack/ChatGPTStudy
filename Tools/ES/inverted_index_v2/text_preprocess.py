# -*- coding: utf-8 -*-
"""Text preprocessing utilities: normalization, tokenization, stop words, stemming.

This module provides a dependency-free Analyzer for English text:
- Unicode normalization (NFKD) and lowercase
- Regex tokenization
- Built-in English stop words
- Porter stemming (see stemmer.py)

Author: Your Name
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from .stemmer import PorterStemmer

# 常见英文停用词表（精选 + 常用，避免外部依赖，覆盖大部分搜索场景）
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but",
    "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with", "from", "have", "has",
    "had", "were", "been", "being", "over", "under", "above", "below",
    "up", "down", "out", "off", "again", "further", "once", "doing", "do",
    "does", "did", "own", "same", "too", "very", "can", "could", "should",
    "would", "may", "might", "must", "shall", "than", "between", "because",
    "until", "while", "where", "when", "why", "how", "what", "who", "whom",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")  # 简化处理英文缩写：don't -> don, t


class Analyzer:
    """Analyzer for English text: normalize, tokenize, stop-words, and stemming."""

    def __init__(self, min_token_len: int = 2) -> None:
        """Initialize the Analyzer.

        Args:
            min_token_len: Minimum token length to keep after normalization.
        """
        # 中文注释：初始化词干提取器，并设定最小词长过滤
        self.stemmer = PorterStemmer()
        self.min_token_len = max(1, int(min_token_len))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize unicode text (lowercase, strip accents)."""
        # 中文注释：NFKD标准化并移除重音，统一小写
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into normalized tokens (raw tokens)."""
        # 中文注释：正则提取单词序列，并按最小长度过滤
        text = self._normalize_text(text)
        tokens = _WORD_RE.findall(text)
        return [t for t in tokens if len(t) >= self.min_token_len]

    def analyze(self, text: str) -> List[str]:
        """Full pipeline: tokenize -> remove stop words -> stem."""
        # 中文注释：分词 -> 停用词过滤 -> 词干化
        tokens = self.tokenize(text)
        tokens = [t for t in tokens if t not in STOP_WORDS]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return [t for t in tokens if t]  # 安全过滤空字符串

    def analyze_with_positions(self, text: str) -> List[tuple[str, int]]:
        """Analyze and keep token positions within the doc (single field)."""
        # 中文注释：保留处理后词项及其在文档中的位置索引（基于处理后序列）
        raw_tokens = self.tokenize(text)
        processed: List[tuple[str, int]] = []
        pos = 0
        for tok in raw_tokens:
            if tok in STOP_WORDS:
                continue
            s = self.stemmer.stem(tok)
            if not s:
                continue
            processed.append((s, pos))
            pos += 1
        return processed

