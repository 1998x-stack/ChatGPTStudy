# -*- coding: utf-8 -*-
"""BM25 ranking model implementation.

This module provides a classic BM25 scorer:
Score(q, d) = sum_i IDF(q_i) * ((tf_i*(k1+1)) / (tf_i + k1*(1 - b + b*|d|/avgdl)))

Author: Your Name
"""

from __future__ import annotations

import math
from typing import Dict, Iterable


class BM25:
    """BM25 scorer with tunable k1 and b.

    Attributes:
        k1: TF saturation parameter (commonly 1.2~2.0).
        b: Length normalization parameter (commonly ~0.75).
        n_docs: Total number of documents.
        avgdl: Average document length (post-processed token count).
    """

    def __init__(self, n_docs: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25.

        Args:
            n_docs: Total document count.
            avgdl: Average document length.
            k1: TF saturation.
            b: Length normalization weight.
        """
        if n_docs <= 0:
            raise ValueError("n_docs must be positive.")
        if avgdl <= 0:
            raise ValueError("avgdl must be positive.")
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = int(n_docs)
        self.avgdl = float(avgdl)

    def idf(self, df: int) -> float:
        """Compute BM25 IDF with standard +0.5 smoothing.

        Args:
            df: Document frequency for a term.

        Returns:
            IDF value.
        """
        # 中文注释：BM25 IDF 平滑形式，避免极端值
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score_tf(self, tf: int, dl: int) -> float:
        """Compute the TF-normalized factor for a single term in a doc.

        Args:
            tf: Term frequency in doc.
            dl: Document length.

        Returns:
            TF component of BM25.
        """
        # 中文注释：长度归一化项
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
        return (tf * (self.k1 + 1.0)) / denom if denom > 0.0 else 0.0

    def score_doc(self, dl: int, tf_by_term: Dict[str, int], df_by_term: Dict[str, int]) -> float:
        """Score a document for a set of query terms.

        Args:
            dl: Document length.
            tf_by_term: Term frequencies for query terms in this doc.
            df_by_term: Document frequency per term.

        Returns:
            BM25 score.
        """
        s = 0.0
        for t, tf in tf_by_term.items():
            if tf <= 0:
                continue
            df = df_by_term.get(t, 0)
            if df <= 0 or df > self.n_docs:
                continue
            s += self.idf(df) * self.score_tf(tf, dl)
        return s

