# -*- coding: utf-8 -*-
"""BM25 and BM25F ranking models."""

from __future__ import annotations

import math
from typing import Dict


class BM25:
    """BM25 scorer with tunable k1 and b."""

    def __init__(self, n_docs: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> None:
        if n_docs <= 0:
            raise ValueError("n_docs must be positive.")
        if avgdl <= 0:
            raise ValueError("avgdl must be positive.")
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = int(n_docs)
        self.avgdl = float(avgdl)

    def idf(self, df: int) -> float:
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score_tf(self, tf: int, dl: int) -> float:
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
        return (tf * (self.k1 + 1.0)) / denom if denom > 0.0 else 0.0

    def score_doc(self, dl: int, tf_by_term: Dict[str, int], df_by_term: Dict[str, int]) -> float:
        s = 0.0
        for t, tf in tf_by_term.items():
            if tf <= 0:
                continue
            df = df_by_term.get(t, 0)
            if df <= 0 or df > self.n_docs:
                continue
            s += self.idf(df) * self.score_tf(tf, dl)
        return s


class BM25F:
    """BM25F scorer (multi-field BM25).

    Simplified BM25F formulation:
      For each term q:
        w_tf(q, d) = sum_f [ w_f * tf_{q,f}(d) / ( (1-b_f) + b_f * len_f(d)/avglen_f ) ]
      Score(q, d) = sum_q IDF(q) * ((w_tf * (k1 + 1)) / (w_tf + k1))

    Attributes:
        n_docs: total documents
        k1: saturation parameter
        field_weights: per-field weight (e.g., {"title":2.0, "body":1.0})
        field_b: per-field b (0~1), length normalization
        avglen: per-field average length
    """

    def __init__(
        self,
        n_docs: int,
        field_weights: Dict[str, float],
        field_b: Dict[str, float],
        avglen: Dict[str, float],
        k1: float = 1.2,
    ) -> None:
        if n_docs <= 0:
            raise ValueError("n_docs must be positive.")
        self.n_docs = n_docs
        self.k1 = float(k1)
        self.field_weights = field_weights
        self.field_b = field_b
        self.avglen = avglen
        for f in field_weights:
            if f not in field_b or f not in avglen:
                raise ValueError(f"Field {f} missing b or avglen in BM25F.")

    @staticmethod
    def _idf(n_docs: int, df: int) -> float:
        # 中文注释：和 BM25 类似，采用平滑 IDF
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _w_tf(self, tf_by_field: Dict[str, int], len_by_field: Dict[str, int]) -> float:
        s = 0.0
        for f, tf in tf_by_field.items():
            if tf <= 0:
                continue
            w_f = self.field_weights.get(f, 1.0)
            b_f = self.field_b.get(f, 0.75)
            avg_f = self.avglen.get(f, 1.0)
            dl_f = max(1, len_by_field.get(f, 1))
            denom = (1.0 - b_f) + b_f * (dl_f / max(1e-9, avg_f))
            s += w_f * (tf / denom)
        return s

    def score_doc(
        self,
        tf_by_term_field: Dict[str, Dict[str, int]],
        df_by_term: Dict[str, int],
        len_by_field: Dict[str, int],
    ) -> float:
        """Score a document with term-field frequencies.

        Args:
            tf_by_term_field: term -> {field -> tf}
            df_by_term: term -> document frequency (any field occurrence)
            len_by_field: {field -> length}

        Returns:
            BM25F score.
        """
        score = 0.0
        for t, tf_fields in tf_by_term_field.items():
            if not tf_fields:
                continue
            wtf = self._w_tf(tf_fields, len_by_field)
            if wtf <= 0:
                continue
            df = df_by_term.get(t, 0)
            if df <= 0 or df > self.n_docs:
                continue
            idf = self._idf(self.n_docs, df)
            score += idf * ((wtf * (self.k1 + 1.0)) / (wtf + self.k1))
        return score

