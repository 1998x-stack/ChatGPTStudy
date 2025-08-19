# -*- coding: utf-8 -*-
"""Inverted index build, query, and BM25 ranking.

This module defines the InvertedIndex class, supporting:
- Building from a folder of English .txt files.
- Boolean queries with AND/OR/NOT and phrase matching.
- BM25 ranking.
- Compression via VB+gap and Skip List acceleration.
- Save/load using pickle (with compressed postings stored as bytes).

Author: Your Name
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .bm25 import BM25
from .postings import Posting, PostingList
from .query import TermToken, PhraseToken, OpToken, parse_query
from .text_preprocess import Analyzer


@dataclass
class DocMeta:
    """Metadata of a document.

    Attributes:
        doc_id: Integer id.
        path: Source file path.
        length: Number of processed tokens kept in index.
    """
    doc_id: int
    path: str
    length: int


class InvertedIndex:
    """In-memory inverted index with compression and BM25 ranking."""

    def __init__(self) -> None:
        # 中文注释：核心结构 term -> PostingList
        self.term_to_pl: Dict[str, PostingList] = {}
        self.docs: Dict[int, DocMeta] = {}
        self.path_to_docid: Dict[str, int] = {}
        self.analyzer = Analyzer()
        self._finalized: bool = False
        self._bm25: Optional[BM25] = None
        self._compressed_terms: Set[str] = set()  # 已压缩的 term（节省内存示例）

    # -------------------------
    # Building
    # -------------------------
    def build_from_folder(self, folder: str) -> None:
        """Build index from a folder of .txt files."""
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        doc_id = 0
        total_len = 0
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt")]
        files.sort()
        for p in files:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            analyzed = self.analyzer.analyze_with_positions(text)
            if not analyzed:
                continue
            # 中文注释：按 term -> positions 收集，构建 posting
            per_term_positions: Dict[str, List[int]] = defaultdict(list)
            for term, pos in analyzed:
                per_term_positions[term].append(pos)
            for term, positions in per_term_positions.items():
                pl = self.term_to_pl.get(term)
                if pl is None:
                    pl = PostingList()
                    self.term_to_pl[term] = pl
                for position in positions:
                    pl.add_occurrence(doc_id=doc_id, position=position)
            dl = len(analyzed)
            self.docs[doc_id] = DocMeta(doc_id=doc_id, path=p, length=dl)
            self.path_to_docid[p] = doc_id
            total_len += dl
            doc_id += 1

        # 完成构建后 finalize
        for term, pl in self.term_to_pl.items():
            pl.finalize()

        n_docs = len(self.docs)
        avgdl = (total_len / n_docs) if n_docs > 0 else 1.0
        self._bm25 = BM25(n_docs=n_docs, avgdl=avgdl, k1=1.5, b=0.75)
        self._finalized = True

        # 打印关键信息
        print(f"[BUILD] Indexed {n_docs} documents from folder: {folder}")
        print(f"[BUILD] Vocabulary size: {len(self.term_to_pl)} terms")
        print(f"[BUILD] Average document length: {avgdl:.2f}")

    # -------------------------
    # Compression
    # -------------------------
    def compress_all(self) -> None:
        """Compress all posting lists to VB+gap encoded bytes (keeps in-memory postings)."""
        if not self._finalized:
            raise RuntimeError("Index must be built/finalized before compression.")
        count = 0
        for term, pl in self.term_to_pl.items():
            pl.to_compressed()
            self._compressed_terms.add(term)
            count += 1
        print(f"[COMPRESS] Compressed {count} posting lists (VB+gap).")

    # -------------------------
    # Query execution helpers
    # -------------------------
    def _get_pl(self, term: str) -> PostingList:
        pl = self.term_to_pl.get(term)
        if pl is None:
            return PostingList()  # empty
        return pl

    def _phrase_docs(self, phrase_terms: List[str]) -> Set[int]:
        """Return doc_ids that match the exact phrase via positions adjacency."""
        if not phrase_terms:
            return set()
        # 先做交集（所有词必须同现）
        pls = [self._get_pl(t) for t in phrase_terms]
        if any(len(pl) == 0 for pl in pls):
            return set()

        # 链式交集
        cur = pls[0]
        for i in range(1, len(pls)):
            cur = cur.intersect(pls[i])

        # 对交集文档，做逐文档 positions 相邻判断
        res: Set[int] = set()
        # 建立 term -> doc_id -> positions 的快速映射
        term_doc_pos: List[Dict[int, List[int]]] = []
        for t in phrase_terms:
            m: Dict[int, List[int]] = {}
            for p in self._get_pl(t).postings:
                m[p.doc_id] = p.positions
            term_doc_pos.append(m)

        for p in cur.postings:
            doc = p.doc_id
            # 逐个位置链式校验：pos0 in t0, pos0+1 in t1, ...
            p0 = term_doc_pos[0].get(doc, [])
            positions_candidates = set(p0)
            for idx in range(1, len(phrase_terms)):
                next_positions = term_doc_pos[idx].get(doc, [])
                # 需要 pos(k) = pos(k-1) + 1
                shifted = {x + 1 for x in positions_candidates}
                positions_candidates = shifted.intersection(next_positions)
                if not positions_candidates:
                    break
            if positions_candidates:
                res.add(doc)
        return res

    def _eval_rpn(self, rpn_tokens) -> Set[int]:
        """Evaluate RPN tokens to get a candidate set of doc_ids (ignoring ranking)."""
        stack: List[Set[int]] = []
        universe: Set[int] = set(self.docs.keys())

        for tok in rpn_tokens:
            if isinstance(tok, TermToken):
                pl = self._get_pl(tok.term)
                stack.append({p.doc_id for p in pl.postings})
            elif isinstance(tok, PhraseToken):
                stack.append(self._phrase_docs(tok.terms))
            elif isinstance(tok, OpToken):
                if tok.op == "NOT":
                    if not stack:
                        stack.append(universe)  # NOT applied to empty -> NOT true -> universe?
                    a = stack.pop()
                    stack.append(universe - a)
                else:
                    if len(stack) < 2:
                        raise ValueError("Insufficient operands for boolean operator.")
                    b = stack.pop()
                    a = stack.pop()
                    if tok.op == "AND":
                        stack.append(a & b)
                    elif tok.op == "OR":
                        stack.append(a | b)
                    else:
                        raise ValueError(f"Unknown operator: {tok.op}")
            else:
                raise ValueError("Unknown token type.")

        if len(stack) != 1:
            raise ValueError("Invalid RPN evaluation (stack not singular).")
        return stack[0]

    def _collect_tf_for_docs(self, doc_ids: Set[int], query_terms: List[str]) -> Dict[int, Counter]:
        """Collect per-doc term frequencies for query terms."""
        # term -> posting map for quick lookup
        term_to_map: Dict[str, Dict[int, int]] = {}
        for t in set(query_terms):
            m: Dict[int, int] = {}
            for p in self._get_pl(t).postings:
                m[p.doc_id] = p.tf
            term_to_map[t] = m

        tf_by_doc: Dict[int, Counter] = {}
        for d in doc_ids:
            c = Counter()
            for t in query_terms:
                tf = term_to_map.get(t, {}).get(d, 0)
                if tf:
                    c[t] = tf
            tf_by_doc[d] = c
        return tf_by_doc

    def _df_by_term(self, terms: Iterable[str]) -> Dict[str, int]:
        """Get document frequency per term."""
        res: Dict[str, int] = {}
        for t in set(terms):
            res[t] = len(self._get_pl(t).postings)
        return res

    def search(self, query: str, topk: int = 10) -> List[Tuple[int, float]]:
        """Search with boolean + phrase, then rank by BM25.

        Args:
            query: User query string.
            topk: Number of top results.

        Returns:
            List of (doc_id, score) sorted by score descending.
        """
        if not self._finalized or self._bm25 is None:
            raise RuntimeError("Index must be built before searching.")
        rpn = parse_query(query, self.analyzer.analyze)
        candidate_docs = self._eval_rpn(rpn)

        # 收集用于 BM25 的查询“词项”（短语被拆开作为独立词参与打分）
        # 注意：NOT 过滤已经在 candidate_docs 中体现
        positive_terms: List[str] = []
        for tok in rpn:
            if isinstance(tok, TermToken):
                positive_terms.append(tok.term)
            elif isinstance(tok, PhraseToken):
                positive_terms.extend(tok.terms)

        df_by_term = self._df_by_term(positive_terms)
        tf_by_doc = self._collect_tf_for_docs(candidate_docs, positive_terms)

        # 计算 BM25 分数
        scored: List[Tuple[int, float]] = []
        for d in candidate_docs:
            dl = self.docs[d].length
            score = self._bm25.score_doc(dl=dl, tf_by_term=tf_by_doc[d], df_by_term=df_by_term)
            scored.append((d, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(topk))]

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path: str) -> None:
        """Serialize index to disk (pickle, with compressed postings)."""
        self.compress_all()
        blob = {
            "docs": self.docs,
            "term_bytes": {t: pl.to_compressed() for t, pl in self.term_to_pl.items()},
            "avgdl": self._bm25.avgdl if self._bm25 else 1.0,
            "n_docs": len(self.docs),
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVE] Index saved to {path} (compressed postings).")

    @staticmethod
    def load(path: str) -> "InvertedIndex":
        """Load index from pickle (rebuild PostingList in memory)."""
        with open(path, "rb") as f:
            blob = pickle.load(f)
        idx = InvertedIndex()
        idx.docs = blob["docs"]
        idx.term_to_pl = {}
        from .postings import PostingList  # avoid cycle
        for t, data in blob["term_bytes"].items():
            idx.term_to_pl[t] = PostingList.from_compressed(data)
        idx._bm25 = BM25(n_docs=blob["n_docs"], avgdl=blob["avgdl"])
        idx._finalized = True
        print(f"[LOAD] Index loaded from {path}. Docs={len(idx.docs)} Terms={len(idx.term_to_pl)}")
        return idx

