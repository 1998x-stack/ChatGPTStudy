# -*- coding: utf-8 -*-
"""Inverted index (in-memory) and FS index (on-disk) APIs."""

from __future__ import annotations

import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .bm25 import BM25, BM25F
from .postings import Posting, PostingList
from .query import TermToken, PhraseToken, OpToken, parse_query
from .text_preprocess import Analyzer
from .storage_fs import (
    build_segments_parallel,
    merge_segments,
    write_manifest,
    write_doclen,
    read_manifest,
    read_doclen,
    load_term_postings_all_segments,
)


# -----------------------
# In-memory index (previous version, kept)
# -----------------------

@dataclass
class DocMeta:
    doc_id: int
    path: str
    length: int


class InvertedIndex:
    """In-memory index with skip list, compression, BM25 (single-field)."""

    def __init__(self) -> None:
        self.term_to_pl: Dict[str, PostingList] = {}
        self.docs: Dict[int, DocMeta] = {}
        self.path_to_docid: Dict[str, int] = {}
        self.analyzer = Analyzer()
        self._finalized: bool = False
        self._bm25: Optional[BM25] = None
        self._compressed_terms: Set[str] = set()

    def build_from_folder(self, folder: str) -> None:
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
        for _, pl in self.term_to_pl.items():
            pl.finalize()
        n_docs = len(self.docs)
        avgdl = (total_len / n_docs) if n_docs > 0 else 1.0
        self._bm25 = BM25(n_docs=n_docs, avgdl=avgdl, k1=1.5, b=0.75)
        self._finalized = True
        print(f"[BUILD] Indexed {n_docs} documents (in-memory)")

    def compress_all(self) -> None:
        if not self._finalized:
            raise RuntimeError("Index must be built/finalized before compression.")
        count = 0
        for term, pl in self.term_to_pl.items():
            pl.to_compressed()
            self._compressed_terms.add(term)
            count += 1
        print(f"[COMPRESS] Compressed {count} posting lists (VB+gap).")

    def _get_pl(self, term: str) -> PostingList:
        return self.term_to_pl.get(term, PostingList())

    def _phrase_docs(self, phrase_terms: List[str]) -> Set[int]:
        if not phrase_terms:
            return set()
        pls = [self._get_pl(t) for t in phrase_terms]
        if any(len(pl) == 0 for pl in pls):
            return set()
        cur = pls[0]
        for i in range(1, len(pls)):
            cur = cur.intersect(pls[i])
        # 简单单字段短语匹配
        res: Set[int] = set()
        term_doc_pos: List[Dict[int, List[int]]] = []
        for t in phrase_terms:
            m: Dict[int, List[int]] = {}
            for p in self._get_pl(t).postings:
                m[p.doc_id] = p.positions
            term_doc_pos.append(m)
        for p in cur.postings:
            doc = p.doc_id
            p0 = term_doc_pos[0].get(doc, [])
            candidates = set(p0)
            for idx in range(1, len(phrase_terms)):
                next_positions = term_doc_pos[idx].get(doc, [])
                shifted = {x + 1 for x in candidates}
                candidates = shifted.intersection(next_positions)
                if not candidates:
                    break
            if candidates:
                res.add(doc)
        return res

    def _eval_rpn(self, rpn_tokens) -> Set[int]:
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
                    a = stack.pop() if stack else set()
                    stack.append(universe - a)
                else:
                    if len(stack) < 2:
                        raise ValueError("Insufficient operands.")
                    b = stack.pop()
                    a = stack.pop()
                    if tok.op == "AND":
                        stack.append(a & b)
                    elif tok.op == "OR":
                        stack.append(a | b)
            else:
                raise ValueError("Unknown token.")
        if len(stack) != 1:
            raise ValueError("Invalid RPN evaluation.")
        return stack[0]

    def _df_by_term(self, terms: Iterable[str]) -> Dict[str, int]:
        return {t: len(self._get_pl(t).postings) for t in set(terms)}

    def _collect_tf_for_docs(self, doc_ids: Set[int], query_terms: List[str]) -> Dict[int, Counter]:
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

    def search(self, query: str, topk: int = 10) -> List[Tuple[int, float]]:
        if not self._finalized or self._bm25 is None:
            raise RuntimeError("Index must be built before searching.")
        rpn = parse_query(query, self.analyzer.analyze)
        candidate_docs = self._eval_rpn(rpn)
        positive_terms: List[str] = []
        for tok in rpn:
            if isinstance(tok, TermToken):
                positive_terms.append(tok.term)
            elif isinstance(tok, PhraseToken):
                positive_terms.extend(tok.terms)
        df_by_term = self._df_by_term(positive_terms)
        tf_by_doc = self._collect_tf_for_docs(candidate_docs, positive_terms)
        scored: List[Tuple[int, float]] = []
        for d in candidate_docs:
            dl = self.docs[d].length
            score = self._bm25.score_doc(dl=dl, tf_by_term=tf_by_doc[d], df_by_term=df_by_term)
            scored.append((d, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(topk))]

    # 保留原 pickle save/load（学习演示用）
    def save(self, path: str) -> None:
        self.compress_all()
        blob = {
            "docs": self.docs,
            "term_bytes": {t: pl.to_compressed() for t, pl in self.term_to_pl.items()},
            "avgdl": self._bm25.avgdl if self._bm25 else 1.0,
            "n_docs": len(self.docs),
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVE] Pickle index -> {path}")

    @staticmethod
    def load(path: str) -> "InvertedIndex":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        idx = InvertedIndex()
        idx.docs = blob["docs"]
        idx.term_to_pl = {}
        from .postings import PostingList
        for t, data in blob["term_bytes"].items():
            idx.term_to_pl[t] = PostingList.from_compressed(data)
        idx._bm25 = BM25(n_docs=blob["n_docs"], avgdl=blob["avgdl"])
        idx._finalized = True
        print(f"[LOAD] Pickle index loaded. Docs={len(idx.docs)} Terms={len(idx.term_to_pl)}")
        return idx


# -----------------------
# FS index (on-disk, parallel segments, BM25F)
# -----------------------

class FSIndex:
    """File-system index reader: on-disk segments + BM25F ranking."""

    def __init__(self, index_dir: str) -> None:
        if not os.path.isdir(index_dir):
            raise FileNotFoundError(index_dir)
        self.index_dir = index_dir
        self.man = read_manifest(index_dir)
        self.doclen = read_doclen(index_dir)  # list of dict
        self.analyzer = Analyzer()

        self.fields: List[str] = self.man["fields"]
        self.field_weights: Dict[str, float] = self.man["field_weights"]
        self.field_b: Dict[str, float] = self.man["field_b"]
        self.n_docs: int = self.man["n_docs"]
        self.avglen: Dict[str, float] = self.man["avglen"]

        self._bm25f = BM25F(
            n_docs=self.n_docs,
            field_weights=self.field_weights,
            field_b=self.field_b,
            avglen=self.avglen,
            k1=1.2,
        )
        # 快速 doc_id -> len_by_field / path
        self.len_by_doc: Dict[int, Dict[str, int]] = {
            x["doc_id"]: {"title": int(x["title"]), "body": int(x["body"])} for x in self.doclen
        }
        self.path_by_doc: Dict[int, str] = {x["doc_id"]: x["path"] for x in self.doclen}

    # ---------- Phrase match (per field only) ----------
    @staticmethod
    def _phrase_match_in_field(positions_lists: List[List[int]]) -> bool:
        """Check whether there exists p in positions_lists[0] so that
        p+1 in positions_lists[1], p+2 in positions_lists[2], ..."""
        if not positions_lists or any(not pl for pl in positions_lists):
            return False
        candidates = set(positions_lists[0])
        for k in range(1, len(positions_lists)):
            shifted = {x + 1 for x in candidates}
            candidates = shifted.intersection(positions_lists[k])
            if not candidates:
                return False
        return True

    def _phrase_docs(self, phrase_terms: List[str]) -> Set[int]:
        """Phrase docs: only if a single field contains consecutive positions."""
        if not phrase_terms:
            return set()
        # 先合并所有 term 的 postings
        per_term_plist: Dict[str, List[Tuple[int, Dict[str, List[int]]]]] = {}
        for t in set(phrase_terms):
            df, plist = load_term_postings_all_segments(self.index_dir, t)
            if df == 0:
                return set()
            per_term_plist[t] = plist
        # 取 doc_id 交集
        docs_sets = [set(d for d, _ in per_term_plist[t]) for t in phrase_terms]
        candidate_docs = set.intersection(*docs_sets)
        res: Set[int] = set()
        for d in sorted(candidate_docs):
            # 对每个字段分别检查短语是否命中
            for f in self.fields:
                seqs = []
                ok = True
                for t in phrase_terms:
                    m = dict(per_term_plist[t])
                    pos_map = m.get(d, {})
                    seqs.append(pos_map.get(f, []))
                if self._phrase_match_in_field(seqs):
                    res.add(d)
                    break
        return res

    # ---------- Boolean evaluation ----------
    def _eval_rpn(self, rpn_tokens) -> Set[int]:
        stack: List[Set[int]] = []
        universe: Set[int] = set(self.len_by_doc.keys())

        # 预取 term -> doc_id set（加速 NOT/AND/OR 组合）
        cache_term_docs: Dict[str, Set[int]] = {}

        for tok in rpn_tokens:
            if isinstance(tok, TermToken):
                t = tok.term
                if t not in cache_term_docs:
                    df, plist = load_term_postings_all_segments(self.index_dir, t)
                    cache_term_docs[t] = {d for d, _ in plist}
                stack.append(cache_term_docs[t])
            elif isinstance(tok, PhraseToken):
                stack.append(self._phrase_docs(tok.terms))
            elif isinstance(tok, OpToken):
                if tok.op == "NOT":
                    a = stack.pop() if stack else set()
                    stack.append(universe - a)
                else:
                    if len(stack) < 2:
                        raise ValueError("Insufficient operands for boolean op.")
                    b = stack.pop()
                    a = stack.pop()
                    if tok.op == "AND":
                        stack.append(a & b)
                    elif tok.op == "OR":
                        stack.append(a | b)
                    else:
                        raise ValueError(f"Unknown operator {tok.op}")
            else:
                raise ValueError("Unknown token type.")
        if len(stack) != 1:
            raise ValueError("Invalid RPN evaluation stack.")
        return stack[0]

    # ---------- Ranking (BM25F) ----------
    def _df_by_term(self, terms: Iterable[str]) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for t in set(terms):
            dfi, _ = load_term_postings_all_segments(self.index_dir, t)
            df[t] = dfi
        return df

    def _tf_by_doc_field_for_terms(self, doc_ids: Set[int], terms: List[str]) -> Dict[int, Dict[str, Dict[str, int]]]:
        """Return doc_id -> term -> field -> tf."""
        # term -> postings
        term_cache: Dict[str, List[Tuple[int, Dict[str, List[int]]]]] = {}
        for t in set(terms):
            _, plist = load_term_postings_all_segments(self.index_dir, t)
            term_cache[t] = plist
        out: Dict[int, Dict[str, Dict[str, int]]] = {d: {} for d in doc_ids}
        for t in terms:
            plist = term_cache[t]
            m = dict(plist)
            for d in doc_ids:
                pos_map = m.get(d)
                if not pos_map:
                    continue
                out[d][t] = {f: len(pos_map.get(f, [])) for f in self.fields}
        return out

    def search(self, query: str, topk: int = 10) -> List[Tuple[int, float]]:
        rpn = parse_query(query, self.analyzer.analyze)
        candidate_docs = self._eval_rpn(rpn)

        positive_terms: List[str] = []
        for tok in rpn:
            if isinstance(tok, TermToken):
                positive_terms.append(tok.term)
            elif isinstance(tok, PhraseToken):
                positive_terms.extend(tok.terms)
        df_by_term = self._df_by_term(positive_terms)
        tf_by_doc_term_field = self._tf_by_doc_field_for_terms(candidate_docs, positive_terms)

        scored: List[Tuple[int, float]] = []
        for d in candidate_docs:
            len_by_field = self.len_by_doc.get(d, {f: 1 for f in self.fields})
            score = self._bm25f.score_doc(
                tf_by_term_field=tf_by_doc_term_field.get(d, {}),
                df_by_term=df_by_term,
                len_by_field=len_by_field,
            )
            scored.append((d, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(topk))]

    def doc_path(self, doc_id: int) -> str:
        return self.path_by_doc.get(doc_id, f"<unknown:{doc_id}>")

    # ---------- Build (orchestrate) ----------
    @staticmethod
    def build_fs(
        data_dir: str,
        out_dir: str,
        workers: int = 4,
        segment_size: int = 1000,
        field_weights: Optional[Dict[str, float]] = None,
        field_b: Optional[Dict[str, float]] = None,
    ) -> None:
        """Build FS index (segments -> merge -> manifest/doclen)."""
        os.makedirs(out_dir, exist_ok=True)
        fields = ["title", "body"]
        if field_weights is None:
            field_weights = {"title": 2.0, "body": 1.0}
        if field_b is None:
            field_b = {"title": 0.6, "body": 0.75}

        # 1) 并行构建分段
        lex_paths, total_docs, total_tokens, df_global, doc_lens = build_segments_parallel(
            data_dir=data_dir,
            index_dir=out_dir,
            workers=workers,
            segment_size=segment_size,
            fields=fields,
        )
        seg_entries = []
        for lex in lex_paths:
            name = os.path.basename(lex).split(".lex.json")[0]
            seg_entries.append({"name": name, "lex": os.path.basename(lex), "postings": f"{name}.postings.bin"})

        # 2) 计算字段平均长度
        if not doc_lens:
            raise RuntimeError("No documents indexed.")
        avglen = {
            "title": sum(d.title_len for d in doc_lens) / len(doc_lens),
            "body": sum(d.body_len for d in doc_lens) / len(doc_lens),
        }

        # 3) 写 doclen.json
        write_doclen(out_dir, doc_lens)

        # 4) 合并所有段 -> consolidated 段
        merged = merge_segments(out_dir, out_seg_name="main", fields=fields, manifest_segments=seg_entries)
        segments = [merged]  # 替换为单段

        # 5) 写 manifest.json
        write_manifest(
            index_dir=out_dir,
            fields=fields,
            field_weights=field_weights,
            field_b=field_b,
            n_docs=total_docs,
            avglen=avglen,
            segments=segments,
        )
        print(f"[FS-BUILD] Done. Docs={total_docs} AvgLen={avglen} Segments={len(segments)}")

