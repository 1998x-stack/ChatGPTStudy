# -*- coding: utf-8 -*-
"""File-system (FS) index storage: segments, lexicon, postings; parallel build and merge.

Disk layout (under index_dir):
- manifest.json
  {
    "version": 1,
    "fields": ["title","body"],
    "field_weights": {"title":2.0,"body":1.0},
    "field_b": {"title":0.6,"body":0.75},
    "n_docs": 12345,
    "avglen": {"title": 8.2, "body": 156.4},
    "segments": [
        {"name":"seg_000001","lex":"seg_000001.lex.json","postings":"seg_000001.postings.bin"},
        ...
    ]
  }

- doclen.json:
  [
    {"doc_id":0,"path": ".../file.txt", "title": 7, "body": 128},
    ...
  ]

- seg_XXXXXX.lex.json:
  {
    "term_a": {"df": 42, "offset": 0,     "length": 1234},
    "term_b": {"df": 10, "offset": 1234,  "length": 512},
    ...
  }

- seg_XXXXXX.postings.bin: binary VB stream for all terms in the segment.

Posting encoding (per term within postings.bin):
  [num_postings VB]
  [doc_gaps VB of length num_postings]
  For each posting (doc):
      For each field in manifest.fields (fixed order):
          [tf_f VB]
          [positions_gaps VB of length tf_f]

Author: Your Name
"""

from __future__ import annotations

import json
import math
import os
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .text_preprocess import Analyzer
from .vb_codec import vb_encode_list, vb_decode_stream, gaps_encode, gaps_decode


# -----------------------
# Data structures
# -----------------------

@dataclass
class DocLen:
    doc_id: int
    path: str
    title_len: int
    body_len: int


# -----------------------
# Helpers
# -----------------------

def _read_txt_title_body(path: str) -> Tuple[str, str]:
    """Read a txt file; first non-empty line as title; rest as body.

    中文注释：若文件为空或仅有一行，则 body 可能为空。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    if not lines:
        return "", ""
    # 找第一行非空作为 title
    title = ""
    i = 0
    while i < len(lines) and not lines[i]:
        i += 1
    if i < len(lines):
        title = lines[i]
        i += 1
    body = "\n".join(ln for ln in lines[i:] if ln)
    return title, body


def _encode_postings_multi_field(
    postings: List[Tuple[int, Dict[str, List[int]]]],
    fields: List[str],
) -> bytes:
    """Encode postings for a term with multi-field positions.

    Args:
        postings: list of (doc_id, {field -> positions(sorted)})
        fields: field order list, e.g., ["title","body"]

    Returns:
        Binary bytes (VB stream).
    """
    postings.sort(key=lambda x: x[0])
    doc_ids = [d for d, _ in postings]
    doc_gaps = gaps_encode(doc_ids)
    out = bytearray()
    out.extend(vb_encode_list([len(postings)]))
    out.extend(vb_encode_list(doc_gaps))
    for _, pos_map in postings:
        for f in fields:
            positions = pos_map.get(f, [])
            positions.sort()
            tf = len(positions)
            out.extend(vb_encode_list([tf]))
            if tf > 0:
                out.extend(vb_encode_list(gaps_encode(positions)))
    return bytes(out)


def _decode_postings_multi_field(
    data: bytes,
    fields: List[str],
) -> List[Tuple[int, Dict[str, List[int]]]]:
    """Decode postings encoded by _encode_postings_multi_field."""
    arr = vb_decode_stream(data)
    if not arr:
        return []
    n = arr[0]
    doc_ids = gaps_decode(arr[1 : 1 + n])
    k = 1 + n
    postings: List[Tuple[int, Dict[str, List[int]]]] = []
    for i in range(n):
        pos_map: Dict[str, List[int]] = {}
        for f in fields:
            tf = arr[k]
            k += 1
            if tf > 0:
                gaps = arr[k : k + tf]
                k += tf
                pos_map[f] = gaps_decode(gaps)
            else:
                pos_map[f] = []
        postings.append((doc_ids[i], pos_map))
    return postings


# -----------------------
# Segment Writer (per worker)
# -----------------------

def _worker_build_segment(
    args: Tuple[List[str], str, str, List[str], int]
) -> Tuple[str, int, int, Dict[str, int], List[DocLen]]:
    """Worker: build a single segment from file paths.

    Args:
        args: (file_paths, index_dir, seg_name, fields, start_doc_id)

    Returns:
        (seg_lex_path, n_indexed_docs, total_tokens, df_by_term, doc_lens)
    """
    file_paths, index_dir, seg_name, fields, start_doc_id = args
    analyzer = Analyzer()
    term_postings: Dict[str, List[Tuple[int, Dict[str, List[int]]]]] = defaultdict(list)
    df_tmp: Dict[str, int] = defaultdict(int)
    doc_lens: List[DocLen] = []

    next_doc_id = start_doc_id
    total_tokens = 0

    for path in file_paths:
        title, body = _read_txt_title_body(path)
        if not title and not body:
            continue
        # 中文注释：分别对 title/body 分析，并保留各自位置
        t_tokens = analyzer.analyze(title) if title else []
        b_tokens = analyzer.analyze(body) if body else []
        # 将字段内的 token 重新编号：位置从 0..N-1
        t_pos_map: Dict[str, List[int]] = defaultdict(list)
        for i, tk in enumerate(t_tokens):
            t_pos_map[tk].append(i)
        b_pos_map: Dict[str, List[int]] = defaultdict(list)
        for i, tk in enumerate(b_tokens):
            b_pos_map[tk].append(i)

        # 合并到多字段 posting：term -> {field -> positions}
        merged_terms = set(t_pos_map.keys()) | set(b_pos_map.keys())
        if not merged_terms:
            continue

        per_doc_term_pos: Dict[str, Dict[str, List[int]]] = {}
        for term in merged_terms:
            per_doc_term_pos[term] = {
                "title": t_pos_map.get(term, []),
                "body": b_pos_map.get(term, []),
            }

        # 收集到段内倒排
        for term, field_pos in per_doc_term_pos.items():
            # 仅当某字段有位置才视为出现
            if not field_pos["title"] and not field_pos["body"]:
                continue
            term_postings[term].append((next_doc_id, field_pos))
            df_tmp[term] += 1

        # 文档长度统计（字段级）
        doc_lens.append(
            DocLen(
                doc_id=next_doc_id,
                path=os.path.abspath(path),
                title_len=len(t_tokens),
                body_len=len(b_tokens),
            )
        )
        total_tokens += len(t_tokens) + len(b_tokens)
        next_doc_id += 1

    # 写段文件：词典 + postings.bin
    lex_path = os.path.join(index_dir, f"{seg_name}.lex.json")
    bin_path = os.path.join(index_dir, f"{seg_name}.postings.bin")
    term_meta = {}
    offset = 0
    with open(bin_path, "wb") as fb:
        for term in sorted(term_postings.keys()):
            plist = term_postings[term]
            blob = _encode_postings_multi_field(plist, fields)
            fb.write(blob)
            term_meta[term] = {"df": df_tmp[term], "offset": offset, "length": len(blob)}
            offset += len(blob)
    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump(term_meta, f, ensure_ascii=False)

    n_docs = len(doc_lens)
    return lex_path, n_docs, total_tokens, df_tmp, doc_lens


# -----------------------
# FS Index Writer (parallel segments)
# -----------------------

def build_segments_parallel(
    data_dir: str,
    index_dir: str,
    workers: int = 4,
    segment_size: int = 1000,
    fields: List[str] | None = None,
) -> Tuple[List[str], int, int, Dict[str, int], List[DocLen]]:
    """Build segments in parallel; return list of lex paths and stats.

    Returns:
        (lex_paths, total_docs, total_tokens, df_global, doc_lens_all)
    """
    if fields is None:
        fields = ["title", "body"]
    os.makedirs(index_dir, exist_ok=True)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    files.sort()
    if not files:
        raise FileNotFoundError("No .txt files under data_dir.")

    # 分段切片
    chunks: List[List[str]] = []
    for i in range(0, len(files), segment_size):
        chunks.append(files[i : i + segment_size])

    # 分配 doc_id 起始编号（每段连续，避免合并时冲突）
    start_ids = []
    _cur = 0
    for c in chunks:
        start_ids.append(_cur)
        _cur += len(c)  # 每个文件至少对应一个 doc_id，占位

    # 并行跑 worker
    args_list = []
    lex_paths: List[str] = []
    total_docs = 0
    total_tokens = 0
    df_global: Dict[str, int] = defaultdict(int)
    doc_lens_all: List[DocLen] = []

    seg_names = [f"seg_{i:06d}" for i in range(len(chunks))]

    if workers <= 1:
        for chunk, seg_name, sid in zip(chunks, seg_names, start_ids):
            lex, n_docs, n_toks, df_tmp, dls = _worker_build_segment((chunk, index_dir, seg_name, fields, sid))
            lex_paths.append(lex)
            total_docs += n_docs
            total_tokens += n_toks
            for k, v in df_tmp.items():
                df_global[k] += v
            doc_lens_all.extend(dls)
    else:
        with mp.Pool(processes=workers) as pool:
            for chunk, seg_name, sid in zip(chunks, seg_names, start_ids):
                args_list.append((chunk, index_dir, seg_name, fields, sid))
            for lex, n_docs, n_toks, df_tmp, dls in pool.imap_unordered(_worker_build_segment, args_list):
                lex_paths.append(lex)
                total_docs += n_docs
                total_tokens += n_toks
                for k, v in df_tmp.items():
                    df_global[k] += v
                doc_lens_all.extend(dls)

    return lex_paths, total_docs, total_tokens, df_global, doc_lens_all


# -----------------------
# Segment Merge
# -----------------------

def _iter_term_postings_from_segments(
    index_dir: str,
    fields: List[str],
    seg_entries: List[Dict],
    term: str,
) -> List[Tuple[int, Dict[str, List[int]]]]:
    """Load postings for a term from all segments and merge by doc_id (sum per-field tfs)."""
    merged: Dict[int, Dict[str, List[int]]] = {}
    for seg in seg_entries:
        lex_path = os.path.join(index_dir, seg["lex"])
        bin_path = os.path.join(index_dir, seg["postings"])
        with open(lex_path, "r", encoding="utf-8") as f:
            lex = json.load(f)
        meta = lex.get(term)
        if not meta:
            continue
        with open(bin_path, "rb") as fb:
            fb.seek(meta["offset"])
            blob = fb.read(meta["length"])
        plist = _decode_postings_multi_field(blob, fields)
        for doc_id, pos_map in plist:
            m = merged.setdefault(doc_id, {f: [] for f in fields})
            for f in fields:
                if pos_map.get(f):
                    # 合并位置（保持有序）——谨慎起见合并后再排序
                    m[f].extend(pos_map[f])
    # 清理并排序
    result: List[Tuple[int, Dict[str, List[int]]]] = []
    for d, pm in merged.items():
        for f in fields:
            pm[f].sort()
        result.append((d, pm))
    result.sort(key=lambda x: x[0])
    return result


def merge_segments(
    index_dir: str,
    out_seg_name: str,
    fields: List[str],
    manifest_segments: List[Dict],
) -> Dict[str, Dict]:
    """Merge many segments into a single consolidated segment; return lex dict."""
    # 中文注释：逐 term 外排合并（为简洁起见，先取所有 term 并排序；大规模可改为多路流式）
    all_terms = set()
    for seg in manifest_segments:
        with open(os.path.join(index_dir, seg["lex"]), "r", encoding="utf-8") as f:
            lex = json.load(f)
        all_terms.update(lex.keys())
    all_terms = sorted(all_terms)

    lex_out_path = os.path.join(index_dir, f"{out_seg_name}.lex.json")
    bin_out_path = os.path.join(index_dir, f"{out_seg_name}.postings.bin")

    term_meta: Dict[str, Dict] = {}
    offset = 0
    with open(bin_out_path, "wb") as fb:
        for term in all_terms:
            plist = _iter_term_postings_from_segments(index_dir, fields, manifest_segments, term)
            if not plist:
                continue
            # df 为文档数
            df = sum(1 for _, pm in plist if any(len(pm[f]) > 0 for f in fields))
            blob = _encode_postings_multi_field(plist, fields)
            fb.write(blob)
            term_meta[term] = {"df": df, "offset": offset, "length": len(blob)}
            offset += len(blob)

    with open(lex_out_path, "w", encoding="utf-8") as f:
        json.dump(term_meta, f, ensure_ascii=False)

    return {"lex": os.path.basename(lex_out_path), "postings": os.path.basename(bin_out_path)}


# -----------------------
# Manifest & Doclen
# -----------------------

def write_manifest(
    index_dir: str,
    fields: List[str],
    field_weights: Dict[str, float],
    field_b: Dict[str, float],
    n_docs: int,
    avglen: Dict[str, float],
    segments: List[Dict],
) -> None:
    man = {
        "version": 1,
        "fields": fields,
        "field_weights": field_weights,
        "field_b": field_b,
        "n_docs": n_docs,
        "avglen": avglen,
        "segments": segments,
    }
    with open(os.path.join(index_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)


def write_doclen(index_dir: str, doc_lens: List[DocLen]) -> None:
    arr = [
        {"doc_id": d.doc_id, "path": d.path, "title": d.title_len, "body": d.body_len}
        for d in sorted(doc_lens, key=lambda x: x.doc_id)
    ]
    with open(os.path.join(index_dir, "doclen.json"), "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False)


def read_manifest(index_dir: str) -> Dict:
    with open(os.path.join(index_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def read_doclen(index_dir: str) -> List[Dict]:
    with open(os.path.join(index_dir, "doclen.json"), "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------
# FS Query-time helpers
# -----------------------

def load_term_postings_all_segments(index_dir: str, term: str) -> Tuple[int, List[Tuple[int, Dict[str, List[int]]]]]:
    """Load postings (merged) for a term from manifest (all segments)."""
    man = read_manifest(index_dir)
    fields = man["fields"]
    segs = man["segments"]
    plist = _iter_term_postings_from_segments(index_dir, fields, segs, term)
    df = len(plist)
    return df, plist

