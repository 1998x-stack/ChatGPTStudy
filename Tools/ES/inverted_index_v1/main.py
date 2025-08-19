# -*- coding: utf-8 -*-
"""CLI entry point for building and querying the inverted index.

Usage:
    python -m inverted_index.main --data ./data --index-out ./index.pkl
    python -m inverted_index.main --load ./index.pkl --query '"deep learning" AND training' --topk 10
    python -m inverted_index.main --selfcheck

Author: Your Name
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .index import InvertedIndex


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverted Index with BM25, skip list, VB compression, phrase/boolean search.")
    parser.add_argument("--data", type=str, help="Folder containing English .txt files.")
    parser.add_argument("--index-out", type=str, help="Path to save built index (pickle).")
    parser.add_argument("--load", type=str, help="Path to load existing index (pickle).")
    parser.add_argument("--query", type=str, help="Query string (use quotes for phrase).")
    parser.add_argument("--topk", type=int, default=10, help="Top K results to display.")
    parser.add_argument("--selfcheck", action="store_true", help="Run internal correctness checks.")
    return parser.parse_args(argv)


def _interactive_loop(idx: InvertedIndex, topk: int) -> None:
    print("\n[INTERACTIVE] Type your query. Use double quotes for phrases. Type :quit to exit.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in (":q", ":quit", ":exit"):
            break
        try:
            results = idx.search(q, topk=topk)
            print(f"[RESULT] Top-{topk}:")
            for rank, (doc_id, score) in enumerate(results, 1):
                meta = idx.docs[doc_id]
                print(f"  {rank:2d}. doc={doc_id} score={score:.4f} path={meta.path} len={meta.length}")
        except Exception as e:
            print(f"[ERROR] {e}")


def _self_check() -> None:
    """Run essential correctness and logic checks."""
    print("[SELFCHECK] Start self-check...")

    # 1) Build tiny synthetic index for deterministic assertions
    idx = InvertedIndex()
    tiny_dir = os.path.join(os.path.dirname(__file__), "_tiny")
    os.makedirs(tiny_dir, exist_ok=True)
    with open(os.path.join(tiny_dir, "a.txt"), "w", encoding="utf-8") as f:
        f.write("I love deep learning. Deep learning is fun.")
    with open(os.path.join(tiny_dir, "b.txt"), "w", encoding="utf-8") as f:
        f.write("I study machine learning and love it.")
    with open(os.path.join(tiny_dir, "c.txt"), "w", encoding="utf-8") as f:
        f.write("This note is a draft about cooking. Nothing about learning.")

    idx.build_from_folder(tiny_dir)

    # 2) Phrase matching sanity
    res1 = idx.search('"deep learning"', topk=10)
    assert len(res1) >= 1 and idx.docs[res1[0][0]].path.endswith("a.txt"), "Phrase query failed."

    # 3) Boolean AND/OR/NOT
    res2 = idx.search('learning AND love', topk=10)
    ids2 = {d for d, _ in res2}
    assert any(idx.docs[d].path.endswith("a.txt") for d in ids2), "Boolean AND missing a.txt"
    assert any(idx.docs[d].path.endswith("b.txt") for d in ids2), "Boolean AND missing b.txt"

    res3 = idx.search('learning AND NOT cooking', topk=10)
    ids3 = {d for d, _ in res3}
    assert all(not idx.docs[d].path.endswith("c.txt") for d in ids3), "NOT filter failed."

    # 4) Compression roundtrip (pick one term)
    term = "learn"  # stem of 'learning'
    pl = idx.term_to_pl.get(term)
    assert pl is not None, "Missing posting list for 'learn'."
    data = pl.to_compressed()
    from .postings import PostingList
    pl2 = PostingList.from_compressed(data)
    assert [p.doc_id for p in pl.postings] == [p.doc_id for p in pl2.postings], "DocID mismatch after compression."
    # 不强制 positions 完全相同（空/非空），但应一致：
    assert [p.positions for p in pl.postings] == [p.positions for p in pl2.postings], "Positions mismatch after compression."

    # 5) Ranking sanity (BM25 should reward 'a.txt' for 'deep learning')
    res4 = idx.search('deep learning', topk=3)
    assert idx.docs[res4[0][0]].path.endswith("a.txt"), "BM25 ranking sanity failed."

    print("[SELFCHECK] All checks passed ✅")


def main() -> None:
    args = _parse_args(sys.argv[1:])

    if args.selfcheck:
        _self_check()
        return

    if args.load:
        idx = InvertedIndex.load(args.load)
    else:
        if not args.data:
            print("[ERROR] --data is required when not using --load")
            sys.exit(2)
        idx = InvertedIndex()
        idx.build_from_folder(args.data)
        if args.index_out:
            idx.save(args.index_out)

    if args.query:
        results = idx.search(args.query, topk=args.topk)
        print(f"[RESULT] Top-{args.topk}:")
        for rank, (doc_id, score) in enumerate(results, 1):
            meta = idx.docs[doc_id]
            print(f"  {rank:2d}. doc={doc_id} score={score:.4f} path={meta.path} len={meta.length}")
    else:
        _interactive_loop(idx, args.topk)


if __name__ == "__main__":
    main()

