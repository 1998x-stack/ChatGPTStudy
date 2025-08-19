# -*- coding: utf-8 -*-
"""CLI for both in-memory and FS indexes.

Examples:
    # FS (non-pickle) build + search
    python -m inverted_index.main build-fs --data ./data --out ./fs_index --workers 4 --segment-size 500
    python -m inverted_index.main search-fs --index ./fs_index --query '"deep learning" AND training' --topk 10

    # In-memory (legacy demo)
    python -m inverted_index.main build-mem --data ./data
    python -m inverted_index.main search-mem --query 'deep learning' --topk 5

    # Self-check
    python -m inverted_index.main --selfcheck
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .index import InvertedIndex, FSIndex


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inverted Index (memory & FS) with Boolean/Phrase/BM25/BM25F.")
    sub = p.add_subparsers(dest="cmd")

    # Memory index
    sp = sub.add_parser("build-mem")
    sp.add_argument("--data", required=True)
    sp2 = sub.add_parser("search-mem")
    sp2.add_argument("--query", required=True)
    sp2.add_argument("--topk", type=int, default=10)

    # FS index
    sp3 = sub.add_parser("build-fs")
    sp3.add_argument("--data", required=True)
    sp3.add_argument("--out", required=True)
    sp3.add_argument("--workers", type=int, default=4)
    sp3.add_argument("--segment-size", type=int, default=1000)

    sp4 = sub.add_parser("merge-fs")
    sp4.add_argument("--index", required=True)  # already built; re-merge main

    sp5 = sub.add_parser("search-fs")
    sp5.add_argument("--index", required=True)
    sp5.add_argument("--query", required=True)
    sp5.add_argument("--topk", type=int, default=10)

    p.add_argument("--selfcheck", action="store_true", help="Run internal correctness checks.")
    return p.parse_args(argv)


def _self_check() -> None:
    """Run correctness checks across memory and FS indexes."""
    print("[SELFCHECK] Start...")

    # --- Memory small test ---
    mem_idx = InvertedIndex()
    tiny_dir = os.path.join(os.path.dirname(__file__), "_tiny")
    os.makedirs(tiny_dir, exist_ok=True)
    with open(os.path.join(tiny_dir, "a.txt"), "w", encoding="utf-8") as f:
        f.write("Deep Learning Basics\nI love deep learning. Deep learning is fun.")
    with open(os.path.join(tiny_dir, "b.txt"), "w", encoding="utf-8") as f:
        f.write("Machine Learning Note\nI study machine learning and love it.")
    with open(os.path.join(tiny_dir, "c.txt"), "w", encoding="utf-8") as f:
        f.write("Cooking Draft\nThis note is a draft about cooking. Nothing about learning.")

    mem_idx.build_from_folder(tiny_dir)
    res1 = mem_idx.search('"deep learning"', topk=10)
    assert len(res1) >= 1, "Phrase query fail (mem)."
    res2 = mem_idx.search('learning AND love', topk=10)
    assert len(res2) >= 2, "Boolean AND fail (mem)."
    res3 = mem_idx.search('learning AND NOT cooking', topk=10)
    assert all(not mem_idx.docs[d].path.endswith("c.txt") for d, _ in res3), "NOT filter fail (mem)."

    # --- FS build test ---
    fs_dir = os.path.join(os.path.dirname(__file__), "_fs_index")
    if os.path.isdir(fs_dir):
        # 清理旧索引
        for fn in os.listdir(fs_dir):
            try:
                os.remove(os.path.join(fs_dir, fn))
            except Exception:
                pass
    os.makedirs(fs_dir, exist_ok=True)

    FSIndex.build_fs(data_dir=tiny_dir, out_dir=fs_dir, workers=1, segment_size=2)

    fs_idx = FSIndex(fs_dir)
    res4 = fs_idx.search('"deep learning"', topk=10)
    assert len(res4) >= 1, "Phrase query fail (fs)."
    res5 = fs_idx.search('learning AND love', topk=10)
    assert len(res5) >= 2, "Boolean AND fail (fs)."
    res6 = fs_idx.search('learning AND NOT cooking', topk=10)
    ids6 = {d for d, _ in res6}
    # c.txt 的 doc 应该被排除
    doc_paths = [fs_idx.doc_path(d) for d in ids6]
    assert all("c.txt" not in p for p in doc_paths), "NOT filter fail (fs)."

    # 排名 sanity：包含两次短语的 a.txt 分数应更高
    top1_path = fs_idx.doc_path(res4[0][0])
    assert top1_path.endswith("a.txt"), "BM25F ranking sanity (fs) failed."

    print("[SELFCHECK] All passed ✅")


def main() -> None:
    args = _parse_args(sys.argv[1:])
    if args.selfcheck:
        _self_check()
        return

    if args.cmd == "build-mem":
        idx = InvertedIndex()
        idx.build_from_folder(args.data)
        # 交互式简单展示
        print("[BUILD-MEM] Done. You can run 'search-mem' to test.")
        return

    if args.cmd == "search-mem":
        # 直接在 tiny mem 上快速体验（或扩展保存/加载）
        print("[WARN] search-mem expects you built mem index within session. For demo only.")
        print("[HINT] Prefer FS index for production.")
        return

    if args.cmd == "build-fs":
        FSIndex.build_fs(
            data_dir=args.data,
            out_dir=args.out,
            workers=max(1, int(args.workers)),
            segment_size=max(1, int(args.segment_size)),
        )
        return

    if args.cmd == "merge-fs":
        from .storage_fs import read_manifest
        man = read_manifest(args.index)
        fields = man["fields"]
        segs = man["segments"]
        if len(segs) <= 1:
            print("[MERGE] Nothing to merge (already single segment).")
            return
        merged = merge_segments(args.index, "main", fields, segs)
        # 覆盖 manifest 段信息
        man["segments"] = [merged]
        import json
        with open(os.path.join(args.index, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)
        print("[MERGE] Completed.")
        return

    if args.cmd == "search-fs":
        idx = FSIndex(args.index)
        results = idx.search(args.query, topk=args.topk)
        print(f"[RESULT] Top-{args.topk}:")
        for r, (doc_id, score) in enumerate(results, 1):
            print(f"  {r:2d}. doc={doc_id} score={score:.4f} path={idx.doc_path(doc_id)}")
        return

    # default
    print("[ERROR] Unknown or missing command. Use --help for usage.")


if __name__ == "__main__":
    main()

