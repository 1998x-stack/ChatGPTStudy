# -*- coding: utf-8 -*-
"""Posting and PostingList with Skip List and VB+gap compression.

This module defines:
- Posting: doc_id, tf, positions
- PostingList: container with sorting, skip pointers, boolean ops (AND/OR/NOT)
  aided by skip pointers, and compression (gap + VB).

Author: Your Name
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .vb_codec import vb_encode_list, vb_decode_stream, gaps_encode, gaps_decode


@dataclass
class Posting:
    """A posting record of a term in a single document.

    Attributes:
        doc_id: Integer document identifier.
        tf: Term frequency in the document.
        positions: Sorted positions where the term occurs within the processed token sequence.
    """
    doc_id: int
    tf: int
    positions: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 中文注释：位置必须严格递增，tf 必须匹配 positions 长度
        if self.doc_id < 0:
            raise ValueError("doc_id must be non-negative.")
        self.positions.sort()
        if self.tf != len(self.positions):
            self.tf = len(self.positions)


class PostingList:
    """PostingList holding postings and skip pointers, with compression support."""

    def __init__(self) -> None:
        # 中文注释：postings 按 doc_id 递增，skip_to 同步维护跳转索引
        self.postings: List[Posting] = []
        self._skip_to: List[int] = []  # -1 表示无跳转
        self._finalized: bool = False
        self._compressed: Optional[bytes] = None  # 压缩存储（可选）

    def add_occurrence(self, doc_id: int, position: int) -> None:
        """Add a single occurrence of the term (doc_id, position).

        Args:
            doc_id: Document id.
            position: Position index in processed tokens.
        """
        # 中文注释：构建时可能多次调用，合并到同一 doc 的 posting 中
        if not self.postings or self.postings[-1].doc_id != doc_id:
            self.postings.append(Posting(doc_id=doc_id, tf=1, positions=[position]))
        else:
            p = self.postings[-1]
            # 保证位置递增
            if p.positions and position < p.positions[-1]:
                raise ValueError("Positions must be non-decreasing during build.")
            p.positions.append(position)
            p.tf += 1

    def finalize(self) -> None:
        """Finalize the posting list: sort, set skip pointers, and lock structure."""
        if self._finalized:
            return
        # 中文注释：确保 doc_id 严格递增
        self.postings.sort(key=lambda x: x.doc_id)
        # 跳表间隔选择：√n
        n = len(self.postings)
        self._skip_to = [-1] * n
        if n > 0:
            step = int(math.sqrt(n))
            if step >= 2:
                for i in range(0, n, step):
                    j = i + step
                    if j < n:
                        self._skip_to[i] = j
        self._finalized = True

    def __len__(self) -> int:
        return len(self.postings)

    def _intersect_with_skip(self, other: "PostingList") -> "PostingList":
        """Efficient intersection using skip pointers."""
        a = self.postings
        b = other.postings
        res = PostingList()
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i].doc_id == b[j].doc_id:
                # 交集时只需 doc_id，positions 留给短语匹配
                res.postings.append(Posting(doc_id=a[i].doc_id, tf=0, positions=[]))
                i += 1
                j += 1
            elif a[i].doc_id < b[j].doc_id:
                # 利用跳表
                si = self._skip_to[i] if i < len(self._skip_to) else -1
                if si != -1 and a[si].doc_id <= b[j].doc_id:
                    i = si
                else:
                    i += 1
            else:
                sj = other._skip_to[j] if j < len(other._skip_to) else -1
                if sj != -1 and b[sj].doc_id <= a[i].doc_id:
                    j = sj
                else:
                    j += 1
        res.finalize()
        return res

    def intersect(self, other: "PostingList") -> "PostingList":
        """Set intersection (AND)."""
        if not self._finalized or not other._finalized:
            raise RuntimeError("PostingList must be finalized before boolean ops.")
        return self._intersect_with_skip(other)

    def union(self, other: "PostingList") -> "PostingList":
        """Set union (OR)."""
        if not self._finalized or not other._finalized:
            raise RuntimeError("PostingList must be finalized before boolean ops.")
        a = self.postings
        b = other.postings
        res = PostingList()
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i].doc_id == b[j].doc_id:
                res.postings.append(Posting(doc_id=a[i].doc_id, tf=0, positions=[]))
                i += 1
                j += 1
            elif a[i].doc_id < b[j].doc_id:
                res.postings.append(Posting(doc_id=a[i].doc_id, tf=0, positions=[]))
                i += 1
            else:
                res.postings.append(Posting(doc_id=b[j].doc_id, tf=0, positions=[]))
                j += 1
        while i < len(a):
            res.postings.append(Posting(doc_id=a[i].doc_id, tf=0, positions=[]))
            i += 1
        while j < len(b):
            res.postings.append(Posting(doc_id=b[j].doc_id, tf=0, positions=[]))
            j += 1
        res.finalize()
        return res

    def difference(self, other: "PostingList") -> "PostingList":
        """Set difference (A - B)."""
        if not self._finalized or not other._finalized:
            raise RuntimeError("PostingList must be finalized before boolean ops.")
        a = self.postings
        b = other.postings
        res = PostingList()
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i].doc_id == b[j].doc_id:
                i += 1
                j += 1
            elif a[i].doc_id < b[j].doc_id:
                res.postings.append(Posting(doc_id=a[i].doc_id, tf=0, positions=[]))
                i += 1
            else:
                j += 1
        while i < len(a):
            res.postings.append(Posting(doc_id=a[i].doc_id, tf=0, positions=[]))
            i += 1
        res.finalize()
        return res

    # -------------------------
    # Compression
    # -------------------------
    def to_compressed(self) -> bytes:
        """Serialize posting list to compressed bytes (gap+VB).

        Format:
            [num_postings VB]
            [doc_gaps VB list]
            For each posting:
                [tf VB]
                [positions_gaps VB list with length=tf]

        Returns:
            Compressed bytes.
        """
        self.finalize()
        n = len(self.postings)
        doc_ids = [p.doc_id for p in self.postings]
        gaps = gaps_encode(doc_ids)
        doc_bytes = vb_encode_list(gaps)
        tail = bytearray()
        for p in self.postings:
            # tf + positions gaps
            pos_gaps = gaps_encode(p.positions) if p.positions else []
            tail.extend(vb_encode_list([p.tf]))
            tail.extend(vb_encode_list(pos_gaps))
        header = vb_encode_list([n])
        self._compressed = bytes(header + doc_bytes + tail)
        return self._compressed

    @staticmethod
    def from_compressed(data: bytes) -> "PostingList":
        """Deserialize from compressed bytes produced by to_compressed."""
        nums = vb_decode_stream(data)
        if not nums:
            return PostingList()
        # First integer = number of postings
        n = nums[0]
        if n < 0:
            raise ValueError("Invalid posting count.")
        # Next: doc gaps of length n
        if len(nums) < 1 + n:
            raise ValueError("Malformed compressed posting list.")
        doc_ids = gaps_decode(nums[1 : 1 + n])
        postings: List[Posting] = []
        k = 1 + n
        for _ in range(n):
            if k >= len(nums):
                raise ValueError("Malformed tail (missing tf/positions).")
            tf = nums[k]
            k += 1
            if tf < 0:
                raise ValueError("Negative tf not allowed.")
            if tf == 0:
                postings.append(Posting(doc_id=doc_ids[len(postings)], tf=0, positions=[]))
                continue
            if k + tf - 1 >= len(nums):
                raise ValueError("Positions length overflow.")
            pos_gaps = nums[k : k + tf]
            k += tf
            positions = gaps_decode(pos_gaps)
            postings.append(Posting(doc_id=doc_ids[len(postings)], tf=tf, positions=positions))

        pl = PostingList()
        pl.postings = postings
        pl.finalize()
        pl._compressed = data
        return pl

