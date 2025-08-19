# -*- coding: utf-8 -*-
"""Posting and PostingList with Skip List and VB+gap compression (single-field).

This module is used by the in-memory index path for skip-list boolean ops demo.
FS index uses storage_fs for multi-field encoding/decoding.

Author: Your Name
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .vb_codec import vb_encode_list, vb_decode_stream, gaps_encode, gaps_decode


@dataclass
class Posting:
    """A posting record of a term in a single document (single-field)."""
    doc_id: int
    tf: int
    positions: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.doc_id < 0:
            raise ValueError("doc_id must be non-negative.")
        self.positions.sort()
        if self.tf != len(self.positions):
            self.tf = len(self.positions)


class PostingList:
    """PostingList holding postings and skip pointers, with compression support."""

    def __init__(self) -> None:
        self.postings: List[Posting] = []
        self._skip_to: List[int] = []
        self._finalized: bool = False
        self._compressed: Optional[bytes] = None

    def add_occurrence(self, doc_id: int, position: int) -> None:
        if not self.postings or self.postings[-1].doc_id != doc_id:
            self.postings.append(Posting(doc_id=doc_id, tf=1, positions=[position]))
        else:
            p = self.postings[-1]
            if p.positions and position < p.positions[-1]:
                raise ValueError("Positions must be non-decreasing during build.")
            p.positions.append(position)
            p.tf += 1

    def finalize(self) -> None:
        if self._finalized:
            return
        self.postings.sort(key=lambda x: x.doc_id)
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
        if not self._finalized or not other._finalized:
            raise RuntimeError("PostingList must be finalized before boolean ops.")
        return self._intersect_with_skip(other)

    def union(self, other: "PostingList") -> "PostingList":
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

    def to_compressed(self) -> bytes:
        """Serialize posting list to compressed bytes (gap+VB)."""
        self.finalize()
        n = len(self.postings)
        doc_ids = [p.doc_id for p in self.postings]
        gaps = gaps_encode(doc_ids)
        doc_bytes = vb_encode_list(gaps)
        tail = bytearray()
        for p in self.postings:
            pos_gaps = gaps_encode(p.positions) if p.positions else []
            tail.extend(vb_encode_list([p.tf]))
            tail.extend(vb_encode_list(pos_gaps))
        header = vb_encode_list([n])
        self._compressed = bytes(header + doc_bytes + tail)
        return self._compressed

    @staticmethod
    def from_compressed(data: bytes) -> "PostingList":
        nums = vb_decode_stream(data)
        if not nums:
            return PostingList()
        n = nums[0]
        if n < 0:
            raise ValueError("Invalid posting count.")
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
            from .vb_codec import gaps_decode
            positions = gaps_decode(pos_gaps)
            postings.append(Posting(doc_id=doc_ids[len(postings)], tf=tf, positions=positions))
        pl = PostingList()
        pl.postings = postings
        pl.finalize()
        pl._compressed = data
        return pl

