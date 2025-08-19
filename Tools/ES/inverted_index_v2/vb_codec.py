# -*- coding: utf-8 -*-
"""Variable-Byte (VB) codec with gap encoding for integer lists."""

from __future__ import annotations
from typing import Iterable, List

def vb_encode_number(n: int) -> bytes:
    if n < 0:
        raise ValueError("vb_encode_number requires non-negative integer.")
    bytes_out = bytearray()
    if n == 0:
        bytes_out.append(0x80)
        return bytes(bytes_out)
    while n > 0:
        bytes_out.insert(0, n & 0x7F)
        n >>= 7
    bytes_out[-1] |= 0x80
    return bytes(bytes_out)

def vb_encode_list(nums: Iterable[int]) -> bytes:
    out = bytearray()
    for n in nums:
        out.extend(vb_encode_number(int(n)))
    return bytes(out)

def vb_decode_stream(data: bytes) -> List[int]:
    res: List[int] = []
    n = 0
    for b in data:
        if b & 0x80:
            n = (n << 7) | (b & 0x7F)
            res.append(n)
            n = 0
        else:
            n = (n << 7) | b
    if n != 0:
        raise ValueError("Malformed VB stream (dangling partial integer).")
    return res

def gaps_encode(sorted_ids: List[int]) -> List[int]:
    if not sorted_ids:
        return []
    prev = None
    gaps: List[int] = []
    for x in sorted_ids:
        if x < 0:
            raise ValueError("IDs must be non-negative.")
        if prev is None:
            gaps.append(x)
            prev = x
            continue
        if x <= prev:
            raise ValueError("IDs must be strictly increasing for gap encoding.")
        gaps.append(x - prev)
        prev = x
    return gaps

def gaps_decode(gaps: List[int]) -> List[int]:
    if not gaps:
        return []
    res: List[int] = []
    total = 0
    for idx, g in enumerate(gaps):
        if g < 0:
            raise ValueError("Gap cannot be negative.")
        if idx == 0:
            total = g
        else:
            total += g
        res.append(total)
    return res

