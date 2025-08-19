# -*- coding: utf-8 -*-
"""Variable-Byte (VB) codec with gap encoding for integer lists.

This module implements classic VB encoding/decoding and gap encoding for
sorted integer sequences (e.g., docIDs, positions). It is optimized for clarity
and correctness with boundary checks.

Author: Your Name
"""

from __future__ import annotations

from typing import Iterable, List


def vb_encode_number(n: int) -> bytes:
    """Encode a non-negative integer using VB encoding.

    Args:
        n: Non-negative integer.

    Returns:
        Encoded bytes (little-end groups, MSB=1 in last byte).

    Raises:
        ValueError: If n is negative.
    """
    # 中文注释：VB 编码将整数拆为 7-bit 分组，最后一组最高位为1
    if n < 0:
        raise ValueError("vb_encode_number requires non-negative integer.")
    bytes_out = bytearray()
    if n == 0:
        bytes_out.append(0x80)
        return bytes(bytes_out)
    while n > 0:
        bytes_out.insert(0, n & 0x7F)
        n >>= 7
    bytes_out[-1] |= 0x80  # 终止标志位
    return bytes(bytes_out)


def vb_encode_list(nums: Iterable[int]) -> bytes:
    """Encode a list of non-negative integers using VB encoding.

    Args:
        nums: Iterable of non-negative integers.

    Returns:
        Concatenated VB-encoded bytes.
    """
    out = bytearray()
    for n in nums:
        out.extend(vb_encode_number(int(n)))
    return bytes(out)


def vb_decode_stream(data: bytes) -> List[int]:
    """Decode a VB-encoded bytes buffer into a list of integers.

    Args:
        data: Bytes buffer.

    Returns:
        Decoded integers.

    Raises:
        ValueError: If the stream is malformed.
    """
    # 中文注释：逐字节解码，遇到最高位1表示一个整数结束
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
    """Gap-encode a strictly increasing integer list.

    Args:
        sorted_ids: Strictly increasing non-negative integers.

    Returns:
        Gap-encoded sequence.

    Raises:
        ValueError: If sequence is not strictly increasing or contains negatives.
    """
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
    """Decode gap-encoded list back to absolute integers.

    Args:
        gaps: Non-negative integers (first is absolute).

    Returns:
        Absolute decoded list.
    """
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

