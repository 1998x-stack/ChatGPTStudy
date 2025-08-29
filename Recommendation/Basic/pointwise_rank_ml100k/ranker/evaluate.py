# -*- coding: utf-8 -*-
"""Evaluation metrics: RMSE, MAE, and NDCG@K."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute MAE."""
    return float(np.mean(np.abs(pred - target)))


def _dcg(rels: List[float]) -> float:
    """Compute DCG for a relevance list."""
    dcg = 0.0
    for i, rel in enumerate(rels, start=1):
        dcg += (2**rel - 1.0) / np.log2(i + 1.0)
    return float(dcg)


def ndcg_at_k(
    user_item_scores: Dict[int, List[Tuple[int, float]]],
    user_item_labels: Dict[int, Dict[int, float]],
    k: int,
) -> float:
    """Compute mean NDCG@K across users.

    Args:
        user_item_scores: Mapping user -> list of (item, score) to rank.
        user_item_labels: Mapping user -> dict of item -> true relevance (rating).
        k: Cutoff K.

    Returns:
        float: Mean NDCG@K.
    """
    ndcgs: List[float] = []
    for u, pairs in user_item_scores.items():
        # 排序取top-k
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)[:k]
        rels = [user_item_labels.get(u, {}).get(i, 0.0) for i, _ in ranked]
        dcg = _dcg(rels)
        # 理想排序（按真实相关度排序）
        ideal_rels = sorted(user_item_labels.get(u, {}).values(), reverse=True)[:k]
        idcg = _dcg(ideal_rels) if ideal_rels else 0.0
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs) if ndcgs else 0.0)
