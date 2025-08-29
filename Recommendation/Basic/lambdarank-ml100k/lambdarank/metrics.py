from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
import torch


def dcg_at_k(rels: Sequence[float], k: int) -> float:
    """Compute DCG@k.

    Args:
        rels: Relevance scores ordered by predicted rank (descending).
        k: Cutoff.

    Returns:
        DCG value at k.
    """
    # 计算 DCG@k：位置越靠前权重越大
    rels = np.asfarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    gains = np.power(2.0, rels) - 1.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(true_rels: Sequence[float], pred_scores: Sequence[float], k: int) -> float:
    """Compute NDCG@k given true relevances and predicted scores.

    Args:
        true_rels: Ground-truth relevance labels.
        pred_scores: Predicted scores (higher means more relevant).
        k: Cutoff.

    Returns:
        NDCG value at k in [0, 1].
    """
    # 先根据预测分数排序，再按该顺序计算 DCG；IDCG 用真实相关性降序排序
    if len(true_rels) == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_scores))
    ranked_rels = np.asarray(true_rels)[order]
    dcg = dcg_at_k(ranked_rels, k)
    ideal_rels = np.sort(np.asarray(true_rels))[::-1]
    idcg = dcg_at_k(ideal_rels, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def map_at_k(true_binary: Sequence[int], pred_scores: Sequence[float], k: int) -> float:
    """Compute MAP@k for binary relevance.

    Args:
        true_binary: Ground-truth binary relevance (0/1).
        pred_scores: Predicted scores.
        k: Cutoff.

    Returns:
        MAP@k value.
    """
    # 计算 MAP@k，仅用于二值相关性
    order = np.argsort(-np.asarray(pred_scores))
    ranked = np.asarray(true_binary)[order][:k]
    if ranked.size == 0:
        return 0.0
    precisions = []
    hit = 0
    for idx, rel in enumerate(ranked, start=1):
        if rel == 1:
            hit += 1
            precisions.append(hit / idx)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(true_binary: Sequence[int], pred_scores: Sequence[float], k: int) -> float:
    """Compute Recall@k for binary relevance.

    Args:
        true_binary: Ground-truth binary relevance (0/1).
        pred_scores: Predicted scores.
        k: Cutoff.

    Returns:
        Recall@k value.
    """
    # 计算 Recall@k，仅用于二值相关性
    total_pos = int(np.sum(true_binary))
    if total_pos == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_scores))
    ranked = np.asarray(true_binary)[order][:k]
    hit = int(np.sum(ranked))
    return float(hit / total_pos)