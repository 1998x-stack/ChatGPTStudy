# -*- coding: utf-8 -*-
"""Utilities: seeding, logging setup, metrics, and parameter counting."""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seeds for reproducibility.

    Args:
        seed: Global random seed.

    中文说明：
        统一设置 Python、NumPy 和 PyTorch 的随机种子，保证实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了更可重复的结果，关闭 cuDNN 的某些非确定性优化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(name: str = "twotower", level: int = logging.INFO) -> logging.Logger:
    """Create a colored console logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recall_at_k(ranked_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
    """Compute Recall@K.

    Args:
        ranked_indices: 2D array [num_users, num_items_sorted] of item ids in ranked order.
        ground_truth: List of ground-truth positive item IDs for each user (single label supported here).
        k: Cutoff.

    Returns:
        Recall@K.

    中文说明：
        简化实现：每个用户只有一个正样本，统计该正样本是否在前 K 内。
    """
    assert ranked_indices.shape[0] == len(ground_truth), "Mismatch users."
    hits = 0
    for i, gt in enumerate(ground_truth):
        topk = set(ranked_indices[i, :k].tolist())
        hits += 1 if gt in topk else 0
    return hits / len(ground_truth) if ground_truth else 0.0


def mrr_at_k(ranked_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
    """Compute MRR@K."""
    assert ranked_indices.shape[0] == len(ground_truth)
    rr_sum = 0.0
    for i, gt in enumerate(ground_truth):
        topk = ranked_indices[i, :k]
        try:
            rank = np.where(topk == gt)[0]
            if rank.size > 0:
                rr_sum += 1.0 / (int(rank[0]) + 1)
        except Exception:
            pass
    return rr_sum / len(ground_truth) if ground_truth else 0.0


def ndcg_at_k(ranked_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
    """Compute NDCG@K for single-positive per user."""
    assert ranked_indices.shape[0] == len(ground_truth)
    dcg_sum = 0.0
    idcg = 1.0  # single positive, ideal DCG@K is always 1 at rank 1
    for i, gt in enumerate(ground_truth):
        topk = ranked_indices[i, :k]
        try:
            rank = np.where(topk == gt)[0]
            if rank.size > 0:
                dcg_sum += 1.0 / math.log2(int(rank[0]) + 2)
        except Exception:
            pass
    return dcg_sum / (len(ground_truth) * idcg) if ground_truth else 0.0


@dataclass
class MetricBundle:
    """Container for ranking metrics."""
    recall: Dict[int, float]
    mrr: Dict[int, float]
    ndcg: Dict[int, float]

    def as_text(self) -> str:
        parts = []
        for d, name in [(self.recall, "Recall"), (self.mrr, "MRR"), (self.ndcg, "NDCG")]:
            pretty = ", ".join([f"@{k}:{v:.4f}" for k, v in sorted(d.items())])
            parts.append(f"{name}[{pretty}]")
        return " | ".join(parts)