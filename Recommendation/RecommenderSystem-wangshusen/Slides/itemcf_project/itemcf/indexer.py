from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from scipy import sparse


@dataclass(frozen=True)
class UserLastNIndex:
    """User → last-N interacted items with weights (like)."""
    last_items: Dict[int, List[Tuple[int, float]]]


@dataclass(frozen=True)
class ItemTopKIndex:
    """Item → top-K similar items with similarity values."""
    topk_items: Dict[int, List[Tuple[int, float]]]


def build_user_lastn_index(
    train_csr: sparse.csr_matrix,
    last_n: int,
) -> UserLastNIndex:
    """Build User→Item last-N index from CSR matrix.

    Strategy: Use input order in CSR indices is arbitrary; we rely on interaction
    recency via timestamps externally if available. Here we use "highest weights" fallback.

    Args:
        train_csr: CSR of user-item interactions (weights as 'like').
        last_n: Keep N items per user.

    Returns:
        UserLastNIndex.

    Raises:
        ValueError: If inputs invalid.
    """
    n_users = train_csr.shape[0]
    if last_n <= 0:
        raise ValueError("last_n must be positive.")

    last_idx: Dict[int, List[Tuple[int, float]]] = {}
    for u in range(n_users):
        start, end = train_csr.indptr[u], train_csr.indptr[u + 1]
        cols = train_csr.indices[start:end]
        vals = train_csr.data[start:end]
        if len(cols) == 0:
            last_idx[u] = []
            continue
        # 按权重从大到小取前 last_n
        order = np.argsort(-vals)
        top = order[:last_n]
        items = [(int(cols[i]), float(vals[i])) for i in top]
        last_idx[u] = items
    logger.info("User→lastN index built for {} users.", n_users)
    return UserLastNIndex(last_items=last_idx)


def build_item_topk_index(
    sim_csr: sparse.csr_matrix,
    top_k: int,
) -> ItemTopKIndex:
    """Build Item→TopK similar items index from similarity CSR (items x items)."""
    n_items = sim_csr.shape[0]
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    topk_map: Dict[int, List[Tuple[int, float]]] = {}

    sim_csr = sim_csr.tocsr()
    for j in range(n_items):
        start, end = sim_csr.indptr[j], sim_csr.indptr[j + 1]
        nbrs = sim_csr.indices[start:end]
        sims = sim_csr.data[start:end]
        # 已按列 TopK，并在 similarity.py 中做了降序
        pairs = [(int(nbrs[k]), float(sims[k])) for k in range(len(nbrs))]
        topk_map[j] = pairs[:top_k]
    logger.info("Item→TopK index built for {} items.", n_items)
    return ItemTopKIndex(topk_items=topk_map)