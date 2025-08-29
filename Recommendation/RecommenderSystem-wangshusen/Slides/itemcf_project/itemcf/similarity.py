from __future__ import annotations

from typing import Tuple

import numpy as np
from loguru import logger
from scipy import sparse


def apply_iuf(csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """Apply inverse user frequency weighting on rows.

    IUF: weight each nonzero by log(1 + n_users / user_interactions)
    This reduces the dominance of heavy users.

    Args:
        csr: User-Item matrix.

    Returns:
        csr_iuf: Weighted matrix in CSR.
    """
    n_users, _ = csr.shape
    # 每个用户的交互数
    row_counts = np.diff(csr.indptr)
    # 防止除零
    row_counts = np.maximum(row_counts, 1)
    weights = np.log1p(n_users / row_counts.astype(np.float64))
    # 将权重应用到每个行的所有非零元素
    data = csr.data.copy().astype(np.float64)
    out_data = data.copy()
    # 快速广播：遍历每个用户行段
    for u in range(n_users):
        start, end = csr.indptr[u], csr.indptr[u + 1]
        if start < end:
            out_data[start:end] = data[start:end] * weights[u]
    csr_iuf = sparse.csr_matrix((out_data, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)
    return csr_iuf


def topk_cosine_similarity(
    csr: sparse.csr_matrix,
    top_k: int,
    sim_shrinkage: float = 1e-12,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Compute item-item TopK cosine similarity from user-item CSR.

    Implementation:
      - Normalize columns (items) by L2 norm.
      - Compute C = X^T @ X on need (sparse-safe via multiplication).
      - Zero-out diagonal and keep per-column TopK.
      - Return as CSR (items x items), and item norms.

    Args:
        csr: User-item matrix (n_users x n_items).
        top_k: Number of top similar items to keep per item.
        sim_shrinkage: Small constant to stabilize denominator.

    Returns:
        sim_csr: CSR of item-item TopK cosine similarity.
        norms: L2 norms of item columns before normalization.

    Raises:
        ValueError: If top_k is invalid or matrix is empty.
    """
    n_users, n_items = csr.shape
    if n_items == 0 or csr.nnz == 0:
        raise ValueError("Empty interaction matrix.")
    if top_k <= 0 or top_k > n_items:
        raise ValueError(f"top_k must be in [1, {n_items}]")

    # 列范数（物品向量 L2）
    col_norms = np.sqrt(csr.power(2).sum(axis=0)).A1  # shape (n_items,)
    col_norms = np.maximum(col_norms, sim_shrinkage)

    # 列归一化：X_norm = X / ||col||
    # 逐列缩放：将稀疏矩阵列除以列范数
    inv_norms = 1.0 / col_norms
    # 利用 CSC 更快地按列缩放
    x_csc = csr.tocsc(copy=True)
    for j in range(n_items):
        start, end = x_csc.indptr[j], x_csc.indptr[j + 1]
        if start < end:
            x_csc.data[start:end] *= inv_norms[j]

    x_norm = x_csc.tocsr()
    # 计算 item-item 相似度近似为 X_norm^T @ X_norm（余弦）
    # 注意：此乘法会得到稀疏，但仍然较大；接下来做 TopK 保留
    logger.info("Multiplying X_norm^T @ X_norm for cosine similarity ...")
    sim = x_norm.T @ x_norm  # (n_items, n_items) 稀疏

    # 去掉自相似对角线
    sim = sim.tolil(copy=True)
    sim.setdiag(0.0)
    sim = sim.tocsr()

    # 保留每列 TopK（按相似度值）
    logger.info("Selecting per-item TopK similar items ...")
    sim = _keep_topk_per_col(sim, top_k=top_k)

    return sim, col_norms


def _keep_topk_per_col(mat: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    """Keep TopK per column for a CSR square matrix.

    Args:
        mat: Square CSR matrix.
        top_k: Number of entries to keep per column.

    Returns:
        CSR matrix with only TopK nonzeros per column retained.
    """
    mat = mat.tocsc(copy=True)
    n = mat.shape[0]
    new_data = []
    new_indices = []
    new_indptr = [0]

    for j in range(n):
        start, end = mat.indptr[j], mat.indptr[j + 1]
        col_data = mat.data[start:end]
        col_rows = mat.indices[start:end]
        if len(col_data) > top_k:
            # 选 TopK 的阈值（避免完全排序开销）
            kth = np.argpartition(-col_data, top_k - 1)[top_k - 1]
            thresh = col_data[kth]
            mask = col_data >= thresh
            col_data = col_data[mask]
            col_rows = col_rows[mask]
            # 再做精排序，保证降序
            order = np.argsort(-col_data)
            col_data = col_data[order]
            col_rows = col_rows[order]
        elif len(col_data) > 0:
            order = np.argsort(-col_data)
            col_data = col_data[order]
            col_rows = col_rows[order]

        new_data.append(col_data)
        new_indices.append(col_rows)
        new_indptr.append(new_indptr[-1] + len(col_data))

    data = np.concatenate(new_data) if new_data else np.array([], dtype=np.float64)
    indices = np.concatenate(new_indices) if new_indices else np.array([], dtype=np.int32)
    indptr = np.array(new_indptr, dtype=np.int32)

    out = sparse.csc_matrix((data, indices, indptr), shape=mat.shape)
    return out.tocsr()