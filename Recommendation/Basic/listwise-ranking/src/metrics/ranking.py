from __future__ import annotations
from typing import Iterable, Tuple
import torch


def dcg_at_k(rel: torch.Tensor, k: int) -> torch.Tensor:
    idxs = torch.arange(rel.shape[-1], device=rel.device)[:k]
    denom = torch.log2(idxs.to(torch.float32) + 2.0)
    gains = (2.0 ** rel[..., :k] - 1.0)
    return (gains / denom).sum(dim=-1)


def ndcg_at_k(labels: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    """Compute NDCG@k for a batch of lists.

    中文注释：
        - 先按预测分数排序得到 DCG；
        - 再按真实标签排序得到 IDCG；
        - 防止除零，用 eps 保证稳定性。
    """
    B, L = scores.shape
    # Mask invalid
    neg_inf = torch.tensor(-1e9, device=scores.device)
    masked_scores = torch.where(mask, scores, neg_inf)
    masked_labels = torch.where(mask, labels, torch.zeros_like(labels))

    # Sort by predicted scores
    _, idx_pred = torch.sort(masked_scores, dim=1, descending=True)
    rel_pred = torch.gather(masked_labels, 1, idx_pred)
    dcg = dcg_at_k(rel_pred, k)

    # Ideal DCG (sort by labels)
    _, idx_true = torch.sort(masked_labels, dim=1, descending=True)
    rel_true = torch.gather(masked_labels, 1, idx_true)
    idcg = dcg_at_k(rel_true, k)

    eps = 1e-9
    ndcg = (dcg / (idcg + eps)).mean().item()
    return float(ndcg)


def mrr(labels: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> float:
    neg_inf = torch.tensor(-1e9, device=scores.device)
    masked_scores = torch.where(mask, scores, neg_inf)
    masked_labels = torch.where(mask, labels, torch.zeros_like(labels))

    _, idx_pred = torch.sort(masked_scores, dim=1, descending=True)
    rel_pred = torch.gather(masked_labels, 1, idx_pred)

    # First relevant position (label>0)
    rel_bin = (rel_pred > 0).to(torch.float32)
    # positions start at 1
    positions = torch.arange(1, rel_bin.shape[1] + 1, device=scores.device).to(torch.float32)
    # Reciprocal only for the first relevant item per row
    # Trick: cumulative sum and pick first position
    cumsum = torch.cumsum(rel_bin, dim=1)
    first_pos = (cumsum == 1).to(torch.float32) * positions
    rr = (first_pos * rel_bin / positions).max(dim=1).values
    return rr.mean().item()