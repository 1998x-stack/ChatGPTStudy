from __future__ import annotations

import torch
from typing import Tuple


def _pairwise_delta_ndcg(
    rels: torch.Tensor,
    scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute |ΔNDCG| matrix for all pairs under current predicted ranks.

    Args:
        rels: Tensor of shape [n] with relevance labels (float or int).
        scores: Tensor of shape [n] with current predicted scores.
        k: Cutoff K for NDCG normalization (IDCG uses all reals but cutoff applies to DCG deltas).

    Returns:
        Tensor [n, n] with absolute NDCG change if swapping i and j, zeros on diagonal.
    """
    # 根据当前 scores 排序得到当前排名位置
    n = rels.numel()
    device = rels.device
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks[order] = torch.arange(n, device=device, dtype=torch.long)  # 0-based ranks

    # 增益与折扣（位置 0 对应 rank=1）
    gains = torch.pow(2.0, rels.to(scores.dtype)) - 1.0
    discounts = torch.log2((ranks + 2).to(scores.dtype))  # log2(i+2)

    # 计算 IDCG（理想 DCG 用真实相关性降序）
    ideal_order = torch.argsort(rels, descending=True)
    ideal_ranks = torch.arange(n, device=device, dtype=torch.long)
    ideal_discounts = torch.log2((ideal_ranks + 2).to(scores.dtype))
    ideal_gains = torch.pow(2.0, rels[ideal_order].to(scores.dtype)) - 1.0
    # 注意：IDCG 在 @k 场景下，通常只累计前 k 位
    idcg_mask = (ideal_ranks < k).to(scores.dtype)
    idcg = torch.sum(ideal_gains * (1.0 / ideal_discounts) * idcg_mask)
    idcg = torch.clamp(idcg, min=1e-8)  # 防止除零

    # 对任意 i, j 交换位置时 DCG 的改变： (gain_i - gain_j) * (1/discount_i - 1/discount_j)
    # 但仅当位置在前 k 位时才对 NDCG@k 起作用，因此对折扣项按 rank<k 加掩码
    dc_mask = (ranks < k).to(scores.dtype)
    inv_discount = (1.0 / discounts) * dc_mask  # [n]

    gi = gains.unsqueeze(1)  # [n,1]
    gj = gains.unsqueeze(0)  # [1,n]
    inv_di = inv_discount.unsqueeze(1)  # [n,1]
    inv_dj = inv_discount.unsqueeze(0)  # [1,n]

    delta_dcg = (gi - gj) * (inv_di - inv_dj)  # [n,n]
    delta_ndcg = torch.abs(delta_dcg) / idcg
    # 对角线为 0（与自身交换无意义）
    delta_ndcg.fill_diagonal_(0.0)
    return delta_ndcg


def lambdarank_pairwise_loss(
    rels: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    k: int,
) -> torch.Tensor:
    """LambdaRank pairwise logistic loss weighted by |ΔNDCG|.

    Args:
        rels: Relevance tensor [n].
        scores: Score tensor [n].
        sigma: Sigmoid slope for pairwise logistic.
        k: Cutoff for NDCG weight.

    Returns:
        Scalar loss tensor.
    """
    # 计算所有成对的 |ΔNDCG|
    delta_ndcg = _pairwise_delta_ndcg(rels, scores, k=k)  # [n, n]

    # 仅保留 rel_i > rel_j 的上三角配对，避免重复计算
    rels_i = rels.unsqueeze(1)
    rels_j = rels.unsqueeze(0)
    pair_mask = (rels_i > rels_j).to(scores.dtype)  # [n, n]，只在 i 的相关性大于 j 时配对

    # 差值与加权
    score_diff = (scores.unsqueeze(1) - scores.unsqueeze(0))  # s_i - s_j
    logit = sigma * score_diff  # [n, n]
    # logistic pairwise loss: log(1 + exp(-logit)) 对 rel_i > rel_j 的对
    pairwise_loss = torch.nn.functional.softplus(-logit)  # log(1 + exp(-x))

    # 仅计入有效 pair，并用 |ΔNDCG| 加权
    weighted = pairwise_loss * delta_ndcg * pair_mask

    # 归一化，防止不同 n 的样本权重不一致（+eps 防止除零）
    denom = torch.clamp(pair_mask.sum(), min=1.0)
    loss = weighted.sum() / denom
    return loss