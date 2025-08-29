from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F


def listmle_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ListMLE loss (Plackett–Luce) with numerically stable log-sum-exp.

    Args:
        scores: (B, L) predicted scores for each item in the list.
        labels: (B, L) ground-truth ratings (higher means better).
        mask:   (B, L) boolean mask for valid entries.

    Returns:
        Scalar tensor: mean ListMLE loss over batch.

    中文注释：
        - 按标签降序对每个列表进行重排，然后计算 PL 概率的负对数；
        - 使用 log-sum-exp 防止数值溢出；
        - mask 避免 padding 对损失的干扰。
    """
    B, L = scores.shape
    device = scores.device

    # Replace invalid positions by -inf so they won't contribute after sorting.
    neg_inf = torch.tensor(-1e9, device=device)
    masked_scores = torch.where(mask, scores, neg_inf)
    masked_labels = torch.where(mask, labels, neg_inf)

    losses = []
    for b in range(B):
        # Select valid positions
        valid = mask[b].nonzero(as_tuple=False).flatten()
        if len(valid) < 2:
            continue
        s = masked_scores[b, valid]
        y = masked_labels[b, valid]
        # Sort by labels descending (ties are allowed but deterministic by index)
        order = torch.argsort(y, descending=True, stable=True)
        s_sorted = s[order]

        # ListMLE: - sum_i log( exp(s_i) / sum_{j>=i} exp(s_j) )
        # = sum_i ( logsumexp(s_{i:}) - s_i )
        # Compute cumulative logsumexp from tail to head.
        # Efficient trick: running logsumexp backward.
        Lq = len(s_sorted)
        # cumulative logsumexp
        cumsum = torch.empty(Lq, device=device)
        running = torch.tensor(float('-inf'), device=device)
        for i in range(Lq - 1, -1, -1):
            running = torch.logaddexp(running, s_sorted[i])
            cumsum[i] = running
        loss_b = (cumsum - s_sorted).sum()
        losses.append(loss_b)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def pairwise_pl_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    q_idx: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic loss consistent with Plackett–Luce.

    We optimize log σ( sign(y_i - y_j) * (s_i - s_j) ).

    Args:
        scores: (B, L)
        labels: (B, L)
        mask:   (B, L)
        q_idx, i_idx, j_idx: sampled pair indices from `sample_pairwise_from_list`.

    Returns:
        Scalar tensor: mean pairwise loss.

    中文注释：
        - 基于分数差与标签差构建二分类对数似然；
        - 与 PL 模型相容的对偶形式；
        - 仅在标签不相等的成对样本上计算。
    """
    if q_idx.numel() == 0:
        return torch.tensor(0.0, device=scores.device)

    s_i = scores[q_idx, i_idx]
    s_j = scores[q_idx, j_idx]

    y_i = labels[q_idx, i_idx]
    y_j = labels[q_idx, j_idx]

    # Only keep pairs where both are valid
    # (在采样阶段已尽量保证，此处再次稳健过滤)
    valid = (y_i > -1e8) & (y_j > -1e8)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    s_i = s_i[valid]
    s_j = s_j[valid]
    y_i = y_i[valid]
    y_j = y_j[valid]

    y_sign = torch.sign(y_i - y_j)
    # Remove zeros (equal labels) to avoid null gradients
    nz = (y_sign != 0)
    if nz.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    logits = (s_i - s_j)[nz] * y_sign[nz]
    # logistic loss: -log sigma(logits)
    loss = F.softplus(-logits).mean()
    return loss