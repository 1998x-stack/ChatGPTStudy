from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class MFScoringModel(nn.Module):
    """A simple user-item embedding scoring model for ranking.

    The score is computed as:
        score(u, i) = <U[u], V[i]> + b_u + b_i

    This is efficient and extensible (you may add MLP or features later).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb_dim: int = 64,
        item_emb_dim: int = 64,
    ) -> None:
        """Initialize the model.

        Args:
            num_users: Number of users.
            num_items: Number of items.
            user_emb_dim: User embedding dimension.
            item_emb_dim: Item embedding dimension.
        """
        super().__init__()
        # 用户与物品嵌入，含偏置项
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # 参数初始化（更稳定的训练）
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute scores for (user_ids, item_ids).

        Args:
            user_ids: Tensor of user indices [batch,].
            item_ids: Tensor of item indices [batch,].

        Returns:
            Scores tensor [batch,].
        """
        u = self.user_emb(user_ids)   # [B, D]
        v = self.item_emb(item_ids)   # [B, D]
        bu = self.user_bias(user_ids) # [B, 1]
        bi = self.item_bias(item_ids) # [B, 1]
        scores = (u * v).sum(dim=1, keepdim=True) + bu + bi  # [B, 1]
        return scores.squeeze(1)  # [B]
