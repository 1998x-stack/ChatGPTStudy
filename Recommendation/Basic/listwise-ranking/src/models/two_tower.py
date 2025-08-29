from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class TwoTowerScorer(nn.Module):
    """Two-tower scorer s(u, i) with MLP head.

    中文注释：
        - 用户塔与物品塔分别嵌入，然后拼接后通过 MLP 输出打分；
        - 支持大规模用户/物品；
        - 生产场景中可导出嵌入用于检索。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score

    def list_scores(self, user_ids: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Compute scores for a batch of users over a padded list of items.

        Args:
            user_ids: (B,)
            items: (B, L)
        Returns:
            scores: (B, L)
        """
        B, L = items.shape
        u = self.user_emb(user_ids).unsqueeze(1).expand(B, L, -1)
        i = self.item_emb(items)
        x = torch.cat([u, i], dim=-1)
        scores = self.mlp(x).squeeze(-1)
        return scores