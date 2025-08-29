# -*- coding: utf-8 -*-
"""Pointwise Ranking Model (Rating Regression).

Architecture:
- User embedding + Item embedding
- Optional MLP head on concatenated embeddings
- Output a single scalar rating prediction

We clamp predictions to rating range [1, 5] at eval time for sensible metrics.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from loguru import logger


class PointwiseRanker(nn.Module):
    """Pointwise rating regression model with embeddings and MLP head."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
    ) -> None:
        """Initialize model.

            Args:
                num_users: Number of users.
                num_items: Number of items.
                embed_dim: Embedding dimension.
                hidden_dims: MLP hidden dims (e.g., [128, 64]).
                dropout: Dropout rate between MLP layers.
        """
        super().__init__()
        # 嵌入层：用户 & 物品
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        layers: List[nn.Module] = []
        in_dim = embed_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters with sensible initializations."""
        # 初始化嵌入，提升收敛稳定性
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        # 初始化线性层
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            users: Tensor of user indices (B,).
            items: Tensor of item indices (B,).

        Returns:
            torch.Tensor: Predicted ratings (B, 1).
        """
        # 边界检查 —— 防止索引越界（生产中经常遇到脏数据）
        assert users.min() >= 0 and items.min() >= 0, "索引出现负数"
        assert users.max() < self.user_emb.num_embeddings, "用户索引越界"
        assert items.max() < self.item_emb.num_embeddings, "物品索引越界"

        u = self.user_emb(users)   # (B, E)
        v = self.item_emb(items)   # (B, E)
        x = torch.cat([u, v], dim=-1)  # (B, 2E)
        y = self.mlp(x)            # (B, 1)
        return y.squeeze(-1)       # (B,)

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss for rating regression."""
        return torch.mean((pred - target) ** 2)