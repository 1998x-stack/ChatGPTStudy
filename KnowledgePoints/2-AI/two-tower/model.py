# -*- coding: utf-8 -*-
"""Two-Tower PyTorch model with InfoNCE loss and in-batch negatives."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class MLPEncoder(nn.Module):
    """Simple MLP encoder for tower.

    Args:
        input_dim: Input dimension (embedding dim of ID).
        hidden_dims: Sequence of hidden sizes.
        output_dim: Output embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class TwoTowerModel(nn.Module):
    """Two-Tower model with user/item embeddings and MLP encoders.

    中文说明：
        - 本模型使用 ID Embedding + MLP 编码器，将用户与物品映射到同一向量空间
        - 支持 L2 归一化与 In-batch Negatives 的 InfoNCE 训练
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.user_vocab_size <= 0 or cfg.item_vocab_size <= 0:
            raise ValueError("Vocab sizes must be set > 0 before model creation.")
        self.cfg = cfg

        # Embedding layers
        self.user_id_emb = nn.Embedding(
            num_embeddings=cfg.user_vocab_size,
            embedding_dim=cfg.embedding_dim,
            padding_idx=cfg.user_pad_idx,
        )
        self.item_id_emb = nn.Embedding(
            num_embeddings=cfg.item_vocab_size,
            embedding_dim=cfg.embedding_dim,
            padding_idx=cfg.item_pad_idx,
        )

        # Encoders
        self.user_mlp = MLPEncoder(
            input_dim=cfg.embedding_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
        )
        if cfg.share_item_user_mlp:
            self.item_mlp = self.user_mlp
        else:
            self.item_mlp = MLPEncoder(
                input_dim=cfg.embedding_dim,
                hidden_dims=cfg.hidden_dims,
                output_dim=cfg.embedding_dim,
                dropout=cfg.dropout,
            )

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with a robust scheme."""
        nn.init.normal_(self.user_id_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_id_emb.weight, mean=0.0, std=0.02)
        if self.user_id_emb.padding_idx is not None:
            with torch.no_grad():
                self.user_id_emb.weight[self.user_id_emb.padding_idx].zero_()
        if self.item_id_emb.padding_idx is not None:
            with torch.no_grad():
                self.item_id_emb.weight[self.item_id_emb.padding_idx].zero_()
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_user(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Encode user indices -> user vector."""
        u = self.user_id_emb(user_idx)  # [B, D]
        u = self.user_mlp(u)            # [B, D]
        if self.cfg.l2norm:
            u = F.normalize(u, p=2, dim=-1)
        return u

    def encode_item(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Encode item indices -> item vector."""
        v = self.item_id_emb(item_idx)  # [B or N, D]
        v = self.item_mlp(v)
        if self.cfg.l2norm:
            v = F.normalize(v, p=2, dim=-1)
        return v

    def compute_logits(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute similarity logits via dot product.

        Args:
            u: [B, D] user vectors.
            v: [B, D] or [N, D] item vectors.

        Returns:
            Logits matrix [B, N] where N=B if in-batch, else arbitrary.
        """
        # [B, D] x [D, N] -> [B, N]
        return torch.matmul(u, v.t())

    def info_nce_loss(
        self,
        batch: Dict[str, torch.Tensor],
        temperature: float,
        in_batch_negatives: bool = True,
    ) -> torch.Tensor:
        """Compute InfoNCE loss with optional in-batch negatives.

        Args:
            batch: Dict with keys user_idx [B], pos_item_idx [B], neg_item_idx [B, Nneg]
            temperature: Temperature > 0.
            in_batch_negatives: If True, use other positives in batch as negatives.

        Returns:
            Scalar loss tensor.
        """
        user_idx: torch.Tensor = batch["user_idx"]
        pos_item_idx: torch.Tensor = batch["pos_item_idx"]
        neg_item_idx: torch.Tensor = batch["neg_item_idx"]  # [B, Nneg] or [B, 0]

        u = self.encode_user(user_idx)  # [B, D]
        v_pos = self.encode_item(pos_item_idx)  # [B, D]

        # 构造候选集合
        if in_batch_negatives:
            # 使用 batch 内所有正样本作为候选（包含自身正样本）
            v_cand = v_pos  # [B, D]
            logits = self.compute_logits(u, v_cand) / temperature  # [B, B]
            labels = torch.arange(u.size(0), device=u.device, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
        else:
            # 使用显式负样本 + 正样本拼接
            if neg_item_idx.numel() == 0:
                raise ValueError("Explicit negatives requested but none provided.")
            B = u.size(0)
            Nneg = neg_item_idx.size(1)
            v_neg = self.encode_item(neg_item_idx.view(-1))  # [B*Nneg, D]
            v_neg = v_neg.view(B, Nneg, -1)                  # [B, Nneg, D]
            # v_cand = [pos, negs...] -> [B, 1+Nneg, D]
            v_cand = torch.cat([v_pos.unsqueeze(1), v_neg], dim=1)
            # [B, D] x [B, D]ᵀ 按行算：我们把每行u与其对应候选做逐行点积
            # 展平计算：
            u_expanded = u.unsqueeze(1)                      # [B, 1, D]
            logits = (u_expanded * v_cand).sum(dim=-1)       # [B, 1+Nneg]
            logits = logits / temperature
            labels = torch.zeros(B, dtype=torch.long, device=u.device)  # 正样本在索引0
            loss = F.cross_entropy(logits, labels)

        return loss