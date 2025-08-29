from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training configuration for LambdaRank.

    Attributes:
        data_dir: Directory to store/download the MovieLens-100k dataset.
        log_dir: Directory for TensorBoard logs.
        ckpt_dir: Directory to save model checkpoints.
        seed: Global random seed for reproducibility.
        device: Torch device string, e.g. "cuda" or "cpu".
        user_emb_dim: User embedding dimension.
        item_emb_dim: Item embedding dimension.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        batch_users: Number of users per training batch (list-wise per user).
        max_items_per_user: Max items per user per step to cap O(n^2) pairs.
        epochs: Number of training epochs.
        patience: Early stopping patience (epochs).
        clip_grad_norm: Gradient clipping max norm (0 to disable).
        ndcg_k: K for NDCG@K metric.
        val_interval: Validate every N steps (batches).
        sigma: Sigma coefficient in pairwise logistic for LambdaRank.
        save_best: Whether to save best checkpoint by validation NDCG.
        quiet: Suppress non-critical logs if True.
    """
    data_dir: str = "./data"
    log_dir: str = "./runs"
    ckpt_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"
    user_emb_dim: int = 64
    item_emb_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_users: int = 64
    max_items_per_user: int = 50
    epochs: int = 30
    patience: int = 5
    clip_grad_norm: float = 5.0
    ndcg_k: int = 10
    val_interval: int = 200
    sigma: float = 1.0
    save_best: bool = True
    quiet: bool = False


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        data_dir: Dataset directory.
        ckpt_path: Model checkpoint path to load.
        device: Torch device string.
        ndcg_k: K for NDCG@K metric.
        max_items_per_user: Limit items per user during eval for determinism.
    """
    data_dir: str = "./data"
    ckpt_path: str = "./checkpoints/best.pt"
    device: str = "cuda"
    ndcg_k: int = 10
    max_items_per_user: int = 200