from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        project_root: Project root folder for data and runs.
        data_dir: Optional explicit data cache dir; defaults to `<project_root>/data`.
        loss: Loss type: 'listmle', 'pairwise_pl', or 'hybrid'.
        hybrid_alpha: Weight for listmle part when loss == 'hybrid'.
        embedding_dim: Embedding dimension for users/items.
        mlp_hidden: Hidden dim for scorer MLP.
        dropout: Dropout rate in MLP.
        epochs: Number of training epochs.
        batch_size: Number of (query) users per batch for listwise; for pairwise, number of pairs per step.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        seed: RNG seed for reproducibility.
        device: 'cpu' or 'cuda'.
        num_workers: DataLoader workers.
        log_interval: Steps between logs.
        eval_interval: Steps between eval.
        early_stop_patience: Early stop patience (eval steps).
        max_listsize: Optional cap on per-user list length for training speed.
        pairwise_per_query: #pairs sampled per query for pairwise objective.
        topk: List of K cutoffs for metrics reporting.
    """

    project_root: str
    data_dir: Optional[str] = None

    loss: Literal['listmle', 'pairwise_pl', 'hybrid'] = 'hybrid'
    hybrid_alpha: float = 0.7

    embedding_dim: int = 64
    mlp_hidden: int = 128
    dropout: float = 0.1

    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 2

    log_interval: int = 50
    eval_interval: int = 400
    early_stop_patience: int = 8

    max_listsize: Optional[int] = 200
    pairwise_per_query: int = 32

    topk: tuple[int, ...] = (5, 10, 20)
