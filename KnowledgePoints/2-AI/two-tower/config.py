# -*- coding: utf-8 -*-
"""Configuration objects and command-line parsing for Two-Tower training.

This module defines dataclasses and helpers for experiment configuration,
with strong validation and sensible defaults suitable for industrial use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class OptimConfig:
    """Optimizer/training hyperparameters.

    Attributes:
        lr: Learning rate for the optimizer.
        weight_decay: L2 regularization coefficient.
        betas: Adam betas.
        eps: Adam epsilon for numerical stability.
        grad_clip_norm: Max norm for gradient clipping; 0 to disable.
        epochs: Max training epochs.
        early_stop_patience: Early stopping patience; 0 to disable.
        amp: Use mixed precision (PyTorch AMP) if True.
        temperature: InfoNCE temperature (>0).
    """
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    epochs: int = 10
    early_stop_patience: int = 3
    amp: bool = True
    temperature: float = 0.07


@dataclass
class ModelConfig:
    """Two-Tower model hyperparameters.

    Attributes:
        embedding_dim: The output embedding dimension of each tower.
        hidden_dims: Hidden sizes for MLP encoders per tower.
        dropout: Dropout rate in MLPs (0-1).
        l2norm: Whether to L2-normalize final embeddings.
        share_item_user_mlp: If True, share the same MLP architecture object
            for both towers (only advisable when modalities are similar).
        user_vocab_size: Size of user ID vocabulary (set at runtime).
        item_vocab_size: Size of item ID vocabulary (set at runtime).
        user_pad_idx: Padding index for user id embedding.
        item_pad_idx: Padding index for item id embedding.
    """
    embedding_dim: int = 128
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    l2norm: bool = True
    share_item_user_mlp: bool = False
    user_vocab_size: int = -1
    item_vocab_size: int = -1
    user_pad_idx: int = 0
    item_pad_idx: int = 0


@dataclass
class DataConfig:
    """Data input and sampling configuration.

    Attributes:
        train_csv: Path to training interactions CSV (user_id,item_id,label,timestamp).
        valid_csv: Path to validation CSV.
        test_csv: Path to test CSV.
        delimiter: CSV delimiter.
        has_header: If True, skip the first header row.
        num_negatives: Number of random negatives per positive in training.
        batch_size: Per-GPU batch size.
        num_workers: Dataloader workers.
        pin_memory: Whether to pin memory for faster host->GPU transfer.
        in_batch_negatives: Use in-batch negatives (InfoNCE in batch).
        strict_id: If True, invalid/unknown id in val/test will error; else map to UNK.
        max_users: Optional cap for users loaded (for prototyping).
        max_items: Optional cap for items loaded (for prototyping).
        seed: Global random seed.
    """
    train_csv: Path = Path("data/train.csv")
    valid_csv: Path = Path("data/valid.csv")
    test_csv: Path = Path("data/test.csv")
    delimiter: str = ","
    has_header: bool = True
    num_negatives: int = 4
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    in_batch_negatives: bool = True
    strict_id: bool = False
    max_users: Optional[int] = None
    max_items: Optional[int] = None
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        k_values: List of K values for Recall@K/MRR@K/NDCG@K.
        full_ranking: If True, brute-force against all items; expensive but exact.
    """
    k_values: Tuple[int, ...] = (10, 50, 100)
    full_ranking: bool = True  # ANN can be plugged later


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration with all grouped sub-configs."""
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    eval: EvalConfig
    output_dir: Path = Path("outputs")

    def validate(self) -> None:
        """Validate configuration values thoroughly.

        Raises:
            ValueError: When any configuration item is invalid.
        """
        if self.optim.lr <= 0:
            raise ValueError("Learning rate must be > 0.")
        if not (0.0 <= self.model.dropout < 1.0):
            raise ValueError("Dropout must be in [0,1).")
        if self.optim.temperature <= 0:
            raise ValueError("InfoNCE temperature must be > 0.")
        if self.data.batch_size <= 0:
            raise ValueError("Batch size must be > 0.")
        if self.data.num_negatives < 0:
            raise ValueError("num_negatives must be >= 0.")
        if any(k <= 0 for k in self.eval.k_values):
            raise ValueError("All k_values must be positive integers.")
        # 输出路径校验（如不存在则创建）
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Return configuration as a flat dictionary."""
        d = asdict(self)
        d["data"]["train_csv"] = str(self.data.train_csv)
        d["data"]["valid_csv"] = str(self.data.valid_csv)
        d["data"]["test_csv"] = str(self.data.test_csv)
        d["output_dir"] = str(self.output_dir)
        return d


def build_config_from_args() -> ExperimentConfig:
    """Parse CLI arguments and build the experiment configuration."""
    parser = argparse.ArgumentParser(description="Two-Tower Training Pipeline")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--valid_csv", type=str, default="data/valid.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--in_batch_negatives", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    data = DataConfig(
        train_csv=Path(args.train_csv),
        valid_csv=Path(args.valid_csv),
        test_csv=Path(args.test_csv),
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        in_batch_negatives=args.in_batch_negatives or True,  # 默认开启
    )
    model = ModelConfig(
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    optim = OptimConfig(
        lr=args.lr,
        epochs=args.epochs,
        amp=not args.no_amp,
    )
    eval_cfg = EvalConfig()

    cfg = ExperimentConfig(data=data, model=model, optim=optim, eval=eval_cfg)
    cfg.validate()
    return cfg