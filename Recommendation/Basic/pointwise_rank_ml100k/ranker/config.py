# -*- coding: utf-8 -*-
"""Configuration management for Pointwise Ranking.

This module defines a dataclass `AppConfig` and argument parser to
configure the training/evaluation process in a scalable and type-safe way.

The design follows Google style docstrings and PEP 8.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AppConfig:
    """Application configuration.

    Attributes:
        data_root: Directory to store/download MovieLens ml-100k.
        run_dir: Directory to save logs, checkpoints, and TB events.
        seed: Global random seed.
        device: Device string ('cuda' or 'cpu'). If None, auto-detect.
        batch_size: Mini-batch size for training and evaluation.
        epochs: Max training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        embed_dim: Embedding dimension for users and items.
        hidden_dims: Hidden dims of MLP head.
        dropout: Dropout rate for MLP.
        early_stop_patience: Early stopping patience on validation RMSE.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for DataLoader when using CUDA.
        amp: Whether to use mixed precision (torch.cuda.amp).
        checkpoint_best_k: Keep top-k checkpoints by val RMSE.
        ndcg_ks: Compute NDCG@K list on eval.
        dry_run: If True, run short sanity checks (fast dev run).
    """
    data_root: str = "./data"
    run_dir: str = "./runs/exp1"
    seed: int = 42
    device: Optional[str] = None

    batch_size: int = 1024
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-6

    embed_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1

    early_stop_patience: int = 5
    num_workers: int = 2
    pin_memory: bool = True
    amp: bool = True
    checkpoint_best_k: int = 3
    ndcg_ks: List[int] = field(default_factory=lambda: [5, 10])

    dry_run: bool = False


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser("Pointwise Ranking (MovieLens 100K)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--run_dir", type=str, default="./runs/exp1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--checkpoint_best_k", type=int, default=3)
    parser.add_argument("--ndcg_ks", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser


def parse_config() -> AppConfig:
    """Parse command line arguments into AppConfig.

    Returns:
        AppConfig: The application config.
    """
    parser = build_argparser()
    args = parser.parse_args()
    cfg = AppConfig(
        data_root=args.data_root,
        run_dir=args.run_dir,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
        checkpoint_best_k=args.checkpoint_best_k,
        ndcg_ks=args.ndcg_ks,
        dry_run=args.dry_run,
    )
    return cfg