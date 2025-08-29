from __future__ import annotations

import argparse
import os

import torch
from loguru import logger

from lambdarank.config import TrainConfig
from lambdarank.data import ML100KDataModule, create_dataloaders
from lambdarank.model import MFScoringModel
from lambdarank.trainer import Trainer
from lambdarank.utils import ensure_dir, set_seed


def parse_args() -> TrainConfig:
    """Parse CLI args to TrainConfig."""
    p = argparse.ArgumentParser(description="Train LambdaRank on MovieLens-100k")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--log_dir", type=str, default="./runs")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--user_emb_dim", type=int, default=64)
    p.add_argument("--item_emb_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--batch_users", type=int, default=64)
    p.add_argument("--max_items_per_user", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--ndcg_k", type=int, default=10)
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        device=args.device,
        user_emb_dim=args.user_emb_dim,
        item_emb_dim=args.item_emb_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_users=args.batch_users,
        max_items_per_user=args.max_items_per_user,
        epochs=args.epochs,
        patience=args.patience,
        clip_grad_norm=args.clip_grad_norm,
        ndcg_k=args.ndcg_k,
        val_interval=args.val_interval,
        sigma=args.sigma,
        save_best=args.save_best,
        quiet=args.quiet,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    ensure_dir(cfg.ckpt_dir)
    ensure_dir(cfg.log_dir)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        cfg.device = "cpu"

    # 数据模块：加载/下载 + 构造 per-user 查询
    dm = ML100KDataModule(
        data_dir=cfg.data_dir,
        max_items_per_user=cfg.max_items_per_user,
        test_size=0.1,
        val_size=0.1,
        seed=cfg.seed,
    )
    dm.setup()

    train_loader, val_loader, _ = create_dataloaders(dm, batch_users=cfg.batch_users, num_workers=0)

    # 模型
    model = MFScoringModel(
        num_users=dm.num_users,
        num_items=dm.num_items,
        user_emb_dim=cfg.user_emb_dim,
        item_emb_dim=cfg.item_emb_dim,
    )

    # 训练器
    trainer = Trainer(model=model, config=cfg)
    trainer.train(train_loader=train_loader, val_loader=val_loader)

    logger.info("Training finished.")


if __name__ == "__main__":
    main()