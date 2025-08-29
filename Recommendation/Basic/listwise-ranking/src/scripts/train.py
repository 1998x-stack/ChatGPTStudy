from __future__ import annotations
import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from loguru import logger

from ..config import TrainConfig
from ..data.movielens import MovieLens100K, ListwiseDataset
from ..data.samplers import collate_listwise
from ..models.two_tower import TwoTowerScorer
from ..engine.trainer import Trainer
from ..utils.seeding import set_global_seed
from ..utils.tb import create_writer


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument('--project_root', type=str, required=True)
    p.add_argument('--loss', type=str, default='hybrid', choices=['listmle', 'pairwise_pl', 'hybrid'])
    p.add_argument('--hybrid_alpha', type=float, default=0.7)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--mlp_hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--eval_interval', type=int, default=400)
    p.add_argument('--early_stop_patience', type=int, default=8)
    p.add_argument('--max_listsize', type=int, default=200)
    p.add_argument('--pairwise_per_query', type=int, default=32)
    args = p.parse_args()

    cfg = TrainConfig(
        project_root=args.project_root,
        data_dir=None,
        loss=args.loss,
        hybrid_alpha=args.hybrid_alpha,
        embedding_dim=args.embedding_dim,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        max_listsize=args.max_listsize,
        pairwise_per_query=args.pairwise_per_query,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    set_global_seed(cfg.seed)
    logger.add(os.path.join(cfg.project_root, 'logs', 'train.log'))

    # Data
    ml = MovieLens100K(cfg.project_root)
    ratings, user2idx, item2idx = ml.prepare()

    train_ds = ListwiseDataset(ratings, split='train', max_listsize=cfg.max_listsize)
    val_ds = ListwiseDataset(ratings, split='val', max_listsize=cfg.max_listsize)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_listwise, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_listwise, pin_memory=True)

    # Model
    num_users = len(user2idx)
    num_items = len(item2idx)
    model = TwoTowerScorer(num_users, num_items, cfg.embedding_dim, cfg.mlp_hidden, cfg.dropout)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # TB Writer
    writer = create_writer(cfg.project_root, run_name=f"{cfg.loss}_ed{cfg.embedding_dim}")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        writer=writer,
        device=cfg.device,
        loss_type=cfg.loss,
        hybrid_alpha=cfg.hybrid_alpha,
        grad_clip=5.0,
        eval_interval=cfg.eval_interval,
        early_stop_patience=cfg.early_stop_patience,
        topk=cfg.topk,
        pairwise_per_query=cfg.pairwise_per_query,
    )

    # Train
    result = trainer.train_epochs(train_loader, val_loader, cfg.epochs)
    logger.info("Training finished. Best NDCG@{} = {}", cfg.topk[0], result['best_ndcg'])


if __name__ == '__main__':
    main()