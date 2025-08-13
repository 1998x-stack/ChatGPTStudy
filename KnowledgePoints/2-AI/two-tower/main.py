# -*- coding: utf-8 -*-
"""Main entry: build data/model, train and test Two-Tower pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import torch

from config import ExperimentConfig, ModelConfig, OptimConfig, DataConfig, EvalConfig, build_config_from_args
from data import build_dataloaders
from model import TwoTowerModel
from trainer import Trainer
from utils import MetricBundle, create_logger, set_global_seed


def _infer_and_fill_vocab_sizes(cfg: ExperimentConfig, user_vocab_size: int, item_vocab_size: int) -> None:
    """Fill model config vocab sizes from data stage and validate."""
    cfg.model.user_vocab_size = user_vocab_size
    cfg.model.item_vocab_size = item_vocab_size
    # pad/unk already fixed at 0/1 by Vocab class order
    cfg.model.user_pad_idx = 0
    cfg.model.item_pad_idx = 0
    # re-validate critical fields
    if cfg.model.embedding_dim <= 0:
        raise ValueError("embedding_dim must be > 0.")


def main() -> None:
    """Run training + evaluation pipeline with strong logging."""
    logger = create_logger()
    cfg = build_config_from_args()
    set_global_seed(cfg.data.seed)

    logger.info(f"Experiment config:\n{json.dumps(cfg.to_dict(), indent=2)}")

    # 1) Data
    train_loader, valid_loader, test_loader, user_vocab, item_vocab = build_dataloaders(cfg.data)
    _infer_and_fill_vocab_sizes(cfg, len(user_vocab), len(item_vocab))

    # 2) Model
    model = TwoTowerModel(cfg.model)
    trainer = Trainer(cfg, model)

    # 3) Training Loop
    logger.info("==== Start Training ====")
    for epoch in range(1, cfg.optim.epochs + 1):
        trainer.train_epoch(train_loader, epoch)
        # 预计算全量物品向量
        all_item_emb = trainer.precompute_all_item_embeddings(cfg.model.item_vocab_size)
        # 在验证集评估
        metrics = trainer.evaluate(valid_loader, all_item_emb)
        logger.info(f"[Epoch {epoch}] Valid: {metrics.as_text()}")
        # 早停监控（以 Recall@50 为例）
        stop = trainer.maybe_early_stop(metrics.recall.get(50, 0.0))
        if stop:
            logger.info("Early stopping triggered.")
            break

    # 4) 最终测试
    logger.info("==== Final Test ====")
    all_item_emb = trainer.precompute_all_item_embeddings(cfg.model.item_vocab_size)
    test_metrics = trainer.evaluate(test_loader, all_item_emb)
    logger.info(f"[Test] {test_metrics.as_text()}")

    # 5) 保存模型
    save_path = Path(cfg.output_dir) / "two_tower.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg.to_dict()}, save_path)
    logger.info(f"Model saved to: {save_path.resolve()}")

    # 关键信息打印
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 兜底：明确报错并退出非零码，便于CI监控
        create_logger().exception(f"Fatal error: {e}")
        sys.exit(2)