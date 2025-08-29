# -*- coding: utf-8 -*-
"""Main entry for training/evaluating pointwise ranking on MovieLens 100K.

Features:
- Configurable CLI (see `config.py`)
- loguru logging
- tensorboardX logging
- AMP, early-stopping, checkpoints
- Evaluation with RMSE/MAE/NDCG@K
- Final self-check notes printed
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger

from .config import AppConfig, parse_config
from .data import DataModule
from .model import PointwiseRanker
from .trainer import Trainer
from .utils import ensure_dir, get_device, seed_everything


def main(cfg: AppConfig) -> None:
    """Run the full training-evaluation pipeline."""
    # 配置运行目录、日志
    run_dir = ensure_dir(cfg.run_dir)
    logger.add(str(run_dir / "train.log"), rotation="5 MB", retention=5, enqueue=True)

    # 固定随机种子
    seed_everything(cfg.seed)

    # 设备选择
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    # 数据模块
    dm = DataModule(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        val_ratio=0.1 if not cfg.dry_run else 0.05,
        test_ratio=0.1 if not cfg.dry_run else 0.05,
        random_state=cfg.seed,
    )
    dm.prepare()

    # 模型
    model = PointwiseRanker(
        num_users=dm.info.num_users,
        num_items=dm.info.num_items,
        embed_dim=cfg.embed_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized. Total params: {total_params:,}")

    # 训练器
    trainer = Trainer(
        model=model,
        device=device,
        run_dir=cfg.run_dir,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs if not cfg.dry_run else 2,
        early_stop_patience=cfg.early_stop_patience if not cfg.dry_run else 1,
        amp=cfg.amp,
        ndcg_ks=cfg.ndcg_ks,
    )

    # 训练
    trainer.fit(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        corpus_info_min_rating=dm.info.min_rating,
        corpus_info_max_rating=dm.info.max_rating,
    )

    # 测试评估
    test_res = trainer.evaluate(
        loader=dm.test_dataloader(),
        clamp=(dm.info.min_rating, dm.info.max_rating),
        collect_for_ranking=True,
    )
    logger.info(
        f"[TEST] RMSE={test_res.rmse:.6f}, MAE={test_res.mae:.6f}, "
        f"NDCGs={ {k: round(v, 6) for k, v in test_res.ndcgs.items()} }"
    )

    # 保存最终报告
    report_path = Path(cfg.run_dir) / "final_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rmse": test_res.rmse,
                "mae": test_res.mae,
                "ndcgs": test_res.ndcgs,
                "params": total_params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Final report saved: {report_path}")

    # ===== 关键步骤：正确性与逻辑自检（静态清单） =====
    # 中文说明：以下打印的自检清单用于“关键步骤的正确性复核”，确保训练流程逻辑与边界条件覆盖。
    logger.info(
        "Self-Check: \n"
        "1) Data split is per-user stratified; each user has train/val/test.\n"
        "2) Index bounds checked in model.forward (no negative or out-of-range indices).\n"
        "3) Ratings are clamped to [1,5] at eval; training uses raw targets.\n"
        "4) Early-stopping on val RMSE; checkpoints keep best K.\n"
        "5) AMP enabled if CUDA; GradScaler used.\n"
        "6) Metrics include RMSE/MAE and NDCG@K for ranking diagnostics.\n"
        "7) Random seed fixed; cudnn deterministic.\n"
        "8) Logging to both loguru file and TensorBoard.\n"
        "9) Strict assertions on ratios, shapes, and NaNs to catch boundary issues early.\n"
    )


if __name__ == "__main__":
    config = parse_config()
    main(config)
