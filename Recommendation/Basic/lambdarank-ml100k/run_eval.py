from __future__ import annotations

import argparse

import torch
from loguru import logger

from lambdarank.config import EvalConfig
from lambdarank.data import ML100KDataModule, create_dataloaders
from lambdarank.evaluation import Evaluator
from lambdarank.model import MFScoringModel
from lambdarank.utils import set_seed


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate LambdaRank model")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--ckpt_path", type=str, default="./checkpoints/best.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ndcg_k", type=int, default=10)
    p.add_argument("--max_items_per_user", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return EvalConfig(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        device=args.device,
        ndcg_k=args.ndcg_k,
        max_items_per_user=args.max_items_per_user,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(42)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
        logger.warning("CUDA not available. Using CPU.")

    dm = ML100KDataModule(
        data_dir=cfg.data_dir,
        max_items_per_user=cfg.max_items_per_user,
        test_size=0.1,
        val_size=0.1,
        seed=42,
    )
    dm.setup()

    # 构建同维度模型并加载权重
    model = MFScoringModel(
        num_users=dm.num_users,
        num_items=dm.num_items,
        user_emb_dim=64,
        item_emb_dim=64,
    ).to(cfg.device)

    state = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {cfg.ckpt_path}")

    evaluator = Evaluator(model=model, cfg=cfg)
    results = evaluator.evaluate_queries(dm.test_queries)
    for k, v in results.items():
        logger.info(f"{k}: {v:.5f}")


if __name__ == "__main__":
    main()