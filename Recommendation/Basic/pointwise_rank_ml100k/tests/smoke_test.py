# -*- coding: utf-8 -*-
"""Smoke test for critical shapes & boundary conditions.

Run:
    python -m tests.smoke_test
"""

from __future__ import annotations

import torch

from ranker.data import DataModule
from ranker.model import PointwiseRanker
from ranker.utils import seed_everything


def main() -> None:
    seed_everything(123)

    dm = DataModule(
        data_root="./data",
        batch_size=256,
        num_workers=0,
        pin_memory=False,
        val_ratio=0.05,
        test_ratio=0.05,
        random_state=123,
    )
    dm.prepare()

    model = PointwiseRanker(
        num_users=dm.info.num_users,
        num_items=dm.info.num_items,
        embed_dim=16,
        hidden_dims=[32, 16],
        dropout=0.1,
    )

    # 从训练集抓一小批，检查形状与范围
    loader = dm.train_dataloader()
    users, items, ratings = next(iter(loader))
    assert users.ndim == 1 and items.ndim == 1 and ratings.ndim == 1
    assert (users >= 0).all() and (items >= 0).all()
    assert ratings.min() >= 1.0 and ratings.max() <= 5.0

    with torch.no_grad():
        pred = model(users, items)
        assert pred.shape == ratings.shape, "输出与目标形状不一致"
        # 推理阶段可截断
        pred = pred.clamp(1.0, 5.0)
        assert (pred >= 1.0).all() and (pred <= 5.0).all()

    print("Smoke test passed ✅")


if __name__ == "__main__":
    main()
