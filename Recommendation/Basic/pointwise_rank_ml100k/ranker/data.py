# -*- coding: utf-8 -*-
"""Data pipeline for MovieLens 100K (pointwise).

Key features:
- Auto-download if not present.
- Map external IDs to contiguous user/item indices with OOV handling.
- Stratified splits per user to preserve per-user distributions.
- Robust boundary checks (min/max item/user, rating range).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from .utils import download_ml100k_if_needed, ensure_dir


@dataclass(frozen=True)
class CorpusInfo:
    """Holds core info of the corpus (vocab sizes and stats)."""
    num_users: int
    num_items: int
    min_rating: float
    max_rating: float
    global_mean: float


class MovieLensPointwiseDataset(Dataset):
    """PyTorch dataset for pointwise rating regression.

    Each sample is a tuple (user_id, item_id, rating).

    Attributes:
        users: Tensor of user indices.
        items: Tensor of item indices.
        ratings: Tensor of float ratings.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Init dataset.

        Args:
            users: User indices (int64).
            items: Item indices (int64).
            ratings: Ratings (float32).
        """
        assert len(users) == len(items) == len(ratings), "长度不一致"
        # 基本的边界条件检查 —— 确保索引非负、评分范围有效
        assert users.min() >= 0 and items.min() >= 0, "用户或物品索引出现负数"
        self.users = torch.from_numpy(users.astype(np.int64))
        self.items = torch.from_numpy(items.astype(np.int64))
        self.ratings = torch.from_numpy(ratings.astype(np.float32))

    def __len__(self) -> int:
        return self.users.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


class DataModule:
    """Data module that prepares datasets and DataLoaders.

    This class encapsulates the necessary steps to create train/val/test
    splits with user-stratification to avoid leakage.

    Note:
        MovieLens 100K `u.data` format: user_id \t item_id \t rating \t timestamp
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int = 2,
        pin_memory: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize the DataModule.

        Args:
            data_root: Path to store dataset.
            batch_size: Batch size for DataLoaders.
            num_workers: DataLoader workers.
            pin_memory: Pin memory for CUDA.
            val_ratio: Validation ratio per user.
            test_ratio: Test ratio per user.
            random_state: Random state for reproducible split.
        """
        # 参数边界检查
        assert 0 < val_ratio < 0.5 and 0 < test_ratio < 0.5, "验证/测试比例需在(0,0.5)间"
        assert val_ratio + test_ratio < 0.9, "验证+测试占比过大，可能影响训练样本量"
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.train_ds: MovieLensPointwiseDataset
        self.val_ds: MovieLensPointwiseDataset
        self.test_ds: MovieLensPointwiseDataset
        self.info: CorpusInfo

    def prepare(self) -> None:
        """Prepare data: download, parse, build splits."""
        ml_dir = download_ml100k_if_needed(self.data_root)
        data_path = ml_dir / "u.data"
        # 读取 u.data（制表符分隔）
        df = pd.read_csv(
            data_path, sep="\t", header=None, names=["user", "item", "rating", "ts"]
        )
        # 基础边界检查 —— 评分范围与缺失值
        assert not df.isna().any().any(), "数据存在缺失值"
        assert df["rating"].between(1, 5).all(), "评分不在1~5范围内"

        # 连续化索引映射（用户/物品外部ID -> 内部ID）
        user_ids = sorted(df["user"].unique().tolist())
        item_ids = sorted(df["item"].unique().tolist())
        user2idx = {u: i for i, u in enumerate(user_ids)}
        item2idx = {m: i for i, m in enumerate(item_ids)}
        df["u_idx"] = df["user"].map(user2idx)
        df["i_idx"] = df["item"].map(item2idx)

        global_mean = float(df["rating"].mean())

        # 用户分层划分：每个用户内部进行 train/val/test
        grouped = df.groupby("u_idx")
        train_rows, val_rows, test_rows = [], [], []

        rng = np.random.RandomState(self.random_state)
        for u, g in grouped:
            g = g.sample(frac=1.0, random_state=rng)  # 打乱
            n = len(g)
            n_test = max(1, int(n * self.test_ratio))
            n_val = max(1, int(n * self.val_ratio))
            # 防止切分和样本过少用户异常
            if n_test + n_val >= n:
                n_test = max(1, int(0.1 * n))
                n_val = max(1, int(0.1 * n))
            test_split = g.iloc[:n_test]
            val_split = g.iloc[n_test : n_test + n_val]
            train_split = g.iloc[n_test + n_val :]
            # 若训练为空，回退策略
            if len(train_split) == 0 and len(g) >= 3:
                train_split = g.iloc[-(n - (n_test + n_val)) :]
            train_rows.append(train_split)
            val_rows.append(val_split)
            test_rows.append(test_split)

        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)

        # 数据集张量化
        self.train_ds = MovieLensPointwiseDataset(
            train_df["u_idx"].to_numpy(),
            train_df["i_idx"].to_numpy(),
            train_df["rating"].to_numpy(),
        )
        self.val_ds = MovieLensPointwiseDataset(
            val_df["u_idx"].to_numpy(),
            val_df["i_idx"].to_numpy(),
            val_df["rating"].to_numpy(),
        )
        self.test_ds = MovieLensPointwiseDataset(
            test_df["u_idx"].to_numpy(),
            test_df["i_idx"].to_numpy(),
            test_df["rating"].to_numpy(),
        )

        self.info = CorpusInfo(
            num_users=len(user2idx),
            num_items=len(item2idx),
            min_rating=1.0,
            max_rating=5.0,
            global_mean=global_mean,
        )

        logger.info(
            f"Data prepared: train={len(self.train_ds)}, val={len(self.val_ds)}, test={len(self.test_ds)}; "
            f"users={self.info.num_users}, items={self.info.num_items}, global_mean={self.info.global_mean:.4f}"
        )

    def train_dataloader(self) -> DataLoader:
        """Train DataLoader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test DataLoader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )