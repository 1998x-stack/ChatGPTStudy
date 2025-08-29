from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens_100k(data_dir: str) -> Path:
    """Download MovieLens 100k if not exists.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        Path to extracted ml-100k directory.

    Raises:
        RuntimeError: If download fails.
    """
    # 自动下载 MovieLens 100k（若目录下不存在）
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / "ml-100k"
    if target_dir.exists():
        logger.info(f"MovieLens-100k found at: {target_dir}")
        return target_dir

    zip_path = root / "ml-100k.zip"
    logger.info("Downloading MovieLens-100k ...")
    try:
        resp = requests.get(ML100K_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MovieLens 100k: {e}. "
            f"Place 'ml-100k' under {data_dir} manually."
        )

    logger.info("Extracting MovieLens-100k ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    logger.info(f"Extracted to {target_dir}")
    return target_dir


@dataclass
class UserQuery:
    """Container for a single query (a user) with multiple items and relevances."""
    user_id: int
    item_ids: np.ndarray  # [n_items]
    rels: np.ndarray      # [n_items], relevance (ratings)


class ML100KDataModule:
    """Data module for MovieLens-100k to produce per-user queries."""

    def __init__(
        self,
        data_dir: str,
        max_items_per_user: int = 50,
        test_size: float = 0.1,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize data module.

        Args:
            data_dir: Dataset directory (auto-download if missing).
            max_items_per_user: Cap items per user to control O(n^2) pair cost.
            test_size: Fraction of users for test split.
            val_size: Fraction of users for validation split.
            seed: Random seed for splits.
        """
        self.data_dir = data_dir
        self.max_items_per_user = max_items_per_user
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

        self.root = download_movielens_100k(data_dir)
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}

        self.train_queries: List[UserQuery] = []
        self.val_queries: List[UserQuery] = []
        self.test_queries: List[UserQuery] = []

        self.num_users: int = 0
        self.num_items: int = 0

    def _load_ratings(self) -> pd.DataFrame:
        """Load ratings data.

        Returns:
            DataFrame with columns [user_id, item_id, rating].
        """
        # 读取 u.data（user item rating timestamp），用制表符分隔
        path = self.root / "u.data"
        df = pd.read_csv(
            path, sep="\t", header=None, names=["user_id", "item_id", "rating", "ts"]
        )
        df = df[["user_id", "item_id", "rating"]]
        return df

    def _reindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex user and item ids to [0..n-1] and build maps."""
        # 将用户和物品 ID 重映射为连续索引，便于 Embedding 处理
        unique_users = df["user_id"].unique()
        unique_items = df["item_id"].unique()
        self.user_map = {u: i for i, u in enumerate(sorted(unique_users))}
        self.item_map = {it: i for i, it in enumerate(sorted(unique_items))}
        df["user_id"] = df["user_id"].map(self.user_map)
        df["item_id"] = df["item_id"].map(self.item_map)
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        return df

    @staticmethod
    def _sample_cap_items(items: np.ndarray, rels: np.ndarray, max_items: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly cap number of items per user for efficiency."""
        # 随机截取用户的部分条目，控制每个查询的最大项目数
        n = items.shape[0]
        if n <= max_items:
            return items, rels
        idx = rng.choice(n, size=max_items, replace=False)
        return items[idx], rels[idx]

    def _build_queries(self, df: pd.DataFrame, users: np.ndarray, rng: np.random.RandomState) -> List[UserQuery]:
        """Construct UserQuery list for a set of users."""
        # 将用户对应的所有 (item, rating) 聚合为一个查询
        queries: List[UserQuery] = []
        g = df.groupby("user_id")
        for u in users:
            if u not in g.groups:
                continue
            sub = g.get_group(u)
            # 过滤边界情况：至少两个不同相关性才有成对比较意义
            if sub.shape[0] < 2:
                continue
            items = sub["item_id"].to_numpy()
            rels = sub["rating"].to_numpy().astype(np.float32)
            # 去除全相同相关性的用户（否则 pair_mask 全 0）
            if np.allclose(rels, rels[0]):
                continue
            items, rels = self._sample_cap_items(items, rels, self.max_items_per_user, rng)
            queries.append(UserQuery(user_id=int(u), item_ids=items, rels=rels))
        return queries

    def setup(self) -> None:
        """Prepare train/val/test queries."""
        df = self._reindex(self._load_ratings())
        users = np.arange(self.num_users)
        # 先划分 test，再从剩余中划分 val
        users_train, users_test = train_test_split(
            users, test_size=self.test_size, random_state=self.seed, shuffle=True
        )
        users_train, users_val = train_test_split(
            users_train, test_size=self.val_size, random_state=self.seed, shuffle=True
        )
        rng = np.random.RandomState(self.seed)
        self.train_queries = self._build_queries(df, users_train, rng)
        self.val_queries = self._build_queries(df, users_val, rng)
        self.test_queries = self._build_queries(df, users_test, rng)

        # 再次检查边界
        assert self.num_users > 0 and self.num_items > 0, "Empty users/items after reindex."
        if len(self.train_queries) == 0:
            raise RuntimeError("No valid training queries found; check filtering conditions.")
        logger.info(
            f"Users: {self.num_users}, Items: {self.num_items} | "
            f"TrainQ: {len(self.train_queries)}, ValQ: {len(self.val_queries)}, TestQ: {len(self.test_queries)}"
        )


class QueryDataset(Dataset):
    """Dataset that yields per-user queries."""

    def __init__(self, queries: List[UserQuery]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> UserQuery:
        return self.queries[idx]


def collate_queries(batch: List[UserQuery]) -> List[UserQuery]:
    """Collate function to keep list of queries as-is."""
    # 不做拼接，每个元素是一个独立查询（用户）
    return batch


def create_dataloaders(
    dm: ML100KDataModule,
    batch_users: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders for train/val/test."""
    train_ds = QueryDataset(dm.train_queries)
    val_ds = QueryDataset(dm.val_queries)
    test_ds = QueryDataset(dm.test_queries)

    train_loader = DataLoader(
        train_ds, batch_size=batch_users, shuffle=True, num_workers=num_workers,
        collate_fn=collate_queries, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_users, shuffle=False, num_workers=0,
        collate_fn=collate_queries, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_users, shuffle=False, num_workers=0,
        collate_fn=collate_queries, drop_last=False,
    )
    return train_loader, val_loader, test_loader