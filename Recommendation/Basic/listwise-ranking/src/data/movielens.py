from __future__ import annotations
import os
import io
import zipfile
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger
from torch.utils.data import Dataset


ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


class MovieLens100K:
    """MovieLens 100K loader that prepares listwise ranking data.

    This loader creates per-user lists of (item, rating) and provides train/val/test
    splits via leave-one-out by time, with the remaining used for training.

    中文注释：
        - 该类负责下载与解析 MovieLens100K，并按用户生成排序列表。
        - 使用时间戳进行留一法划分：每个用户最后一条为测试、倒数第二条为验证，其余为训练。
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self.dataset_dir = os.path.join(root, "data", "ml-100k")
        os.makedirs(self.dataset_dir, exist_ok=True)

    def prepare(self) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """Download, extract and return ratings with user/item id maps.

        Returns:
            ratings: DataFrame with columns [user_id, item_id, rating, timestamp]
            user2idx: Mapping of raw user id to contiguous index starting at 0
            item2idx: Mapping of raw item id to contiguous index starting at 0
        """
        ratings_path = os.path.join(self.dataset_dir, "u.data")
        if not os.path.exists(ratings_path):
            self._download_and_extract()

        # Load tab-separated: user, item, rating, timestamp
        ratings = pd.read_csv(
            ratings_path, sep='\t', header=None,
            names=['user', 'item', 'rating', 'timestamp']
        )

        # Reindex to 0..U-1 and 0..I-1 for efficient embedding lookup.
        user_ids = sorted(ratings['user'].unique())
        item_ids = sorted(ratings['item'].unique())
        user2idx = {str(u): i for i, u in enumerate(user_ids)}
        item2idx = {str(it): i for i, it in enumerate(item_ids)}

        ratings['user_id'] = ratings['user'].astype(str).map(user2idx).astype(int)
        ratings['item_id'] = ratings['item'].astype(str).map(item2idx).astype(int)
        ratings = ratings[['user_id', 'item_id', 'rating', 'timestamp']]
        return ratings, user2idx, item2idx

    def _download_and_extract(self) -> None:
        zpath = os.path.join(self.dataset_dir, "ml-100k.zip")
        os.makedirs(self.dataset_dir, exist_ok=True)
        logger.info("Downloading MovieLens100K ...")
        with requests.get(ML100K_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zpath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        logger.info("Extracting ml-100k.zip ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(self.dataset_dir)
        # Move files up if extracted into subdir ml-100k/
        inner = os.path.join(self.dataset_dir, 'ml-100k')
        if os.path.isdir(inner):
            for fn in os.listdir(inner):
                shutil.move(os.path.join(inner, fn), os.path.join(self.dataset_dir, fn))
            shutil.rmtree(inner)
        logger.info("MovieLens100K ready at {}", self.dataset_dir)


class ListwiseDataset(Dataset):
    """Listwise dataset yielding per-user lists with variable length.

    Produces a dict: {"user_id": int, "items": np.ndarray[int], "labels": np.ndarray[float]}

    中文注释：
        - 针对每个用户，提供其观测过的 item 列表与评分；
        - 支持按时间留一划分；
        - 训练集可选截断 list 长度以提升速度。
    """

    def __init__(
        self,
        ratings: pd.DataFrame,
        split: str = 'train',
        max_listsize: Optional[int] = None,
        min_items: int = 5,
    ) -> None:
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.max_listsize = max_listsize

        # Sort by time and split per user.
        ratings = ratings.sort_values(['user_id', 'timestamp'])
        grouped = ratings.groupby('user_id')

        rows: List[Dict[str, np.ndarray]] = []
        for uid, g in grouped:
            if len(g) < min_items:
                continue  # 中文：过滤交互过少的用户，提高训练稳定性
            # Leave-one-out by time: ... train ... | val | test
            if len(g) >= 3:
                test_row = g.iloc[-1]
                val_row = g.iloc[-2]
                train_part = g.iloc[:-2]
            else:
                # Fallback: minimal split
                test_row = g.iloc[-1]
                val_row = g.iloc[-1]
                train_part = g.iloc[:-1]

            if split == 'train':
                part = train_part
            elif split == 'val':
                part = pd.DataFrame([val_row])
            else:
                part = pd.DataFrame([test_row])

            items = part['item_id'].to_numpy(dtype=np.int64)
            labels = part['rating'].astype(np.float32).to_numpy()

            # Optional truncate long lists for efficiency
            if max_listsize is not None and len(items) > max_listsize:
                # Keep latest max_listsize interactions (by time already sorted)
                items = items[-max_listsize:]
                labels = labels[-max_listsize:]

            if len(items) < 2:
                continue

            rows.append({"user_id": int(uid), "items": items, "labels": labels})

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.rows[idx]