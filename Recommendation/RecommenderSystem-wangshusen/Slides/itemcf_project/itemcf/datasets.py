from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
from loguru import logger
from scipy import sparse


ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass(frozen=True)
class DatasetArtifacts:
    """Container for dataset artifacts.

    Attributes:
        user_mapping: Dict original_user_id -> contiguous int id
        item_mapping: Dict original_item_id -> contiguous int id
        train_csr: CSR matrix for train interactions (users x items)
        test_pos: Dict[int, list[int]] positive test items per user
        train_seen: Dict[int, set[int]] items seen in train per user
        timestamps: Optional array of timestamps aligned with interactions (for last-N)
    """
    user_mapping: Dict[Any, int]
    item_mapping: Dict[Any, int]
    train_csr: sparse.csr_matrix
    test_pos: Dict[int, list[int]]
    train_seen: Dict[int, set[int]]
    timestamps: np.ndarray


def _download_ml100k(data_dir: str) -> str:
    """Download MovieLens 100k and return extracted folder path."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    folder_path = os.path.join(data_dir, "ml-100k")
    if os.path.isdir(folder_path) and os.path.isfile(os.path.join(folder_path, "u.data")):
        logger.info("ml-100k already downloaded at {}", folder_path)
        return folder_path

    logger.info("Downloading MovieLens 100k from {}", ML_100K_URL)
    r = requests.get(ML_100K_URL, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    logger.info("Extracted to {}", folder_path)
    return folder_path


def _load_raw_ml100k(folder_path: str) -> pd.DataFrame:
    """Load raw u.data as DataFrame with columns: user, item, rating, ts."""
    path = os.path.join(folder_path, "u.data")
    names = ["user", "item", "rating", "ts"]
    df = pd.read_csv(path, sep="\t", names=names, engine="python")
    if df.empty:
        raise ValueError("Loaded u.data is empty.")
    return df


def _apply_min_filters(df: pd.DataFrame, min_user_inter: int, min_item_inter: int) -> pd.DataFrame:
    """Filter by minimum user/item interactions with iterative pruning."""
    prev_shape = None
    cur = df
    # 反复过滤直到稳定 —— 防止单次过滤后出现新的冷用户/冷物品
    while prev_shape != cur.shape:
        prev_shape = cur.shape
        users_ok = cur.groupby("user").size()
        items_ok = cur.groupby("item").size()
        keep_users = users_ok[users_ok >= min_user_inter].index
        keep_items = items_ok[items_ok >= min_item_inter].index
        cur = cur[cur["user"].isin(keep_users) & cur["item"].isin(keep_items)]
    return cur


def _binarize_if_needed(df: pd.DataFrame, implicit_like: bool, threshold: float) -> pd.DataFrame:
    """Binarize ratings to implicit feedback if requested (>=threshold -> 1)."""
    if implicit_like:
        df = df.copy()
        df["rating"] = (df["rating"] >= threshold).astype(np.float32)  # 二值化评分
    else:
        df["rating"] = df["rating"].astype(np.float32)
    return df


def _split_train_test(
    df: pd.DataFrame, holdout: str, ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/test by holdout strategy."""
    if holdout == "leave_one":
        # 每个用户按时间排序，最后一条留作测试
        df = df.sort_values(["user", "ts"])
        test = df.groupby("user").tail(1)
        train = df.drop(test.index)
    else:
        # 按比例分层抽样
        def split_grp(g: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            g = g.sort_values("ts")
            n = len(g)
            cut = int(max(1, n * (1 - ratio)))
            return g.iloc[:cut], g.iloc[cut:]

        parts = [split_grp(g) for _, g in df.groupby("user")]
        train = pd.concat([p[0] for p in parts], ignore_index=True)
        test = pd.concat([p[1] for p in parts], ignore_index=True)

    if train.empty or test.empty:
        raise ValueError("Train/test split resulted in empty partition.")
    return train, test


def _build_mappings(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[dict, dict]:
    """Build contiguous id mappings for users and items based on train+test."""
    users = pd.Index(sorted(set(train_df["user"]).union(set(test_df["user"])))).tolist()
    items = pd.Index(sorted(set(train_df["item"]).union(set(test_df["item"])))).tolist()
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(items)}
    return user_map, item_map


def _to_csr(
    df: pd.DataFrame, user_map: dict, item_map: dict
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Convert (user,item,rating,ts) to CSR matrix and timestamp array."""
    rows = df["user"].map(user_map).values
    cols = df["item"].map(item_map).values
    data = df["rating"].astype(np.float32).values
    ts = df["ts"].astype(np.int64).values  # 时间戳保留以便 last-N
    n_users = len(user_map)
    n_items = len(item_map)
    csr = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    csr.sort_indices()
    return csr, ts


def load_dataset_artifacts(
    data_dir: str,
    min_user_inter: int,
    min_item_inter: int,
    holdout: str,
    ratio: float,
    implicit_like: bool,
    implicit_threshold: float,
) -> DatasetArtifacts:
    """Load and prepare dataset artifacts for ItemCF pipeline.

    Returns:
        DatasetArtifacts: All prepared structures for downstream components.
    """
    folder = _download_ml100k(data_dir)
    raw = _load_raw_ml100k(folder)
    raw = _apply_min_filters(raw, min_user_inter, min_item_inter)
    raw = _binarize_if_needed(raw, implicit_like, implicit_threshold)

    train_df, test_df = _split_train_test(raw, holdout, ratio)

    user_map, item_map = _build_mappings(train_df, test_df)
    train_csr, train_ts = _to_csr(train_df, user_map, item_map)

    # 测试集正样本（每用户一个或多个，取决于切分策略）
    test_pos = {}
    for u, g in test_df.groupby("user"):
        uid = user_map[u]
        test_pos[uid] = [item_map[i] for i in g["item"].tolist()]

    # 训练集中每个用户见过的物品集合
    train_seen = {}
    for u_idx in range(train_csr.shape[0]):
        row_start, row_end = train_csr.indptr[u_idx], train_csr.indptr[u_idx + 1]
        seen_items = set(train_csr.indices[row_start:row_end].tolist())
        train_seen[u_idx] = seen_items

    logger.info(
        "Dataset ready: users={}, items={}, train_nnz={}, test_users={}",
        train_csr.shape[0],
        train_csr.shape[1],
        train_csr.nnz,
        len(test_pos),
    )

    return DatasetArtifacts(
        user_mapping=user_map,
        item_mapping=item_map,
        train_csr=train_csr,
        test_pos=test_pos,
        train_seen=train_seen,
        timestamps=train_ts,
    )