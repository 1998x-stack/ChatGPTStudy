# -*- coding: utf-8 -*-
"""Utility helpers for Pointwise Ranking project.

This file includes:
- Reproducibility tooling.
- Directory helpers.
- Download and checksum verify for MovieLens.
- Metric helpers (simple).
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests
import torch
from loguru import logger


ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_ZIP_MD5 = "0e33842e24a9c977be4e0107933c0723"  # 官方md5


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Global seed.
    """
    # 设置所有随机源 —— 确保可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对多GPU也生效
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to: {seed}")


def ensure_dir(path: str | Path, empty: bool = False) -> Path:
    """Ensure a directory exists.

    Args:
        path: Directory path.
        empty: If True, clear existing content.

    Returns:
        Path: The ensured directory path.
    """
    p = Path(path)
    if p.exists() and empty:
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 checksum for a file.

    Args:
        path: File path.
        chunk_size: Chunk size to read.

    Returns:
        str: MD5 hex.
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ml100k_if_needed(data_root: str | Path) -> Path:
    """Download MovieLens 100K if not exists; verify checksum.

    Args:
        data_root: Directory to place dataset.

    Returns:
        Path: Path to extracted `ml-100k` directory.
    """
    data_root = Path(data_root)
    ensure_dir(data_root)
    ml_dir = data_root / "ml-100k"
    if ml_dir.exists():
        logger.info(f"Found dataset at {ml_dir}")
        return ml_dir

    # 下载 zip
    zip_path = data_root / "ml-100k.zip"
    logger.info("Downloading MovieLens 100K ...")
    with requests.get(ML100K_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    # 校验 MD5
    md5 = md5_of_file(zip_path)
    if md5 != ML100K_ZIP_MD5:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError("MD5 mismatch for ml-100k.zip, download may be corrupted.")

    # 解压
    shutil.unpack_archive(str(zip_path), extract_dir=str(data_root))
    logger.info(f"Extracted dataset to {data_root}")
    return ml_dir


def get_device(force: Optional[str] = None) -> torch.device:
    """Get torch device.

    Args:
        force: Force 'cuda' or 'cpu'; if None, auto detect.

    Returns:
        torch.device: Selected device.
    """
    if force is not None:
        return torch.device(force)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")