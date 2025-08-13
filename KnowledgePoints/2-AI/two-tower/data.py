# -*- coding: utf-8 -*-
"""Dataset, vocab builders, samplers and DataLoader for Two-Tower pipeline.

This module supports:
- Reading CSV interactions (user_id,item_id,label,timestamp)
- Building user/item vocabularies with PAD/UNK
- Negative sampling for training (random negatives)
- In-batch negatives compatible collate function
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig
from utils import create_logger


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class Interaction:
    """One user-item interaction (binary label)."""
    user_id: str
    item_id: str
    label: int  # 1 for positive, 0 for negative
    timestamp: int


class Vocab:
    """Vocabulary mapping for users/items with PAD/UNK handling."""

    def __init__(self, specials: Optional[List[str]] = None) -> None:
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        specials = specials or [PAD_TOKEN, UNK_TOKEN]
        for tok in specials:
            self.add_token(tok)

    def add_token(self, tok: str) -> int:
        if tok not in self.stoi:
            idx = len(self.itos)
            self.stoi[tok] = idx
            self.itos.append(tok)
        return self.stoi[tok]

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)


def read_interactions(
    path: Path, delimiter: str = ",", has_header: bool = True
) -> List[Interaction]:
    """Read interactions CSV as Interaction list.

    CSV Format: user_id,item_id,label,timestamp
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    interactions: List[Interaction] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 4:
                # 容错：若缺失列，跳过该行
                continue
            user_id, item_id, label, ts = row[:4]
            try:
                interactions.append(
                    Interaction(
                        user_id=str(user_id),
                        item_id=str(item_id),
                        label=int(label),
                        timestamp=int(ts),
                    )
                )
            except Exception:
                # 非法行：跳过
                continue
    return interactions


def build_vocabs(
    train: List[Interaction],
    max_users: Optional[int] = None,
    max_items: Optional[int] = None,
) -> Tuple[Vocab, Vocab]:
    """Build user and item vocabularies from training interactions."""
    user_vocab = Vocab()
    item_vocab = Vocab()
    # 统计并构建词表
    u_cnt: Dict[str, int] = defaultdict(int)
    i_cnt: Dict[str, int] = defaultdict(int)
    for it in train:
        if it.label == 1:  # 只对正样本计数以控制规模（也可包含负样本）
            u_cnt[it.user_id] += 1
            i_cnt[it.item_id] += 1
    # 可选上限控制
    users_sorted = sorted(u_cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    items_sorted = sorted(i_cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_users is not None:
        users_sorted = users_sorted[:max_users]
    if max_items is not None:
        items_sorted = items_sorted[:max_items]
    for uid, _ in users_sorted:
        user_vocab.add_token(uid)
    for iid, _ in items_sorted:
        item_vocab.add_token(iid)
    return user_vocab, item_vocab


class TwoTowerTrainDataset(Dataset):
    """Training dataset with random negative sampling.

    Each __getitem__ yields one positive (user,item_pos) and `num_negatives` negatives.
    """

    def __init__(
        self,
        interactions: List[Interaction],
        user_vocab: Vocab,
        item_vocab: Vocab,
        num_negatives: int = 4,
    ) -> None:
        if num_negatives < 0:
            raise ValueError("num_negatives must be >= 0.")
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.num_negatives = num_negatives
        # 仅保留正样本用于驱动负采样
        self.pos_samples: List[Tuple[int, int]] = []
        user_pos_items: Dict[int, set] = defaultdict(set)
        for it in interactions:
            if it.label != 1:
                continue
            uidx = user_vocab.stoi.get(it.user_id, user_vocab.unk_idx)
            iidx = item_vocab.stoi.get(it.item_id, item_vocab.unk_idx)
            self.pos_samples.append((uidx, iidx))
            user_pos_items[uidx].add(iidx)
        self.user_pos_items = user_pos_items
        self.num_items = len(item_vocab)
        if len(self.pos_samples) == 0:
            raise ValueError("No positive samples for training.")

    def __len__(self) -> int:
        return len(self.pos_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uidx, iidx_pos = self.pos_samples[idx]
        # 负采样：从全集随机采样，排除用户的正集合
        negs: List[int] = []
        if self.num_negatives > 0:
            pos_set = self.user_pos_items.get(uidx, set())
            # 边界：若正集合已覆盖全部（病态极少），则退化为随机
            attempts = 0
            while len(negs) < self.num_negatives and attempts < self.num_negatives * 20:
                cand = np.random.randint(0, self.num_items)
                if cand not in pos_set and cand != iidx_pos:
                    negs.append(int(cand))
                attempts += 1
            # 若仍不足，填充为任意随机（保证 batch 形状）
            while len(negs) < self.num_negatives:
                negs.append(int(np.random.randint(0, self.num_items)))
        return {
            "user_idx": torch.tensor(uidx, dtype=torch.long),
            "pos_item_idx": torch.tensor(iidx_pos, dtype=torch.long),
            "neg_item_idx": torch.tensor(negs, dtype=torch.long)
            if len(negs) > 0
            else torch.empty(0, dtype=torch.long),
        }


class TwoTowerEvalDataset(Dataset):
    """Evaluation dataset holding single positive interactions per row."""

    def __init__(
        self,
        interactions: List[Interaction],
        user_vocab: Vocab,
        item_vocab: Vocab,
        strict_id: bool = False,
    ) -> None:
        self.samples: List[Tuple[int, int]] = []
        self.strict = strict_id
        for it in interactions:
            if it.label != 1:
                continue
            uidx = user_vocab.stoi.get(it.user_id, user_vocab.unk_idx)
            iidx = item_vocab.stoi.get(it.item_id, item_vocab.unk_idx)
            # 严格模式：如果映射到 UNK，直接跳过该样本
            if strict_id and (uidx == user_vocab.unk_idx or iidx == item_vocab.unk_idx):
                continue
            self.samples.append((uidx, iidx))
        if len(self.samples) == 0:
            raise ValueError("No positive samples for evaluation.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uidx, iidx = self.samples[idx]
        return {
            "user_idx": torch.tensor(uidx, dtype=torch.long),
            "pos_item_idx": torch.tensor(iidx, dtype=torch.long),
        }


def collate_train(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for training.

    Returns:
        A dict with tensors:
            user_idx: [B]
            pos_item_idx: [B]
            neg_item_idx: [B, N] or [B, 0] if N=0
    """
    user_idx = torch.stack([b["user_idx"] for b in batch], dim=0)
    pos_item_idx = torch.stack([b["pos_item_idx"] for b in batch], dim=0)
    if batch[0]["neg_item_idx"].numel() > 0:
        neg_item_idx = torch.stack([b["neg_item_idx"] for b in batch], dim=0)
    else:
        neg_item_idx = torch.empty((len(batch), 0), dtype=torch.long)
    return {
        "user_idx": user_idx,
        "pos_item_idx": pos_item_idx,
        "neg_item_idx": neg_item_idx,
    }


def collate_eval(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for evaluation."""
    user_idx = torch.stack([b["user_idx"] for b in batch], dim=0)
    pos_item_idx = torch.stack([b["pos_item_idx"] for b in batch], dim=0)
    return {"user_idx": user_idx, "pos_item_idx": pos_item_idx}


def build_dataloaders(cfg: DataConfig):
    """Build train/val/test dataloaders + vocabs.

    中文说明：
        1) 读取CSV交互数据
        2) 基于训练集构建 user/item 词表
        3) 负采样训练集、评测集
        4) 返回 DataLoader 和词表
    """
    logger = create_logger()
    train = read_interactions(cfg.train_csv, cfg.delimiter, cfg.has_header)
    valid = read_interactions(cfg.valid_csv, cfg.delimiter, cfg.has_header)
    test = read_interactions(cfg.test_csv, cfg.delimiter, cfg.has_header)

    logger.info(f"Loaded interactions: train={len(train)}, valid={len(valid)}, test={len(test)}")

    user_vocab, item_vocab = build_vocabs(train, cfg.max_users, cfg.max_items)
    logger.info(
        f"Vocab sizes: users={len(user_vocab)} (pad={user_vocab.pad_idx}, unk={user_vocab.unk_idx}), "
        f"items={len(item_vocab)} (pad={item_vocab.pad_idx}, unk={item_vocab.unk_idx})"
    )

    train_ds = TwoTowerTrainDataset(train, user_vocab, item_vocab, cfg.num_negatives)
    valid_ds = TwoTowerEvalDataset(valid, user_vocab, item_vocab, strict_id=cfg.strict_id)
    test_ds = TwoTowerEvalDataset(test, user_vocab, item_vocab, strict_id=cfg.strict_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_train,
        drop_last=True,  # 保证 in-batch negative 的稳定性
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_eval,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_eval,
    )
    return train_loader, valid_loader, test_loader, user_vocab, item_vocab