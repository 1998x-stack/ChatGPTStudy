from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch


def collate_listwise(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length lists into dense tensors with mask.

    中文注释：
        - 将不同用户的变长列表对齐（padding），并生成 mask；
        - mask=True 表示有效条目，False 为 padding。
    """
    user_ids = [b['user_id'] for b in batch]
    max_len = max(len(b['items']) for b in batch)

    items = np.zeros((len(batch), max_len), dtype=np.int64)
    labels = np.zeros((len(batch), max_len), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.bool_)

    for i, b in enumerate(batch):
        L = len(b['items'])
        items[i, :L] = b['items']
        labels[i, :L] = b['labels']
        mask[i, :L] = True

    return {
        'user_ids': torch.tensor(user_ids, dtype=torch.long),
        'items': torch.tensor(items, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.float32),
        'mask': torch.tensor(mask, dtype=torch.bool),
    }


def sample_pairwise_from_list(
    labels: torch.Tensor,
    mask: torch.Tensor,
    per_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample pair indices (i, j) within each query where label_i != label_j.

    Returns:
        q_idx: (B*per_query,) query indices
        i_idx: (B*per_query,) position i in list
        j_idx: (B*per_query,) position j in list

    中文注释：
        - 在每个用户列表中随机采样若干对 (i, j)，要求标签不同；
        - 用于 pairwise PL 损失构造。
    """
    B, L = labels.shape
    q_list = []
    i_list = []
    j_list = []
    for q in range(B):
        valid = mask[q].nonzero(as_tuple=False).flatten()
        if len(valid) < 2:
            continue
        y = labels[q, valid]
        # Only consider pairs with different labels
        diff_positions = []
        idxs = valid.cpu().numpy().tolist()
        for _ in range(per_query):
            a, b = np.random.choice(idxs, 2, replace=False)
            if labels[q, a] != labels[q, b]:
                q_list.append(q)
                i_list.append(a)
                j_list.append(b)
    if len(q_list) == 0:
        # Return dummy zero-sized tensors if no pairs
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
    return (
        torch.tensor(q_list, dtype=torch.long),
        torch.tensor(i_list, dtype=torch.long),
        torch.tensor(j_list, dtype=torch.long),
    )