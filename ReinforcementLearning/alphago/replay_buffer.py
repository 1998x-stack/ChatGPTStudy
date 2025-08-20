# -*- coding: utf-8 -*-
"""自博弈经验回放缓存（线程安全性可按需扩展）."""

from __future__ import annotations
from typing import Deque, Tuple, List
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """固定容量环形缓存."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.obs: Deque[np.ndarray] = deque(maxlen=capacity)
        self.pi: Deque[np.ndarray] = deque(maxlen=capacity)
        self.z: Deque[float] = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, pi: np.ndarray, z: float) -> None:
        """压入一条样本."""
        self.obs.append(obs.astype(np.float32))
        self.pi.append(pi.astype(np.float32))
        self.z.append(float(z))

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机采样一个批次."""
        idx = np.random.choice(len(self.obs), size=min(batch_size, len(self.obs)), replace=False)
        obs = torch.from_numpy(np.stack([self.obs[i] for i in idx], axis=0)).to(device)
        pi = torch.from_numpy(np.stack([self.pi[i] for i in idx], axis=0)).to(device)
        z = torch.from_numpy(np.array([self.z[i] for i in idx], dtype=np.float32)).unsqueeze(-1).to(device)
        return obs, pi, z
