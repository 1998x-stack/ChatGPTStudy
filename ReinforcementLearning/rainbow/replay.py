# -*- coding: utf-8 -*-
"""Experience Replay buffers (Uniform and Prioritized) with n-step support."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class NStepBuffer:
    """Temporary n-step buffer to accumulate (s,a,r,done) before pushing to RB.

    This helps building n-step transitions online.

    Attributes:
        n: n-step length.
        gamma: Discount factor.
    """

    def __init__(self, n: int, gamma: float):
        if n <= 0:
            raise ValueError("n must be positive.")
        self.n = n
        self.gamma = gamma
        self.deque: Deque[Tuple[np.ndarray, int, float, bool]] = deque()

    def push(self, s: np.ndarray, a: int, r: float, done: bool) -> None:
        """Push a one-step transition into the n-step buffer."""
        self.deque.append((s, a, r, done))

    def can_pop(self) -> bool:
        """Whether we can pop an n-step transition."""
        return len(self.deque) >= self.n

    def pop(self, s_next: np.ndarray) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Build n-step transition (s0, a0, R^(n), s_n, done_n)."""
        assert self.can_pop()
        s0, a0, _, _ = self.deque[0]
        R = 0.0
        done_n = False
        # 中文：累积未来n步奖励，并在碰到done时提前截断
        for i in range(self.n):
            s, a, r, done = self.deque[i]
            R += (self.gamma ** i) * r
            if done:
                done_n = True
                break
        # 中文：弹出一条transition
        self.deque.popleft()
        return s0, a0, R, s_next, done_n

    def flush(self) -> None:
        """Clear buffer."""
        self.deque.clear()


class SumTree:
    """SumTree for proportional PER."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data: List = [None] * capacity
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, p: float, data) -> None:
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        # 中文：向上更新父节点
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def get(self, s: float) -> Tuple[int, float, any]:
        """Sample a leaf index by cumulative sum value s."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                idx = left + 1
                s -= self.tree[left]
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer with proportional sampling."""

    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta0: float,
        beta_steps: int,
        eps: float = 1e-6,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1].")
        if not (0.0 < beta0 <= 1.0):
            raise ValueError("beta0 must be in (0,1].")
        if beta_steps <= 0:
            raise ValueError("beta_steps must be positive.")
        self.capacity = capacity
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_steps = beta_steps
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        # Storage arrays
        self.obs = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.tree.size

    def beta_by_step(self, step: int) -> float:
        """Anneal beta linearly to 1.0."""
        t = min(max(step, 0), self.beta_steps)
        return self.beta0 + (1.0 - self.beta0) * (t / float(self.beta_steps))

    def add(self, obs, action, reward, next_obs, done) -> None:
        idx = self.tree.write
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.tree.add(self.max_priority, idx)

    def sample(self, batch_size: int, step: int) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        batch = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "dones": []}
        idxs = np.zeros((batch_size,), dtype=np.int64)
        priorities = np.zeros((batch_size,), dtype=np.float32)
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data_idx = self.tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            j = data_idx
            batch["obs"].append(self.obs[j])
            batch["actions"].append(self.actions[j])
            batch["rewards"].append(self.rewards[j])
            batch["next_obs"].append(self.next_obs[j])
            batch["dones"].append(self.dones[j])

        for k in batch:
            batch[k] = np.stack(batch[k], axis=0)

        # IS weights
        probs = priorities / (self.tree.total() + 1e-8)
        probs = np.maximum(probs, 1e-12)
        weights = (self.tree.size * probs) ** -self.beta_by_step(step)
        weights = weights / weights.max()

        return batch, idxs, priorities, weights

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray) -> None:
        for leaf_idx, p in zip(idxs, new_priorities):
            p = float(max(p, self.eps))
            self.tree.update(int(leaf_idx), p)
            self.max_priority = max(self.max_priority, p)


class UniformReplayBuffer:
    """Uniform replay buffer with same interface (for ablation)."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self.capacity = capacity
        self.size = 0
        self.write = 0

        self.obs = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.write] = obs
        self.actions[self.write] = action
        self.rewards[self.write] = reward
        self.next_obs[self.write] = next_obs
        self.dones[self.write] = float(done)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, step: int):
        idxs = np.random.randint(0, self.size, size=(batch_size,))
        batch = {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }
        # Return placeholders to keep interface compatible with PER
        priorities = np.ones_like(idxs, dtype=np.float32)
        weights = np.ones_like(priorities, dtype=np.float32)
        return batch, idxs, priorities, weights

    def update_priorities(self, idxs, new_priorities) -> None:
        # Uniform buffer ignores priority updates.
        pass
