# -*- coding: utf-8 -*-
"""Rollout storage with GAE.

中文说明：
    - 存储一段 rollout 的轨迹（NT 条），并计算 GAE 优势与返回。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """Flattened batch for PPO updates."""
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Rollout storage for PPO with GAE.

    中文说明：
        - 存储 (num_steps, num_envs, ...) 的张量；
        - rollout 结束后调用 compute_returns() 计算优势与回报；
        - 提供打乱后的小批量迭代器。
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        num_envs: int,
        num_steps: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Add a single step transition."""
        assert self.ptr < self.num_steps, "Rollout 指针越界，请先 compute_returns 再继续采样"
        self.obs[self.ptr] = obs
        # 动作 shape 对齐（离散动作存 float，再在更新时转 long）
        self.actions[self.ptr] = action if action.dtype == torch.float32 else action.float()
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    @torch.inference_mode()
    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        last_done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        中文说明：
            - 使用 bootstrap 的 last_value；
            - 对 done 进行掩码，episode 结束处不穿越。
        """
        advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.num_steps)):
            next_nonterminal = 1.0 - (self.dones[t] if t < self.num_steps - 1 else last_done)
            next_values = self.values[t + 1] if t < self.num_steps - 1 else last_value
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        # 展平为 (batch, ...)
        batch = RolloutBatch(
            obs=self.obs.reshape(self.num_steps * self.num_envs, *self.obs.shape[2:]),
            actions=self.actions.reshape(self.num_steps * self.num_envs, *self.actions.shape[2:]),
            logprobs=self.logprobs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=self.values.reshape(-1),
        )
        # 重置指针
        self.ptr = 0
        return batch.advantages, batch.returns

    def iter_minibatches(self, num_minibatches: int):
        """Yield shuffled minibatches."""
        batch_size = self.num_steps * self.num_envs
        indices = torch.randperm(batch_size, device=self.device)
        mb_size = batch_size // num_minibatches
        for i in range(num_minibatches):
            mb_idx = indices[i * mb_size:(i + 1) * mb_size]
            yield (
                self.obs.reshape(batch_size, *self.obs.shape[2:])[mb_idx],
                self.actions.reshape(batch_size, *self.actions.shape[2:])[mb_idx],
                self.logprobs.reshape(batch_size)[mb_idx],
                self.values.reshape(batch_size)[mb_idx],
                mb_idx,  # 索引用于提取 advantages/returns
            )
