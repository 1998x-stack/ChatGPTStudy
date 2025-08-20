# -*- coding: utf-8 -*-
"""Utility helpers for Rainbow DQN."""

from __future__ import annotations

import math
import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_global_seeds(seed: int) -> None:
    """Set global seeds for reproducibility."""
    # 中文：设置随机种子，保证实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_schedule(start: float, final: float, t: int, t_final: int) -> float:
    """Linear decay schedule.

    Args:
        start: Starting value.
        final: Final value.
        t: Current step (>= 0).
        t_final: Steps over which to decay.

    Returns:
        Decayed value clipped to [min(start, final), max(start, final)].
    """
    if t_final <= 0:
        return final
    ratio = min(max(t / float(t_final), 0.0), 1.0)
    return start + ratio * (final - start)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy parameters from source to target (hard update)."""
    # 中文：硬更新目标网络参数
    target.load_state_dict(source.state_dict())


def projection_distribution(
    next_support: torch.Tensor,
    next_prob: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    vmin: float,
    vmax: float,
    atoms: int,
) -> torch.Tensor:
    """Project the target distribution (C51) onto fixed support.

    This is the standard C51 projection (Bellemare et al. 2017).

    Args:
        next_support: (B, atoms) target value support z'.
        next_prob: (B, atoms) target probabilities p'.
        rewards: (B,) rewards.
        dones: (B,) terminal flags (float 0/1).
        gamma: Discount factor compounded by n-step already.
        vmin: Minimum of value support.
        vmax: Maximum of value support.
        atoms: Number of atoms.

    Returns:
        Projected distribution (B, atoms).
    """
    # 中文：C51投影操作，将目标分布投影回固定的支持点上
    batch_size = rewards.shape[0]
    delta_z = (vmax - vmin) / (atoms - 1)
    support = torch.linspace(vmin, vmax, atoms, device=next_support.device)
    rewards = rewards.unsqueeze(1).expand(batch_size, atoms)
    dones = dones.unsqueeze(1).expand(batch_size, atoms)
    tz = rewards + (1.0 - dones) * gamma * next_support  # (B, atoms)
    tz = tz.clamp(min=vmin, max=vmax)

    b = (tz - vmin) / delta_z  # (B, atoms)
    l = b.floor().to(torch.int64)
    u = b.ceil().to(torch.int64)

    proj = torch.zeros_like(next_prob)
    offset = (
        torch.linspace(0, (batch_size - 1) * atoms, batch_size, device=next_prob.device)
        .unsqueeze(1)
        .expand(batch_size, atoms)
        .to(torch.int64)
    )
    proj.view(-1).index_add_(0, (l + offset).view(-1), (next_prob * (u - b)).view(-1))
    proj.view(-1).index_add_(0, (u + offset).view(-1), (next_prob * (b - l)).view(-1))
    return proj


def safe_log_prob(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute log(x) safely with clamping."""
    return torch.log(x.clamp(min=eps))


def compute_kl_div(
    target: torch.Tensor,
    pred: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute KL(target || pred)."""
    # 中文：计算KL散度，作为分布式损失或PER优先级依据
    t = target.clamp(min=eps)
    p = pred.clamp(min=eps)
    kl = (t * (torch.log(t) - torch.log(p))).sum(dim=1)
    if reduction == "mean":
        return kl.mean()
    return kl


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - Var[y - y_hat] / Var[y]."""
    # 中文：评估拟合程度的指标
    var_y = np.var(y_true)
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))
