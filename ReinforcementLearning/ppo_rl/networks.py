# -*- coding: utf-8 -*-
"""Actor-Critic networks and distributions.

中文说明：
    - 提供统一的 Actor-Critic 网络，支持离散与连续动作空间。
    - 连续动作使用 SquashedDiagGaussian（tanh 变换 + 对数概率修正）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def orthogonal_init(m: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to linear layers.

    中文说明：
        - Orthogonal 初始化在 PPO 等策略梯度中比较稳健。
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class SquashedDiagGaussian:
    """Tanh-squashed diagonal Gaussian distribution.

    中文说明：
        - 对连续动作空间：Normal -> tanh -> 映射到 [-1, 1] 再缩放到 action 空间范围。
        - 对数概率需要考虑 tanh 的雅可比修正项：log(1 - tanh(x)^2)。

    References:
        - Appendix C of SAC, and common PPO implementations with Tanh-Normal.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        self.mean = mean
        self.log_std = torch.clamp(log_std, -20, 2)  # 中文：稳定性边界
        self.std = torch.exp(self.log_std)
        self.base_dist = Normal(self.mean, self.std)
        self.low = low
        self.high = high
        self.eps = eps

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return (action, log_prob)."""
        u = self.base_dist.rsample()  # reparameterization
        y = torch.tanh(u)
        # 修正项：sum(log(1 - tanh(u)^2)), 加 eps 防止 log(0)
        log_prob = self.base_dist.log_prob(u) - torch.log(1 - y.pow(2) + self.eps)
        log_prob = log_prob.sum(-1)

        # scale to [low, high]
        act = (y + 1) / 2 * (self.high - self.low) + self.low
        return act, log_prob

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Compute log-probability for given action (inverse of squashing & scaling)."""
        # 反变换到 [-1,1]
        y = 2 * (action - self.low) / (self.high - self.low) - 1.0
        y = torch.clamp(y, -1 + self.eps, 1 - self.eps)
        # 反 tanh：atanh(y) = 0.5 * log((1+y)/(1-y))
        u = 0.5 * (torch.log1p(y + self.eps) - torch.log1p(-y + self.eps))
        # 基础高斯的 log_prob
        log_prob = self.base_dist.log_prob(u) - torch.log(1 - y.pow(2) + self.eps)
        return log_prob.sum(-1)

    def entropy(self) -> torch.Tensor:
        """Approximate entropy via base Gaussian's entropy minus squashing penalty (rough)."""
        # 严格的熵需要期望项，这里用基高斯熵作为近似，足够做 PPO 的探索奖励
        return self.base_dist.entropy().sum(-1)


class ActorCritic(nn.Module):
    """Unified Actor-Critic network for discrete and continuous control.

    中文说明：
        - 共享特征提取 MLP，分支输出：策略（离散 logits 或连续均值/对数方差）、价值函数。
        - 连续动作使用可训练 log_std（全局向量），也可改为 state-dependent std。
    """

    def __init__(
        self,
        obs_dim: int,
        action_space,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.is_discrete = hasattr(action_space, "n")
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        if self.is_discrete:
            self.action_dim = action_space.n
            self.policy_head = nn.Linear(hidden_size, self.action_dim)
        else:
            self.action_dim = action_space.shape[0]
            self.policy_head = nn.Linear(hidden_size, self.action_dim)
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
            # 动作范围缓存
            low = torch.as_tensor(action_space.low, dtype=torch.float32)
            high = torch.as_tensor(action_space.high, dtype=torch.float32)
            self.register_buffer("act_low", low)
            self.register_buffer("act_high", high)

        self.value_head = nn.Linear(hidden_size, 1)

        # 初始化
        self.apply(orthogonal_init)
        if self.is_discrete:
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward backbone."""
        return self.backbone(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s)."""
        h = self.forward_features(x)
        return self.value_head(h).squeeze(-1)

    def get_dist(self, x: torch.Tensor):
        """Get action distribution given observation."""
        h = self.forward_features(x)
        if self.is_discrete:
            logits = self.policy_head(h)
            return Categorical(logits=logits)
        else:
            mean = self.policy_head(h)
            return SquashedDiagGaussian(mean, self.log_std, self.act_low, self.act_high)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) action, log_prob, entropy and value.

        中文说明：
            - 训练时若未提供 action，则从分布抽样；
            - 更新阶段会传入历史动作，计算 log_prob 和 value 以构造损失。
        """
        dist = self.get_dist(x)
        value = self.get_value(x)

        if action is None:
            if self.is_discrete:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            else:
                action, log_prob = dist.sample()
                entropy = dist.entropy()
        else:
            if self.is_discrete:
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            else:
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

        return action, log_prob, entropy, value
