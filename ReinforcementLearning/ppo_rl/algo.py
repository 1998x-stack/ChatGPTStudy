# -*- coding: utf-8 -*-
"""PPO algorithm core (update step).

中文说明：
    - 实现 PPO 更新：clip objective、value clipping、entropy bonus、target KL 提前停止等。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .configs import PPOConfig
from .networks import ActorCritic


@dataclass
class PPOUpdateStats:
    """Statistics from a PPO update step."""
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float


class PPOAgent:
    """PPO algorithm wrapper around an ActorCritic model."""

    def __init__(self, model: ActorCritic, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    def set_lr(self, lr: float) -> None:
        """Dynamically update optimizer learning rate (for annealing)."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def update(
        self,
        b_obs: torch.Tensor,
        b_actions: torch.Tensor,
        b_logprobs: torch.Tensor,
        b_advantages: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
    ) -> PPOUpdateStats:
        """Run PPO multiple epochs over minibatches.

        中文说明：
            - 使用打乱的小批量进行 K 轮更新；
            - 计算 approx_kl 与 clip_frac，必要时提前停止本次更新。
        """
        cfg = self.cfg
        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // cfg.num_minibatches

        # 标准化优势（常见 trick）
        advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []

        for epoch in range(cfg.update_epochs):
            # 随机打乱索引
            indices = torch.randperm(batch_size, device=b_obs.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logprobs = b_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_old_values = b_values[mb_idx]

                # 对离散动作做 Long 类型转换
                if mb_actions.dtype != torch.float32 and mb_actions.dim() == 1:
                    mb_actions = mb_actions.long()

                _, new_logprobs, entropy, new_values = self.model.get_action_and_value(mb_obs, mb_actions)

                # ratio
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                # PPO-Clip policy loss
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                # value function loss (with optional clip)
                if cfg.value_clip > 0.0:
                    v_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -cfg.value_clip, cfg.value_clip)
                    v_loss_unclipped = (new_values - mb_returns).pow(2)
                    v_loss_clipped = (v_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
                else:
                    value_loss = 0.5 * torch.mean((new_values - mb_returns).pow(2))

                entropy_loss = torch.mean(entropy)

                # 总损失
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # 统计
                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1) - log_ratio).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_coef).float()).item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

            # target KL 提前停止（防止更新过猛）
            if cfg.target_kl and (sum(approx_kls[-cfg.num_minibatches:]) / cfg.num_minibatches) > cfg.target_kl:
                break

        return PPOUpdateStats(
            policy_loss=float(sum(policy_losses) / max(1, len(policy_losses))),
            value_loss=float(sum(value_losses) / max(1, len(value_losses))),
            entropy=float(sum(entropies) / max(1, len(entropies))),
            approx_kl=float(sum(approx_kls) / max(1, len(approx_kls))),
            clip_frac=float(sum(clip_fracs) / max(1, len(clip_fracs))),
        )
