# -*- coding: utf-8 -*-
"""Rainbow DQN Agent with 6 pluggable extensions."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import AgentExtensions, TrainConfig
from .networks import RainbowQNetwork
from .replay import NStepBuffer, PrioritizedReplayBuffer, UniformReplayBuffer
from .utils import (
    compute_kl_div,
    hard_update,
    linear_schedule,
    projection_distribution,
)


class RainbowAgent:
    """Rainbow DQN Agent supporting pluggable extensions."""

    def __init__(self, obs_shape, num_actions: int, cfg: TrainConfig, ext: AgentExtensions):
        ext.validate()
        cfg.validate(ext)

        self.cfg = cfg
        self.ext = ext
        self.num_actions = num_actions

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # 中文：构造网络 (online / target)
        self.q_net = RainbowQNetwork(
            num_actions=num_actions,
            frame_stack=cfg.frame_stack,
            dueling=ext.use_dueling,
            noisy=ext.use_noisy_nets,
            distributional=ext.use_distributional,
            atoms=cfg.atoms,
        ).to(self.device)
        self.target_q_net = RainbowQNetwork(
            num_actions=num_actions,
            frame_stack=cfg.frame_stack,
            dueling=ext.use_dueling,
            noisy=ext.use_noisy_nets,
            distributional=ext.use_distributional,
            atoms=cfg.atoms,
        ).to(self.device)
        hard_update(self.target_q_net, self.q_net)

        # 中文：优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

        # 中文：分布式支持
        if self.ext.use_distributional:
            self.support = torch.linspace(cfg.vmin, cfg.vmax, cfg.atoms, device=self.device)

        # 中文：n步缓存
        self.n_buffer = NStepBuffer(cfg.n_step if ext.use_n_step else 1, cfg.gamma)

        # 中文：经验回放
        if ext.use_per:
            self.replay = PrioritizedReplayBuffer(
                capacity=cfg.buffer_size,
                alpha=cfg.per_alpha,
                beta0=cfg.per_beta0,
                beta_steps=cfg.per_beta_steps,
            )
        else:
            self.replay = UniformReplayBuffer(capacity=cfg.buffer_size)

        # 中文：epsilon-greedy参数（当不使用NoisyNets）
        self.eps_start = cfg.epsilon_start
        self.eps_final = cfg.epsilon_final
        self.eps_decay = cfg.epsilon_decay_frames

        # 统计
        self.train_step = 0

    def act(self, obs4: np.ndarray, global_step: int) -> int:
        """Select action given stacked observation."""
        if not self.ext.use_noisy_nets:
            eps = linear_schedule(self.eps_start, self.eps_final, global_step, self.eps_decay)
            if np.random.rand() < eps:
                return np.random.randint(self.num_actions)

        # 中文：网络前向，选择期望Q最大动作
        with torch.no_grad():
            x = torch.from_numpy(obs4.transpose(2, 0, 1)).unsqueeze(0).to(self.device)  # (1,4,84,84)
            q_out = self.q_net(x)
            if self.ext.use_distributional:
                q_vals = self.q_net.q_values_from_dist(q_out, self.support)
            else:
                q_vals = q_out
            action = int(torch.argmax(q_vals, dim=1).item())
        return action

    def store(self, obs, action, reward, next_obs, done) -> None:
        """Store transition using n-step if enabled."""
        self.n_buffer.push(obs, action, reward, done)
        # 中文：当n步缓存可弹出时，转换为n步transition存入回放
        if self.n_buffer.can_pop():
            s0, a0, Rn, sN, doneN = self.n_buffer.pop(next_obs)
            self.replay.add(s0, a0, Rn, sN, doneN)

    def update(self, global_step: int) -> Dict[str, float]:
        """One optimization step if ready."""
        self.train_step += 1
        if len(self.replay) < self.cfg.learning_starts:
            return {}

        batch, idxs, priorities, weights = self.replay.sample(self.cfg.batch_size, global_step)
        obs = torch.from_numpy(batch["obs"].transpose(0, 3, 1, 2)).float().to(self.device)  # (B,4,84,84)
        next_obs = torch.from_numpy(batch["next_obs"].transpose(0, 3, 1, 2)).float().to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)
        isw = torch.from_numpy(weights).float().to(self.device)  # (B,)

        if self.ext.use_distributional:
            # 中文：分布式目标构造
            with torch.no_grad():
                next_dist = self.q_net(next_obs)  # (B,A,Z)
                target_next_dist = self.target_q_net(next_obs)  # (B,A,Z)

                if self.ext.use_double_q:
                    q_vals_online = self.q_net.q_values_from_dist(next_dist, self.support)  # (B,A)
                    a_star = torch.argmax(q_vals_online, dim=1)  # (B,)
                else:
                    q_vals_target = self.target_q_net.q_values_from_dist(target_next_dist, self.support)
                    a_star = torch.argmax(q_vals_target, dim=1)

                next_prob = target_next_dist[torch.arange(next_obs.size(0)), a_star]  # (B,Z)
                gamma_n = self.cfg.gamma ** (self.cfg.n_step if self.ext.use_n_step else 1)
                target_proj = projection_distribution(
                    next_support=self.support.unsqueeze(0).expand_as(next_prob),
                    next_prob=next_prob,
                    rewards=rewards,
                    dones=dones,
                    gamma=gamma_n,
                    vmin=self.cfg.vmin,
                    vmax=self.cfg.vmax,
                    atoms=self.cfg.atoms,
                )  # (B,Z)

            # 当前分布
            dist = self.q_net(obs)  # (B,A,Z)
            logp = torch.log(dist[torch.arange(obs.size(0)), actions].clamp(min=1e-6))  # (B,Z)

            loss_per = -(target_proj * logp).sum(dim=1)  # cross-entropy ~ KL
            loss = (loss_per * isw).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.gradient_clip_norm)
            self.optimizer.step()

            # 中文：更新优先级
            if self.ext.use_per:
                if self.ext.per_priority_type == "kl":
                    prio = loss_per.detach().cpu().numpy() + 1e-6
                else:  # abs_td fallback: 以期望差近似
                    with torch.no_grad():
                        q_curr = self.q_net.q_values_from_dist(dist, self.support)
                        q_next = (target_proj * self.support).sum(dim=1)
                        td_abs = (q_next - q_curr[torch.arange(obs.size(0)), actions]).abs()
                        prio = td_abs.detach().cpu().numpy() + 1e-6
                self.replay.update_priorities(idxs, prio)

            # 目标网络同步
            if self.train_step % self.cfg.target_update_interval == 0:
                hard_update(self.target_q_net, self.q_net)

            return {"loss": float(loss.item()), "loss_mean": float(loss_per.mean().item())}

        # ===== 非分布式分支（用于消融） =====
        with torch.no_grad():
            q_next_online = self.q_net(next_obs)  # (B,A)
            q_next_target = self.target_q_net(next_obs)  # (B,A)
            if self.ext.use_double_q:
                a_star = torch.argmax(q_next_online, dim=1)
                q_next = q_next_target[torch.arange(actions.size(0)), a_star]
            else:
                q_next = q_next_target.max(dim=1).values
            gamma_n = self.cfg.gamma ** (self.cfg.n_step if self.ext.use_n_step else 1)
            target = rewards + (1.0 - dones) * gamma_n * q_next  # (B,)

        q_curr = self.q_net(obs)[torch.arange(actions.size(0)), actions]
        td_error = target - q_curr
        loss_per = torch.nn.functional.smooth_l1_loss(q_curr, target, reduction="none")
        loss = (loss_per * isw).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.gradient_clip_norm)
        self.optimizer.step()

        if self.ext.use_per:
            prio = td_error.detach().abs().cpu().numpy() + 1e-6
            self.replay.update_priorities(idxs, prio)

        if self.train_step % self.cfg.target_update_interval == 0:
            hard_update(self.target_q_net, self.q_net)

        return {"loss": float(loss.item()), "loss_mean": float(loss_per.mean().item())}
