# -*- coding: utf-8 -*-
"""AlphaGo(Zero 风格) 智能体：自博弈、训练、策略选择."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from go_env import GoEnv
from mcts import MCTS
from replay_buffer import ReplayBuffer


class AlphaGoAgent:
    """AlphaGo 风格智能体（单网 + MCTS）."""

    def __init__(
        self,
        env: GoEnv,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mcts: MCTS,
    ) -> None:
        self.env = env
        self.net = net
        self.optimizer = optimizer
        self.device = device
        self.mcts = mcts

    def self_play_episode(self, temperature: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """进行一局自博弈，返回状态、策略目标 pi、赢家.

        - 前若干步使用 temperature=1 采样，后续趋近于贪心
        """
        env = self.env.clone()
        states: List[np.ndarray] = []
        target_pis: List[np.ndarray] = []
        players: List[int] = []

        while True:
            pi, _ = self.mcts.run(env)
            # 温度采样/贪心
            if temperature > 1e-6:
                action = np.random.choice(env.action_size, p=pi)
            else:
                action = int(np.argmax(pi))

            states.append(env.features())          # 存玩家视角状态
            target_pis.append(pi.copy())           # 存目标策略
            players.append(env.current_player)     # 存执手方

            # 执行动作
            _, _, done, _ = env.step(action if action < env.action_size - 1 else env.action_size - 1)
            if done:
                score = env._tromp_taylor_score()
                winner = 1 if score > 0 else -1 if score < 0 else 0
                break

            # 降温：前 X 步温度=1，之后接近 0
            if len(states) >= self.mcts.temperature_moves:
                temperature = 1e-8

        # 生成 z 标签（按每步执手方视角）
        zs: List[float] = []
        for p in players:
            if winner == 0:
                zs.append(0.0)
            else:
                zs.append(1.0 if winner == p else -1.0)
        return states, target_pis, winner, zs

    def update(self, batch) -> Tuple[float, float, float]:
        """一次参数更新：返回 loss, policy_loss, value_loss."""
        obs, pi, z = batch  # obs: (B,3,N,N), pi:(B,A), z:(B,1)
        self.net.train()
        logits, value = self.net(obs)
        # 策略损失：交叉熵（目标为 pi 分布）
        logp = F.log_softmax(logits, dim=-1)
        policy_loss = -(pi * logp).sum(dim=-1).mean()
        # 价值损失：MSE
        value_loss = F.mse_loss(value, z)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
        self.optimizer.step()
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())
