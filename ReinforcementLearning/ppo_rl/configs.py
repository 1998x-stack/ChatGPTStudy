# -*- coding: utf-8 -*-
"""PPO configuration dataclasses.

This module defines configuration objects for PPO training.

中文说明：
    - 该模块提供 PPO 训练的配置数据类，包含超参数、环境、日志、硬件等设置。
    - 提供基本的配置校验函数以避免常见边界错误（如 batch 划分不整除）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """Configuration for PPO training.

    Attributes:
        env_id: Gymnasium environment ID (e.g., 'CartPole-v1', 'Pendulum-v1').
        total_timesteps: Total environment steps to train.
        num_envs: Parallel environments for vectorized rollout.
        num_steps: Steps per environment per PPO rollout (T).
        update_epochs: PPO update epochs per rollout (K).
        num_minibatches: Number of minibatches per update (M).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_coef: PPO clip epsilon.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping norm (0 = disable).
        learning_rate: Optimizer learning rate.
        anneal_lr: Whether to linearly anneal LR to 0.
        value_clip: Value function clipping epsilon (0 = disable).
        target_kl: Early stop if approx KL exceeds this (<=0 disables).
        seed: Random seed.
        device: Torch device string ('cpu', 'cuda', etc.).
        log_dir: Directory for logs and checkpoints.
        save_interval: Save model every N updates (0 disables).
        eval_interval: Evaluate every N updates (0 disables).
        eval_episodes: Episodes per evaluation.
    """

    env_id: str = "CartPole-v1"
    total_timesteps: int = 200_000
    num_envs: int = 8
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    value_clip: float = 0.2
    target_kl: float = 0.0
    seed: int = 42
    device: str = "cpu"
    log_dir: str = "./runs/ppo"
    save_interval: int = 10
    eval_interval: int = 10
    eval_episodes: int = 5

    # 可选：观测归一化、奖励归一化（工业落地可能需要）
    obs_norm: bool = False
    rew_norm: bool = False

    # 可选：网络宽度
    hidden_size: int = 64


def check_config(cfg: PPOConfig) -> None:
    """Validate config and raise AssertionError if invalid.

    中文说明：
        - 对配置做基本完整性检查，避免训练过程中 shape 不匹配或 batch 划分错误。

    Args:
        cfg: PPOConfig instance.

    Raises:
        AssertionError: If invalid settings detected.
    """
    assert cfg.total_timesteps > 0
    assert cfg.num_envs > 0 and cfg.num_steps > 0
    batch_size = cfg.num_envs * cfg.num_steps
    assert cfg.num_minibatches > 0 and batch_size % cfg.num_minibatches == 0, (
        "num_envs * num_steps 必须能整除 num_minibatches，"
        "以便构造等大小的 minibatch"
    )
    assert 0.0 < cfg.gamma <= 0.999999, "gamma 合理范围 (0, 1)"
    assert 0.0 <= cfg.gae_lambda <= 1.0
    assert cfg.clip_coef >= 0.0
    assert cfg.learning_rate > 0.0
    assert cfg.update_epochs > 0
    if cfg.target_kl < 0:
        raise AssertionError("target_kl 不能为负数")
