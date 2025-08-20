# -*- coding: utf-8 -*-
"""全局配置文件（可按需修改）.

此处集中管理棋盘大小、MCTS 参数、训练超参、日志与存储等配置。
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BoardConfig:
    """棋盘与规则相关配置."""
    board_size: int = 9                # 可改 19
    max_moves: int = 9 * 9 * 3         # 防止极端对局过长
    consecutive_pass_to_end: int = 2   # 连续 PASS 次数到达即结束
    simple_ko: bool = True             # 简单打劫：禁止立即复现上一步局面
    allow_suicide: bool = False        # 是否允许自杀（围棋规则通常不允许）


@dataclass(frozen=True)
class MCTSConfig:
    """MCTS 搜索相关配置（PUCT）."""
    num_simulations: int = 200
    c_puct: float = 2.0
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 10        # 前若干步采用温度=1，之后降为接近 0
    virtual_loss: float = 0.0          # 预留并行用（此版本串行，设 0）


@dataclass(frozen=True)
class TrainConfig:
    """训练与优化相关配置."""
    seed: int = 42
    device: str = "cuda"               # "cpu" 或 "cuda"
    total_iterations: int = 50
    self_play_episodes_per_iter: int = 8
    train_steps_per_iter: int = 200
    batch_size: int = 128
    buffer_capacity: int = 100_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./runs/alphago"
    print_model_summary: bool = True


@dataclass(frozen=True)
class NetConfig:
    """神经网络结构相关配置."""
    channels: int = 128
    num_res_blocks: int = 6
    policy_conv_channels: int = 2
    value_conv_channels: int = 2
    value_hidden: int = 256


@dataclass(frozen=True)
class Config:
    """聚合全局配置."""
    board: BoardConfig = BoardConfig()
    mcts: MCTSConfig = MCTSConfig()
    train: TrainConfig = TrainConfig()
    net: NetConfig = NetConfig()
