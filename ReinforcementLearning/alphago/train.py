# -*- coding: utf-8 -*-
"""训练主入口：自博弈生成、经验回放、参数更新、日志记录."""

from __future__ import annotations
from typing import Tuple
import os
import random
import numpy as np
import torch
import torch.optim as optim

from config import Config
from go_env import GoEnv
from models import AlphaGoZeroNet
from mcts import MCTS
from replay_buffer import ReplayBuffer
from alphago import AlphaGoAgent
from utils.logger import TrainLogger
from utils.serialization import save_checkpoint, load_checkpoint


def set_seed(seed: int) -> None:
    """设置随机种子，增强可复现性."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = Config()
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # 1) 环境
    env = GoEnv(
        board_size=cfg.board.board_size,
        max_moves=cfg.board.max_moves,
        consecutive_pass_to_end=cfg.board.consecutive_pass_to_end,
        simple_ko=cfg.board.simple_ko,
        allow_suicide=cfg.board.allow_suicide,
        seed=cfg.train.seed,
    )

    # 2) 模型与优化器
    net = AlphaGoZeroNet(
        board_size=cfg.board.board_size,
        channels=cfg.net.channels,
        num_res_blocks=cfg.net.num_res_blocks,
        policy_conv_channels=cfg.net.policy_conv_channels,
        value_conv_channels=cfg.net.value_conv_channels,
        value_hidden=cfg.net.value_hidden,
    ).to(device)

    if cfg.train.print_model_summary:
        dummy = torch.zeros(1, 3, cfg.board.board_size, cfg.board.board_size, device=device)
        p, v = net(dummy)
        print(f"[Model] policy logits shape: {tuple(p.shape)}, value shape: {tuple(v.shape)}")

    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # 3) MCTS
    mcts = MCTS(
        net=net,
        device=device,
        c_puct=cfg.mcts.c_puct,
        num_simulations=cfg.mcts.num_simulations,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
        temperature_moves=cfg.mcts.temperature_moves,
    )

    # 4) Agent + Buffer + Logger
    agent = AlphaGoAgent(env=env, net=net, optimizer=optimizer, device=device, mcts=mcts)
    buffer = ReplayBuffer(capacity=cfg.train.buffer_capacity)
    tlogger = TrainLogger(log_dir=cfg.train.log_dir)

    # 5) 可选加载断点
    start_iter = 0
    ckpt_latest = os.path.join(cfg.train.ckpt_dir, "latest.pt")
    if os.path.exists(ckpt_latest):
        print(f"[INFO] Load checkpoint: {ckpt_latest}")
        start_iter = load_checkpoint(ckpt_latest, model=net, optimizer=optimizer, map_location=str(device))

    global_step = 0

    # 6) 训练循环
    for it in range(start_iter, cfg.train.total_iterations):
        # 6.1 自博弈收集
        wins = 0
        for ep in range(cfg.train.self_play_episodes_per_iter):
            temp = 1.0  # 前若干步温度 1
            states, pis, winner, zs = agent.self_play_episode(temperature=temp)
            # 压入缓存
            for s, pi, z in zip(states, pis, zs):
                buffer.push(s, pi, z)
            if winner == 1:
                wins += 1

        tlogger.scalar("selfplay/buffer_size", len(buffer), it)
        tlogger.scalar("selfplay/black_win_rate", wins / max(1, cfg.train.self_play_episodes_per_iter), it)

        # 6.2 参数更新
        train_losses = []
        policy_losses = []
        value_losses = []
        net.train()
        for _ in range(cfg.train.train_steps_per_iter):
            if len(buffer) < cfg.train.batch_size:
                break
            batch = buffer.sample(cfg.train.batch_size, device=device)
            loss, pl, vl = agent.update(batch)
            train_losses.append(loss)
            policy_losses.append(pl)
            value_losses.append(vl)
            global_step += 1

        # 6.3 日志与保存
        if train_losses:
            tlogger.scalar("train/loss", float(np.mean(train_losses)), it)
            tlogger.scalar("train/policy_loss", float(np.mean(policy_losses)), it)
            tlogger.scalar("train/value_loss", float(np.mean(value_losses)), it)

        save_path = os.path.join(cfg.train.ckpt_dir, "latest.pt")
        save_checkpoint(save_path, net, optimizer, step=it, extra={"global_step": global_step})

        print(f"[Iter {it}] buffer={len(buffer)} "
              f"loss={np.mean(train_losses) if train_losses else None:.4f} "
              f"pl={np.mean(policy_losses) if policy_losses else None:.4f} "
              f"vl={np.mean(value_losses) if value_losses else None:.4f}")

    tlogger.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
