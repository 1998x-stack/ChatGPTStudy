# -*- coding: utf-8 -*-
"""Training/Evaluation runner, ablation scaffolding and logging."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter

from .agent import RainbowAgent
from .config import AgentExtensions, TrainConfig
from .utils import set_global_seeds
from .wrappers import make_atari_env


def evaluate_agent(
    env_id: str,
    agent: RainbowAgent,
    episodes: int,
    frame_stack: int,
    clip_reward: bool,
    seed: int,
) -> Tuple[float, float]:
    """Run evaluation episodes and return average reward and length."""
    env = make_atari_env(env_id, frame_stack=frame_stack, clip_reward=clip_reward)
    env.reset(seed=seed + 999)  # eval seed offset
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.act(obs, global_step=0)  # eval无需epsilon退火
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


def train_single_run(cfg: TrainConfig, ext: AgentExtensions) -> Dict[str, float]:
    """Train Rainbow on a single env with given extensions and return summary."""
    set_global_seeds(cfg.seed)
    # 中文：日志与目录
    run_dir = os.path.join(cfg.log_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    logger.add(os.path.join(run_dir, "train.log"), rotation="10 MB")

    logger.info(f"Run name: {cfg.run_name}")
    logger.info(f"Extensions: {ext}")
    logger.info(f"Config: {cfg}")

    env = make_atari_env(cfg.env_id, noop_max=cfg.max_noop, frame_stack=cfg.frame_stack, clip_reward=cfg.clip_reward)
    obs, info = env.reset(seed=cfg.seed)

    agent = RainbowAgent(
        obs_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        cfg=cfg,
        ext=ext,
    )

    episode_reward = 0.0
    episode_len = 0
    ep_count = 0

    best_eval = -float("inf")
    start_time = time.time()

    for global_step in range(1, cfg.total_frames + 1):
        action = agent.act(obs, global_step)

        next_obs, reward, done, truncated, info = env.step(action)
        episode_reward += float(reward)
        episode_len += 1

        agent.store(obs, action, float(reward), next_obs, bool(done))
        obs = next_obs

        # 中文：回合完成，记录日志与重置
        if done or truncated:
            ep_count += 1
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/episode_length", episode_len, global_step)
            logger.info(f"Step={global_step} Episode#{ep_count} Reward={episode_reward:.1f} Len={episode_len}")

            obs, info = env.reset()
            episode_reward = 0.0
            episode_len = 0

        # 中文：按频率进行一次优化（学习起步后）
        if global_step > cfg.learning_starts and global_step % cfg.train_freq == 0:
            stat = agent.update(global_step)
            if stat:
                writer.add_scalar("loss/loss", stat["loss"], global_step)
                writer.add_scalar("loss/loss_mean", stat["loss_mean"], global_step)

        # 中文：评估
        if global_step % cfg.eval_interval == 0:
            avg_ret, avg_len = evaluate_agent(
                env_id=cfg.env_id,
                agent=agent,
                episodes=cfg.eval_episodes,
                frame_stack=cfg.frame_stack,
                clip_reward=cfg.clip_reward,
                seed=cfg.seed,
            )
            writer.add_scalar("eval/avg_return", avg_ret, global_step)
            writer.add_scalar("eval/avg_length", avg_len, global_step)
            logger.info(f"[EVAL] Step={global_step} AvgReturn={avg_ret:.2f} AvgLen={avg_len:.1f}")
            best_eval = max(best_eval, avg_ret)

        # 中文：保存checkpoint
        if global_step % cfg.save_interval == 0 or global_step == cfg.total_frames:
            ckpt_path = os.path.join(run_dir, f"ckpt_{global_step}.pt")
            torch.save(
                {
                    "q_net": agent.q_net.state_dict(),
                    "target_q_net": agent.target_q_net.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "cfg": asdict(cfg),
                    "ext": asdict(ext),
                    "global_step": global_step,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # 重要信息打印
        if global_step % 50_000 == 0:
            elapsed = time.time() - start_time
            fps = global_step / max(elapsed, 1e-6)
            logger.info(f"Progress: {global_step}/{cfg.total_frames} ({100.0 * global_step/cfg.total_frames:.1f}%), FPS≈{fps:.1f}")

    env.close()
    writer.close()
    return {"best_eval": best_eval}


def ablation_matrix() -> List[Tuple[str, AgentExtensions]]:
    """Define a standard Rainbow ablation suite."""
    full = AgentExtensions(True, True, True, True, True, True, "kl")
    variants = [
        ("rainbow_full", full),
        ("no_distributional", AgentExtensions(True, True, True, True, False, True, "abs_td")),
        ("no_multistep", AgentExtensions(True, True, True, False, True, True, "kl")),
        ("no_per", AgentExtensions(True, False, True, True, True, True, "kl")),
        ("no_noisy", AgentExtensions(True, True, True, True, True, False, "kl")),
        ("no_dueling", AgentExtensions(True, True, False, True, True, True, "kl")),
        ("no_double", AgentExtensions(False, True, True, True, True, True, "kl")),
    ]
    return variants
