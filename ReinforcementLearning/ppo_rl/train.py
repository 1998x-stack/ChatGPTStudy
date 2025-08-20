# -*- coding: utf-8 -*-
"""PPO training entrypoint.

中文说明：
    - 训练主循环：采样 → 计算 GAE → PPO 更新 → 日志/评估/保存。
    - 支持离散与连续动作空间。
"""

from __future__ import annotations

import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from .configs import PPOConfig, check_config
from .envs import make_vec_envs
from .networks import ActorCritic
from .storage import RolloutBuffer
from .algo import PPOAgent
from .utils import set_seed, linear_schedule, ensure_dir, explain_shape
from .logger import CSVLogger


def evaluate_policy(env_id: str, model: ActorCritic, episodes: int, seed: int) -> float:
    """Evaluate policy with greedy actions (mean for continuous)."""
    env = gym.make(env_id)
    env.reset(seed=seed + 10_000)
    returns = []
    for _ in range(episodes):
        done, total_r = False, 0.0
        obs, _ = env.reset()
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = model.get_dist(obs_t)
                if model.is_discrete:
                    action = torch.argmax(dist.logits if hasattr(dist, "logits") else dist.probs, dim=-1)
                    act = action.item()
                else:
                    # 连续：取均值并做 squashing & scaling
                    mean = model.policy_head(model.forward_features(obs_t))
                    # 利用连续分布封装进行反向映射
                    # 这里直接用 mean 通过 tanh 和缩放（略去 log prob）
                    y = torch.tanh(mean)
                    act = ((y + 1) / 2 * (model.act_high - model.act_low) + model.act_low).squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            total_r += float(r)
        returns.append(total_r)
    env.close()
    return float(np.mean(returns))


def train(cfg: PPOConfig) -> None:
    """Main training loop for PPO."""
    check_config(cfg)
    set_seed(cfg.seed)
    ensure_dir(cfg.log_dir)

    # 构建向量环境
    envs = make_vec_envs(cfg.env_id, cfg.num_envs, cfg.seed)
    obs, _ = envs.reset(seed=cfg.seed)

    # 模型与存储
    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_space = envs.single_action_space
    if hasattr(action_space, "n"):
        action_shape = ()
    else:
        action_shape = action_space.shape

    device = torch.device(cfg.device)
    model = ActorCritic(obs_dim=obs_dim, action_space=action_space, hidden_size=cfg.hidden_size).to(device)
    buffer = RolloutBuffer(obs_shape, action_shape, cfg.num_envs, cfg.num_steps, device)
    agent = PPOAgent(model, cfg)

    # 日志器
    logger = CSVLogger(cfg.log_dir)
    header = {
        "update": 0, "global_step": 0, "lr": cfg.learning_rate,
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
        "clip_frac": 0.0, "avg_return": 0.0,
    }
    logger.set_headers(header)

    global_step = 0
    num_updates = cfg.total_timesteps // (cfg.num_envs * cfg.num_steps)

    # 主循环
    for update in range(1, num_updates + 1):
        # 学习率退火
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            agent.set_lr(cfg.learning_rate * frac)

        # 采样 rollout
        last_done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
        for step in range(cfg.num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(cfg.num_envs, -1)
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs_t)
            # 将动作转为 numpy
            if model.is_discrete:
                action_np = action.cpu().numpy()
            else:
                action_np = action.cpu().numpy()

            next_obs, reward, terminated, truncated, _ = envs.step(action_np)
            done = np.logical_or(terminated, truncated).astype(np.float32)

            # 写入 buffer
            buffer.add(
                obs=torch.as_tensor(obs, dtype=torch.float32, device=device),
                action=torch.as_tensor(action_np, device=device),
                logprob=logprob.detach(),
                reward=torch.as_tensor(reward, dtype=torch.float32, device=device),
                done=torch.as_tensor(done, dtype=torch.float32, device=device),
                value=value.detach(),
            )

            obs = next_obs
            last_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            global_step += cfg.num_envs

        # 计算 GAE / returns
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(cfg.num_envs, -1)
            _, _, _, last_value = model.get_action_and_value(obs_t)
        advantages, returns = buffer.compute_returns(
            last_value=last_value, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, last_done=last_done
        )

        # 展平 batch
        batch_size = cfg.num_envs * cfg.num_steps
        b_obs = buffer.obs.reshape(batch_size, -1)
        b_actions = buffer.actions.reshape(batch_size, *buffer.actions.shape[2:])
        # 离散动作转 long
        if model.is_discrete:
            b_actions = b_actions.squeeze(-1).long()
        b_logprobs = buffer.logprobs.reshape(-1)
        b_values = buffer.values.reshape(-1)
        b_advantages = advantages
        b_returns = returns

        # PPO 更新
        stats = agent.update(b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values)

        # 评估与日志
        avg_ret = 0.0
        if cfg.eval_interval and update % cfg.eval_interval == 0:
            avg_ret = evaluate_policy(cfg.env_id, model, cfg.eval_episodes, cfg.seed)

        logger.log({
            "update": update,
            "global_step": global_step,
            "lr": agent.optimizer.param_groups[0]["lr"],
            "policy_loss": round(stats.policy_loss, 6),
            "value_loss": round(stats.value_loss, 6),
            "entropy": round(stats.entropy, 6),
            "approx_kl": round(stats.approx_kl, 6),
            "clip_frac": round(stats.clip_frac, 6),
            "avg_return": round(avg_ret, 3),
        })

        # 保存
        if cfg.save_interval and update % cfg.save_interval == 0:
            ensure_dir(cfg.log_dir)
            path = os.path.join(cfg.log_dir, f"ppo_ckpt_{update}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)

    envs.close()
    logger.close()


if __name__ == "__main__":
    # 默认配置：CartPole 离散动作；将 device 调为 'cuda' 可用 GPU
    cfg = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=100_000,
        num_envs=8,
        num_steps=128,
        update_epochs=4,
        num_minibatches=4,
        learning_rate=3e-4,
        device="cpu",
        log_dir="./runs/ppo_cartpole",
        eval_interval=5,
        save_interval=10,
        target_kl=0.02,
    )
    train(cfg)
