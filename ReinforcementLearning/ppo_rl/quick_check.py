# -*- coding: utf-8 -*-
"""Quick correctness sanity check.

中文说明：
    - 以极小配置快速跑几次更新，验证 shape、loss 是否正常下降，是否能学到 > 随机策略的回报。
    - 该脚本可以作为基本的回归测试。
"""

from __future__ import annotations

from ppo_rl.configs import PPOConfig
from ppo_rl.train import train


def main() -> None:
    cfg = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=20_000,  # 小步数快速检验
        num_envs=4,
        num_steps=64,
        update_epochs=4,
        num_minibatches=4,
        learning_rate=3e-4,
        clip_coef=0.2,
        value_clip=0.2,
        target_kl=0.03,
        device="cpu",
        log_dir="./runs/quick_check",
        eval_interval=2,
        save_interval=0,
    )
    train(cfg)


if __name__ == "__main__":
    main()
