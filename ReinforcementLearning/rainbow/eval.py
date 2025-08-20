# -*- coding: utf-8 -*-
"""Evaluate a saved Rainbow checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from loguru import logger

from rainbow.agent import RainbowAgent
from rainbow.config import AgentExtensions, TrainConfig
from rainbow.runner import evaluate_agent
from rainbow.wrappers import make_atari_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Rainbow checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = TrainConfig(**ckpt["cfg"])
    ext = AgentExtensions(**ckpt["ext"])

    env = make_atari_env(cfg.env_id, frame_stack=cfg.frame_stack, clip_reward=cfg.clip_reward)
    agent = RainbowAgent(obs_shape=env.observation_space.shape, num_actions=env.action_space.n, cfg=cfg, ext=ext)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_q_net.load_state_dict(ckpt["target_q_net"])

    avg_ret, avg_len = evaluate_agent(cfg.env_id, agent, args.episodes, cfg.frame_stack, cfg.clip_reward, cfg.seed)
    logger.info(f"[EVAL] Episodes={args.episodes} AvgReturn={avg_ret:.2f} AvgLen={avg_len:.1f}")

    result = {"episodes": args.episodes, "avg_return": avg_ret, "avg_length": avg_len}
    out = os.path.join(os.path.dirname(args.ckpt), "eval_result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved eval to {out}")


if __name__ == "__main__":
    main()
