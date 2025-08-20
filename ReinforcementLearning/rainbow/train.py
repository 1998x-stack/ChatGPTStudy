# -*- coding: utf-8 -*-
"""Train Rainbow DQN and run ablation study on Atari.

Usage:
    python train.py --env ALE/Breakout-v5 --total_frames 2000000 --run rainbow_breakout
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

from loguru import logger

from rainbow.config import AgentExtensions, TrainConfig
from rainbow.runner import ablation_matrix, train_single_run


def parse_args() -> Tuple[TrainConfig, bool, bool]:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Rainbow DQN with ablation.")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5", help="Atari env ID")
    parser.add_argument("--total_frames", type=int, default=2_000_000)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--run", type=str, default="rainbow_exp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablate", action="store_true", help="Run ablation suite.")
    parser.add_argument("--single_ext", type=str, default="", help="One extension set: full/no_distributional/no_multistep/no_per/no_noisy/no_dueling/no_double")
    args = parser.parse_args()

    cfg = TrainConfig(
        env_id=args.env,
        total_frames=args.total_frames,
        run_name=args.run,
        log_dir=args.log_dir,
        device=args.device,
    )
    return cfg, args.ablate, bool(args.single_ext)


def main() -> None:
    cfg, do_ablation, single = parse_args()
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))  # mirror to stdout

    if do_ablation:
        for name, ext in ablation_matrix():
            run_name = f"{cfg.run_name}_{name}"
            cfg_i = TrainConfig(**{**cfg.__dict__, "run_name": run_name})
            train_single_run(cfg_i, ext)
    else:
        # 单一设置（默认：full Rainbow）
        if single:
            mapping = dict(ablation_matrix())
            name = os.environ.get("EXT_NAME")
            if not name:
                raise ValueError("When --single_ext is set, provide EXT_NAME in env to match ablation key.")
            ext = mapping.get(name)
            if ext is None:
                raise ValueError(f"Unknown EXT_NAME={name}.")
            cfg.run_name = f"{cfg.run_name}_{name}"
            train_single_run(cfg, ext)
        else:
            ext = AgentExtensions()  # full
            train_single_run(cfg, ext)


if __name__ == "__main__":
    main()
