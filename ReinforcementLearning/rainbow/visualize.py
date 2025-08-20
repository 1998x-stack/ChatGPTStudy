# -*- coding: utf-8 -*-
"""Aggregate TensorBoard logs and visualize ablation comparisons."""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _extract_scalars(event_dir: str, tag: str) -> List[Tuple[int, float]]:
    """Extract scalar tag from TensorBoard event files."""
    event_files = glob.glob(os.path.join(event_dir, "events.*"))
    if not event_files:
        return []
    # 读取最新的一个event文件
    event = EventAccumulator(event_dir)
    event.Reload()
    if tag not in event.Tags().get("scalars", []):
        return []
    scalars = event.Scalars(tag)
    return [(s.step, s.value) for s in scalars]


def compare_runs(run_dirs: Dict[str, str], tag: str, out_png: str) -> None:
    """Plot tag curves for multiple runs.

    Args:
        run_dirs: Mapping name -> tb_dir
        tag: Scalar tag path in TB (e.g., 'eval/avg_return').
        out_png: Output figure path.
    """
    plt.figure(figsize=(10, 6))
    for name, tb_dir in run_dirs.items():
        points = _extract_scalars(tb_dir, tag)
        if not points:
            logger.warning(f"No data for {name} at {tag}")
            continue
        steps, vals = zip(*points)
        steps = np.array(steps)
        vals = np.array(vals)
        # 简单平滑
        if len(vals) > 5:
            window = min(21, max(5, len(vals) // 10))
            vals = pd.Series(vals).rolling(window, min_periods=1, center=True).mean().values
        plt.plot(steps, vals, label=name)
    plt.xlabel("Steps")
    plt.ylabel(tag)
    plt.title(f"Ablation Comparison: {tag}")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    logger.info(f"Saved figure to {out_png}")


def main() -> None:
    # 使用方式：将各个 run 的目录映射到名称（可手工编辑或按规则搜集）
    # 例如：runs/rainbow_exp_rainbow_full, runs/rainbow_exp_no_per, ...
    base = "runs"
    candidates = glob.glob(os.path.join(base, "*"))
    mapping = {}
    for d in candidates:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "train.log")):
            name = os.path.basename(d).replace("rainbow_exp_", "")
            mapping[name] = d

    if not mapping:
        logger.error("No run directories with train.log found under 'runs/'.")
        return

    # 对比 eval/avg_return
    compare_runs(mapping, "eval/avg_return", os.path.join("plots", "ablation_avg_return.png"))
    # 对比 charts/episode_reward
    compare_runs(mapping, "charts/episode_reward", os.path.join("plots", "ablation_episode_reward.png"))


if __name__ == "__main__":
    main()