# -*- coding: utf-8 -*-
"""Environment factories and wrappers.

中文说明：
    - 提供向量化环境工厂函数，支持 Gymnasium 环境。
    - 包含基础的观测/奖励归一化选项（可扩展）。"""

from __future__ import annotations

import functools
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int, idx: int, capture_video: bool = False) -> Callable[[], gym.Env]:
    """Create a single environment thunk.

    中文说明：
        - 按需创建单个环境实例的 thunk，用于 SyncVectorEnv。
        - 为不同子进程（或索引）设置不同 seed，保证并行采样的随机性。

    Args:
        env_id: Gymnasium environment id.
        seed: Base random seed.
        idx: Environment index.
        capture_video: Not used (placeholder for future video wrapper).

    Returns:
        A callable that returns a gym.Env when invoked.
    """
    def _thunk() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        return env
    return _thunk


def make_vec_envs(env_id: str, num_envs: int, seed: int) -> gym.vector.SyncVectorEnv:
    """Create a vectorized environment.

    中文说明：
        - 使用 SyncVectorEnv 以简化依赖与可复现性。
        - 若追求吞吐量，可替换为 SubprocVectorEnv。

    Args:
        env_id: Gymnasium environment id.
        num_envs: Number of parallel envs.
        seed: Base seed.

    Returns:
        A SyncVectorEnv instance.
    """
    thunks = [make_env(env_id, seed, i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(thunks)