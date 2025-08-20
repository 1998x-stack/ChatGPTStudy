# -*- coding: utf-8 -*-
"""Atari environment wrappers (Gymnasium)."""

from __future__ import annotations

import collections
import cv2
import gymnasium as gym
import numpy as np
from typing import Deque, Tuple


cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    """Sample initial number of no-ops on reset."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for environments that need it."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame with max pooling over last two."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 grayscale."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, 1}."""

    def reward(self, reward):
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """Stack last k frames along channel axis."""

    def __init__(self, env, k: int):
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = collections.deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=2)


def make_atari_env(env_id: str, noop_max: int = 30, frame_stack: int = 4, clip_reward: bool = True):
    """Create a preprocessed Atari environment."""
    # 中文：按NoFrameskip流程封装Atari环境，确保输入尺寸与DQN一致
    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.25, render_mode=None)
    env = NoopResetEnv(env, noop_max=noop_max)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=frame_stack)
    return env
