# -*- coding: utf-8 -*-
"""Utilities: seeding, schedules, math helpers."""

from __future__ import annotations

import os
import random
from typing import Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_schedule(initial_value: float, current_step: int, max_steps: int) -> float:
    """Linear LR annealing from initial_value to 0."""
    frac = 1.0 - (current_step / max_steps)
    return max(initial_value * frac, 0.0)


def explain_shape(x: torch.Tensor, name: str) -> str:
    """Helper to pretty-print a tensor shape for debugging."""
    return f"{name}.shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"


def ensure_dir(path: str) -> None:
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)
