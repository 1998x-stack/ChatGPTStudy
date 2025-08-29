from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    # 设置全局随机种子，确保结果可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Ensure the given directory exists."""
    # 确保目录存在
    Path(path).mkdir(parents=True, exist_ok=True)


class TBLogger:
    """A thin wrapper around tensorboardX SummaryWriter."""

    def __init__(self, log_dir: str) -> None:
        """Initialize TB logger.

        Args:
            log_dir: Directory to write event files.
        """
        # 初始化 TensorBoardX 写入器
        ensure_dir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag: str, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalars under a main tag.

        Args:
            tag: Main tag name.
            metrics: Dict of metric name to value.
            step: Global step.
        """
        for k, v in metrics.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    # 统计可训练参数个数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)