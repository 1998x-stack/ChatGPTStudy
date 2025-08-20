# -*- coding: utf-8 -*-
"""日志与可视化记录：loguru + tensorboardX."""

from __future__ import annotations
from typing import Optional
from loguru import logger
from tensorboardX import SummaryWriter
import os
import sys
import time


class TrainLogger:
    """训练日志管理器."""

    def __init__(self, log_dir: str, stdout_level: str = "INFO") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        logger.remove()
        logger.add(sys.stdout, level=stdout_level)
        logger.add(os.path.join(log_dir, f"train_{int(time.time())}.log"), level="DEBUG", rotation="10 MB")

    def scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量."""
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """关闭资源."""
        self.writer.flush()
        self.writer.close()
