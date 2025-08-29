from __future__ import annotations

from loguru import logger
import sys
from typing import Optional


def init_logger(level: str = "INFO", to_file: Optional[str] = None) -> None:
    """Initialize loguru logger.

    Args:
        level: Log level string.
        to_file: Optional file path to write logs.

    Raises:
        ValueError: If level is invalid or file path is not writable.
    """
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=True, diagnose=False)  # 控制台输出
    if to_file:
        logger.add(to_file, level=level, rotation="10 MB", retention="7 days", enqueue=True)  # 文件输出