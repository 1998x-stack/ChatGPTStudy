# -*- coding: utf-8 -*-
"""Simple CSV logger with console printing.

中文说明：
    - 轻量 CSV 日志器，便于训练过程可视化与对比。
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Any

from .utils import ensure_dir


class CSVLogger:
    """CSV logger that also prints to console."""

    def __init__(self, log_dir: str, filename: str = "metrics.csv") -> None:
        ensure_dir(log_dir)
        self.path = os.path.join(log_dir, filename)
        self._writer = None
        self._headers_written = False

    def log(self, metrics: Dict[str, Any]) -> None:
        """Append a row of metrics to CSV and print."""
        if not self._headers_written:
            self._open()
            self._writer.writeheader()
            self._headers_written = True
        self._writer.writerow(metrics)
        print(" | ".join(f"{k}={v}" for k, v in metrics.items()))

    def _open(self) -> None:
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        fieldnames = []
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)

    def close(self) -> None:
        """Close underlying file."""
        if getattr(self, "_file", None):
            self._file.close()

    def set_headers(self, headers: Dict[str, Any]) -> None:
        """Define CSV headers before first log() call."""
        self._open()
        self._writer.fieldnames = list(headers.keys())
        self._writer.writeheader()
        self._headers_written = True
