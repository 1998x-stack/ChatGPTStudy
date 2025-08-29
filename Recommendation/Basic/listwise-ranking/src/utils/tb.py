from __future__ import annotations
from tensorboardX import SummaryWriter
from datetime import datetime
import os


def create_writer(project_root: str, run_name: str | None = None) -> SummaryWriter:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = run_name or f"run_{ts}"
    logdir = os.path.join(project_root, 'runs', name)
    os.makedirs(logdir, exist_ok=True)
    return SummaryWriter(logdir)