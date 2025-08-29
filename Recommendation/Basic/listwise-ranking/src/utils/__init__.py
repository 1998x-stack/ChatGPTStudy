from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    中文注释：
        - 统一设置随机种子，保证可复现；
        - 在 GPU 环境下尽量固定性，但 cuDNN 某些算法仍可能非确定。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False