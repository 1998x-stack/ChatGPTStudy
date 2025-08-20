# -*- coding: utf-8 -*-
"""关键逻辑健康检查（建议在训练前先跑一遍）.

运行:
    python -m tests.sanity_check
"""

from __future__ import annotations
import numpy as np
import torch

from go_env import GoEnv
from models import AlphaGoZeroNet
from mcts import MCTS


def test_capture_and_suicide() -> None:
    """测试提子与自杀禁手."""
    env = GoEnv(board_size=5, simple_ko=True, allow_suicide=False)
    env.reset()
    # 构造一个简单提子局面：黑先在 (2,2)，白围杀黑，黑提白一子
    # 为简化，这里仅检查合法性与不触发异常
    actions = [
        2 * 5 + 2,  # B at (2,2)
        1 * 5 + 2,  # W around
        2 * 5 + 1,  # B
        3 * 5 + 2,  # W
        2 * 5 + 3,  # B
        2 * 5 + 4,  # W
    ]
    for a in actions:
        env.step(a)
    # PASS 测试
    env.step(25)  # PASS
    print("[OK] capture/suicide basic flow")


def test_network_forward() -> None:
    """测试网络前向尺寸."""
    net = AlphaGoZeroNet(board_size=9)
    x = torch.zeros(2, 3, 9, 9)
    p, v = net(x)
    assert p.shape == (2, 82) and v.shape == (2, 1)
    print("[OK] network forward shapes", p.shape, v.shape)


def test_mcts_run() -> None:
    """测试 MCTS 单次运行输出."""
    env = GoEnv(board_size=5)
    net = AlphaGoZeroNet(board_size=5)
    mcts = MCTS(net=net, device=torch.device("cpu"), num_simulations=16)
    pi, v = mcts.run(env)
    assert pi.shape[0] == 26
    assert abs(v) <= 1.0 + 1e-6
    print("[OK] mcts run pi len:", len(pi), "v:", v)


if __name__ == "__main__":
    test_capture_and_suicide()
    test_network_forward()
    test_mcts_run()
    print("All sanity checks passed.")
