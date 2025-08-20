# -*- coding: utf-8 -*-
"""AlphaGo Zero 风格共享干 + 策略头 + 价值头（PyTorch）."""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """标准 ResBlock：Conv-BN-ReLU-Conv-BN + 残差连接."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaGoZeroNet(nn.Module):
    """共享干网络 + 策略头 + 价值头."""

    def __init__(
        self,
        board_size: int,
        channels: int = 128,
        num_res_blocks: int = 6,
        policy_conv_channels: int = 2,
        value_conv_channels: int = 2,
        value_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1

        # 输入 3 通道（我方、对方、执手平面）
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_conv_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_conv_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_conv_channels * board_size * board_size, self.action_size)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_conv_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_conv_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(value_conv_channels * board_size * board_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算.

        Args:
            x: 形状 (B, 3, N, N)

        Returns:
            policy_logits: (B, A)
            value: (B, 1), tanh 限幅到 [-1, 1]
        """
        b = self.stem(x)
        b = self.resblocks(b)

        p = self.policy_head(b)
        p = p.reshape(p.shape[0], -1)
        policy_logits = self.policy_fc(p)

        v = self.value_head(b)
        v = v.reshape(v.shape[0], -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return policy_logits, v
