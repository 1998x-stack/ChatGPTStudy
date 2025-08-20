# -*- coding: utf-8 -*-
"""Noisy Linear layer with factorized Gaussian noise (Fortunato et al., 2017)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _f_epsilon(x: torch.Tensor) -> torch.Tensor:
    """Factorized noise helper: sign(x) * sqrt(|x|)."""
    return x.sign() * torch.sqrt(x.abs())


class NoisyLinear(nn.Module):
    """Factorized Gaussian Noisy Linear layer.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        sigma0: Initial std coefficient.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features/out_features must be positive.")
        self.in_features = in_features
        self.out_features = out_features

        # 中文：可学习的权重与偏置（均值项）
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 中文：因子化噪声向量（不作为参数）
        self.register_buffer("eps_in", torch.zeros(1, in_features))
        self.register_buffer("eps_out", torch.zeros(out_features, 1))

        # 中文：初始化
        mu_range = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(sigma0 / math.sqrt(in_features))
        self.bias_sigma.data.fill_(sigma0 / math.sqrt(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fresh noise each time."""
        # 中文：因子化噪声采样，并构造噪声矩阵
        eps_in = _f_epsilon(torch.randn_like(self.eps_in))
        eps_out = _f_epsilon(torch.randn_like(self.eps_out))
        weight_epsilon = eps_out @ eps_in  # (out, in)
        bias_epsilon = eps_out.squeeze(1)  # (out,)

        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        bias = self.bias_mu + self.bias_sigma * bias_epsilon
        return F.linear(x, weight, bias)
