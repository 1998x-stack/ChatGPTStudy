# -*- coding: utf-8 -*-
"""Q-network architectures for Rainbow DQN."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noisy import NoisyLinear


class FeatureExtractor(nn.Module):
    """Atari conv feature extractor.

    Input: (N, C=4, H=84, W=84) -> (N, 3136) with classic DQN conv stack.
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        # 中文：经典DQN卷积提取器
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0  # 中文：归一化到[0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)  # (N, 3136)


class DQNHead(nn.Module):
    """Q-value head supporting (dueling vs non-dueling) and (noisy vs linear)."""

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        dueling: bool = True,
        noisy: bool = True,
        hidden: int = 512,
    ):
        super().__init__()
        if in_features <= 0 or num_actions <= 0:
            raise ValueError("in_features/num_actions must be positive.")
        self.dueling = dueling
        self.noisy = noisy

        Linear = NoisyLinear if noisy else nn.Linear

        if dueling:
            # 中文：决斗结构，分别估计V和A
            self.value_fc = Linear(in_features, hidden)
            self.value_out = Linear(hidden, 1)

            self.adv_fc = Linear(in_features, hidden)
            self.adv_out = Linear(hidden, num_actions)
        else:
            self.fc = Linear(in_features, hidden)
            self.out = Linear(hidden, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            v = F.relu(self.value_fc(features))
            v = self.value_out(v)  # (N, 1)
            a = F.relu(self.adv_fc(features))
            a = self.adv_out(a)  # (N, A)
            q = v + a - a.mean(dim=1, keepdim=True)
            return q
        out = F.relu(self.fc(features))
        return self.out(out)


class C51Head(nn.Module):
    """Distributional head (C51) with optional dueling/noisy."""

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        atoms: int,
        dueling: bool = True,
        noisy: bool = True,
        hidden: int = 512,
    ):
        super().__init__()
        if atoms <= 1:
            raise ValueError("atoms must be > 1 for C51.")
        self.atoms = atoms
        self.num_actions = num_actions
        self.dueling = dueling
        self.noisy = noisy

        Linear = NoisyLinear if noisy else nn.Linear

        if dueling:
            # 中文：分布版本的决斗结构，V输出为atoms，A输出为A*atoms
            self.value_fc = Linear(in_features, hidden)
            self.value_out = Linear(hidden, atoms)

            self.adv_fc = Linear(in_features, hidden)
            self.adv_out = Linear(hidden, num_actions * atoms)
        else:
            self.fc = Linear(in_features, hidden)
            self.out = Linear(hidden, num_actions * atoms)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            v = F.relu(self.value_fc(features))
            v = self.value_out(v).view(-1, 1, self.atoms)  # (N,1,Z)

            a = F.relu(self.adv_fc(features))
            a = self.adv_out(a).view(-1, self.num_actions, self.atoms)  # (N,A,Z)
            a = a - a.mean(dim=1, keepdim=True)
            logits = v + a  # (N,A,Z)
        else:
            out = F.relu(self.fc(features))
            logits = self.out(out).view(-1, self.num_actions, self.atoms)
        probs = F.softmax(logits, dim=2)  # (N,A,Z)
        return probs


class RainbowQNetwork(nn.Module):
    """Rainbow Q-Network wrapper selecting scalar or distributional head."""

    def __init__(
        self,
        num_actions: int,
        frame_stack: int = 4,
        dueling: bool = True,
        noisy: bool = True,
        distributional: bool = True,
        atoms: int = 51,
    ):
        super().__init__()
        self.feature = FeatureExtractor(in_channels=frame_stack)
        if distributional:
            self.head = C51Head(
                in_features=3136,
                num_actions=num_actions,
                atoms=atoms,
                dueling=dueling,
                noisy=noisy,
            )
            self.distributional = True
            self.atoms = atoms
        else:
            self.head = DQNHead(
                in_features=3136,
                num_actions=num_actions,
                dueling=dueling,
                noisy=noisy,
            )
            self.distributional = False
            self.atoms = 1
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            If distributional=True: (N, A, Z) probabilities.
            Else: (N, A) Q-values.
        """
        feats = self.feature(x)
        out = self.head(feats)
        return out

    def q_values_from_dist(self, probs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute Q-values as expectation over distribution (N,A)."""
        assert probs.dim() == 3, "probs must be (N,A,Z)."
        assert support.dim() == 1 and support.numel() == probs.size(2)
        return torch.tensordot(probs, support, dims=([2], [0]))  # (N,A)
