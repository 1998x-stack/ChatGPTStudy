# -*- coding: utf-8 -*-
"""PUCT 蒙特卡洛树搜索（AlphaGo Zero 风格）.

- 扩展时使用神经网络给出先验策略与价值
- 无随机 rollout，叶子估值直接使用 value head
- 根节点注入 Dirichlet 噪声增强探索
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import numpy as np
import torch

from go_env import GoEnv, PASS_ACTION


@dataclass
class NodeStats:
    """节点统计量（PUCT 所需）."""
    prior: float       # P(s, a)
    visit_count: int   # N(s, a)
    total_value: float # W(s, a)
    mean_value: float  # Q(s, a) = W / N

    def update(self, value: float) -> None:
        """回传更新统计量."""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class MCTSNode:
    """MCTS 树节点."""
    def __init__(
        self,
        env: GoEnv,
        prior: float,
        parent: Optional["MCTSNode"] = None
    ) -> None:
        self.env = env
        self.parent = parent
        self.player_to_move = env.current_player
        self.stats = NodeStats(prior=prior, visit_count=0, total_value=0.0, mean_value=0.0)
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded: bool = False

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray) -> None:
        """根据先验（策略分布）与合法掩码展开子节点."""
        self.is_expanded = True
        priors = priors.copy()
        priors[~legal_mask] = 0.0
        s = priors.sum()
        if s <= 0.0:
            # 全部非法（极罕见），退化为均匀分布于合法动作
            priors = legal_mask.astype(np.float32)
            s = priors.sum()
        priors /= s
        for action, p in enumerate(priors):
            if p <= 0.0 or not legal_mask[action]:
                continue
            env_child = self.env.clone()
            if action == env_child.N * env_child.N:
                env_child._apply_pass()
            else:
                r, c = divmod(action, env_child.N)
                env_child._apply_move(r, c, env_child.current_player)  # _apply_move 内部会切换 current_player
            self.children[action] = MCTSNode(env_child, prior=float(p), parent=self)

    def is_leaf(self) -> bool:
        return not self.is_expanded


class MCTS:
    """PUCT MCTS 搜索器."""

    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device,
        c_puct: float = 2.0,
        num_simulations: int = 200,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        temperature_moves: int = 10
    ) -> None:
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_moves = temperature_moves

    @torch.no_grad()
    def run(self, root_env: GoEnv) -> Tuple[np.ndarray, float]:
        """从根环境执行若干次模拟，返回访问计数分布与根估值.

        Returns:
            pi: 归一化的访问计数 N(s, a) 分布
            root_value: 根节点的 value 估计（当前执子方视角）
        """
        root = MCTSNode(root_env.clone(), prior=1.0, parent=None)

        # 评估根节点先验与 value
        obs = torch.from_numpy(root.env.features()).unsqueeze(0).to(self.device)  # (1,C,N,N)
        logits, value = self.net(obs)  # logits: (1, A), value: (1, 1)
        policy = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        value_root = float(value.cpu().numpy()[0, 0])

        legal_mask = root.env.legal_actions()
        # 根节点注入 Dirichlet 噪声增强探索
        noise = np.random.dirichlet([self.dirichlet_alpha] * policy.shape[0])
        policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
        root.expand(policy, legal_mask)

        # 多次模拟
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # 1) Selection：沿 UCB 选择
            while not node.is_leaf():
                action = self._select_action(node)
                node = node.children[action]
                path.append(node)

            # 2) Expand & Evaluate
            if node.env._is_terminal():
                leaf_value = self._terminal_value(node)
            else:
                obs = torch.from_numpy(node.env.features()).unsqueeze(0).to(self.device)
                logits, value = self.net(obs)
                policy = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                legal_mask = node.env.legal_actions()
                node.expand(policy, legal_mask)
                leaf_value = float(value.cpu().numpy()[0, 0])

            # 3) Backup：价值沿路径反向传播（对手视角取反）
            for n in reversed(path):
                n.stats.update(leaf_value if n.player_to_move == path[-1].player_to_move else -leaf_value)

        # 访问计数转 pi
        visit_counts = np.zeros(root.env.action_size, dtype=np.float32)
        for a, child in root.children.items():
            visit_counts[a] = child.stats.visit_count

        # 归一化
        if visit_counts.sum() > 0:
            pi = visit_counts / visit_counts.sum()
        else:
            # 极小概率无子可走，退化为 PASS
            pi = np.zeros_like(visit_counts)
            pi[-1] = 1.0

        return pi, value_root

    def _select_action(self, node: MCTSNode) -> int:
        """PUCT 选择：argmax(Q + U)."""
        parent_visits = max(1, node.stats.visit_count)
        best_score = -1e9
        best_action = None
        for a, child in node.children.items():
            q = child.stats.mean_value
            u = self.c_puct * child.stats.prior * math.sqrt(parent_visits) / (1 + child.stats.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a
        assert best_action is not None
        return int(best_action)

    def _terminal_value(self, node: MCTSNode) -> float:
        """终局估值：按当前节点执手方视角返回 [-1, 1]."""
        score = node.env._tromp_taylor_score()
        # 若当前节点执手方是 node.player_to_move，则：
        # score>0 代表黑优势；值函数定义为“当前执手一方视角”
        if node.player_to_move == 1:  # 黑
            return 1.0 if score > 0 else -1.0 if score < 0 else 0.0
        else:  # 白
            return 1.0 if score < 0 else -1.0 if score > 0 else 0.0
