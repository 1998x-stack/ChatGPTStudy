# -*- coding: utf-8 -*-
"""围棋环境（9x9/19x19），Tromp-Taylor 计分，简单打劫，自杀禁手.

核心特性：
- 状态表示：numpy int8 矩阵（0=空, 1=黑, -1=白）
- 动作空间：N*N + 1（最后一个为 PASS）
- 规则：提子判定、禁自杀、简单打劫（禁止立刻回到上一局面）、两次连续 PASS 结束
- 计分：Tromp-Taylor（己方棋子数 + 被己方完全包围的空点）

注意：本实现聚焦训练稳定性与可扩展性，未实现超级打劫（superko）。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

PASS_ACTION = -1  # 内部使用，外部暴露为 N*N 索引


@dataclass
class StepResult:
    """一步走子后的结果信息."""
    next_player: int
    done: bool
    captured: int
    info: Dict[str, np.ndarray]


class GoEnv:
    """围棋环境对象."""

    def __init__(
        self,
        board_size: int = 9,
        max_moves: int = 9 * 9 * 3,
        consecutive_pass_to_end: int = 2,
        simple_ko: bool = True,
        allow_suicide: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """初始化环境.

        Args:
            board_size: 棋盘边长 N.
            max_moves: 最多步数（防止极端对局）.
            consecutive_pass_to_end: 连续 PASS 次数达到后结束.
            simple_ko: 是否启用简单打劫.
            allow_suicide: 是否允许自杀（通常 False）.
            seed: 随机种子（用于复现）.
        """
        self.N = board_size
        self.action_size = self.N * self.N + 1  # 最后一维为 PASS
        self.max_moves = max_moves
        self.consecutive_pass_to_end = consecutive_pass_to_end
        self.simple_ko = simple_ko
        self.allow_suicide = allow_suicide

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------------------- 对外主要接口 ----------------------

    def reset(self) -> np.ndarray:
        """重置对局并返回初始状态（玩家视角的特征平面）."""
        self.board = np.zeros((self.N, self.N), dtype=np.int8)
        self.current_player = 1  # 黑先
        self.num_moves = 0
        self.consecutive_passes = 0
        self.prev_board: Optional[np.ndarray] = None  # 简单打劫用
        return self.features()

    def features(self) -> np.ndarray:
        """返回玩家视角的特征平面.

        Returns:
            形状为 (C, N, N) 的特征，其中：
            - C=3：当前玩家的棋子平面、对手棋子平面、全 1 平面（编码当前执子方）
        """
        me = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == -self.current_player).astype(np.float32)
        turn = np.ones_like(me, dtype=np.float32)
        return np.stack([me, opp, turn], axis=0)

    def legal_actions(self) -> np.ndarray:
        """返回当前玩家的合法动作掩码（长度 N*N+1）."""
        mask = np.zeros(self.action_size, dtype=np.bool_)
        for idx in range(self.N * self.N):
            r, c = divmod(idx, self.N)
            if self.board[r, c] != 0:
                continue
            if self._is_legal_move(r, c, self.current_player):
                mask[idx] = True
        # PASS 永远合法
        mask[-1] = True
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作.

        Args:
            action: 0..N*N-1 对应落子点；N*N 对应 PASS.

        Returns:
            next_obs, reward, done, info:
            - reward: 最终结束时，胜方=+1，负方=-1；未终局时为 0
        """
        if action < 0 or action >= self.action_size:
            raise ValueError(f"非法动作索引: {action}")
        if action == self.N * self.N:
            # PASS
            self._apply_pass()
        else:
            r, c = divmod(action, self.N)
            if not self._is_legal_move(r, c, self.current_player):
                raise ValueError(f"非法走子（自杀/打劫/占用）：{(r, c)} by {self.current_player}")
            self._apply_move(r, c, self.current_player)

        self.num_moves += 1
        done = self._is_terminal()
        reward = 0.0
        if done:
            score = self._tromp_taylor_score()
            reward = 1.0 if (score * self.current_player) < 0 else -1.0
            # 注意：此处 reward 是给“刚才下完棋后将要执子的一方”的，
            # 在训练中我们通常不直接使用 step 的 reward，而是用 z 赋值。

        obs = self.features()
        info = {"board": self.board.copy()}
        return obs, reward, done, info

    def clone(self) -> "GoEnv":
        """创建环境深拷贝（用于 MCTS 分支模拟）."""
        env = GoEnv(
            board_size=self.N,
            max_moves=self.max_moves,
            consecutive_pass_to_end=self.consecutive_passes,
            simple_ko=self.simple_ko,
            allow_suicide=self.allow_suicide,
        )
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.num_moves = self.num_moves
        env.consecutive_passes = self.consecutive_passes
        env.prev_board = None if self.prev_board is None else self.prev_board.copy()
        return env

    # ---------------------- 私有工具方法 ----------------------

    def _apply_pass(self) -> None:
        """执行 PASS 动作."""
        self.prev_board = self.board.copy()
        self.consecutive_passes += 1
        self.current_player = -self.current_player

    def _apply_move(self, r: int, c: int, player: int) -> None:
        """在 (r,c) 落子并处理提子/禁手/打劫."""
        self.prev_board = self.board.copy()
        self.board[r, c] = player

        # 提取相邻对手棋群，若无气则提走
        opp = -player
        captured = 0
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == opp:
                if self._count_liberties(nr, nc) == 0:
                    captured += self._remove_group(nr, nc)

        # 检查自杀（如果没有提走对方且己方落子后无气则非法）
        if not self.allow_suicide:
            if captured == 0 and self._count_liberties(r, c) == 0:
                # 回滚（保持接口简洁，实际 step 前已检查合法性）
                self.board = self.prev_board
                raise ValueError("自杀禁手")

        self.consecutive_passes = 0
        self.current_player = -self.current_player

    def _is_legal_move(self, r: int, c: int, player: int) -> bool:
        """判断某一落子是否合法（占用/自杀/打劫）."""
        if self.board[r, c] != 0:
            return False

        # 试落
        snapshot = self.board.copy()
        self.board[r, c] = player

        opp = -player
        captured = 0
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == opp and self._count_liberties(nr, nc) == 0:
                captured += self._remove_group(nr, nc)

        legal = True
        # 自杀检测
        if not self.allow_suicide and captured == 0:
            if self._count_liberties(r, c) == 0:
                legal = False

        # 打劫检测：不能复现上一局面
        if legal and self.simple_ko and self.prev_board is not None:
            if np.array_equal(self.board, self.prev_board):
                legal = False

        # 回滚
        self.board = snapshot
        return legal

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """返回 (r,c) 的上下左右邻接点坐标."""
        res: List[Tuple[int, int]] = []
        if r > 0:
            res.append((r - 1, c))
        if r < self.N - 1:
            res.append((r + 1, c))
        if c > 0:
            res.append((r, c - 1))
        if c < self.N - 1:
            res.append((r, c + 1))
        return res

    def _count_liberties(self, r: int, c: int) -> int:
        """计算 (r,c) 所在棋群的气."""
        color = self.board[r, c]
        if color == 0:
            return 0
        visited = set()
        stack = [(r, c)]
        liberties = 0
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                if self.board[nr, nc] == 0:
                    liberties += 1
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return liberties

    def _remove_group(self, r: int, c: int) -> int:
        """提掉 (r,c) 所在棋群，返回提子数量."""
        color = self.board[r, c]
        to_remove = []
        visited = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            to_remove.append((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                if self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        for rr, cc in to_remove:
            self.board[rr, cc] = 0
        return len(to_remove)

    def _is_terminal(self) -> bool:
        """是否终局（两次连续 PASS 或达到步数上限）."""
        if self.consecutive_passes >= 2:
            return True
        if self.num_moves >= self.max_moves:
            return True
        return False

    def _tromp_taylor_score(self) -> int:
        """Tromp-Taylor 计分（黑为 +，白为 -）."""
        score = int(np.sum(self.board))  # 黑子 +1，白子 -1
        # 计算空点归属
        visited = np.zeros_like(self.board, dtype=np.bool_)
        for r in range(self.N):
            for c in range(self.N):
                if self.board[r, c] != 0 or visited[r, c]:
                    continue
                # BFS 找空域
                empties: List[Tuple[int, int]] = []
                border_colors = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if visited[cr, cc]:
                        continue
                    visited[cr, cc] = True
                    empties.append((cr, cc))
                    for nr, nc in self._neighbors(cr, cc):
                        if self.board[nr, nc] == 0 and not visited[nr, nc]:
                            stack.append((nr, nc))
                        elif self.board[nr, nc] != 0:
                            border_colors.add(self.board[nr, nc])
                if len(border_colors) == 1:
                    color = border_colors.pop()
                    score += color * len(empties)
        return score
