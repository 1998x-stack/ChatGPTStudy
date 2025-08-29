from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from .indexer import UserLastNIndex, ItemTopKIndex


class ItemCFRecommender:
    """ItemCF recommender that scores candidates with like × similarity.

    This component assumes both indices are prepared offline.
    """

    def __init__(
        self,
        user_lastn: UserLastNIndex,
        item_topk: ItemTopKIndex,
        exclude_seen: bool = True,
        score_clip_min: float = -1e9,
        score_clip_max: float = 1e9,
        train_seen: Dict[int, set[int]] | None = None,
    ) -> None:
        """Constructor.

        Args:
            user_lastn: User→last-N items index.
            item_topk: Item→Top-K similar items index.
            exclude_seen: Whether to exclude items seen in train.
            score_clip_min: Minimum clipping for scores.
            score_clip_max: Maximum clipping for scores.
            train_seen: Train seen items per user for exclusion.
        """
        self.user_lastn = user_lastn
        self.item_topk = item_topk
        self.exclude_seen = exclude_seen
        self.clip_min = score_clip_min
        self.clip_max = score_clip_max
        self.train_seen = train_seen or {}

    def recommend(self, user_id: int, topn: int = 100) -> List[Tuple[int, float]]:
        """Recommend items for a user.

        Args:
            user_id: Internal integer user id.
            topn: Number of items in the final list.

        Returns:
            List of (item_id, score) in descending order.
        """
        seed = self.user_lastn.last_items.get(user_id, [])
        if not seed:
            # 没有行为的用户返回空（冷启动可接混合策略）
            logger.debug("User {} has no seed items.", user_id)
            return []

        scores: Dict[int, float] = {}

        # 遍历用户最近的物品，聚合相似物品得分
        for item_i, like_val in seed:
            nbrs = self.item_topk.topk_items.get(item_i, [])
            for item_j, sim in nbrs:
                score = scores.get(item_j, 0.0) + float(like_val) * float(sim)
                # 分数裁剪，避免极端值
                if score < self.clip_min:
                    score = self.clip_min
                elif score > self.clip_max:
                    score = self.clip_max
                scores[item_j] = score

        # 训练中看过的物品不推荐（可配置）
        if self.exclude_seen:
            seen = self.train_seen.get(user_id, set())
            for item in list(scores.keys()):
                if item in seen:
                    del scores[item]

        # 排序取 TopN
        if not scores:
            return []
        pairs = list(scores.items())
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:topn]