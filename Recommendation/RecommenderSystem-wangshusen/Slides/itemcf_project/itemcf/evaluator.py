from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from loguru import logger


def hit_rate_at_k(reco_items: Sequence[int], pos_items: Sequence[int], k: int) -> float:
    """HitRate@K for a single user."""
    topk = set(reco_items[:k])
    pos = set(pos_items)
    return 1.0 if topk & pos else 0.0


def recall_at_k(reco_items: Sequence[int], pos_items: Sequence[int], k: int) -> float:
    """Recall@K for a single user (for implicit positives)."""
    topk = set(reco_items[:k])
    pos = set(pos_items)
    if not pos:
        return 0.0
    return len(topk & pos) / float(len(pos))


def ndcg_at_k(reco_items: Sequence[int], pos_items: Sequence[int], k: int) -> float:
    """NDCG@K for a single user with binary relevance."""
    topk = reco_items[:k]
    pos = set(pos_items)
    dcg = 0.0
    for idx, it in enumerate(topk, start=1):
        if it in pos:
            dcg += 1.0 / math.log2(idx + 1)
    # ideal DCG
    ideal_hits = min(len(pos), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_all(
    user_ids: List[int],
    recommender,
    test_pos: Dict[int, List[int]],
    ks: Tuple[int, ...] = (10, 20, 50, 100),
) -> Dict[int, Dict[str, float]]:
    """Evaluate recommender with HitRate/Recall/NDCG for a list of users.

    Args:
        user_ids: Users to evaluate.
        recommender: ItemCFRecommender instance.
        test_pos: Positive test items per user.
        ks: Cutoffs.

    Returns:
        Dict mapping K -> metrics dict (avg over users).
    """
    # 累积器
    acc = {k: {"HitRate": 0.0, "Recall": 0.0, "NDCG": 0.0} for k in ks}
    valid_users = 0

    for u in user_ids:
        if u not in test_pos or len(test_pos[u]) == 0:
            continue
        reco_pairs = recommender.recommend(u, topn=max(ks))
        reco_items = [it for it, _ in reco_pairs]
        pos_items = test_pos[u]
        if not reco_items:
            continue
        valid_users += 1
        for k in ks:
            acc[k]["HitRate"] += hit_rate_at_k(reco_items, pos_items, k)
            acc[k]["Recall"] += recall_at_k(reco_items, pos_items, k)
            acc[k]["NDCG"] += ndcg_at_k(reco_items, pos_items, k)

    if valid_users == 0:
        logger.warning("No valid users for evaluation.")
        return acc

    for k in ks:
        for name in ["HitRate", "Recall", "NDCG"]:
            acc[k][name] /= float(valid_users)

    return acc