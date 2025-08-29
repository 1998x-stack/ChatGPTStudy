from __future__ import annotations
from typing import Dict, Tuple
import torch
from ..metrics.ranking import ndcg_at_k, mrr


def evaluate_batch(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, topk: Tuple[int, ...]) -> Dict[str, float]:
    out = {}
    for k in topk:
        out[f"NDCG@{k}"] = ndcg_at_k(labels, scores, mask, k)
    out["MRR"] = mrr(labels, scores, mask)
    return out