from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from loguru import logger

from .config import EvalConfig
from .data import ML100KDataModule, UserQuery, create_dataloaders
from .metrics import ndcg_at_k, map_at_k, recall_at_k


class Evaluator:
    """Evaluator for ranking metrics."""

    def __init__(self, model: torch.nn.Module, cfg: EvalConfig) -> None:
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            cfg: Evaluation config.
        """
        self.model = model.to(cfg.device)
        self.cfg = cfg

    @torch.no_grad()
    def evaluate_queries(self, queries: List[UserQuery]) -> Dict[str, float]:
        """Evaluate metrics over a list of queries."""
        self.model.eval()
        ndcgs: List[float] = []
        maps: List[float] = []
        recalls: List[float] = []

        for q in queries:
            device = self.cfg.device
            user = torch.full((len(q.item_ids),), q.user_id, dtype=torch.long, device=device)
            items = torch.tensor(q.item_ids, dtype=torch.long, device=device)
            rels = torch.tensor(q.rels, dtype=torch.float32, device=device)

            scores = self.model(user, items).detach().cpu().numpy()
            rels_np = rels.detach().cpu().numpy()

            ndcgs.append(ndcg_at_k(rels_np, scores, k=self.cfg.ndcg_k))

            # 对 MAP/Recall 使用二值化（>=4 为正例）
            binary = (rels_np >= 4.0).astype(int)
            maps.append(map_at_k(binary, scores, k=self.cfg.ndcg_k))
            recalls.append(recall_at_k(binary, scores, k=self.cfg.ndcg_k))

        def _safe_mean(x: List[float]) -> float:
            return float(np.mean(x)) if len(x) > 0 else 0.0

        results = {
            f"NDCG@{self.cfg.ndcg_k}": _safe_mean(ndcgs),
            f"MAP@{self.cfg.ndcg_k}": _safe_mean(maps),
            f"Recall@{self.cfg.ndcg_k}": _safe_mean(recalls),
        }
        return results