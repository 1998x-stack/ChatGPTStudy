from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW

from .config import TrainConfig
from .data import UserQuery
from .loss import lambdarank_pairwise_loss
from .metrics import ndcg_at_k
from .utils import TBLogger, count_parameters


class Trainer:
    """Trainer for LambdaRank."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model.
            config: Train configuration.
            optimizer: Optional optimizer (default AdamW).
        """
        self.model = model.to(config.device)
        self.cfg = config
        self.optimizer = optimizer or AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.tb = TBLogger(self.cfg.log_dir)
        self.best_val = -1.0
        self.steps_since_improve = 0
        logger.info(f"Trainable params: {count_parameters(self.model):,}")

    def _step_on_query(self, query: UserQuery) -> torch.Tensor:
        """Compute loss for a single user query (list-wise)."""
        device = self.cfg.device
        user = torch.full(
            (len(query.item_ids),), fill_value=query.user_id, dtype=torch.long, device=device
        )
        items = torch.tensor(query.item_ids, dtype=torch.long, device=device)
        rels = torch.tensor(query.rels, dtype=torch.float32, device=device)

        # 模型打分
        scores = self.model(user, items)  # [n]
        # 计算 LambdaRank pairwise loss（按 |ΔNDCG| 加权）
        loss = lambdarank_pairwise_loss(rels, scores, sigma=self.cfg.sigma, k=self.cfg.ndcg_k)
        return loss

    @torch.no_grad()
    def _val_metrics_on_batch(self, batch: List[UserQuery]) -> Dict[str, float]:
        """Compute average NDCG on a batch of queries."""
        self.model.eval()
        ndcgs: List[float] = []
        for q in batch:
            device = self.cfg.device
            user = torch.full((len(q.item_ids),), q.user_id, dtype=torch.long, device=device)
            items = torch.tensor(q.item_ids, dtype=torch.long, device=device)
            rels = torch.tensor(q.rels, dtype=torch.float32, device=device)

            scores = self.model(user, items)
            ndcg = ndcg_at_k(
                true_rels=rels.detach().cpu().numpy(),
                pred_scores=scores.detach().cpu().numpy(),
                k=self.cfg.ndcg_k,
            )
            if not np.isnan(ndcg):
                ndcgs.append(ndcg)
        avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        return {"ndcg@k": avg_ndcg}

    def train(
        self,
        train_loader,
        val_loader,
    ) -> None:
        """Run training with early stopping."""
        global_step = 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                # 对一个 batch 内多个用户查询累积 loss
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                valid_queries = 0
                for q in batch:
                    loss = self._step_on_query(q)
                    if torch.isfinite(loss):
                        batch_loss += loss
                        valid_queries += 1
                if valid_queries == 0:
                    continue
                batch_loss = batch_loss / valid_queries
                batch_loss.backward()

                # 梯度裁剪，稳定训练
                if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)

                self.optimizer.step()

                epoch_loss += float(batch_loss.item())
                global_step += 1

                if global_step % 50 == 0 and not self.cfg.quiet:
                    logger.info(f"[Ep {epoch}] step {global_step} | loss={float(batch_loss):.5f}")
                    self.tb.log_scalars("train", {"loss": float(batch_loss)}, global_step)

                # 间隔验证
                if global_step % self.cfg.val_interval == 0:
                    val_metrics = self.validate(val_loader)
                    self.tb.log_scalars("val", val_metrics, global_step)
                    improved = val_metrics["ndcg@k"] > self.best_val
                    if improved:
                        self.best_val = val_metrics["ndcg@k"]
                        self.steps_since_improve = 0
                        if self.cfg.save_best:
                            self._save_ckpt("best.pt")
                            logger.info(f"New best NDCG@{self.cfg.ndcg_k}: {self.best_val:.5f} - checkpoint saved.")
                    else:
                        self.steps_since_improve += 1
                        logger.info(f"No improvement steps: {self.steps_since_improve}/{self.cfg.patience}")
                        if self.steps_since_improve >= self.cfg.patience:
                            logger.warning("Early stopping triggered.")
                            return

            # 记录每个 epoch 的平均 loss
            epoch_loss /= max(len(train_loader), 1)
            self.tb.log_scalars("epoch", {"loss": epoch_loss}, epoch)
            logger.info(f"Epoch {epoch} done. avg_loss={epoch_loss:.6f}")

        logger.info("Training complete.")

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate and return averaged metrics."""
        self.model.eval()
        ndcgs: List[float] = []
        for batch in val_loader:
            metrics = self._val_metrics_on_batch(batch)
            ndcgs.append(metrics["ndcg@k"])
        avg = float(np.mean(ndcgs)) if ndcgs else 0.0
        logger.info(f"[VAL] NDCG@{self.cfg.ndcg_k}={avg:.5f}")
        return {"ndcg@k": avg}

    def _save_ckpt(self, name: str) -> None:
        """Save model checkpoint."""
        path = f"{self.cfg.ckpt_dir}/{name}"
        state = {"model": self.model.state_dict(), "config": self.cfg.__dict__}
        torch.save(state, path)