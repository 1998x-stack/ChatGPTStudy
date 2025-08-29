from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from tensorboardX import SummaryWriter

from ..losses.listmle import listmle_loss, pairwise_pl_loss
from ..data.samplers import sample_pairwise_from_list
from ..engine.evaluate import evaluate_batch


class Trainer:
    """Generic trainer for listwise/pairwise objectives.

    中文注释：
        - 训练循环：前向、计算损失、反向传播、评估与早停；
        - 使用梯度裁剪、数值检查，保证工业稳定性。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        device: str = 'cuda',
        loss_type: str = 'hybrid',
        hybrid_alpha: float = 0.7,
        grad_clip: float = 5.0,
        eval_interval: int = 400,
        early_stop_patience: int = 8,
        topk: Tuple[int, ...] = (5, 10, 20),
        pairwise_per_query: int = 32,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.loss_type = loss_type
        self.hybrid_alpha = hybrid_alpha
        self.grad_clip = grad_clip
        self.eval_interval = eval_interval
        self.early_stop_patience = early_stop_patience
        self.topk = topk
        self.pairwise_per_query = pairwise_per_query

        self.global_step = 0
        self.best_metric = float('-inf')
        self.no_improve = 0

    def train_epochs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, float]:
        self.model.to(self.device)
        for epoch in range(1, epochs + 1):
            logger.info("Epoch {}", epoch)
            self._train_one_epoch(train_loader)
            # Force an eval at end of each epoch
            metrics = self._evaluate(val_loader)
            logger.info("Validation metrics: {}", metrics)
            self._early_stop_check(metrics)
            if self.no_improve >= self.early_stop_patience:
                logger.warning("Early stopping triggered.")
                break
        return {"best_ndcg": self.best_metric}

    def _train_one_epoch(self, train_loader: DataLoader) -> None:
        self.model.train()
        for batch in train_loader:
            self.global_step += 1
            user_ids = batch['user_ids'].to(self.device)
            items = batch['items'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            scores = self.model.list_scores(user_ids, items)

            # Compute losses according to type
            loss_lmle = listmle_loss(scores, labels, mask)
            q_idx, i_idx, j_idx = sample_pairwise_from_list(labels, mask, self.pairwise_per_query)
            loss_pair = pairwise_pl_loss(scores, labels, mask, q_idx.to(self.device), i_idx.to(self.device), j_idx.to(self.device))

            if self.loss_type == 'listmle':
                loss = loss_lmle
            elif self.loss_type == 'pairwise_pl':
                loss = loss_pair
            else:  # hybrid
                loss = self.hybrid_alpha * loss_lmle + (1 - self.hybrid_alpha) * loss_pair

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if self.global_step % 50 == 0:
                self.writer.add_scalar('train/loss_total', loss.item(), self.global_step)
                self.writer.add_scalar('train/loss_listmle', loss_lmle.item(), self.global_step)
                self.writer.add_scalar('train/loss_pairwise', loss_pair.item(), self.global_step)

            if self.global_step % self.eval_interval == 0:
                metrics = self._evaluate(val_loader)
                self._early_stop_check(metrics)
                if self.no_improve >= self.early_stop_patience:
                    break

    @torch.no_grad()
    def _evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics_sum: Dict[str, float] = {}
        n = 0
        for batch in data_loader:
            user_ids = batch['user_ids'].to(self.device)
            items = batch['items'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)
            scores = self.model.list_scores(user_ids, items)
            metrics = evaluate_batch(scores, labels, mask, self.topk)
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            n += 1
        for k in metrics_sum.keys():
            metrics_sum[k] /= max(1, n)
            self.writer.add_scalar(f'val/{k}', metrics_sum[k], self.global_step)
        return metrics_sum

    def _early_stop_check(self, metrics: Dict[str, float]) -> None:
        target = metrics.get(f"NDCG@{self.topk[0]}", 0.0)
        if target > self.best_metric:
            self.best_metric = target
            self.no_improve = 0
        else:
            self.no_improve += 1