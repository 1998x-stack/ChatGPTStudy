# -*- coding: utf-8 -*-
"""Trainer for pointwise ranking.

This module provides a high-level Trainer with:
- AMP mixed precision
- Early stopping on validation RMSE
- Checkpointing top-K best models
- TensorBoard & loguru logging
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import mae, rmse, ndcg_at_k
from .model import PointwiseRanker


@dataclass
class EvalResult:
    """Holds evaluation results."""
    rmse: float
    mae: float
    ndcgs: Dict[int, float]  # {k: ndcg@k}


class Trainer:
    """High-level training orchestrator for Pointwise Ranking."""

    def __init__(
        self,
        model: PointwiseRanker,
        device: torch.device,
        run_dir: str | Path,
        lr: float,
        weight_decay: float,
        epochs: int,
        early_stop_patience: int,
        amp: bool,
        ndcg_ks: List[int],
    ) -> None:
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logdir=str(self.run_dir))
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.amp = amp
        self.ndcg_ks = ndcg_ks

        self.opt = torch.optim.Adam(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

        self.best_heap: List[Tuple[float, str]] = []  # (rmse, ckpt_path)
        self.best_rmse: float = float("inf")
        self.no_improve = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        corpus_info_min_rating: float = 1.0,
        corpus_info_max_rating: float = 5.0,
    ) -> None:
        """Run training with validation and checkpointing."""
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_res = self.evaluate(val_loader, clamp=(corpus_info_min_rating, corpus_info_max_rating))
            logger.info(
                f"[Epoch {epoch}] TrainLoss={train_loss:.6f} "
                f"ValRMSE={val_res.rmse:.6f} ValMAE={val_res.mae:.6f} "
                f"NDCGs={ {k: round(v, 6) for k, v in val_res.ndcgs.items()} }"
            )
            # TensorBoard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("val/rmse", val_res.rmse, epoch)
            self.writer.add_scalar("val/mae", val_res.mae, epoch)
            for k, v in val_res.ndcgs.items():
                self.writer.add_scalar(f"val/ndcg@{k}", v, epoch)

            # Early stopping & checkpoints
            improved = val_res.rmse < self.best_rmse - 1e-6
            if improved:
                self.best_rmse = val_res.rmse
                self.no_improve = 0
                self._save_checkpoint(epoch, val_res.rmse)
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loader, desc="Train", leave=False)
        for users, items, ratings in pbar:
            users = users.to(self.device, non_blocking=True)
            items = items.to(self.device, non_blocking=True)
            ratings = ratings.to(self.device, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                pred = self.model(users, items)
                loss = self.model.loss_fn(pred, ratings)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += float(loss.item()) * users.size(0)
            n += users.size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        clamp: Tuple[float, float] = (1.0, 5.0),
        collect_for_ranking: bool = True,
    ) -> EvalResult:
        """Evaluate model on a data loader.

        We clamp predictions to rating range for sensible metrics.

        Args:
            loader: DataLoader for evaluation.
            clamp: (min, max) clamp range.
            collect_for_ranking: Whether to collect user-item predictions for NDCG.

        Returns:
            EvalResult: RMSE/MAE and NDCG@K results.
        """
        self.model.eval()
        preds: List[float] = []
        targets: List[float] = []

        # 用于 NDCG 的结构：用户 -> [(item, score)...]；以及真实标签
        user_item_scores: Dict[int, List[Tuple[int, float]]] = {}
        user_item_labels: Dict[int, Dict[int, float]] = {}

        for users, items, ratings in tqdm(loader, desc="Eval", leave=False):
            users = users.to(self.device, non_blocking=True)
            items = items.to(self.device, non_blocking=True)
            ratings = ratings.to(self.device, non_blocking=True)

            pred = self.model(users, items).detach()
            # 推理阶段对评分做合理截断
            pred = pred.clamp(min=clamp[0], max=clamp[1])

            preds.append(pred.cpu().numpy())
            targets.append(ratings.cpu().numpy())

            if collect_for_ranking:
                for u, i, s, r in zip(users.cpu().numpy(), items.cpu().numpy(), pred.cpu().numpy(), ratings.cpu().numpy()):
                    user_item_scores.setdefault(int(u), []).append((int(i), float(s)))
                    user_item_labels.setdefault(int(u), {})[int(i)] = float(r)

        y_pred = np.concatenate(preds, axis=0)
        y_true = np.concatenate(targets, axis=0)

        res_rmse = rmse(y_pred, y_true)
        res_mae = mae(y_pred, y_true)
        res_ndcgs = {}
        if collect_for_ranking:
            for k in self.ndcg_ks:
                res_ndcgs[k] = ndcg_at_k(user_item_scores, user_item_labels, k=k)
        else:
            res_ndcgs = {k: 0.0 for k in self.ndcg_ks}

        return EvalResult(rmse=res_rmse, mae=res_mae, ndcgs=res_ndcgs)

    def _save_checkpoint(self, epoch: int, val_rmse: float) -> None:
        """Save a checkpoint and keep top-K best."""
        ckpt_path = self.run_dir / f"epoch_{epoch}_valrmse_{val_rmse:.6f}.pt"
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "val_rmse": val_rmse}, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # 维护最优K个模型（最小RMSE）
        heapq.heappush(self.best_heap, (-val_rmse, str(ckpt_path)))
        while len(self.best_heap) > 3:
            _, worst_path = heapq.heappop(self.best_heap)
            try:
                os.remove(worst_path)
                logger.info(f"Removed old checkpoint: {worst_path}")
            except OSError:
                pass
