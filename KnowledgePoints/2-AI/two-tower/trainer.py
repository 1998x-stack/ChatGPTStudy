# -*- coding: utf-8 -*-
"""Trainer with early stopping, AMP, gradient clipping, and ranking metrics."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from config import ExperimentConfig, ModelConfig, OptimConfig
from model import TwoTowerModel
from utils import (
    MetricBundle,
    count_parameters,
    create_logger,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class Trainer:
    """Encapsulates training and evaluation routines for Two-Tower."""

    def __init__(self, cfg: ExperimentConfig, model: TwoTowerModel) -> None:
        self.cfg = cfg
        self.model = model
        self.logger = create_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.amp)

        self.best_metric = -float("inf")
        self.bad_epochs = 0

        # 重要信息打印
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Trainable params: {count_parameters(self.model):,}")
        self.logger.info(f"Optimizer: Adam(lr={cfg.optim.lr}, wd={cfg.optim.weight_decay})")
        self.logger.info(f"AMP: {cfg.optim.amp}, GradClip: {cfg.optim.grad_clip_norm}")

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch, return average loss."""
        self.model.train()
        losses = []
        t0 = time.time()
        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.cfg.optim.amp):
                loss = self.model.info_nce_loss(
                    batch,
                    temperature=self.cfg.optim.temperature,
                    in_batch_negatives=self.cfg.data.in_batch_negatives,
                )
            self.scaler.scale(loss).backward()
            if self.cfg.optim.grad_clip_norm and self.cfg.optim.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                self.logger.info(
                    f"[Epoch {epoch}] Step {step+1}/{len(loader)} "
                    f"Loss={np.mean(losses):.4f}"
                )
        elapsed = time.time() - t0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        self.logger.info(f"[Epoch {epoch}] Train loss={avg_loss:.4f} (took {elapsed:.1f}s)")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, all_item_emb: torch.Tensor) -> MetricBundle:
        """Evaluate ranking metrics using brute-force full ranking."""
        self.model.eval()
        device = self.device

        all_item_ids = torch.arange(all_item_emb.size(0), device=device, dtype=torch.long)
        gt_items: List[int] = []
        ranked: List[np.ndarray] = []

        t0 = time.time()
        for batch in loader:
            user_idx = batch["user_idx"].to(device, non_blocking=True)
            pos_item_idx = batch["pos_item_idx"].to(device, non_blocking=True)

            u = self.model.encode_user(user_idx)            # [B, D]
            # 与所有物品向量点积，得到打分
            logits = torch.matmul(u, all_item_emb.t())      # [B, I]
            # 排序，获取物品排名索引
            sorted_idx = torch.argsort(logits, dim=1, descending=True)  # [B, I]
            ranked.append(sorted_idx.detach().cpu().numpy())
            gt_items.extend(pos_item_idx.detach().cpu().numpy().tolist())

        ranked_indices = np.concatenate(ranked, axis=0)
        metrics = {}
        recall = {}
        mrr = {}
        ndcg = {}
        for k in self.cfg.eval.k_values:
            recall[k] = recall_at_k(ranked_indices, gt_items, k)
            mrr[k] = mrr_at_k(ranked_indices, gt_items, k)
            ndcg[k] = ndcg_at_k(ranked_indices, gt_items, k)
        elapsed = time.time() - t0
        self.logger.info(
            f"[Eval] {len(gt_items)} users evaluated in {elapsed:.1f}s -> "
            f"Recall={recall} MRR={mrr} NDCG={ndcg}"
        )
        return MetricBundle(recall=recall, mrr=mrr, ndcg=ndcg)

    @torch.no_grad()
    def precompute_all_item_embeddings(self, num_items: int, batch_size: int = 4096) -> torch.Tensor:
        """Precompute all item embeddings [I, D] for fast evaluation."""
        self.model.eval()
        device = self.device
        ids = torch.arange(num_items, device=device, dtype=torch.long)
        vecs = []
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            v = self.model.encode_item(ids[start:end])  # [B, D]
            vecs.append(v)
        return torch.cat(vecs, dim=0)  # [I, D]

    def maybe_early_stop(self, score: float) -> bool:
        """Check early stopping. Returns True if should stop."""
        if score > self.best_metric:
            self.best_metric = score
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.cfg.optim.early_stop_patience > 0 and (
                self.bad_epochs >= self.cfg.optim.early_stop_patience
            )
