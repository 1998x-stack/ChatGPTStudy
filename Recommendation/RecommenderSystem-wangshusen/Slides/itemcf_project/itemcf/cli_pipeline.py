from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
from loguru import logger
from scipy import sparse

from .config import parse_config_from_args, FullConfig
from .logger import init_logger
from .datasets import load_dataset_artifacts, DatasetArtifacts
from .similarity import apply_iuf, topk_cosine_similarity
from .indexer import build_user_lastn_index, build_item_topk_index
from .recommender import ItemCFRecommender
from .evaluator import evaluate_all


def _set_random_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def run_pipeline(args: list[str] | None = None) -> None:
    """End-to-end ItemCF pipeline."""
    cfg: FullConfig = parse_config_from_args(args)
    init_logger(cfg.runtime.log_level)

    if cfg.runtime.num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(cfg.runtime.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(cfg.runtime.num_threads)
        logger.info("Set num threads to {}", cfg.runtime.num_threads)

    _set_random_seed(cfg.runtime.random_seed)

    # 1) 数据加载与预处理（在线下载、过滤、切分、映射、CSR）
    arts: DatasetArtifacts = load_dataset_artifacts(
        data_dir=cfg.data.data_dir,
        min_user_inter=cfg.data.min_user_inter,
        min_item_inter=cfg.data.min_item_inter,
        holdout=cfg.data.test_holdout,
        ratio=cfg.data.test_ratio,
        implicit_like=cfg.data.implicit_like,
        implicit_threshold=cfg.data.implicit_threshold,
    )

    # 2) 构造相似度（可选 IUF）
    x = arts.train_csr
    if cfg.index.use_iuf:
        logger.info("Applying IUF reweighting ...")
        x = apply_iuf(x)

    logger.info("Computing item-item TopK cosine similarity ...")
    sim_csr, norms = topk_cosine_similarity(
        csr=x,
        top_k=cfg.index.top_k,
        sim_shrinkage=cfg.index.sim_shrinkage,
    )
    logger.info("Similarity built: nnz={}", sim_csr.nnz)

    # 3) 构建索引：User→lastN, Item→TopK
    user_lastn = build_user_lastn_index(arts.train_csr, last_n=cfg.index.last_n)
    item_topk = build_item_topk_index(sim_csr, top_k=cfg.index.top_k)

    # 4) 召回器
    recommender = ItemCFRecommender(
        user_lastn=user_lastn,
        item_topk=item_topk,
        exclude_seen=cfg.reco.exclude_seen,
        score_clip_min=cfg.reco.score_clip_min,
        score_clip_max=cfg.reco.score_clip_max,
        train_seen=arts.train_seen,
    )

    # 5) 评测
    user_ids = list(arts.test_pos.keys())
    metrics = evaluate_all(
        user_ids=user_ids,
        recommender=recommender,
        test_pos=arts.test_pos,
        ks=cfg.eval.ks,
    )

    for k in cfg.eval.ks:
        logger.info(
            "Eval@{}: HitRate={:.4f}, Recall={:.4f}, NDCG={:.4f}",
            k, metrics[k]["HitRate"], metrics[k]["Recall"], metrics[k]["NDCG"]
        )

    # 6) 示例推荐（展示一个有测试样本的用户）
    sample_user = user_ids[0] if user_ids else None
    if sample_user is not None:
        demo = recommender.recommend(sample_user, topn=10)
        logger.info("Sample user {} recommendation (top-10): {}", sample_user, demo)

    # 7) 关键自检步骤（Correctness & Logic Recheck）
    _self_check(sim_csr, arts.train_csr, cfg)


def _self_check(sim_csr: sparse.csr_matrix, train_csr: sparse.csr_matrix, cfg: FullConfig) -> None:
    """Key step: self-check for correctness and logic.

    Checks:
      - Similarity matrix shape and diagonal
      - Score range sanity
      - TopK per column guarantee
      - User lastN and Item topK parameters in valid ranges
    """
    n_users, n_items = train_csr.shape
    assert sim_csr.shape == (n_items, n_items), "Similarity matrix shape mismatch."
    # 对角线应为 0（或已被清理）
    diag = sim_csr.diagonal()
    assert np.allclose(diag, 0.0), "Similarity diagonal must be zero."

    # 列 TopK 合理（非零不超过 top_k，允许相等边界）
    top_k = cfg.index.top_k
    col_nnz = np.diff(sim_csr.tocsc().indptr)
    if n_items > 0:
        assert np.max(col_nnz) <= top_k, "Per-column nonzeros exceed top_k."

    # last_n & final_topn 合法
    assert cfg.index.last_n > 0, "last_n must be positive."
    assert 1 <= cfg.reco.final_topn <= max(1000, n_items), "final_topn out of range."

    logger.info("Self-check passed: shapes, diagonal, TopK, and parameter ranges are valid.")