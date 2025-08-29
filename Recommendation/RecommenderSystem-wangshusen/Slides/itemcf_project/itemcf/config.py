from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    """Data configuration.

    Attributes:
        dataset_name: Built-in dataset name. Currently supports 'ml-100k'.
        data_dir: Local directory to cache the dataset files.
        min_user_inter: Minimum interactions per user kept.
        min_item_inter: Minimum interactions per item kept.
        test_holdout: Holdout strategy. 'leave_one' (per user leave last one) or 'ratio'.
        test_ratio: Ratio for test split if test_holdout=='ratio'.
        implicit_like: Whether to binarize ratings as implicit feedback.
        implicit_threshold: Threshold for binarization (>= threshold => like=1).
    """
    dataset_name: str = "ml-100k"
    data_dir: str = "./data"
    min_user_inter: int = 5
    min_item_inter: int = 5
    test_holdout: str = "leave_one"
    test_ratio: float = 0.2
    implicit_like: bool = True
    implicit_threshold: float = 4.0


@dataclass(frozen=True)
class IndexConfig:
    """Index configuration.

    Attributes:
        last_n: Number of most-recent items kept per user.
        top_k: Number of top similar items kept per item.
        sim_shrinkage: Small constant added to denominator to stabilize cosine.
        use_iuf: Whether to apply inverse user frequency reweighting.
    """
    last_n: int = 200
    top_k: int = 50
    sim_shrinkage: float = 1e-12
    use_iuf: bool = True


@dataclass(frozen=True)
class RecoConfig:
    """Recommendation configuration.

    Attributes:
        final_topn: Number of final recommended items per user.
        exclude_seen: Whether to exclude items seen in train from recommendation.
        score_clip_min: Minimum score allowed after accumulation (for stability).
        score_clip_max: Maximum score allowed after accumulation (for stability).
    """
    final_topn: int = 100
    exclude_seen: bool = True
    score_clip_min: float = -1e9
    score_clip_max: float = 1e9


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        ks: List of cutoffs for metrics like HitRate@K / Recall@K / NDCG@K.
    """
    ks: tuple[int, ...] = (10, 20, 50, 100)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration.

    Attributes:
        random_seed: Random seed for reproducibility.
        num_threads: CPU threads for numpy/scipy where applicable.
        log_level: Log level for loguru.
    """
    random_seed: int = 2025
    num_threads: int = 0
    log_level: str = "INFO"


@dataclass(frozen=True)
class FullConfig:
    """Aggregate configuration."""
    data: DataConfig = DataConfig()
    index: IndexConfig = IndexConfig()
    reco: RecoConfig = RecoConfig()
    eval: EvalConfig = EvalConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI arg parser."""
    parser = argparse.ArgumentParser("ItemCF Pipeline")
    # Data
    parser.add_argument("--dataset_name", type=str, default="ml-100k")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--min_user_inter", type=int, default=5)
    parser.add_argument("--min_item_inter", type=int, default=5)
    parser.add_argument("--test_holdout", type=str, default="leave_one", choices=["leave_one", "ratio"])
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--implicit_like", action="store_true", default=True)
    parser.add_argument("--implicit_threshold", type=float, default=4.0)

    # Index
    parser.add_argument("--last_n", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--sim_shrinkage", type=float, default=1e-12)
    parser.add_argument("--use_iuf", action="store_true", default=True)

    # Reco
    parser.add_argument("--final_topn", type=int, default=100)
    parser.add_argument("--exclude_seen", action="store_true", default=True)
    parser.add_argument("--score_clip_min", type=float, default=-1e9)
    parser.add_argument("--score_clip_max", type=float, default=1e9)

    # Eval
    parser.add_argument("--ks", type=str, default="10,20,50,100")

    # Runtime
    parser.add_argument("--random_seed", type=int, default=2025)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def parse_config_from_args(args: Optional[list[str]] = None) -> FullConfig:
    """Parse FullConfig from CLI args.

    Args:
        args: Optional list of CLI arguments.

    Returns:
        FullConfig: Aggregated configuration instance.
    """
    parser = build_argparser()
    ns = parser.parse_args(args=args)

    ks_tuple = tuple(int(x) for x in ns.ks.split(",") if x.strip())

    return FullConfig(
        data=DataConfig(
            dataset_name=ns.dataset_name,
            data_dir=ns.data_dir,
            min_user_inter=ns.min_user_inter,
            min_item_inter=ns.min_item_inter,
            test_holdout=ns.test_holdout,
            test_ratio=ns.test_ratio,
            implicit_like=ns.implicit_like,
            implicit_threshold=ns.implicit_threshold,
        ),
        index=IndexConfig(
            last_n=ns.last_n,
            top_k=ns.top_k,
            sim_shrinkage=ns.sim_shrinkage,
            use_iuf=ns.use_iuf,
        ),
        reco=RecoConfig(
            final_topn=ns.final_topn,
            exclude_seen=ns.exclude_seen,
            score_clip_min=ns.score_clip_min,
            score_clip_max=ns.score_clip_max,
        ),
        eval=EvalConfig(ks=ks_tuple),
        runtime=RuntimeConfig(
            random_seed=ns.random_seed,
            num_threads=ns.num_threads,
            log_level=ns.log_level,
        ),
    )
