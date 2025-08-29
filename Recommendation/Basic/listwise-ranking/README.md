# ListMLE + Pairwise (Plackett–Luce) on MovieLens 100K

```
Project Structure
├── README.md
├── requirements.txt
└── src/
    ├── config.py
    ├── data/
    │   ├── __init__.py
    │   ├── movielens.py
    │   └── samplers.py
    ├── engine/
    │   ├── __init__.py
    │   ├── trainer.py
    │   └── evaluate.py
    ├── losses/
    │   ├── __init__.py
    │   └── listmle.py
    ├── metrics/
    │   ├── __init__.py
    │   └── ranking.py
    ├── models/
    │   ├── __init__.py
    │   └── two_tower.py
    ├── utils/
    │   ├── __init__.py
    │   ├── seeding.py
    │   ├── tb.py
    │   └── timer.py
    └── scripts/
        ├── train.py
        └── export_embeddings.py
```

An industrial‑style, scalable Learning‑to‑Rank training stack implementing **ListMLE** (listwise maximum likelihood via Plackett–Luce) and a **pairwise PL logistic** loss, with:

- **PyTorch** (GPU ready),
- **Loguru** structured logging,
- **tensorboardX** for metrics & loss curves,
- Solid engineering practices (masking, numerics, gradient clipping, early stopping),
- Google‑style docstrings, type hints, and Chinese comments for key steps.

## Features
- Load & preprocess **MovieLens 100K** into per‑user ranking lists (queries).
- Train a Two‑Tower scorer `s(u, i)` with configurable MLP head.
- Choose from: `listmle`, `pairwise_pl`, or a **hybrid** weighted objective.
- Robust batching of **variable‑length** lists with masks.
- Metrics: NDCG@K, MRR, MAP, Recall.

## Quick Start
```bash
python -m pip install -r requirements.txt
python -m src.scripts.train --project_root . --loss listmle --epochs 20 --batch_size 1024 --learning_rate 3e-4
# TensorBoard
tensorboard --logdir runs
```

## Repro Tips
- Use `--seed 42` for determinism.
- For faster sweeps, set `--loss pairwise_pl` and `--pairwise_per_query 16`.
- For best ranking quality, keep `--loss listmle` or `--loss hybrid` (default α=0.7 → ListMLE‑heavy).

## Export
```bash
python -m src.scripts.export_embeddings --project_root . --ckpt path/to/best.ckpt
```