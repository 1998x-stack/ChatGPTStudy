# Pointwise Ranking on MovieLens 100K (PyTorch)

This project implements a **Pointwise Ranking** (rating regression) pipeline using **MovieLens 100K**.
Tech stack: **PyTorch, loguru, tensorboardX**.  
It includes industrial features: clean structure, robust boundary checks, stratified splits, AMP, early-stopping, checkpointing, and NDCG@K evaluation.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Train & Evaluate
python -m ranker.main \
  --data_root ./data \
  --run_dir ./runs/exp1 \
  --embed_dim 64 \
  --hidden_dims 128 64 \
  --batch_size 1024 \
  --epochs 30 \
  --lr 0.001 \
  --weight_decay 1e-6 \
  --ndcg_ks 5 10 \
  --early_stop_patience 5
````

TensorBoard:

```bash
tensorboard --logdir ./runs
```

## Notes

* Dataset will be auto-downloaded (ml-100k) if not exists.
* Pointwise objective: **MSE** over ratings.
* Metrics: RMSE, MAE, and NDCG\@K (diagnostic ranking metric).


# 📁 工程目录（File Category & File-by-File）

```
pointwise_rank_ml100k/
├─ README.md
├─ requirements.txt
├─ ranker/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ data.py
│  ├─ model.py
│  ├─ evaluate.py
│  ├─ trainer.py
│  └─ main.py
└─ tests/
   └─ smoke_test.py
```

> 说明
>
> * `ranker/main.py`：项目入口，命令行参数、日志、TensorBoard、训练/验证/测试全流程。
> * `ranker/data.py`：MovieLens 100K 数据读取、ID 映射、用户分层划分、DataLoader。
> * `ranker/model.py`：Pointwise 回归模型（User/Item Embedding + MLP）。
> * `ranker/trainer.py`：训练器，含 AMP、早停、断点保存、指标评估与日志。
> * `ranker/evaluate.py`：RMSE/MAE/NDCG\@K 指标实现。
> * `ranker/utils.py`：随机种子、目录工具、下载与完整性校验、通用工具。
> * `ranker/config.py`：配置数据类与解析器。
> * `tests/smoke_test.py`：关键路径冒烟测试（shape/边界条件自检）。
> * `README.md`：运行指导。
> * `requirements.txt`：依赖列表。