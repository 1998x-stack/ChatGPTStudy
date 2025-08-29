# LambdaRank on MovieLens 100K (PyTorch + loguru + tensorboardX)

```
lambdarank-ml100k/
├── README.md
├── requirements.txt
├── run_train.py
├── run_eval.py
└── lambdarank/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── metrics.py
    ├── loss.py
    ├── model.py
    ├── trainer.py
    ├── evaluation.py
    └── utils.py
```

> 说明：
>
> * `run_train.py`：训练入口（含自动下载/解析 MovieLens-100k，日志与 TensorBoard）
> * `run_eval.py`：评测入口（载入 checkpoint 计算 NDCG/MAP/Recall）
> * `lambdarank/`：核心库模块（数据、模型、损失、训练、评测、工具）


This repository provides an industrial-grade implementation of **LambdaRank** trained on **MovieLens-100k**.  
It is production-oriented: modular structure, robust boundary checks, Google-style docstrings, PEP 8/257 compliant, with Chinese comments.

## Features
- Pairwise LambdaRank with |ΔNDCG| weighting
- Efficient per-query vectorized pair construction with caps
- Stable training: gradient clipping, early stopping, ckpt
- Logging via loguru; metrics to TensorBoard via tensorboardX
- Clean abstractions: DataModule, Trainer, Evaluator, Config

## Quickstart
```bash
pip install -r requirements.txt
# Train
python run_train.py --data_dir ./data --log_dir ./runs --ckpt_dir ./checkpoints
# Evaluate
python run_eval.py --data_dir ./data --ckpt_path ./checkpoints/best.pt
````

## Arguments

See `run_train.py -h` and `run_eval.py -h` for full options.

## Notes

* The script auto-downloads MovieLens-100k if not found (needs internet).
* If offline, place extracted `ml-100k` under `--data_dir`.