# ItemCF Industrial-Style Pipeline (MovieLens-100K)

Features:
- Online dataset download (ml-100k, GroupLens)
- Loguru logging
- Robust configs & CLI overrides
- IUF weighting (optional)
- Sparse cosine TopK similarity
- User→lastN & Item→TopK indices
- ItemCF scoring (like × sim)
- Evaluation: HitRate@K / Recall@K / NDCG@K
- Built-in self-check to verify correctness & logic

## Quickstart

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py \
  --data_dir ./data \
  --min_user_inter 5 --min_item_inter 5 \
  --test_holdout leave_one \
  --implicit_like --implicit_threshold 4.0 \
  --last_n 200 --top_k 50 \
  --final_topn 100 \
  --log_level INFO
````

## Notes

* `--implicit_like` converts rating >= threshold to 1.0 (implicit feedback).
* Similarity is cosine with per-column TopK.
* Excludes train-seen items by default in final recommendation.

````

---

# 三、运行与扩展建议

- 运行命令（示例）：
  ```bash
  python -m pip install -r requirements.txt
  python scripts/run_pipeline.py --top_k 50 --last_n 200 --final_topn 100 --log_level INFO
````

* 可选参数：

  * `--use_iuf`（默认 True）：控制 IUF 重加权。
  * `--test_holdout ratio --test_ratio 0.2`：切换为比例切分。
  * `--implicit_like` / `--implicit_threshold 4.0`：控制隐式反馈与阈值。

---

# 四、关键“正确性自检”说明（已在 `cli_pipeline.py` 实现）

* ✅ 相似度矩阵尺寸/对角线为 0；
* ✅ 每列非零不超过 `top_k`（保障 TopK 约束）；
* ✅ `last_n`、`final_topn` 等参数范围合法；
* ✅ 运行日志将打印数据规模、相似度非零元、评测指标与一个样例用户的 Top-10 推荐列表。

---