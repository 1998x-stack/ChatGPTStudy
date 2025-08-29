---

## 📌 分类 1：**分类任务（文本分类、情感分析）**

| 指标               | 说明                           | 是否敏感于类别不均衡 |
| ---------------- | ---------------------------- | ---------- |
| Accuracy（准确率）    | 预测正确样本数 / 总样本数               | 是（对类不均衡敏感） |
| Precision（精确率）   | TP / (TP + FP)，预测为正中实际为正的比例  | 否          |
| Recall（召回率）      | TP / (TP + FN)，实际为正中被预测为正的比例 | 否          |
| F1-score         | 精确率与召回率的调和平均                 | 否          |
| Macro / Micro 平均 | 多标签场景中评估不同类别时使用              | -          |

✅ **建议测试方式**：使用 `sklearn.metrics` 提供的 `classification_report`，可一键输出上述指标。

---

## 📌 分类 2：**序列标注任务（NER、POS、分词）**

| 指标                         | 说明                |
| -------------------------- | ----------------- |
| Precision / Recall / F1    | 多用于按实体/标签进行评估     |
| Exact Match Rate           | 标签序列完全匹配的比例       |
| Overlap F1 / Partial Match | 宽松匹配度量，如开始/结束位置相交 |

✅ **推荐库**：`seqeval`（支持BIO标签），或 HuggingFace 的 `evaluate` 工具。

---

## 📌 分类 3：**生成任务（摘要、翻译、问答、对话）**

| 指标                              | 说明                   | 特点                           |
| ------------------------------- | -------------------- | ---------------------------- |
| **BLEU**                        | 主要用于机器翻译，基于n-gram重合度 | 越高表示生成文本越接近参考                |
| **ROUGE**                       | 主要用于摘要，基于重合度统计       | 包括 ROUGE-1、ROUGE-2、ROUGE-L 等 |
| **METEOR**                      | 考虑词形还原、同义词的 BLEU 改进版 | 更注重语义匹配                      |
| **BERTScore**                   | 基于 BERT 向量表示的语义相似度   | 更适合生成类任务                     |
| **N-gram Diversity**            | 衡量重复性，防止“模型啰嗦”       | 用于生成文本的多样性评估                 |
| **Average Length / Perplexity** | 评估语言模型的自然程度          | PPL 越低表示语言模型困惑越少             |

✅ **BLEU & ROUGE 工具推荐**：

```bash
pip install sacrebleu rouge-score bert-score
```

---

## 📌 分类 4：**排序任务（搜索、推荐、问答排序）**

| 指标                            | 说明                | 是否考虑排序位置     |
| ----------------------------- | ----------------- | ------------ |
| **MRR**（Mean Reciprocal Rank） | 正确答案第一次出现的倒数的平均值  | 是            |
| **Precision\@K**              | 前K个结果中，正确结果的比例    | 否            |
| **Recall\@K**                 | 前K中召回了多少比例的所有正确答案 | 否            |
| **NDCG\@K**                   | 考虑排序位置和相关度打分的指标   | 是（对位次越靠前越敏感） |

✅ **评估示例**：

```python
from sklearn.metrics import ndcg_score
```

---

## 📌 分类 5：**语言模型（预训练LM）**

| 指标                   | 说明                   |
| -------------------- | -------------------- |
| **Perplexity (PPL)** | 评估语言模型对测试集的困惑程度，越低越好 |
| **Loss（交叉熵）**        | 与 PPL 有直接数学关系        |
| **Accuracy\@Top-K**  | 模型预测前K个词中是否包含正确答案    |

---

## 🎯 快速总结闪卡

| 🎯 指标         | 📚 适用任务 | 🧠 特点      |
| ------------- | ------- | ---------- |
| Accuracy / F1 | 文本分类    | 易理解，对不均衡敏感 |
| BLEU / ROUGE  | 翻译 / 摘要 | 统计式指标      |
| BERTScore     | 生成      | 语义感知能力强    |
| NDCG / MRR    | 排序      | 考虑位置的重要性   |
| Perplexity    | 语言建模    | 越低越好       |

---



在推荐系统中，**评估指标**是衡量推荐质量、排序效果和用户满意度的核心。以下是推荐系统中最常用的指标分类及测试方法，适用于**离线评估（Offline Evaluation）**。

---

## 🧩 1. **评分预测类指标（Rating Prediction）**

用于评估模型预测的评分与用户真实评分之间的差异。

| 指标                               | 公式                                                    | 说明                    |    |             |
| -------------------------------- | ----------------------------------------------------- | --------------------- | -- | ----------- |
| **MAE**（Mean Absolute Error）     | \$\frac{1}{N} \sum                                    | \hat{r}*{ui} - r*{ui} | \$ | 平均绝对误差，越小越好 |
| **RMSE**（Root Mean Square Error） | \$\sqrt{\frac{1}{N} \sum (\hat{r}*{ui} - r*{ui})^2}\$ | 均方根误差，对极端误差敏感         |    |             |
| **MSE**（Mean Squared Error）      | 与 RMSE 相似，不取根号                                        | 主要用于优化目标              |    |             |

✅ 适用场景：MovieLens、NetFlix 等评分预测型任务

---

## 📊 2. **排序类指标（Ranking Metrics）**

用于 Top-K 推荐中，评估推荐列表的相关性、排序和覆盖度。

### 🔹 常规 Top-K 相关性指标：

| 指标               | 说明                       | 是否考虑排序？  |
| ---------------- | ------------------------ | -------- |
| **Precision\@K** | 前K个推荐中，命中的比例             | ❌（不考虑位置） |
| **Recall\@K**    | 命中个数 / 用户所有感兴趣项目数        | ❌        |
| **F1\@K**        | Precision 与 Recall 的调和平均 | ❌        |

### 🔹 排序敏感指标：

| 指标                                 | 说明           | 特点            |
| ---------------------------------- | ------------ | ------------- |
| **MAP\@K**（Mean Average Precision） | 所有相关项目的平均精确率 | 对位置敏感         |
| **MRR\@K**（Mean Reciprocal Rank）   | 正确项首次出现的位置倒数 | 越靠前越好         |
| **NDCG\@K**                        | 综合考虑相关性和排名位置 | 高相关性排得越前，得分越高 |

**NDCG 公式简略**：

$$
DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}, \quad NDCG_k = \frac{DCG_k}{IDCG_k}
$$

---

## 🌍 3. **多样性 & 覆盖度类指标（Diversity / Coverage）**

反映推荐结果是否单一化或覆盖足够物品。

| 指标                       | 说明                |
| ------------------------ | ----------------- |
| **Coverage**             | 被推荐的不同物品数量 / 总物品数 |
| **Novelty**              | 推荐物品的流行度越低越新颖     |
| **Intra-list Diversity** | 推荐列表中物品的相似度越低越多样  |

✅ 用于防止热门物品垄断推荐（提高用户满意度）

---

## 📦 4. **用户满意度类指标（User-Oriented）**

通常通过用户反馈或模拟点击数据计算：

| 指标                | 说明                |
| ----------------- | ----------------- |
| **Hit Rate**（命中率） | 推荐列表中至少一个被用户点击/交互 |
| **CTR**（点击率）      | 被点击项目数 / 展示项目数    |
| **AUC**（ROC下的面积）  | 衡量正例排在负例前面的概率     |

---

## 🧪 快速测试代码（基于 Python）

你可以使用 [RecBole](https://recbole.io/)、[Surprise](https://surpriselib.com/)、或以下手动实现来测试模型效果：

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 真实评分 vs 预测评分
y_true = [4.0, 3.5, 5.0, 2.0]
y_pred = [3.8, 3.0, 4.8, 2.2]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE = {mae:.4f}, RMSE = {rmse:.4f}")
```

要测试 NDCG / Precision\@K，可使用：

```python
from sklearn.metrics import ndcg_score

true_relevance = [[1, 0, 1, 0, 1]]
scores = [[0.9, 0.1, 0.8, 0.2, 0.6]]

print("NDCG@3:", ndcg_score(true_relevance, scores, k=3))
```

---

## 🎯 总结闪卡

| 类别   | 指标                | 关注点        |
| ---- | ----------------- | ---------- |
| 评分预测 | MAE, RMSE         | 准确性，适合评分任务 |
| 排名   | NDCG\@K, MRR, MAP | 排序质量       |
| 点击行为 | HitRate, CTR, AUC | 模拟用户行为     |
| 多样性  | Coverage, Novelty | 用户满意度与体验   |

---
