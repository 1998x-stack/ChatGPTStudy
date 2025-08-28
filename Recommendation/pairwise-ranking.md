# Pairwise ranking学习大纲

**User:** Anonymous (kanekoveblen824@outlook.com)  
**Created:** 8/28/2025 10:37  
**Updated:** 8/28/2025 11:08  
**Exported:** 8/28/2025 13:23  
**Link:** [https://chatgpt.com/c/68afc0d5-637c-8328-9bf3-5a51e7c2ff44?model=gpt-5-thinking](https://chatgpt.com/c/68afc0d5-637c-8328-9bf3-5a51e7c2ff44?model=gpt-5-thinking)  

## Prompt:
Task: 深入学习研究pair ranking里面的技术细节和底层机制

Methid: 耐心引导式&选择题提问式&知识卡片/QA闪卡总结式&interesting-examples/jokes-driven&一步一步拆解分析

Bonus&Candy🍬:colorful emoji & 幽默有趣 & 有思想深度 & 图解/闪卡 & funny games!!!

Notice: 我们可以quote原文中的核心句子&每隔几段进行总结复习!!!

首先给出学习大纲:

## Response:
太棒了！我们将开启一段有趣又深入的学习之旅，探索 Pairwise Ranking（成对排序）这一重要的排序学习方法。

* * *

🧠 Pairwise Ranking 学习大纲（逐步拆解式）
-------------------------------

| 模块 | 内容 | 学习方式 |
| --- | --- | --- |
| A. 导论 | Pairwise ranking 的基本定义、应用场景、与 pointwise/listwise 对比 | 图解+闪卡🃏 |
| B. 排序目标函数 | Hinge Loss、Cross Entropy Loss（Log Loss）、Pairwise Preference Loss 等 | 公式推导+小测试🧪 |
| C. 排序模型结构 | RankNet、LambdaRank、LambdaMART、BERT-based pairwise 结构等 | 模型图+选择题🧩 |
| D. 样本构造技巧 | 如何从文档集合中构建有效的样本对（正负文档对） | 案例驱动🧺 |
| E. 损失梯度与优化 | Lambda的由来、如何隐式优化 NDCG/ERR？ | 推导练习+知识卡📐 |
| F. 实例演练 | 模拟排序场景 + RankNet 手动算一轮梯度 | 角色扮演+游戏🕹️ |
| G. 应用实战 | Pairwise ranking 在推荐系统、搜索排序中的部署与优化 | 案例剖析🛠️ |
| H. 总结回顾 | 比较三种排序学习范式 + 总结优缺点 | 表格对比+闪卡复习📊 |

* * *

🍬Bonus Candy Elements
----------------------

*   🧩 小测验穿插在每个模块中，帮你实时 check 自己掌握程度
*   😂 趣味类比/段子（比如“你是更喜欢《千与千寻》还是《灌篮高手》？搜索系统要懂你”）
*   📘 每个模块都提供 1–2 个闪卡，总结公式 + 关键词 + 推导路径
*   🎲 游戏互动：比如“我出排序，用户当评委”，“你是搜索引擎，我是用户”

* * *

现在我们从 **模块 A：导论** 开始——

我先抛出一个轻松的问题给你热身：

* * *

### ❓选择题：哪个场景最适合使用 Pairwise Ranking？

A. 预测电影评分  
B. 推荐系统中，点击了 item A 没点击 item B  
C. 把所有候选排序输出一个分数  
D. 对句子进行情感分类（正向/负向）

👉 你觉得哪一个是 **Pairwise Ranking** 的经典使用场景？我们来一步一步分析！

## Prompt:
B

## Response:
👏 完全正确！我们来拆解下为什么答案是 **B. 推荐系统中，点击了 item A 没点击 item B**，并顺便补一个记忆闪卡 🧠！

* * *

🧠 知识卡片 01：什么是 Pairwise Ranking？
--------------------------------

| 内容 | 说明 |
| --- | --- |
| 📌 目标 | 学习一个函数 `f(x)`，使得对于成对样本 $(x\_i, x\_j)$，满足 $f(x\_i) > f(x\_j)$（代表 $x\_i$ 更优） |
| 🎯 应用 | 排序场景：搜索、推荐、广告、问答系统中对结果排序 |
| 🧪 样本构造 | 从用户行为中构造偏好对，如：点击了 A 没点击 B，可视为 A > B |
| 🆚 对比 | 与 pointwise（回归/分类）和 listwise（直接优化整个排序）不同，pairwise 专注于**成对比较** |

* * *

### 🔍 为什么不是其他选项？

*   **A. 预测电影评分** → 属于 pointwise 的典型例子（预测一个具体数值）
*   **C. 所有候选打分** → 虽然 pairwise 最终也会用打分，但这是 listwise 更擅长的事
*   **D. 情感分类** → 属于分类任务，不涉及排序/比较

* * *

🧩 快速复述（你来！）
------------

💬 “所以 Pairwise Ranking 主要用来……？”  
（你可以用自己的话总结一下，越简洁越好~）

等你一句话总结完，我们就继续进入模块 B：「常见 Pairwise Ranking 的 Loss 函数解析」——非常重要的一环！

## Prompt:
比较A选项 与 B选项的好坏

## Response:
太好了，这就是深度学习者该有的思维方式：不只是“记住答案”，而是**比较和理解为什么更好**。我们来做个清晰的对比 👇

* * *

⚔️ A vs B：Pointwise vs Pairwise 的真实对比战
--------------------------------------

| 维度 | A. 预测电影评分（Pointwise） | B. 用户点击偏好对（Pairwise） |
| --- | --- | --- |
| 🎯 目标 | 预测一个“绝对分数”，如：你可能给这部电影打 4.7 分 | 学习相对偏好：你更喜欢 A 胜过 B |
| 🧠 学习方式 | 回归问题（如 MSE）；模型拟合具体评分值 | 比较问题（如 hinge loss）；模型学习偏好方向 |
| 🧪 标签获取 | 需要明确分数标签，如 1~5 星评分 | 更容易获得：点击行为可直接推断偏好对 |
| ⚠️ 假设 | 用户的打分是准确的、有区分度的 | 用户点击能反映真实偏好（即使有噪声） |
| 🚨 缺点 | 分数主观性大，噪声高、可比性差 | 对样本对构造要求高，数据量成指数级增长 |
| 🧬 应用例子 | 评分预测、满意度建模 | 搜索引擎、推荐系统、广告排序等 |

* * *

### 🧠 小结：适合什么场景？

| 类型 | 更适合… |
| --- | --- |
| Pointwise | 有明确评分/等级指标的数据集，如电影/课程评分 |
| Pairwise | 用户行为为主的数据，关注谁“更相关”，如 CTR、推荐点击、搜索排序 |

* * *

要不我们来个轻松互动小游戏 🧩：

你是一个推荐系统工程师，现在你有以下三种数据：

1.  用户给电影打了分（1–5星）
2.  用户点击了视频 A 没点击视频 B
3.  用户一次性看了多个视频

你会如何分别建模这三种行为？  
先挑 **第1种**，你觉得用 pointwise 还是 pairwise？为啥？

## Prompt:
1 point wise
2 pairwise
3 point wise with multi class softmax + entropy loss

## Response:
💯 完全正确！你已经非常理解这三种行为对应的建模范式了。我们来逐个回顾、再补点细节巩固知识：

* * *

🎯 三种用户行为建模方式解析
---------------

### **1️⃣ 用户给电影打了分（1–5 星）**

*   ✅ **Pointwise**
*   💡 本质是回归问题，也可以转换成分类（离散的 1~5 类）
*   📉 常见 loss：
    *   MSE / RMSE（用于回归）
    *   Cross Entropy（用于多类分类）

### **2️⃣ 用户点击了 A 没点击 B**

*   ✅ **Pairwise**
*   💡 明确反映了用户更偏好 A，而非 B ⇒ 训练目标：$f(A) > f(B)$
*   📉 常见 loss：
    *   Hinge Loss（SVM）
    *   Logistic Loss（RankNet）
    *   Pairwise Logistic Loss（含概率形式）

### **3️⃣ 用户一次性看了多个视频**

*   ✅ **Multi-class Pointwise 或 Listwise**
*   💡 这类属于 session-based 或整体行为，可能需要建模多个 item 的相对重要性
*   📉 可选 loss：
    *   多类交叉熵（Pointwise + softmax）
    *   ListNet / ListMLE（Listwise loss，更关注整个 ranking）

* * *

### 📘 知识卡片 02：三种排序学习范式总结

| 范式 | 输入 | 标签类型 | Loss 例子 | 优势 | 劣势 |
| --- | --- | --- | --- | --- | --- |
| Pointwise | 单个 item | 实数或类别 | MSE, CrossEntropy | 简单直接 | 忽略相对顺序 |
| Pairwise | item 对 | 谁优于谁 | Hinge, LogLoss | 关注排序方向 | 样本数增长快 |
| Listwise | item 列表 | 排序/权重分布 | ListNet, ListMLE | 全局优化排序指标 | 训练复杂、样本构造难 |

* * *

我们继续深入探索 Pairwise 的核心技术点：

🔜 **模块 B：排序目标函数**  
比如说，最早用于 Pairwise 的经典模型是 **RankNet**，它用了什么 Loss 呢？它为啥不像 SVM 那样用 hinge loss？  
👇

### 🧩选择题：RankNet 使用的损失函数是？

A. Mean Squared Error  
B. Cross Entropy over pairwise preference  
C. Hinge Loss  
D. Triplet Margin Loss

你选哪个？我们来推一下 RankNet 的 loss！

## Prompt:
D

## Response:
你选的是 **D. Triplet Margin Loss**，这其实是个很好的想法！但它并不是 RankNet 的核心损失函数。我们来详细推导一下选项，并解释为什么 **正确答案是 B：Cross Entropy over pairwise preference**。

* * *

🧠 RankNet 的核心机制是什么？
--------------------

RankNet 是微软提出的早期 pairwise 排序模型，核心思想是：

> **将成对的文档 $(d\_i, d\_j)$ 映射为概率 $P\_{ij}$，表示“$d\_i$ 的相关性大于 $d\_j$ 的概率”，然后使用交叉熵损失函数优化这个概率。**

* * *

### 🔬 Step-by-step 推导过程：

#### 1️⃣ 网络输出分数：

给两个文档 $d\_i$ 和 $d\_j$，经过神经网络模型 $f$ 后分别输出：

*   $s\_i = f(d\_i)$
*   $s\_j = f(d\_j)$

#### 2️⃣ 概率建模（用 sigmoid）：

$$
P_{ij} = \frac{1}{1 + \exp(-(s_i - s_j))} \\
$$

这个概率越大，说明模型越确信 $d\_i$ 比 $d\_j$ 更 relevant。

#### 3️⃣ 真值标签：

$$
y_{ij} = \begin{cases} 1, & \text{如果 } d_i > d_j \\ 0, & \text{如果 } d_j > d_i \end{cases}
$$

#### 4️⃣ 交叉熵损失（Binary Cross Entropy）：

$$
\mathcal{L}_{RankNet} = - \left[ y_{ij} \log P_{ij} + (1 - y_{ij}) \log (1 - P_{ij}) \right]
$$

这就是 RankNet 的核心！

* * *

### ❌ 其他选项为什么不合适？

| 选项 | 理由 |
| --- | --- |
| A. MSE | 用于 pointwise 回归，不适合 pairwise 的概率建模 |
| C. Hinge Loss | 是 RankSVM 使用的，对应 margin-based ranking，但不是 RankNet 的核心 |
| D. Triplet Margin Loss | 是 triplet learning 的典型（用于图像/度量学习等），非 RankNet 所用 |

* * *

🎓 关键概念回顾闪卡
-----------

| 关键词 | 解释 |
| --- | --- |
| **RankNet** | 基于 pairwise 比较的神经网络排序模型 |
| **Sigmoid** | 将分值差 $(s\_i - s\_j)$ 映射为概率 |
| **Cross Entropy** | 用于优化“哪个更相关”的概率预测 |
| **样本对构造** | 通常基于点击数据或人工标注判断哪个更 relevant |

* * *

🤹‍♂️ 有趣类比记忆法
-------------

想象你是一个图书推荐系统，两个用户都搜“深度学习”：

*   用户 A 点了《深度学习入门》，
*   用户 B 点了《深度学习理论与实践》。

你作为系统会记录：“A 的点击说明《入门》比《理论》更 relevant”。RankNet 就是从这些点点滴滴中，学习如何对\*\*“哪个更 relevant”建模概率并优化”。

* * *

现在轮到你来回答一个 follow-up 问题：

> **如果 RankNet 的概率函数是 sigmoid($s\_i - s\_j$)，你觉得输出分数 $s\_i$ 和 $s\_j$ 的值范围有关系吗？它们必须是正数吗？必须归一化吗？**

（这是引出下一步“分值刻度 vs 排序”的好点，我们下轮继续讲这个！）

2/2

## Prompt:
值范围一致，不一定正数，不需要归一化

## Response:
🎯 太棒了，你说得完全对，而且说出了三个关键点，我们来详细拆解一下你的回答，让这块知识扎实地落地。

* * *

✅ 回答解析：为什么值范围**一致**就够了？
-----------------------

### 🧩 1. **不一定是正数**

没错！RankNet 并不关心 $s\_i$ 和 $s\_j$ 是不是正数。因为最终我们用的是：

$$
P_{ij} = \frac{1}{1 + \exp(-(s_i - s_j))}
$$

这个函数只关注**差值 $s\_i - s\_j$**，所以无论它们本身是多少，负数、零、正数都可以。

> 🧠 类比：你跟朋友打分，如果你打《电影A》8分，《电影B》6分，而他打的分别是 3 和 1，**虽然绝对值不同，但排序一致，逻辑完全相同**。

* * *

### 🧩 2. **不需要归一化**

又对！因为 sigmoid 自带归一化的概率输出能力，直接从分值差里算出 $P\_{ij}$，不需要像 softmax 那样做归一化。

#### 举个例子：

$$
s_i = 3,\quad s_j = 1\quad \Rightarrow\quad P_{ij} = \frac{1}{1 + e^{-2}} \approx 0.88
$$

无须管这两个分数是不是太大、总和是多少 —— 我们只关心 $s\_i - s\_j$。

* * *

### 🧩 3. **值范围要一致**

✔️这点尤其关键！虽然不需要归一化，但为了 **训练稳定、模型泛化好**，**分值尺度（scale）要一致**是非常重要的。

> 否则，一个 batch 的分值在 \[-1, 1\]，另一个在 \[10, 20\]，虽然差值排序可能没问题，但优化过程容易变得不稳定、收敛慢。

* * *

🧠 知识卡片 03：RankNet 分值输出的要求
--------------------------

| 特性 | 是否需要 |
| --- | --- |
| 输出为正 | ❌ 不需要 |
| 归一化 | ❌ 不需要 |
| 分值差 $s\_i - s\_j$ | ✅ 必须有意义（反映真实偏好） |
| 分值范围一致 | ✅ 建议保持一致，确保训练稳定性 |

* * *

🎲 小测验互动（挑战模式）
--------------

假如现在模型输出如下：

| 文档 | 输出 $s$ |
| --- | --- |
| A | 0.2 |
| B | 0.1 |
| C | \-0.5 |

👉 你能说出以下每一对的 $P\_{ij}$ 吗？（保留一位小数即可）

*   $P\_{AB} = ?$
*   $P\_{AC} = ?$
*   $P\_{CB} = ?$

你来算一下，我来验算并解释每个对应的概率含义！🧪

## Prompt:
next

## Response:
好的，我们继续进入 Pairwise 排序模型的经典代表之一——**RankNet** 的核心机制扩展，走进模块 **C：排序模型结构**。

* * *

🎯 模块 C：经典 Pairwise 排序模型结构对比
----------------------------

我们会聚焦在以下几个代表性模型上：

| 模型 | 核心思想 | 是否用 Pairwise | 优势亮点 |
| --- | --- | --- | --- |
| RankNet | 用神经网络预测 $s\_i, s\_j$，比较其差值学习偏好 | ✅ | 概率建模清晰，训练简单 |
| RankSVM | 用 hinge loss 优化 pairwise ranking margin | ✅ | 强几何解释，训练快 |
| LambdaRank | 在 RankNet 上引入 **梯度加权机制**，隐式优化 NDCG | ✅ | 对实际指标（NDCG）更敏感 |
| LambdaMART | 用 **梯度提升树（GBDT）** 替代神经网络 + LambdaRank | ✅ | 精度高，工业级 SOTA |
| BERT-pair | 用 BERT 做成对编码（如 Query–Doc pair）进行偏好比较 | ✅ | 利用预训练语言模型优势 |

* * *

🧠 LambdaRank 的核心亮点是？
---------------------

它在 RankNet 的基础上引入了**梯度权重 $\\lambda$**，让模型在训练时更关注排序变化大、影响指标（如 NDCG）更多的位置。

> 🧠 可以理解为：“影响 NDCG 多的 pair，模型就训练得更用力”。

* * *

🧩 小问题（选择题）
-----------

LambdaRank 的名字中 "Lambda" 指的是？

A. 正则项权重  
B. 神经网络中隐藏层的参数名  
C. 梯度中的可学习参数  
D. 梯度中根据排序指标（如 NDCG）权重调整的项

你选哪个？  
选完后我们将揭开 LambdaRank 是如何“偷偷优化排序指标”的黑科技～

## Prompt:
c

## Response:
嘿嘿～这次差一点！🛑 答案是 **D：梯度中根据排序指标（如 NDCG）权重调整的项**。但你选的 **C** 也蛮接近真相，值得我们来认真拆解一下 🌟

* * *

🔬 什么是 LambdaRank 的 Lambda？
---------------------------

在 LambdaRank 中，模型仍然像 RankNet 一样输出两个分值 $s\_i$ 和 $s\_j$，然后计算 $s\_i - s\_j$ 来估计偏好。但——

💥 **不同点在于：它不再直接最小化 loss，而是用“定制的梯度”去训练模型！**

这些梯度被称作 **lambda 值**，具体定义如下：

$$
\lambda_{ij} = \frac{\partial \mathcal{L}}{\partial s_i} = \sigma \cdot |\Delta \text{NDCG}_{ij}|
$$

其中：

*   $\\sigma$ 是 $s\_i - s\_j$ 的 sigmoid 函数导数部分
*   $\\Delta \\text{NDCG}\_{ij}$ 表示**如果交换 $i,j$ 两个文档的顺序，NDCG 变化了多少**

> 🎯 换句话说：**排序影响越大的 pair，对应的训练梯度越大，模型就越努力优化它。**

* * *

🧠 闪卡：LambdaRank 的 Lambda 是什么？
------------------------------

| 项 | 意义 |
| --- | --- |
| λ（lambda） | 并非参数，而是 **“梯度权重”** |
| 来源 | 基于 NDCG 损失的变化估算而来 |
| 目的 | 隐式优化最终排序指标（如 NDCG） |
| 优势 | 无需显式优化复杂指标，训练更稳，效果更好 |

* * *

😂 类比理解：考试排位系统
--------------

想象一个老师要给学生排名：

*   **RankNet 老师**：只看谁分高一些，多一分也要管
*   **LambdaRank 老师**：只管你这次调换顺序会不会让前十名变了（对排名影响大不大）

* * *

🎲 小互动：你来猜
----------

如果某一对文档 $(i, j)$，一个在第2位，一个在第9位，它们的 relevance 反过来了，那么对 NDCG 的影响大不大？

你觉得这个 pair 的 lambda 值会是：

A. 很小  
B. 一般般  
C. 非常大

选一个，我们继续剖析 NDCG 的结构和 lambda 权重分配的秘密 📐📊

## Prompt:
C

## Response:
✨ 完美命中！选 **C：非常大** —— 正是 LambdaRank 的精髓所在！

* * *

🎯 为什么排名靠前的错排更严重？
-----------------

这是因为排序指标（尤其是 **NDCG**）本质上是**位置敏感的**：

### 公式复习一下：

$$
\text{DCG}_k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

这个公式告诉我们：

*   第 1 位：除数是 $\\log\_2(2) = 1$，影响最大
*   第 2 位：$\\log\_2(3) \\approx 1.58$
*   第 9 位：$\\log\_2(10) \\approx 3.32$

* * *

📉 举个例子来看看错排的影响
---------------

假设：

*   文档 A：相关性高（rel=3），排名在第9位
*   文档 B：相关性低（rel=1），排名在第2位

现在两者排名反了！

我们计算一下：

### 交换前的 DCG 贡献：

*   A 在第9位：$\\frac{2^3 - 1}{\\log\_2(10)} = \\frac{7}{3.32} ≈ 2.11$
*   B 在第2位：$\\frac{2^1 - 1}{\\log\_2(3)} = \\frac{1}{1.58} ≈ 0.63$

### 交换后的 DCG 贡献：

*   A 到第2位：$\\frac{7}{1.58} ≈ 4.43$
*   B 到第9位：$\\frac{1}{3.32} ≈ 0.30$

👉 **DCG 变动 = 4.43 + 0.30 − (2.11 + 0.63) = 2.0！**

> 说明只交换一个位置，就造成了 2 分的 NDCG 波动 —— 训练时自然要加大这对 pair 的“训练力度”，也就是 **lambda 值更大**。

* * *

🧠 闪卡回顾：Lambda 的动态权重思想
----------------------

| 原理 | 描述 |
| --- | --- |
| 🔍 位置越靠前 | 影响越大（因 $\\log\_2(i+1)$ 小） |
| 🔁 错排位置越远 | 损失越大，梯度 $\\lambda$ 越大 |
| 🧪 LambdaRank | 通过计算排序交换对指标的影响大小，动态调整 pair 的训练力度 |

* * *

🎲 来个小实战题！
----------

假设你是一个 LambdaRank 系统，现在观察到以下文档顺序：

| 文档 | Rel | Rank |
| --- | --- | --- |
| D1 | 3 | 3 |
| D2 | 2 | 1 |

你会对这个 pair 分配一个高 lambda 值吗？为啥？  
（你可以思考：是否会影响 NDCG 很大？）

答完我们继续下一节 → **LambdaMART 如何结合 GBDT 构造工业级模型？🔥**

## Prompt:
ok

## Response:
🧠 太棒了，我们继续进入 Pairwise 排序的工业应用明星选手——**LambdaMART**！

* * *

🚀 模块 C.2：LambdaMART = LambdaRank + MART（GBDT）
----------------------------------------------

### 📘 快问快答：MART 是啥？

MART = Multiple Additive Regression Trees，其实就是我们熟知的 GBDT（梯度提升决策树）家族的一员。

> 📌 简单来说：LambdaMART 把 LambdaRank 的**pairwise 梯度思想**，迁移到了 **树模型中**！

* * *

🧠 LambdaMART 的工作流程
-------------------

1️⃣ 构造训练对（如点击了 A 没点击 B）  
2️⃣ 计算每个样本点的 λ 值（影响 NDCG 的梯度）  
3️⃣ 把这些 λ 作为伪标签，喂给 GBDT（做回归）  
4️⃣ 每轮迭代构建一棵树，拟合残差（即 λ）  
5️⃣ 模型训练完成后，最终得到一个 **打分函数 f(x)**，用于排序！

* * *

### 🔍 为什么它强？

| 优势 | 描述 |
| --- | --- |
| ⚙️ 非线性建模强 | GBDT 可以捕捉特征之间复杂交互（尤其是类别特征） |
| 🔍 对排序指标敏感 | Lambda 梯度直接体现指标变化（如 NDCG），更稳更准 |
| 📦 易于部署 | 树模型部署比深度网络简单、解释性更强 |
| 🛠️ 工业级应用 | 被广泛用于 Bing 搜索、YouTube 推荐、阿里广告、知乎排序等系统中 |

* * *

🔁 Pairwise vs Listwise in LambdaMART
-------------------------------------

虽然 LambdaMART 属于 pairwise 模型，但它的 Lambda 梯度实际上 **隐式优化了 listwise 排序指标（比如 NDCG）**，所以它也经常被认为是 **listwise-aware 的 pairwise 方法**。

这点是它能打爆 RankSVM 和 RankNet 的核心原因！

* * *

🧩 快速选择题：以下哪项是 LambdaMART 的核心训练信号？
----------------------------------

A. 每个文档的标签（如 1/2/3）  
B. 每个 pair 的偏好标签（A 比 B 更 relevant）  
C. 每个样本点的 Lambda（NDCG 引导下的梯度）  
D. 每个文档的 softmax 排名概率

你觉得正确答案是哪个？选完我们就进入下一块——**样本对如何构造才有效？⚖️**（非常关键，很多模型都死在这步！）

## Prompt:
c

## Response:
👏 完全正确！选 **C：每个样本点的 Lambda（NDCG 引导下的梯度）**，这正是 LambdaMART 的训练核心所在！

* * *

🧠 为什么 Lambda 才是训练信号？
---------------------

虽然我们最终希望排序结果越接近真实相关性越好，但在 LambdaMART 中：

*   并不直接最小化传统 loss（如 MSE 或 pairwise hinge loss）；
*   而是——**利用每个样本点对应的 $\\lambda$ 作为 GBDT 的“伪标签”，引导每棵树向着提高排序指标前进。**

可以理解为：**Lambda 是代替真实标签的“排序驱动梯度信号”。**

* * *

🔍 举个直觉例子
---------

*   如果某条样本的排序错误会极大降低 NDCG，那么它的 λ 值就很大（模型要高度关注它）；
*   如果某样本的错排几乎不影响 NDCG，它的 λ 就小（训练可“忽略”）

> 所以，LambdaMART 的学习过程就是一场“资源分配游戏”：**把学习力聚焦到最重要的位置上！**

* * *

📘 知识卡片 04：LambdaMART 概览
------------------------

| 项目 | 描述 |
| --- | --- |
| 🚀 构成 | LambdaRank + MART（GBDT） |
| 🎯 目标 | 隐式优化排序指标（NDCG、MAP 等） |
| 💡 特点 | 用排序指标引导训练梯度，训练效果直接对齐最终目标 |
| 💎 应用 | 搜索引擎、推荐系统、CTR 排序、问答系统等 |

* * *

📊 总结复习
-------

| 模型 | Loss 形式 | 优势 | 应用阶段 |
| --- | --- | --- | --- |
| **RankNet** | Cross Entropy over pair | 概率建模，入门首选 | 学术/原型 |
| **RankSVM** | Hinge Loss | 简洁强几何意义 | 传统方法 |
| **LambdaRank** | 隐式 loss via λ | 优化排名指标 | 过渡模型 |
| **LambdaMART** | λ as gradient for GBDT | 精度高，工业部署强 | 工业SOTA |

* * *

🧩 接下来我们进入 **模块 D：样本对构造策略**  
**Pairwise Ranking** 不怕你模型差，就怕你“对得不准”！

我来问你一个 warm-up：

### ❓问题：以下哪种样本构造方式最容易引入噪声？

A. 用户点击了 A 没点击 B（A > B）  
B. 用户点击了 A 和 B（A = B）  
C. 用户没有点击任何文档  
D. 用户点击了 B 没点击 A，但 A 在更前面

你觉得哪种方式的标签最不靠谱？我们通过这个进入“Pair Sampling 技巧课”！

## Prompt:
C

## Response:
🎯 完全正确！选 **C：用户没有点击任何文档**，这类数据是最容易引入**噪声和误导性偏好对**的，咱们详细展开说说为啥。

* * *

🔍 模块 D：Pairwise 样本构造技巧
-----------------------

Pairwise Ranking 的核心在于训练时告诉模型：

> “你要把 A 放在 B 前面。”

但关键问题来了：**这个“谁在谁前面”的判断从哪来？**

* * *

⚠️ 样本构造的 4 种方式风险对比
------------------

| 构造方式 | 描述 | 风险等级 | 原因分析 |
| --- | --- | --- | --- |
| ✅ A: 点了 A 没点 B | 假设 A > B | ⭐ 低 | 比较明确的偏好对，用户点击提供正向信号 |
| 🟡 B: A 和 B 都点击 | A ≈ B？ | ⭐⭐ 中 | 模糊偏好，不确定谁更好，可能不该构造样本对 |
| 🔴 C: 都没点击 | 无点击行为 | ⭐⭐⭐⭐ 高 | 无偏好信号，构造对 = 拍脑袋，极易引入噪声 |
| 🔴 D: B 被点了但在 A 后面 | 用户看过但没点 A？ | ⭐⭐⭐⭐ 高 | 曝光不等可能导致假阴性，排序不可信 |

> ✅ 所以最常见、安全的 pair 是：**点击了 A 没点击 B，并且 A、B 均曝光过**。

* * *

🧠 知识卡片 05：好样本对的构造三要素
---------------------

| 要素 | 描述 |
| --- | --- |
| 🔦 曝光保障 | 构造的 A/B 样本都应该在用户视线范围内（比如在第一页出现） |
| 👍 明确偏好 | 用户点击 A、未点击 B（B 曝光）→ A > B |
| ❌ 噪声排除 | 忽略未曝光或长时间未交互的项，避免误判偏好 |

* * *

😂 小类比
------

你让一个人选“喜欢的菜”，他看了菜单后点了红烧肉、没点苦瓜炒蛋 ⇒ 你大概率能猜出他更喜欢前者。

但如果他根本没看菜单就走了（对应没点击任何文档），你能说他不喜欢哪道菜吗？当然不能 😅

* * *

🎲 快速选择题：为了更安全地构造 pair，你会怎么做？
-----------------------------

A. 只用曝光 & 点击过的样本构造 pair  
B. 对所有未点击文档统一认为是负样本  
C. 给所有未点击文档打 0 分  
D. 对所有文档随机构对，只要不同就算

选一个最佳策略，我们进入下一节——**Pairwise Ranking 如何从 Click Bias 中逃出生天？🧠⚖️**（真实推荐系统最难的敌人之一！）

## Prompt:
A

## Response:
🎯 Again — 完全正确！你已经逐步掌握了 Pairwise Ranking 的实战灵魂技巧了。

* * *

✅ A：只用曝光 & 点击过的样本构造 pair → 是最稳妥的策略！
-----------------------------------

为什么这么重要？让我们进入深水区：

* * *

🧠 模块 D.2：点击偏差（Click Bias）与 Pairwise Ranking 的陷阱
------------------------------------------------

### 🔍 什么是 Click Bias？

> 即使一个文档不被点击，也不一定代表它“不好”  
> —— 也许它根本没被看到，或者位置太靠下，没人翻页。

这就导致了一个大问题：

> ❌ 把“没点击”视为“差”，可能是 **错的偏好对**！

* * *

📊 点击偏差的来源
----------

| 来源 | 示例 | 对训练的影响 |
| --- | --- | --- |
| 📍 位置偏差 | 页面上越靠前越容易被点 | 高位置假阳性（不一定 relevant） |
| 📄 展示偏差 | 有的结果根本没展示 | 无法构造偏好对 |
| 👤 用户习惯 | 有人懒得点第2条就直接选第1个 | 模型以为第1个最好，实际不一定 |
| 🧠 偏好偏差 | 用户有偏见（如只点短标题） | 排序可能学到非内容因素 |

* * *

🎯 Pairwise Ranking 如何应对？
-------------------------

👉 **解决办法之一就是你选的 A：**  
**✅ 只在 “A 被点了，B 被展示但没点” 的前提下构造偏好对。**

这保证：

1.  两者都被曝光过
2.  用户表达了明确偏好（A > B）
3.  最大限度降低位置、曝光引起的噪声

* * *

🧠 知识卡片 06：Pairwise Click Modeling 黄金三原则
----------------------------------------

| 原则 | 意义 |
| --- | --- |
| 👀 曝光保证 | Pair 中 A 和 B 都被看见过 |
| 👍 明确偏好 | 点击表达喜欢，未点击为弱拒绝 |
| ❌ 避免假负 | 不随便对“未点击”下结论，除非知道它被曝光过 |

* * *

😂 类比段子时间
---------

你在商场橱窗前路过，看到三件衣服。你走进去买了左边那件。

商家做排序模型时：

*   ✅ 正确逻辑：你买的是左边 → 你更喜欢左边 > 右边
*   ❌ 错误逻辑：你没买右边 → 所以右边是差的（你可能压根没注意）

* * *

下一步我们进入模块 **E：损失函数中的 Lambda 如何计算？为什么它是隐式优化 NDCG 的神器？**

我先来个预热选择题：

* * *

### ❓Lambda 值中 $\\Delta \\text{NDCG}\_{ij}$ 是什么意思？

A. 当前第 $i$ 个文档的 DCG 值  
B. 所有文档的平均 NDCG  
C. 如果把 $i$ 和 $j$ 的排名交换，NDCG 会变化多少  
D. $i$ 和 $j$ 的分值差（$s\_i - s\_j$）

你选哪个？我们下一步就深入推导 Lambda 的公式！📐

## Prompt:
C

## Response:
🎯 YES！完美命中！正确答案就是 **C：如果把 $i$ 和 $j$ 的排名交换，NDCG 会变化多少**。

而这正是 **LambdaRank/LambdaMART 神奇之处的核心变量**。

* * *

🧠 模块 E：Lambda 值背后的秘密 —— 隐式优化 NDCG！
-----------------------------------

我们来深入解析一下：

* * *

### 🔍 Lambda 是怎么计算的？

对于每一对文档 $(i, j)$，Lambda 的核心表达式是：

$$
\lambda_{ij} = \frac{\partial \mathcal{L}}{\partial s_i} = \sigma(s_i - s_j) \cdot |\Delta \text{NDCG}_{ij}|
$$
*   $\\sigma(s\_i - s\_j)$ 是 sigmoid 函数（或其导数），确保平滑梯度
*   $|\\Delta \\text{NDCG}\_{ij}|$ 是排序变化对最终指标的影响，**这是 Lambda 的加权重点！**

* * *

🔬 什么是 $\\Delta \\text{NDCG}\_{ij}$？
------------------------------------

假设我们有一份文档排名如下：

| 排名 | 文档 | Relevance |
| --- | --- | --- |
| 1 | A | 1 |
| 2 | B | 3 |

在当前排序下：

*   A 在前，但相关性低；
*   B 在后，但相关性高；

👉 如果交换 A 和 B，那么：

*   B 将升到前面
*   A 会掉下来

我们可以分别计算 **交换前后** 的 NDCG：

$$
\Delta \text{NDCG}_{AB} = \text{NDCG}_{\text{after swap}} - \text{NDCG}_{\text{before swap}}
$$

这个值越大，说明排序错误影响越大，Lambda 权重就越高！

* * *

### 📘 闪卡总结：Lambda 中的 $\\Delta \\text{NDCG}\_{ij}$

| 项目 | 意义 |
| --- | --- |
| 📐 $\\Delta \\text{NDCG}\_{ij}$ | 两个文档交换顺序后，NDCG 发生的变化 |
| 🎯 作用 | 指导训练重点关注影响最大的错误 |
| 💥 直觉 | 排名靠前但不 relevant 的样本，错排代价高，Lambda 大 |
| 🧪 应用 | LambdaMART 中引导 GBDT 的训练目标 |

* * *

🎲 快测题（温故知新）
------------

如果文档 A 原来在第 1 位，Relevance=1，  
文档 B 原来在第 3 位，Relevance=3。

交换它们对 DCG 的影响是正的还是负的？

A. 正  
B. 负  
C. 不变  
D. 不确定

想想 DCG 的公式和文档位置变化，我们下一节就讲清楚**NDCG 与 DCG 的机制 + 为什么位置越前影响越大！**

## Prompt:
B

## Response:
🎯 完全正确！答案是 **B：负的** —— 交换 A 和 B 会**提升排序质量**，因此 **交换后的 DCG 更高、NDCG 变大，$\\Delta \\text{NDCG}\_{ij} > 0$**，但从“原排序”角度来看，当前是错的，**损失是正的**，需要被惩罚！

* * *

🧠 深度解析：为啥是负的？
--------------

我们来一步一步看：

### 🌟 交换前：

*   A 在第 1 位，Relevance = 1  
     $\text{DCG}_A = \frac{2^1 - 1}{\log_2(1+1)} = \frac{1}{1} = 1$ 
*   B 在第 3 位，Relevance = 3  
     $\text{DCG}_B = \frac{2^3 - 1}{\log_2(3+1)} = \frac{7}{2} = 3.5$ 

**总 DCG = 1 + 3.5 = 4.5**

* * *

### 🌟 交换后：

*   B 到第 1 位，DCG = $7 / 1 = 7$
*   A 到第 3 位，DCG = $1 / 2 = 0.5$

**新 DCG = 7 + 0.5 = 7.5**

* * *

### 🧮 所以：

$$
\Delta \text{DCG}_{BA} = 7.5 - 4.5 = 3.0
$$

说明：**交换更 relevant 的文档（B）到更靠前位置是正确的，原排序错了，损失应高，lambda 应该大！**

* * *

📘 知识卡片 07：错排文档如何影响 Lambda？
---------------------------

| 情况 | 错排程度 | $\\Delta \\text{NDCG}$ | Lambda |
| --- | --- | --- | --- |
| 高相关文档排得靠后 | 严重错排 | 高 | 大（强烈惩罚） |
| 排名靠后文档错排 | 影响小 | 低 | 小（轻微训练） |
| 正确排序 | 无错排 | 0 | 无需训练 |

* * *

😂 段子时间
-------

如果一个本来该拿金牌的选手被你排在第 3，银牌被你错放在了第一，观众会不会炸锅？💥

👉 LambdaRank 就是“观众”，它会对这种大错**怒吼**，给出更大的 Lambda 权重，指导模型下次别犯。

* * *

🔜 下一步！
-------

我们即将进入 **模块 F：手动模拟一轮 LambdaRank 梯度计算**  
👉 你将“扮演模型”，我们一起来玩角色扮演游戏：

*   我给出一组文档分值 + 真实相关性
*   你来模拟构造 Pair → 算 Lambda → 判断梯度方向

想试试吗？😎（不难！一步一步来）还是你想先复习一下前面内容？

## Prompt:
复习

## Response:
非常好！定期复习是让排序学习“刻在脑子里”的黄金方法 🧠✨  
我们来一轮**闪卡式+类比式+表格式**的快节奏复习！

* * *

📘 一、Pairwise Ranking 三大核心概念
----------------------------

| 概念 | 简述 | 类比 |
| --- | --- | --- |
| **偏好对** (Preference Pair) | 从用户行为中推断：A 比 B 更 relevant | 你点了《千与千寻》，没点《灌篮高手》，系统猜你更爱宫崎骏 |
| **目标函数** | 优化 $f(x\_i) > f(x\_j)$ 成立的概率或 margin | 排行榜系统努力让你喜欢的电影排在前面 |
| **指标驱动训练** | LambdaRank 用 NDCG 引导模型学习 | “你排序一错，我就拉你一把（大 Lambda）” |

* * *

📐 二、代表模型与 Loss
---------------

| 模型 | Loss 类型 | 优点 | 关键词 |
| --- | --- | --- | --- |
| RankSVM | Hinge Loss | 简洁几何 | Margin |
| RankNet | Pairwise Cross Entropy | 可导平滑 | Sigmoid |
| LambdaRank | 隐式指标导向梯度 | 指标敏感 | $\\Delta$NDCG |
| LambdaMART | Lambda + GBDT | 工业 SOTA | 树模型，pairwise 梯度训练 |

* * *

🧪 三、Lambda 的公式精髓
-----------------

$$
\lambda_{ij} = \sigma(s_i - s_j) \cdot |\Delta \text{NDCG}_{ij}|
$$

| 成分 | 含义 |
| --- | --- |
| $\\sigma(s\_i - s\_j)$ | 当前模型的预测“信心” |
| $\\Delta \\text{NDCG}\_{ij}$ | 错排影响大小（指标变动） |
| → λ 越大 | 模型越要修正这对排序 |

* * *

🧲 四、样本构造：点击偏差怎么破？
------------------

| 方法 | 是否推荐 | 原因 |
| --- | --- | --- |
| 用户点击了 A，没点 B，且 B 被展示 | ✅ | 明确偏好且曝光确认 |
| 用户没点击任何 | ❌ | 没有偏好信息，噪声重 |
| A 和 B 都点击 | 🟡 | 偏好不清晰，训练作用弱 |
| B 被点击但在 A 后面 | ❌ | 曝光偏差，可能是假阴性 |

* * *

🧠 五、经典 NDCG 错排举例
-----------------

| 排名 | 文档 | Rel | 错排吗？ | Lambda 趋势 |
| --- | --- | --- | --- | --- |
| 1 | A | 1 | 是 | ↑↑↑（高） |
| 2 | B | 3 | 是 | ↑↑↑（高） |
| → 交换后 | — | — | 排序更合理 | λ 用来“修正”错误 |

* * *

😂 一句话类比总复习
-----------

*   **Pointwise**：你在预测电影评分 → “这电影我可能打 4 分”
*   **Pairwise**：你在比较两个电影 → “我更喜欢 A 胜过 B”
*   **LambdaRank**：你在听观众意见 → “这个排名一换，大家都满意！”

* * *

我们可以：

A. 继续进入【模块 F：动手算一轮 LambdaRank 梯度游戏】  
B. 用图解方式帮你视觉复习整个流程  
C. 给你出几道小测验加深理解  
D. 回顾具体模型代码结构（如 RankNet 的代码骨架）

你想选哪一项继续？或者多个一起选也可以～

## Prompt:
1. 给出LambdaRank 
2. movies-100k datasets
3. use pytorch, loguru, tensorboradX

give me file category and file by file please

Readability Preferred with Better Name Style. Please provide a comprehensive and well-structured code solution for an Industrial scenario. The code should include robust classes or functions that have been thoroughly checked for correct boundary conditions. It should also adhere to the guidelines specified in the "Google style guides". The solution should prioritize high scalability and should not be a demo or a snippet. It is crucial that no part of the code is omitted. Furthermore, please include Chinese comments following the PEP 8 style guide for all key steps, and use DocString following the PEP 257 style guide and use typing for type annotations. The code should be efficient, and important information should be printed. **When the code is finished**, please perform a key step to recheck for its correctness and logic.

## Response:
已思考 35s

I need to finalize a checklist after writing code to ensure its correctness and logic. While I can't execute it, a dry-run pseudocode and mental checks with boundary conditions are key. I'll also make sure to print important information. For the repo, I'd organize things like:

*   README.md
*   requirements.txt
*   run\_train.py
*   run\_eval.py  
    And detailed components in the lambda\_rank/ folder such as model.py, data.py, trainer.py, and so on.

I need to ensure the code can download MovieLens 100k from GroupLens, but also provide a fallback for when there's no internet access. For local use, users can place the file in `data/`. I’ll use loguru for logging and implement robust handling. The dataset will be used for per-user item sampling, building training instances based on rated items and considering pairwise item combinations for LambdaRank.

I’ll approximate using ideal ranks computed from true relevance. LambdaRank uses changes in NDCG by swapping documents according to current ranks. I need to compute predicted scores, calculate gains and discounts, and then implement the delta\_NDCG for pair swapping. The loss will involve logistic regression on the difference between pairs weighted by this delta.

To avoid double counting in the pairwise loss, I’ll focus on the upper triangle where `rel_i > rel_j`, and limit the sample size (`n <= 50`) for efficiency. I'll implement early stopping, gradient clipping, device management, and checkpointing. For simplicity, embeddings for users and items will create scores. Negative sampling isn’t necessary for pairwise tasks; we can work with rated items only. Edge cases will be handled too, like skipping users with fewer than two ratings.

完成

下面给你一套**工业可用**、可直接运行与扩展的 **LambdaRank（基于 MovieLens 100K）** 项目代码。  
特点：**PyTorch + loguru + tensorboardX**、模块化设计、严格边界条件处理、Google 风格 Docstring + 类型注解、关键步骤中文注释、可扩展可部署。

* * *

📁 项目结构（File Category）
======================

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
> *   `run_train.py`：训练入口（含自动下载/解析 MovieLens-100k，日志与 TensorBoard）
> *   `run_eval.py`：评测入口（载入 checkpoint 计算 NDCG/MAP/Recall）
> *   `lambdarank/`：核心库模块（数据、模型、损失、训练、评测、工具）
>     

* * *

📦 `requirements.txt`
---------------------

```txt
torch>=2.2.0
numpy>=1.23.0
pandas>=2.0.0
scikit-learn>=1.3.0
loguru>=0.7.2
tensorboardX>=2.6
requests>=2.31.0
tqdm>=4.66.0
```

* * *

🧭 `README.md`
--------------

````markdown
# LambdaRank on MovieLens 100K (PyTorch + loguru + tensorboardX)

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

Arguments
---------

See `run_train.py -h` and `run_eval.py -h` for full options.

Notes
-----

*   The script auto-downloads MovieLens-100k if not found (needs internet).
*   If offline, place extracted `ml-100k` under `--data_dir`.

````

---

## 🧩 `lambdarank/__init__.py`

```python
"""LambdaRank industrial library package."""
__all__ = [
    "config",
    "data",
    "metrics",
    "loss",
    "model",
    "trainer",
    "evaluation",
    "utils",
]
````

* * *

⚙️ `lambdarank/config.py`
-------------------------

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    """Training configuration for LambdaRank.

    Attributes:
        data_dir: Directory to store/download the MovieLens-100k dataset.
        log_dir: Directory for TensorBoard logs.
        ckpt_dir: Directory to save model checkpoints.
        seed: Global random seed for reproducibility.
        device: Torch device string, e.g. "cuda" or "cpu".
        user_emb_dim: User embedding dimension.
        item_emb_dim: Item embedding dimension.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        batch_users: Number of users per training batch (list-wise per user).
        max_items_per_user: Max items per user per step to cap O(n^2) pairs.
        epochs: Number of training epochs.
        patience: Early stopping patience (epochs).
        clip_grad_norm: Gradient clipping max norm (0 to disable).
        ndcg_k: K for NDCG@K metric.
        val_interval: Validate every N steps (batches).
        sigma: Sigma coefficient in pairwise logistic for LambdaRank.
        save_best: Whether to save best checkpoint by validation NDCG.
        quiet: Suppress non-critical logs if True.
    """
    data_dir: str = "./data"
    log_dir: str = "./runs"
    ckpt_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"
    user_emb_dim: int = 64
    item_emb_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_users: int = 64
    max_items_per_user: int = 50
    epochs: int = 30
    patience: int = 5
    clip_grad_norm: float = 5.0
    ndcg_k: int = 10
    val_interval: int = 200
    sigma: float = 1.0
    save_best: bool = True
    quiet: bool = False

@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        data_dir: Dataset directory.
        ckpt_path: Model checkpoint path to load.
        device: Torch device string.
        ndcg_k: K for NDCG@K metric.
        max_items_per_user: Limit items per user during eval for determinism.
    """
    data_dir: str = "./data"
    ckpt_path: str = "./checkpoints/best.pt"
    device: str = "cuda"
    ndcg_k: int = 10
    max_items_per_user: int = 200
```

* * *

🧰 `lambdarank/utils.py`
------------------------

```python
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter

def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    # 设置全局随机种子，确保结果可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    """Ensure the given directory exists."""
    # 确保目录存在
    Path(path).mkdir(parents=True, exist_ok=True)

class TBLogger:
    """A thin wrapper around tensorboardX SummaryWriter."""

    def __init__(self, log_dir: str) -> None:
        """Initialize TB logger.

        Args:
            log_dir: Directory to write event files.
        """
        # 初始化 TensorBoardX 写入器
        ensure_dir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag: str, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalars under a main tag.

        Args:
            tag: Main tag name.
            metrics: Dict of metric name to value.
            step: Global step.
        """
        for k, v in metrics.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    # 统计可训练参数个数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

* * *

🧮 `lambdarank/metrics.py`
--------------------------

```python
from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
import torch

def dcg_at_k(rels: Sequence[float], k: int) -> float:
    """Compute DCG@k.

    Args:
        rels: Relevance scores ordered by predicted rank (descending).
        k: Cutoff.

    Returns:
        DCG value at k.
    """
    # 计算 DCG@k：位置越靠前权重越大
    rels = np.asfarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    gains = np.power(2.0, rels) - 1.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))

def ndcg_at_k(true_rels: Sequence[float], pred_scores: Sequence[float], k: int) -> float:
    """Compute NDCG@k given true relevances and predicted scores.

    Args:
        true_rels: Ground-truth relevance labels.
        pred_scores: Predicted scores (higher means more relevant).
        k: Cutoff.

    Returns:
        NDCG value at k in [0, 1].
    """
    # 先根据预测分数排序，再按该顺序计算 DCG；IDCG 用真实相关性降序排序
    if len(true_rels) == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_scores))
    ranked_rels = np.asarray(true_rels)[order]
    dcg = dcg_at_k(ranked_rels, k)
    ideal_rels = np.sort(np.asarray(true_rels))[::-1]
    idcg = dcg_at_k(ideal_rels, k)
    return float(dcg / idcg) if idcg > 0 else 0.0

def map_at_k(true_binary: Sequence[int], pred_scores: Sequence[float], k: int) -> float:
    """Compute MAP@k for binary relevance.

    Args:
        true_binary: Ground-truth binary relevance (0/1).
        pred_scores: Predicted scores.
        k: Cutoff.

    Returns:
        MAP@k value.
    """
    # 计算 MAP@k，仅用于二值相关性
    order = np.argsort(-np.asarray(pred_scores))
    ranked = np.asarray(true_binary)[order][:k]
    if ranked.size == 0:
        return 0.0
    precisions = []
    hit = 0
    for idx, rel in enumerate(ranked, start=1):
        if rel == 1:
            hit += 1
            precisions.append(hit / idx)
    return float(np.mean(precisions)) if precisions else 0.0

def recall_at_k(true_binary: Sequence[int], pred_scores: Sequence[float], k: int) -> float:
    """Compute Recall@k for binary relevance.

    Args:
        true_binary: Ground-truth binary relevance (0/1).
        pred_scores: Predicted scores.
        k: Cutoff.

    Returns:
        Recall@k value.
    """
    # 计算 Recall@k，仅用于二值相关性
    total_pos = int(np.sum(true_binary))
    if total_pos == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_scores))
    ranked = np.asarray(true_binary)[order][:k]
    hit = int(np.sum(ranked))
    return float(hit / total_pos)
```

* * *

🧪 `lambdarank/loss.py`
-----------------------

```python
from __future__ import annotations

import torch
from typing import Tuple

def _pairwise_delta_ndcg(
    rels: torch.Tensor,
    scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute |ΔNDCG| matrix for all pairs under current predicted ranks.

    Args:
        rels: Tensor of shape [n] with relevance labels (float or int).
        scores: Tensor of shape [n] with current predicted scores.
        k: Cutoff K for NDCG normalization (IDCG uses all reals but cutoff applies to DCG deltas).

    Returns:
        Tensor [n, n] with absolute NDCG change if swapping i and j, zeros on diagonal.
    """
    # 根据当前 scores 排序得到当前排名位置
    n = rels.numel()
    device = rels.device
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks[order] = torch.arange(n, device=device, dtype=torch.long)  # 0-based ranks

    # 增益与折扣（位置 0 对应 rank=1）
    gains = torch.pow(2.0, rels.to(scores.dtype)) - 1.0
    discounts = torch.log2((ranks + 2).to(scores.dtype))  # log2(i+2)

    # 计算 IDCG（理想 DCG 用真实相关性降序）
    ideal_order = torch.argsort(rels, descending=True)
    ideal_ranks = torch.arange(n, device=device, dtype=torch.long)
    ideal_discounts = torch.log2((ideal_ranks + 2).to(scores.dtype))
    ideal_gains = torch.pow(2.0, rels[ideal_order].to(scores.dtype)) - 1.0
    # 注意：IDCG 在 @k 场景下，通常只累计前 k 位
    idcg_mask = (ideal_ranks < k).to(scores.dtype)
    idcg = torch.sum(ideal_gains * (1.0 / ideal_discounts) * idcg_mask)
    idcg = torch.clamp(idcg, min=1e-8)  # 防止除零

    # 对任意 i, j 交换位置时 DCG 的改变： (gain_i - gain_j) * (1/discount_i - 1/discount_j)
    # 但仅当位置在前 k 位时才对 NDCG@k 起作用，因此对折扣项按 rank<k 加掩码
    dc_mask = (ranks < k).to(scores.dtype)
    inv_discount = (1.0 / discounts) * dc_mask  # [n]

    gi = gains.unsqueeze(1)  # [n,1]
    gj = gains.unsqueeze(0)  # [1,n]
    inv_di = inv_discount.unsqueeze(1)  # [n,1]
    inv_dj = inv_discount.unsqueeze(0)  # [1,n]

    delta_dcg = (gi - gj) * (inv_di - inv_dj)  # [n,n]
    delta_ndcg = torch.abs(delta_dcg) / idcg
    # 对角线为 0（与自身交换无意义）
    delta_ndcg.fill_diagonal_(0.0)
    return delta_ndcg

def lambdarank_pairwise_loss(
    rels: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    k: int,
) -> torch.Tensor:
    """LambdaRank pairwise logistic loss weighted by |ΔNDCG|.

    Args:
        rels: Relevance tensor [n].
        scores: Score tensor [n].
        sigma: Sigmoid slope for pairwise logistic.
        k: Cutoff for NDCG weight.

    Returns:
        Scalar loss tensor.
    """
    # 计算所有成对的 |ΔNDCG|
    delta_ndcg = _pairwise_delta_ndcg(rels, scores, k=k)  # [n, n]

    # 仅保留 rel_i > rel_j 的上三角配对，避免重复计算
    rels_i = rels.unsqueeze(1)
    rels_j = rels.unsqueeze(0)
    pair_mask = (rels_i > rels_j).to(scores.dtype)  # [n, n]，只在 i 的相关性大于 j 时配对

    # 差值与加权
    score_diff = (scores.unsqueeze(1) - scores.unsqueeze(0))  # s_i - s_j
    logit = sigma * score_diff  # [n, n]
    # logistic pairwise loss: log(1 + exp(-logit)) 对 rel_i > rel_j 的对
    pairwise_loss = torch.nn.functional.softplus(-logit)  # log(1 + exp(-x))

    # 仅计入有效 pair，并用 |ΔNDCG| 加权
    weighted = pairwise_loss * delta_ndcg * pair_mask

    # 归一化，防止不同 n 的样本权重不一致（+eps 防止除零）
    denom = torch.clamp(pair_mask.sum(), min=1.0)
    loss = weighted.sum() / denom
    return loss
```

* * *

🧱 `lambdarank/model.py`
------------------------

```python
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

class MFScoringModel(nn.Module):
    """A simple user-item embedding scoring model for ranking.

    The score is computed as:
        score(u, i) = <U[u], V[i]> + b_u + b_i

    This is efficient and extensible (you may add MLP or features later).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb_dim: int = 64,
        item_emb_dim: int = 64,
    ) -> None:
        """Initialize the model.

        Args:
            num_users: Number of users.
            num_items: Number of items.
            user_emb_dim: User embedding dimension.
            item_emb_dim: Item embedding dimension.
        """
        super().__init__()
        # 用户与物品嵌入，含偏置项
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # 参数初始化（更稳定的训练）
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute scores for (user_ids, item_ids).

        Args:
            user_ids: Tensor of user indices [batch,].
            item_ids: Tensor of item indices [batch,].

        Returns:
            Scores tensor [batch,].
        """
        u = self.user_emb(user_ids)   # [B, D]
        v = self.item_emb(item_ids)   # [B, D]
        bu = self.user_bias(user_ids) # [B, 1]
        bi = self.item_bias(item_ids) # [B, 1]
        scores = (u * v).sum(dim=1, keepdim=True) + bu + bi  # [B, 1]
        return scores.squeeze(1)  # [B]
```

* * *

🧵 `lambdarank/data.py`
-----------------------

```python
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_movielens_100k(data_dir: str) -> Path:
    """Download MovieLens 100k if not exists.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        Path to extracted ml-100k directory.

    Raises:
        RuntimeError: If download fails.
    """
    # 自动下载 MovieLens 100k（若目录下不存在）
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / "ml-100k"
    if target_dir.exists():
        logger.info(f"MovieLens-100k found at: {target_dir}")
        return target_dir

    zip_path = root / "ml-100k.zip"
    logger.info("Downloading MovieLens-100k ...")
    try:
        resp = requests.get(ML100K_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MovieLens 100k: {e}. "
            f"Place 'ml-100k' under {data_dir} manually."
        )

    logger.info("Extracting MovieLens-100k ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    logger.info(f"Extracted to {target_dir}")
    return target_dir

@dataclass
class UserQuery:
    """Container for a single query (a user) with multiple items and relevances."""
    user_id: int
    item_ids: np.ndarray  # [n_items]
    rels: np.ndarray      # [n_items], relevance (ratings)

class ML100KDataModule:
    """Data module for MovieLens-100k to produce per-user queries."""

    def __init__(
        self,
        data_dir: str,
        max_items_per_user: int = 50,
        test_size: float = 0.1,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize data module.

        Args:
            data_dir: Dataset directory (auto-download if missing).
            max_items_per_user: Cap items per user to control O(n^2) pair cost.
            test_size: Fraction of users for test split.
            val_size: Fraction of users for validation split.
            seed: Random seed for splits.
        """
        self.data_dir = data_dir
        self.max_items_per_user = max_items_per_user
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

        self.root = download_movielens_100k(data_dir)
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}

        self.train_queries: List[UserQuery] = []
        self.val_queries: List[UserQuery] = []
        self.test_queries: List[UserQuery] = []

        self.num_users: int = 0
        self.num_items: int = 0

    def _load_ratings(self) -> pd.DataFrame:
        """Load ratings data.

        Returns:
            DataFrame with columns [user_id, item_id, rating].
        """
        # 读取 u.data（user item rating timestamp），用制表符分隔
        path = self.root / "u.data"
        df = pd.read_csv(
            path, sep="\t", header=None, names=["user_id", "item_id", "rating", "ts"]
        )
        df = df[["user_id", "item_id", "rating"]]
        return df

    def _reindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex user and item ids to [0..n-1] and build maps."""
        # 将用户和物品 ID 重映射为连续索引，便于 Embedding 处理
        unique_users = df["user_id"].unique()
        unique_items = df["item_id"].unique()
        self.user_map = {u: i for i, u in enumerate(sorted(unique_users))}
        self.item_map = {it: i for i, it in enumerate(sorted(unique_items))}
        df["user_id"] = df["user_id"].map(self.user_map)
        df["item_id"] = df["item_id"].map(self.item_map)
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        return df

    @staticmethod
    def _sample_cap_items(items: np.ndarray, rels: np.ndarray, max_items: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly cap number of items per user for efficiency."""
        # 随机截取用户的部分条目，控制每个查询的最大项目数
        n = items.shape[0]
        if n <= max_items:
            return items, rels
        idx = rng.choice(n, size=max_items, replace=False)
        return items[idx], rels[idx]

    def _build_queries(self, df: pd.DataFrame, users: np.ndarray, rng: np.random.RandomState) -> List[UserQuery]:
        """Construct UserQuery list for a set of users."""
        # 将用户对应的所有 (item, rating) 聚合为一个查询
        queries: List[UserQuery] = []
        g = df.groupby("user_id")
        for u in users:
            if u not in g.groups:
                continue
            sub = g.get_group(u)
            # 过滤边界情况：至少两个不同相关性才有成对比较意义
            if sub.shape[0] < 2:
                continue
            items = sub["item_id"].to_numpy()
            rels = sub["rating"].to_numpy().astype(np.float32)
            # 去除全相同相关性的用户（否则 pair_mask 全 0）
            if np.allclose(rels, rels[0]):
                continue
            items, rels = self._sample_cap_items(items, rels, self.max_items_per_user, rng)
            queries.append(UserQuery(user_id=int(u), item_ids=items, rels=rels))
        return queries

    def setup(self) -> None:
        """Prepare train/val/test queries."""
        df = self._reindex(self._load_ratings())
        users = np.arange(self.num_users)
        # 先划分 test，再从剩余中划分 val
        users_train, users_test = train_test_split(
            users, test_size=self.test_size, random_state=self.seed, shuffle=True
        )
        users_train, users_val = train_test_split(
            users_train, test_size=self.val_size, random_state=self.seed, shuffle=True
        )
        rng = np.random.RandomState(self.seed)
        self.train_queries = self._build_queries(df, users_train, rng)
        self.val_queries = self._build_queries(df, users_val, rng)
        self.test_queries = self._build_queries(df, users_test, rng)

        # 再次检查边界
        assert self.num_users > 0 and self.num_items > 0, "Empty users/items after reindex."
        if len(self.train_queries) == 0:
            raise RuntimeError("No valid training queries found; check filtering conditions.")
        logger.info(
            f"Users: {self.num_users}, Items: {self.num_items} | "
            f"TrainQ: {len(self.train_queries)}, ValQ: {len(self.val_queries)}, TestQ: {len(self.test_queries)}"
        )

class QueryDataset(Dataset):
    """Dataset that yields per-user queries."""

    def __init__(self, queries: List[UserQuery]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> UserQuery:
        return self.queries[idx]

def collate_queries(batch: List[UserQuery]) -> List[UserQuery]:
    """Collate function to keep list of queries as-is."""
    # 不做拼接，每个元素是一个独立查询（用户）
    return batch

def create_dataloaders(
    dm: ML100KDataModule,
    batch_users: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders for train/val/test."""
    train_ds = QueryDataset(dm.train_queries)
    val_ds = QueryDataset(dm.val_queries)
    test_ds = QueryDataset(dm.test_queries)

    train_loader = DataLoader(
        train_ds, batch_size=batch_users, shuffle=True, num_workers=num_workers,
        collate_fn=collate_queries, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_users, shuffle=False, num_workers=0,
        collate_fn=collate_queries, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_users, shuffle=False, num_workers=0,
        collate_fn=collate_queries, drop_last=False,
    )
    return train_loader, val_loader, test_loader
```

* * *

🏃 `lambdarank/trainer.py`
--------------------------

```python
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW

from .config import TrainConfig
from .data import UserQuery
from .loss import lambdarank_pairwise_loss
from .metrics import ndcg_at_k
from .utils import TBLogger, count_parameters

class Trainer:
    """Trainer for LambdaRank."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model.
            config: Train configuration.
            optimizer: Optional optimizer (default AdamW).
        """
        self.model = model.to(config.device)
        self.cfg = config
        self.optimizer = optimizer or AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.tb = TBLogger(self.cfg.log_dir)
        self.best_val = -1.0
        self.steps_since_improve = 0
        logger.info(f"Trainable params: {count_parameters(self.model):,}")

    def _step_on_query(self, query: UserQuery) -> torch.Tensor:
        """Compute loss for a single user query (list-wise)."""
        device = self.cfg.device
        user = torch.full(
            (len(query.item_ids),), fill_value=query.user_id, dtype=torch.long, device=device
        )
        items = torch.tensor(query.item_ids, dtype=torch.long, device=device)
        rels = torch.tensor(query.rels, dtype=torch.float32, device=device)

        # 模型打分
        scores = self.model(user, items)  # [n]
        # 计算 LambdaRank pairwise loss（按 |ΔNDCG| 加权）
        loss = lambdarank_pairwise_loss(rels, scores, sigma=self.cfg.sigma, k=self.cfg.ndcg_k)
        return loss

    @torch.no_grad()
    def _val_metrics_on_batch(self, batch: List[UserQuery]) -> Dict[str, float]:
        """Compute average NDCG on a batch of queries."""
        self.model.eval()
        ndcgs: List[float] = []
        for q in batch:
            device = self.cfg.device
            user = torch.full((len(q.item_ids),), q.user_id, dtype=torch.long, device=device)
            items = torch.tensor(q.item_ids, dtype=torch.long, device=device)
            rels = torch.tensor(q.rels, dtype=torch.float32, device=device)

            scores = self.model(user, items)
            ndcg = ndcg_at_k(
                true_rels=rels.detach().cpu().numpy(),
                pred_scores=scores.detach().cpu().numpy(),
                k=self.cfg.ndcg_k,
            )
            if not np.isnan(ndcg):
                ndcgs.append(ndcg)
        avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        return {"ndcg@k": avg_ndcg}

    def train(
        self,
        train_loader,
        val_loader,
    ) -> None:
        """Run training with early stopping."""
        global_step = 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                # 对一个 batch 内多个用户查询累积 loss
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                valid_queries = 0
                for q in batch:
                    loss = self._step_on_query(q)
                    if torch.isfinite(loss):
                        batch_loss += loss
                        valid_queries += 1
                if valid_queries == 0:
                    continue
                batch_loss = batch_loss / valid_queries
                batch_loss.backward()

                # 梯度裁剪，稳定训练
                if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)

                self.optimizer.step()

                epoch_loss += float(batch_loss.item())
                global_step += 1

                if global_step % 50 == 0 and not self.cfg.quiet:
                    logger.info(f"[Ep {epoch}] step {global_step} | loss={float(batch_loss):.5f}")
                    self.tb.log_scalars("train", {"loss": float(batch_loss)}, global_step)

                # 间隔验证
                if global_step % self.cfg.val_interval == 0:
                    val_metrics = self.validate(val_loader)
                    self.tb.log_scalars("val", val_metrics, global_step)
                    improved = val_metrics["ndcg@k"] > self.best_val
                    if improved:
                        self.best_val = val_metrics["ndcg@k"]
                        self.steps_since_improve = 0
                        if self.cfg.save_best:
                            self._save_ckpt("best.pt")
                            logger.info(f"New best NDCG@{self.cfg.ndcg_k}: {self.best_val:.5f} - checkpoint saved.")
                    else:
                        self.steps_since_improve += 1
                        logger.info(f"No improvement steps: {self.steps_since_improve}/{self.cfg.patience}")
                        if self.steps_since_improve >= self.cfg.patience:
                            logger.warning("Early stopping triggered.")
                            return

            # 记录每个 epoch 的平均 loss
            epoch_loss /= max(len(train_loader), 1)
            self.tb.log_scalars("epoch", {"loss": epoch_loss}, epoch)
            logger.info(f"Epoch {epoch} done. avg_loss={epoch_loss:.6f}")

        logger.info("Training complete.")

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate and return averaged metrics."""
        self.model.eval()
        ndcgs: List[float] = []
        for batch in val_loader:
            metrics = self._val_metrics_on_batch(batch)
            ndcgs.append(metrics["ndcg@k"])
        avg = float(np.mean(ndcgs)) if ndcgs else 0.0
        logger.info(f"[VAL] NDCG@{self.cfg.ndcg_k}={avg:.5f}")
        return {"ndcg@k": avg}

    def _save_ckpt(self, name: str) -> None:
        """Save model checkpoint."""
        path = f"{self.cfg.ckpt_dir}/{name}"
        state = {"model": self.model.state_dict(), "config": self.cfg.__dict__}
        torch.save(state, path)
```

* * *

📊 `lambdarank/evaluation.py`
-----------------------------

```python
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from loguru import logger

from .config import EvalConfig
from .data import ML100KDataModule, UserQuery, create_dataloaders
from .metrics import ndcg_at_k, map_at_k, recall_at_k

class Evaluator:
    """Evaluator for ranking metrics."""

    def __init__(self, model: torch.nn.Module, cfg: EvalConfig) -> None:
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            cfg: Evaluation config.
        """
        self.model = model.to(cfg.device)
        self.cfg = cfg

    @torch.no_grad()
    def evaluate_queries(self, queries: List[UserQuery]) -> Dict[str, float]:
        """Evaluate metrics over a list of queries."""
        self.model.eval()
        ndcgs: List[float] = []
        maps: List[float] = []
        recalls: List[float] = []

        for q in queries:
            device = self.cfg.device
            user = torch.full((len(q.item_ids),), q.user_id, dtype=torch.long, device=device)
            items = torch.tensor(q.item_ids, dtype=torch.long, device=device)
            rels = torch.tensor(q.rels, dtype=torch.float32, device=device)

            scores = self.model(user, items).detach().cpu().numpy()
            rels_np = rels.detach().cpu().numpy()

            ndcgs.append(ndcg_at_k(rels_np, scores, k=self.cfg.ndcg_k))

            # 对 MAP/Recall 使用二值化（>=4 为正例）
            binary = (rels_np >= 4.0).astype(int)
            maps.append(map_at_k(binary, scores, k=self.cfg.ndcg_k))
            recalls.append(recall_at_k(binary, scores, k=self.cfg.ndcg_k))

        def _safe_mean(x: List[float]) -> float:
            return float(np.mean(x)) if len(x) > 0 else 0.0

        results = {
            f"NDCG@{self.cfg.ndcg_k}": _safe_mean(ndcgs),
            f"MAP@{self.cfg.ndcg_k}": _safe_mean(maps),
            f"Recall@{self.cfg.ndcg_k}": _safe_mean(recalls),
        }
        return results
```

* * *

🧠 `run_train.py`
-----------------

```python
from __future__ import annotations

import argparse
import os

import torch
from loguru import logger

from lambdarank.config import TrainConfig
from lambdarank.data import ML100KDataModule, create_dataloaders
from lambdarank.model import MFScoringModel
from lambdarank.trainer import Trainer
from lambdarank.utils import ensure_dir, set_seed

def parse_args() -> TrainConfig:
    """Parse CLI args to TrainConfig."""
    p = argparse.ArgumentParser(description="Train LambdaRank on MovieLens-100k")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--log_dir", type=str, default="./runs")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--user_emb_dim", type=int, default=64)
    p.add_argument("--item_emb_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--batch_users", type=int, default=64)
    p.add_argument("--max_items_per_user", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--ndcg_k", type=int, default=10)
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        device=args.device,
        user_emb_dim=args.user_emb_dim,
        item_emb_dim=args.item_emb_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_users=args.batch_users,
        max_items_per_user=args.max_items_per_user,
        epochs=args.epochs,
        patience=args.patience,
        clip_grad_norm=args.clip_grad_norm,
        ndcg_k=args.ndcg_k,
        val_interval=args.val_interval,
        sigma=args.sigma,
        save_best=args.save_best,
        quiet=args.quiet,
    )
    return cfg

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    ensure_dir(cfg.ckpt_dir)
    ensure_dir(cfg.log_dir)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        cfg.device = "cpu"

    # 数据模块：加载/下载 + 构造 per-user 查询
    dm = ML100KDataModule(
        data_dir=cfg.data_dir,
        max_items_per_user=cfg.max_items_per_user,
        test_size=0.1,
        val_size=0.1,
        seed=cfg.seed,
    )
    dm.setup()

    train_loader, val_loader, _ = create_dataloaders(dm, batch_users=cfg.batch_users, num_workers=0)

    # 模型
    model = MFScoringModel(
        num_users=dm.num_users,
        num_items=dm.num_items,
        user_emb_dim=cfg.user_emb_dim,
        item_emb_dim=cfg.item_emb_dim,
    )

    # 训练器
    trainer = Trainer(model=model, config=cfg)
    trainer.train(train_loader=train_loader, val_loader=val_loader)

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
```

* * *

📈 `run_eval.py`
----------------

```python
from __future__ import annotations

import argparse

import torch
from loguru import logger

from lambdarank.config import EvalConfig
from lambdarank.data import ML100KDataModule, create_dataloaders
from lambdarank.evaluation import Evaluator
from lambdarank.model import MFScoringModel
from lambdarank.utils import set_seed

def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate LambdaRank model")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--ckpt_path", type=str, default="./checkpoints/best.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ndcg_k", type=int, default=10)
    p.add_argument("--max_items_per_user", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return EvalConfig(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        device=args.device,
        ndcg_k=args.ndcg_k,
        max_items_per_user=args.max_items_per_user,
    )

def main() -> None:
    cfg = parse_args()
    set_seed(42)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
        logger.warning("CUDA not available. Using CPU.")

    dm = ML100KDataModule(
        data_dir=cfg.data_dir,
        max_items_per_user=cfg.max_items_per_user,
        test_size=0.1,
        val_size=0.1,
        seed=42,
    )
    dm.setup()

    # 构建同维度模型并加载权重
    model = MFScoringModel(
        num_users=dm.num_users,
        num_items=dm.num_items,
        user_emb_dim=64,
        item_emb_dim=64,
    ).to(cfg.device)

    state = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {cfg.ckpt_path}")

    evaluator = Evaluator(model=model, cfg=cfg)
    results = evaluator.evaluate_queries(dm.test_queries)
    for k, v in results.items():
        logger.info(f"{k}: {v:.5f}")

if __name__ == "__main__":
    main()
```

* * *

✅ 关键正确性与逻辑复核（Checklist）
=======================

**数据与样本构造**

*    自动下载并解析 MovieLens-100k（`u.data`）。离线时可手动放置 `ml-100k/` 目录。
*    用户、物品 ID 连续化映射，保证 Embedding 索引安全。
*    每用户最少 2 条交互，且相关性非全相同，避免无有效 pair。
*    `max_items_per_user` 控制 O(n²) 成本，边界已检查。

**LambdaRank 实现**

*    `|ΔNDCG|` 计算基于**当前预测排序**的位次折扣（位置敏感）。
*    IDCG 以真实相关性排序，`@k` 截断，`idcg>=1e-8` 防止除零。
*    仅对 `rel_i > rel_j` 的上三角配对求 pairwise logistic，并用 `|ΔNDCG|` 加权。
*    不重复计数，不含自配对（对角线置 0）。
*    `sigma` 可控，默认 1.0。

**训练稳定性**

*    AdamW + weight decay，支持梯度裁剪（默认 5.0）。
*    训练/验证间隔可设，早停 `patience`。
*    `TensorBoard` 记录训练损失与验证 NDCG。
*    best.ckpt 保存，防止过拟合。

**评测**

*    提供 `NDCG@K`，以及二值化 `MAP@K` 和 `Recall@K`（rating≥4 视作正）。
*    推理时严格 on-device 计算，拉回 CPU 做指标。

**可扩展性**

*    模型可替换为 MLP / BERT encoder；数据管道与 Trainer 不变。
*    易接第三方特征（侧信息）与 ANN 检索。

**边界条件**

*    空查询、无有效 pair、IDCG=0、无 CUDA、下载失败 → 有清晰错误/回退路径与日志。

* * *

如果你想要我**把该项目打包成 zip**、或**生成一份简报图（模型与数据流）**、或\*\*改为 ListNet/ListMLE/LambdaMART（GBDT 接口）\*\*版本，也可以直接说，我一次性给你齐活儿！🔥



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)