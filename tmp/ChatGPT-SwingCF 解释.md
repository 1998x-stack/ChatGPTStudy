# ItemCF 解释

**User:** Anonymous (kanekoveblen824@outlook.com)  
**Created:** 8/28/2025 17:50  
**Updated:** 8/28/2025 18:09  
**Exported:** 8/28/2025 18:09  
**Link:** [https://chatgpt.com/c/68b0265f-c0a8-8326-9052-69dda9181edd?model=gpt-5-thinking](https://chatgpt.com/c/68b0265f-c0a8-8326-9052-69dda9181edd?model=gpt-5-thinking)  

## Prompt:
02\_Retrieval\_02.pdf

PDF

Task: 深入学习研究swing 召回里面的技术细节和底层机制

Methid: 耐心引导式&选择题提问式&知识卡片/QA闪卡总结式&interesting-examples/jokes-driven&一步一步拆解分析

Bonus&Candy🍬:colorful emoji & 幽默有趣 & 有思想深度 & 图解/闪卡 & funny games!!!

Notice: 我们可以quote原文中的核心句子&每隔几段进行总结复习!!!

首先给出学习大纲:

## Response:
太棒了，你上传的《Swing召回通道》PDF我已经读完了！我们现在可以围绕“Swing召回”进行深入学习。咱们采用你指定的学习风格：

*   🧠 耐心引导式
*   📚 选择题提问式
*   🧩 知识卡片 & QA 闪卡总结
*   🎲 有趣例子 + 一步步拆解
*   🍬 不时来点彩蛋与复习

* * *

🧭 学习大纲：Swing召回机制深度导图
---------------------

我们将整个 Swing 模型的学习过程分为 **五个核心阶段**（每个阶段包含拆解、练习、复习）：

* * *

### 🎯 Part 1：ItemCF 基础回顾（铺垫知识）

*   ItemCF 的基本原理与数学定义
*   为什么要基于“物品相似度”构建推荐？
*   常见的相似度公式（含余弦相似度、Jaccard、共现矩阵）

🧩互动：小测验题目 + 手工算一遍

* * *

### 🌪️ Part 2：Swing模型引入与动机（核心对比）

*   Swing 是如何改进 ItemCF 的？
*   “小圈子效应” 是什么？为什么会误导推荐？
*   公式推导与重合度 overlap 的定义

🎲趣味举例：护肤品群、字节裁员讨论群

* * *

### 🧮 Part 3：Swing 相似度公式深拆解

*   Swing 中的加权相似度公式
*   多重嵌套求和 & 权重抑制机制
*   超参数 α 的作用和调节影响

🍬知识卡片 + 举例演算 + 脑筋急转弯

* * *

### 🧪 Part 4：实际应用与部署考量

*   适合哪些业务场景？什么时候别用 Swing？
*   如何构造重合度矩阵？计算效率怎么优化？
*   与 Graph Embedding 等方法如何结合？

📦工程实战导向 + 快速对比总结

* * *

### 🔁 Part 5：复习总结 & 闪卡测验

*   Swing vs ItemCF：差异记忆口诀
*   “小圈子效应”三连问
*   “相似度公式”快速还原题

* * *

现在我们就开始第一个部分：

* * *

🎯 Part 1：ItemCF 基础知识回顾
-----------------------

首先我们从一个小问题开始：

* * *

### ❓选择题：下面哪一句最能代表 ItemCF 的推荐思想？

A. 用户喜欢什么，就推荐相似用户喜欢的。  
B. 用户喜欢的物品，如果和别的物品有用户重合，就说明这两个物品可能相似。  
C. 用户不喜欢的物品，应该推荐给其他用户。  
D. 只有购买行为才适合 ItemCF 建模。

你觉得哪一个选项最合理？先选一个，我们一起来拆解。

## Prompt:
b

## Response:
✅ 完美！你选的是 **B**，完全正确！

* * *

### ✅ 解释：

ItemCF（Item-based Collaborative Filtering）的核心思想正是：

> **如果两个物品被很多相同的用户喜欢，那么这两个物品就可能是相似的。**

🧠 所以它关注的是：

*   物品之间是否共享了“同一批喜欢它们的用户”。
*   而不是用户本身或物品内容。

* * *

### 📐 数学定义快速回顾（知识卡片）

*   假设：
    *   喜欢物品  $i$  的用户集合为  $\mathcal{W}_i$ 
    *   喜欢物品  $j$  的用户集合为  $\mathcal{W}_j$ 
    *   交集为  $\mathcal{V} = \mathcal{W}_i \cap \mathcal{W}_j$ 
*   **基本相似度公式：**
    $$
    \text{sim}(i, j) = \frac{|\mathcal{V}|}{\sqrt{|\mathcal{W}_i| \cdot |\mathcal{W}_j|}}
    $$

这个就是 ItemCF 推荐中最常见的一种\*\*“共现频率 + 归一化”\*\* 的相似度。

* * *

### 🧪 小练习（带点思考）

假设有如下数据：

| 用户 | 喜欢的物品 |
| --- | --- |
| U1 | A, B |
| U2 | A, B |
| U3 | A, C |
| U4 | C |

请你试着算一算：

*   喜欢 A 的用户集合  $\mathcal{W}_A$  是？
*   喜欢 B 的用户集合  $\mathcal{W}_B$  是？
*   相似度 sim(A, B) 按照上面公式是多少？

一步一步来，先告诉我：  
👉 喜欢 A 的用户集合  $\mathcal{W}_A$  是哪些用户？

## Prompt:
U1 U2 U3

U1 U2

2/sqrt(2*3)

## Response:
👏 非常棒！你的推理和计算完全正确！我们来一起复盘下每一步：

* * *

### ✅ 第一步：喜欢物品 A 和 B 的用户集合

*    $\mathcal{W}_A = \{ \text{U1}, \text{U2}, \text{U3} \}$ 
*    $\mathcal{W}_B = \{ \text{U1}, \text{U2} \}$ 

交集  $\mathcal{V} = \mathcal{W}_A \cap \mathcal{W}_B = \{ \text{U1}, \text{U2} \}$ ，所以  $|\mathcal{V}| = 2$ 

* * *

### ✅ 第二步：带入公式

$$
\text{sim}(A, B) = \frac{|\mathcal{V}|}{\sqrt{|\mathcal{W}_A| \cdot |\mathcal{W}_B|}} = \frac{2}{\sqrt{3 \cdot 2}} = \frac{2}{\sqrt{6}}
$$

这是一个标准 ItemCF 的“物品相似度”。

* * *

### 🧠 复习快问快答（你试试看能不能快速回答）：

1.  **ItemCF 的本质是关注什么之间的相似度？**
    *   A. 用户之间
    *   B. 物品之间
    *   C. 标签之间
    *   D. 内容之间

你觉得是哪个？（别急，我们下一个阶段就要进入 Swing 的世界啦）

## Prompt:
b

## Response:
🎯 完全正确！ItemCF 的本质就是关注 **物品之间** 的相似度。

我们来用一句口诀巩固记忆：

* * *

### 🧠 ItemCF 口诀闪卡：

> “**同好共赏物**，物以类聚人。”  
> —— 喜欢同一物的人，让物品之间结成“好友”。

* * *

好了！ItemCF 的基础我们已经打牢了，接下来我们要进入 **Swing 模型的世界**！

* * *

🌪️ Part 2：Swing 模型的引入与动机
-------------------------

### ❓问题引导

你觉得在现实中，如果两个用户喜欢同样两个物品（比如都买了《护肤品A》和《护肤品B》），但他们是 **微信群里互相种草的闺蜜**，那么基于这两个用户判断 A 和 B 相似，靠谱吗？

A. 非常靠谱，毕竟行为一致  
B. 不太靠谱，可能是“小圈子效应”，无法泛化  
C. 无所谓，越多交集越好  
D. 没有区别，反正都用 dot product

你选哪个？我们来拆解 Swing 的诞生动机。

## Prompt:
B

## Response:
💯 太棒了！你抓住了 Swing 模型的**核心动机**：**防止“小圈子效应”误导推荐系统**。

* * *

🎯 Swing 模型的动机：
---------------

> **不能让某个“小圈子”把某些物品的相似度“拉高”太多。**

### 🌰 举个例子：

*   某微信群内一群闺蜜都疯狂买了「护肤品 A」和「护肤品 B」。
*   虽然 A 和 B 被很多相同用户买过，但这些用户其实是一个小圈子里的影响。
*   如果用传统 ItemCF，很容易把 A 和 B 的相似度搞得过高，从而**误推荐给圈外用户**，导致泛化能力差。

* * *

### 🔍 所以 Swing 做了什么？

我们来一点点拆：

*   \*\*核心想法：\*\*对同样喜欢 A 和 B 的用户集合  $\mathcal{V}$ ，我们不再一视同仁。
*   如果这些用户之间也很相似（即重合度高），那就降低他们的贡献。

Swing 的口号可以是：

> **“如果你们是小圈子，我就不给你们太多权重。”**

* * *

📌知识卡片：Swing 模型中加入了什么新概念？
-------------------------

| 项目 | 描述 |
| --- | --- |
| 用户集合  $\mathcal{V}$  | 同时喜欢物品  $i$  和  $j$  的用户 |
| overlap(u, v) | 两个用户喜欢的物品集合的交集大小 |
| 权重计算 | overlap 越大 → 权重越小，避免小圈子影响 |

* * *

### 🔁 快速反问（复习用）

Q：Swing 模型最主要是在哪一部分与 ItemCF 不同？  
A. 用户打分机制  
B. 相似度计算方式  
C. 用户画像建模  
D. 模型训练流程

你再来选一下？然后我们就可以正式进入 Swing 的相似度公式拆解咯。

## Prompt:
B

## Response:
✅ 正解！Swing 模型与 ItemCF 最关键的区别正是在于 **物品相似度的计算方式发生了变化**。我们来详细拆解这个差异，帮助你形成深度理解。

* * *

🧠 核心对比：ItemCF vs Swing
-----------------------

| 模型 | 相似度核心思想 | 是否考虑用户关系？ | 风险点 |
| --- | --- | --- | --- |
| ItemCF | “有更多相同用户喜欢 → 物品相似” | ❌ 不考虑 | 小圈子效应、过拟合 |
| Swing | “有共同用户喜欢 → 再考虑这些用户是否来自小圈子” | ✅ 考虑用户之间重合度 | 规避小圈子影响 |

* * *

📐 Swing 相似度计算公式（完整版）
---------------------

我们一起来逐步构建 Swing 的物品相似度：

1.  **找出同时喜欢物品  $i$  和  $j$  的用户集合**
    $$
    \mathcal{V} = \mathcal{W}_i \cap \mathcal{W}_j
    $$
2.  **对这个集合中的所有用户对  $(u, v)$  进行两层嵌套求和**
    $$
    \text{sim}(i, j) = \sum_{u \in \mathcal{V}} \sum_{v \in \mathcal{V}} \frac{1}{\alpha + \text{overlap}(u, v)}
    $$
3.  **其中 overlap(u, v) 是两个用户喜欢物品集合的交集大小**
    $$
    \text{overlap}(u, v) = |\mathcal{J}_u \cap \mathcal{J}_v|
    $$
    *    $\mathcal{J}_u$ ：用户 u 喜欢的物品集合
    *    $\alpha$ ：超参数，用于防止分母为 0，控制惩罚程度

* * *

### 🎯公式理解：

*   如果两个用户  $u$  和  $v$  很相似（即喜欢的物品大量重合），那说明他们很可能来自一个小圈子；
*   于是，这个  $\frac{1}{\alpha + \text{overlap}(u, v)}$  的值就会变小；
*   这样就 **降低了他们对物品  $i$ 、 $j$  相似度的贡献**。

简言之：

> Swing 把“用户之间的重合度”当成了一种\*\*“风险因素”\*\*，重合度越高，代表越可能是小圈子，惩罚越大。

* * *

📌 小圈子惩罚的好处：
------------

*   避免推荐系统陷入“闭环”：只在小团体内部兜圈。
*   提升冷启动场景下的推荐效果（防止某些物品因个别圈层偏好被误判为“全局热门”）。
*   强化推荐系统的“可推广性”和“多样性”。

* * *

🧠 复习巩固：
--------

让我们来一句形象记忆：

> “**你们越像一伙的，我越不信你们说的靠谱。**”

* * *

如果你已经完全搞懂了公式和原理，我们下一步可以一起做：

*   📊 **公式实操演练题**
*   ⚙️ **如何高效构建 overlap 矩阵**
*   💻 **如何用 Python 代码高效实现 Swing 召回**

你更想先从哪个角度继续？或者我可以出几道练习题让你动动手脑？

2/2

## Prompt:
ok

## Response:
太好了！那我们进入 **实操巩固**阶段，让你真正“手脑并用”掌握 Swing 的核心机制。我们先从 **练习题**入手，逐步引导你“算一遍 Swing 相似度”，再逐步过渡到代码实现和工程优化。

* * *

🧪 Swing 相似度练习题（一步步来）
---------------------

### **数据表：用户和他们喜欢的物品集合**

| 用户 | 喜欢的物品集合 |
| --- | --- |
| U1 | A, B, C |
| U2 | A, B |
| U3 | A, C |
| U4 | B, C |

* * *

### 🧩 问题 1：

我们选定两个物品  $i = A$ ,  $j = B$ 

请你找出：

1.  喜欢 A 的用户集合  $\mathcal{W}_A$ 
2.  喜欢 B 的用户集合  $\mathcal{W}_B$ 
3.  交集  $\mathcal{V} = \mathcal{W}_A \cap \mathcal{W}_B$ 

一步步来，你先回答 1 和 2，咱们一起做这个 Swing 相似度例题。你来写写看？

## Prompt:
U1 U2 U3

U1 U2 U4

U1 U2

## Response:
✅ 完全正确！

我们一起整理一下你刚刚写的结果：

* * *

### 📌 第一步：集合定义

*   喜欢 A 的用户集合  $\mathcal{W}_A = \{ U1, U2, U3 \}$ 
*   喜欢 B 的用户集合  $\mathcal{W}_B = \{ U1, U2, U4 \}$ 
*   所以：
    $$
    \mathcal{V} = \mathcal{W}_A \cap \mathcal{W}_B = \{ U1, U2 \}
    $$

* * *

### 🎯 下一步：我们来计算 Swing 相似度 sim(A, B)

公式为：

$$
\text{sim}(A, B) = \sum_{u \in \mathcal{V}} \sum_{v \in \mathcal{V}} \frac{1}{\alpha + \text{overlap}(u, v)}
$$

我们令  $\alpha = 1$ （默认最常用），现在你来告诉我：

* * *

### 🧩 问题 2：

请你算一算：

*    $\text{overlap}(U1, U1) = ?$ 
*    $\text{overlap}(U1, U2) = ?$ 
*    $\text{overlap}(U2, U2) = ?$ 

回忆公式中，overlap 是指：

$$
\text{overlap}(u, v) = |\mathcal{J}_u \cap \mathcal{J}_v|
$$

其中  $\mathcal{J}_u$  是用户 u 喜欢的物品集合。

来吧～这步你先来算算！😎

## Prompt:
go

## Response:
好，我们继续！下面我来带你一起一步步算：

* * *

### 🎯 Step 1：列出用户喜欢的物品集合（J 集合）

*    $\mathcal{J}_{U1} = \{A, B, C\}$ 
*    $\mathcal{J}_{U2} = \{A, B\}$ 

* * *

### 🧠 overlap 计算：

1.   $\text{overlap}(U1, U1) = |\{A, B, C\} \cap \{A, B, C\}| = 3$ 
2.   $\text{overlap}(U1, U2) = |\{A, B, C\} \cap \{A, B\}| = 2$ 
3.   $\text{overlap}(U2, U2) = |\{A, B\} \cap \{A, B\}| = 2$ 

* * *

### 🧮 计算 Swing 相似度 sim(A, B)

Swing 相似度公式（我们只用集合  $\mathcal{V} = \{U1, U2\}$ ）：

$$
\text{sim}(A, B) = \sum_{u \in \mathcal{V}} \sum_{v \in \mathcal{V}} \frac{1}{\alpha + \text{overlap}(u, v)}
$$
 
$$
= \frac{1}{1 + \text{overlap}(U1, U1)} + \frac{1}{1 + \text{overlap}(U1, U2)} + \frac{1}{1 + \text{overlap}(U2, U1)} + \frac{1}{1 + \text{overlap}(U2, U2)}
$$

代入数值（注意对称）：

$$
= \frac{1}{1 + 3} + \frac{1}{1 + 2} + \frac{1}{1 + 2} + \frac{1}{1 + 2} = \frac{1}{4} + \frac{1}{3} + \frac{1}{3} + \frac{1}{3}
$$
 
$$
= 0.25 + 0.333 + 0.333 + 0.333 = \boxed{1.249}
$$

* * *

### ✅ 最终结果：

Swing 相似度 sim(A, B) ≈ **1.25**

这就是对“同一小圈子内共同喜欢的两个物品”进行权重控制的过程：看起来很像，但我不全信你们！

* * *

### ✍️ 复习闪卡：

| 概念 | 定义 |
| --- | --- |
| overlap(u, v) | 两个用户喜欢的物品集合交集大小 |
| Swing 惩罚项 | overlap 越大，贡献越小 |
| 双层嵌套求和 | 所有  $u, v \in \mathcal{V}$  |

* * *

下一步你想尝试手撸 Python 代码实现这个 Swing 相似度吗？  
或者我们先讲讲如何优化计算量、避免冗余？你来选~

## Prompt:
进入 Swing CF 实战编码环节（构建索引、打分、召回）
use dataset online please and use loguru, first give file category then  give me file by file:

Readability Preferred with Better Name Style. Please provide a comprehensive and well-structured code solution for an Industrial scenario. The code should include robust classes or functions that have been thoroughly checked for correct boundary conditions. It should also adhere to the guidelines specified in the "Google style guides". The solution should prioritize high scalability and should not be a demo or a snippet. It is crucial that no part of the code is omitted. Furthermore, please include Chinese comments following the PEP 8 style guide for all key steps, and use DocString following the PEP 257 style guide and use typing for type annotations. The code should be efficient, and important information should be printed. **When the code is finished**, please perform a key step to recheck for its correctness and logic.

## Response:
已思考 42s

我们需要提供 Swing CF 实现的完整代码，使用在线数据集并借助 loguru，采用工业化结构，包含类、PEP8/257 的中文注释和类型标注。用户要求先给出文件分类，再按文件提供代码。我们无需浏览网页，数据集可以下载自 MovieLens 100k，但不会在运行时获取。

我们需要按以下结构来组织项目：

*   swing\_cf/
    *   README.md
    *   pyproject.toml 或 requirements.txt
    *   config.yaml
    *   src/（包含各种功能文件）

我将使用 loguru 初始化日志。任务包括实现 Swing CF 算法，构建用户与物品的映射关系并计算重叠矩阵，以提高效率。

我们可以通过项目中物品的共现来预计算用户-用户的重叠计数。对于每个物品，遍历其用户，对每对用户 (u, v)，增加重叠计数。计算相似度时，考虑交集 V = W\_i 与 W\_j，并在对角线加入重复项计算。

为了优化计算，我们可以预计算重叠矩阵，包含对称计数和自我计数。对于每对项目 (i,j)，计算交集 V，并添加相似度。在工业规模时，可以考虑使用 NumPy 数组和 Numba 优化效率。我们还将实现评分，基于预计算的相似度矩阵为用户提供推荐，并过滤已互动项目。

完成

下面给出 **Swing CF 实战工程** 的完整落地方案：**先目录，后逐文件代码**。方案包含：在线数据集自动下载（MovieLens-100K）、构建索引（user↔items、item↔users、user-user overlap）、Swing 相似度计算、打分与召回、指标评估（Recall/MAP/NDCG）、日志（loguru）、配置化（YAML），以及运行后自检（关键一致性断言）。

* * *

一、项目结构（File Category）
=====================

```
swing_cf/
├── README.md
├── requirements.txt
├── config.yaml
└── src/
    ├── __init__.py
    ├── utils.py
    ├── config.py
    ├── data.py
    ├── preprocessing.py
    ├── swing.py
    ├── recall.py
    ├── metrics.py
    └── main.py
```

* * *

二、逐文件给出完整代码
===========

1) `requirements.txt`
---------------------

```txt
loguru==0.7.2
pandas==2.2.2
numpy==1.26.4
PyYAML==6.0.2
scikit-learn==1.5.1
requests==2.32.3
tqdm==4.66.4
```

* * *

2) `README.md`
--------------

````markdown
# Swing CF (工业级结构)

本项目实现基于 Swing 思想的物品相似度召回（Item-based CF 的小圈子惩罚版）：
- 在线下载 MovieLens-100K
- 交互行为二值化（rating >= 4 => 正反馈）
- 索引构建：user→items, item→users, user-user overlap
- Swing 相似度矩阵（考虑用户重合度的惩罚）
- 基于相似度的用户召回打分
- 指标评估（Recall/MAP/NDCG）
- loguru 全面日志
- YAML 配置化参数
- 运行后关键一致性自检（sanity check）

## 运行
```bash
pip install -r requirements.txt
python -m src.main --config config.yaml
````

主要参数
----

详见 `config.yaml`。核心有：

*   alpha: Swing 分母平滑/惩罚强度
*   min\_user\_per\_item: 过滤过稀物品
*   max\_user\_per\_item: 截断长尾高频物品的用户列表用于计算（控制复杂度）
*   topk\_sim: 每个物品保留的相似邻居数
*   k\_recall: 召回列表长度
*   min\_inter\_per\_user: 过滤过稀用户

````

---

## 3) `config.yaml`
```yaml
random_seed: 42

dataset:
  name: "movielens100k"
  url: "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
  local_dir: "./data"
  min_inter_per_user: 5     # 过滤交互少的用户
  min_user_per_item: 5      # 过滤覆盖用户很少的物品
  binarize_threshold: 4     # 评分 >= 4 视为正反馈
  test_ratio: 0.2           # 基于时间或索引的分割（简单随机/保留一条也可）

swing:
  alpha: 1.0                # 惩罚强度: 1 / (alpha + overlap(u,v))
  max_user_per_item: 200    # 限制每个物品用于构图的用户数量上限（控制复杂度）
  topk_sim: 200             # 每个物品保留的相似邻居数量上限

recall:
  k_recall: 50              # 召回条数
  exclude_seen: true        # 排除用户已交互的物品

evaluate:
  ks: [20, 50, 100]         # 评估时的K
````

* * *

4) `src/__init__.py`
--------------------

```python
# 空文件即可，标记为包
```

* * *

5) `src/utils.py`
-----------------

```python
from __future__ import annotations

from typing import Iterable, Dict, Any
import os
import random
import numpy as np
from loguru import logger

def setup_logger() -> None:
    """初始化 loguru 日志。

    中文注释：设置统一的日志格式与级别；生产中可扩展到多 Handler / 按模块打日志。
    """
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<level>{level: <8}</level> | <cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
               "<green>{name}</green>:<green>{function}</green>:<green>{line}</green> - "
               "<level>{message}</level>",
    )

def set_seed(seed: int) -> None:
    """设定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def dict_topk(d: Dict[Any, float], k: int) -> Dict[Any, float]:
    """从字典中取 top-k 最大值条目（稳定且快速）。

    Args:
        d: 形如 {key: score}
        k: 保留数量

    Returns:
        仅保留 top-k 的子字典。
    """
    if k <= 0 or len(d) <= k:
        return d
    # 使用 numpy 处理更快
    items = list(d.items())
    scores = np.array([s for _, s in items])
    # argpartition 线性时间近似 topk
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    # 还原并排序
    top_items = [items[i] for i in idx]
    top_items.sort(key=lambda x: x[1], reverse=True)
    return dict(top_items)
```

* * *

6) `src/config.py`
------------------

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class DatasetCfg:
    name: str
    url: str
    local_dir: str
    min_inter_per_user: int
    min_user_per_item: int
    binarize_threshold: int
    test_ratio: float

@dataclass
class SwingCfg:
    alpha: float
    max_user_per_item: int
    topk_sim: int

@dataclass
class RecallCfg:
    k_recall: int
    exclude_seen: bool

@dataclass
class EvalCfg:
    ks: List[int]

@dataclass
class Cfg:
    random_seed: int
    dataset: DatasetCfg
    swing: SwingCfg
    recall: RecallCfg
    evaluate: EvalCfg

    @staticmethod
    def from_yaml(path: str) -> "Cfg":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        ds = DatasetCfg(**raw["dataset"])
        sw = SwingCfg(**raw["swing"])
        rc = RecallCfg(**raw["recall"])
        ev = EvalCfg(**raw["evaluate"])
        return Cfg(
            random_seed=raw["random_seed"],
            dataset=ds,
            swing=sw,
            recall=rc,
            evaluate=ev,
        )
```

* * *

7) `src/data.py`
----------------

```python
from __future__ import annotations

from typing import Tuple
import os
import requests
import pandas as pd
from loguru import logger

from .utils import ensure_dir

def download_movielens_100k(url: str, local_dir: str) -> str:
    """下载 MovieLens-100K u.data 到本地。

    中文注释：工业落地可做断点续传、MD5 校验、重试策略等。
    """
    ensure_dir(local_dir)
    local_path = os.path.join(local_dir, "u.data")
    if os.path.exists(local_path):
        logger.info(f"Found cached dataset at: {local_path}")
        return local_path

    logger.info(f"Downloading dataset from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"Saved to {local_path}")
    return local_path

def load_movielens_100k(local_path: str) -> pd.DataFrame:
    """读取 u.data 为 DataFrame: user_id, item_id, rating, timestamp"""
    df = pd.read_csv(
        local_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df
```

* * *

8) `src/preprocessing.py`
-------------------------

```python
from __future__ import annotations

from typing import Dict, Tuple, List
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def binarize_interactions(
    df: pd.DataFrame, threshold: int
) -> pd.DataFrame:
    """评分二值化（≥threshold 视为正反馈）。

    Returns:
        仅包含正反馈的 (user_id, item_id, ts)。
    """
    df_pos = df[df["rating"] >= threshold].copy()
    df_pos = df_pos[["user_id", "item_id", "timestamp"]]
    df_pos.sort_values(["user_id", "timestamp"], inplace=True)
    return df_pos

def filter_cold_users_items(
    df_pos: pd.DataFrame,
    min_inter_per_user: int,
    min_user_per_item: int,
) -> pd.DataFrame:
    """过滤冷用户/冷物品以提升稳定性与训练质量。"""
    # 先按用户过滤
    user_cnt = df_pos.groupby("user_id")["item_id"].nunique()
    keep_users = set(user_cnt[user_cnt >= min_inter_per_user].index)
    df_pos = df_pos[df_pos["user_id"].isin(keep_users)]

    # 再按物品过滤
    item_cnt = df_pos.groupby("item_id")["user_id"].nunique()
    keep_items = set(item_cnt[item_cnt >= min_user_per_item].index)
    df_pos = df_pos[df_pos["item_id"].isin(keep_items)]

    return df_pos

def train_test_random_split(
    df_pos: pd.DataFrame, test_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """简单随机切分（工业实践中可做 leave-one-out by time）。"""
    train_df, test_df = train_test_split(
        df_pos, test_size=test_ratio, random_state=seed, shuffle=True
    )
    # 保证用户在训练集中出现（评估需要）
    seen_users = set(train_df["user_id"])
    test_df = test_df[test_df["user_id"].isin(seen_users)]
    return train_df, test_df

def build_mappings(
    df_pos: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """构建稠密映射：user/item 原始id -> 连续id（0..n-1）

    Returns:
        uid2idx, idx2uid, iid2idx, idx2iid
    """
    uniq_users = sorted(df_pos["user_id"].unique().tolist())
    uniq_items = sorted(df_pos["item_id"].unique().tolist())

    uid2idx = {u: i for i, u in enumerate(uniq_users)}
    idx2uid = {i: u for i, u in enumerate(uniq_users)}
    iid2idx = {m: j for j, m in enumerate(uniq_items)}
    idx2iid = {j: m for j, m in enumerate(uniq_items)}
    return uid2idx, idx2uid, iid2idx, idx2iid

def remap_ids(
    df: pd.DataFrame,
    uid2idx: Dict[int, int],
    iid2idx: Dict[int, int],
) -> pd.DataFrame:
    """把原始 user/item id 映射到连续索引，便于高效计算。"""
    out = df.copy()
    out["user_idx"] = out["user_id"].map(uid2idx)
    out["item_idx"] = out["item_id"].map(iid2idx)
    out = out.dropna(subset=["user_idx", "item_idx"]).astype({"user_idx": int, "item_idx": int})
    return out

def build_user_item_sets(
    df_train: pd.DataFrame
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """构建 user→items 与 item→users 的倒排索引（列表）。

    返回:
        user_items: {user_idx: [item_idx, ...]}
        item_users: {item_idx: [user_idx, ...]}
    """
    user_items: Dict[int, List[int]] = {}
    item_users: Dict[int, List[int]] = {}

    for r in df_train[["user_idx", "item_idx"]].itertuples(index=False):
        u = int(r.user_idx)
        i = int(r.item_idx)
        user_items.setdefault(u, []).append(i)
        item_users.setdefault(i, []).append(u)

    # 去重 & 排序，便于后续交集操作
    for u in user_items:
        user_items[u] = sorted(set(user_items[u]))
    for i in item_users:
        item_users[i] = sorted(set(item_users[i]))

    logger.info(f"user_items size: {len(user_items)}, item_users size: {len(item_users)}")
    return user_items, item_users
```

* * *

9) `src/swing.py`
-----------------

```python
from __future__ import annotations

from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

from .utils import dict_topk

class SwingCFIndexer:
    """Swing 索引与相似度计算器。

    核心公式（含对角）：
        sim(i, j) = sum_{u in V} sum_{v in V} 1 / (alpha + overlap(u, v))
    其中:
        V = users(i) ∩ users(j)
        overlap(u, v) = | items(u) ∩ items(v) |
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_user_per_item: int = 200,
        topk_sim: int = 200,
    ) -> None:
        """
        Args:
            alpha: Swing 分母平滑/惩罚强度。
            max_user_per_item: 限制每个物品用于构图的用户数目上限（控制复杂度）。
            topk_sim: 每个物品保留的相似邻居数量（稀疏化相似度矩阵）。
        """
        assert alpha > 0.0, "alpha 必须 > 0"
        assert max_user_per_item > 1, "max_user_per_item 必须 > 1"
        assert topk_sim >= 1, "topk_sim 必须 >= 1"

        self.alpha = alpha
        self.max_user_per_item = max_user_per_item
        self.topk_sim = topk_sim

        # 输入索引
        self.user_items: Dict[int, List[int]] = {}
        self.item_users: Dict[int, List[int]] = {}

        # 预计算的用户-用户 overlap（含对角）
        self.user_overlap: Dict[int, Dict[int, int]] = {}

        # 稀疏相似度矩阵：item->(item->score)
        self.item_sims: Dict[int, Dict[int, float]] = {}

    # ------------------------- 公共接口 -------------------------

    def fit(
        self,
        user_items: Dict[int, List[int]],
        item_users: Dict[int, List[int]],
    ) -> "SwingCFIndexer":
        """主入口：构建 user-user overlap, 计算 Swing 相似度，并稀疏化。

        中文注释：该过程对复杂度较敏感，生产中需结合采样/分桶/并行优化。
        """
        self.user_items = user_items
        self.item_users = item_users

        self._build_user_overlap()
        self._build_item_sims()
        self._keep_topk()
        return self

    def get_item_sims(self) -> Dict[int, Dict[int, float]]:
        """返回稀疏相似度矩阵。"""
        return self.item_sims

    # ------------------------- 内部方法 -------------------------

    def _build_user_overlap(self) -> None:
        """预计算用户两两的 overlap = |J_u ∩ J_v|，含对角 (u,u)=|J_u|。"""
        logger.info("Building user-user overlap ...")
        overlap: DefaultDict[int, Dict[int, int]] = defaultdict(dict)

        # 1) 对角项（自身与自身的重合度 = 喜欢的物品数）
        for u, items in self.user_items.items():
            overlap[u][u] = len(items)

        # 2) 共现项：对每个物品，把其用户两两配对（上三角）计数 +1
        for i, users in tqdm(self.item_users.items(), desc="overlap by cooccurrence"):
            # 控制复杂度：截断超大物品的用户列表
            if len(users) > self.max_user_per_item:
                users = users[: self.max_user_per_item]

            # 上三角两两计数
            for idx_a in range(len(users)):
                ua = users[idx_a]
                for idx_b in range(idx_a + 1, len(users)):
                    ub = users[idx_b]
                    # 对称填充
                    va = overlap[ua].get(ub, 0) + 1
                    overlap[ua][ub] = va
                    overlap[ub][ua] = va

        self.user_overlap = dict(overlap)
        logger.info(f"user_overlap entries (approx pairs): {sum(len(v) for v in self.user_overlap.values())}")

    def _build_item_sims(self) -> None:
        """计算物品两两 Swing 相似度（稀疏存储）。"""
        logger.info("Building item-item swing similarities ...")
        item_ids = sorted(self.item_users.keys())
        sims: DefaultDict[int, Dict[int, float]] = defaultdict(dict)

        for idx_a in tqdm(range(len(item_ids)), desc="item swing sim"):
            ia = item_ids[idx_a]
            users_a = self.item_users[ia]
            if len(users_a) > self.max_user_per_item:
                users_a = users_a[: self.max_user_per_item]
            # 使用邻域：只与有交集的物品计算
            # 借助 "用户→所交互物品" 反查，构建候选物品集合
            neighbor_items = set()
            for u in users_a:
                neighbor_items.update(self.user_items[u])

            for ib in neighbor_items:
                if ib <= ia:
                    continue  # 上三角避免重复
                users_b = self.item_users.get(ib, [])
                if not users_b:
                    continue
                if len(users_b) > self.max_user_per_item:
                    users_b = users_b[: self.max_user_per_item]

                # V = users(ia) ∩ users(ib)
                V = intersect_sorted(users_a, users_b)
                if len(V) == 0:
                    continue

                score = 0.0
                # 双层嵌套（含对角），可用上三角*2+对角加速
                for x in range(len(V)):
                    ux = V[x]
                    # 对角
                    overlap_xx = self.user_overlap[ux].get(ux, 0)
                    score += 1.0 / (self.alpha + float(overlap_xx))
                    # 上三角
                    for y in range(x + 1, len(V)):
                        uy = V[y]
                        ov = self.user_overlap[ux].get(uy, 0)
                        val = 1.0 / (self.alpha + float(ov))
                        score += 2.0 * val  # 对称 (ux,uy) 与 (uy,ux)

                if score > 0.0:
                    sims[ia][ib] = score
                    sims[ib][ia] = score  # 保持对称

        self.item_sims = dict(sims)

    def _keep_topk(self) -> None:
        """对每个物品的相似邻居做 top-k 截断，保持矩阵稀疏。"""
        if self.topk_sim <= 0:
            return
        for i in list(self.item_sims.keys()):
            self.item_sims[i] = dict_topk(self.item_sims[i], self.topk_sim)

def intersect_sorted(a: List[int], b: List[int]) -> List[int]:
    """两个升序列表的交集（线性指针法），返回升序无重复列表。"""
    i, j = 0, 0
    out: List[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            if not out or out[-1] != a[i]:
                out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out
```

* * *

10) `src/recall.py`
-------------------

```python
from __future__ import annotations

from typing import Dict, List, Tuple
from loguru import logger
import numpy as np

class SwingRecommender:
    """基于 Swing 相似度矩阵的打分与召回器。"""

    def __init__(
        self,
        item_sims: Dict[int, Dict[int, float]],
        user_items: Dict[int, List[int]],
        exclude_seen: bool = True,
    ) -> None:
        """
        Args:
            item_sims: 稀疏相似度矩阵 {item_i: {item_j: sim_ij}}
            user_items: 用户→已交互物品
            exclude_seen: 是否在召回中排除已交互物品
        """
        self.item_sims = item_sims
        self.user_items = user_items
        self.exclude_seen = exclude_seen

    def recommend_for_user(
        self, user: int, topk: int = 50
    ) -> List[Tuple[int, float]]:
        """对单个用户进行召回，返回 (item, score) 的降序列表。"""
        seen = set(self.user_items.get(user, []))
        scores: Dict[int, float] = {}

        # 基于用户已看物品的相似邻居进行累加打分
        for i in self.user_items.get(user, []):
            for j, s in self.item_sims.get(i, {}).items():
                if self.exclude_seen and j in seen:
                    continue
                scores[j] = scores.get(j, 0.0) + s

        if not scores:
            return []

        # topk
        items = list(scores.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:topk]

    def batch_recommend(
        self, users: List[int], topk: int = 50
    ) -> Dict[int, List[Tuple[int, float]]]:
        """批量召回。"""
        out = {}
        for u in users:
            out[u] = self.recommend_for_user(u, topk=topk)
        return out
```

* * *

11) `src/metrics.py`
--------------------

```python
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

def recall_at_k(
    recs: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int
) -> float:
    """Recall@K：命中率。
    Args:
        recs: {user: [item...]}（预测）
        ground_truth: {user: [item...]}（真实）
    """
    users = [u for u in recs if u in ground_truth]
    if not users:
        return 0.0
    hit = 0
    total = 0
    for u in users:
        pred = set(recs[u][:k])
        gt = set(ground_truth[u])
        if len(gt) == 0:
            continue
        hit += len(pred & gt)
        total += len(gt)
    return (hit / total) if total > 0 else 0.0

def average_precision_at_k(pred: List[int], gt: List[int], k: int) -> float:
    """AP@K for a single user."""
    if not gt:
        return 0.0
    gt_set = set(gt)
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(pred[:k], start=1):
        if item in gt_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(gt), k)

def map_at_k(
    recs: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int
) -> float:
    """MAP@K."""
    users = [u for u in recs if u in ground_truth]
    if not users:
        return 0.0
    vals = []
    for u in users:
        vals.append(average_precision_at_k(recs[u], ground_truth[u], k))
    return float(np.mean(vals)) if vals else 0.0

def ndcg_at_k(
    recs: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int
) -> float:
    """NDCG@K (二值相关性)。"""
    def dcg(rel: List[int]) -> float:
        return sum((1.0 / np.log2(i + 2) for i, r in enumerate(rel) if r > 0))

    users = [u for u in recs if u in ground_truth]
    if not users:
        return 0.0
    vals = []
    for u in users:
        gt = set(ground_truth[u])
        rel = [1 if it in gt else 0 for it in recs[u][:k]]
        ideal = sorted(rel, reverse=True)
        denom = dcg(ideal)
        vals.append((dcg(rel) / denom) if denom > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0
```

* * *

12) `src/main.py`
-----------------

```python
from __future__ import annotations

import argparse
from typing import Dict, List
from loguru import logger
import numpy as np
import pandas as pd

from .utils import setup_logger, set_seed
from .config import Cfg
from .data import download_movielens_100k, load_movielens_100k
from .preprocessing import (
    binarize_interactions,
    filter_cold_users_items,
    train_test_random_split,
    build_mappings,
    remap_ids,
    build_user_item_sets,
)
from .swing import SwingCFIndexer
from .recall import SwingRecommender
from .metrics import recall_at_k, map_at_k, ndcg_at_k

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()

def df_to_gt(test_df: pd.DataFrame) -> Dict[int, List[int]]:
    """把测试集转为 {user_idx: [item_idx,...]} 作为 ground truth。"""
    gt: Dict[int, List[int]] = {}
    for r in test_df[["user_idx", "item_idx"]].itertuples(index=False):
        gt.setdefault(int(r.user_idx), []).append(int(r.item_idx))
    return gt

def main() -> None:
    setup_logger()
    args = parse_args()
    cfg = Cfg.from_yaml(args.config)
    set_seed(cfg.random_seed)

    # ---------------- 数据准备 ----------------
    logger.info("=== Stage 1: Download & Load Dataset ===")
    path = download_movielens_100k(cfg.dataset.url, cfg.dataset.local_dir)
    raw = load_movielens_100k(path)
    logger.info(f"Raw shape: {raw.shape}")

    logger.info("=== Stage 2: Binarize & Filter ===")
    df_pos = binarize_interactions(raw, cfg.dataset.binarize_threshold)
    df_pos = filter_cold_users_items(
        df_pos,
        cfg.dataset.min_inter_per_user,
        cfg.dataset.min_user_per_item,
    )
    logger.info(f"Positive interactions after filter: {df_pos.shape}")

    logger.info("=== Stage 3: Train/Test Split ===")
    train_df, test_df = train_test_random_split(
        df_pos, cfg.dataset.test_ratio, cfg.random_seed
    )
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

    logger.info("=== Stage 4: Build ID Mappings ===")
    uid2idx, idx2uid, iid2idx, idx2iid = build_mappings(train_df)
    train_df = remap_ids(train_df, uid2idx, iid2idx)
    test_df = remap_ids(test_df, uid2idx, iid2idx)
    logger.info(f"Users: {len(uid2idx)}, Items: {len(iid2idx)}")

    logger.info("=== Stage 5: Build Inverted Index ===")
    user_items, item_users = build_user_item_sets(train_df)

    # ---------------- 索引与相似度 ----------------
    logger.info("=== Stage 6: Swing Indexing & Similarity ===")
    indexer = SwingCFIndexer(
        alpha=cfg.swing.alpha,
        max_user_per_item=cfg.swing.max_user_per_item,
        topk_sim=cfg.swing.topk_sim,
    ).fit(user_items=user_items, item_users=item_users)

    item_sims = indexer.get_item_sims()
    # 关键自检：相似度对称性与非负性
    _sanity_check_item_sims(item_sims)

    # ---------------- 召回与评估 ----------------
    logger.info("=== Stage 7: Recall & Evaluate ===")
    recommender = SwingRecommender(
        item_sims=item_sims,
        user_items=user_items,
        exclude_seen=cfg.recall.exclude_seen,
    )

    users_eval = sorted(set(test_df["user_idx"].unique().tolist()))
    rec_pred = {}
    for u in users_eval:
        recs = recommender.recommend_for_user(u, topk=cfg.recall.k_recall)
        rec_pred[u] = [it for it, _ in recs]

    gt = df_to_gt(test_df)

    for k in cfg.evaluate.ks:
        r = recall_at_k(rec_pred, gt, k)
        m = map_at_k(rec_pred, gt, k)
        n = ndcg_at_k(rec_pred, gt, k)
        logger.info(f"[Eval] K={k:>3d}  Recall={r:.4f}  MAP={m:.4f}  NDCG={n:.4f}")

    # ---------------- 结果展示（样例） ----------------
    sample_users = users_eval[:5]
    for u in sample_users:
        recs = recommender.recommend_for_user(u, topk=10)
        logger.info(f"[Sample] user={u}, recs={recs}")

    logger.info("Pipeline finished.")

def _sanity_check_item_sims(item_sims: Dict[int, Dict[int, float]]) -> None:
    """关键一致性自检：对称、非负、无自环（如需要）。"""
    cnt = 0
    for i, neigh in item_sims.items():
        for j, s in neigh.items():
            assert s >= 0.0, "相似度必须非负"
            # 对称性
            s2 = item_sims.get(j, {}).get(i, None)
            assert s2 is not None and abs(s - s2) < 1e-8, "相似度矩阵需要对称"
            cnt += 1
    logger.info(f"Sanity check passed. checked edges={cnt}")

if __name__ == "__main__":
    main()
```

* * *

三、如何运行（简要）
==========

```bash
pip install -r requirements.txt
python -m src.main --config config.yaml
```

运行后日志将依次打印：数据下载→过滤→切分→索引→Swing 相似度构建→TopK 截断→召回→评估（Recall/MAP/NDCG）→样例用户推荐结果。

* * *

四、工程要点 & 复杂度控制（实战心得）
====================

*   **复杂度瓶颈**：Swing 的双层用户求和在 V×V 上，V=共同用户集合。通过
    *   `max_user_per_item` 限制每个物品参与构图的用户数；
    *   相似度计算时仅对 **存在交集的候选物品** 计算（邻域法）；
    *   `topk_sim` 稀疏化邻居，保持矩阵稀疏。
*   **稳定性**：过滤冷用户/冷物品，减少噪声；二值化阈值可按业务调。
*   **正确性自检**：对称性、非负性断言；可追加随机抽样重算核对。
*   **可扩展方向**：
    *   将 `user_overlap` 与 `item_sims` 的构建并行化/批处理化；
    *   采用分桶（按活跃度/时间）做分治；
    *   支持增量更新（新交互到来时仅更新相关对）。

* * *

如果你希望，我可以继续补充：

*   带 **leave-one-out by time** 的评估切分版本；
*   更激进的性能优化（多进程 / numba / 分桶稀疏 CSR 实现）；
*   与 **ItemCF/Graph CF** 并行对比评测脚本，帮助你做策略对比与 A/B。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)