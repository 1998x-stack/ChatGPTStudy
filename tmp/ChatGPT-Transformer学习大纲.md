# Transformer学习大纲

**User:** Anonymous (bonedarwin561@outlook.com)  
**Created:** 8/19/2025 17:40  
**Updated:** 8/20/2025 10:11  
**Exported:** 8/20/2025 10:13  
**Link:** [https://chatgpt.com/c/68a4469a-28cc-8329-8076-5f5698870277?model=gpt-5-thinking](https://chatgpt.com/c/68a4469a-28cc-8329-8076-5f5698870277?model=gpt-5-thinking)  

## Prompt:
01-Transformer.pdf

PDF

Task: 深入学习研究Transformer论文里面的技术细节和底层机制

Methid: 耐心引导式&选择题提问式&知识卡片/QA闪卡总结式&interesting-examples/jokes-driven&一步一步拆解分析

Bonus&Candy🍬:colorful emoji & 幽默有趣 & 有思想深度 & 图解/闪卡 & funny games

Notice: 我们可以quote原文中的核心句子&每隔几段进行总结复习

首先给出学习大纲

## Response:
我们将用你喜欢的引导式 + 闪卡式 + 趣味元素（🍬🎯🧠）来系统学习 Transformer 的核心技术细节和底层机制，基于论文《Attention is All You Need》。整个过程会分为 **逐节精讲 + 提问互动 + 总结回顾** 的方式进行。

* * *

🎯Transformer 论文精学大纲（根据原文结构+学习逻辑整理）
-----------------------------------

### 🌟 Part 0. 导学导图 & 提问目标

*   Transformer 到底解决了什么问题？为什么“Attention is All You Need”？
*   核心模块到底有哪几个？每个模块干啥的？

* * *

### 🧠 Part 1. 总体结构与设计目标（Sections 1~2）

> 【关键词】Seq2Seq、RNN、CNN、Self-Attention、Parallelism、路径长度、建模长距离依赖

*   🎯1.1 传统模型瓶颈：RNN 与 CNN 有哪些限制？
*   🎯1.2 Transformer 用了哪些新思路？“全 Attention 架构”的意义是什么？
*   🎯1.3 Self-Attention 是怎么取代 RNN 的？有什么数学优势？

* * *

### 🧩 Part 2. 模型结构全览（Section 3）

> 【关键词】Encoder-Decoder、Residual、LayerNorm、Feed-Forward

*   🎯2.1 Transformer 模型图精读：每个箭头背后代表什么操作？
*   🎯2.2 Encoder 和 Decoder 各有几个子模块？哪一层用了 mask？
*   🎯2.3 Feed-Forward 有哪些特殊设计？为啥是 2048 维？
*   🎯2.4 为什么每层都要残差+LayerNorm？

* * *

### 🧲 Part 3. 注意力机制详解（Section 3.2）

> 【关键词】Scaled Dot-Product Attention、Multi-Head Attention、Query/Key/Value、Masking

*   🎯3.1 Scaled Dot-Product Attention 数学公式详解（推导 + 直觉）
*   🎯3.2 为什么要除以 √dk？不除会怎样？
*   🎯3.3 Multi-Head Attention 有什么结构优势？为什么要并行多个 Head？
*   🎯3.4 三种 Attention 应用场景（Encoder Self、Decoder Self、Encoder-Decoder）

* * *

### 🔢 Part 4. Embedding 与位置编码（Section 3.4 ~ 3.5）

> 【关键词】Positional Encoding、Sin/Cos 编码、权重共享

*   🎯4.1 为什么没有 RNN 时必须加位置编码？
*   🎯4.2 Positional Encoding 是怎么用正余弦实现的？可以学习吗？
*   🎯4.3 为什么输入/输出 embedding 和 softmax 前线性层可以共用参数？

* * *

### ⚙️ Part 5. 模型优势解析：Why Self-Attention（Section 4）

> 【关键词】复杂度分析、路径长度、可并行性、解释性

*   🎯5.1 Self-Attention vs RNN/CNN：复杂度和路径长度对比
*   🎯5.2 为什么说 Self-Attention 更“可解释”？
*   🎯5.3 对长距离依赖建模，谁更强？

* * *

### 🧪 Part 6. 训练机制 & 超参技巧（Section 5）

> 【关键词】优化器、Warmup、Label Smoothing、Dropout、学习率调度

*   🎯6.1 学习率调度曲线长什么样？为啥这样设计？
*   🎯6.2 为什么要 Label Smoothing？它到底做了什么？
*   🎯6.3 Dropout 在哪里用了？参数设多少？

* * *

### 🏆 Part 7. 实验结果分析 & Ablation Study（Sections 6~7）

> 【关键词】BLEU、头数变化、位置编码对比、大模型效果

*   🎯7.1 哪些模块对性能影响最大？Ablation Table 怎么读？
*   🎯7.2 Transformer (big) 有什么改变？
*   🎯7.3 除了翻译还能干啥？比如 Constituency Parsing

* * *

### 🎓 Part 8. 总结与拓展思考

> 【关键词】下一步、效率瓶颈、未来方向

*   🎯8.1 Transformer 的瓶颈有哪些？（比如长序列）
*   🎯8.2 后来的改进版本（比如 BERT、GPT、Performer）做了什么改变？
*   🎯8.3 你能设计一个比 Transformer 更强的模型吗？🤯

* * *

是否现在从 **Part 1 开始学习**？我会先引导你思考 Transformer 为什么“只用 Attention 就能通吃”，然后引入 Self-Attention 的关键优势。也可以从你想学的某一部分直接开始！你选~

## Prompt:
ok

## Response:
太棒啦！那我们就从 **Part 1：总体结构与设计目标** 正式开始 🎯

* * *

🧠 Part 1：总体结构与设计目标
-------------------

> 💡 引导我们走向“Attention is All You Need”的问题链！

* * *

### 🎯 Step 1：Transformer 想解决什么问题？

我们先来回忆一下，在 Transformer 出现之前，最常用的序列建模方法是：

*   ✅ RNN / LSTM：依赖时间步 t 的串行状态传播，训练慢，不易并行。
*   ✅ CNN：可以并行但接收域有限，长距离建模困难。

那我们来做一个 **判断题** 🧩：

> Transformer 的设计初衷是：
> 
> A. 更少的参数、更小的模型  
> B. 更强的表达能力，更强的记忆性  
> C. 减少序列依赖，加快训练并提升长距离建模能力  
> D. 模拟语言的语法结构

你选哪个？（可以多选）

## Prompt:
c

## Response:
💯 完美！就是 **C. 减少序列依赖，加快训练并提升长距离建模能力**！

* * *

### 🧠 快速知识卡片：为什么要放弃 RNN 和 CNN？

| 特性 🧩 | RNN / LSTM | CNN | Transformer (Self-Attention) ⚡ |
| --- | --- | --- | --- |
| 串行依赖性 | 强（不能并行） | 无串行依赖 | 完全并行 🧨 |
| 长距离依赖建模 | 路径长，梯度易消失 | 路径短但受限 | 所有位置间一跳到达 🛸 |
| 训练时间 | 长 | 中等 | 快 ⚡ |
| 参数共享 | 按时间共享 | 卷积核共享 | 多头注意力独立 |

* * *

### 🎯 Step 2：那 Self-Attention 到底好在哪里？

Transformer 最大的创新就是👇

> 🔥 “We propose the Transformer, a model architecture **eschewing recurrence** and instead relying entirely on an **attention mechanism** to draw global dependencies between input and output.”

我们来个 **脑筋急转弯式选择题** 🧠：

> Self-Attention 和 RNN 最大的“计算路径”差别在哪？
> 
> A. RNN 要一步一步传，路径长，信息传播慢  
> B. Self-Attention 是每个词都能直接看到所有词，路径是常数  
> C. RNN 更容易过拟合  
> D. Self-Attention 训练更难

你觉得是？（可以选多个）

## Prompt:
b

## Response:
✅ 正确！就是 **B：Self-Attention 是每个词都能直接看到所有词，路径是常数**！

* * *

### 🧠 知识闪卡：路径长度对建模有什么影响？

> 原文中给出了这样一张关键对比表👇：

| Layer 类型 | 每层复杂度 | 最少串行操作数 | 最大路径长度 |
| --- | --- | --- | --- |
| Self-Attention | 𝑂(n²·d) | **1** | **1** |
| RNN | 𝑂(n·d²) | **n** | **n** |
| CNN | 𝑂(k·n·d²) | 1 | logₖ(n) |

其中 `n` 是序列长度，`d` 是维度，`k` 是卷积核宽度。

✨ 所以 Self-Attention 真的做到了 **全局建模 + 并行训练 + 短路径传播**，比起 RNN 和 CNN 都更合适语言建模。

* * *

### ✅ 快速回顾 Check-in

我来考考你（别担心，我们一起答 😄）：

> 🚀 为什么 Transformer 能比 RNN 训练快、效果还更好？请选出所有合适的解释：
> 
> A. 用了残差连接  
> B. 自注意力不依赖时间序列，因此可并行  
> C. 所有词之间直接连接，路径最短  
> D. 模型用了正余弦位置编码

你来选~

## Prompt:
b

## Response:
🥇没错！**B 是核心关键点！**

Self-Attention 不像 RNN 一样必须一步步传 —— 它允许**所有位置之间直接交互**，因此能充分利用 GPU 并行计算，加速训练 🚀

* * *

不过我们也来看下其他选项值不值得选：

| 选项 | 是否也对？ | 理由 |
| --- | --- | --- |
| A. 残差连接 | ✅ | 虽然不是“并行训练”的原因，但它确实让深层网络更易训练（防梯度消失） |
| B. 自注意力可并行 | ✅✅✅ | 正确！训练速度革命性提升关键 |
| C. 路径短 | ✅ | 路径长度为常数 ⇒ 长依赖更易学 |
| D. 正余弦编码 | 🟡 | 编码顺序信息很重要，但和“训练快”没直接关系 |

所以 B 是主要原因，C 和 A 是支持原因，D 更偏向功能补充 💡

* * *

### 🎁小总结 🎓（速记口诀）

> **“并行快，全局看，路径短”**
> 
> Transformer 比 RNN 强的三大优点：

*   ⚡ 并行快（Self-Attention 不串行）
*   👀 全局看（任意两词直接交互）
*   🔁 路径短（长依赖好学）

* * *

下一步，我们就进入 **Part 2：模型结构全览** —— 拆解整张 Transformer 架构图，看看那些 “Box” 背后到底藏了什么玄学🧙！

👉 要继续进入 Part 2 吗？还是你想先复习 / 提问一下这部分？

## Prompt:
go

## Response:
好的，出发！🎯

* * *

🧩 Part 2：模型结构全览
----------------

> 拆解 Transformer 图，每个模块到底干了啥？

* * *

### 🎯Step 1：总览结构图

先来看论文中的核心结构图（你脑海里可能已经有印象了）：

📌 图中左边是 **Encoder Stack**，右边是 **Decoder Stack**。  
每一边都有 **若干层叠加结构**，都用到了：

*   Multi-Head Attention 🎯
*   Feed-Forward Network ⚙️
*   Residual + LayerNorm 🌀

> 原文引用句：
> 
> > _"The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder."_

* * *

### 🧠 快速脑图拆分

| 模块 | 子结构 | 说明 |
| --- | --- | --- |
| **Encoder 每层** | 🧠 Multi-Head Self-Attention + FFN + 残差+Norm | 编码器每层结构完全一致 |
| **Decoder 每层** | 🧠 Masked Self-Attention + Encoder-Decoder Attention + FFN + 残差+Norm | Decoder 会看 Encoder 输出，还要 Mask 自己的未来 |
| **嵌入部分** | Input Embedding + Positional Encoding | 加上位置信息（否则没有顺序感） |
| **输出部分** | Linear + Softmax | 输出预测词的概率 |

* * *

### 🎯 Step 2：提问练习 🧪

下面哪个说法是对的？来选一选 🧐

> Transformer 的 Decoder 比 Encoder 多了一个什么组件？
> 
> A. 多了 ReLU  
> B. 多了一个 Self-Attention  
> C. 多了一个 Encoder-Decoder Attention  
> D. 多了一个 Softmax

你选？

## Prompt:
c

## Response:
🎯正确！就是 **C. 多了一个 Encoder-Decoder Attention**！

* * *

### 🧠 解释一下为什么：

*   Encoder 中每一层只有两个子层：
    1.  Multi-Head Self-Attention（可以全局看自己）
    2.  Feed-Forward（两个全连接 + ReLU）
*   Decoder 中每层有三个子层：
    1.  Masked Multi-Head Self-Attention（**不能偷看后面的词**）
    2.  **Encoder-Decoder Attention**（查询来自 Decoder，键值来自 Encoder）
    3.  Feed-Forward

💡 这个 **Encoder-Decoder Attention** 就是帮 Decoder “看” 输入序列。

* * *

### 🎁趣味比喻时间 🍬

你可以把 Decoder 想象成一个 **小学生作文选手✍️**：

*   自己（Decoder）要一句句地写（自回归）
*   不能提前偷看后面（Mask）
*   老师（Encoder）给了你一篇英文文章（输入序列）
*   你要**一边写一边回看老师讲了啥**（Encoder-Decoder Attention）

* * *

### ✅ 快速复盘 Check-in：

我来小考一下 🤓

> Encoder 和 Decoder 都用了 Residual Connection + LayerNorm 吗？  
> 为什么这么设计？（可以选多个）

A. 因为可以提升表达能力  
B. 因为训练更稳定  
C. 因为能防梯度消失  
D. 因为它是 Attention 的替代品

你来选~

## Prompt:
c

## Response:
✅ \*\*C 是正确的！\*\*残差连接确实能帮助防止梯度消失 —— 尤其是层数变深的时候，这是深度学习的经典 trick 💡

不过我们也来看看其他选项是否也合理：

| 选项 | 是否也对？ | 解释 |
| --- | --- | --- |
| A. 提升表达能力 | 🟡 部分对 | LayerNorm + 残差能帮助保留多层表示，但它们本身不是核心决定表达力的因子 |
| B. 训练更稳定 | ✅ 是的 | LayerNorm + Residual 是深度网络收敛的“镇宅之宝”之一🏡 |
| C. 防梯度消失 | ✅✅✅ | 残差最初就是为此设计的（ResNet 起家） |
| D. Attention 替代品 | ❌ 完全错误 | 它是辅助模块，不是替代品；Attention 才是核心表达方式 |

* * *

🎓 **小总结：**

> 每一层都加 Residual + LayerNorm，有助于：
> 
> *   梯度传播更稳定
> *   多层结构更好训练
> *   保留低层信息不被淹没
>     

* * *

🧩 好了，现在你已经理解了整张 Transformer 架构图背后的结构，我们就可以 zoom in 聚焦每一个关键模块啦！

下一步，我们进入 Transformer 的灵魂：

🧠 Part 3：Attention 机制拆解！
-------------------------

从 Scaled Dot-Product 到 Multi-Head Attention，一步步推公式 + 图解直观理解 👀

要继续吗？🧪

## Prompt:
OK

## Response:
太好啦！🎯  
现在我们正式进入 Transformer 的“灵魂核心发动机”：

* * *

🧠 Part 3：Attention 机制拆解
------------------------

> Scaled Dot-Product ➕ Multi-Head ➕ 应用场景 🧪 图解 + 公式 + 推理直觉

* * *

### 🎯 Step 1：什么是 Attention？

一句话直白解释👇：

> Attention = 用 Query 去找最相关的 Key → 用权重加权对应的 Value  
> 🤝 你提问（Query），我搜数据库（Key）→ 给你答案（Value）

* * *

### 🧠 闪卡记忆：Scaled Dot-Product Attention 公式

原文给出👇公式（也是面试常考！）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

| 符号 | 含义 |
| --- | --- |
| Q | Queries（你要找啥） |
| K | Keys（有哪些可以匹配的） |
| V | Values（返回啥信息） |
| √dk | 缩放因子，防止 softmax 梯度太小 |

📌 这是 “点积注意力”+缩放，因此叫 Scaled Dot-Product Attention。

* * *

### 🎯 Step 2：快问快答🧪

> 为什么要除以  $\sqrt{d_k}$  ？  
> A. 防止点积过大，导致 softmax 梯度消失  
> B. 为了更快收敛  
> C. 是模型作者随意加的  
> D. 只有在训练中才除

你觉得是？

## Prompt:
A

## Response:
🎯 完美命中！答案正是 **A：防止点积过大，导致 softmax 梯度消失**！

* * *

### 🧠 小推理时间（逻辑脑锻炼💪）

如果我们不除以 √dk，会发生什么？

*   假设 Q 和 K 的维度是 512，那 Q·K 就是 512 个数的乘积和，加起来可能非常大。
*   这会让 softmax 进入**极端区域** —— 非常陡峭，梯度趋近于 0 😵
*   训练将变得困难，无法学习细微的注意力变化。

所以这一步其实是 “梯度保护装置”🛡️

* * *

### 🧩小知识延伸

> **Additive Attention**（Bahdanau Attention）就没有除 √dk，而是用个小网络去拟合 Compatibility

论文里其实也比较过两者：

*   Dot Product + Scaling：快，空间省，更适合矩阵乘优化 ✅
*   Additive：更灵活，但慢 ❌

* * *

### ✅ 快速复盘 Check-in

请简要说明：

> Attention 是怎么从 Q、K、V 三者中计算出输出的？  
> （我们可以一起回答~）

你先说说思路？我可以补充 💬

## Prompt:
ok

## Response:
好，来我们一起梳理一下 Attention 的核心逻辑，帮你牢牢刻在脑中🧠：

* * *

### 🧠 Attention 计算流程（可以边画边记）

> 给定 Q（查询）、K（键）、V（值），一步步计算如下：

1.  **计算相似度分数**
    *   点积 Q·Kᵀ，得到每个 Query 和所有 Key 的相似度
    *   大小越大，表示越相关
2.  **缩放保护**
    *   除以 √dk，避免点积过大导致 softmax 梯度消失
3.  **归一化成概率**
    *   对每个 Query 的相似度分数进行 softmax，变成权重（注意力分布）
4.  **加权求和**
    *   用这些权重加权对应的 Value 向量，得到输出结果

* * *

### 🧠 图解复现一下：

$$
\textcolor{#FF7F50}{\text{Attention}(Q, K, V)} = \underbrace{\text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right)}_{\textcolor{#1E90FF}{\text{权重}}} \cdot V
$$

🧠 它像在说：“我问的问题 Q，和所有数据库 K 比比谁最像，然后从 V 里抽取答案，按权重平均。”

* * *

✅ 你能完整复述一下 Attention 的 4 步计算过程吗？  
👉 如果可以了，我们就进入 **Multi-Head Attention 的魔法增强版** 🪄！  
要不要你来挑战一下？😄

## Prompt:
ok

## Response:
太棒了！👏 那请你来简单说说：  
👉 从 Q、K、V 怎么一步步计算出 Attention 的输出？

你可以一句话一个步骤，我会帮你补全或修正。大胆来试试吧~

## Prompt:
1. 根据 query & key 点积计算 相似度分数
2. 除以 sqrt d_k ，进行缩放保护
3. 归一化softmax成为权重
4. 权重和value进行点积，加权分数，得到输出结果

## Response:
👏👏👏 完全正确，而且表达非常清晰有条理！

我们稍微来润色一下，把你刚说的变成一张**复习闪卡**，随时能用来回忆：

* * *

### 🧠 Attention 计算 4 步法（闪卡）

1️⃣ **点积比相似**  
→ 用每个 Query 和所有 Key 做点积，得相似度分数

2️⃣ **缩放保梯度**  
→ 除以  $\sqrt{d_k}$ ，避免 softmax 梯度消失

3️⃣ **softmax 变权重**  
→ 对每个 Query 的分数做 softmax，得注意力权重分布

4️⃣ **加权出结果**  
→ 权重 × Value 向量，加权求和，输出结果 🎯

* * *

🎓 现在你已经掌握了单个 Attention 的核心！

接下来我们就要讲讲最有魔力的一招 —— **Multi-Head Attention** 🪄

👉 它不是重复上面这个流程一次，而是开多个并行注意力通道，每个看不同的子空间！

要进入 Multi-Head Attention 吗？😎

## Prompt:
OK

## Response:
好耶！现在我们进入 Transformer 最魔性的一招：

* * *

🧠 Part 3.2：Multi-Head Attention 魔法扩展 🪄
----------------------------------------

> “不是你一个人在战斗，是八个自己一起注意不同信息！”

* * *

### 🎯为什么要多个头（multi-head）？

先问你一个引导问题👇：

> 如果我们只用一个注意力头（即只执行一次 Attention），你觉得会有什么问题？🤔  
> （大胆猜一猜）

* * *

提示：想象一个人同时要判断句子结构、主谓关系、动词短语……只能用一个「注意点」，会发生啥？

## Prompt:
只能聚焦一方面信息

## Response:
🎯 正确！就是这个意思！

> **单个注意力头太“专一”，只能看一处；多个头才能分工合作👀👀👀👀👀👀👀👀**

* * *

### 🧠 Multi-Head Attention 核心思想

不是重复执行同一个 Attention，而是用 **不同线性投影** 把 Q、K、V 分别映射成多个子空间，在每个空间里做不同的注意力计算。

然后再把所有头的结果**拼接 + 映射**回来：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

其中每个 head 是：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

* * *

### 🎯为什么这样更强？

我们来做一个 **类比选择题**：

> Multi-Head Attention 就像是：  
> A. 让一组专家各自去看句子的不同结构  
> B. 把一句话复制多次再求平均  
> C. 模拟词向量的加法  
> D. 给模型增加深度

你选？🤔

## Prompt:
A

## Response:
💯 完美命中！就是 **A：让一组专家各自去看句子的不同结构**！

* * *

### 🧠 再来个有趣比喻 🍬

> Multi-Head Attention 就像是：
> 
> 一群“专注力不同的侦探🕵️‍♂️🕵️‍♀️🕵️‍♂️🕵️‍♀️”在看同一封信：
> 
> *   头1 看主语-动词结构
> *   头2 看时间短语
> *   头3 看指代关系
> *   …
> *   然后大家开会合并情报，总结成最终输出
>     

* * *

### 📐 模型参数也能解释这个设计：

假设总维度是  $d_{model} = 512$ ，我们用  $h = 8$  个头：

*   每个头的 Query、Key、Value 都被投影到 64 维（因为  $512 / 8 = 64$ ）
*   8个头分别计算 Attention → 结果拼接 → 还原成 512 维

✨ 这样总计算量不变，但**每个头能学到不一样的东西**，相当于做了**信息多视角建模**。

* * *

✅ 快速复盘 Check-in：

> Multi-Head Attention 为什么更强？请用自己的话讲讲~  
> 你来一句我一句？我们一起总结下 😊

## Prompt:
让一组专家各自去看句子的不同结构

## Response:
完美说法！👏👏👏

我来帮你接下一句补充一下：

> ✅ 每个注意力头都从不同角度理解输入信息，然后 **合并结果，形成更丰富的表示**。

* * *

### 🧠 快速闪卡总结：为什么要 Multi-Head Attention？

| 优势 🧠 | 原因说明 ✍️ |
| --- | --- |
| 👀 多角度表达 | 不同头看不同关系（语法、语义、时序） |
| 🧠 增强建模能力 | 每个子空间可以学习不同类型的注意力分布 |
| ⚙️ 计算成本不变 | 因为每个头降维后计算，再拼接回来 |
| 🔁 更丰富的特征融合 | 拼接再投影让模型集成多个“专家”输出 |

* * *

🥳 到这里，Single Attention ➕ Multi-Head Attention 都掌握了！

下一节我们进入：

🎯 Part 3.3：三种 Attention 的具体使用位置（Encoder vs Decoder）
----------------------------------------------------

→ 分清楚哪些地方是 **Self-Attention**，哪些是 **Encoder-Decoder Attention**

要继续进入吗？😎

## Prompt:
ok

## Response:
太棒了！我们继续冲 🚀！

* * *

🎯 Part 3.3：Attention 用在哪里？三种注意力场景大拆解！
--------------------------------------

Transformer 总共用了三种不同的 Attention，用法不同、目的不同，但**都用的是同一个机制（Scaled Dot-Product）**。

* * *

### 🧠 三种 Attention 总览（一定要牢牢记住！）

| 模块位置 | Attention 类型 | Query 来自 | Key & Value 来自 | 用途 |
| --- | --- | --- | --- | --- |
| Encoder | **Self-Attention** | Encoder 自己 | Encoder 自己 | 编码器内部信息交互 🧠 |
| Decoder | **Masked Self-Attention** | Decoder 自己 | Decoder 自己（masked） | 防止看未来，只看自己和过去 ⏳ |
| Decoder | **Encoder-Decoder Attention** | Decoder 自己 | Encoder 输出 | Decoder 查 Encoder 输出内容 🔍 |

* * *

### 🎯 提问小测试：

> 以下哪些地方用了 Masked Self-Attention？  
> A. Encoder Stack  
> B. Decoder Stack  
> C. Encoder-Decoder Attention  
> D. 输出 Softmax 层

你选？😏

## Prompt:
a

## Response:
🛑 小小地踩了一下坑哈~ 其实正确答案是：

> ✅ **B. Decoder Stack**

* * *

### 🧠 解释一下：

*   **Encoder Stack** 是 **普通的 Self-Attention**，可以自由地看整个输入序列，因为输入已经是已知的 📖
*   **Decoder Stack** 中第一层是 **Masked Self-Attention**，它必须**防止看到未来词**，因为 Decoder 是逐步生成的 ✍️
*   **Encoder-Decoder Attention** 是 Decoder 的第二层，用来看 Encoder 的输出，是“查参考资料” 🔍

* * *

### 🎓 快速口诀记忆法（背下来绝对有用！）：

> **“三 Attention 分工忙，一自一掩一参考”**
> 
> *   Encoder：**自看**
> *   Decoder：**掩看（不能偷看后面）**
> *   Decoder→Encoder：**参考 Encoder 输出**
>     

* * *

我们可以用一张图来巩固理解（用脑内画画也行）：

```
Encoder Stack:
  Q ← Self
  K ← Self
  V ← Self

Decoder Stack:
  第1层:
    Q ← Self
    K ← Self (masked)
    V ← Self
  第2层:
    Q ← Self
    K ← Encoder 输出
    V ← Encoder 输出
```

* * *

要不要试着自己总结一下：

> Transformer 总共用了哪三种 Attention？每种干嘛的？

说完我来帮你打分！😎

## Prompt:
encoder : self-attention query-value-key 都是 encoder，将编码器内部信息进行交互
docoder：masked self-attention, query-value-key 都是 decoder, 防止看到未来，只看到过去

## Response:
👏👏太棒了！你已经把前两种说得非常清晰了！

只差最后一个小伙伴啦 👉 **Encoder-Decoder Attention**。我帮你补全一下：

* * *

### 🧠 三种 Attention 完整总结

1.  **Encoder Self-Attention**
    *   Q, K, V 都来自 **Encoder 自己**
    *   作用：让输入序列里的所有词互相关注，捕捉上下文依赖
2.  **Decoder Masked Self-Attention**
    *   Q, K, V 都来自 **Decoder 自己**
    *   作用：生成时只能看前面的词，**保证自回归**（不能偷看未来）
3.  **Encoder-Decoder Attention**
    *   Q 来自 **Decoder**
    *   K, V 来自 **Encoder 输出**
    *   作用：帮助 Decoder 在生成每个词时查阅 Encoder 提供的“翻译上下文/原文信息”

* * *

### 🎓 口诀复习

> **“自看、掩看、参考看”**

*   编码器：自看自己
*   解码器：掩着看自己
*   解码器：参考看编码器

* * *

✅ Check-in 小测：  
如果 Decoder 没有 **Encoder-Decoder Attention**，会发生什么问题呢？🤔

## Prompt:
就不能获得从encoder传来的编码信息

## Response:
💯 完全正确！👏

如果没有 **Encoder-Decoder Attention**，Decoder 就会“闭门造句”，只能依赖自己生成过的词，而完全没法利用输入序列（比如英文句子）的语义信息。

那样的话，模型就**没法翻译**，只能变成一个“语言模型”，像 GPT 一样只会续写。📜

* * *

### 🎁 小总结闪卡

> Decoder = 语言模型 + 查资料功能
> 
> *   **Masked Self-Attention** → 像语言模型一样，逐字生成
> *   **Encoder-Decoder Attention** → 相当于查输入句子的“资料库”，保证翻译对齐
>     

* * *

🎯 到这里，我们已经把 \*\*Attention 模块（单头 + 多头 + 三种应用场景）\*\*全部拆清楚了。  
接下来可以进入 **Part 4：Embedding & Positional Encoding**，聊聊 Transformer 怎么解决“没有顺序感”的问题 👀

要不要继续？

## Prompt:
OK

## Response:
太好啦 🚀 我们继续深入！

* * *

🧠 Part 4：Embedding & Positional Encoding
-----------------------------------------

> Transformer 没有 RNN/CNN，天然就没有“顺序感”🕰️  
> 那它怎么知道单词的顺序呢？答案就是 **位置编码（Positional Encoding）**

* * *

### 🎯 Step 1：Embedding 基础

*   输入的每个 token（比如单词、子词）先映射成一个 **dmodel 维向量**
*   论文里用的是  $d_{model} = 512$ 
*   输出端同样有 embedding → 然后接 softmax 预测下一个 token

📌 一个 trick：  
输入 embedding 和 输出 softmax 的权重矩阵是**共享的**（减少参数，还能稳定训练）✍️

* * *

### 🎯 Step 2：Positional Encoding 数学公式

论文采用 **正余弦函数** 编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
 
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

📌 意思是：

*   每个维度对应一个不同频率的正弦波
*   组合在一起，就能唯一表示每个位置
*   且能让模型学会相对位置信息（比如 pos+k 可以通过线性关系表示）

* * *

### 🎁 趣味比喻

你可以把 **Positional Encoding** 想象成：

> 🎶 每个单词都有一段“背景音乐”，位置不同旋律不同
> 
> 模型一听，就知道这是句子的第 3 个词，还是第 10 个词

* * *

### 🎯 Step 3：小测问题

> Transformer 为什么不直接用“位置索引（1,2,3,4…）”作为输入，而要用这种正余弦函数呢？🤔

A. 因为整数索引不能被梯度优化  
B. 因为正余弦可以表示相对位置关系  
C. 因为正余弦更节省显存  
D. 因为 Vaswani 作者喜欢三角函数

你选？

## Prompt:
b

## Response:
✅ 完美！答案正是 **B：正余弦可以表示相对位置关系** 🎯

* * *

### 🧠 解释一下为什么：

*   如果直接用位置索引（1,2,3,4…），模型只能学到“绝对位置”，却很难直接捕捉 **相对位置**（比如“距离 3 个词”这种信息）。
*   正余弦的波形特性 → 不同维度对应不同频率，就像组合音符 🎶
*   因此 **任意两个位置之间的差值（相对位置）** 都可以通过线性关系表示。

这就是为什么论文中说：

> _“We hypothesized it would allow the model to easily learn to attend by relative positions.”_

* * *

### 🎓 小总结闪卡

> **Positional Encoding = 给词加上“节奏感”🎵**

*   每个维度是一个不同频率的波
*   相对位置 → 线性可表达
*   模型因此能分辨顺序信息，而不需要 RNN/CNN

* * *

✅ 到这里，我们就把 **输入模块**（Embedding + Positional Encoding）搞定了！

接下来，我们可以进入 **Part 5：Why Self-Attention 比 RNN/CNN 更优**（论文里的核心对比表格和理论优势）⚡

要不要继续冲？

## Prompt:
ok

## Response:
太好了 🚀！我们进入论文里最精彩的理论部分：

* * *

🧠 Part 5：Why Self-Attention 更优？
--------------------------------

> 原文第 4 节直接拿 Self-Attention 和 RNN、CNN 作了对比。  
> 关键词：**计算复杂度、并行度、路径长度**。

* * *

### 🎯 Step 1：三大比较指标

1.  **计算复杂度 per layer**
    *   RNN:  $O(n \cdot d^2)$ 
    *   CNN:  $O(k \cdot n \cdot d^2)$ 
    *   Self-Attention:  $O(n^2 \cdot d)$ 
2.  **最少串行操作数**
    *   RNN:  $O(n)$ （必须按时间步来）
    *   CNN:  $O(1)$ 
    *   Self-Attention:  $O(1)$ （全并行⚡）
3.  **最大路径长度**
    *   RNN:  $O(n)$ （长依赖传递慢）
    *   CNN:  $O(\log_k(n))$ 
    *   Self-Attention:  $O(1)$ （一步直达🚀）

* * *

### 🎁 有趣比喻

*   **RNN**：像是一个人要逐字读信，每个字都要记住，传到最后容易忘记前面的内容（梯度消失）。
*   **CNN**：像放大镜🔍在窗口里看，能快点滑动，但看全局还要好多层。
*   **Self-Attention**：像无人机航拍 🛸，一眼就能看到全局。

* * *

### 🎯 Step 2：小测问题

> 以下哪一点 **不是** Self-Attention 的优势？🤔
> 
> A. 训练时能并行  
> B. 长距离依赖路径短  
> C. 每层复杂度总是比 RNN 更低  
> D. 模型更易解释

你觉得是哪一个？

## Prompt:
C

## Response:
🎯 完全正确！就是 **C**。

* * *

### 🧠 解释一下：

*   Self-Attention 的 **并行性** (A) → ✅ 大优势
*   长距离依赖路径短 (B) → ✅ 只要 1 步直达
*   更易解释 (D) → ✅ 注意力权重能可视化，直观看到模型关注了哪里

但 (C) 就不对啦：

*   Self-Attention 的复杂度是  $O(n^2 \cdot d)$ ，当序列长度  $n$  很大时，可能比 RNN ( $O(n \cdot d^2)$ ) 还贵。
*   所以后面才有一堆改进（比如 Longformer, Performer, Linformer 等），专门优化长序列的注意力计算。

* * *

### 🎓 小总结闪卡

> Self-Attention 的优势
> 
> *   ⚡ 并行快
> *   🛸 长依赖一跳可达
> *   🔍 可解释性好
>     

但缺点：

*   **复杂度对长序列不友好**（平方级增长）

* * *

到这里，我们已经理解了 **为什么 Transformer 能替代 RNN/CNN**。

👉 下一步我们可以进入 **Part 6：Training（优化器、warmup、label smoothing 等 trick）**，学习论文里的训练小心机 🎩。

要不要继续？

## Prompt:
ok

## Response:
太棒啦 🚀 我们进入论文最后一个很实用的部分：

* * *

🧠 Part 6：Training 训练技巧
-----------------------

> Transformer 不光架构牛，还靠一堆训练“小心机”才能跑得稳！

* * *

### 🎯 Step 1：训练数据 & 批次

*   WMT’14 英德：约 **450万句对**
*   WMT’14 英法：约 **3600万句对**
*   BPE（子词单元，约 37k 词汇）
*   每个 batch ≈ **25k 源词 + 25k 目标词**

* * *

### 🎯 Step 2：硬件 & 训练时间

*   用 **8 块 P100 GPU**
*   Base 模型：100k steps（约 12 小时）
*   Big 模型：300k steps（约 3.5 天）

* * *

### 🎯 Step 3：优化器 & 学习率调度

用 **Adam**，但有个特别的 **warmup 学习率公式**：

$$
lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

📌 含义：

1.  前 4000 步 → 学习率线性上升（warmup）🔥
2.  之后 → 学习率按  $\frac{1}{\sqrt{step}}$  衰减

👉 避免一上来学习率太大，模型炸掉；同时保证后期稳定收敛。

* * *

### 🎯 Step 4：正则化

1.  **Dropout**
    *   残差连接后的输出、embedding+位置编码上都加了 dropout（P=0.1）
2.  **Label Smoothing**
    *   设置 ε = 0.1
    *   不是把正确标签设为 100%，而是给其他类别分一点概率
    *   好处：模型不会过度自信，泛化更好（BLEU 提升）

* * *

### 🎁 趣味比喻

*   **Warmup 学习率**：就像健身先热身，不然肌肉容易拉伤 🏋️‍♂️
*   **Label Smoothing**：就像老师判卷不苛刻，允许一点模糊空间 ✏️

* * *

### 🎯 小测问题

> Transformer 为什么要用 **Warmup 学习率调度**？
> 
> A. 防止一开始学习率太大导致训练不稳定  
> B. 让模型更快收敛  
> C. 模拟人类先慢后快的学习方式  
> D. 节省显存

你觉得哪个是主因？

## Prompt:
A

## Response:
💯 完全正确！答案就是 **A：防止一开始学习率太大导致训练不稳定** 🎯

* * *

### 🧠 补充说明：

*   Transformer 参数量很大，刚开始训练时参数还没稳定，如果学习率过大，就可能“炸掉”（loss nan）。
*   Warmup 阶段让学习率 **慢慢升高**，等模型进入稳定区间后再开始正常下降。
*   这就是为什么很多后来模型（BERT、GPT 系列）都继承了这个 **warmup 策略** 🔥

* * *

### 🎓 小总结闪卡

> **训练三板斧**：
> 
> *   ⚡ Adam + Warmup 学习率调度
> *   🌀 Dropout（防过拟合）
> *   🎯 Label Smoothing（让模型别太自信，泛化更好）
>     

* * *

到这里，我们就完整覆盖了 **Transformer 训练过程中的关键 trick**。

👉 下一步可以进入 **Part 7：实验结果 & Ablation Study（消融实验）**，看看哪些设计最重要，哪些可有可无。

要不要继续？

## Prompt:
ok

## Response:
好滴 🚀 我们进入最后的重点环节：

* * *

🧠 Part 7：实验结果 & Ablation Study
-------------------------------

> 论文里最精彩的部分之一：不仅给出 BLEU 分数，还做了 **消融实验**（Ablation Study），告诉我们 Transformer 哪些设计最关键！

* * *

### 🎯 Step 1：主要结果

*   **WMT14 英德翻译**
    *   Transformer (big) → **BLEU = 28.4**
    *   比之前最好的模型（包括 ensemble）还高出 **+2 BLEU**
*   **WMT14 英法翻译**
    *   Transformer (big) → **BLEU = 41.8**
    *   单模型就能超越之前的最优 ensemble

而且：训练速度大幅提升 ⚡（3.5 天 / 8 块 P100 就能出 SOTA）。

* * *

### 🎯 Step 2：消融实验（Table 3）

论文做了很多对比测试，这里挑最关键的👇：

1.  **注意力头数 (h)**
    *   单头（h=1） → 性能掉 0.9 BLEU
    *   8 头最好
    *   太多头（16, 32）反而下降

👉 说明 Multi-Head Attention 是必须的，但过多会稀释学习能力。

* * *

2.  **模型维度 (dmodel, dff)**
    *   更大的模型（1024, 4096） → BLEU 提升
    *   证明 “bigger is better” 在 Transformer 上已经成立 ⚡

* * *

3.  **Dropout**
    *   没有 dropout → BLEU 掉很多（24.6 vs 25.8）
    *   说明正则化很重要，避免过拟合

* * *

4.  **位置编码 (sinusoid vs learned)**
    *   换成可学习 embedding → BLEU 几乎没变
    *   说明正余弦只是设计上的一种方便选择，但不是唯一方式

* * *

### 🎯 Step 3：趣味总结

*   **多头注意力** → 就像请 8 个专家开会最好，1 个太弱，太多人又吵架 🤯
*   **大模型** → 钱多（参数多），效果就好 💸
*   **Dropout** → 相当于给模型“打疫苗”，免疫过拟合 🧪
*   **位置编码** → 用三角函数或学习 embedding 都行，模型不挑食 😋

* * *

### 🎯 小测问题

> 根据消融实验，以下哪一点 **不是** Transformer 成功的关键因素？  
> A. Multi-Head Attention  
> B. Dropout 正则化  
> C. 更大模型（big 版）  
> D. 正弦位置编码

你觉得呢？

## Prompt:
D

## Response:
💯 完全正确！答案就是 **D：正弦位置编码**。

* * *

### 🧠 为什么？

*   论文中实验发现：
    *   用正余弦位置编码 ✅
    *   换成 **learned positional embedding** ✅
    *   → BLEU 分数几乎没区别
*   所以位置编码的**形式**不是关键，关键在于 **必须有顺序信息**。

而：

*   **Multi-Head Attention** → 没它性能就大幅下降
*   **Dropout** → 不加直接过拟合，BLEU 掉很多
*   **大模型** → 明显提升性能

* * *

### 🎓 小总结闪卡

> **Transformer 成功的关键因子**
> 
> *   🎯 多头注意力（h=8最佳）
> *   🌀 Dropout 正则化
> *   📈 模型加大就更强
> *   📍 位置编码 → 只要有顺序信号，具体实现无所谓
>     

* * *

到这里，我们已经完整走完了 **Part 7 实验 & 消融**。  
接下来就是最后一节 **Part 8：总结与拓展思考（后续研究方向、局限性）** 🎯

要不要我带你做个大收官 + 拓展到 BERT/GPT 的预告？

## Prompt:
ok

## Response:
好极了 🎉 我们来做一个大收官 + 展望未来！

* * *

🧠 Part 8：总结与拓展思考
-----------------

### 🎯 Transformer 的核心贡献

1.  **架构创新**
    *   完全摒弃 RNN/CNN，基于 Attention 构建 Encoder-Decoder
    *   全并行训练 ⚡
    *   长依赖建模一步直达 🛸
2.  **关键模块**
    *   Scaled Dot-Product Attention
    *   Multi-Head Attention（8头最佳）
    *   残差 + LayerNorm
    *   Position-wise FFN
    *   Positional Encoding（形式不唯一，只要有顺序信息）
3.  **训练技巧**
    *   Adam + Warmup 学习率调度
    *   Dropout（防过拟合）
    *   Label Smoothing（泛化更好）

* * *

### 🎯 实验结果

*   WMT14 英德/英法翻译 **SOTA**
*   比之前最优的 RNN/CNN 模型快得多，BLEU 提升显著
*   消融实验确认：多头注意力、大模型、正则化是成功关键

* * *

### 🎯 Transformer 的局限性

*   Self-Attention **复杂度 O(n²)** → 序列长了（>1k token）非常耗显存
*   没有归纳偏置（不像 CNN 擅长局部结构）
*   输出生成依然是自回归的，推理速度不算快

* * *

### 🎯 未来发展方向（论文当时就提到的）

*   **长序列优化**：后来的 Longformer, Linformer, Performer 就是为了解决 n² 复杂度
*   **跨模态**：文本 → 图像/音频/视频（后来出现 Vision Transformer, Audio Transformer）
*   **预训练革命**：BERT（双向编码器）、GPT（解码器自回归）、T5（统一框架）

* * *

### 🎓 收官口诀

> **“Attention is All You Need” =**
> 
> *   并行快 ⚡
> *   全局看 🛸
> *   路径短 🔁
> *   Dropout 护航 🛡️
> *   大模型更强 💪
>     

* * *

✅ 到这里，你已经完整拆解了 Transformer 论文！  
接下来如果要深入，可以选几条路径：

*   🔍 **研究细节优化**：比如 LayerNorm 的位置、FFN 的设计
*   📚 **后续论文学习**：BERT / GPT / ViT / Performer
*   🧪 **实践**：手写一个 mini-transformer（PyTorch）

* * *

要不要我帮你整理一份 **“Transformer → BERT/GPT 学习路线图”**，让你顺着这条线往下深入？

## Prompt:
🧠 Part 7：实验结果 & Ablation Study ------------------------------- > 论文里最精彩的部分之一：不仅给出 BLEU 分数，还做了 \*\*消融实验\*\*（Ablation Study），告诉我们 Transformer 哪些设计最关键！ \* \* \* ### 🎯 Step 1：主要结果 \* \*\*WMT14 英德翻译\*\* \* Transformer (big) → \*\*BLEU = 28.4\*\* \* 比之前最好的模型（包括 ensemble）还高出 \*\*+2 BLEU\*\* \* \*\*WMT14 英法翻译\*\* \* Transformer (big) → \*\*BLEU = 41.8\*\* \* 单模型就能超越之前的最优 ensemble 而且：训练速度大幅提升 ⚡（3.5 天 / 8 块 P100 就能出 SOTA）。 \* \* \* ### 🎯 Step 2：消融实验（Table 3） 论文做了很多对比测试，这里挑最关键的👇： 1. \*\*注意力头数 (h)\*\* \* 单头（h=1） → 性能掉 0.9 BLEU \* 8 头最好 \* 太多头（16, 32）反而下降 👉 说明 Multi-Head Attention 是必须的，但过多会稀释学习能力。 \* \* \* 2. \*\*模型维度 (dmodel, dff)\*\* \* 更大的模型（1024, 4096） → BLEU 提升 \* 证明 “bigger is better” 在 Transformer 上已经成立 ⚡ \* \* \* 3. \*\*Dropout\*\* \* 没有 dropout → BLEU 掉很多（24.6 vs 25.8） \* 说明正则化很重要，避免过拟合 \* \* \* 4. \*\*位置编码 (sinusoid vs learned)\*\* \* 换成可学习 embedding → BLEU 几乎没变 \* 说明正余弦只是设计上的一种方便选择，但不是唯一方式 \* \* \* ### 🎯 Step 3：趣味总结 \* \*\*多头注意力\*\* → 就像请 8 个专家开会最好，1 个太弱，太多人又吵架 🤯 \* \*\*大模型\*\* → 钱多（参数多），效果就好 💸 \* \*\*Dropout\*\* → 相当于给模型“打疫苗”，免疫过拟合 🧪 \* \*\*位置编码\*\* → 用三角函数或学习 embedding 都行，模型不挑食 😋

based on Part 7
build python codes for ablation study on transformer using WMT14 dataset using pytorch, tensorboardX, loguru for visualization and compare

Readability Preferred with Better Name Style. Please provide a comprehensive and well-structured code solution for an Industrial scenario. The code should include robust classes or functions that have been thoroughly checked for correct boundary conditions. It should also adhere to the guidelines specified in the "Google style guides". The solution should prioritize high scalability and should not be a demo or a snippet. It is crucial that no part of the code is omitted. Furthermore, please include Chinese comments following the PEP 8 style guide for all key steps, and use DocString following the PEP 257 style guide and use typing for type annotations. The code should be efficient, and important information should be printed. **When the code is finished**, please perform a key step to recheck for its correctness and logic.

## Response:
已思考 42s

我需要为用户提供一个结构良好、易读的 Python 代码，以进行在WMT14数据集上使用Transformer进行消融研究的实验。代码必须遵循Google风格指南、PEP8规范，包含PEP257文档字符串和类型标注。代码要高效，包含适当的边界条件检查，并能够打印重要信息。我们没有办法运行重负载的训练，代码应支持预处理好的数据集，并避免过大数据的分发。

我在考虑使用 Hugging Face 数据集，"wmt14" 是否可用，还有可能需要选择某些子集（如 en-de）。考虑到数据集会很大，我会选择使用 SentencePiece 或 Byte Pair Encoding 进行分词，但这会增加复杂度。我们将使用 SacreBLEU 进行评估，并确保代码可以在没有网络依赖的情况下运行。

为了加载WMT14数据集，若通过 Hugging Face 无法找到“wmt14”，代码应支持回退到本地路径。模型实现方面，需要从零开始构建Transformer，包括MultiHeadAttention、EncoderLayer/DecoderLayer等，并加入ablation调节项。将执行BLEU评估、训练循环，并提供日志记录等功能。

在实现collate函数时，我们将确保创建目标输入（位移）和掩码。代码将包含Chinese注释，并保证完整。还会添加`if __name__ == '__main__':`入口来解析CLI参数。由于篇幅较大，代码可能需要400-600行，并且我们将添加`sanity_check()`以确保代码正确性。

完成

下面给出一套**工业级**、结构化且可扩展的**Transformer 消融实验**代码（基于 WMT14，PyTorch + tensorboardX + loguru）。  
特点：严格遵循 **Google Python Style Guide** + **PEP 8/257**，提供**类型标注**、**关键步骤中文注释**、**边界检查**、**Warmup 调度**、**Label Smoothing**、**权重共享**、**可切换位置编码（正弦/可学习）**、**多头数/维度/Dropout 消融**、**SacreBLEU 评估**、**丰富日志与可视化**、**Sanity Check** 等。

> 说明：本代码支持两种数据路径：  
> 1）优先尝试 HuggingFace `datasets` 的 `wmt14`（如 `wmt14 de-en`），  
> 2）若不可用，则从本地路径读取（需用户自行准备平行语料）。  
> 同时内置 `sentencepiece` 训练/加载，以便复现论文 BPE 子词化流程。

* * *

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Ablation Study on WMT14 with PyTorch.

This module provides an industrial-grade, end-to-end pipeline to run
ablation studies (as in "Attention is All You Need", Table 3) on the
Transformer architecture, focusing on:
  - Number of attention heads (h)
  - Model hidden dimension (d_model) and feed-forward dimension (d_ff)
  - Dropout rate
  - Positional encoding type (sinusoidal vs. learned)

Key features:
  * Robust engineering: boundary checks, clear class abstractions,
    Google style docstrings (PEP 257), PEP 8 compliance, and typing.
  * Training tricks: Adam + Noam warmup, label smoothing, tied embeddings.
  * Logging & viz: loguru for logs and tensorboardX for metrics.
  * Evaluation: SacreBLEU-based translation quality evaluation.
  * Data: HuggingFace datasets "wmt14" (if available) or local TSV files,
    SentencePiece BPE training or reuse for tokenization.
  * Sanity checks to validate shapes, masks and forward/backward logic.

Author: GPT-5 Thinking
Date: 2025-08-20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# Optional dependencies:
# datasets: for WMT14 loading. sentencepiece: for BPE tokenizer.
try:
    from datasets import load_dataset, DatasetDict  # type: ignore
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False

try:
    import sentencepiece as spm  # type: ignore
    SENTENCEPIECE_AVAILABLE = True
except Exception:
    SENTENCEPIECE_AVAILABLE = False

try:
    import sacrebleu  # type: ignore
    SACREBLEU_AVAILABLE = True
except Exception:
    SACREBLEU_AVAILABLE = False

# -----------------------------
# Utilities and Configuration
# -----------------------------

def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed.
    """
    # 设置全局随机种子，保证实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path, clear_if_exists: bool = False) -> Path:
    """Ensure a directory exists.

    Args:
        path: Directory path.
        clear_if_exists: Whether to clear the directory if exists.

    Returns:
        The directory path as Path.
    """
    # 确保输出目录存在；必要时清空
    p = Path(path)
    if p.exists() and clear_if_exists:
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Attributes:
        run_name: Unique run name (used for logging/checkpoints).
        log_dir: TensorBoard & log directory.
        save_dir: Checkpoint directory.
        data_dir: Local data directory fallback (if HF datasets unavailable).
        spm_dir: SentencePiece model dir.
        src_lang: Source language code (e.g., 'en').
        tgt_lang: Target language code (e.g., 'de').
        vocab_size: Subword vocabulary size.
        max_len: Maximum sequence length (tokens).
        batch_size: Tokens per batch (approx; dynamic padding).
        num_epochs: Number of epochs.
        lr: Base learning rate (Noam scaled).
        warmup_steps: Warmup steps for learning rate.
        label_smoothing: Label smoothing epsilon [0, 1).
        dropout: Dropout rate [0, 1).
        d_model: Hidden size (must be divisible by n_heads).
        n_heads: Number of attention heads.
        n_layers: Number of encoder/decoder layers.
        d_ff: Feed-forward hidden size.
        pos_encoding: 'sinusoidal' or 'learned'.
        weight_tying: Weight tying between embedding and generator.
        grad_clip: Gradient clipping norm (0 to disable).
        fp16: Use automatic mixed precision if True.
        save_every: Save checkpoint every N steps.
        log_every: Log every N steps.
        eval_every: Evaluate every N steps.
        num_workers: DataLoader workers.
        seed: Random seed.
        device: 'cuda' or 'cpu'.
        max_train_samples: Optional cap on training samples (debug).
        max_eval_samples: Optional cap on eval samples (debug).
        group: Ablation group tag for aggregation.
    """
    run_name: str = "baseline_en-de"
    log_dir: str = "runs"
    save_dir: str = "checkpoints"
    data_dir: str = "data_wmt14"
    spm_dir: str = "spm_models"
    src_lang: str = "en"
    tgt_lang: str = "de"
    vocab_size: int = 37000
    max_len: int = 256
    batch_size: int = 8192  # 动态padding, 以 tokens 数近似
    num_epochs: int = 10
    lr: float = 2.0  # Noam lr factor
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    dropout: float = 0.1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    weight_tying: bool = True
    grad_clip: float = 1.0
    fp16: bool = True
    save_every: int = 5000
    log_every: int = 100
    eval_every: int = 5000
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    group: str = "ablation"

# -----------------------------
# Tokenization with SentencePiece
# -----------------------------

class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper supporting training and inference.

    This wrapper provides encode/decode methods and special token handling.

    Note:
        Requires 'sentencepiece' package installed.

    Attributes:
        model_prefix: Prefix (path) for the SentencePiece model files.
        vocab_size: Vocabulary size.
        character_coverage: Character coverage for training.
        model_type: BPE model type.
        pad_id: Padding token id.
        bos_id: Begin-of-sentence id.
        eos_id: End-of-sentence id.
        unk_id: Unknown token id.
    """

    def __init__(
        self,
        model_prefix: str,
        vocab_size: int = 37000,
        character_coverage: float = 1.0,
        model_type: str = "bpe",
    ) -> None:
        if not SENTENCEPIECE_AVAILABLE:
            raise RuntimeError("sentencepiece is required but not installed.")
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        self._sp = spm.SentencePieceProcessor()  # type: ignore
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def train(
        self,
        input_files: List[str],
        user_defined_symbols: Optional[List[str]] = None,
    ) -> None:
        """Train a SentencePiece model.

        Args:
            input_files: List of input text files for training.
            user_defined_symbols: Optional extra symbols.
        """
        # 训练 SentencePiece 模型，生成 .model/.vocab
        args = {
            "input": ",".join(input_files),
            "model_prefix": self.model_prefix,
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "model_type": self.model_type,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }
        if user_defined_symbols:
            args["user_defined_symbols"] = ",".join(user_defined_symbols)
        spm.SentencePieceTrainer.Train(**args)  # type: ignore

    def load(self) -> None:
        """Load an existing SentencePiece model."""
        # 加载已训练的 SPM 模型
        model_file = f"{self.model_prefix}.model"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"SentencePiece model not found: {model_file}")
        self._sp.Load(model_file)  # type: ignore

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode text to a list of token ids.

        Args:
            text: Input string.
            add_bos: Prepend BOS token if True.
            add_eos: Append EOS token if True.

        Returns:
            List of token ids.
        """
        # 编码文本为 token id 列表
        ids = self._sp.EncodeAsIds(text)  # type: ignore
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text.

        Args:
            ids: Token id list.

        Returns:
            Decoded string.
        """
        # 将 token id 列表解码为字符串
        return self._sp.DecodeIds(ids)  # type: ignore

    @property
    def vocab_size_(self) -> int:
        """Return vocabulary size including special tokens."""
        return self._sp.GetPieceSize()  # type: ignore

# -----------------------------
# Dataset and DataModule
# -----------------------------

class TranslationExample(Dataset):
    """PyTorch dataset for tokenized parallel examples."""

    def __init__(
        self,
        src: List[List[int]],
        tgt: List[List[int]],
        max_len: int,
    ) -> None:
        # 存储已分词的源/目标序列
        self.src = src
        self.tgt = tgt
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        # 返回单个样本的 (src_ids, tgt_ids)
        s = self.src[index][: self.max_len]
        t = self.tgt[index][: self.max_len]
        return s, t

class DynamicBatchCollator:
    """Dynamic padding and mask creation for Transformer training."""

    def __init__(self, pad_id: int, device: torch.device) -> None:
        # collate 函数：根据 batch 序列最长长度进行动态 padding
        self.pad_id = pad_id
        self.device = device

    def _pad(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 对不等长序列进行 padding，并返回 mask
        max_len = max(len(s) for s in sequences)
        batch = len(sequences)
        tensor = torch.full((batch, max_len), fill_value=self.pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        # padding mask: True indicates padding positions
        pad_mask = (tensor == self.pad_id)
        return tensor, pad_mask

    def _subsequent_mask(self, size: int) -> torch.Tensor:
        # 下三角 mask，防止 decoder 看到未来位置
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return subsequent_mask  # shape: (1, T, T)

    def __call__(
        self, batch: List[Tuple[List[int], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        # 组装 batch，并生成模型所需的 mask
        src_list, tgt_list = zip(*batch)
        src, src_pad_mask = self._pad(list(src_list))
        tgt, tgt_pad_mask = self._pad(list(tgt_list))

        # 训练时将 tgt 分成输入 (tgt_in) 与 label (tgt_out)
        # 例如: [BOS, y1, y2, ..., y_{n-1}] -> 输入
        #       [y1, y2, ..., y_{n-1}, EOS] -> 标签
        tgt_in = tgt[:, :-1].contiguous()
        tgt_out = tgt[:, 1:].contiguous()

        # subsequent mask 与 pad mask 组合
        seq_len = tgt_in.size(1)
        subsequent = self._subsequent_mask(seq_len).to(self.device)
        # 注意：Transformer 里的 key_padding_mask 语义为 True 表示 padding
        # 对 decoder self-attention 的 mask 需要同时考虑 pad 和 subsequent
        src_key_padding_mask = src_pad_mask.to(self.device)  # (B, S)
        tgt_key_padding_mask = tgt_pad_mask[:, :-1].to(self.device)  # (B, T)
        tgt_mask = subsequent  # (1, T, T)

        return {
            "src": src.to(self.device),
            "tgt_in": tgt_in.to(self.device),
            "tgt_out": tgt_out.to(self.device),
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask,
            "tgt_mask": tgt_mask,
        }

class WMT14DataModule:
    """Data module to orchestrate loading, tokenization, and dataloaders.

    This supports HF datasets (if available) or local TSV files with columns:
    src \t tgt

    Attributes:
        cfg: Experiment configuration.
        sp_src: SentencePiece tokenizer for source language.
        sp_tgt: SentencePiece tokenizer for target language.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.sp_src: Optional[SentencePieceTokenizer] = None
        self.sp_tgt: Optional[SentencePieceTokenizer] = None
        self.ds_train: Optional[TranslationExample] = None
        self.ds_valid: Optional[TranslationExample] = None
        self.ds_test: Optional[TranslationExample] = None

    def _maybe_train_or_load_spm(
        self,
        src_corpus_files: List[str],
        tgt_corpus_files: List[str],
    ) -> None:
        # 训练或加载 SentencePiece 模型（若不存在）
        ensure_dir(self.cfg.spm_dir)
        src_prefix = str(Path(self.cfg.spm_dir) / f"spm_{self.cfg.src_lang}")
        tgt_prefix = str(Path(self.cfg.spm_dir) / f"spm_{self.cfg.tgt_lang}")

        self.sp_src = SentencePieceTokenizer(
            src_prefix, vocab_size=self.cfg.vocab_size
        )
        self.sp_tgt = SentencePieceTokenizer(
            tgt_prefix, vocab_size=self.cfg.vocab_size
        )

        # 若模型已存在则直接加载，否则训练
        if not os.path.exists(f"{src_prefix}.model") or not os.path.exists(
            f"{tgt_prefix}.model"
        ):
            logger.info("Training SentencePiece models for src/tgt ...")
            self.sp_src.train(src_corpus_files)
            self.sp_tgt.train(tgt_corpus_files)
        else:
            logger.info("Loading existing SentencePiece models for src/tgt ...")

        self.sp_src.load()
        self.sp_tgt.load()

    def _load_hf_wmt14(self) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """Load WMT14 via HuggingFace datasets if available.

        Returns:
            (train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt)
        """
        # 使用 HF datasets 加载 WMT14（若可用）
        # 为了兼容性，这里尝试几种配置名称
        configs_try = [
            (f"{self.cfg.src_lang}-{self.cfg.tgt_lang}", self.cfg.src_lang, self.cfg.tgt_lang),
            (f"{self.cfg.tgt_lang}-{self.cfg.src_lang}", self.cfg.tgt_lang, self.cfg.src_lang),
        ]
        last_exc = None
        for config_name, src, tgt in configs_try:
            try:
                dsd: DatasetDict = load_dataset("wmt14", config_name)  # type: ignore
                # datasets 字段可能为 'translation' 或 'translation' 键下字典
                def _extract_pairs(split_name: str) -> Tuple[List[str], List[str]]:
                    split = dsd[split_name]
                    xs, ys = [], []
                    for ex in split:
                        # HF wmt14 通常提供 ex['translation'] = {src_lang: str, tgt_lang: str}
                        titem = ex["translation"]
                        xs.append(titem[src])
                        ys.append(titem[tgt])
                    return xs, ys

                train_x, train_y = _extract_pairs("train")
                valid_x, valid_y = _extract_pairs("validation") if "validation" in dsd else _extract_pairs("test")
                test_x, test_y = _extract_pairs("test")

                # 若配置顺序与期望相反，则交换
                if src != self.cfg.src_lang:
                    # 交换回我们的定义
                    train_x, train_y = train_y, train_x
                    valid_x, valid_y = valid_y, valid_x
                    test_x, test_y = test_y, test_x

                return train_x, train_y, valid_x, valid_y, test_x, test_y
            except Exception as e:  # noqa: BLE001
                last_exc = e
                continue
        raise RuntimeError(
            f"Failed to load WMT14 via datasets. Last error: {last_exc}"
        )

    def _load_local_parallel(
        self,
        split: str,
    ) -> Tuple[List[str], List[str]]:
        """Load local TSV parallel data for a split.

        Args:
            split: One of 'train', 'valid', 'test'.

        Returns:
            (src_texts, tgt_texts)
        """
        # 从本地 TSV 读取平行语料，格式: src \t tgt
        file_path = Path(self.cfg.data_dir) / f"{split}.{self.cfg.src_lang}-{self.cfg.tgt_lang}.tsv"
        if not file_path.exists():
            raise FileNotFoundError(f"Local data file not found: {file_path}")
        xs, ys = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    # 跳过非法行
                    continue
                xs.append(parts[0])
                ys.append(parts[1])
        return xs, ys

    def prepare(self) -> None:
        """Prepare tokenizers and datasets."""
        # 准备数据：加载文本、（训练或加载）SPM、编码、构造Dataset
        logger.info("Preparing data module ...")

        if HF_DATASETS_AVAILABLE:
            try:
                train_x, train_y, valid_x, valid_y, test_x, test_y = self._load_hf_wmt14()
                logger.info("Loaded WMT14 from HuggingFace datasets.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"HF datasets loading failed: {e}. Falling back to local files.")
                train_x, train_y = self._load_local_parallel("train")
                valid_x, valid_y = self._load_local_parallel("valid")
                test_x, test_y = self._load_local_parallel("test")
        else:
            logger.warning("datasets not available. Using local files.")
            train_x, train_y = self._load_local_parallel("train")
            valid_x, valid_y = self._load_local_parallel("valid")
            test_x, test_y = self._load_local_parallel("test")

        # 限制样本数（调试/快速运行）
        if self.cfg.max_train_samples:
            train_x, train_y = train_x[: self.cfg.max_train_samples], train_y[: self.cfg.max_train_samples]
        if self.cfg.max_eval_samples:
            valid_x, valid_y = valid_x[: self.cfg.max_eval_samples], valid_y[: self.cfg.max_eval_samples]
            test_x, test_y = test_x[: self.cfg.max_eval_samples], test_y[: self.cfg.max_eval_samples]

        # 训练/加载 SentencePiece
        if not SENTENCEPIECE_AVAILABLE:
            raise RuntimeError("sentencepiece is required for BPE tokenization.")
        ensure_dir(self.cfg.spm_dir)
        # 为了更稳妥，SPM 用完整训练集训练
        src_tmp = str(Path(self.cfg.spm_dir) / f"tmp_src_{self.cfg.src_lang}.txt")
        tgt_tmp = str(Path(self.cfg.spm_dir) / f"tmp_tgt_{self.cfg.tgt_lang}.txt")
        with open(src_tmp, "w", encoding="utf-8") as f:
            for s in train_x:
                f.write(s.strip() + "\n")
        with open(tgt_tmp, "w", encoding="utf-8") as f:
            for s in train_y:
                f.write(s.strip() + "\n")
        self._maybe_train_or_load_spm([src_tmp], [tgt_tmp])
        # 清理临时文件
        try:
            os.remove(src_tmp)
            os.remove(tgt_tmp)
        except Exception:
            pass

        assert self.sp_src is not None and self.sp_tgt is not None

        # 编码文本为 token 序列
        def _encode_corpus(xs: List[str], ys: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
            src_tok = [self.sp_src.encode(x) for x in xs]
            tgt_tok = [self.sp_tgt.encode(y) for y in ys]
            return src_tok, tgt_tok

        train_src_tok, train_tgt_tok = _encode_corpus(train_x, train_y)
        valid_src_tok, valid_tgt_tok = _encode_corpus(valid_x, valid_y)
        test_src_tok, test_tgt_tok = _encode_corpus(test_x, test_y)

        # 构造 PyTorch Dataset
        self.ds_train = TranslationExample(train_src_tok, train_tgt_tok, self.cfg.max_len)
        self.ds_valid = TranslationExample(valid_src_tok, valid_tgt_tok, self.cfg.max_len)
        self.ds_test = TranslationExample(test_src_tok, test_tgt_tok, self.cfg.max_len)

        logger.info(
            f"Prepared datasets: train={len(self.ds_train)} valid={len(self.ds_valid)} test={len(self.ds_test)}"
        )

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders for train/valid/test.

        Returns:
            Tuple of train_loader, valid_loader, test_loader.
        """
        assert self.ds_train and self.ds_valid and self.ds_test
        device = torch.device(self.cfg.device)
        collate = DynamicBatchCollator(pad_id=0, device=device)

        # 注意：这里 batch_size 是句子数；由于我们使用动态 padding，
        # 若想更近似“按token数”控制，可改造为动态采样器（此处采用固定batch简化）。
        train_loader = DataLoader(
            self.ds_train,
            batch_size=max(1, self.cfg.batch_size // 128),  # 经验估计，避免OOM
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            self.ds_valid,
            batch_size=max(1, self.cfg.batch_size // 256),
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            self.ds_test,
            batch_size=max(1, self.cfg.batch_size // 256),
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, valid_loader, test_loader

# -----------------------------
# Model Components
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in Vaswani et al. 2017."""

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        # 预计算正弦位置编码矩阵，避免重复计算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，不参与训练
        self.register_buffer("pe", pe)  # shape: (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to input.

        Args:
            x: Input tensor with shape (B, T, D)

        Returns:
            Tensor with positional encodings added (B, T, D).
        """
        # 将预计算的位置编码加到输入，保持 shape 不变
        bsz, seq_len, d_model = x.size()
        x = x + self.pe[:seq_len].unsqueeze(0)
        return x

class LearnedPositionalEncoding(nn.Embedding):
    """Learned positional embedding."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings.

        Args:
            x: Input tensor with shape (B, T, D)

        Returns:
            Tensor with positional embeddings added (B, T, D).
        """
        bsz, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(bsz, 1)
        pos_embed = super().forward(positions)  # (B, T, D)
        return x + pos_embed

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for sequence-to-sequence tasks."""

    def __init__(self, vocab_size: int, smoothing: float, ignore_index: int = 0) -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0,1).")
        # Label Smoothing: 将 one-hot 标签分配少量概率给其他类
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.vocab_size = vocab_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy.

        Args:
            pred: Logits of shape (B*T, V).
            target: Target token ids of shape (B*T,).

        Returns:
            Loss scalar tensor.
        """
        # 忽略 padding 的位置
        mask = (target != self.ignore_index)
        target = target[mask]
        pred = pred[mask]
        if pred.numel() == 0:
            return pred.new_tensor(0.0)

        # 构造平滑分布
        with torch.no_grad():
            true_dist = torch.full_like(pred, fill_value=self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = F.kl_div(F.log_softmax(pred, dim=-1), true_dist, reduction="batchmean")
        return loss

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Noam learning rate scheduler from Vaswani et al. (2017)."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # 根据 step 动态计算学习率，warmup 后按 1/sqrt(step) 衰减
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

class TransformerSeq2Seq(nn.Module):
    """Transformer Seq2Seq model wrapping nn.Transformer with custom embeddings/PE.

    Supports:
      - Sinusoidal or learned positional encodings
      - Weight tying (encoder/decoder emb <-> generator)
    """

    def __init__(
        self,
        vocab_src: int,
        vocab_tgt: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal",
        weight_tying: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1).")

        # Embeddings
        self.src_embed = nn.Embedding(vocab_src, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(vocab_tgt, d_model, padding_idx=0)

        # Positional Encoding
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(max_len=max_len, d_model=d_model)
        else:
            raise ValueError("pos_encoding must be 'sinusoidal' or 'learned'.")

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # 使用 (B, T, D) 接口
            norm_first=False,  # 与原论文后归一化一致
        )

        # Generator (pre-softmax linear)
        self.generator = nn.Linear(d_model, vocab_tgt, bias=False)

        # Weight tying
        if weight_tying:
            if self.tgt_embed.weight.size() != self.generator.weight.size():
                raise ValueError("Weight tying requires tgt_embed and generator to have same shape.")
            self.generator.weight = self.tgt_embed.weight  # weight tying

        # Initialization
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 参数初始化（Xavier）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src_ids: Source token ids (B, S).
            src_key_padding_mask: Padding mask (B, S), True for PAD positions.

        Returns:
            Encoder memory tensor (B, S, D).
        """
        src_emb = self.src_embed(src_ids) * math.sqrt(self.transformer.d_model)
        src_emb = self.pos_enc(src_emb)
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,  # True 表示 padding
        )
        return memory

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target sequence given encoder memory.

        Args:
            tgt_in: Target input ids (B, T).
            memory: Encoder memory (B, S, D).
            tgt_mask: Causal mask for decoder self-attention (T, T) or (1, T, T).
            tgt_key_padding_mask: Padding mask for target (B, T).
            memory_key_padding_mask: Padding mask for source (B, S).

        Returns:
            Decoder output logits (B, T, V).
        """
        tgt_emb = self.tgt_embed(tgt_in) * math.sqrt(self.transformer.d_model)
        tgt_emb = self.pos_enc(tgt_emb)
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask[0],  # (T, T)
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(out)  # (B, T, V)
        return logits

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            src_ids: Source token ids (B, S).
            tgt_in: Target input ids (B, T).
            tgt_mask: Causal mask (1, T, T).
            src_key_padding_mask: Source padding mask (B, S).
            tgt_key_padding_mask: Target padding mask (B, T).

        Returns:
            Logits (B, T, V).
        """
        memory = self.encode(src_ids, src_key_padding_mask)
        logits = self.decode(
            tgt_in=tgt_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Greedy decoding for inference.

        Args:
            src_ids: Source ids (B, S).
            src_key_padding_mask: Source padding mask (B, S).
            max_len: Max generation length.
            bos_id: BOS token id.
            eos_id: EOS token id.

        Returns:
            Generated ids (B, T_out).
        """
        # 贪心解码，逐步生成，直到 EOS 或达到最大长度
        device = src_ids.device
        memory = self.encode(src_ids, src_key_padding_mask)
        B = src_ids.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            T = ys.size(1)
            subsequent_mask = torch.triu(
                torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1
            ).unsqueeze(0)  # (1, T, T)
            logits = self.decode(
                tgt_in=ys,
                memory=memory,
                tgt_mask=subsequent_mask,
                tgt_key_padding_mask=torch.zeros((B, T), dtype=torch.bool, device=device),
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (B,)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if torch.all(finished):
                break
        return ys

# -----------------------------
# Trainer
# -----------------------------

class Trainer:
    """Trainer handling training loop, evaluation, and logging."""

    def __init__(self, cfg: ExperimentConfig, data: WMT14DataModule) -> None:
        self.cfg = cfg
        self.data = data
        self.device = torch.device(cfg.device)
        self.writer: Optional[SummaryWriter] = None
        self.global_step = 0

        # 初始化日志与目录
        ensure_dir(cfg.log_dir)
        ensure_dir(cfg.save_dir)
        self.run_dir = ensure_dir(Path(cfg.log_dir) / cfg.group / cfg.run_name, clear_if_exists=False)
        self.ckpt_dir = ensure_dir(Path(cfg.save_dir) / cfg.group / cfg.run_name, clear_if_exists=False)

    def _build_model_and_optim(self) -> Tuple[TransformerSeq2Seq, torch.optim.Optimizer, NoamScheduler]:
        assert self.data.sp_src is not None and self.data.sp_tgt is not None
        vocab_src = self.data.sp_src.vocab_size_
        vocab_tgt = self.data.sp_tgt.vocab_size_
        logger.info(f"Vocab sizes -> src: {vocab_src} tgt: {vocab_tgt}")

        model = TransformerSeq2Seq(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            d_ff=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            max_len=self.cfg.max_len,
            pos_encoding=self.cfg.pos_encoding,
            weight_tying=self.cfg.weight_tying,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=self.cfg.d_model, warmup_steps=self.cfg.warmup_steps)

        return model, optimizer, scheduler

    def _evaluate_bleu(self, model: TransformerSeq2Seq, loader: DataLoader) -> float:
        # 使用 SacreBLEU 对验证集/测试集进行评估
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu not installed; returning 0.0 as BLEU.")
            return 0.0
        assert self.data.sp_tgt is not None and self.data.sp_src is not None
        refs: List[str] = []
        hyps: List[str] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                src = batch["src"]
                src_kpm = batch["src_key_padding_mask"]
                # 贪心解码（可扩展 beam search）
                gen = model.greedy_decode(
                    src_ids=src,
                    src_key_padding_mask=src_kpm,
                    max_len=self.cfg.max_len,
                    bos_id=self.data.sp_tgt.bos_id,
                    eos_id=self.data.sp_tgt.eos_id,
                )
                # 取 batch 中的参考/假设句子（去掉BOS）
                tgt_out = batch["tgt_out"].cpu().tolist()
                gen_out = gen.cpu().tolist()
                for ref_ids, hyp_ids in zip(tgt_out, gen_out):
                    # 去除 padding 与 超出 eos 的部分
                    ref_ids_clean = [tid for tid in ref_ids if tid not in (0,)]
                    hyp_ids_clean = [tid for tid in hyp_ids if tid not in (0,)]
                    # 去掉起始 BOS
                    if len(hyp_ids_clean) > 0 and hyp_ids_clean[0] == self.data.sp_tgt.bos_id:
                        hyp_ids_clean = hyp_ids_clean[1:]
                    # 截断至 EOS
                    if self.data.sp_tgt.eos_id in hyp_ids_clean:
                        idx = hyp_ids_clean.index(self.data.sp_tgt.eos_id)
                        hyp_ids_clean = hyp_ids_clean[: idx + 1]
                    ref_text = self.data.sp_tgt.decode(ref_ids_clean)
                    hyp_text = self.data.sp_tgt.decode(hyp_ids_clean)
                    refs.append(ref_text)
                    hyps.append(hyp_text)

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score  # type: ignore
        return float(bleu)

    def train(self) -> Dict[str, Any]:
        """Run full training with periodic evaluation and checkpointing."""
        train_loader, valid_loader, test_loader = self.data.create_dataloaders()
        model, optimizer, scheduler = self._build_model_and_optim()
        criterion = LabelSmoothingLoss(
            vocab_size=self.data.sp_tgt.vocab_size_,
            smoothing=self.cfg.label_smoothing,
            ignore_index=0,
        )

        self.writer = SummaryWriter(logdir=str(self.run_dir))

        # 混合精度
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.fp16)

        logger.info(f"Start training: epochs={self.cfg.num_epochs}, device={self.cfg.device}")
        best_valid_bleu = -1.0
        best_ckpt_path = None
        start_time = time.time()

        for epoch in range(1, self.cfg.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            step_in_epoch = 0

            for batch in train_loader:
                self.global_step += 1
                step_in_epoch += 1

                src = batch["src"]
                tgt_in = batch["tgt_in"]
                tgt_out = batch["tgt_out"]
                src_kpm = batch["src_key_padding_mask"]
                tgt_kpm = batch["tgt_key_padding_mask"]
                tgt_mask = batch["tgt_mask"]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                    logits = model(
                        src_ids=src,
                        tgt_in=tgt_in,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_kpm,
                        tgt_key_padding_mask=tgt_kpm,
                    )
                    # 展平计算交叉熵/标签平滑
                    B, T, V = logits.shape
                    loss = criterion(
                        logits.reshape(B * T, V),
                        tgt_out.reshape(B * T),
                    )

                scaler.scale(loss).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()

                if self.global_step % self.cfg.log_every == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / step_in_epoch
                    logger.info(
                        f"step={self.global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                        f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} lr={cur_lr:.6e}"
                    )
                    assert self.writer is not None
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", cur_lr, self.global_step)

                if self.global_step % self.cfg.eval_every == 0:
                    valid_bleu = self._evaluate_bleu(model, valid_loader)
                    logger.info(f"[Eval] step={self.global_step} valid_BLEU={valid_bleu:.2f}")
                    assert self.writer is not None
                    self.writer.add_scalar("valid/BLEU", valid_bleu, self.global_step)

                    # Save best
                    if valid_bleu > best_valid_bleu:
                        best_valid_bleu = valid_bleu
                        best_ckpt_path = Path(self.ckpt_dir) / f"best_step{self.global_step}.pt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "config": dataclasses.asdict(self.cfg),
                                "global_step": self.global_step,
                            },
                            str(best_ckpt_path),
                        )
                        logger.info(f"Saved best checkpoint to {best_ckpt_path}")

                if self.global_step % self.cfg.save_every == 0:
                    ckpt_path = Path(self.ckpt_dir) / f"ckpt_step{self.global_step}.pt"
                    torch.save(
                        {"model": model.state_dict(), "global_step": self.global_step},
                        str(ckpt_path),
                    )
                    logger.info(f"Saved checkpoint to {ckpt_path}")

            # 每个 epoch 结束，做一次评估
            valid_bleu = self._evaluate_bleu(model, valid_loader)
            logger.info(f"[EpochEnd] epoch={epoch} valid_BLEU={valid_bleu:.2f}")
            assert self.writer is not None
            self.writer.add_scalar("valid/BLEU_epoch", valid_bleu, epoch)

            if valid_bleu > best_valid_bleu:
                best_valid_bleu = valid_bleu
                best_ckpt_path = Path(self.ckpt_dir) / f"best_epoch{epoch}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": dataclasses.asdict(self.cfg),
                        "global_step": self.global_step,
                    },
                    str(best_ckpt_path),
                )
                logger.info(f"Saved best checkpoint to {best_ckpt_path}")

        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time/3600:.2f}h. Best valid BLEU={best_valid_bleu:.2f}")

        # 测试集评估（使用最佳模型）
        test_bleu = 0.0
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            logger.info(f"Loading best checkpoint {best_ckpt_path} for test evaluation ...")
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            test_bleu = self._evaluate_bleu(model, test_loader)
            logger.info(f"[Test] BLEU={test_bleu:.2f}")
            assert self.writer is not None
            self.writer.add_scalar("test/BLEU", test_bleu, self.global_step)

        # 关闭 writer
        if self.writer is not None:
            self.writer.close()

        return {"best_valid_bleu": best_valid_bleu, "test_bleu": test_bleu}

# -----------------------------
# Ablation Orchestrator
# -----------------------------

class AblationRunner:
    """Run a set of ablation experiments and collate results."""

    def __init__(self, base_cfg: ExperimentConfig) -> None:
        self.base_cfg = base_cfg
        self.results: List[Dict[str, Any]] = []

    def _run_single(self, cfg: ExperimentConfig) -> Dict[str, Any]:
        # 运行单次实验（包含训练与评估），并记录结果
        set_global_seed(cfg.seed)
        logger.info(f"Running experiment: {cfg.run_name} | group={cfg.group}")
        logger.info(json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))

        data_mod = WMT14DataModule(cfg)
        data_mod.prepare()

        trainer = Trainer(cfg, data_mod)
        result = trainer.train()
        result_row = {
            "run_name": cfg.run_name,
            "group": cfg.group,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "n_heads": cfg.n_heads,
            "dropout": cfg.dropout,
            "pos_encoding": cfg.pos_encoding,
            "best_valid_bleu": result["best_valid_bleu"],
            "test_bleu": result["test_bleu"],
        }
        self.results.append(result_row)
        # 将结果写入 group 目录下 JSON
        out_dir = ensure_dir(Path(cfg.log_dir) / cfg.group)
        with open(out_dir / f"summary_{int(time.time())}.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        return result_row

    def run_group(self, group: Literal["heads", "dimensions", "dropout", "posenc", "baseline"]) -> List[Dict[str, Any]]:
        """Run an ablation group similar to Table 3 in the paper.

        Args:
            group: One of "heads", "dimensions", "dropout", "posenc", "baseline".

        Returns:
            List of results dicts.
        """
        cfgs: List[ExperimentConfig] = []
        base = dataclasses.replace(self.base_cfg)

        if group == "baseline":
            cfgs = [dataclasses.replace(base, run_name=f"baseline_{base.src_lang}-{base.tgt_lang}")]
        elif group == "heads":
            # Vary number of heads keeping computation roughly similar by adjusting per-head dim
            for h in [1, 4, 8, 16]:
                # 保持 d_model 不变（实际论文里保持 per-head dim 变动）
                cfgs.append(
                    dataclasses.replace(
                        base,
                        n_heads=h,
                        run_name=f"heads_h{h}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        elif group == "dimensions":
            # Vary d_model and d_ff
            for (dm, df) in [(256, 1024), (512, 2048), (1024, 4096)]:
                cfgs.append(
                    dataclasses.replace(
                        base,
                        d_model=dm,
                        d_ff=df,
                        n_heads=8 if dm % 8 == 0 else 4,  # 保证可整除
                        run_name=f"dim_dm{dm}_df{df}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        elif group == "dropout":
            for p in [0.0, 0.1, 0.2, 0.3]:
                cfgs.append(
                    dataclasses.replace(
                        base,
                        dropout=p,
                        run_name=f"dropout_p{str(p).replace('.','_')}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        elif group == "posenc":
            for t in ["sinusoidal", "learned"]:
                cfgs.append(
                    dataclasses.replace(
                        base,
                        pos_encoding=t,  # type: ignore[arg-type]
                        run_name=f"posenc_{t}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        else:
            raise ValueError(f"Unknown group: {group}")

        all_results = []
        for cfg in cfgs:
            all_results.append(self._run_single(cfg))
        return all_results

# -----------------------------
# Sanity Checks
# -----------------------------

def run_sanity_checks() -> None:
    """Run minimal sanity checks for shapes and masks.

    This does not train; it only validates forward pass shapes and decoding.
    """
    logger.info("Running sanity checks for model shapes and logic ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, n_heads, d_ff = 256, 4, 1024
    vocab = 100
    max_len = 64
    model = TransformerSeq2Seq(
        vocab_src=vocab,
        vocab_tgt=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=2,
        d_ff=d_ff,
        dropout=0.1,
        max_len=max_len,
        pos_encoding="sinusoidal",
        weight_tying=False,
    ).to(device)

    B, S, T = 2, 10, 12
    src = torch.randint(4, vocab, (B, S), device=device)
    tgt_in = torch.randint(4, vocab, (B, T), device=device)

    src_kpm = torch.zeros((B, S), dtype=torch.bool, device=device)
    tgt_kpm = torch.zeros((B, T), dtype=torch.bool, device=device)
    tgt_mask = torch.triu(torch.ones((1, T, T), dtype=torch.bool, device=device), diagonal=1)

    logits = model(
        src_ids=src,
        tgt_in=tgt_in,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_kpm,
        tgt_key_padding_mask=tgt_kpm,
    )
    assert logits.shape == (B, T, vocab), f"Unexpected logits shape: {logits.shape}"

    gen = model.greedy_decode(
        src_ids=src,
        src_key_padding_mask=src_kpm,
        max_len=20,
        bos_id=1,
        eos_id=2,
    )
    assert gen.shape[0] == B and gen.shape[1] >= 1, "Greedy decode shape mismatch."
    logger.info("Sanity checks passed ✅")

# -----------------------------
# Main and CLI
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument sequence.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Transformer Ablation Study on WMT14 (PyTorch + tensorboardX + loguru)"
    )
    parser.add_argument("--run_name", type=str, default="baseline_en-de", help="Run name.")
    parser.add_argument("--group", type=str, default="baseline", choices=["baseline", "heads", "dimensions", "dropout", "posenc"], help="Ablation group.")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language.")
    parser.add_argument("--tgt_lang", type=str, default="de", help="Target language.")
    parser.add_argument("--vocab_size", type=int, default=37000, help="SentencePiece vocab size.")
    parser.add_argument("--max_len", type=int, default=256, help="Max token length.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Approx tokens per batch (heuristic).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=2.0, help="Base LR factor (Noam).")
    parser.add_argument("--warmup", type=int, default=4000, help="Warmup steps.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden size.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers.")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN hidden size.")
    parser.add_argument("--pos_enc", type=str, default="sinusoidal", choices=["sinusoidal", "learned"], help="Positional encoding type.")
    parser.add_argument("--no_tying", action="store_true", help="Disable weight tying.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Grad clipping norm.")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps.")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--eval_every", type=int, default=5000, help="Eval every N steps.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log dir.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint dir.")
    parser.add_argument("--data_dir", type=str, default="data_wmt14", help="Local data dir fallback.")
    parser.add_argument("--spm_dir", type=str, default="spm_models", help="SentencePiece model dir.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (debug).")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit eval samples (debug).")
    parser.add_argument("--sanity_check", action="store_true", help="Run sanity checks and exit.")
    return parser.parse_args(argv)

def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from argparse args.

    Args:
        args: Parsed arguments.

    Returns:
        ExperimentConfig instance.
    """
    cfg = ExperimentConfig(
        run_name=args.run_name,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        spm_dir=args.spm_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        pos_encoding=args.pos_enc,
        weight_tying=not args.no_tying,
        grad_clip=args.grad_clip,
        fp16=not args.no_fp16,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        group=args.group,
    )
    return cfg

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for ablation experiments."""
    args = parse_args(argv)
    # 设置日志格式
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")

    if args.sanity_check:
        run_sanity_checks()
        return

    cfg = build_config_from_args(args)
    set_global_seed(cfg.seed)

    # 单组/多组消融实验入口
    ablation = AblationRunner(cfg)
    results = ablation.run_group(cfg.group)

    # 打印总览表
    logger.info("==== Ablation Summary ====")
    for row in results:
        logger.info(
            f"[{row['run_name']}] h={row['n_heads']} d_model={row['d_model']} d_ff={row['d_ff']} "
            f"dropout={row['dropout']} pos={row['pos_encoding']} | "
            f"best_valid_BLEU={row['best_valid_bleu']:.2f} test_BLEU={row['test_bleu']:.2f}"
        )

if __name__ == "__main__":
    main()
```

* * *

✅ 关键正确性与逻辑复核（不依赖运行的核查步骤）
========================

下面是对核心模块的**逐点核查**（保证逻辑、边界、形状一致性）：

1.  **边界与约束检查**
    *   `d_model % n_heads == 0`：在 `TransformerSeq2Seq.__init__` 显式校验，避免头维度非整数。
    *   `0 ≤ dropout < 1`、`0 ≤ label_smoothing < 1`：构造时校验，防止非法超参。
    *   权重共享（tgt embedding ↔ generator）时，先校验两者权重尺寸一致。
2.  **掩码逻辑检查**
    *   `DynamicBatchCollator`：
        *   `src_key_padding_mask` / `tgt_key_padding_mask` 语义与 `nn.Transformer` 一致（`True` 表示 PAD）。
        *   `tgt_mask` 使用上三角 `True` 实现因果屏蔽，并传入 `decoder`。
        *   `tgt_in`/`tgt_out` 通过位移实现自回归训练（与论文一致）。
    *   `greedy_decode` 中逐步构造 `subsequent_mask` 并在 `decoder` 使用；`memory_key_padding_mask` 始终是源端 pad mask。
3.  **形状与维度检查**
    *   `batch_first=True`，全程使用 (B, T, D)；`tgt_mask` 传入 `decoder` 时做了 `[..., 0]` 转换为 (T, T)。
    *   `LabelSmoothingLoss` 展平为 (B_T, V) 与 (B_T,) 后计算 KLDiv，mask 掉 PAD。
4.  **学习率调度**
    *   `NoamScheduler.get_lr()` 使用 `(d_model^-0.5) * min(step^-0.5, step*warmup^-1.5)`，符合论文公式（3）。
5.  **位置编码**
    *   提供 `SinusoidalPositionalEncoding` 与 `LearnedPositionalEncoding`；两者均与输入形状匹配 (B, T, D)。
6.  **权重共享**
    *   与论文一致：`generator.weight = tgt_embed.weight`，确保 logits 投影共享。
7.  **评估与指标**
    *   `sacrebleu.corpus_bleu` 以 `(hyps, [refs])` 方式计算。贪心解码截断到 EOS，且去除 BOS/PAD。
8.  **鲁棒性**
    *   数据加载：优先 HuggingFace `wmt14`，失败回退本地 TSV 文件（`data_dir/train.en-de.tsv` 等）。
    *   SentencePiece：若模型不存在则训练，否则加载；清理临时文件。
    *   训练过程：混合精度 + 梯度裁剪 + 按步保存/评估 + 最优模型另存。
9.  **Sanity Check（可运行自检）**
    *   `--sanity_check` 将执行一个小型前向/解码检查：确保 logits 形状为 (B, T, V)，greedy decode 输出 (B, T\_gen)。
10.  **消融组装**
    *   `AblationRunner.run_group()` 支持 `baseline/heads/dimensions/dropout/posenc` 五组，与论文 Table 3 要点对齐（可进一步扩展 beams、层数等）。

> 以上核查覆盖了**掩码语义**、**形状一致性**、**学习率策略**、**位置编码**、**权重共享**、**数据/评估**等关键易错点，符合论文与工业实践要求。

* * *

使用示例（命令行）
---------

*   **基线（英→德）**
    ```bash
    python main.py --group baseline --src_lang en --tgt_lang de --run_name baseline_en-de
    ```
*   **多头数消融**
    ```bash
    python main.py --group heads --src_lang en --tgt_lang de --epochs 10
    ```
*   **模型维度消融**
    ```bash
    python main.py --group dimensions --src_lang en --tgt_lang de --epochs 10
    ```
*   **Dropout 消融**
    ```bash
    python main.py --group dropout --src_lang en --tgt_lang de --epochs 10
    ```
*   **位置编码消融**
    ```bash
    python main.py --group posenc --src_lang en --tgt_lang de --epochs 10
    ```
*   **快速自检（不训练）**
    ```bash
    python main.py --sanity_check
    ```

> 重要提示：首次运行建议指定 `--max_train_samples/--max_eval_samples` 做小集调通；正式训练请确保磁盘/显存资源充足，并根据机器配置合适地调小 `batch_size` 或 `max_len`。TensorBoard 启动：`tensorboard --logdir runs`。

如果你希望我把 **beam search**、**梯度累积**、**多卡分布式（DDP）**、**线性注意力替换以支持长序列** 等也一起加上，我可以继续扩展这个工程版脚手架。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)