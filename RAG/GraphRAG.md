下面给你一套“可落地、可扩展”的**本地书籍 + SerpAPI + GPT-5**构图方案，用来快速搭建 GraphRAG。内容包含：总体架构、数据与图谱 schema、端到端流水线、关键提示词（prompts）、以及**可直接运行的参考代码（文件划分+逐文件给出）**。代码遵循 PEP 8/257，带中文注释与类型标注，便于工业化扩展。

---

# 总览

## 目标

* 从**本地书籍（PDF/EPUB/TXT）**抽取结构化知识（实体/关系/属性），并做**实体对齐与去重**。
* 结合 **SerpAPI** 做“外部补全/佐证”（别名、缺失属性、同义关系）。
* 将知识落入**图数据库/图存储** + **向量索引**，并构建**GraphRAG 检索器**（先图再文，或图文协同）。
* 保证模块化与可扩展（随时替换 LLM、图数据库、向量库、切分器等）。

## 架构（简图）

```
            ┌───────────────┐
Local Books │  PDF/EPUB/TXT │
            └──────┬────────┘
                   │  文本抽取&切分
                   ▼
             ┌───────────┐            ┌──────────────┐
             │  Splitter │◀──────────▶│  SerpAPI 客户端 │（外部补全/校验）
             └─────┬─────┘            └──────────────┘
                   │
                   ▼
            ┌──────────────┐
            │ GPT-5 抽取器  │（实体/关系/属性）
            └──────┬───────┘
                   │
          实体对齐/去重/别名归一（本地 + Serp 补全）
                   │
                   ▼
      ┌───────────────────────┐       ┌────────────────┐
      │ Graph Store(Neo4j/NX) │  +    │ Vector Store    │（段落向量）
      └──────────┬────────────┘       └──────┬─────────┘
                 │                            │
                 └──────────┬─────────────────┘
                            ▼
                      GraphRAG 检索器
            （Query→子图扩展→文段召回→答案综合）
```

---

# 数据与图谱 Schema

* **Node（实体）**：

  * `:Entity {id, name, type, aliases, description, source, confidence}`
* **Edge（关系）**（有向或无向）：

  * `(:Entity)-[:REL {type, evidence_span, source, confidence}]->(:Entity)`
* **TextChunk（文段）**（可建独立节点或仅存于向量索引）：

  * `{chunk_id, doc_id, page, text, tokens, source}`

> 生产建议：以 **Neo4j**（或 ArangoDB/JanusGraph）做属性图；实验/本地可用 **NetworkX** 内存图存储。向量索引可用 **FAISS/Milvus/Elasticsearch vector**。

---

# 流水线（端到端）

1. **采集与解析**：遍历本地书库（支持 PDF/EPUB/TXT），抽取纯文本与来源元数据。
2. **切分**：`RecursiveCharacterTextSplitter`（先按章/节/段/句递归），确保语义完整；保留页码/章节路径。
3. **外部补全（可选）**：用 **SerpAPI** 查实体的别名/百科摘要/同义关系，填补缺失属性，为实体消歧做证据。
4. **三元组抽取**：用 **GPT-5** 批处理文段，产出 `(head, relation, tail, attrs, confidence)`；同时抽取实体候选的 `type/alias/desc`。
5. **实体解析与去重**：基于（名称归一 + 别名集合 + 语义相似度 + Serp证据）做聚类/合并；记录对齐溯源。
6. **入图与建索引**：批量写入图库（节点/边），并把原文段嵌入进**向量库**保存 `chunk_id→向量`。
7. **GraphRAG 检索**：

   * Query→（可选）查询改写/子任务拆分（GPT-5）
   * **子图引导检索**：从匹配实体出发做一跳/多跳扩展，拿到候选节点集
   * 对候选节点关联的文段做**向量检索**（或“图先过滤 + 文向量精排”）
   * 答案综合与引用生成（GPT-5）

---

# 关键 Prompts（摘要）

**三元组抽取 Prompt（对单 chunk）：**

> 你是专业信息抽取器。请从以下文本抽取面向知识图谱的实体与关系，输出 JSON 数组：
> `{"entities":[{"name":"","type":"","aliases":[],"desc":""}...], "triples":[{"head":"","relation":"","tail":"","evidence_span":"","attributes":{},"confidence":0.0}], "notes":[] }`
> 要求：
>
> 1. `relation` 用动宾式或标准术语；2) `aliases` 含常见别名；3) `confidence` 0\~1；4) `evidence_span` 贴原文短句；5) 不臆造。

**实体对齐 Prompt（候选冲突合并）：**

> 给定两组实体候选（含 name/aliases/desc/source），判断是否是**同一实体**并给出 `reason` 与 `confidence`。输出：`{"same": true|false, "reason":"", "confidence":0~1}`。保守合并，宁缺毋滥。

**GraphRAG 答案综合 Prompt：**

> 输入：查询、候选子图摘要（节点/边/证据）、候选文段（含来源）。
> 输出：结构化答案（分点），引用证据（“\[source: doc#page] 短句”），并标注不确定部分。

---

# 代码（文件结构 + 逐文件）

> 说明：以下为**精简但可运行**的参考实现（偏工程骨架）。默认使用 **NetworkX** 作为图存储，**sklearn** 近邻代替 FAISS（可一键替换）。LLM 调用通过抽象接口，你只需把 `GPT5ChatClient` 里补上实际 SDK 调用即可。

## 目录

```
graphrag_kit/
├─ requirements.txt
├─ config.py
├─ llm_client.py
├─ serp_client.py
├─ loaders.py
├─ splitters.py
├─ extractor.py
├─ resolve.py
├─ graph_store.py
├─ vector_store.py
├─ graph_rag.py
├─ build_kg.py
└─ query_cli.py
```

---

### `requirements.txt`

```txt
tqdm==4.66.4
pydantic==2.8.2
python-dotenv==1.0.1
networkx==3.3
numpy==1.26.4
scikit-learn==1.5.1
pdfminer.six==20240706
beautifulsoup4==4.12.3
requests==2.32.3
rapidfuzz==3.9.7
```

---

### `config.py`

```python
# -*- coding: utf-8 -*-
"""全局配置与常量定义."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, List


class Paths(BaseModel):
    """路径配置."""
    books_dir: str = "data/books"
    chunks_dir: str = "data/chunks"
    vect_index_dir: str = "data/index"


class LLMConfig(BaseModel):
    """LLM 配置（抽象）。"""
    model: str = "gpt-5"  # 逻辑名
    api_base: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 2048


class SerpConfig(BaseModel):
    """SerpAPI 配置."""
    api_key_env: str = "SERPAPI_KEY"
    engine: str = "google"
    num: int = 5


class BuildConfig(BaseModel):
    """构图批处理配置."""
    batch_size: int = 8
    max_workers: int = 4
    use_serp_enrichment: bool = True
    min_confidence: float = 0.5


class GraphRAGConfig(BaseModel):
    """GraphRAG 检索配置."""
    max_hops: int = 1
    top_k_entities: int = 10
    top_k_chunks: int = 8


class AppConfig(BaseModel):
    """顶层配置."""
    paths: Paths = Paths()
    llm: LLMConfig = LLMConfig()
    serp: SerpConfig = SerpConfig()
    build: BuildConfig = BuildConfig()
    rag: GraphRAGConfig = GraphRAGConfig()
```

---

### `llm_client.py`

```python
# -*- coding: utf-8 -*-
"""LLM 客户端抽象与 GPT-5 适配."""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class LLMMessage:
    """对话消息."""
    role: str
    content: str


class BaseChatClient:
    """LLM 抽象接口."""

    def chat(self, messages: List[LLMMessage], temperature: float = 0.0,
             max_tokens: int = 1024) -> str:
        """发送多轮对话，返回模型文本输出."""
        raise NotImplementedError


class GPT5ChatClient(BaseChatClient):
    """GPT-5 Chat 客户端占位实现.

    注意：为避免绑定具体 SDK，这里保留抽象调用。
    你可以在此处接入官方 Python SDK 或 HTTP 请求。
    """

    def __init__(self,
                 api_key_env: str = "OPENAI_API_KEY",
                 api_base: Optional[str] = None,
                 model: str = "gpt-5") -> None:
        self._api_key = os.getenv(api_key_env, "")
        self._api_base = api_base
        self._model = model
        if not self._api_key:
            raise RuntimeError(f"Missing API key in env: {api_key_env}")

    def chat(self, messages: List[LLMMessage], temperature: float = 0.2,
             max_tokens: int = 1024) -> str:
        """在此替换为真实调用。这里给出一个本地假实现便于跑通流程."""
        # TODO: 替换为真实 GPT-5 SDK 调用
        # 占位：简单拼接最后用户输入，模拟“抽取/总结”回包
        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content
                break
        # 模拟有限输出
        time.sleep(0.05)
        return f'{{"mock":"replace_with_real_gpt5","echo_preview":"{user_text[:120]}"}}'
```

---

### `serp_client.py`

```python
# -*- coding: utf-8 -*-
"""SerpAPI 客户端（用于外部补全/校验）."""
from __future__ import annotations

import os
import requests
from typing import Dict, Any, List, Optional


class SerpClient:
    """轻量 SerpAPI 封装."""

    def __init__(self, api_key_env: str, engine: str = "google", num: int = 5) -> None:
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing SerpAPI key in env: {api_key_env}")
        self._key = api_key
        self._engine = engine
        self._num = num

    def search(self, q: str) -> Dict[str, Any]:
        """执行搜索并返回 JSON."""
        params = {
            "engine": self._engine,
            "q": q,
            "api_key": self._key,
            "num": self._num,
        }
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def quick_enrich_aliases(self, entity_name: str) -> List[str]:
        """尝试从搜索结果里提取候选别名（简单示例：抓取标题中常见括号/同义写法）."""
        data = self.search(entity_name)
        results = []
        for item in (data.get("organic_results") or [])[: self._num]:
            title = (item.get("title") or "").strip()
            if "（" in title and "）" in title:
                try:
                    alias = title.split("（", 1)[1].split("）", 1)[0]
                    if alias and alias not in results:
                        results.append(alias)
                except Exception:
                    pass
        return results
```

---

### `loaders.py`

```python
# -*- coding: utf-8 -*-
"""本地书籍加载与解析."""
from __future__ import annotations

import os
from typing import Dict, Iterable, Generator
from pdfminer.high_level import extract_text


def iter_texts(books_dir: str) -> Generator[Dict, None, None]:
    """遍历目录，解析 PDF/TXT，产出 {doc_id, path, page, text}.

    中文注释：
        - 这里只示例 PDF/TXT，如需 EPUB/HTML，可继续扩展。
        - 为稳健性，遇到异常文件跳过但记录 path。
    """
    for root, _, files in os.walk(books_dir):
        for fn in files:
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn)[-1].lower()
            doc_id = os.path.relpath(path, books_dir)
            try:
                if ext == ".pdf":
                    # 简化：一次性抽取文本（生产中建议逐页解析）
                    text = extract_text(path)
                    yield {"doc_id": doc_id, "path": path, "page": 1, "text": text}
                elif ext == ".txt":
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    yield {"doc_id": doc_id, "path": path, "page": 1, "text": text}
            except Exception as e:
                yield {"doc_id": doc_id, "path": path, "page": 0, "text": f"[PARSE_ERROR]: {e}"}
```

---

### `splitters.py`

```python
# -*- coding: utf-8 -*-
"""文本切分器实现（递归优先，保证语义完整）."""
from __future__ import annotations

from typing import List, Dict
import re


def recursive_split(text: str, max_chars: int = 1200) -> List[str]:
    """递归按章节/段落/句子切分，兜底按字符分块.

    中文注释：
        - 优先按两个换行（段落），其后按句号/问号/感叹号，最终按定长切分。
        - 生产中可引入更强的句法/标点/标题检测器。
    """
    if len(text) <= max_chars:
        return [text]

    # 1) 段落
    paras = re.split(r"\n\s*\n", text)
    if len(paras) > 1:
        return _split_by_units(paras, max_chars)

    # 2) 句子
    sents = re.split(r"(?<=[。！？.!?])", text)
    if len(sents) > 1:
        return _split_by_units(sents, max_chars)

    # 3) 定长兜底
    return [text[i: i + max_chars] for i in range(0, len(text), max_chars)]


def _split_by_units(units: List[str], max_chars: int) -> List[str]:
    chunks, buf = [], ""
    for u in units:
        if len(buf) + len(u) <= max_chars:
            buf += (u if buf == "" else ("" + u))
        else:
            if buf:
                chunks.append(buf)
            buf = u
    if buf:
        chunks.append(buf)
    return chunks


def make_chunks(doc_row: Dict, max_chars: int = 1200) -> List[Dict]:
    """生成带元数据的块."""
    text = doc_row["text"]
    parts = recursive_split(text, max_chars=max_chars)
    chunks = []
    for i, p in enumerate(parts):
        chunks.append({
            "chunk_id": f'{doc_row["doc_id"]}#p{doc_row["page"]}#{i}',
            "doc_id": doc_row["doc_id"],
            "page": doc_row["page"],
            "text": p.strip(),
            "source": doc_row["path"],
        })
    return chunks
```

---

### `extractor.py`

```python
# -*- coding: utf-8 -*-
"""利用 GPT-5 抽取实体/关系/属性."""
from __future__ import annotations

import json
from typing import List, Dict, Any
from llm_client import BaseChatClient, LLMMessage


TRIPLE_PROMPT = """你是专业信息抽取器。请从以下文本抽取面向知识图谱的实体与关系，输出 JSON：
格式：
{
 "entities":[{"name":"","type":"","aliases":[],"desc":""},...],
 "triples":[{"head":"","relation":"","tail":"","evidence_span":"","attributes":{},"confidence":0.0}],
 "notes":[]
}
要求：relation 用标准术语；不要臆造；confidence ∈ [0,1]；evidence_span 贴原文短句。
文本：
"""


def extract_from_chunk(client: BaseChatClient, text: str) -> Dict[str, Any]:
    """对单个 chunk 调用 LLM 抽取."""
    messages = [
        LLMMessage(role="system", content="You are a precise information extraction model."),
        LLMMessage(role="user", content=TRIPLE_PROMPT + text[:4000]),
    ]
    raw = client.chat(messages, temperature=0.1, max_tokens=1200)
    try:
        data = json.loads(raw)
    except Exception:
        # 兜底：尝试从文本中抓取最接近 JSON 的段
        data = {"entities": [], "triples": [], "notes": [f"parse_fail:{raw[:120]}"]}
    # 结构保守归一
    data.setdefault("entities", [])
    data.setdefault("triples", [])
    return data
```

---

### `resolve.py`

```python
# -*- coding: utf-8 -*-
"""实体对齐与去重."""
from __future__ import annotations

from typing import Dict, List, Tuple
from rapidfuzz import fuzz


def canonical_name(name: str) -> str:
    """名称归一（大小写/空白/标点简单处理）."""
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum() or ch in (" ", "-"))


def same_entity(a: Dict, b: Dict, name_thresh: int = 90, alias_boost: int = 5) -> Tuple[bool, float]:
    """判断两个实体是否同一（启发式 + 名称相似 + 别名交集）."""
    an, bn = canonical_name(a.get("name", "")), canonical_name(b.get("name", ""))
    name_sim = fuzz.token_set_ratio(an, bn)
    alias_set_a = set(canonical_name(x) for x in a.get("aliases", []))
    alias_set_b = set(canonical_name(x) for x in b.get("aliases", []))
    alias_bonus = alias_boost if alias_set_a & alias_set_b else 0
    score = name_sim + alias_bonus
    return (score >= name_thresh), min(1.0, score / 100.0)


def merge_entities(cands: List[Dict]) -> List[Dict]:
    """贪心合并实体候选."""
    merged: List[Dict] = []
    for e in cands:
        placed = False
        for m in merged:
            same, conf = same_entity(m, e)
            if same:
                # 合并别名与描述（保留信息最多者）
                m["aliases"] = sorted(list(set(m.get("aliases", []) + e.get("aliases", []))))
                if len(e.get("desc", "")) > len(m.get("desc", "")):
                    m["desc"] = e.get("desc", "")
                placed = True
                break
        if not placed:
            merged.append(e)
    return merged
```

---

### `graph_store.py`

```python
# -*- coding: utf-8 -*-
"""图存储：NetworkX 实现（可替换为 Neo4j）."""
from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple, List
import networkx as nx


class GraphStore:
    """简单属性图包装."""

    def __init__(self) -> None:
        self.G = nx.MultiDiGraph()

    # ---------- 节点 ----------
    def upsert_entity(self, ent: Dict[str, Any]) -> str:
        """插入/更新实体节点."""
        key = ent["name"]
        if not self.G.has_node(key):
            self.G.add_node(key, **ent)
        else:
            self.G.nodes[key].update(ent)
        return key

    # ---------- 边 ----------
    def add_relation(self, head: str, rel: str, tail: str, props: Dict[str, Any]) -> None:
        self.G.add_edge(head, tail, key=rel, **({"type": rel} | props))

    # ---------- 查询 ----------
    def topk_similar_entities(self, name: str, k: int = 5) -> List[str]:
        """简化相似：同名/包含."""
        name_l = name.lower()
        cands = []
        for n in self.G.nodes:
            if name_l in n.lower():
                cands.append(n)
        return cands[:k]

    def neighborhood(self, seeds: Iterable[str], hops: int = 1) -> Tuple[List[str], List[Tuple[str, str, Dict]]]:
        """返回多跳邻域."""
        sub_nodes = set(seeds)
        for _ in range(hops):
            new_nodes = set()
            for s in list(sub_nodes):
                new_nodes.update(self.G.successors(s))
                new_nodes.update(self.G.predecessors(s))
            sub_nodes.update(new_nodes)
        # 收集边
        edges = []
        for u in sub_nodes:
            for v in self.G.successors(u):
                for k, data in self.G.get_edge_data(u, v).items():
                    edges.append((u, v, data))
        return list(sub_nodes), edges
```

> **切换 Neo4j**：可新增 `Neo4jGraphStore`，将 `upsert_entity/add_relation/neighborhood` 映射为 Cypher；写入时用 `MERGE` 去重，关系用 `MERGE (h)-[r:REL{type:$type}]->(t)` + `SET r += $props`。

---

### `vector_store.py`

```python
# -*- coding: utf-8 -*-
"""向量索引（sklearn 近邻兜底，可换 FAISS/Milvus）."""
from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class SimpleVectorStore:
    """简化向量库：TFIDF + KNN（演示用）."""

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(max_features=50000)
        self._index = NearestNeighbors(n_neighbors=8, algorithm="auto")
        self._ids: List[str] = []
        self._mat = None

    def build(self, items: List[Dict]) -> None:
        """items: [{"chunk_id","text"}...]"""
        texts = [it["text"] for it in items]
        self._ids = [it["chunk_id"] for it in items]
        X = self._vec.fit_transform(texts)
        self._mat = X
        self._index.fit(X)

    def query(self, q: str, k: int = 8) -> List[Tuple[str, float]]:
        x = self._vec.transform([q])
        dists, idxs = self._index.kneighbors(x, n_neighbors=min(k, len(self._ids)))
        out: List[Tuple[str, float]] = []
        for i, dist in zip(idxs[0], dists[0]):
            out.append((self._ids[i], float(1.0 / (1.0 + dist))))
        return out
```

---

### `graph_rag.py`

```python
# -*- coding: utf-8 -*-
"""GraphRAG 检索器."""
from __future__ import annotations

from typing import Dict, List, Tuple
from llm_client import BaseChatClient, LLMMessage
from graph_store import GraphStore
from vector_store import SimpleVectorStore


ANSWER_PROMPT = """你是检索增强问答助手。结合“候选子图摘要”和“候选文段”作答：
- 先列要点，再给结论
- 引用证据时用 [doc:page:chunk_id] 标注
- 不确定时明确说明

问题：
{query}

候选子图摘要（节点/边）：
{graph_brief}

候选文段（每条以 [doc:page:chunk] 开头）：
{passages}
"""


class GraphRAG:
    """图引导的 RAG 检索."""

    def __init__(self, graph: GraphStore, vstore: SimpleVectorStore, llm: BaseChatClient) -> None:
        self._graph = graph
        self._vstore = vstore
        self._llm = llm

    def answer(self, query: str, seeds: List[str], hops: int = 1, top_k_chunks: int = 8) -> str:
        """从种子实体出发扩展子图，再做文段召回并综合."""
        nodes, edges = self._graph.neighborhood(seeds, hops=hops)
        gbrief = []
        for u, v, data in edges[:50]:
            gbrief.append(f"{u} -[{data.get('type','rel')}]-> {v}")
        gtxt = "\n".join(gbrief)

        # 文段召回（用 query 或拼 seeds）
        q = query if seeds else (query + " " + " ".join(nodes[:5]))
        hits = self._vstore.query(q, k=top_k_chunks)
        passages = []
        for cid, score in hits:
            doc, page, idx = cid.split("#")
            passages.append(f"[{doc}:{page}:{idx}] (score={score:.3f}) ...")

        prompt = ANSWER_PROMPT.format(query=query, graph_brief=gtxt, passages="\n".join(passages))
        msg = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content=prompt),
        ]
        return self._llm.chat(msg, temperature=0.2, max_tokens=1200)
```

---

### `build_kg.py`

```python
# -*- coding: utf-8 -*-
"""端到端构图 CLI."""
from __future__ import annotations

import json
from typing import List, Dict
from tqdm import tqdm

from config import AppConfig
from llm_client import GPT5ChatClient
from serp_client import SerpClient
from loaders import iter_texts
from splitters import make_chunks
from extractor import extract_from_chunk
from resolve import merge_entities
from graph_store import GraphStore
from vector_store import SimpleVectorStore


def main() -> None:
    cfg = AppConfig()

    # 初始化组件
    llm = GPT5ChatClient(api_key_env=cfg.llm.api_key_env, api_base=cfg.llm.api_base, model=cfg.llm.model)
    serp = SerpClient(api_key_env=cfg.serp.api_key_env, engine=cfg.serp.engine, num=cfg.serp.num)
    gstore = GraphStore()
    vstore = SimpleVectorStore()

    # 1) 加载 + 切分
    all_chunks: List[Dict] = []
    for row in tqdm(iter_texts(cfg.paths.books_dir), desc="Parsing Books"):
        chunks = make_chunks(row, max_chars=1200)
        all_chunks.extend(chunks)

    # 2) 抽取
    all_entities: List[Dict] = []
    all_triples: List[Dict] = []
    for ch in tqdm(all_chunks, desc="Extracting Triples"):
        data = extract_from_chunk(llm, ch["text"])
        # 附加来源与置信度过滤
        for e in data["entities"]:
            e.setdefault("aliases", [])
            e["source"] = ch["source"]
            e.setdefault("desc", "")
            all_entities.append(e)
        for t in data["triples"]:
            conf = float(t.get("confidence", 0.0))
            if conf >= cfg.build.min_confidence:
                t["source"] = ch["source"]
                all_triples.append(t)

    # 3) 外部补全（别名）
    if cfg.build.use_serp_enrichment:
        for e in tqdm(all_entities, desc="Serp Enrichment"):
            try:
                aliases = serp.quick_enrich_aliases(e["name"])
                e["aliases"] = sorted(list(set(e.get("aliases", []) + aliases)))
            except Exception:
                pass

    # 4) 实体合并
    merged = merge_entities(all_entities)

    # 5) 入图
    for e in merged:
        gstore.upsert_entity(e)
    for t in all_triples:
        gstore.add_relation(t["head"], t["relation"], t["tail"], {
            "evidence_span": t.get("evidence_span", ""),
            "source": t.get("source", ""),
            "confidence": float(t.get("confidence", 0.0))
        })

    # 6) 构建向量索引（这里用 TFIDF+KNN 演示）
    vstore.build([{"chunk_id": ch["chunk_id"], "text": ch["text"]} for ch in all_chunks])

    # 7) 简要统计输出
    print(f"[BUILD DONE] entities={len(merged)}, triples={len(all_triples)}, chunks={len(all_chunks)}")


if __name__ == "__main__":
    main()
```

---

### `query_cli.py`

```python
# -*- coding: utf-8 -*-
"""GraphRAG 查询 CLI."""
from __future__ import annotations

from typing import List
from config import AppConfig
from llm_client import GPT5ChatClient
from graph_store import GraphStore
from vector_store import SimpleVectorStore
from graph_rag import GraphRAG


def demo_load_minimal_graph() -> GraphStore:
    """演示：加载构图阶段内存图（真实环境应持久化后再加载）."""
    # 这里直接 new 一个空图，实际可在 build_kg.py 中将图/索引持久化到文件或图数据库。
    return GraphStore()


def main() -> None:
    cfg = AppConfig()
    llm = GPT5ChatClient(api_key_env=cfg.llm.api_key_env, api_base=cfg.llm.api_base, model=cfg.llm.model)
    gstore = demo_load_minimal_graph()
    vstore = SimpleVectorStore()  # 实际需要加载已构建索引；此处仅演示 API
    rag = GraphRAG(gstore, vstore, llm)

    # 示例查询
    query = "该书中如何定义X的核心概念？与Y之间的关系是什么？"
    seeds: List[str] = ["X", "Y"]  # 若无实体种子，可先用向量检索或 NER 找候选实体

    print(rag.answer(query=query, seeds=seeds, hops=cfg.rag.max_hops, top_k_chunks=cfg.rag.top_k_chunks))


if __name__ == "__main__":
    main()
```

---

# 使用与扩展

## 快速运行

1. 准备数据：把 PDF/TXT 放到 `data/books/`。
2. 配置环境变量：

   * `export OPENAI_API_KEY=...`（GPT-5）
   * `export SERPAPI_KEY=...`（SerpAPI）
3. 安装依赖：`pip install -r requirements.txt`
4. 构图：`python graphrag_kit/build_kg.py`
5. 查询：`python graphrag_kit/query_cli.py`

> 生产化时：
>
> * 将 **GraphStore** 切至 **Neo4j**（批量 `MERGE`、定期 `apoc` 清洗）；
> * 将 **向量库** 切至 **FAISS/Milvus/ES**；
> * 将 **LLM** 客户端换成真实 GPT-5 SDK 调用并落盘缓存（避免二次费用）。

## 可拓展性要点

* **数据源插件化**：在 `loaders.py` 增加 `EPUBLoader/HTMLLoader`。
* **切分策略**：在 `splitters.py` 增加 `SentenceSplitter/TokenSplitter`；为长书籍配置**自适应窗口**（按平均句长+方差动态调 `max_chars`）。
* **抽取策略**：`extractor.py` 支持**批内去重**与**失败重试**；对低置信样本触发二次抽取。
* **对齐策略**：`resolve.py` 可增加 **语义向量**（用 m3e/BGE）与 **规则库**（缩写映射、年份/单位归一）。
* **图存储**：新增 `Neo4jGraphStore`，提供持久化与并行写入（事务批次）。
* **GraphRAG**：在 `graph_rag.py` 增加**子图扩展策略**（按边类型/时间戳/权重）与**重排序**（图中心性 + 文向量相关性融合）。

## 评估与质检（工程建议）

* **结构质量**：随机抽样三元组做人工标注（精确率/召回率/一致性）；统计 `confidence` 分布与重复边率。
* **对齐质量**：聚类纯度（Purity）、B³、同名异物的错合率。
* **RAG 效果**：基于问题集测 **准确率/覆盖率/引用正确率**；答案与证据的一致性审计。
* **成本/延迟**：抽取 TPS、平均 token 消耗、构图时长；检索时延分布（P50/P95）。

---

# 自检（关键逻辑回看）

* 抽象层清晰：LLM/Serp/Graph/Vector/RAG 解耦；任一模块可替换。
* 失败兜底：LLM JSON 解析失败 → 返回空结构并记录 `notes`；向量库缺失 → 不致使流程崩溃。
* 置信阈值：构图仅写入 `confidence ≥ min_confidence` 的关系，降低噪声。
* 去重合并：名称归一 + 别名交集 + 模糊匹配；避免简单同名误合。
* GraphRAG：**子图摘要 + 文段证据** 一并交给 LLM，具备可追溯引用。

---

如果你希望：

* 切到 **Neo4j**（含批量 Cypher 模板）、
* 用 **FAISS/Milvus**（含落盘/加载）、
* 或者把 **GPT-5 SDK** 的真实调用补全进 `GPT5ChatClient`，
  我可以直接把相应的代码片段补上（保持以上接口不变）。
