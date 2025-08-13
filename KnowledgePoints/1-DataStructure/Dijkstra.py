from __future__ import annotations
from typing import Dict, Hashable, Iterable, List, Tuple, Optional
import heapq


class Graph:
    """
    简单加权图（默认无向，可选有向），使用邻接表存储：adj[u] = [(v, w), ...]
    - 顶点类型用 Hashable，方便用字符串或数字当作节点ID
    - 边权必须非负，否则 Dijkstra 不适用
    """

    def __init__(self, directed: bool = False):
        self.adj: Dict[Hashable, List[Tuple[Hashable, float]]] = {}
        self.directed = directed

    def add_edge(self, u: Hashable, v: Hashable, w: float) -> None:
        """添加一条边 u->v，权重 w（若无向图则同时添加 v->u）。"""
        if w < 0:
            raise ValueError("Dijkstra 要求非负边权，检测到负权边。")
        self.adj.setdefault(u, []).append((v, w))
        self.adj.setdefault(v, [])  # 确保 v 也在图里
        if not self.directed:
            self.adj.setdefault(v, []).append((u, w))
            self.adj.setdefault(u, [])  # 确保 u 在图里

    def nodes(self) -> Iterable[Hashable]:
        return self.adj.keys()


def dijkstra_heap(
    graph: Graph, source: Hashable
) -> Tuple[Dict[Hashable, float], Dict[Hashable, Optional[Hashable]]]:
    """
    堆优化 Dijkstra：O((V+E)logV)
    返回:
      dist: 最短距离字典（不可达为 float('inf')）
      prev: 前驱字典（用于路径重建；source 的前驱为 None）
    """
    dist: Dict[Hashable, float] = {u: float("inf") for u in graph.nodes()}
    prev: Dict[Hashable, Optional[Hashable]] = {u: None for u in graph.nodes()}
    dist[source] = 0.0

    # 小根堆：元素是 (当前距离, 节点)
    pq: List[Tuple[float, Hashable]] = [(0.0, source)]

    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        # 懒删除：如果弹出的距离不是当前最优，跳过
        if d != dist[u]:
            continue
        if u in visited:  # 可选：visited 不是必须（上面的懒删除已足够）
            continue
        visited.add(u)

        for v, w in graph.adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev: Dict[Hashable, Optional[Hashable]], target: Hashable) -> List[Hashable]:
    """根据前驱表重建从源点到 target 的路径。若不可达，返回空列表。"""
    if prev.get(target) is None and target not in prev:
        # 目标不在图中
        return []
    path: List[Hashable] = []
    cur: Optional[Hashable] = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    # 若路径长度为1 且 前驱为 None，可能是源点或不可达（区分方式：依赖 dist[target]）
    return path


# --------- 可选：密集图朴素版（邻接矩阵或需要 O(V^2) 时）---------
def dijkstra_naive(
    graph: Graph, source: Hashable
) -> Tuple[Dict[Hashable, float], Dict[Hashable, Optional[Hashable]]]:
    """
    O(V^2) 朴素版：每次线性扫描找到未确定的最小 dist 顶点。
    适合小图或非常稠密图（且没有现成堆结构时）。
    """
    nodes = list(graph.nodes())
    dist = {u: float("inf") for u in nodes}
    prev: Dict[Hashable, Optional[Hashable]] = {u: None for u in nodes}
    dist[source] = 0.0
    used = {u: False for u in nodes}

    for _ in range(len(nodes)):
        # 选未确定中 dist 最小的顶点
        u = min((x for x in nodes if not used[x]), key=lambda x: dist[x], default=None)
        if u is None or dist[u] == float("inf"):
            break
        used[u] = True
        for v, w in graph.adj[u]:
            if not used[v] and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
    return dist, prev


# ----------------------- Demo：地铁最短时间 -----------------------
if __name__ == "__main__":
    g = Graph(directed=False)
    # 示例网络（与我们课堂推导一致）
    edges = [
        ("S", "A", 4),
        ("S", "B", 2),
        ("A", "B", 1),
        ("A", "C", 5),
        ("B", "C", 8),
        ("B", "D", 10),
        ("C", "D", 2),
        ("C", "E", 6),
        ("D", "E", 3),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    dist, prev = dijkstra_heap(g, "S")
    # 打印最短距离
    print("Shortest Times from S:")
    for node in sorted(g.nodes()):
        print(f"  S -> {node}: {dist[node]}")

    # 打印到每个点的一条最短路径
    print("\nPaths:")
    for node in sorted(g.nodes()):
        path = reconstruct_path(prev, node)
        if dist[node] == float("inf"):
            print(f"  S -> {node}: unreachable")
        else:
            print(f"  S -> {node}: {' -> '.join(path)} (time={dist[node]})")
