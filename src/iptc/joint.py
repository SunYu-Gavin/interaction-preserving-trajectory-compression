"""两档块精度分配：强制升级 + 代价感知顶点覆盖。"""
from __future__ import annotations
import itertools
from collections import defaultdict
import numpy as np


def exact_cover(cost: dict[int,int], edges: set[tuple[int,int]], forced: set[int] | None = None) -> tuple[set[int],int]:
    forced = set(forced or [])
    residual = {e for e in edges if not forced.intersection(e)}
    vertices = sorted({v for e in residual for v in e})
    if len(vertices) > 20:
        raise ValueError("Exhaustive solver is limited to 20 active vertices")
    best,best_cost = set(vertices),sum(cost[v] for v in vertices)
    for bits in itertools.product((0,1),repeat=len(vertices)):
        selected = {v for v,b in zip(vertices,bits) if b}
        value = sum(cost[v] for v in selected)
        if value < best_cost and all(selected.intersection(e) for e in residual):
            best,best_cost = selected,value
    return best|forced,best_cost+sum(cost[v] for v in forced)


def greedy_cover(cost: dict[int,int], edges: set[tuple[int,int]], forced: set[int] | None = None) -> tuple[set[int],dict]:
    if any(v < 0 for v in cost.values()) or any(v not in cost for e in edges for v in e):
        raise ValueError("Coverage costs must exist and be nonnegative")
    forced = set(forced or [])
    selected = set(forced)
    remaining = {e for e in edges if not selected.intersection(e)}
    while remaining:
        degree = defaultdict(int)
        for u,v in remaining:
            degree[u] += 1
            if u != v:
                degree[v] += 1
        vertex = min(degree,key=lambda v:(cost[v]/degree[v],cost[v],v))
        selected.add(vertex)
        remaining = {e for e in remaining if vertex not in e}
    # 逆代价顺序删除冗余节点，同时保持覆盖约束可行。
    for v in sorted(selected-forced,key=lambda v:(-cost[v],v)):
        reduced = selected-{v}
        if all(reduced.intersection(e) for e in edges):
            selected = reduced
    if not all(selected.intersection(e) for e in edges) or not forced.issubset(selected):
        raise RuntimeError("Cover solver returned an infeasible solution")
    return selected,dict(forced_upgrades=len(forced),heuristic_upgrades=len(selected-forced),
                         logical_upgrade_bytes=sum(cost[v] for v in selected))


def small_component_gaps(cost: dict[int,int], edges: set[tuple[int,int]], forced: set[int],
                         max_nodes: int = 16, max_components: int = 20) -> list[dict]:
    residual = {e for e in edges if not forced.intersection(e)}
    adj = defaultdict(set)
    for u,v in residual:
        adj[u].add(v)
        adj[v].add(u)
    seen,result = set(),[]
    for start in sorted(adj):
        if start in seen:
            continue
        todo,component = [start],set()
        while todo:
            v = todo.pop()
            if v in component:
                continue
            component.add(v)
            todo.extend(adj[v]-component)
        seen.update(component)
        if len(component) > max_nodes:
            continue
        subedges = {e for e in residual if e[0] in component}
        exact,optimum = exact_cover(cost,subedges)
        greedy,stats = greedy_cover(cost,subedges)
        value = stats["logical_upgrade_bytes"]
        result.append(dict(nodes=len(component),edges=len(subedges),exact_bytes=optimum,
                           heuristic_bytes=value,absolute_gap=value-optimum,
                           ratio=value/optimum if optimum else (1 if value==0 else None)))
        if len(result) >= max_components:
            break
    return result
