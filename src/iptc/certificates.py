"""证书失败和实际翻转分开计数；证书只给充分条件。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .data import Recording
from .geometry import Block
from .events import frames, spatial_pairs


def constraint_case(eu: float, ev: float, margin: float, guard: float = 0.0) -> str:
    if min(eu,ev,margin,guard) < 0 or not np.isfinite([eu,ev,margin,guard]).all():
        raise ValueError("Errors, margin and guard must be finite and nonnegative")
    if margin <= guard:
        return "both"
    if eu+ev+guard < margin:
        return "none"
    u_ok = ev+guard < margin
    v_ok = eu+guard < margin
    if u_ok and v_ok:
        return "cover"
    if u_ok:
        return "u"
    if v_ok:
        return "v"
    return "both"


@dataclass
class Constraints:
    forced: set[int]
    edges: set[tuple[int,int]]
    uncertified: int
    candidate_pairs: int
    cases: dict[str,int]


def build_constraints(rec: Recording, coarse: np.ndarray, blocks: list[Block],
                      radii: list[float], epsilon: float) -> Constraints:
    errors = np.linalg.norm(coarse-rec.xy,axis=1)
    if errors.max() > epsilon:
        raise ValueError("Coarse reconstruction violates epsilon")
    nodes = np.empty(len(rec.time),dtype=np.int64)
    for n,b in enumerate(blocks):
        nodes[b.rows] = n
    forced,edges = set(),set()
    count,total = 0,0
    cases = {k:0 for k in ["none","cover","u","v","both"]}
    # 若原始距离 > r_max+2epsilon，重建距离仍 > r_max。
    # 数值防护加在候选半径和证书侧，不改变查询阈值。
    max_coord = float(np.abs(rec.xy).max())
    global_guard = 64*np.finfo(np.float64).eps*(1+2*max_coord+max(radii))
    radius = max(radii)+2*epsilon+global_guard
    for _,rows in frames(rec):
        pairs = spatial_pairs(rec.xy[rows],radius)
        total += len(pairs)
        if not len(pairs):
            continue
        distance = np.linalg.norm(rec.xy[rows[pairs[:,0]]]-rec.xy[rows[pairs[:,1]]],axis=1)
        for p,(i,j) in enumerate(pairs):
            a,b = int(rows[i]),int(rows[j])
            u,v = sorted((int(nodes[a]),int(nodes[b])))
            # u/v 标签要和对应误差对齐。
            eu,ev = (errors[a],errors[b]) if nodes[a] == u else (errors[b],errors[a])
            for r in radii:
                case = constraint_case(float(eu),float(ev),abs(float(distance[p])-r),global_guard)
                cases[case] += 1
                if case == "none":
                    continue
                count += 1
                if case == "cover":
                    edges.add((u,v))
                elif case == "u":
                    forced.add(u)
                elif case == "v":
                    forced.add(v)
                else:
                    forced.update((u,v))
    return Constraints(forced,edges,count,total,cases)
