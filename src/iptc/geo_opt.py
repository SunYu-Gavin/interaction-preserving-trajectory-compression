"""强几何对照：全原始候选点上的同步误差可达图与最少点路径。"""
from __future__ import annotations
import numpy as np
from .data import Recording
from .geometry import Block,interpolate
from .prune import shortest_subsequence


def segment_errors(rec: Recording, blocks: list[Block]) -> list[np.ndarray]:
    from .resources import check
    graphs=[]
    for block in blocks:
        check();rows=block.rows;t=rec.time[rows];xy=rec.xy[rows];size=len(rows)
        error=np.full((size,size),np.inf,dtype=np.float64)
        for left in range(size):
            error[left,left]=0
            if left+1<size:error[left,left+1]=0
            for right in range(left+2,size):
                candidate=interpolate(t[left:right+1],t[[left,right]],xy[[left,right]])
                error[left,right]=np.linalg.norm(candidate-xy[left:right+1],axis=1).max()
        graphs.append(error)
    return graphs


def optimal_knots(graphs: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    if epsilon<0 or not np.isfinite(epsilon):raise ValueError('Invalid geometric error')
    return [shortest_subsequence(np.arange(len(graph)),lambda a,b:graph[a,b]<=epsilon) for graph in graphs]
