"""强独立保真对照：依据每时刻最近查询边界分配保守误差预算。"""
from __future__ import annotations
import numpy as np
from .data import Recording
from .geometry import Block
from .prune import shortcut_xy
from .refinement_query import RefinementQuery


def margin_budgets(rec: Recording,radii: list[float],max_epsilon: float) -> np.ndarray:
    if max_epsilon<=0 or not np.isfinite(max_epsilon):raise ValueError('Positive finite maximum epsilon required')
    plan=RefinementQuery(rec,radii,max_epsilon);plan.relations(rec);pairs=plan.rows
    budgets=np.full(len(rec.time),max_epsilon,dtype=np.float64)
    if len(pairs):
        distances=np.linalg.norm(rec.xy[pairs[:,0]]-rec.xy[pairs[:,1]],axis=1)
        margins=np.min(np.abs(distances[:,None]-np.asarray(radii)[None,:]),axis=1)
        # 只收紧证书，不改变评价的 radius 或 epsilon。近边界原样兜底。
        scale=np.maximum(1,np.max(np.abs(rec.xy[pairs]),axis=(1,2))+distances)
        guard=64*np.finfo(np.float64).eps*scale
        allowed=np.nextafter(np.maximum(0,(margins-guard)/2),0)
        np.minimum.at(budgets,pairs[:,0],allowed);np.minimum.at(budgets,pairs[:,1],allowed)
    return budgets


def margin_graphs(rec: Recording,blocks: list[Block],radii: list[float],max_epsilon: float) -> tuple[list[np.ndarray],np.ndarray]:
    from .resources import check
    budgets=margin_budgets(rec,radii,max_epsilon);graphs=[]
    for block in blocks:
        check();rows=block.rows;t=rec.time[rows];xy=rec.xy[rows];local_budget=budgets[rows];size=len(rows)
        graph=np.full((size,size),np.inf,dtype=np.float64)
        np.fill_diagonal(graph,0)
        for left in range(size):
            if left+1<size:graph[left,left+1]=0
            for right in range(left+2,size):
                error=np.linalg.norm(shortcut_xy(t,xy,left,right)-xy[left:right+1],axis=1)
                if np.all(error<=local_budget[left:right+1]):graph[left,right]=float(error.max())
        graphs.append(graph)
    return graphs,budgets
