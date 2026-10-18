"""局部采样点修复：严格复核几何误差和所有声明关系，单调加点保证终止。"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .data import Recording
from .geometry import Block,reconstruct
from .joint import greedy_cover
from .refinement_query import RefinementQuery


def state_case(valid: list[bool]) -> str:
    """顺序 00/10/01/11，当前错误关系的四种精确端点状态。"""
    if len(valid)!=4 or not valid[3]:
        raise ValueError("Exact/exact must reproduce the original relationship")
    if valid[0]:
        return "none"
    if valid[1] and valid[2]:
        return "cover"
    if valid[1]:
        return "u"
    if valid[2]:
        return "v"
    return "both"


def refine(rec: Recording,blocks: list[Block],coarse: list[np.ndarray],radii: list[float],
           epsilon: float,truth: dict,max_iterations: int,deadline: float,
           mode: str,scratch_dir: Path | None = None) -> tuple[list[np.ndarray],dict]:
    if mode not in {"POINT-REPAIR","ENDPOINT-REPAIR","LOCAL-JOINT"}:
        raise ValueError("Unknown local refinement mode")
    if epsilon<0 or not np.isfinite(epsilon) or max_iterations<0:
        raise ValueError("Invalid refinement epsilon/iteration budget")
    rec.validate()
    knots=[k.copy() for k in coarse]
    owner=np.empty(len(rec.time),dtype=np.int64)
    position=np.empty(len(rec.time),dtype=np.int64)
    for n,b in enumerate(blocks):
        owner[b.rows]=n
        position[b.rows]=np.arange(len(b.rows))
    lookup={(int(t),int(v)):i for i,(t,v) in enumerate(zip(rec.time,rec.track))}
    retained={int(row) for b,k in zip(blocks,knots,strict=True) for row in b.rows[k]}
    initial=len(retained)
    geom_total,relation_total=0,0
    forced_total,heuristic_total=0,0
    solve_seconds=0.0
    history=[]
    query=RefinementQuery(rec,radii,epsilon)
    for iteration in range(max_iterations+1):
        if time.perf_counter()>deadline:
            raise TimeoutError("Local refinement wall-time budget exceeded")
        if scratch_dir is not None:
            from .codecs import encode,decode
            path=scratch_dir/f"{mode}_iteration_{iteration}.npz"
            encode(path,rec,blocks,knots,radii)
            rebuilt,_,_=decode(path)
            path.unlink()
        else:
            rebuilt=Recording(rec.metadata,rec.tracks,rec.track,rec.time,reconstruct(rec,blocks,knots),rec.dt)
        error=np.linalg.norm(rec.xy-rebuilt.xy,axis=1)
        bad_geometry=set(map(int,np.flatnonzero(error>epsilon)))
        actual=query.relations(rebuilt)
        flips=set().union(*(truth[float(r)] ^ actual[float(r)] for r in radii))
        item=dict(iteration=iteration,max_error_m=float(error.max()),
                  geometric_violations=len(bad_geometry),flipped_pair_times=len(flips),
                  retained_points=len(retained))
        history.append(item)
        if not bad_geometry and not flips:
            return knots,dict(local_iterations=iteration,local_added_points=len(retained)-initial,
                              geometry_repair_points=geom_total,relation_repair_points=relation_total,
                              local_forced_points=forced_total,local_heuristic_points=heuristic_total,
                              solve_s=solve_seconds,refinement_history=history)
        if iteration==max_iterations:
            raise TimeoutError("Local refinement iteration budget exceeded")
        if retained.intersection(bad_geometry):
            raise RuntimeError("Retained exact point violates geometry: implementation error")
        selected=set(bad_geometry)
        edges=set()
        forced=set(bad_geometry)
        cases={k:0 for k in ["none","cover","u","v","both"]}
        for t,i,j in sorted(flips):
            u,v=lookup[(t,i)],lookup[(t,j)]
            if u in retained and v in retained:
                raise RuntimeError("Two exact sample points still flip: implementation error")
            if mode=="POINT-REPAIR":
                selected.update((u,v))
                continue
            # 四个状态只在当前时刻计算；加点对其他时刻的影响必须下一轮重新查。
            delta=np.vstack((rebuilt.xy[u]-rebuilt.xy[v],rec.xy[u]-rebuilt.xy[v],
                             rebuilt.xy[u]-rec.xy[v],rec.xy[u]-rec.xy[v]))
            distance=np.linalg.norm(delta,axis=1)
            labels=distance[:,None]<=np.asarray(radii)[None,:]
            gold=np.array([(t,i,j) in truth[float(r)] for r in radii])
            case=state_case(np.all(labels==gold[None,:],axis=1).tolist())
            cases[case]+=1
            if case=="none":
                raise RuntimeError("Actual flip disagrees with state enumeration")
            if case=="cover":
                edges.add(tuple(sorted((u,v))))
            elif case=="u":
                forced.add(u)
            elif case=="v":
                forced.add(v)
            else:
                forced.update((u,v))
        if mode in {"ENDPOINT-REPAIR","LOCAL-JOINT"}:
            item.update(state_cases=cases,merged_cover_edges=len(edges),forced_sample_points=len(forced))
            nodes=forced|{v for e in edges for v in e}
            costs={v:0 if v in retained else 24 for v in nodes}
            start=time.perf_counter()
            if mode=="LOCAL-JOINT":
                selected,stats=greedy_cover(costs,edges,forced)
            else:
                # 强简单对照：按错误边顺序先选误差较大端点，跳过已覆盖边。
                selected=set(forced)
                for u,v in sorted(edges):
                    if not selected.intersection((u,v)):
                        selected.add(max((u,v),key=lambda node:(error[node],-node)))
                for node in sorted(selected-forced,reverse=True):
                    reduced=selected-{node}
                    if all(reduced.intersection(edge) for edge in edges):
                        selected=reduced
                if not forced.issubset(selected) or not all(selected.intersection(edge) for edge in edges):
                    raise RuntimeError("Endpoint repair produced infeasible selections")
            solve_seconds+=time.perf_counter()-start
            forced_total+=len(forced-retained-bad_geometry)
            heuristic_total+=len(selected-forced-retained)
        new=selected-retained
        if not new:
            raise RuntimeError("Refinement made no progress despite violations")
        geometry_new=new & bad_geometry
        geom_total+=len(geometry_new)
        relation_total+=len(new-geometry_new)
        item.update(new_geometry_points=len(geometry_new),new_relation_points=len(new-geometry_new))
        additions={}
        for v in new:
            additions.setdefault(int(owner[v]),[]).append(int(position[v]))
        for n,points in additions.items():
            knots[n]=np.union1d(knots[n],points).astype(np.int64)
        retained.update(new)
