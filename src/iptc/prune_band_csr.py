"""固定其他车辆时，条件可行捷径图上的保留点回收。"""
from __future__ import annotations
import time
from collections.abc import Callable
from pathlib import Path
import numpy as np
from .data import Recording
from .geometry import Block, interpolate, reconstruct
from .refinement_query import RefinementQuery
from .danger_band import active_predicates


def shortcut_xy(times: np.ndarray, xy: np.ndarray, left: int, right: int) -> np.ndarray:
    """内部快路径：调用处已验证递增时间和块边界，数值运算与解码插值相同。"""
    anchor_t=times[[left,right]];anchor_xy=xy[[left,right]]
    result=np.column_stack([np.interp(times[left:right+1],anchor_t,anchor_xy[:,d]) for d in range(2)])
    result[0]=anchor_xy[0];result[-1]=anchor_xy[1]
    return result


def shortest_subsequence(knots: np.ndarray, feasible: Callable[[int,int],bool]) -> np.ndarray:
    """原候选保留点构成 DAG；相邻原边已可行，返回最少顶点子路径。"""
    if len(knots)<2:
        return knots.copy()
    size=len(knots)
    cost=np.arange(1,size+1,dtype=np.int64)
    previous=np.arange(-1,size-1,dtype=np.int64)
    for j in range(2,size):
        cost[j]=cost[j-1]+1;previous[j]=j-1
        # i 从早到晚，代价相等时保留已确定路径，确定性破平局。
        for i in range(j-1):
            if cost[i]+1<cost[j] and feasible(int(knots[i]),int(knots[j])):
                cost[j]=cost[i]+1;previous[j]=i
                if cost[j]==2:
                    break
    path=[];j=size-1
    while j>=0:
        path.append(int(knots[j]));j=int(previous[j])
    return np.array(path[::-1],dtype=np.int64)


def greedy_subsequence(knots: np.ndarray, feasible: Callable[[int,int],bool]) -> np.ndarray:
    """强简单回收对照：逐点删除，重复直到没有单点可删。"""
    kept=list(map(int,knots));changed=True
    while changed:
        changed=False;i=1
        while i<len(kept)-1:
            if feasible(kept[i-1],kept[i+1]):
                kept.pop(i);changed=True
            else:
                i+=1
    return np.array(kept,dtype=np.int64)


def shortest_feasible_path(size: int, feasible: Callable[[int,int],bool]) -> np.ndarray:
    """全原始候选图；单独恢复一个原始点未必关系可行，故不假定相邻边存在。"""
    if size<1:raise ValueError('Nonempty candidate graph required')
    cost=np.full(size,size+1,dtype=np.int64);cost[0]=1
    previous=np.full(size,-1,dtype=np.int64)
    for j in range(1,size):
        for i in range(j):
            if cost[i]+1<cost[j] and feasible(i,j):
                cost[j]=cost[i]+1;previous[j]=i
                if cost[j]==2:break
    if cost[-1]>size:raise ValueError('No feasible path through original candidate graph')
    path=[];j=size-1
    while j>=0:
        path.append(j);j=int(previous[j])
    return np.array(path[::-1],dtype=np.int64)


def _prune_csr(rec: Recording, blocks: list[Block], initial: list[np.ndarray], radii: list[float],
          epsilon: float, truth: dict, mode: str, max_sweeps: int, deadline: float,
          scratch_dir: Path, geometric_lower_counts: list[int] | None = None,
          active_block_schedule: bool = False,
          candidate_pools: list[np.ndarray] | None = None,
          danger_band: bool = True) -> tuple[list[np.ndarray],dict]:
    if mode not in {'GREEDY-PRUNE','DAG-PRUNE','FULL-DAG','SEED-DAG'} or max_sweeps<1:
        raise ValueError('Pruning requires known mode and positive sweep budget')
    from .codecs import encode,decode
    from .events import relations
    from .resources import check
    wall_start=time.perf_counter()
    profile=dict(seed_validation_s=0.,candidate_build_s=0.,geometry_s=0.,relation_check_s=0.,
                 sweep_serialization_s=0.,sweep_validation_s=0.,dynamic_predicate_tests=0,
                 dynamic_pair_distances=0)
    knots=[k.copy() for k in initial]
    if geometric_lower_counts is not None:
        if mode not in {'FULL-DAG','SEED-DAG'} or len(geometric_lower_counts)!=len(blocks):
            raise ValueError('Certified geometric bounds require FULL-DAG/SEED-DAG and one count per block')
        if any(not 1<=bound<=len(k) for bound,k in zip(geometric_lower_counts,knots,strict=True)):
            raise ValueError('Invalid geometric minimum count')
    if mode=='SEED-DAG':
        if candidate_pools is None or len(candidate_pools)!=len(blocks):raise ValueError('One fixed candidate pool per block required')
        candidate_pools=[p.copy() for p in candidate_pools]
        for p,k,b in zip(candidate_pools,knots,blocks,strict=True):
            if not np.issubdtype(p.dtype,np.integer) or not len(p) or p[0]!=0 or p[-1]!=len(b.rows)-1 or np.any(np.diff(p)<=0) or not set(map(int,k)).issubset(set(map(int,p))):
                raise ValueError('Fixed pool must contain incumbent path and valid block endpoints')
    current=Recording(rec.metadata,rec.tracks,rec.track,rec.time,reconstruct(rec,blocks,knots),rec.dt)
    if np.linalg.norm(current.xy-rec.xy,axis=1).max()>epsilon or relations(current,radii)!=truth:
        raise ValueError('Pruning seed must satisfy geometry and every declared relationship')
    profile['seed_validation_s']=time.perf_counter()-wall_start
    build_start=time.perf_counter()
    plan=RefinementQuery(rec,radii,epsilon)
    plan._build()
    pairs=plan.rows
    original_distance=np.linalg.norm(rec.xy[pairs[:,0]]-rec.xy[pairs[:,1]],axis=1)
    radii_array=np.asarray(radii,dtype=float)
    active=active_predicates(original_distance,radii_array,epsilon,rec.xy) if danger_band else np.ones((len(pairs),len(radii)),dtype=bool)
    profile['spatial_pairs']=len(pairs)
    profile['full_predicates']=int(active.size)
    profile['band_predicates']=int(active.sum())
    keep=active.any(axis=1)
    pairs=pairs[keep];active=active[keep]
    profile['dynamic_pairs']=len(pairs)
    dirty=set(range(len(blocks))) if active_block_schedule else None
    neighbours=None
    if active_block_schedule:
        owner=np.empty(len(rec.time),dtype=np.int64)
        for n,b in enumerate(blocks):owner[b.rows]=n
        neighbours=[set() for _ in blocks]
        for a,b in np.unique(np.sort(owner[pairs],axis=1),axis=0):
            if a!=b:neighbours[int(a)].add(int(b));neighbours[int(b)].add(int(a))
    profile['dependency_edges']=sum(map(len,neighbours))//2 if neighbours is not None else None
    directed=np.concatenate((pairs,pairs[:,::-1]))
    active=np.concatenate((active,active))
    order=np.argsort(directed[:,0],kind='stable');directed=directed[order];active=active[order]
    offsets=np.searchsorted(directed[:,0],np.arange(len(rec.time)+1))
    original_distance=np.linalg.norm(rec.xy[directed[:,0]]-rec.xy[directed[:,1]],axis=1)
    radii_array=np.asarray(radii,dtype=float)
    gold=original_distance[:,None]<=radii_array
    pi,ri=np.nonzero(active)
    directed=directed[pi]
    predicate_radii=radii_array[ri]
    gold=gold[pi,ri]
    offsets=np.searchsorted(directed[:,0],np.arange(len(rec.time)+1))
    del active,pi,ri,original_distance
    profile['constraint_layout']='predicate-CSR'
    profile['candidate_build_s']=time.perf_counter()-build_start
    start_count=sum(map(len,knots));history=[];trials=0;geometric_rejects=0;relation_rejects=0
    geometry_cache=[None]*len(blocks) if mode in {'FULL-DAG','SEED-DAG'} else None
    reintroduced=0;updates=0;skipped_by_bound=0;skipped_unchanged=0
    for sweep in range(max_sweeps):
        before=sum(map(len,knots))
        for block_index,block in enumerate(blocks):
            check()
            if time.perf_counter()>deadline:
                raise TimeoutError('Pruning wall-time budget exceeded')
            if dirty is not None:
                if block_index not in dirty:
                    skipped_unchanged+=1;continue
                dirty.remove(block_index)
            retained=knots[block_index]
            if len(retained)<=2:
                continue
            if geometric_lower_counts is not None and len(retained)==geometric_lower_counts[block_index]:
                skipped_by_bound+=1;continue
            rows=block.rows;times=rec.time[rows];xy=rec.xy[rows]
            if geometry_cache is not None and geometry_cache[block_index] is None:
                geometry_cache[block_index]=np.full((len(rows),len(rows)),np.nan)
            # 块内原始行在规范车辆—时间序中连续；缺口和公共块边界已切开。
            if not np.array_equal(rows,np.arange(rows[0],rows[-1]+1)):
                raise ValueError('Pruning requires canonical contiguous block rows')
            lo,hi=offsets[rows[0]],offsets[rows[-1]+1]
            local=directed[lo:hi,0]-rows[0];other=directed[lo:hi,1]
            gold_block=gold[lo:hi];threshold_block=predicate_radii[lo:hi]
            cache={}
            if mode in {'FULL-DAG','SEED-DAG'}:
                for a,b in zip(retained[:-1],retained[1:]):cache[int(a),int(b)]=True

            def feasible(left: int,right: int) -> bool:
                nonlocal trials,geometric_rejects,relation_rejects
                key=(left,right)
                if key in cache:
                    return cache[key]
                trials+=1
                geo_start=time.perf_counter()
                if geometry_cache is not None and geometry_cache[block_index][left,right]>epsilon:
                    profile['geometry_s']+=time.perf_counter()-geo_start
                    geometric_rejects+=1;cache[key]=False;return False
                candidate=shortcut_xy(times,xy,left,right)
                geometric_error=float(np.linalg.norm(candidate-xy[left:right+1],axis=1).max())
                if geometry_cache is not None:geometry_cache[block_index][left,right]=geometric_error
                if geometric_error>epsilon:
                    profile['geometry_s']+=time.perf_counter()-geo_start
                    geometric_rejects+=1;cache[key]=False;return False
                profile['geometry_s']+=time.perf_counter()-geo_start
                rel_start=time.perf_counter()
                a=int(offsets[int(rows[0])+left]-lo)
                b=int(offsets[int(rows[0])+right+1]-lo)
                if a==b:
                    valid=True
                else:
                    distance=np.linalg.norm(candidate[local[a:b]-left]-current.xy[other[a:b]],axis=1)
                    valid=bool(np.all((distance<=threshold_block[a:b])==gold_block[a:b]))
                profile['dynamic_pair_distances']+=b-a
                profile['dynamic_predicate_tests']+=b-a
                profile['relation_check_s']+=time.perf_counter()-rel_start
                relation_rejects+=int(not valid);cache[key]=valid
                return valid

            if mode=='FULL-DAG':replacement=shortest_feasible_path(len(rows),feasible)
            elif mode=='SEED-DAG':
                pool=candidate_pools[block_index]
                path=shortest_feasible_path(len(pool),lambda a,b:feasible(int(pool[a]),int(pool[b])))
                replacement=pool[path]
            else:
                choose=shortest_subsequence if mode=='DAG-PRUNE' else greedy_subsequence
                replacement=choose(retained,feasible)
            if len(replacement)<len(retained):
                if geometric_lower_counts is not None and len(replacement)<geometric_lower_counts[block_index]:
                    raise RuntimeError('Feasible path contradicts certified geometric minimum')
                reintroduced+=len(set(map(int,replacement))-set(map(int,retained)));updates+=1
                knots[block_index]=replacement
                current.xy[rows]=interpolate(times,times[replacement],xy[replacement])
                if dirty is not None:dirty.update(neighbours[block_index])
        serial_start=time.perf_counter()
        path=scratch_dir/f'{mode}_sweep_{sweep}.npz'
        encode(path,rec,blocks,knots,radii);decoded,_,_=decode(path);path.unlink()
        profile['sweep_serialization_s']+=time.perf_counter()-serial_start
        validation_start=time.perf_counter()
        # 每轮用独立原查询完整核验，候选图不是最终验收器。
        error=float(np.linalg.norm(decoded.xy-rec.xy,axis=1).max())
        if error>epsilon or relations(decoded,radii)!=truth:
            raise RuntimeError('Pruned serialized representation failed independent checks')
        profile['sweep_validation_s']+=time.perf_counter()-validation_start
        current=decoded
        after=sum(map(len,knots));history.append(dict(sweep=sweep,points_before=before,points_after=after,max_error_m=error))
        if after==before:
            profile['total_s']=time.perf_counter()-wall_start
            profile['conditional_other_s']=profile['total_s']-sum(profile[k] for k in ['seed_validation_s','candidate_build_s','geometry_s','relation_check_s','sweep_serialization_s','sweep_validation_s'])
            return knots,dict(profile=profile,removed_points=start_count-after,sweeps=sweep+1,shortcut_trials=trials,
                              geometric_rejects=geometric_rejects,relation_rejects=relation_rejects,
                              reintroduced_points=reintroduced,successful_block_updates=updates,
                              skipped_by_geometry_bound=skipped_by_bound,
                              skipped_unchanged_blocks=skipped_unchanged,history=history)
    raise TimeoutError('Pruning sweep budget exhausted before fixed point')


def prune(rec: Recording, blocks: list[Block], initial: list[np.ndarray], radii: list[float],
          epsilon: float, truth: dict, mode: str, max_sweeps: int, deadline: float,
          scratch_dir: Path, geometric_lower_counts: list[int] | None = None,
          active_block_schedule: bool = False,
          candidate_pools: list[np.ndarray] | None = None,
          danger_band: bool = True) -> tuple[list[np.ndarray],dict]:

    # The comparator is the exact same instrumented full-check implementation
    # used before CSR refinement; no sparse-index overhead is added to it.
    if not danger_band:
        from .prune_band import prune as full_checks
        chosen=full_checks
    else:
        chosen=_prune_csr
    return chosen(rec,blocks,initial,radii,epsilon,truth,mode,max_sweeps,deadline,
                  scratch_dir,geometric_lower_counts,active_block_schedule,
                  candidate_pools,danger_band)
