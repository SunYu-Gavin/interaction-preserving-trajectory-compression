"""从规范原始数据独立编码；不读取预先计算的种子、图或真值缓存。"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from .data import Recording
from .geometry import make_blocks
from .baselines import raw_knots
from .geo_opt import segment_errors,optimal_knots
from .margin import margin_graphs
from .local import refine
from .prune import prune
from .events import relations,pack_index
from .paper_codec import encode,decode

METHODS=('RAW','GEOMETRY','FULL-INDEX','XOR-INDEX','MARGIN-DAG','RAW-DAG','FULL-DAG')


def compress(rec: Recording,epsilon: float,radii: list[float],block_seconds: float,
             method: str,deadline: float,scratch: Path) -> tuple:
    if method not in METHODS:raise ValueError(f'Unknown encoder {method}')
    if epsilon<=0 or not np.isfinite(epsilon):raise ValueError('Positive finite epsilon required')
    scratch.mkdir(parents=True,exist_ok=True)
    timing={};start=time.perf_counter();blocks=make_blocks(rec,block_seconds);timing['blocks_s']=time.perf_counter()-start
    truth=None;mode='none';index=None
    if method=='RAW':return blocks,raw_knots(blocks),index,mode,timing,truth
    if method in {'FULL-INDEX','XOR-INDEX','RAW-DAG','FULL-DAG'}:
        start=time.perf_counter();truth=relations(rec,radii);timing['truth_s']=time.perf_counter()-start
    if method=='RAW-DAG':
        start=time.perf_counter()
        knots,stats=prune(rec,blocks,raw_knots(blocks),radii,epsilon,truth,'FULL-DAG',1000,deadline,scratch,None,True)
        timing['conditional_s']=time.perf_counter()-start;timing['conditional_stats']=stats
        return blocks,knots,None,'none',timing,truth
    start=time.perf_counter()
    graphs,_=margin_graphs(rec,blocks,radii,epsilon) if method=='MARGIN-DAG' else (segment_errors(rec,blocks),None)
    timing['graph_build_s']=time.perf_counter()-start
    start=time.perf_counter();knots=optimal_knots(graphs,epsilon);timing['geometry_solve_s']=time.perf_counter()-start
    if method in {'GEOMETRY','MARGIN-DAG'}:return blocks,knots,None,'none',timing,truth
    if method=='FULL-INDEX':
        start=time.perf_counter();index=pack_index(truth,radii,rec.dt);timing['index_build_s']=time.perf_counter()-start;mode='full'
    elif method=='XOR-INDEX':
        start=time.perf_counter();temporary=scratch/'xor_geometry_reference.npz'
        encode(temporary,rec,blocks,knots,radii)
        try:
            rebuilt,_,_,_=decode(temporary);actual=relations(rebuilt,radii)
            index=pack_index({float(r):truth[float(r)]^actual[float(r)] for r in radii},radii,rec.dt)
        finally:temporary.unlink()
        timing['index_build_s']=time.perf_counter()-start;mode='xor'
    elif method=='FULL-DAG':
        lower=list(map(len,knots));start=time.perf_counter()
        knots,stats=refine(rec,blocks,knots,radii,epsilon,truth,1000,deadline,'ENDPOINT-REPAIR',scratch)
        timing['seed_repair_s']=time.perf_counter()-start;timing['seed_stats']=stats
        start=time.perf_counter()
        knots,stats=prune(rec,blocks,knots,radii,epsilon,truth,'DAG-PRUNE',1000,deadline,scratch)
        timing['restricted_s']=time.perf_counter()-start;timing['restricted_stats']=stats
        start=time.perf_counter()
        knots,stats=prune(rec,blocks,knots,radii,epsilon,truth,'FULL-DAG',1000,deadline,scratch,lower,True)
        timing['conditional_s']=time.perf_counter()-start;timing['conditional_stats']=stats
    return blocks,knots,index,mode,timing,truth
