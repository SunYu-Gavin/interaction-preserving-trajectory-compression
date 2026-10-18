"""Additional baselines without modifying any of the 32 frozen source files."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .geometry import make_blocks
from .baselines import raw_knots
from .geo_opt import segment_errors,optimal_knots
from .events import relations,pack_index
from .local import refine
from .prune import prune
from .paper_codec import encode,decode

METHODS=('RAW-GREEDY','SEED-DAG','XOR-SCAN')
SCAN_FRACTIONS=(0.,.25,.5,.75,1.)

def compress(rec,epsilon,radii,block_seconds,method,deadline,scratch):
    if method not in METHODS:raise ValueError(method)
    scratch.mkdir(parents=True,exist_ok=True);timing={};begin=time.perf_counter()
    blocks=make_blocks(rec,block_seconds);timing['blocks_s']=time.perf_counter()-begin
    begin=time.perf_counter();truth=None if method in {'SQUISH-E','CISED-S'} else relations(rec,radii);timing['truth_s']=time.perf_counter()-begin
    if method=='RAW-GREEDY':
        begin=time.perf_counter();knots,stats=prune(rec,blocks,raw_knots(blocks),radii,epsilon,truth,'GREEDY-PRUNE',1000,deadline,scratch)
        timing['optimization_s']=time.perf_counter()-begin;timing['optimization_stats']=stats
        return blocks,knots,None,'none',timing,truth
    if method=='SEED-DAG':
        begin=time.perf_counter();graphs=segment_errors(rec,blocks);coarse=optimal_knots(graphs,epsilon)
        timing['geometry_s']=time.perf_counter()-begin;lower=list(map(len,coarse))
        begin=time.perf_counter();knots,stats=refine(rec,blocks,coarse,radii,epsilon,truth,1000,deadline,'ENDPOINT-REPAIR',scratch)
        timing['repair_s']=time.perf_counter()-begin;timing['repair_stats']=stats;pools=[k.copy() for k in knots]
        begin=time.perf_counter();knots,stats=prune(rec,blocks,knots,radii,epsilon,truth,'DAG-PRUNE',1000,deadline,scratch)
        timing['restricted_s']=time.perf_counter()-begin;timing['restricted_stats']=stats
        begin=time.perf_counter();knots,stats=prune(rec,blocks,knots,radii,epsilon,truth,'SEED-DAG',1000,deadline,scratch,lower,True,pools)
        timing['optimization_s']=time.perf_counter()-begin;timing['optimization_stats']=stats
        return blocks,knots,None,'none',timing,truth
    if method=='XOR-SCAN':
        begin=time.perf_counter();graphs=segment_errors(rec,blocks);timing['graph_s']=time.perf_counter()-begin
        candidates=[];best=None;begin=time.perf_counter()
        for label,fraction in [('RAW',None),*[(str(f),f) for f in SCAN_FRACTIONS]]:
            if time.perf_counter()>deadline:raise TimeoutError('Index budget scan deadline')
            knots=raw_knots(blocks) if fraction is None else optimal_knots(graphs,epsilon*fraction)
            path=scratch/f'scan_geometry_{label}.npz';encode(path,rec,blocks,knots,radii)
            rebuilt,_,_,_=decode(path);path.unlink();actual=relations(rebuilt,radii)
            correction=pack_index({float(r):truth[float(r)]^actual[float(r)] for r in radii},radii,rec.dt)
            path=scratch/f'scan_index_{label}.npz';size=encode(path,rec,blocks,knots,radii,correction,'xor');path.unlink()
            candidates.append(dict(label=label,error_fraction=fraction,points=sum(map(len,knots)),bytes=size,correction_intervals=len(correction)))
            if best is None or size<best[0]:best=(size,knots,correction,label)
        timing['scan_s']=time.perf_counter()-begin;timing['scan_candidates']=candidates;timing['selected_candidate']=best[3]
        return blocks,best[1],best[2],'xor',timing,truth
    raise ValueError(method)
