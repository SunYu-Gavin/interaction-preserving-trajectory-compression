"""补充强简单对照；用同一冻结测量器注册独立编码函数，不改主方法文件。"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import yaml
from .geometry import make_blocks,independent_knots
from .geo_opt import segment_errors,optimal_knots
from .events import relations
from .local import refine
from .prune import prune

METHODS=('RDP-REPAIR','RDP-GREEDY','GEO-GREEDY')


def compress(rec,epsilon,radii,block_seconds,method,deadline,scratch):
    if method not in METHODS:raise ValueError('Unknown strong control')
    scratch.mkdir(parents=True,exist_ok=True);timing={}
    begin=time.perf_counter();blocks=make_blocks(rec,block_seconds);timing['blocks_s']=time.perf_counter()-begin
    begin=time.perf_counter();truth=relations(rec,radii);timing['truth_s']=time.perf_counter()-begin
    begin=time.perf_counter()
    if method=='GEO-GREEDY':
        graphs=segment_errors(rec,blocks);coarse=optimal_knots(graphs,epsilon)
    else:coarse=independent_knots(rec,blocks,epsilon)
    timing['geometry_s']=time.perf_counter()-begin
    begin=time.perf_counter();knots,stats=refine(rec,blocks,coarse,radii,epsilon,truth,1000,deadline,'ENDPOINT-REPAIR',scratch)
    timing['repair_s']=time.perf_counter()-begin;timing['repair_stats']=stats
    if method!='RDP-REPAIR':
        begin=time.perf_counter();knots,stats=prune(rec,blocks,knots,radii,epsilon,truth,'GREEDY-PRUNE',1000,deadline,scratch)
        timing['greedy_s']=time.perf_counter()-begin;timing['greedy_stats']=stats
    return blocks,knots,None,'none',timing,truth
