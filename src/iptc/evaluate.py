"""独立评价只依据写入后解码的坐标和事件；不读取求解成功标记。"""
from __future__ import annotations
import time
import numpy as np
from .data import Recording
from .events import relations, intervals, expand_index


def evaluate(raw: Recording, decoded: Recording, radii: list[float], epsilon: float | None,
             truth: dict, index: np.ndarray | None = None) -> tuple[list[dict],dict]:
    if raw.tracks != decoded.tracks or raw.metadata != decoded.metadata:
        raise ValueError("Decoder identity mapping differs from reference")
    if not np.array_equal(raw.time,decoded.time) or not np.array_equal(raw.track,decoded.track):
        raise ValueError("Decoder valid time/vehicle grid differs from reference")
    error = np.linalg.norm(raw.xy-decoded.xy,axis=1)
    begin = time.perf_counter()
    actual = relations(decoded,radii)
    query_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    indexed = expand_index(index,radii,decoded.dt) if index is not None else None
    index_seconds = time.perf_counter()-begin if indexed is not None else None
    results = []
    example = None
    for r in radii:
        gold,pred = truth[float(r)],actual[float(r)]
        fp,fn = pred-gold,gold-pred
        if example is None and (fp or fn):
            t,i,j = min(fp|fn)
            rows = [int(np.flatnonzero((raw.time==t)&(raw.track==v))[0]) for v in (i,j)]
            example = dict(radius=r,time=t,track_ids=[raw.tracks[i],raw.tracks[j]],
                           raw_xy=raw.xy[rows].tolist(),decoded_xy=decoded.xy[rows].tolist(),
                           raw_distance=float(np.linalg.norm(raw.xy[rows[0]]-raw.xy[rows[1]])),
                           decoded_distance=float(np.linalg.norm(decoded.xy[rows[0]]-decoded.xy[rows[1]])))
        interval_diff = len(intervals(gold,raw.dt)^intervals(pred,raw.dt))
        row = dict(radius_m=r,max_error_m=float(error.max()),mean_error_m=float(error.mean()),
                   geometry_pass=epsilon is None or float(error.max()) <= epsilon,
                   flips=len(fp)+len(fn),false_positive=len(fp),false_negative=len(fn),
                   positive_relations=len(gold),no_positive_samples=not bool(gold),
                   flip_per_positive=(len(fp)+len(fn))/len(gold) if gold else None,
                   interval_symmetric_difference=interval_diff,coordinate_query_s=query_seconds,
                   index_query_s=index_seconds,index_flips=None,index_interval_difference=None)
        if indexed is not None:
            row["index_flips"] = len(gold ^ indexed[float(r)])
            row["index_interval_difference"] = len(intervals(gold,raw.dt)^intervals(indexed[float(r)],raw.dt))
        results.append(row)
    return results,dict(flip_example=example)
