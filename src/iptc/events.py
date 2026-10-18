"""稀疏关系查询与离散有效采样区间。查询不需要原始轨迹。"""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import numpy as np
from .data import Recording

Event = tuple[int, int, int]  # time, track_i, track_j
Interval = tuple[int, int, int, int, int]  # i,j,start,end,count


def frames(rec: Recording):
    order = np.argsort(rec.time, kind="stable")
    cuts = np.flatnonzero(np.diff(rec.time[order]) != 0) + 1
    for rows in np.split(order, cuts):
        from .resources import check
        check()
        yield int(rec.time[rows[0]]), rows


def spatial_pairs(xy: np.ndarray, radius: float) -> np.ndarray:
    """半径大小网格：同格及 8 邻格完整覆盖圆内点对，无近邻数量截断。"""
    if radius <= 0 or not np.isfinite(radius) or not np.isfinite(xy).all():
        raise ValueError("Spatial query requires positive finite radius and finite points")
    cells = defaultdict(list)
    pairs = []
    for i, cell in enumerate(np.floor(xy / radius).astype(np.int64)):
        a,b = map(int,cell)
        for da in (-1,0,1):
            for db in (-1,0,1):
                for j in cells[(a+da,b+db)]:
                    pairs.append((j,i))
        cells[(a,b)].append(i)
    pairs = np.array(pairs, dtype=np.int64).reshape(-1,2)
    if len(pairs):
        distance = np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)
        pairs = pairs[distance <= radius]
    return pairs


def relations(rec: Recording, radii: list[float], exhaustive: bool = False) -> dict[float,set[Event]]:
    if not radii or min(radii) <= 0 or not np.isfinite(radii).all():
        raise ValueError("Query radii must be finite and positive")
    result = {float(r):set() for r in radii}
    for t, rows in frames(rec):
        xy = rec.xy[rows]
        pairs = np.array(list(combinations(range(len(rows)),2)), dtype=np.int64).reshape(-1,2) if exhaustive else spatial_pairs(xy,max(radii))
        if not len(pairs):
            continue
        distance = np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)
        for r in radii:
            for i,j in pairs[distance <= r]:
                a,b = sorted((int(rec.track[rows[i]]),int(rec.track[rows[j]])))
                result[float(r)].add((t,a,b))
    return result


def intervals(events: set[Event], dt: int) -> set[Interval]:
    if dt <= 0:
        raise ValueError("Interval time step must be positive")
    grouped = defaultdict(list)
    for t,i,j in events:
        grouped[(i,j)].append(t)
    result = set()
    for (i,j), times in grouped.items():
        times.sort()
        start = end = times[0]
        count = 1
        for t in times[1:]:
            if t-end != dt:
                result.add((i,j,start,end,count))
                start,count = t,0
            end,count = t,count+1
        result.add((i,j,start,end,count))
    return result


def expand_index(index: np.ndarray, radii: list[float], dt: int) -> dict[float,set[Event]]:
    result = {float(r):set() for r in radii}
    for ri,i,j,start,end,count in index:
        times = range(int(start),int(end)+1,dt)
        if len(times) != count:
            raise ValueError("Corrupt interval index count")
        result[float(radii[int(ri)])].update((t,int(i),int(j)) for t in times)
    return result


def pack_index(events: dict[float,set[Event]], radii: list[float], dt: int) -> np.ndarray:
    rows = [(ri,*v) for ri,r in enumerate(radii) for v in sorted(intervals(events[float(r)],dt))]
    return np.array(rows,dtype=np.int64).reshape(-1,6)
