"""编码器内部复用候选对；最终独立评价仍使用 events.relations。"""
from __future__ import annotations
import numpy as np
from .data import Recording
from .events import frames, spatial_pairs, Event


class RefinementQuery:
    """原始距离 <= r_max + 2 * 实测最大误差涵盖所有重建正关系。

    仅供离线编码器使用，不能作为部署查询接口。误差超过已建候选
    上界时重新构建；保守浮点余量只扩充候选，不放宽最终阈值。
    """

    def __init__(self, original: Recording, radii: list[float], initial_bound: float):
        if not radii or min(radii) <= 0 or not np.isfinite(radii).all():
            raise ValueError("Refinement radii must be finite and positive")
        if initial_bound < 0 or not np.isfinite(initial_bound):
            raise ValueError("Refinement error bound must be finite and nonnegative")
        self.original = original
        self.radii = radii
        self.bound = initial_bound
        self.rows = None
        self.keys = None
        self.rebuilds = 0

    def _build(self) -> None:
        rec = self.original
        guard = 64*np.finfo(np.float64).eps*(1+2*float(np.abs(rec.xy).max())+max(self.radii)+2*self.bound)
        radius = max(self.radii)+2*self.bound+guard
        pairs, times = [], []
        for t, rows in frames(rec):
            found = spatial_pairs(rec.xy[rows], radius)
            if len(found):
                pairs.append(rows[found])
                times.append(np.full(len(found), t, dtype=np.int64))
        self.rows = np.concatenate(pairs) if pairs else np.empty((0,2),dtype=np.int64)
        ids = rec.track[self.rows]
        self.keys = np.column_stack((np.concatenate(times) if times else np.empty(0,dtype=np.int64),
                                     np.minimum(ids[:,0],ids[:,1]),np.maximum(ids[:,0],ids[:,1])))
        self.rebuilds += 1

    def relations(self, rebuilt: Recording) -> dict[float,set[Event]]:
        from .resources import check
        check()
        rec = self.original
        if (rebuilt.tracks != rec.tracks or not np.array_equal(rebuilt.time,rec.time)
                or not np.array_equal(rebuilt.track,rec.track) or not np.isfinite(rebuilt.xy).all()):
            raise ValueError("Refinement query requires identical valid vehicle/time grid")
        measured = float(np.linalg.norm(rebuilt.xy-rec.xy,axis=1).max())
        if not np.isfinite(measured):
            raise ValueError("Nonfinite refinement error")
        if self.rows is None or measured > self.bound:
            self.bound = max(self.bound, measured)
            self._build()
        distance = np.linalg.norm(rebuilt.xy[self.rows[:,0]]-rebuilt.xy[self.rows[:,1]],axis=1)
        result = {float(r):set(map(tuple,self.keys[distance<=r].tolist())) for r in self.radii}
        check()
        return result
