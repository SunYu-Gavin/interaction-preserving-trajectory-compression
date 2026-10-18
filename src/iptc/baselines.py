"""基础方法及实际错误修复；每次升级后重新检查全套关系。"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .data import Recording
from .geometry import Block, reconstruct
from .events import relations


def method_names(cfg: dict) -> list[str]:
    extras=cfg.get("local_methods",[])
    if not isinstance(extras,list) or any(m not in {"POINT-REPAIR","ENDPOINT-REPAIR","LOCAL-JOINT"} for m in extras) or len(set(extras))!=len(extras):
        raise ValueError("local_methods must be a unique list of supported local refinements")
    return ["RAW",*[f"UNIFORM-{s}" for s in cfg["uniform_steps"]],"INDEPENDENT","REPAIR","INDEX","JOINT",*extras]


def raw_knots(blocks: list[Block]) -> list[np.ndarray]:
    return [np.arange(len(b.rows),dtype=np.int64) for b in blocks]


def uniform_knots(blocks: list[Block], step: int) -> list[np.ndarray]:
    if step < 1:
        raise ValueError("Uniform step must be >= 1")
    return [np.union1d(np.arange(0,len(b.rows),step),[len(b.rows)-1]).astype(np.int64) for b in blocks]


def upgraded_knots(blocks: list[Block], coarse: list[np.ndarray], selected: set[int]) -> list[np.ndarray]:
    return [np.arange(len(b.rows),dtype=np.int64) if n in selected else k.copy()
            for n,(b,k) in enumerate(zip(blocks,coarse,strict=True))]


def repair(rec: Recording, blocks: list[Block], coarse: list[np.ndarray], radii: list[float],
           truth: dict, max_iterations: int, deadline: float,
           scratch_dir: Path | None = None) -> tuple[list[np.ndarray],dict]:
    selected = set()
    row_lookup = {(int(t),int(v)):i for i,(t,v) in enumerate(zip(rec.time,rec.track))}
    nodes = np.empty(len(rec.time),dtype=np.int64)
    for n,b in enumerate(blocks):
        nodes[b.rows] = n
    for iteration in range(max_iterations+1):
        if time.perf_counter() > deadline:
            raise TimeoutError("REPAIR time budget exceeded")
        knots = upgraded_knots(blocks,coarse,selected)
        if scratch_dir is not None:
            # 实验中每轮修复都真正写入、重新解码，再查关系。
            from .codecs import encode,decode
            path = scratch_dir/f"repair_iteration_{iteration}.npz"
            encode(path,rec,blocks,knots,radii)
            rebuilt,_,_ = decode(path)
            path.unlink()  # 仅删除本轮自建的临时文件，不触及原始数据。
        else:
            rebuilt = Recording(rec.metadata,rec.tracks,rec.track,rec.time,reconstruct(rec,blocks,knots),rec.dt)
        actual = relations(rebuilt,radii)
        flips = set().union(*(truth[float(r)] ^ actual[float(r)] for r in radii))
        if not flips:
            return knots,dict(repair_iterations=iteration,repair_upgrades=len(selected))
        new = set()
        for t,i,j in flips:
            u = int(nodes[row_lookup[(t,i)]])
            v = int(nodes[row_lookup[(t,j)]])
            if u in selected and v in selected:
                raise RuntimeError("Exact blocks still flip: implementation error")
            new.update((u,v))
        selected.update(new)
    raise TimeoutError("REPAIR iteration budget exceeded")
