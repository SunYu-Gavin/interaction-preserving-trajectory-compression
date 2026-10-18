"""统一自包含 NPZ 格式：无对象数组，所有方法使用 ZIP DEFLATE。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .data import Recording
from .geometry import Block, interpolate


def encode(path: Path, rec: Recording, blocks: list[Block], knots: list[np.ndarray],
           radii: list[float], index: np.ndarray | None = None) -> int:
    rec.validate()
    times, points, positions = [], [], []
    offsets, knot_offsets, tracks = [0],[0],[]
    for b,k in zip(blocks,knots,strict=True):
        if len(k) == 0 or k[0] != 0 or k[-1] != len(b.rows)-1:
            raise ValueError("Every block must retain first and last valid sample")
        times.extend(rec.time[b.rows])
        positions.extend(k)
        points.extend(rec.xy[b.rows[k]])
        offsets.append(len(times))
        knot_offsets.append(len(points))
        tracks.append(b.track)
    metadata = dict(format="IPTC-NPZ-1", recording=rec.metadata, tracks=rec.tracks,
                    dt=rec.dt,radii=radii,index_present=index is not None,
                    coordinate_dtype="float64",compression="zip-deflate")
    arrays = dict(metadata=np.frombuffer(json.dumps(metadata,ensure_ascii=False,sort_keys=True).encode("utf-8"),dtype=np.uint8),
                  time=np.array(times,dtype=np.int64),block_offsets=np.array(offsets,dtype=np.int64),
                  block_track=np.array(tracks,dtype=np.int32),knot_offsets=np.array(knot_offsets,dtype=np.int64),
                  knot_position=np.array(positions,dtype=np.int64),xy=np.array(points,dtype=np.float64).reshape(-1,2))
    if index is not None:
        arrays["event_index"] = index
    path.parent.mkdir(parents=True,exist_ok=True)
    # 独占创建，防止覆盖已经完成的成果。
    with path.open("xb") as f:
        np.savez_compressed(f,**arrays)
    return path.stat().st_size


def decode(path: Path) -> tuple[Recording,np.ndarray | None,list[float]]:
    with np.load(path,allow_pickle=False) as z:
        meta = json.loads(z["metadata"].tobytes().decode("utf-8"))
        if meta["format"] != "IPTC-NPZ-1":
            raise ValueError("Unknown compressed format")
        time,offsets,koffsets = z["time"],z["block_offsets"],z["knot_offsets"]
        # NPZ 下标读取会重新解压整份数组；每个数组只读取一次。
        block_track = z["block_track"]
        knot_position = z["knot_position"]
        knot_xy = z["xy"]
        track = np.empty(len(time),dtype=np.int32)
        xy = np.empty((len(time),2),dtype=np.float64)
        for b in range(len(block_track)):
            start,end = int(offsets[b]),int(offsets[b+1])
            ks,ke = int(koffsets[b]),int(koffsets[b+1])
            k = knot_position[ks:ke]
            xy[start:end] = interpolate(time[start:end],time[start:end][k],knot_xy[ks:ke])
            track[start:end] = block_track[b]
        rec = Recording(meta["recording"],meta["tracks"],track,time.copy(),xy,int(meta["dt"]))
        index = z["event_index"].copy() if meta["index_present"] else None
    rec.validate()
    return rec,index,meta["radii"]
