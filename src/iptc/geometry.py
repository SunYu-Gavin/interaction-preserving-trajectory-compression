"""按时间线性插值的同步欧氏误差简化；不使用二维垂距 RDP。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .data import Recording


@dataclass
class Block:
    rows: np.ndarray
    track: int


def make_blocks(rec: Recording, seconds: float) -> list[Block]:
    ticks = seconds * rec.metadata["ticks_per_second"]
    if seconds <= 0 or ticks != int(ticks):
        raise ValueError("Block duration must be a positive integer number of ticks")
    blocks = []
    for track in range(len(rec.tracks)):
        rows = np.flatnonzero(rec.track == track)
        if not len(rows):
            continue
        # 共同录制时间轴从 0 开始，块互斥；缺失处另起块。
        groups = rec.time[rows] // int(ticks)
        splits = np.flatnonzero((np.diff(groups) != 0) | (np.diff(rec.time[rows]) != rec.dt)) + 1
        for chunk in np.split(rows, splits):
            blocks.append(Block(chunk, track))
    return blocks


def interpolate(t: np.ndarray, knot_time: np.ndarray, knot_xy: np.ndarray) -> np.ndarray:
    if len(knot_time) == 0 or np.any(np.diff(knot_time) <= 0):
        raise ValueError("Knots must be nonempty and strictly increasing")
    if t.min() < knot_time[0] or t.max() > knot_time[-1]:
        raise ValueError("Extrapolation is not permitted")
    result = np.column_stack([np.interp(t, knot_time, knot_xy[:, d]) for d in range(2)])
    # 精确恢复保留点，避免端点算术导致零误差兜底失效。
    idx = np.searchsorted(t, knot_time)
    if np.any(idx >= len(t)) or not np.array_equal(t[idx], knot_time):
        raise ValueError("Knots must lie on the valid time grid")
    result[idx] = knot_xy
    return result


def simplify(t: np.ndarray, xy: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon < 0 or not np.isfinite(epsilon) or len(t) == 0:
        raise ValueError("epsilon must be finite/nonnegative and trajectory nonempty")
    if np.any(np.diff(t) <= 0) or not np.isfinite(xy).all():
        raise ValueError("Simplifier requires finite positions and unique increasing times")
    keep = {0, len(t) - 1}
    stack = [(0, len(t) - 1)]
    while stack:
        left, right = stack.pop()
        if right - left <= 1:
            continue
        middle = np.arange(left + 1, right)
        # 使用与解码器相同的 np.interp 计算；严格比较，不放宽 epsilon。
        estimate = np.column_stack([np.interp(t[middle], t[[left,right]], xy[[left,right], d]) for d in range(2)])
        error = np.linalg.norm(estimate - xy[middle], axis=1)
        arg = int(np.argmax(error))
        if error[arg] > epsilon:
            split = int(middle[arg])
            keep.add(split)
            stack.extend([(left, split), (split, right)])
    knots = np.array(sorted(keep), dtype=np.int64)
    # 全局重建复核；必要时保留边界数值误差点，不能放宽误差要求。
    while True:
        errors = np.linalg.norm(interpolate(t, t[knots], xy[knots]) - xy, axis=1)
        bad = np.flatnonzero(errors > epsilon)
        if not len(bad):
            return knots
        knots = np.union1d(knots, bad).astype(np.int64)


def independent_knots(rec: Recording, blocks: list[Block], epsilon: float) -> list[np.ndarray]:
    return [simplify(rec.time[b.rows], rec.xy[b.rows], epsilon) for b in blocks]


def reconstruct(rec: Recording, blocks: list[Block], knots: list[np.ndarray]) -> np.ndarray:
    xy = np.empty_like(rec.xy)
    for b, k in zip(blocks, knots, strict=True):
        xy[b.rows] = interpolate(rec.time[b.rows], rec.time[b.rows[k]], rec.xy[b.rows[k]])
    return xy
