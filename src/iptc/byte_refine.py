"""Feasible anchor search scored by the unchanged paper NPZ byte objective.

This is an experimental refinement, separate from the frozen paper encoders.
"""
from __future__ import annotations
import io
import time
import zlib
import numpy as np
from .geometry import interpolate, reconstruct
from .refinement_query import RefinementQuery
from .danger_band import active_predicates
from .prune_band_csr import shortcut_xy


def compressed_array_size(array):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return len(zlib.compress(stream.getvalue()))


def variable_bytes(rec, blocks, knots):
    """Exactly the three varying DEFLATE streams; other ZIP bytes are constant."""
    counts = np.array([len(k) for k in knots], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    positions = np.concatenate(knots).astype(np.int64)
    xy = np.concatenate([rec.xy[b.rows[k]] for b, k in zip(blocks, knots, strict=True)])
    return sum(compressed_array_size(a) for a in (offsets, positions, xy))


def candidate_paths(size, feasible, width=8):
    """A bounded deterministic list of shortest feasible DAG paths.

    Keep alternatives per vertex, including one extra anchor beyond its shortest
    count. This exposes ties suppressed by the point-count-only solver.
    """
    paths = [[] for _ in range(size)]
    paths[0] = [(0,)]
    for right in range(1, size):
        choices = []
        for left in range(right):
            if paths[left] and feasible(left, right):
                choices.extend(p + (right,) for p in paths[left])
        if choices:
            choices.sort(key=lambda p: (len(p), p))
            minimum = len(choices[0])
            shortest = [p for p in choices if len(p) == minimum]
            longer = [p for p in choices if len(p) == minimum + 1]
            def spread(items, limit):
                if len(items) <= limit:
                    return items
                return [items[i] for i in np.linspace(0, len(items)-1, limit, dtype=int)]
            paths[right] = spread(shortest, width) + spread(longer, max(2, width//2))
    return [np.asarray(p, dtype=np.int64) for p in paths[-1]]


def refine_bytes(rec, blocks, initial, radii, epsilon, *, sweeps=2, width=8,
                 deadline=float('inf'), progress=None):
    start = time.perf_counter()
    knots = [k.copy() for k in initial]
    current = reconstruct(rec, blocks, knots)
    plan = RefinementQuery(rec, radii, epsilon)
    plan._build()
    pairs = plan.rows
    distance = np.linalg.norm(rec.xy[pairs[:, 0]]-rec.xy[pairs[:, 1]], axis=1)
    active = active_predicates(distance, np.asarray(radii), epsilon, rec.xy)
    keep = active.any(axis=1)
    pairs, active = pairs[keep], active[keep]
    directed = np.concatenate((pairs, pairs[:, ::-1]))
    active = np.concatenate((active, active))
    order = np.argsort(directed[:, 0], kind='stable')
    directed, active = directed[order], active[order]
    offsets = np.searchsorted(directed[:, 0], np.arange(len(rec.time)+1))
    original = np.linalg.norm(rec.xy[directed[:, 0]]-rec.xy[directed[:, 1]], axis=1)
    gold = original[:, None] <= np.asarray(radii)
    geometry = [dict() for _ in blocks]
    score = variable_bytes(rec, blocks, knots)
    history = []
    trials = accepted = 0
    for sweep in range(sweeps):
        before = score
        for bi, block in enumerate(blocks):
            if time.perf_counter() > deadline:
                raise TimeoutError('Byte refinement deadline')
            rows = block.rows
            if len(rows) <= 2:
                continue
            times, xy = rec.time[rows], rec.xy[rows]
            lo, hi = offsets[rows[0]], offsets[rows[-1]+1]
            local = directed[lo:hi, 0]-rows[0]
            other = directed[lo:hi, 1]
            cache = {(int(a), int(b)): True for a, b in zip(knots[bi][:-1], knots[bi][1:])}
            def feasible(left, right):
                key = left, right
                if key in cache:
                    return cache[key]
                candidate = shortcut_xy(times, xy, left, right)
                if key not in geometry[bi]:
                    geometry[bi][key] = float(np.linalg.norm(candidate-xy[left:right+1], axis=1).max())
                if geometry[bi][key] > epsilon:
                    cache[key] = False
                    return False
                a = int(offsets[rows[0]+left]-lo)
                b = int(offsets[rows[0]+right+1]-lo)
                distances = np.linalg.norm(candidate[local[a:b]-left]-current[other[a:b]], axis=1)
                valid = bool(np.all(((distances[:, None] <= np.asarray(radii)) == gold[lo+a:lo+b]) | ~active[lo+a:lo+b]))
                cache[key] = valid
                return valid
            incumbent = knots[bi]
            best, best_score = incumbent, score
            for candidate in candidate_paths(len(rows), feasible, width):
                if np.array_equal(candidate, incumbent):
                    continue
                knots[bi] = candidate
                trial = variable_bytes(rec, blocks, knots)
                trials += 1
                if trial < best_score:
                    best, best_score = candidate, trial
            knots[bi] = best
            if best_score < score:
                accepted += 1
                current[rows] = interpolate(times, times[best], xy[best])
                score = best_score
            if progress and bi % 200 == 0:
                progress(dict(sweep=sweep, block=bi, blocks=len(blocks), gain=before-score,
                              elapsed_s=time.perf_counter()-start))
        history.append(dict(sweep=sweep, variable_bytes_before=before, variable_bytes_after=score))
        if before == score:
            break
    return knots, dict(history=history, candidate_files=trials, accepted=accepted,
                       refinement_s=time.perf_counter()-start)
