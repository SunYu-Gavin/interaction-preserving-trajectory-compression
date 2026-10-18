"""Full raw-input encoder for the experimental byte-aware FULL-DAG variant."""
from __future__ import annotations
import time
from .paper_encoder import compress as original_compress
from .byte_refine import refine_bytes


def compress(rec, epsilon, radii, block_seconds, method, deadline, scratch):
    if method != 'FULL-DAG-BYTE':
        raise ValueError(method)
    blocks, knots, index, mode, timing, truth = original_compress(
        rec, epsilon, radii, block_seconds, 'FULL-DAG', deadline, scratch)
    start = time.perf_counter()
    knots, stats = refine_bytes(rec, blocks, knots, radii, epsilon,
                               sweeps=2, width=8, deadline=deadline)
    timing['byte_refinement_s'] = time.perf_counter()-start
    timing['byte_refinement_stats'] = stats
    return blocks, knots, index, mode, timing, truth
