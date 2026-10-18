"""Certified per-threshold filtering for geometrically feasible states only."""
import numpy as np


def active_predicates(distance, radii, epsilon, xy):
    """Closed 2ε bands, expanded only for conservative floating arithmetic.

    Both endpoints must already have error at most epsilon. The bound also
    applies to a proposed shortcut only after its geometric check succeeds.
    This routine is deliberately not used in initialization repair.
    """
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError('Finite nonnegative epsilon required')
    guard=128*np.finfo(np.float64).eps*(1+2*float(np.max(np.abs(xy)))+float(np.max(radii))+2*epsilon)
    return np.abs(np.asarray(distance)[:,None]-np.asarray(radii)[None,:]) <= 2*epsilon+guard
