"""Garbage-overwrite test for lookahead.

Every row that became known after its entity's cutoff is overwritten with garbage (huge values, sign flips and
NaNs). A window builder that reads only point-in-time data returns exactly the same output before and after; any
difference means it looked ahead.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class LookaheadError(AssertionError):
    """Raised when a window builder's output depends on rows unknown at the cutoff."""


def _future_mask(df, known_at_col: str, cutoff, id_col: str | None) -> np.ndarray:
    known_at = df[known_at_col].to_numpy()
    if id_col is None or np.isscalar(cutoff):
        return known_at > cutoff
    cut = df[id_col].map(lambda k: cutoff[k]).to_numpy()
    return known_at > cut


def _same(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.all(a == b))


def check_no_lookahead(builder: Callable, df, cutoff, known_at_col: str, columns: Sequence[str],
                       id_col: str | None = None, n_trials: int = 3, seed: int = 0):
    """Assert that ``builder(df)`` ignores every row known after the cutoff.

    Parameters
    ----------
    builder : callable taking the table and returning arrays (or a tuple of arrays), e.g.
        ``lambda d: point_in_time_windows(d, ..., cutoff=cutoff, ...)`` or a function that also encodes them.
    cutoff : scalar, or a mapping from entity id to cutoff (then ``id_col`` is required).
    columns : the value columns to overwrite (never the id, time or known-at columns).
    n_trials : number of independent garbage draws.

    Returns the builder's output on the clean table. Raises :class:`LookaheadError` on any difference.
    """
    if not np.isscalar(cutoff) and id_col is None:
        raise ValueError("a per-entity cutoff needs id_col")
    clean = builder(df.copy())
    future = _future_mask(df, known_at_col, cutoff, id_col)
    if not future.any():
        return clean
    rng = np.random.default_rng(seed)
    for t in range(n_trials):
        dirty = df.copy()
        for c in columns:
            g = rng.choice([-1e9, 1e9, np.nan, 7.77], size=int(future.sum())) * (1 + rng.random(int(future.sum())))
            col = dirty[c].to_numpy(dtype=np.float64, copy=True)
            col[future] = g
            dirty[c] = col
        if not _same(clean, builder(dirty)):
            raise LookaheadError(f"output changed when rows known after the cutoff were overwritten (trial {t})")
    return clean
