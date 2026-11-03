"""Point-in-time windows from a long table.

Each row of the table belongs to one entity (``id_col``), is stamped at ``time_col`` and becomes known at
``known_at_col`` (stamp plus the source's publication latency). For a cutoff t*, the window ends at the latest row
that was already known at t* (known_at <= t*), not at the row nominally stamped t*. Rows inside the window that
arrived late (known_at > t*) are blanked to NaN, so nothing that was unknown at the cutoff can reach a model.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _cutoff_of(cutoff, ids):
    if isinstance(cutoff, Mapping):
        return [cutoff[i] for i in ids]
    if hasattr(cutoff, "loc") and hasattr(cutoff, "index"):
        return [cutoff.loc[i] for i in ids]
    return [cutoff] * len(ids)


def point_in_time_windows(df, id_col: str, time_col: str, known_at_col: str, cutoff, length: int,
                          channels: Sequence[str], ids: Sequence | None = None, freq=None):
    """Build one window per entity from the rows known at its cutoff.

    Parameters
    ----------
    df : pandas DataFrame in long format (one row per entity and time stamp).
    cutoff : a scalar (shared by every entity), or a mapping / Series from entity id to its cutoff.
    length : window length; rows when ``freq`` is None, else grid steps of size ``freq``.
    channels : value columns, in channel order.
    ids : entities to build, in output order (default: the sorted unique ids of ``df``).
    freq : if given, the window is the regular grid end - (length - 1) * freq, ..., end, where end is the stamp
        of the last known row; grid points with no row are NaN.

    Returns
    -------
    (X, last_row) : X float64 (n_ids, length, C), NaN where missing or not yet known, oldest step first;
        last_row (n_ids,) object array of the ``df`` index label of each window's end row (None if the entity
        had no known row, its window is then all NaN).
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    channels = list(channels)
    ids = sorted(df[id_col].unique()) if ids is None else list(ids)
    X = np.full((len(ids), length, len(channels)), np.nan)
    last_row = np.empty(len(ids), dtype=object)
    groups = {k: g for k, g in df.groupby(id_col, sort=False)}
    for i, (key, cut) in enumerate(zip(ids, _cutoff_of(cutoff, ids))):
        g = groups.get(key)
        if g is None:
            continue
        g = g.sort_values(time_col, kind="stable")
        known = (g[known_at_col] <= cut).to_numpy()
        if not known.any():
            continue
        times = g[time_col].to_numpy()
        pos_end = int(np.flatnonzero(known)[-1])
        last_row[i] = g.index[pos_end]
        vals = g[channels].to_numpy(dtype=np.float64, na_value=np.nan).copy()
        vals[~known] = np.nan
        if freq is None:
            lo = max(0, pos_end - length + 1)
            block = vals[lo:pos_end + 1]
            X[i, length - len(block):] = block
            continue
        grid = times[pos_end] - np.arange(length - 1, -1, -1) * freq
        sel = known & (times <= times[pos_end]) & (times >= grid[0])
        for t, row in zip(times[sel], vals[sel]):
            j = np.flatnonzero(grid == t)
            if j.size:
                X[i, j[-1]] = row
    return X, last_row
