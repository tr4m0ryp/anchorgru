"""Paired per-example log-loss differences recomputed from the per-split prediction files.

Each split file ``<raw>/<dataset>/<split>.npz`` holds ``keys``, ``y`` and ``p_<arm>`` for one train / test split.
Differences are ours minus arm per test example, averaged over the splits that share an example (as the benchmark
runner does), then summarised three ways:
  iid      sd / sqrt(n) over examples (independent test rows)
  block    moving-block bootstrap over contiguous rows (serially dependent rows), same block, draws and seed as the
           benchmark runner, so it must reproduce the reported SE
  split    SE of the per-split mean difference over the S training splits, sd / sqrt(S); it carries the
           training-subsample and seed noise that the per-example SEs leave out
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-12


def example_ll(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def load_splits(raw: Path, size: int):
    files = sorted(raw.glob(f"n{size}_*.npz"))
    return [dict(np.load(f)) for f in files]


def pooled_diff(splits, arm: str):
    """Per-example ours - arm, averaged over splits sharing a key; returned in key order."""
    keys = np.unique(np.concatenate([z["keys"] for z in splits]))
    tot, cnt = np.zeros(len(keys)), np.zeros(len(keys))
    for z in splits:
        pos = np.searchsorted(keys, z["keys"])
        d = example_ll(z["y"], z["p_ours"]) - example_ll(z["y"], z[f"p_{arm}"])
        np.add.at(tot, pos, d)
        np.add.at(cnt, pos, 1)
    return tot / cnt


def split_means(splits, arm: str):
    return np.array([np.mean(example_ll(z["y"], z["p_ours"]) - example_ll(z["y"], z[f"p_{arm}"]))
                     for z in splits])


def iid_se(d) -> float:
    return float(np.std(d, ddof=1) / np.sqrt(len(d)))


def block_se(d, block: int, draws: int = 2000, seed: int = 0) -> float:
    """Moving-block bootstrap SE of the mean (identical algorithm and seed to the benchmark runner)."""
    n = len(d)
    b = max(1, min(block, n))
    k = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - b + 1, size=(draws, k))
    csum = np.concatenate([[0.0], np.cumsum(d)])
    means = (csum[starts + b] - csum[starts]).sum(axis=1) / (k * b)
    return float(means.std(ddof=1))


def split_se(m) -> float:
    return float(np.std(m, ddof=1) / np.sqrt(len(m))) if len(m) > 1 else float("nan")


def summarise(raw: Path, size: int, arm: str, block: int | None) -> dict:
    sp = load_splits(raw, size)
    d = pooled_diff(sp, arm)
    m = split_means(sp, arm)
    out = {"mean": float(d.mean()), "n": int(len(d)), "se_iid": iid_se(d), "se_split": split_se(m),
           "split_means": m.tolist()}
    if block:
        out["se_block"] = block_se(d, block)
    return out


def reliability(y, p, bins: int = 10):
    """Equal-mass bins: (mean p, observed rate, count) per bin, plus ECE over the same bins."""
    order = np.argsort(p, kind="stable")
    rows, err = [], 0.0
    for part in np.array_split(order, bins):
        if part.size:
            mp, my = float(np.mean(p[part])), float(np.mean(y[part]))
            rows.append((mp, my, part.size))
            err += part.size * abs(mp - my)
    return rows, err / len(p)
