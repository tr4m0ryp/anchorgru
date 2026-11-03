"""PhysioNet/CinC Challenge 2012 (open access): 12,000 ICU stays, 48 h, in-hospital mortality.

Downloads sets A, B, C and their outcome files to ``<cache>/physionet2012`` and builds an hourly grid
(n, 48, C) with NaN = not measured (the last measurement inside each hour), the static admission values and the
SAPS-I score (-1 = missing in the source). The grid is cached as ``grid.npz`` next to the downloads.
"""
from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://physionet.org/files/challenge-2012/1.0.0/"
SETS = ("a", "b", "c")
STATIC = ("Age", "Gender", "Height", "ICUType", "Weight")
CACHE = Path(os.environ.get("ANCHORGRU_DATA", "/home/tr4m0ryp/projects/oss/_data"))


def _get(dirn: Path, name: str) -> Path:
    p = dirn / name
    if not p.exists():
        dirn.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL + name, p)
    return p


def _parse(text: str):
    rows = [ln.split(",") for ln in text.strip().splitlines()[1:]]
    return [(int(t[:2]) * 60 + int(t[3:5]), k, float(v)) for t, k, v in rows if v != ""]


def load(cache: Path = CACHE, hours: int = 48):
    """Return dict: X (n, hours, C) float64, static (n, 5), saps (n,), y (n,), ids (n,), set (n,), channels."""
    d = cache / "physionet2012"
    out = d / "grid.npz"
    if out.exists():
        z = np.load(out, allow_pickle=False)
        return {k: z[k] for k in z.files}
    recs, outcomes = {}, {}
    for s in SETS:
        for ln in _get(d, f"Outcomes-{s}.txt").read_text().strip().splitlines()[1:]:
            v = ln.split(",")
            outcomes[int(v[0])] = (float(v[1]), int(v[5]), s)
        with tarfile.open(_get(d, f"set-{s}.tar.gz")) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".txt"):
                    recs[int(Path(m.name).stem)] = _parse(io.TextIOWrapper(tf.extractfile(m)).read())
    ids = np.array(sorted(set(recs) & set(outcomes)))
    chans = sorted({k for r in recs.values() for _, k, _ in r} - set(STATIC) - {"RecordID", ""})
    ci = {k: j for j, k in enumerate(chans)}
    X = np.full((len(ids), hours, len(chans)), np.nan)
    st = np.full((len(ids), len(STATIC)), np.nan)
    for i, rid in enumerate(ids):
        for t, k, v in recs[rid]:
            if k in STATIC:
                if t == 0:
                    st[i, STATIC.index(k)] = np.nan if v < 0 else v
            elif k in ci and v >= 0:
                X[i, min(t // 60, hours - 1), ci[k]] = v
    saps = np.array([outcomes[r][0] for r in ids])
    y = np.array([outcomes[r][1] for r in ids])
    sets = np.array([outcomes[r][2] for r in ids])
    np.savez_compressed(out, X=X, static=st, saps=saps, y=y, ids=ids, set=sets, channels=np.array(chans))
    return load(cache, hours)
