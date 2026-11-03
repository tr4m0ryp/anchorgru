"""Electricity (elec2, OpenML 151): 45,312 half-hourly rows; label = price above its 24 h average."""
from __future__ import annotations

import numpy as np

from .physionet import CACHE

NUMERIC = ("nswprice", "nswdemand", "vicprice", "vicdemand", "transfer")
CURRENT = ("day", "period") + NUMERIC


def load(cache=CACHE):
    """Return dict: rows (n, 7) current-row features, seq (n, 5) the numeric channels, y (n,) in {0, 1}."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=151, as_frame=True, data_home=str(cache / "openml"), parser="auto")
    df = d.frame
    y = (df["class"].astype(str) == "UP").astype(int).to_numpy()
    rows = df[list(CURRENT)].astype(float).to_numpy()
    return {"rows": rows, "seq": df[list(NUMERIC)].astype(float).to_numpy(), "y": y}
