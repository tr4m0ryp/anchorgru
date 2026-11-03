"""Point-in-time windows: the end row honours publication latency, late rows are blanked, nothing leaks."""
import numpy as np
import pandas as pd
import pytest

from anchorgru.windows import LookaheadError, check_no_lookahead, encode, last_observed, point_in_time_windows


def _table(seed=0, n_ids=6, steps=40, latency=2.0, late_share=0.1):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_ids):
        for t in range(steps):
            if rng.random() < 0.1:
                continue
            delay = latency + (rng.uniform(5, 15) if rng.random() < late_share else 0.0)
            rows.append({"id": f"e{e}", "t": float(t), "known": t + delay,
                         "a": rng.normal(), "p": rng.uniform(0.05, 0.95)})
    return pd.DataFrame(rows)


def test_window_ends_at_the_last_known_row_not_the_cutoff_stamp():
    df = pd.DataFrame({"id": "x", "t": np.arange(10.0), "known": np.arange(10.0) + 1.5, "a": np.arange(10.0)})
    X, last = point_in_time_windows(df, "id", "t", "known", cutoff=6.0, length=3, channels=["a"])
    np.testing.assert_array_equal(X[0, :, 0], [2.0, 3.0, 4.0])
    assert last[0] == 4


def test_late_rows_inside_the_window_are_blanked():
    df = pd.DataFrame({"id": "x", "t": [0.0, 1, 2, 3], "known": [0.0, 9, 2, 3], "a": [10.0, 11, 12, 13]})
    X, _ = point_in_time_windows(df, "id", "t", "known", cutoff=3.0, length=4, channels=["a"])
    np.testing.assert_array_equal(X[0, :, 0], [10.0, np.nan, 12, 13])


def test_short_history_is_left_padded_and_unknown_entity_is_empty():
    df = pd.DataFrame({"id": "x", "t": [0.0, 1], "known": [0.0, 1], "a": [1.0, 2]})
    X, last = point_in_time_windows(df, "id", "t", "known", cutoff=5.0, length=4, channels=["a"], ids=["x", "y"])
    np.testing.assert_array_equal(X[0, :, 0], [np.nan, np.nan, 1, 2])
    assert np.isnan(X[1]).all() and last[1] is None


def test_regular_grid_option():
    df = pd.DataFrame({"id": "x", "t": [0.0, 2, 3, 6], "known": [0.0, 2, 3, 6], "a": [1.0, 2, 3, 4]})
    X, _ = point_in_time_windows(df, "id", "t", "known", cutoff=5.0, length=4, channels=["a"], freq=1.0)
    np.testing.assert_array_equal(X[0, :, 0], [1.0, np.nan, 2.0, 3.0])


@pytest.mark.parametrize("length", [1, 2, 5, 13, 40])
@pytest.mark.parametrize("freq", [None, 1.0])
def test_garbage_after_the_cutoff_changes_nothing(length, freq):
    df = _table()
    cut = {f"e{e}": 20.0 + 3 * e for e in range(6)}

    def build(d):
        X, last = point_in_time_windows(d, "id", "t", "known", cut, length, ["a", "p"], freq=freq)
        seq, valid = encode(X)
        return X, last, seq, valid, last_observed(X, 1)

    check_no_lookahead(build, df, cut, "known", ["a", "p"], id_col="id")


def test_shared_scalar_cutoff():
    df = _table(seed=3)
    check_no_lookahead(lambda d: point_in_time_windows(d, "id", "t", "known", 25.0, 8, ["a"]),
                       df, 25.0, "known", ["a"])


def test_a_builder_that_reads_by_stamp_is_caught():
    df = _table(seed=1, latency=3.0)

    def leaky(d):
        d = d[d["t"] <= 20.0].sort_values(["id", "t"])
        return np.array([g["a"].to_numpy()[-5:] for _, g in d.groupby("id")])

    with pytest.raises(LookaheadError):
        check_no_lookahead(leaky, df, 20.0, "known", ["a"])


def test_per_entity_cutoff_needs_id_col():
    with pytest.raises(ValueError):
        check_no_lookahead(lambda d: 0, _table(), {"e0": 1.0}, "known", ["a"])
