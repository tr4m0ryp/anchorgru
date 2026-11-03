"""BlockedTimeSplit: every row tested once, blocks never split, embargo respected."""
import numpy as np
import pytest

from anchorgru import BlockedTimeSplit


@pytest.mark.parametrize("embargo", [0, 1, 3])
def test_kfold_covers_rows_once_and_keeps_the_embargo(embargo):
    groups = np.repeat(np.arange(20), 5)
    X = np.zeros((len(groups), 1))
    seen = np.zeros(len(groups), int)
    for tr, te in BlockedTimeSplit(4, embargo=embargo).split(X, groups=groups):
        seen[te] += 1
        tb, rb = set(groups[te]), set(groups[tr])
        assert not tb & rb
        assert min(abs(a - b) for a in tb for b in rb) > embargo
    assert np.all(seen == 1)


def test_forward_chaining_trains_only_on_the_past():
    X = np.zeros((50, 1))
    s = BlockedTimeSplit(5, embargo=2, forward=True)
    splits = list(s.split(X))
    assert len(splits) == s.get_n_splits(X) == 4
    for tr, te in splits:
        assert tr.max() < te.min() - 2


def test_groups_order_by_first_appearance():
    groups = np.array(["b", "b", "a", "a", "c", "c"])
    folds = [te for _, te in BlockedTimeSplit(3).split(np.zeros((6, 1)), groups=groups)]
    assert [list(f) for f in folds] == [[0, 1], [2, 3], [4, 5]]


def test_bad_arguments():
    with pytest.raises(ValueError):
        BlockedTimeSplit(1)
    with pytest.raises(ValueError):
        list(BlockedTimeSplit(5).split(np.zeros((3, 1))))
