"""A cross-validation splitter for serially dependent rows: contiguous blocks with an embargo gap."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class BlockedTimeSplit(BaseCrossValidator):
    """Split time-ordered rows into contiguous folds and drop an embargo around each test fold.

    Rows are assumed to be in time order. ``groups`` (optional) gives each row a block id (e.g. a day or an
    episode); blocks are ordered by first appearance and never split between train and test. Without groups every
    row is its own block.

    Parameters
    ----------
    n_splits : number of folds (contiguous runs of blocks).
    embargo : number of blocks on each side of a test fold removed from training, so neighbouring blocks that
        share information with the test fold do not train the model.
    forward : if True, train only on blocks before the test fold (forward chaining); the first fold, which has
        no past, is skipped, so ``n_splits - 1`` splits are produced.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0, forward: bool = False):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits, self.embargo, self.forward = n_splits, embargo, forward

    def _blocks(self, n: int, groups):
        if groups is None:
            return np.arange(n), n
        _, first, inv = np.unique(np.asarray(groups), return_index=True, return_inverse=True)
        order = np.argsort(np.argsort(first))
        return order[inv], len(first)

    def split(self, X, y=None, groups=None):
        n = len(X)
        block, n_blocks = self._blocks(n, groups)
        if n_blocks < self.n_splits:
            raise ValueError(f"{n_blocks} blocks cannot make {self.n_splits} folds")
        edges = np.linspace(0, n_blocks, self.n_splits + 1).round().astype(int)
        for k in range(self.n_splits):
            lo, hi = edges[k], edges[k + 1]
            test = (block >= lo) & (block < hi)
            if self.forward:
                train = block < lo - self.embargo
                if k == 0 or not train.any():
                    continue
            else:
                train = (block < lo - self.embargo) | (block >= hi + self.embargo)
            yield np.flatnonzero(train), np.flatnonzero(test)

    def get_n_splits(self, X=None, y=None, groups=None):
        if self.forward and X is not None:
            return sum(1 for _ in self.split(X, y, groups))
        return self.n_splits - 1 if self.forward else self.n_splits
