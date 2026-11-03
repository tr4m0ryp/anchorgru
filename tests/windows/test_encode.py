"""Input encoding: forward fill, relative values, masks, stride, validity gate."""
import numpy as np
import pytest

from anchorgru.windows import encode, fit_scale, forward_fill, last_observed, step_indices

nan = np.nan


def test_forward_fill_carries_values_and_keeps_leading_gaps():
    X = np.array([[[nan], [1.0], [nan], [3.0], [nan]]])
    np.testing.assert_array_equal(forward_fill(X)[0, :, 0], [nan, 1, 1, 3, 3])


def test_relative_values_masks_and_last_step_zero():
    X = np.array([[[nan, 5.0], [2.0, nan], [nan, 7.0], [4.0, nan]]])
    seq, valid = encode(X, min_observed=0.0)
    vals, mask = seq[0, :, :2], seq[0, :, 2:]
    np.testing.assert_array_equal(mask, [[0, 1], [1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(vals[:, 0], [0, -2, -2, 0])
    np.testing.assert_allclose(vals[:, 1], [-2, -2, 0, 0])
    assert valid[0] == 1


def test_absolute_values_when_relative_is_off():
    X = np.array([[[1.0], [nan], [3.0]]])
    seq, _ = encode(X, relative=False)
    np.testing.assert_allclose(seq[0, :, 0], [1, 1, 3])


def test_stride_counts_back_from_the_last_step():
    np.testing.assert_array_equal(step_indices(10, 3), [0, 3, 6, 9])
    np.testing.assert_array_equal(step_indices(7, 1), np.arange(7))
    X = np.arange(10.0)[None, :, None]
    seq, _ = encode(X, stride=3, relative=False)
    np.testing.assert_allclose(seq[0, :, 0], [0, 3, 6, 9])
    with pytest.raises(ValueError):
        step_indices(5, 0)


def test_window_takes_the_last_steps():
    X = np.arange(10.0)[None, :, None]
    seq, _ = encode(X, window=4, relative=False)
    np.testing.assert_allclose(seq[0, :, 0], [6, 7, 8, 9])


def test_validity_gate_threshold():
    X = np.full((3, 8, 1), nan)
    X[0, :2] = 1.0           # 2 / 8 observed: exactly the default threshold
    X[1, :1] = 1.0           # 1 / 8
    seq, valid = encode(X)
    np.testing.assert_array_equal(valid, [1, 0, 0])
    np.testing.assert_array_equal(encode(X, min_observed=0.1)[1], [1, 1, 0])
    assert np.all(np.isfinite(seq))


def test_never_observed_channel_is_zero_with_zero_mask():
    X = np.random.default_rng(0).normal(size=(4, 6, 2))
    X[:, :, 1] = nan
    seq, valid = encode(X)
    assert np.all(seq[:, :, 1] == 0) and np.all(seq[:, :, 3] == 0)
    assert np.all(valid == 1)


def test_last_observed_and_scale():
    X = np.array([[[0.2], [nan]], [[nan], [nan]]])
    np.testing.assert_array_equal(last_observed(X, 0), [0.2, nan])
    seq, valid = encode(np.random.default_rng(0).normal(size=(50, 9, 2)) * [1.0, 10.0])
    s = fit_scale(seq, valid)
    assert s[1] > 5 * s[0]
    seq2, _ = encode(np.random.default_rng(0).normal(size=(50, 9, 2)) * [1.0, 10.0], scale=s)
    np.testing.assert_allclose(fit_scale(seq2, valid), 1.0, rtol=1e-5)


def test_shape_checks():
    assert encode(np.zeros((3, 5)))[0].shape == (3, 5, 2)
    with pytest.raises(ValueError):
        encode(np.zeros((3,)))
