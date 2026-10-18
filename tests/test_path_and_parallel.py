import numpy as np
import pytest

import cdtw

from .reference_dtw import distance_matrix, dtw_path_reference

METRICS = ["euclidean", "cosine"]


# --- return_path ---------------------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("window", [None, 0, 3])
@pytest.mark.parametrize(
    "shape_a,shape_b", [((1, 3), (1, 3)), ((1, 3), (9, 3)), ((9, 3), (1, 3)), ((17, 5), (23, 5)), ((40, 2), (33, 2))]
)
def test_path_matches_reference(metric, window, shape_a, shape_b):
    rng = np.random.default_rng(20)
    XA, XB = rng.standard_normal(shape_a), rng.standard_normal(shape_b)
    cost, path = cdtw.dtw(XA, XB, metric=metric, window=window, return_path=True)
    ref_cost, ref_path = dtw_path_reference(XA, XB, metric=metric, window=window)
    assert cost == pytest.approx(ref_cost)
    np.testing.assert_array_equal(path, ref_path)
    assert cost == cdtw.dtw(XA, XB, metric=metric, window=window)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_path_is_a_valid_warping_path_with_matching_cost(dtype):
    rng = np.random.default_rng(21)
    XA, XB = rng.standard_normal((50, 13)).astype(dtype), rng.standard_normal((61, 13)).astype(dtype)
    cost, path = cdtw.dtw(XA, XB, window=15, return_path=True)
    assert path.dtype == np.intp and path.ndim == 2 and path.shape[1] == 2
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (49, 60)
    steps = {tuple(s) for s in np.diff(path, axis=0)}
    assert steps <= {(0, 1), (1, 0), (1, 1)}
    assert np.all(np.abs(path[:, 0] - path[:, 1]) <= 15)
    D = distance_matrix(XA.astype(np.float64), XB.astype(np.float64))
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    assert D[path[:, 0], path[:, 1]].sum() == pytest.approx(cost, rel=rtol)


def test_path_of_identical_sequences_is_the_diagonal():
    X = np.random.default_rng(22).standard_normal((30, 4))
    cost, path = cdtw.dtw(X, X, return_path=True)
    assert cost == 0.0
    np.testing.assert_array_equal(path, np.stack([np.arange(30)] * 2, axis=1))


def test_path_is_empty_when_cost_is_nan():
    XA = np.array([[0.0, 0.0], [1.0, 2.0]])
    cost, path = cdtw.dtw(XA, XA, metric="cosine", return_path=True)
    assert np.isnan(cost) and path.shape == (0, 2)


def test_return_path_and_n_jobs_are_keyword_only():
    X = np.zeros((3, 2))
    with pytest.raises(TypeError):
        cdtw.dtw(X, X, "euclidean", None, True)


# --- n_jobs for a single pair ----------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("window", [None, 40])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel_single_pair_is_bit_identical(metric, window, dtype):
    # 600 x 650 = 390,000 cells: above the threshold for splitting a pair
    rng = np.random.default_rng(23)
    XA, XB = rng.standard_normal((600, 3)).astype(dtype), rng.standard_normal((650, 3)).astype(dtype)
    one = cdtw.dtw(XA, XB, metric=metric, window=window, n_jobs=1)
    assert cdtw.dtw(XA, XB, metric=metric, window=window, n_jobs=-1) == one
    assert cdtw.dtw(XA, XB, metric=metric, window=window, n_jobs=3) == one
    cost1, path1 = cdtw.dtw(XA, XB, metric=metric, window=window, return_path=True, n_jobs=1)
    costn, pathn = cdtw.dtw(XA, XB, metric=metric, window=window, return_path=True, n_jobs=None)
    assert cost1 == costn == one
    np.testing.assert_array_equal(path1, pathn)


@pytest.mark.parametrize("bad", [0, -2])
def test_dtw_invalid_n_jobs_raises(bad):
    with pytest.raises(ValueError, match="n_jobs"):
        cdtw.dtw(np.zeros((2, 2)), np.ones((2, 2)), n_jobs=bad)
