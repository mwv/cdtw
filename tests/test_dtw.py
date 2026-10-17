import numpy as np
import pytest
import cdtw
from .reference_dtw import dtw_reference


@pytest.mark.parametrize("metric_int,metric_str", [(0, "euclidean"), (1, "cosine")])
def test_dtw_matches_reference(metric_int, metric_str):
    rng = np.random.default_rng(0)
    XA = rng.random((10, 4))
    XB = rng.random((13, 4))
    assert cdtw.dtw(XA, XB, metric=metric_int) == pytest.approx(
        dtw_reference(XA, XB, metric=metric_str)
    )


def test_dtw_single_sample_sequences():
    XA = np.array([[1.0, 2.0, 3.0]])
    XB = np.array([[4.0, 5.0, 6.0]])
    expected = dtw_reference(XA, XB, metric="euclidean")
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(expected)


def test_dtw_mismatched_feature_dims_raises():
    XA = np.zeros((3, 2))
    XB = np.zeros((3, 5))
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB)


def test_dtw_unknown_metric_raises():
    XA = np.zeros((3, 2))
    XB = np.zeros((4, 2))
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB, metric=99)


def test_dtw_non_contiguous_input_is_coerced():
    rng = np.random.default_rng(1)
    XA = rng.random((5, 8))[:, ::2]  # non-contiguous view, shape (5, 4)
    XB = rng.random((7, 4))
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(
        dtw_reference(XA, XB, metric="euclidean")
    )


def test_dtw_transposed_input_is_coerced():
    rng = np.random.default_rng(2)
    XA = rng.random((4, 10)).T  # feature-major, transposed to samples x features
    XB = rng.random((4, 13)).T
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(
        dtw_reference(XA, XB, metric="euclidean")
    )


def test_dtw_read_only_input_is_accepted():
    rng = np.random.default_rng(3)
    XA = rng.random((5, 4))
    XB = rng.random((6, 4))
    XA.flags.writeable = False
    XB.flags.writeable = False
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(
        dtw_reference(XA, XB, metric="euclidean")
    )


def test_dtw_float32_input_is_computed_in_single_precision():
    rng = np.random.default_rng(4)
    XA = rng.random((5, 4)).astype(np.float32)
    XB = rng.random((6, 4)).astype(np.float32)
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(
        dtw_reference(XA.astype(np.float64), XB.astype(np.float64), metric="euclidean"), rel=1e-5
    )


@pytest.mark.parametrize("shape_a,shape_b", [((1, 1), (1, 1)), ((1, 3), (5, 3)), ((5, 3), (1, 3))])
@pytest.mark.parametrize("metric_int,metric_str", [(0, "euclidean"), (1, "cosine")])
def test_dtw_single_row_boundary(shape_a, shape_b, metric_int, metric_str):
    rng = np.random.default_rng(5)
    XA = rng.random(shape_a)
    XB = rng.random(shape_b)
    assert cdtw.dtw(XA, XB, metric=metric_int) == pytest.approx(
        dtw_reference(XA, XB, metric=metric_str)
    )


@pytest.mark.parametrize("shape_a,shape_b", [((0, 3), (5, 3)), ((5, 3), (0, 3)), ((0, 3), (0, 3))])
def test_dtw_empty_sequence_raises(shape_a, shape_b):
    XA = np.zeros(shape_a)
    XB = np.zeros(shape_b)
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB)


def test_dtw_cosine_metric_all_zero_row_returns_nan_instead_of_raising():
    XA = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    XB = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    result = cdtw.dtw(XA, XB, metric=1)
    assert np.isnan(result)


# --- string metrics, windows, 1-D input ---------------------------------------

METRICS = ["euclidean", "cosine"]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("window", [None, 0, 1, 5, 1000])
@pytest.mark.parametrize("shape_a,shape_b", [((30, 13), (37, 13)), ((9, 3), (8, 3)), ((1, 2), (20, 2))])
def test_dtw_window_matches_reference(metric, window, shape_a, shape_b):
    rng = np.random.default_rng(6)
    XA = rng.standard_normal(shape_a)
    XB = rng.standard_normal(shape_b)
    assert cdtw.dtw(XA, XB, metric=metric, window=window) == pytest.approx(
        dtw_reference(XA, XB, metric=metric, window=window)
    )


def test_dtw_window_never_increases_cost_when_widened():
    rng = np.random.default_rng(7)
    XA, XB = rng.standard_normal((40, 5)), rng.standard_normal((45, 5))
    costs = [cdtw.dtw(XA, XB, window=w) for w in (0, 2, 8, 32, None)]
    assert costs == sorted(costs, reverse=True)


def test_dtw_identical_sequences_cost_zero():
    X = np.random.default_rng(8).standard_normal((50, 13))
    assert cdtw.dtw(X, X) == 0.0


def test_dtw_accepts_1d_sequences():
    a = np.array([0.0, 1.0, 2.0, 1.0])
    b = np.array([0.0, 2.0, 1.0])
    assert cdtw.dtw(a, b) == pytest.approx(dtw_reference(a[:, None], b[:, None]))
    assert cdtw.dtw([0, 1, 2, 1], [0, 2, 1]) == pytest.approx(cdtw.dtw(a, b))


def test_dtw_mixed_dtypes_use_double_precision():
    rng = np.random.default_rng(9)
    XA = rng.random((5, 4)).astype(np.float32)
    XB = rng.random((6, 4))
    assert cdtw.dtw(XA, XB) == pytest.approx(dtw_reference(XA.astype(np.float64), XB), rel=1e-12)


@pytest.mark.parametrize("bad", ["manhattan", 2, -1, True, None, 0.0])
def test_unknown_metric_raises(bad):
    with pytest.raises(ValueError, match="metric"):
        cdtw.dtw(np.zeros((2, 2)), np.ones((2, 2)), metric=bad)


def test_negative_window_raises():
    with pytest.raises(ValueError, match="window"):
        cdtw.dtw(np.zeros((2, 2)), np.ones((2, 2)), window=-1)


def test_3d_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        cdtw.dtw(np.zeros((2, 2, 2)), np.ones((2, 2)))


# --- cdist / pdist -------------------------------------------------------------


def _random_seqs(rng, n, dim, dtype=np.float64):
    return [rng.standard_normal((int(rng.integers(1, 40)), dim)).astype(dtype) for _ in range(n)]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("n_jobs", [None, 1, 2, -1])
def test_cdist_matches_pairwise_dtw(metric, window, n_jobs):
    rng = np.random.default_rng(10)
    A, B = _random_seqs(rng, 7, 6), _random_seqs(rng, 5, 6)
    D = cdtw.cdist(A, B, metric=metric, window=window, n_jobs=n_jobs)
    assert D.shape == (7, 5) and D.dtype == np.float64
    expected = [[cdtw.dtw(a, b, metric=metric, window=window) for b in B] for a in A]
    np.testing.assert_array_equal(D, expected)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("n_jobs", [None, 1, 3])
def test_pdist_is_condensed_upper_triangle(metric, n_jobs):
    rng = np.random.default_rng(11)
    S = _random_seqs(rng, 9, 4)
    d = cdtw.pdist(S, metric=metric, n_jobs=n_jobs)
    expected = [cdtw.dtw(S[i], S[j], metric=metric) for i in range(9) for j in range(i + 1, 9)]
    np.testing.assert_array_equal(d, expected)


def test_pdist_matches_cdist_upper_triangle():
    rng = np.random.default_rng(12)
    S = _random_seqs(rng, 6, 3)
    D = cdtw.cdist(S, S)
    np.testing.assert_allclose(cdtw.pdist(S), D[np.triu_indices(6, k=1)], rtol=1e-12)


def test_cdist_float32():
    rng = np.random.default_rng(13)
    A, B = _random_seqs(rng, 4, 8, np.float32), _random_seqs(rng, 3, 8, np.float32)
    D = cdtw.cdist(A, B)
    assert D.dtype == np.float64
    expected = [[dtw_reference(a.astype(np.float64), b.astype(np.float64)) for b in B] for a in A]
    np.testing.assert_allclose(D, expected, rtol=1e-5)


def test_cdist_accepts_generators_and_3d_arrays():
    X = np.random.default_rng(14).standard_normal((4, 10, 3))
    D = cdtw.cdist((x for x in X), X)
    assert D.shape == (4, 4)
    np.testing.assert_allclose(np.diag(D), 0.0)


def test_cdist_pdist_empty_collections():
    assert cdtw.cdist([], [np.zeros((2, 2))]).shape == (0, 1)
    assert cdtw.pdist([]).shape == (0,)
    assert cdtw.pdist([np.zeros((2, 2))]).shape == (0,)


def test_cdist_dimension_mismatch_names_the_sequence():
    with pytest.raises(ValueError, match=r"seqs_b\[1\]"):
        cdtw.cdist([np.zeros((3, 2))], [np.zeros((3, 2)), np.zeros((3, 4))])


def test_pdist_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty"):
        cdtw.pdist([np.zeros((3, 2)), np.zeros((0, 2))])


@pytest.mark.parametrize("bad", [0, -2])
def test_invalid_n_jobs_raises(bad):
    with pytest.raises(ValueError, match="n_jobs"):
        cdtw.cdist([np.zeros((2, 2))], [np.zeros((2, 2))], n_jobs=bad)


def test_cosine_zero_frame_is_nan_in_cdist():
    A = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 2.0]])]
    D = cdtw.cdist(A, A, metric="cosine")
    assert np.isnan(D[0]).all() and np.isnan(D[:, 0]).all()
    assert D[1, 1] == pytest.approx(0.0, abs=1e-12)
