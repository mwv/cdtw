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


def test_dtw_float32_input_is_coerced():
    rng = np.random.default_rng(4)
    XA = rng.random((5, 4)).astype(np.float32)
    XB = rng.random((6, 4)).astype(np.float32)
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(
        dtw_reference(XA.astype(np.float64), XB.astype(np.float64), metric="euclidean")
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
