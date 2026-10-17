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


def test_dtw_non_contiguous_input_raises_or_is_coerced():
    XA = np.zeros((5, 4))[:, ::2]  # non-contiguous view
    XB = np.zeros((5, 2))
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB)
