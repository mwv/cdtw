import numpy as np

from cdtw._cdtw import dtw as _dtw

__all__ = ["dtw"]


def dtw(XA, XB, metric=0):
    """calculate dtw between two observation x features arrays.

    Accepts any array-like input; non-contiguous, non-float64, or
    non-writable arrays are coerced to a contiguous float64 copy first.

    :param XA: array-like of samples x features
    :param XB: array-like of samples x features
    :param metric: switch between euclidean (0) and cosine (1) distances
    """
    XA = np.ascontiguousarray(XA, dtype=np.float64)
    XB = np.ascontiguousarray(XB, dtype=np.float64)
    return _dtw(XA, XB, metric)
