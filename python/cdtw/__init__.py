"""Fast dynamic time warping for sequences of feature vectors.

Sequences are ``frames x features`` arrays (1-D arrays are treated as one
feature per frame). The DTW cost is the minimum, over all monotone warping
paths from the first to the last frame pair using steps ``(1, 0)``, ``(0, 1)``
and ``(1, 1)``, of the summed frame distances along the path.

``window`` restricts warping paths to a Sakoe-Chiba band ``|i - j| <= r``
with ``r = max(window, |len(a) - len(b)|)``; the widening guarantees that a
path exists.

Inputs that are all ``float32`` are computed in single precision (faster);
anything else is converted to ``float64``. The DP always accumulates in
double precision.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from importlib.metadata import version
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from cdtw import _cdtw

__all__ = ["dtw", "cdist", "pdist"]
__version__ = version("cdtw")

# integer codes are accepted for compatibility with cdtw < 0.3
_METRICS = {"euclidean": 0, "cosine": 1, 0: 0, 1: 1}


def _metric(metric: str | int) -> int:
    # exact types only: 0.0 or True must not silently select a metric
    is_int = isinstance(metric, (int, np.integer)) and not isinstance(metric, (bool, np.bool_))
    if (isinstance(metric, str) or is_int) and metric in _METRICS:
        return _METRICS[metric]
    raise ValueError(f"unknown metric {metric!r}; expected 'euclidean' or 'cosine'")


def _window(window: int | None) -> int | None:
    if window is None:
        return None
    window = operator.index(window)
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    return window


def _n_jobs(n_jobs: int | None) -> int | None:
    if n_jobs is None or n_jobs == -1:
        return None
    n_jobs = operator.index(n_jobs)
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer, -1 or None, got {n_jobs}")
    return n_jobs


def _as_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(f"sequences must be 1-D or 2-D (frames x features), got {arr.ndim}-D")
    return arr


def _coerce(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], bool]:
    """Make all arrays C-contiguous with one dtype; returns (arrays, is_float32)."""
    f32 = bool(arrays) and all(a.dtype == np.float32 for a in arrays)
    dtype = np.float32 if f32 else np.float64
    return [np.ascontiguousarray(a, dtype=dtype) for a in arrays], f32


@overload
def dtw(
    a: ArrayLike,
    b: ArrayLike,
    metric: str = ...,
    window: int | None = ...,
    *,
    return_path: Literal[False] = ...,
    n_jobs: int | None = ...,
) -> float: ...
@overload
def dtw(
    a: ArrayLike,
    b: ArrayLike,
    metric: str = ...,
    window: int | None = ...,
    *,
    return_path: Literal[True],
    n_jobs: int | None = ...,
) -> tuple[float, np.ndarray]: ...
def dtw(
    a: ArrayLike,
    b: ArrayLike,
    metric: str = "euclidean",
    window: int | None = None,
    *,
    return_path: bool = False,
    n_jobs: int | None = 1,
) -> float | tuple[float, np.ndarray]:
    """DTW cost between two sequences.

    :param a: frames x features array (or 1-D array)
    :param b: frames x features array (or 1-D array)
    :param metric: frame distance, ``"euclidean"`` or ``"cosine"``
    :param window: Sakoe-Chiba band radius in frames; ``None`` for no constraint
    :param return_path: also return an optimal warping path
    :param n_jobs: threads for this one pair; ``None`` or ``-1`` uses all
        cores. Only pairs of more than about 260,000 band cells (roughly
        500 x 500 frames) are split; smaller pairs always run on one thread.
        For many pairs, use :func:`cdist` or :func:`pdist` instead.
    :returns: the cost, or ``(cost, path)`` if ``return_path``. ``path`` is an
        ``(L, 2)`` integer array of the ``(i, j)`` frame pairs on the path,
        from ``(0, 0)`` to ``(len(a) - 1, len(b) - 1)``; among equally cheap
        paths, diagonal steps are preferred. It is empty when the cost is
        ``nan``. Computing the path needs one byte per band cell.
    """
    (a, b), f32 = _coerce([_as_array(a), _as_array(b)])
    args = (a, b, _metric(metric), _window(window), _n_jobs(n_jobs))
    if return_path:
        return (_cdtw.dtw_path_f32 if f32 else _cdtw.dtw_path_f64)(*args)
    return (_cdtw.dtw_f32 if f32 else _cdtw.dtw_f64)(*args)


def cdist(
    seqs_a: Iterable[ArrayLike],
    seqs_b: Iterable[ArrayLike],
    metric: str = "euclidean",
    window: int | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """DTW cost between every sequence in ``seqs_a`` and every sequence in ``seqs_b``.

    Sequences may have different lengths but must share the number of features.

    :param n_jobs: number of threads; ``None`` or ``-1`` uses all cores
    :returns: ``(len(seqs_a), len(seqs_b))`` float64 array
    """
    a = [_as_array(s) for s in seqs_a]
    b = [_as_array(s) for s in seqs_b]
    arrays, f32 = _coerce(a + b)
    a, b = arrays[: len(a)], arrays[len(a) :]
    fn = _cdtw.cdist_f32 if f32 else _cdtw.cdist_f64
    return fn(a, b, _metric(metric), _window(window), _n_jobs(n_jobs))


def pdist(
    seqs: Iterable[ArrayLike],
    metric: str = "euclidean",
    window: int | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """DTW cost between every pair of sequences in ``seqs``.

    Returns the condensed form used by :func:`scipy.spatial.distance.pdist`:
    pairs ``(0, 1), (0, 2), ..., (1, 2), ...``; use
    :func:`scipy.spatial.distance.squareform` to get the square matrix.

    :param n_jobs: number of threads; ``None`` or ``-1`` uses all cores
    :returns: float64 array of length ``n * (n - 1) / 2``
    """
    arrays, f32 = _coerce([_as_array(s) for s in seqs])
    fn = _cdtw.pdist_f32 if f32 else _cdtw.pdist_f64
    return fn(arrays, _metric(metric), _window(window), _n_jobs(n_jobs))
