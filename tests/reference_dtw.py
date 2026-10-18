import numpy as np


def distance_matrix(XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Frame distances ``D[i, j]`` between ``XA[i]`` and ``XB[j]``."""
    if metric == "euclidean":
        diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    if metric == "cosine":
        norms_a = np.linalg.norm(XA, axis=1)[:, np.newaxis]
        norms_b = np.linalg.norm(XB, axis=1)[np.newaxis, :]
        return 1 - (XA @ XB.T) / (norms_a * norms_b)
    raise ValueError(f"unknown metric: {metric}")


def _cost_matrix(XA, XB, metric, window):
    mA, mB = XA.shape[0], XB.shape[0]
    D = distance_matrix(XA, XB, metric)
    r = max(mA, mB) if window is None else max(window, abs(mA - mB))
    H = np.full((mA + 1, mB + 1), np.inf)
    H[0, 0] = 0.0
    for i in range(1, mA + 1):
        for j in range(max(1, i - r), min(mB, i + r) + 1):
            H[i, j] = min(H[i - 1, j], H[i, j - 1], H[i - 1, j - 1]) + D[i - 1, j - 1]
    return H


def dtw_reference(
    XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean", window: int | None = None
) -> float:
    """Pure NumPy reference DTW implementation, independent of the compiled extension.

    ``window`` is a Sakoe-Chiba radius, widened to the length difference.
    """
    return float(_cost_matrix(XA, XB, metric, window)[-1, -1])


def dtw_path_reference(
    XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean", window: int | None = None
) -> tuple[float, np.ndarray]:
    """Reference cost and warping path, breaking ties like cdtw: diagonal
    first, then up (along ``XA``), and left only when strictly cheaper."""
    H = _cost_matrix(XA, XB, metric, window)
    i, j = H.shape[0] - 1, H.shape[1] - 1
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            up, diag, left = H[i - 1, j], H[i - 1, j - 1], H[i, j - 1]
            if left < min(up, diag):
                j -= 1
            elif diag <= up:
                i, j = i - 1, j - 1
            else:
                i -= 1
        path.append((i - 1, j - 1))
    return float(H[-1, -1]), np.array(path[::-1])
