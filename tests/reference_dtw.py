import numpy as np


def dtw_reference(
    XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean", window: int | None = None
) -> float:
    """Pure NumPy reference DTW implementation, independent of the compiled extension.

    ``window`` is a Sakoe-Chiba radius, widened to the length difference.
    """
    mA = XA.shape[0]
    mB = XB.shape[0]

    if metric == "euclidean":
        diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=2))
    elif metric == "cosine":
        norms_a = np.linalg.norm(XA, axis=1)[:, np.newaxis]
        norms_b = np.linalg.norm(XB, axis=1)[np.newaxis, :]
        dot = XA @ XB.T
        D = 1 - dot / (norms_a * norms_b)
    else:
        raise ValueError(f"unknown metric: {metric}")

    r = max(mA, mB) if window is None else max(window, abs(mA - mB))
    H = np.full((mA + 1, mB + 1), np.inf)
    H[0, 0] = 0.0

    for i in range(1, mA + 1):
        for j in range(max(1, i - r), min(mB, i + r) + 1):
            H[i, j] = min(H[i - 1, j], H[i, j - 1], H[i - 1, j - 1]) + D[i - 1, j - 1]

    return float(H[mA, mB])
