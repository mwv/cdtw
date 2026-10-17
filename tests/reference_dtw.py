import numpy as np


def dtw_reference(XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean") -> float:
    """Pure NumPy reference DTW implementation, independent of the Cython extension."""
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

    H = np.empty((mA + 1, mB + 1), dtype=np.float64)
    H[1:, 0] = np.finfo(np.float32).max
    H[0, 1:] = np.finfo(np.float32).max
    H[0, 0] = 0.0

    for i in range(1, mA + 1):
        for j in range(1, mB + 1):
            H[i, j] = min(H[i - 1, j], H[i, j - 1], H[i - 1, j - 1]) + D[i - 1, j - 1]

    return float(H[mA, mB])
