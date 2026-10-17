# cython version of dtw, provides ~10x speedup over python version

import numpy as np
cimport cython

cdef extern from "math.h":
    double pow(double x, double y)
    double sqrt(double x)
    double fabs(double x)

cdef extern from "float.h":
    double FLT_MAX

@cython.boundscheck(False)
def dtw(double[:, ::1] XA, double[:, ::1] XB, int metric=0):
    """calculate dtw between two observation x features arrays.

    :param XA: ndarray of samples x features
    :param XB: ndarray of samples x features
    :param metric: switch between euclidean (0) and cosine (1) distances
    """
    cdef unsigned int mA, mB, dim, i, j, k
    cdef double powA, powB, enumerator, denominator
    cdef double d
    cdef double[:, :] H
    cdef double[:, :] D

    mA = XA.shape[0]
    mB = XB.shape[0]
    dim = XA.shape[1]

    if XB.shape[1] != dim:
        raise ValueError("XA and XB must have the same number of features")
    if metric not in (0, 1):
        raise ValueError(f"unknown metric: {metric}")

    D = np.empty([mA, mB], dtype=np.double)
    for i in range(0, mA):
        for j in range(0, mB):
            if metric == 0:
                d = 0.0
                for k in range(0, dim):
                    d += pow(XA[i, k] - XB[j, k], 2)
                D[i, j] = sqrt(d)
            elif metric == 1:
                powA = 0.0
                powB = 0.0
                enumerator = 0.0
                for k in range(0, dim):
                    powA += pow(XA[i, k], 2)
                    powB += pow(XB[j, k], 2)
                    enumerator += XA[i, k] * XB[j, k]
                denominator = sqrt(powA) * sqrt(powB)
                D[i, j] = 1 - enumerator / denominator

    H = np.empty([mA+1, mB+1], dtype=np.double)
    for i in range(1, mA+1):
        H[i, 0] = FLT_MAX
    for i in range(1, mB+1):
        H[0, i] = FLT_MAX
    H[0, 0] = 0.0

    for i in range(1, mA+1):
        for j in range(1, mB+1):
            d = D[i-1, j-1]
            H[i, j] = min(H[i-1, j],
                          H[i, j-1],
                          H[i-1, j-1]) + d
    d = H[mA, mB]
    return d
