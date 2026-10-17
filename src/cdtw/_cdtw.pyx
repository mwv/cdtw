# cython version of dtw, provides ~10x speedup over python version
# 1. convert to c code:
# $ cython cdtw.pyx
# 2. compile to .so
# gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o cdtw.so cdtw.c

import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef extern from "math.h":
    double pow(double x, double y)
    double sqrt(double x)
    double fabs(double x)

cdef extern from "float.h":
    double FLT_MAX

# @cython.cdivision(True)
@cython.boundscheck(False)
def dtw(np.ndarray[DTYPE_t, ndim=2] XA,
        np.ndarray[DTYPE_t, ndim=2] XB, int metric=0):
    """calculate dtw between two observation x features arrays.

    :param XA: ndarray of samples x features
    :param XB: ndarray of samples x features
    :param metric: switch between euclidean (0) and cosine (1) distances
    """
    cdef unsigned int mA, mB, dim, i, j, k
    cdef double powA, powB, enumerator, denominator
    cdef DTYPE_t d
    cdef DTYPE_t[:,:] H
    cdef DTYPE_t[:,:] D

    mA = XA.shape[0]
    mB = XB.shape[0]
    dim = XA.shape[1]

    D = np.empty([mA, mB], dtype=DTYPE)
    for i in xrange(0, mA):
        for j in xrange(0, mB):
            if metric == 0:
                d = 0.0
                for k in xrange(0, dim):
                    d += pow(XA[i, k] - XB[j, k], 2)
                D[i, j] = sqrt(d)
            elif metric == 1:
                powA = 0.0
                powB = 0.0
                enumerator = 0.0
                for k in xrange(0, dim):
                    powA += pow(XA[i, k], 2)
                    powB += pow(XB[j, k], 2)
                    enumerator += XA[i, k] * XB[j, k]
                denominator = sqrt(powA) * sqrt(powB)
                D[i, j] = 1 - enumerator / denominator

    H = np.empty([mA+1, mB+1], dtype=DTYPE)
    for i in xrange(1, mA+1):
        H[i, 0] = FLT_MAX
    for i in xrange(1, mB+1):
        H[0, i] = FLT_MAX
    H[0, 0] = 0.0

    for i in xrange(1, mA+1):
        for j in xrange(1, mB+1):
            d = D[i-1, j-1]
            H[i,j] = min(H[i-1, j],
                         H[i, j-1],
                         H[i-1, j-1]) + d
    d = H[mA, mB]
    return d
