===============================
cdtw
===============================

Fast dtw in python/cython.

* Free software: GPLv3 license

Quickstart
----------

Install with uv::

    $ uv add cdtw

or pip::

    $ pip install cdtw

Usage::

    import numpy as np
    import cdtw

    XA = np.random.rand(10, 4)  # 10 samples x 4 features
    XB = np.random.rand(13, 4)  # 13 samples x 4 features

    cdtw.dtw(XA, XB, metric=0)  # euclidean distance
    cdtw.dtw(XA, XB, metric=1)  # cosine distance
