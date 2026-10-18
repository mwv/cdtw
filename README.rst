===============================
cdtw
===============================

Fast dynamic time warping for speech and other feature sequences.

* Rust core with explicit SIMD (NEON, SSE2, AVX2+FMA), multi-threaded
  ``cdist``/``pdist``, no runtime dependencies beyond NumPy.
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

    XA = np.random.rand(10, 4)  # 10 frames x 4 features
    XB = np.random.rand(13, 4)  # 13 frames x 4 features

    cdtw.dtw(XA, XB)                       # euclidean frame distance
    cdtw.dtw(XA, XB, metric="cosine")      # cosine frame distance
    cdtw.dtw(XA, XB, window=3)             # Sakoe-Chiba band

    # the optimal warping path: (L, 2) array of (i, j) frame pairs
    cost, path = cdtw.dtw(XA, XB, return_path=True)

    # one long pair on all cores (pairs under ~500 x 500 frames stay single-threaded)
    cdtw.dtw(np.random.rand(5000, 39), np.random.rand(5200, 39), n_jobs=-1)

    # many pairs at once, on all cores (n_jobs=1 for a single thread)
    segments = [np.random.rand(n, 39) for n in (20, 35, 50, 80)]
    D = cdtw.cdist(segments, segments, metric="cosine")  # 4 x 4 matrix
    d = cdtw.pdist(segments)  # condensed, as scipy.spatial.distance.pdist

Semantics
---------

* The cost is the sum of frame distances along the cheapest monotone warping
  path (steps ``(1, 0)``, ``(0, 1)``, ``(1, 1)``), not normalised by length.
* ``window=r`` restricts paths to ``|i - j| <= max(r, |len(a) - len(b)|)``.
* Inputs that are all ``float32`` are computed in single precision, roughly
  twice as fast; anything else is computed in ``float64``. Accumulation is
  always in ``float64``.
* 1-D arrays are treated as one feature per frame.
* With ``metric="cosine"``, a sequence containing an all-zero frame has
  distance ``nan`` to everything (and an empty path).
* Among equally cheap warping paths, ``return_path`` prefers diagonal steps.
  It needs one byte of memory per cell inside the band, e.g. 1 MB for
  1000 x 1000 frames without a window.
* Single pairs and paths are bit-identical whatever ``n_jobs`` is.
* ``metric=0`` / ``metric=1`` are accepted for compatibility with cdtw < 0.3.

Benchmarks
----------

``bench/bench.py`` runs deterministic benchmark suites and compares saved
runs::

    $ uv run python bench/bench.py run --suite quick --out bench/results/before.json
    # ... make changes, rebuild with `uv sync` ...
    $ uv run python bench/bench.py run --suite quick --out bench/results/after.json
    $ uv run python bench/bench.py compare bench/results/before.json bench/results/after.json

Use ``--suite full`` for the complete matrix, ``--filter`` to select
scenarios, and ``--libs cdtw,dtaidistance`` to include other libraries.
``cargo run --release -p cdtw-core --example profile`` runs a single pair in
a tight loop for profilers.
