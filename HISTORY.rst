.. :changelog:

History
-------

0.3.0 (unreleased)
++++++++++++++++++

* ``dtw(..., return_path=True)`` returns the optimal warping path.
* ``dtw(..., n_jobs=...)`` splits a single large pair across threads
  (about 5x on 8 cores for 4000 x 4400 frames); results are bit-identical.

* Rewritten in Rust (PyO3/maturin), replacing the Cython extension: fused
  distance/DP kernel with explicit SIMD and O(len) memory; roughly 5-20x
  faster single-threaded than 0.2.0.
* New ``cdist`` and ``pdist`` for many pairs at once, multi-threaded
  (``n_jobs``), with the GIL released.
* New ``window`` argument (Sakoe-Chiba band).
* Metrics are named (``"euclidean"``, ``"cosine"``); ``0``/``1`` still work.
* ``float32`` inputs are computed in single precision; 1-D inputs accepted.
* Cosine distance involving a sequence with an all-zero frame is now defined
  to be ``nan``.
* Added ``bench/bench.py`` benchmark toolkit.

0.2.0 (2026-10-17)
++++++++++++++++++

* Rewritten for Python 3: modernized Cython extension (typed memoryviews),
  pyproject.toml/uv/hatchling packaging, pytest test suite. Dropped Python 2
  support.

0.1.0 (2014-01-16)
++++++++++++++++++

* First release on PyPI.