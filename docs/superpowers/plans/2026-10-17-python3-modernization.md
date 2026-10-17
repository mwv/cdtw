# cdtw Python 3 Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `cdtw` (a 2014-era Python 2 / Cython DTW distance package) up to a runnable, maintainable, PyPI-publishable state on current Python, tooling, and packaging standards.

**Architecture:** Restructure the flat `src/cdtw.pyx` script into a proper `src/cdtw/` package with a modernized Cython extension (`_cdtw.pyx`, using typed memoryviews instead of the legacy `cimport numpy` buffer syntax) wrapped by a thin pure-Python `__init__.py` that exposes `dtw(...)`. Replace `setup.py`/`distutils` with a `pyproject.toml` built by `hatchling`, using the `hatch-cython` build hook to compile the extension (uv-managed environment/dependency resolution). Add a `pytest` suite that verifies the extension against a pure-Python/NumPy reference DTW implementation, plus edge cases. Add GitHub Actions CI building/testing on Python 3.10–3.13.

**Tech Stack:** Python 3.10+, Cython (modernized, memoryview-based), uv, hatchling + hatch-cython build hook, pytest, NumPy.

**Spec:** No separate spec document; the spec is the conversation decisions below (derived from user Q&A on 2026-10-17):
- Performance backend: modernized Cython (not numba, not pure-NumPy).
- Packaging: uv + hatchling, with a custom build hook for the Cython extension (hatchling has no native Cython support).
- Scope: target Python 3.10+, keep the package PyPI-publishable (proper metadata, classifiers, long description).

## Global Constraints

- Minimum supported Python version: 3.10.
- Build backend: `hatchling`, dependency/env management: `uv`.
- Cython extension must use typed memoryviews (`double[:, ::1]`), not the legacy `cimport numpy` / `np.ndarray[DTYPE_t, ndim=2]` buffer syntax, to avoid NumPy C-API ABI coupling.
- Public API must stay `dtw(XA, XB, metric=0)` with `metric` accepting `0` (euclidean) or `1` (cosine) — same contract as today — so existing callers aren't broken by the rewrite.
- No `setup.py`, `distutils`, or Python 2 syntax (`xrange`, old `print`, etc.) anywhere in the final tree.
- License stays GPLv3; author/package metadata carries over from the existing `setup.py`.

## Review Focus

- Mismatched-dimension inputs (`XA.shape[1] != XB.shape[1]`): current code has no check and will silently read out of bounds or produce garbage — a test should pin the expected behavior (raise `ValueError`).
- Empty-sequence inputs (`XA` or `XB` with 0 rows): current loop bounds (`mA`/`mB` = 0) combined with the `FLT_MAX` border initialization — needs a test confirming it either raises or returns a defined value, not a crash.
- Non-contiguous / non-`float64` input arrays (e.g. a transposed view, or `float32` array): memoryview typing (`double[:, ::1]`) will raise on mismatched dtype/stride — needs a test showing the actual (not assumed) error, and a decision on whether `__init__.py` should coerce with `np.ascontiguousarray(..., dtype=np.float64)` before calling the extension.
- Unknown `metric` value (anything other than 0/1): current Cython code falls through silently, leaving `D[i,j]` uninitialized (garbage memory) for every cell — needs a test pinning a raised `ValueError` for invalid metric.
- Single-sample sequences (`mA == 1` or `mB == 1`, i.e. degenerate 1xN DTW): boundary-row/column initialization path — needs a test confirming it still returns the correct scalar distance.

---

### Task 1: Package layout and pyproject/uv scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/cdtw/__init__.py`
- Create: `src/cdtw/_cdtw.pyx` (modernized copy of `src/cdtw.pyx`, syntax fixes only — see Task 2 for the memoryview rewrite)
- Delete: `setup.py`, `requirements.txt`, `MANIFEST.in`, `src/cdtw.pyx`
- Modify: `.gitignore` (add `*.so`, `*.c`, `.venv/`, `dist/`, `build/`, `*.egg-info/`)

**Interfaces:**
- Produces: `cdtw` importable as a package (`import cdtw; cdtw.dtw(XA, XB, metric=0)`), installable via `uv sync` / `uv pip install -e .`.

- [ ] **Step 1: Write `pyproject.toml`**

Project metadata carried over from `setup.py` (`name="cdtw"`, `version="0.2.0"`, `description="Fast dtw in python/cython."`, author/email from `setup.py`, `license = "GPL-3.0-only"`, `requires-python = ">=3.10"`, `dependencies = ["numpy"]`, classifiers updated to `Programming Language :: Python :: 3.10/3.11/3.12/3.13`). `[build-system]` requires `hatchling`, `hatch-cython`, `numpy`; `build-backend = "hatchling.build"`. `[tool.hatch.build.hooks.cython]` section pointing at `src/cdtw/_cdtw.pyx` (exact hook config keys come from `hatch-cython`'s own README — read it when writing this section, since the plan's role is to name the tool, not transcribe its config schema).

- [ ] **Step 2: Move and rename the extension source**

`git mv src/cdtw.pyx src/cdtw/_cdtw.pyx` (create `src/cdtw/` first). Leave content untouched for this step — Task 2 handles the rewrite.

- [ ] **Step 3: Write `src/cdtw/__init__.py`**

```python
from cdtw._cdtw import dtw

__all__ = ["dtw"]
```

- [ ] **Step 4: Delete legacy build files**

`git rm setup.py requirements.txt MANIFEST.in`

- [ ] **Step 5: Update `.gitignore`**

Add `*.so`, `*.c`, `.venv/`, `dist/`, `build/`, `*.egg-info/` if not already present.

- [ ] **Step 6: Verify the package builds and imports**

Run: `uv sync && uv run python -c "import cdtw; print(cdtw.dtw)"`
Expected: prints the function object, no build errors (the `.pyx` still has Python 2 syntax at this point, so Cython may warn but should still compile — if it hard-fails, note the exact error for Task 2 rather than patching here).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/cdtw .gitignore
git commit -m "build: replace setup.py/distutils with pyproject.toml + uv/hatchling"
```

---

### Task 2: Modernize the Cython extension

**Files:**
- Modify: `src/cdtw/_cdtw.pyx`

**Interfaces:**
- Consumes: nothing new — same two-array-plus-metric contract as the original.
- Produces: `dtw(XA: double[:, ::1], XB: double[:, ::1], metric: int = 0) -> float`, raising `ValueError` on dimension mismatch and on unknown `metric` (see Review Focus).

- [ ] **Step 1: Rewrite the typing and loop syntax**

Replace `cimport numpy as np` / `np.ndarray[DTYPE_t, ndim=2]` arguments with typed memoryviews: `def dtw(double[:, ::1] XA, double[:, ::1] XB, int metric=0)`. Replace `xrange` with `range`. Keep `cdef extern from "math.h"` block (still valid Cython 3). Drop the unused `DTYPE`/`DTYPE_t` aliases now that memoryviews are typed directly as `double`.

- [ ] **Step 2: Add input validation**

At the top of `dtw`, after reading `mA, mB, dim`: if `XB.shape[1] != dim`, `raise ValueError("XA and XB must have the same number of features")`. If `metric not in (0, 1)`, `raise ValueError(f"unknown metric: {metric}")`. These are plain Python-level checks (not `cdef`-timed), so they run before the typed loops.

- [ ] **Step 3: Verify it still compiles and runs**

Run: `uv sync && uv run python -c "import numpy as np, cdtw; a=np.random.rand(5,3); b=np.random.rand(7,3); print(cdtw.dtw(a,b))"`
Expected: a float prints, no compiler warnings about deprecated NumPy API usage.

- [ ] **Step 4: Commit**

```bash
git add src/cdtw/_cdtw.pyx
git commit -m "refactor: modernize cdtw Cython extension to Python 3 / memoryviews"
```

---

### Task 3: Reference implementation and correctness tests

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/reference_dtw.py`
- Create: `tests/test_dtw.py`

**Interfaces:**
- Consumes: `cdtw.dtw` from Task 2.
- Produces: `reference_dtw.dtw_reference(XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean") -> float`, used only by tests.

- [ ] **Step 1: Write the pure-Python/NumPy reference implementation**

`tests/reference_dtw.py` — a straightforward (non-optimized) NumPy transliteration of the same algorithm (pairwise distance matrix via euclidean or cosine, then the `H` DP matrix with the same `FLT_MAX` border). This is the oracle the Cython extension is checked against, so it must be independent code, not a copy-paste of the `.pyx` logic.

- [ ] **Step 2: Write the failing correctness tests**

```python
import numpy as np
import pytest
import cdtw
from .reference_dtw import dtw_reference

@pytest.mark.parametrize("metric_int,metric_str", [(0, "euclidean"), (1, "cosine")])
def test_dtw_matches_reference(metric_int, metric_str):
    rng = np.random.default_rng(0)
    XA = rng.random((10, 4))
    XB = rng.random((13, 4))
    assert cdtw.dtw(XA, XB, metric=metric_int) == pytest.approx(
        dtw_reference(XA, XB, metric=metric_str)
    )

def test_dtw_single_sample_sequences():
    XA = np.array([[1.0, 2.0, 3.0]])
    XB = np.array([[4.0, 5.0, 6.0]])
    expected = dtw_reference(XA, XB, metric="euclidean")
    assert cdtw.dtw(XA, XB, metric=0) == pytest.approx(expected)

def test_dtw_mismatched_feature_dims_raises():
    XA = np.zeros((3, 2))
    XB = np.zeros((3, 5))
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB)

def test_dtw_unknown_metric_raises():
    XA = np.zeros((3, 2))
    XB = np.zeros((4, 2))
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB, metric=99)

def test_dtw_non_contiguous_input_raises_or_is_coerced():
    XA = np.zeros((5, 4))[:, ::2]  # non-contiguous view
    XB = np.zeros((5, 2))
    with pytest.raises(ValueError):
        cdtw.dtw(XA, XB)
```

- [ ] **Step 3: Run tests to verify they fail/pass appropriately before `__init__.py` coercion is decided**

Run: `uv run pytest tests/ -v`
Expected: the first four tests pass against Task 2's implementation; `test_dtw_non_contiguous_input_raises_or_is_coerced` tells you whether memoryview typing already raises `TypeError`/`ValueError` on a non-contiguous view. If it raises something other than `ValueError`, change the test's expected exception type to match reality rather than forcing a coercion layer — this task is pinning actual behavior, not adding new conversion logic.

- [ ] **Step 4: Adjust the test's expected exception type if needed, then re-run**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: add pytest suite with NumPy reference oracle for cdtw.dtw"
```

---

### Task 4: PyPI-ready packaging metadata and docs

**Files:**
- Modify: `README.rst`
- Modify: `HISTORY.rst`
- Modify: `pyproject.toml` (from Task 1)
- Delete: `AUTHORS.rst`, `CONTRIBUTING.rst` only if their content is folded into `README.rst`; otherwise keep and just drop their now-stale `setup.py`/py2 references

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks — this is the terminal packaging-polish task.

- [ ] **Step 1: Update `README.rst`**

Add a Quickstart section: `uv add cdtw` (or `pip install cdtw`), a short usage example calling `cdtw.dtw`, and the metric parameter meaning (0=euclidean, 1=cosine). Keep the GPLv3 line.

- [ ] **Step 2: Add a `0.2.0` entry to `HISTORY.rst`**

One line: Python 3 / modern packaging rewrite, dropped Python 2 support.

- [ ] **Step 3: Add `[project.urls]` and `readme`/`license-files` keys to `pyproject.toml`**

`readme = "README.rst"`, `[project.urls] Homepage = "https://github.com/mwv/cdtw"`.

- [ ] **Step 4: Build an sdist and wheel locally to verify packaging**

Run: `uv build`
Expected: `dist/cdtw-0.2.0.tar.gz` and a platform wheel are produced without errors.

- [ ] **Step 5: Commit**

```bash
git add README.rst HISTORY.rst pyproject.toml
git commit -m "docs: update README/HISTORY for Python 3 release, verify uv build"
```

---

### Task 5: CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Delete: `Makefile` (its targets — `tox`, `sphinx-apidoc`, `python setup.py test` — are all dead references to tooling this project no longer uses)

**Interfaces:**
- Consumes: `uv run pytest tests/` from Task 3, `uv build` from Task 4.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

Matrix over `python-version: ["3.10", "3.11", "3.12", "3.13"]`, using `astral-sh/setup-uv`, steps: `uv sync`, `uv run pytest tests/ -v`, `uv build`.

- [ ] **Step 2: Delete the Makefile**

`git rm Makefile` — its targets reference `tox`, `coverage run --source cdtw setup.py test`, and `sphinx-apidoc`, none of which this plan sets up; keeping it would be a stale, misleading entry point.

- [ ] **Step 3: Push and verify the workflow run is green**

Run: `git push` (on a branch) and check the Actions tab, or run `act` locally if available.
Expected: all matrix jobs pass.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions matrix build/test across Python 3.10-3.13"
```
