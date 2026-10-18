"""Benchmark toolkit for cdtw.

Run a suite and save the results::

    uv run python bench/bench.py run --suite quick --out bench/results/my-change.json

Compare two saved runs (speedup = baseline time / candidate time)::

    uv run python bench/bench.py compare bench/results/baseline.json bench/results/my-change.json

Scenarios are deterministic (fixed seeds), so results from different commits
on the same machine are directly comparable. Each scenario is timed with
``repeats`` independent measurements; each measurement runs the call as many
times as needed to fill ``--min-time`` seconds. The median per-call time is
the headline number; ``cells/s`` is the number of DP cells (inside the
window) evaluated per second, which normalises across sequence lengths.

The script also runs against older cdtw versions (integer metrics, no
``cdist``/``pdist``/``window``): unsupported scenarios are skipped and
``cdist``/``pdist`` are emulated with a Python loop over ``dtw``.

Optional competitor libraries can be included with ``--libs``. They compute
slightly different quantities (e.g. dtaidistance sums squared distances and
takes the square root at the end) so only their speed is comparable.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


@dataclasses.dataclass(frozen=True)
class Scenario:
    kind: str  # "dtw", "cdist" or "pdist"
    metric: str
    dim: int
    dtype: str = "float64"
    window: int | None = None
    # cdist/pdist: None = all cores; dtw: None = the default (one thread), -1 = all cores
    n_jobs: int | None = None
    # dtw: also compute the warping path
    path: bool = False
    # dtw: lengths of the two sequences
    len_a: int = 0
    len_b: int = 0
    # cdist/pdist: number of sequences and their length range
    n_a: int = 0
    n_b: int = 0
    min_len: int = 0
    max_len: int = 0

    @property
    def id(self) -> str:
        if self.kind == "dtw":
            shape = f"{self.len_a}x{self.len_b}"
        elif self.kind == "cdist":
            shape = f"{self.n_a}x{self.n_b}[{self.min_len}-{self.max_len}]"
        else:
            shape = f"{self.n_a}[{self.min_len}-{self.max_len}]"
        parts = [self.kind, shape, f"d{self.dim}", self.metric, self.dtype]
        if self.window is not None:
            parts.append(f"w{self.window}")
        if self.n_jobs is not None:
            parts.append(f"j{self.n_jobs}")
        if self.path:
            parts.append("path")
        return "/".join(parts)


def _suite_quick() -> list[Scenario]:
    s = []
    for n, dim in [(50, 13), (200, 39), (1000, 39), (1000, 768)]:
        for metric in ("euclidean", "cosine"):
            s.append(Scenario("dtw", metric, dim, len_a=n, len_b=n + n // 10))
    s.append(Scenario("dtw", "euclidean", 39, len_a=1000, len_b=1100, dtype="float32"))
    s.append(Scenario("dtw", "euclidean", 39, len_a=1000, len_b=1100, window=50))
    s.append(Scenario("dtw", "euclidean", 39, len_a=1000, len_b=1100, path=True))
    # one long pair, split across cores
    s.append(Scenario("dtw", "euclidean", 39, len_a=4000, len_b=4400))
    s.append(Scenario("dtw", "euclidean", 39, len_a=4000, len_b=4400, n_jobs=-1))
    # word-sized speech segments, as in ABX / query-by-example evaluation
    for n_jobs in (1, None):
        s.append(Scenario("cdist", "cosine", 39, n_a=100, n_b=100, min_len=20, max_len=80, n_jobs=n_jobs))
    s.append(Scenario("pdist", "euclidean", 39, n_a=150, min_len=20, max_len=80))
    return s


def _suite_full() -> list[Scenario]:
    s = []
    for n in (20, 50, 200, 1000, 4000):
        for dim in (1, 13, 39, 80, 768):
            if n * n * dim > 4000 * 4000 * 80:
                continue
            for metric in ("euclidean", "cosine"):
                for dtype in ("float64", "float32"):
                    s.append(Scenario("dtw", metric, dim, dtype=dtype, len_a=n, len_b=n + n // 10))
    for window in (10, 50, 200):
        s.append(Scenario("dtw", "euclidean", 39, len_a=4000, len_b=4400, window=window))
    for n_jobs in (2, 4, -1):
        s.append(Scenario("dtw", "euclidean", 39, len_a=4000, len_b=4400, n_jobs=n_jobs))
        s.append(Scenario("dtw", "cosine", 39, dtype="float32", len_a=4000, len_b=4400, n_jobs=n_jobs))
        s.append(Scenario("dtw", "euclidean", 39, len_a=4000, len_b=4400, window=200, n_jobs=n_jobs))
    s.append(Scenario("dtw", "euclidean", 39, len_a=1000, len_b=1100, n_jobs=-1))
    for n, dim in [(200, 39), (1000, 39), (4000, 39), (1000, 768)]:
        s.append(Scenario("dtw", "euclidean", dim, len_a=n, len_b=n + n // 10, path=True))
    s.append(Scenario("dtw", "euclidean", 39, len_a=4000, len_b=4400, path=True, n_jobs=-1))
    for n_jobs in (1, 2, 4, None):
        s.append(Scenario("cdist", "cosine", 39, n_a=200, n_b=200, min_len=20, max_len=80, n_jobs=n_jobs))
    for dim in (13, 80, 768):
        s.append(Scenario("cdist", "cosine", dim, n_a=100, n_b=100, min_len=20, max_len=80))
    s.append(Scenario("cdist", "euclidean", 39, n_a=100, n_b=100, min_len=20, max_len=80, window=10))
    s.append(Scenario("cdist", "euclidean", 39, n_a=50, n_b=50, min_len=200, max_len=600, dtype="float32"))
    s.append(Scenario("pdist", "euclidean", 39, n_a=300, min_len=20, max_len=80))
    s.append(Scenario("pdist", "cosine", 80, n_a=300, min_len=20, max_len=80, dtype="float32"))
    return s


SUITES: dict[str, Callable[[], list[Scenario]]] = {"quick": _suite_quick, "full": _suite_full}


# ---------------------------------------------------------------- data


def _make_seqs(sc: Scenario, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(sc.min_len, sc.max_len + 1, size=count)
    return [rng.standard_normal((int(n), sc.dim)).astype(sc.dtype) for n in lengths]


def _make_inputs(sc: Scenario):
    rng = np.random.default_rng(0)
    if sc.kind == "dtw":
        a = rng.standard_normal((sc.len_a, sc.dim)).astype(sc.dtype)
        b = rng.standard_normal((sc.len_b, sc.dim)).astype(sc.dtype)
        return a, b
    if sc.kind == "cdist":
        return _make_seqs(sc, sc.n_a, 1), _make_seqs(sc, sc.n_b, 2)
    return (_make_seqs(sc, sc.n_a, 1),)


def _band_cells(n: int, m: int, window: int | None) -> int:
    """Number of DP cells inside the Sakoe-Chiba band (cdtw's convention)."""
    if window is None:
        return n * m
    r = max(window, abs(n - m))
    i = np.arange(n)
    lo = np.maximum(0, i - r)
    hi = np.minimum(m - 1, i + r)
    return int(np.sum(hi - lo + 1))


def _cells(sc: Scenario, inputs) -> int:
    if sc.kind == "dtw":
        return _band_cells(sc.len_a, sc.len_b, sc.window)
    if sc.kind == "cdist":
        la, lb = inputs
        return sum(_band_cells(len(a), len(b), sc.window) for a in la for b in lb)
    (seqs,) = inputs
    n = len(seqs)
    return sum(
        _band_cells(len(seqs[i]), len(seqs[j]), sc.window) for i in range(n) for j in range(i + 1, n)
    )


# ---------------------------------------------------------------- adapters


class Skip(Exception):
    pass


def _cdtw_adapter(sc: Scenario, inputs) -> Callable[[], object]:
    import cdtw

    params = inspect.signature(cdtw.dtw).parameters
    legacy = "window" not in params
    if legacy:
        if sc.window is not None:
            raise Skip("window not supported")
        metric = {"euclidean": 0, "cosine": 1}[sc.metric]
        kw = {}
    else:
        metric = sc.metric
        kw = {"window": sc.window}

    if sc.kind == "dtw":
        a, b = inputs
        if sc.n_jobs is not None:
            if "n_jobs" not in params:
                raise Skip("no single-pair n_jobs")
            kw["n_jobs"] = sc.n_jobs
        if sc.path:
            if "return_path" not in params:
                raise Skip("no return_path")
            kw["return_path"] = True
        return lambda: cdtw.dtw(a, b, metric=metric, **kw)

    if hasattr(cdtw, sc.kind):
        fn = getattr(cdtw, sc.kind)
        return lambda: fn(*inputs, metric=metric, n_jobs=sc.n_jobs, **kw)

    # emulate cdist/pdist on versions that lack them (single-threaded)
    if sc.n_jobs not in (None, 1):
        raise Skip("no parallel cdist/pdist")
    dtw = cdtw.dtw
    if sc.kind == "cdist":
        la, lb = inputs
        return lambda: np.array([[dtw(a, b, metric=metric) for b in lb] for a in la])
    (seqs,) = inputs
    n = len(seqs)
    return lambda: np.array(
        [dtw(seqs[i], seqs[j], metric=metric) for i in range(n) for j in range(i + 1, n)]
    )


def _dtaidistance_adapter(sc: Scenario, inputs) -> Callable[[], object]:
    from dtaidistance import dtw_ndim

    if sc.metric != "euclidean":
        raise Skip("dtaidistance only supports euclidean")
    kw = {} if sc.window is None else {"window": sc.window + 1}
    if sc.kind == "dtw":
        a, b = (np.ascontiguousarray(x, dtype=np.float64) for x in inputs)
        return lambda: dtw_ndim.distance_fast(a, b, **kw)
    if sc.kind == "pdist":
        (seqs,) = inputs
        seqs = [np.ascontiguousarray(x, dtype=np.float64) for x in seqs]
        parallel = sc.n_jobs != 1
        return lambda: dtw_ndim.distance_matrix_fast(seqs, compact=True, parallel=parallel, **kw)
    raise Skip("no cdist equivalent")


ADAPTERS = {"cdtw": _cdtw_adapter, "dtaidistance": _dtaidistance_adapter}


# ---------------------------------------------------------------- timing


def _measure(fn: Callable[[], object], repeats: int, min_time: float) -> list[float]:
    """Return ``repeats`` per-call times in seconds."""
    fn()  # warm-up (thread pools, caches, lazy imports)
    # calibrate the number of calls per measurement
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        elapsed = time.perf_counter() - t0
        if elapsed >= min_time / 4 or number >= 1 << 20:
            break
        number *= 4
    number = max(1, int(number * min_time / max(elapsed, 1e-9)))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    return times


def _fmt_time(s: float) -> str:
    for unit, scale in (("s", 1), ("ms", 1e-3), ("us", 1e-6)):
        if s >= scale:
            return f"{s / scale:7.2f} {unit}"
    return f"{s / 1e-9:7.1f} ns"


# ---------------------------------------------------------------- metadata


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _metadata(libs: list[str]) -> dict:
    versions = {}
    for lib in libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = None
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": _cpu(),
        "cpu_count": os.cpu_count(),
        "libs": versions,
    }


# ---------------------------------------------------------------- commands


def cmd_run(args) -> None:
    scenarios = SUITES[args.suite]()
    if args.filter:
        scenarios = [sc for sc in scenarios if all(f in sc.id for f in args.filter)]
    libs = args.libs.split(",")
    meta = _metadata(libs)
    print(f"# {meta['cpu']} ({meta['cpu_count']} cpus), python {meta['python']}, "
          f"commit {meta['git_commit']}{' (dirty)' if meta['git_dirty'] else ''}")
    results = []
    if not scenarios:
        print("no scenarios match the filters (repeated --filter values must all match)")
        return
    width = max(len(sc.id) for sc in scenarios)
    for sc in scenarios:
        inputs = _make_inputs(sc)
        cells = _cells(sc, inputs)
        for lib in libs:
            if meta["libs"].get(lib) is None:
                continue
            try:
                fn = ADAPTERS[lib](sc, inputs)
            except Skip as e:
                print(f"{sc.id:<{width}}  {lib:<12} skipped: {e}")
                continue
            try:
                times = _measure(fn, args.repeats, args.min_time)
            except Exception as e:  # e.g. a competitor without its compiled extension
                if lib == "cdtw":
                    raise
                print(f"{sc.id:<{width}}  {lib:<12} skipped: {type(e).__name__}: {e}".splitlines()[0])
                continue
            med = statistics.median(times)
            spread = (max(times) - min(times)) / med * 100
            print(f"{sc.id:<{width}}  {lib:<12} {_fmt_time(med)}  ±{spread:4.1f}%  "
                  f"{cells / med / 1e6:9.1f} Mcells/s", flush=True)
            results.append({
                "id": sc.id, "lib": lib, "scenario": dataclasses.asdict(sc),
                "cells": cells, "times": times, "median": med,
            })
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"# wrote {out}")


def cmd_compare(args) -> None:
    base = json.loads(Path(args.baseline).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    for label, run in (("baseline", base), ("candidate", cand)):
        m = run["meta"]
        print(f"# {label:<9}: commit {m['git_commit']}{' (dirty)' if m['git_dirty'] else ''}, "
              f"{m['cpu']}, {m['timestamp']}")
    if base["meta"]["cpu"] != cand["meta"]["cpu"]:
        print("# WARNING: runs are from different CPUs")
    base_by_key = {(r["id"], r["lib"]): r for r in base["results"]}
    rows = [(r, base_by_key.get((r["id"], args.baseline_lib or r["lib"])))
            for r in cand["results"]]
    rows = [(c, b) for c, b in rows if b is not None]
    if not rows:
        print("no common scenarios")
        return
    width = max(len(c["id"]) for c, _ in rows)
    speedups = []
    for c, b in rows:
        sp = b["median"] / c["median"]
        speedups.append(sp)
        print(f"{c['id']:<{width}}  {c['lib']:<12} {_fmt_time(b['median'])} -> "
              f"{_fmt_time(c['median'])}  {sp:8.2f}x")
    geo = float(np.exp(np.mean(np.log(speedups))))
    print(f"# geometric mean speedup over {len(speedups)} scenarios: {geo:.2f}x")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a benchmark suite")
    r.add_argument("--suite", choices=SUITES, default="quick")
    r.add_argument("--filter", action="append", help="only scenarios whose id contains this (repeatable)")
    r.add_argument("--libs", default="cdtw", help="comma-separated: " + ",".join(ADAPTERS))
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--min-time", type=float, default=0.2, help="seconds per measurement")
    r.add_argument("--out", help="write results as JSON")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare", help="compare two result files")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--baseline-lib", help="compare every candidate lib against this baseline lib")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
