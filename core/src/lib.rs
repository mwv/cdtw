//! Dynamic time warping between sequences of feature vectors.
//!
//! The DTW cost is the minimum, over all monotone warping paths from
//! `(0, 0)` to `(n - 1, m - 1)` using steps `(1, 0)`, `(0, 1)` and `(1, 1)`,
//! of the summed frame distances along the path. Distances are not
//! normalised by path length.
//!
//! `window` restricts paths to a Sakoe-Chiba band `|i - j| <= r` with
//! `r = max(window, |n - m|)`; the widening guarantees that a path exists.

mod float;
mod kernel;
mod simd;
mod wavefront;

use std::fmt;

use rayon::prelude::*;

pub use float::Float;
use kernel::{dtw_pair, dtw_pair_full, prepare_cols, prepare_rows, Scratch};

/// Frame-level distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// `||a - b||`
    Euclidean,
    /// `1 - a.b / (||a|| ||b||)`. A sequence containing an all-zero frame
    /// has an undefined distance to everything (`NaN`).
    Cosine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub metric: Metric,
    /// Sakoe-Chiba band radius in frames; `None` for unconstrained DTW.
    pub window: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options { metric: Metric::Euclidean, window: None }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// A borrowed `len x dim` row-major sequence of frames.
#[derive(Clone, Copy, Debug)]
pub struct Seq<'a, T> {
    data: &'a [T],
    len: usize,
    dim: usize,
}

impl<'a, T> Seq<'a, T> {
    pub fn new(data: &'a [T], len: usize, dim: usize) -> Result<Self, Error> {
        if len.checked_mul(dim) != Some(data.len()) {
            return Err(Error(format!(
                "sequence data has {} values, expected {len} x {dim}",
                data.len()
            )));
        }
        Ok(Seq { data, len, dim })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Check that every sequence is non-empty and they all share one dimension.
fn validate<T>(groups: &[(&str, &[Seq<'_, T>])]) -> Result<(), Error> {
    let mut dim: Option<(usize, &str, usize)> = None;
    for (name, seqs) in groups {
        for (i, s) in seqs.iter().enumerate() {
            if s.len == 0 {
                return Err(Error(format!("{name}[{i}] is empty; sequences must have at least one frame")));
            }
            match dim {
                None => dim = Some((s.dim, name, i)),
                Some((d, n0, i0)) if d != s.dim => {
                    return Err(Error(format!(
                        "{name}[{i}] has {} features but {n0}[{i0}] has {d}; all sequences must have the same number of features",
                        s.dim
                    )))
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn check_pair<T>(a: &Seq<'_, T>, b: &Seq<'_, T>) -> Result<(), Error> {
    if a.len == 0 || b.len == 0 {
        return Err(Error("sequences must have at least one frame".into()));
    }
    if a.dim != b.dim {
        return Err(Error(format!(
            "sequences must have the same number of features (got {} and {})",
            a.dim, b.dim
        )));
    }
    Ok(())
}

/// A warping path: `(i, j)` frame pairs from `(0, 0)` to `(n - 1, m - 1)`.
pub type Path = Vec<(usize, usize)>;

fn single<T: Float>(a: Seq<'_, T>, b: Seq<'_, T>, opts: &Options, path: bool, n_jobs: Option<usize>) -> Result<(f64, Option<Path>), Error> {
    check_pair(&a, &b)?;
    let rows = prepare_rows(&a, opts.metric);
    let cols = prepare_cols(&b, opts.metric);
    with_threads(n_jobs, |parallel| dtw_pair_full(&rows, &cols, opts.metric, opts.window, path, parallel))
}

/// DTW cost between two sequences.
///
/// `n_jobs` sets the threads used for this one pair, with the same
/// convention as [`cdist`]. Pairs with fewer than 2^18 (about 260,000) band
/// cells always run on the calling thread.
pub fn dtw<T: Float>(a: Seq<'_, T>, b: Seq<'_, T>, opts: &Options, n_jobs: Option<usize>) -> Result<f64, Error> {
    Ok(single(a, b, opts, false, n_jobs)?.0)
}

/// DTW cost and an optimal warping path: the `(i, j)` frame pairs it visits,
/// from `(0, 0)` to `(a.len() - 1, b.len() - 1)`. Among equally cheap
/// predecessors the diagonal step is preferred, then the step along `a`.
///
/// Needs one byte of memory per band cell. The path is empty when the cost
/// is NaN. `n_jobs` as for [`dtw`].
pub fn dtw_path<T: Float>(a: Seq<'_, T>, b: Seq<'_, T>, opts: &Options, n_jobs: Option<usize>) -> Result<(f64, Path), Error> {
    let (cost, path) = single(a, b, opts, true, n_jobs)?;
    Ok((cost, path.unwrap_or_default()))
}

/// Run `f` on `n_jobs` threads: `None` uses rayon's global pool (all cores
/// unless `RAYON_NUM_THREADS` is set), `Some(1)` runs on the calling thread.
fn with_threads<R: Send>(n_jobs: Option<usize>, f: impl FnOnce(bool) -> R + Send) -> Result<R, Error> {
    match n_jobs {
        Some(0) => Err(Error("n_jobs must be at least 1".into())),
        Some(1) => Ok(f(false)),
        None => Ok(f(true)),
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| Error(format!("could not start thread pool: {e}")))?;
            Ok(pool.install(|| f(true)))
        }
    }
}

/// Evaluate `pair(scratch, k)` for `k in 0..out.len()` into `out`.
fn fill<T: Float>(out: &mut [f64], parallel: bool, pair: impl Fn(&mut Scratch<T>, usize) -> f64 + Sync) {
    if parallel {
        out.par_iter_mut()
            .enumerate()
            .for_each_init(Scratch::new, |s, (k, o)| *o = pair(s, k));
    } else {
        let mut s = Scratch::new();
        out.iter_mut().enumerate().for_each(|(k, o)| *o = pair(&mut s, k));
    }
}

/// DTW cost between every pair of `a[i]` and `b[j]`, as a row-major
/// `a.len() x b.len()` matrix.
pub fn cdist<T: Float>(a: &[Seq<'_, T>], b: &[Seq<'_, T>], opts: &Options, n_jobs: Option<usize>) -> Result<Vec<f64>, Error> {
    validate(&[("seqs_a", a), ("seqs_b", b)])?;
    let (na, nb) = (a.len(), b.len());
    let mut out = vec![0.0; na * nb];
    if out.is_empty() {
        return Ok(out);
    }
    with_threads(n_jobs, |parallel| {
        let (rows, cols) = if parallel {
            rayon::join(
                || a.par_iter().map(|s| prepare_rows(s, opts.metric)).collect::<Vec<_>>(),
                || b.par_iter().map(|s| prepare_cols(s, opts.metric)).collect::<Vec<_>>(),
            )
        } else {
            (
                a.iter().map(|s| prepare_rows(s, opts.metric)).collect(),
                b.iter().map(|s| prepare_cols(s, opts.metric)).collect(),
            )
        };
        fill(&mut out, parallel, |s, k| dtw_pair(&rows[k / nb], &cols[k % nb], opts.metric, opts.window, s));
    })?;
    Ok(out)
}

/// DTW cost between every pair `i < j` of `seqs`, in condensed order
/// (`(0,1), (0,2), ..., (0,n-1), (1,2), ...`), as `scipy.spatial.distance.pdist`.
pub fn pdist<T: Float>(seqs: &[Seq<'_, T>], opts: &Options, n_jobs: Option<usize>) -> Result<Vec<f64>, Error> {
    validate(&[("seqs", seqs)])?;
    let n = seqs.len();
    let mut out = vec![0.0; n * n.saturating_sub(1) / 2];
    if out.is_empty() {
        return Ok(out);
    }
    // condensed index of the first pair in row i
    let starts: Vec<usize> = (0..n).map(|i| i * n - i * (i + 1) / 2).collect();
    with_threads(n_jobs, |parallel| {
        let m = opts.metric;
        let prepared: Vec<_> = if parallel {
            seqs.par_iter().map(|s| (prepare_rows(s, m), prepare_cols(s, m))).collect()
        } else {
            seqs.iter().map(|s| (prepare_rows(s, m), prepare_cols(s, m))).collect()
        };
        fill(&mut out, parallel, |s, k| {
            let i = starts.partition_point(|&st| st <= k) - 1;
            let j = i + 1 + (k - starts[i]);
            dtw_pair(&prepared[i].0, &prepared[j].1, opts.metric, opts.window, s)
        });
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests;
