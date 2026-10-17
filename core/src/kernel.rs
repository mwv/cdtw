//! The pairwise DTW kernel.
//!
//! The cost matrix is never materialised. Rows of the DP are processed in
//! blocks of `R`: first the frame distances for the block (restricted to the
//! union of the rows' bands) are computed into a small tile, then the DP is
//! advanced over the block, keeping only the last row.
//!
//! * Distances: a register-tiled micro-kernel computes `MR x JB` distances at
//!   a time. The column-side sequence is stored transposed (feature-major) so
//!   the inner loop over columns is contiguous and vectorises; each loaded
//!   column value is reused for `MR` rows. Cosine inputs are pre-normalised,
//!   so cosine distance is `1 - dot`.
//! * DP: within a block the `R` rows are advanced as a skewed wavefront (row
//!   `r` lags row `r-1` by one column), so the block carries `R` independent
//!   dependency chains instead of one. Values are identical to the textbook
//!   recurrence; only the evaluation order changes.

use std::borrow::Cow;

use crate::float::Float;
use crate::simd::Simd;
use crate::{Metric, Seq};

/// DP rows per block.
const R: usize = 8;
/// Rows per distance micro-kernel call.
const MR: usize = 4;

const INF: f64 = f64::INFINITY;

/// Row-side sequence: row-major frames (normalised for cosine).
pub struct Rows<'a, T: Float> {
    data: Cow<'a, [T]>,
    len: usize,
    dim: usize,
    /// Cosine with an all-zero frame: every warping path visits every frame,
    /// so the distance is undefined.
    nan: bool,
}

/// Column-side sequence, packed into panels of `T::JB` consecutive frames.
/// Each panel is stored feature-major (`dim x JB`, contiguous), so the
/// micro-kernel streams one contiguous block per tile column chunk. The last
/// panel is zero-padded, so the micro-kernel never needs a remainder loop.
pub struct Cols<T> {
    data: Vec<T>,
    len: usize,
    dim: usize,
    nan: bool,
}

fn normalised<T: Float>(seq: &Seq<'_, T>) -> (Vec<T>, bool) {
    let mut out = seq.data.to_vec();
    let mut nan = false;
    if seq.dim > 0 {
        for frame in out.chunks_exact_mut(seq.dim) {
            let norm = frame.iter().fold(T::ZERO, |acc, &x| x.fma(x, acc)).sqrt();
            if norm == T::ZERO {
                nan = true;
            } else {
                frame.iter_mut().for_each(|x| *x = *x / norm);
            }
        }
    } else {
        nan = true;
    }
    (out, nan)
}

pub fn prepare_rows<'a, T: Float>(seq: &Seq<'a, T>, metric: Metric) -> Rows<'a, T> {
    let (data, nan) = match metric {
        Metric::Euclidean => (Cow::Borrowed(seq.data), false),
        Metric::Cosine => {
            let (v, nan) = normalised(seq);
            (Cow::Owned(v), nan)
        }
    };
    Rows { data, len: seq.len, dim: seq.dim, nan }
}

pub fn prepare_cols<T: Float>(seq: &Seq<'_, T>, metric: Metric) -> Cols<T> {
    let (len, dim) = (seq.len, seq.dim);
    let (src, nan) = match metric {
        Metric::Euclidean => (Cow::Borrowed(seq.data), false),
        Metric::Cosine => {
            let (v, nan) = normalised(seq);
            (Cow::Owned(v), nan)
        }
    };
    let jb = T::JB;
    let mut data = vec![T::ZERO; len.div_ceil(jb) * dim * jb];
    for (j, frame) in src.chunks_exact(dim.max(1)).enumerate().take(len) {
        let panel = &mut data[(j / jb) * dim * jb..];
        for (k, &x) in frame.iter().enumerate() {
            panel[k * jb + j % jb] = x;
        }
    }
    Cols { data, len, dim, nan }
}

/// Per-thread reusable buffers.
#[derive(Default)]
pub struct Scratch<T> {
    tile: Vec<T>,
    prev: Vec<f64>,
    cur: Vec<f64>,
}

impl<T> Scratch<T> {
    pub fn new() -> Self {
        Scratch { tile: Vec::new(), prev: Vec::new(), cur: Vec::new() }
    }
}

/// `MR x (NV * V::LANES)` block of squared euclidean distances or dot products.
///
/// # Safety
/// Each `a[r]` must point to `dim` readable values; `panel` to a packed
/// `dim x (NV * V::LANES)` panel; each `out[r]` must be writable for
/// `NV * V::LANES` values. `V`'s target features must be enabled.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // fixed-size index loops keep accumulators in registers
unsafe fn micro<T: Float, V: Simd<T>, const NV: usize, const COS: bool>(
    a: [*const T; MR],
    panel: *const T,
    dim: usize,
    out: [*mut T; MR],
) {
    let mut acc = [[V::zero(); NV]; MR];
    for k in 0..dim {
        let bk = panel.add(k * NV * V::LANES);
        let mut b = [V::zero(); NV];
        for v in 0..NV {
            b[v] = V::load(bk.add(v * V::LANES));
        }
        for r in 0..MR {
            let x = V::splat(*a[r].add(k));
            for v in 0..NV {
                acc[r][v] = if COS {
                    x.mul_add(b[v], acc[r][v])
                } else {
                    let d = x.sub(b[v]);
                    d.mul_add(d, acc[r][v])
                };
            }
        }
    }
    for r in 0..MR {
        for v in 0..NV {
            acc[r][v].store(out[r].add(v * V::LANES));
        }
    }
}

/// Fill `tile[r * tw + (j - la)]` with the raw kernel values (squared
/// distance or dot product) of rows `i0..i0 + R` against columns `la..=h`,
/// where `la` is `l` rounded down to a panel boundary (a multiple of `T::JB`).
/// Rows past the end of `a` are computed as duplicates of its last row.
#[inline(always)]
pub fn fill_tile<T: Float, V: Simd<T>, const NV: usize, const COS: bool>(
    a: &Rows<T>,
    b: &Cols<T>,
    i0: usize,
    l: usize,
    h: usize,
    tile: &mut [T],
    tw: usize,
) {
    const { assert!(MR == 4) };
    let jb = NV * V::LANES;
    let dim = a.dim;
    assert!(jb == T::JB && dim == b.dim && h < b.len && i0 < a.len && l <= h);
    assert!(a.data.len() >= a.len * dim && b.data.len() >= b.len.div_ceil(jb) * dim * jb);
    let la = l / jb * jb;
    assert!(tile.len() >= R * tw && tw >= h - la + jb);
    let row_ptr = |r: usize| unsafe { a.data.as_ptr().add((i0 + r).min(a.len - 1) * dim) };
    let tp = tile.as_mut_ptr();
    let mut j0 = la;
    while j0 <= h {
        // panel j0 / jb exists since j0 <= h < b.len; it starts at (j0 / jb) * dim * jb = j0 * dim
        let panel = unsafe { b.data.as_ptr().add(j0 * dim) };
        for r0 in (0..R).step_by(MR) {
            let rows = [row_ptr(r0), row_ptr(r0 + 1), row_ptr(r0 + 2), row_ptr(r0 + 3)];
            // (r0 + r) * tw + (j0 - la) + jb <= R * tw since j0 - la + jb <= tw.
            let out = [0, 1, 2, 3].map(|r| unsafe { tp.add((r0 + r) * tw + (j0 - la)) });
            unsafe { micro::<T, V, NV, COS>(rows, panel, dim, out) };
        }
        j0 += jb;
    }
}

/// Minimum of two non-NaN values (NaN inputs are rejected up front or give
/// unspecified results): a single `fminnm` on aarch64 and `minsd` on x86.
#[inline(always)]
fn min(a: f64, b: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        a.min(b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if a < b {
            a
        } else {
            b
        }
    }
}

/// One DP cell of row `r` at column offset `q` in the skewed block sweep.
///
/// `c[r]` holds row `r`'s most recent value (its left neighbour), `c[r-1]`
/// the value just above, and `lu[r]` the value above-left.
///
/// # Safety
/// `l + q + 1 < prev.len()`, `l + q + 1 < cur.len()`, `r * tw + q < tile.len()`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn cell<T: Float>(
    r: usize,
    q: usize,
    l: usize,
    tile: &[T],
    tw: usize,
    prev: &[f64],
    cur: &mut [f64],
    c: &mut [f64; R],
    lu: &mut [f64; R],
) {
    let up = if r == 0 { *prev.get_unchecked(l + q + 1) } else { c[r - 1] };
    let v = min(min(up, lu[r]), c[r]) + tile.get_unchecked(r * tw + q).to_f64();
    lu[r] = up;
    c[r] = v;
    if r == R - 1 {
        *cur.get_unchecked_mut(l + q + 1) = v;
    }
}

/// Advance the DP over a full block of `R` rows, columns `l..=h`.
fn dp_block<T: Float>(tile: &[T], tw: usize, prev: &[f64], cur: &mut [f64], l: usize, h: usize) {
    let w = h - l + 1;
    assert!(h + 1 < prev.len() && h + 1 < cur.len() && (R - 1) * tw + w <= tile.len());
    let mut c = [INF; R];
    let mut lu = [INF; R];
    lu[0] = prev[l];
    // At step t, row r is at column offset t - r. Steps R-1..w have every row
    // active; the ramps before and after are handled cell by cell.
    let ramp = |t: usize, c: &mut [f64; R], lu: &mut [f64; R], cur: &mut [f64]| {
        let r_lo = t.saturating_sub(w - 1);
        for r in (r_lo..=t.min(R - 1)).rev() {
            unsafe { cell(r, t - r, l, tile, tw, prev, cur, c, lu) };
        }
    };
    for t in 0..(R - 1).min(w + R - 1) {
        ramp(t, &mut c, &mut lu, cur);
    }
    if w >= R {
        unsafe { dp_steady(tile, tw, prev, cur, l, w, &mut c, &mut lu) };
    }
    for t in w.max(R - 1)..(w + R - 1) {
        ramp(t, &mut c, &mut lu, cur);
    }
}

/// Steady state of [`dp_block`]: steps `R-1..w`, all `R` rows active.
///
/// The row state lives in named locals so it stays in registers; as arrays,
/// LLVM vectorises across rows and spills the state every step.
///
/// # Safety
/// Same bounds as asserted in [`dp_block`], and `w >= R`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dp_steady<T: Float>(
    tile: &[T],
    tw: usize,
    prev: &[f64],
    cur: &mut [f64],
    l: usize,
    w: usize,
    c: &mut [f64; R],
    lu: &mut [f64; R],
) {
    const { assert!(R == 8) };
    let [mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7] = *c;
    let [mut u0, mut u1, mut u2, mut u3, mut u4, mut u5, mut u6, mut u7] = *lu;
    // row r's cell at step t is tile[r * tw + t - r] = d[r][t]
    let d: [*const T; R] = std::array::from_fn(|r| tile.as_ptr().add(r * tw).sub(r));
    let pp = prev.as_ptr().add(l + 1);
    let cp = cur.as_mut_ptr().add(l + 1);
    macro_rules! cell {
        ($t:expr, $r:literal, $c:ident, $u:ident, $up:expr) => {{
            let up = $up;
            $c = min(min(up, $u), $c) + (*d[$r].add($t)).to_f64();
            $u = up;
        }};
    }
    for t in (R - 1)..w {
        // descending, so each row reads the row above before it advances
        cell!(t, 7, c7, u7, c6);
        cell!(t, 6, c6, u6, c5);
        cell!(t, 5, c5, u5, c4);
        cell!(t, 4, c4, u4, c3);
        cell!(t, 3, c3, u3, c2);
        cell!(t, 2, c2, u2, c1);
        cell!(t, 1, c1, u1, c0);
        cell!(t, 0, c0, u0, *pp.add(t));
        *cp.add(t - (R - 1)) = c7; // row 7 is at offset t - 7
    }
    *c = [c0, c1, c2, c3, c4, c5, c6, c7];
    *lu = [u0, u1, u2, u3, u4, u5, u6, u7];
}

/// Advance the DP over `rows < R` rows one row at a time (last block).
fn dp_rows<T: Float>(tile: &[T], tw: usize, prev: &mut Vec<f64>, cur: &mut Vec<f64>, rows: usize, l: usize, h: usize) {
    for r in 0..rows {
        let row = &tile[r * tw..r * tw + (h - l + 1)];
        let mut left = INF;
        cur[l] = INF;
        for (q, &d) in row.iter().enumerate() {
            let j = l + q;
            let v = min(min(prev[j + 1], prev[j]), left) + d.to_f64();
            cur[j + 1] = v;
            left = v;
        }
        std::mem::swap(prev, cur);
    }
}

pub fn dtw_pair<T: Float>(a: &Rows<T>, b: &Cols<T>, metric: Metric, window: Option<usize>, s: &mut Scratch<T>) -> f64 {
    if a.nan || b.nan {
        return f64::NAN;
    }
    let (na, nb) = (a.len, b.len);
    debug_assert!(na > 0 && nb > 0);
    if a.dim == 0 {
        return 0.0; // euclidean between empty frames
    }
    let radius = match window {
        None => na.max(nb),
        Some(w) => w.max(na.abs_diff(nb)),
    };
    let band = |i: usize| (i.saturating_sub(radius), (i.saturating_add(radius)).min(nb - 1));

    let tw = nb + T::JB;
    s.tile.resize(R * tw, T::ZERO);
    s.prev.clear();
    s.prev.resize(nb + 1, INF);
    s.prev[0] = 0.0; // virtual row -1, column -1
    s.cur.clear();
    s.cur.resize(nb + 1, INF);
    let cosine = metric == Metric::Cosine;

    let mut i0 = 0;
    while i0 < na {
        let rows = R.min(na - i0);
        let l = band(i0).0;
        let h = band(i0 + rows - 1).1;
        let w = h - l + 1;
        T::fill_tile(a, b, i0, l, h, &mut s.tile, tw, cosine);
        // the tile starts at the panel boundary below l; view it from column l
        let tile = &mut s.tile[l % T::JB..];

        // kernel values -> distances; out-of-band cells -> inf
        for r in 0..rows {
            let (lo, hi) = band(i0 + r);
            let row = &mut tile[r * tw..r * tw + w];
            let (before, rest) = row.split_at_mut(lo - l);
            let (inside, after) = rest.split_at_mut(hi - lo + 1);
            before.fill(T::INFINITY);
            after.fill(T::INFINITY);
            if cosine {
                inside.iter_mut().for_each(|x| *x = T::ONE - *x);
            } else {
                inside.iter_mut().for_each(|x| *x = x.sqrt());
            }
        }

        if rows == R {
            s.cur[l] = INF;
            dp_block(tile, tw, &s.prev, &mut s.cur, l, h);
            std::mem::swap(&mut s.prev, &mut s.cur);
        } else {
            dp_rows(tile, tw, &mut s.prev, &mut s.cur, rows, l, h);
        }
        // Columns the next block may read beyond this block's range.
        let end = (h + 2 + R).min(nb + 1);
        if h + 2 < end {
            s.prev[h + 2..end].fill(INF);
        }
        i0 += rows;
    }
    s.prev[nb]
}
