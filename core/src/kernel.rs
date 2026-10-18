//! The pairwise DTW kernel.
//!
//! The cost matrix is never materialised. Rows of the DP are processed in
//! blocks of `R`: first the frame distances for the block (restricted to the
//! union of the rows' bands) are computed into a small tile, then the DP is
//! advanced over the block, keeping only the last row.
//!
//! The DP runs over rectangles ([`run_rect`]) that take the costs left of and
//! above them as input. Single-threaded DTW is one rectangle covering the
//! matrix; `wavefront` runs many in parallel. With `PATH`, every cell also
//! records which predecessor it came from ([`Dirs`]), for backtracking.
//!
//! * Distances: a register-tiled micro-kernel computes `MR x JB` distances at
//!   a time. The column-side sequence is packed feature-major in panels of
//!   `JB` frames, so the inner loop over columns is contiguous and
//!   vectorises; each loaded column value is reused for `MR` rows. Cosine
//!   inputs are pre-normalised, so cosine distance is `1 - dot`.
//! * DP: within a block the `R` rows are advanced as a skewed wavefront (row
//!   `r` lags row `r-1` by one column), so the block carries `R` independent
//!   dependency chains instead of one. Values are identical to the textbook
//!   recurrence; only the evaluation order changes.

use std::borrow::Cow;

use crate::float::Float;
use crate::simd::Simd;
use crate::{Metric, Seq};

/// DP rows per block.
pub const R: usize = 8;
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
///
/// On aarch64 this uses the NEON intrinsic rather than `f64::min`: newer
/// LLVM lowers `f64::min` with extra NaN-canonicalising `fminnm x, x`
/// instructions and vectorises it across DP rows, both of which lengthen
/// the DP's dependency chains.
#[inline(always)]
fn min(a: f64, b: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vdup_n_f64, vget_lane_f64, vminnm_f64};
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vget_lane_f64::<0>(vminnm_f64(vdup_n_f64(a), vdup_n_f64(b))) }
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

// ------------------------------------------------------------------ band

/// Sakoe-Chiba band: row `i` may visit columns `row(i).0..=row(i).1`.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub na: usize,
    pub nb: usize,
    radius: usize,
}

impl Band {
    pub fn new(na: usize, nb: usize, window: Option<usize>) -> Self {
        let radius = match window {
            None => na.max(nb),
            Some(w) => w.max(na.abs_diff(nb)),
        };
        Band { na, nb, radius }
    }

    #[inline]
    pub fn row(&self, i: usize) -> (usize, usize) {
        (i.saturating_sub(self.radius), i.saturating_add(self.radius).min(self.nb - 1))
    }

    /// Number of DP row blocks.
    pub fn blocks(&self) -> usize {
        self.na.div_ceil(R)
    }

    /// First row, row count and band union `(l, h)` of row block `rb`.
    #[inline]
    pub fn block(&self, rb: usize) -> (usize, usize, usize, usize) {
        let i0 = rb * R;
        let rows = R.min(self.na - i0);
        (i0, rows, self.row(i0).0, self.row(i0 + rows - 1).1)
    }

    /// Number of cells inside the band.
    pub fn cells(&self) -> usize {
        (0..self.na).map(|i| self.row(i).1 - self.row(i).0 + 1).sum()
    }
}

// ------------------------------------------------------------------ path

const DIAG: u8 = 0;
const UP: u8 = 1;
const LEFT: u8 = 2;

/// Which predecessor each cell's DP minimum came from (one byte per cell),
/// for every cell of each row block's band union, block by block.
pub struct Dirs {
    data: Vec<u8>,
    /// start of each row block in `data`
    offsets: Vec<usize>,
}

impl Dirs {
    pub fn new(band: &Band) -> Self {
        let mut offsets = Vec::with_capacity(band.blocks());
        let mut total = 0;
        for rb in 0..band.blocks() {
            let (_, rows, l, h) = band.block(rb);
            offsets.push(total);
            total += rows * (h - l + 1);
        }
        Dirs { data: vec![0; total], offsets }
    }

    /// A view that rectangles running on different threads can write their
    /// own (disjoint) cells through.
    pub fn view(&mut self) -> DirsView<'_> {
        DirsView { data: self.data.as_mut_ptr(), offsets: &self.offsets }
    }

    fn get(&self, band: &Band, i: usize, j: usize) -> u8 {
        let rb = i / R;
        let (i0, _, l, h) = band.block(rb);
        self.data[self.offsets[rb] + (i - i0) * (h - l + 1) + (j - l)]
    }

    /// The optimal warping path, from `(0, 0)` to `(na - 1, nb - 1)`.
    pub fn backtrack(&self, band: &Band) -> Vec<(usize, usize)> {
        let (mut i, mut j) = (band.na - 1, band.nb - 1);
        let mut path = Vec::with_capacity(band.na + band.nb);
        path.push((i, j));
        while i > 0 || j > 0 {
            match (i, j) {
                (0, _) => j -= 1,
                (_, 0) => i -= 1,
                _ => match self.get(band, i, j) {
                    DIAG => (i, j) = (i - 1, j - 1),
                    UP => i -= 1,
                    _ => j -= 1,
                },
            }
            path.push((i, j));
        }
        path.reverse();
        path
    }
}

/// Shared write access to a [`Dirs`]; see [`Dirs::view`].
#[derive(Clone, Copy)]
pub struct DirsView<'a> {
    data: *mut u8,
    offsets: &'a [usize],
}

// SAFETY: writers only touch the cells of their own rectangle (run_rect).
unsafe impl Send for DirsView<'_> {}
unsafe impl Sync for DirsView<'_> {}

impl DirsView<'_> {
    /// Pointer to the decision of cell `(i0, ls)` of row block `rb` and the
    /// row stride; cell `(i0 + r, ls + q)` is at `ptr + r * stride + q`.
    fn block_ptr(&self, band: &Band, rb: usize, ls: usize) -> (*mut u8, usize) {
        let (_, _, l, h) = band.block(rb);
        (self.data.wrapping_add(self.offsets[rb] + (ls - l)), h - l + 1)
    }
}

// ------------------------------------------------------------------ DP

/// The DP recurrence: `min(up, diag, left) + d`, plus (if `PATH`) which of
/// the three was taken. Ties prefer the diagonal, then up.
#[inline(always)]
fn relax<const PATH: bool>(up: f64, diag: f64, left: f64, d: f64) -> (f64, u8) {
    let m = min(up, diag);
    let v = min(m, left) + d;
    let dir = if !PATH {
        0
    } else if left < m {
        LEFT
    } else if diag <= up {
        DIAG
    } else {
        UP
    };
    (v, dir)
}

/// One DP cell of row `r` at column offset `q` in the skewed block sweep.
///
/// `c[r]` holds row `r`'s most recent value (its left neighbour), `c[r-1]`
/// the value just above, and `lu[r]` the value above-left. `row[q]` holds
/// the row above the block on entry; row `R-1` overwrites it in place, `R-1`
/// columns behind row 0's reads.
///
/// # Safety
/// `q < w` for the block's width `w`; `row` valid for `w` values, `d` for
/// `(R-1) * tw + w`, `dirs` (if `PATH`) for `(R-1) * ds + w`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn cell<T: Float, const PATH: bool>(
    r: usize,
    q: usize,
    d: &[T],
    tw: usize,
    row: *mut f64,
    dirs: *mut u8,
    ds: usize,
    c: &mut [f64; R],
    lu: &mut [f64; R],
) {
    let up = if r == 0 { *row.add(q) } else { c[r - 1] };
    let (v, dir) = relax::<PATH>(up, lu[r], c[r], d.get_unchecked(r * tw + q).to_f64());
    if PATH {
        *dirs.add(r * ds + q) = dir;
    }
    lu[r] = up;
    c[r] = v;
    if r == R - 1 {
        *row.add(q) = v;
    }
}

/// Advance the DP over a full block of `R` rows and `w` columns.
///
/// `lin[0]` is the value above-left of the block, `lin[1 + r]` the value left
/// of row `r`. Returns each row's last value.
///
/// # Safety
/// As for [`cell`].
unsafe fn dp_block<T: Float, const PATH: bool>(
    d: &[T],
    tw: usize,
    w: usize,
    row: *mut f64,
    lin: &[f64; R + 1],
    dirs: *mut u8,
    ds: usize,
) -> [f64; R] {
    debug_assert!((R - 1) * tw + w <= d.len());
    let mut c: [f64; R] = std::array::from_fn(|r| lin[r + 1]);
    let mut lu: [f64; R] = std::array::from_fn(|r| lin[r]);
    // At step t, row r is at column offset t - r. Steps R-1..w have every row
    // active; the ramps before and after are handled cell by cell.
    let ramp = |t: usize, c: &mut [f64; R], lu: &mut [f64; R]| {
        let r_lo = t.saturating_sub(w - 1);
        for r in (r_lo..=t.min(R - 1)).rev() {
            cell::<T, PATH>(r, t - r, d, tw, row, dirs, ds, c, lu);
        }
    };
    for t in 0..(R - 1).min(w + R - 1) {
        ramp(t, &mut c, &mut lu);
    }
    if w >= R {
        dp_steady::<T, PATH>(d, tw, w, row, dirs, ds, &mut c, &mut lu);
    }
    for t in w.max(R - 1)..(w + R - 1) {
        ramp(t, &mut c, &mut lu);
    }
    c
}

/// Steady state of [`dp_block`]: steps `R-1..w`, all `R` rows active.
///
/// The row state lives in named locals so it stays in registers; as arrays,
/// LLVM vectorises across rows and spills the state every step.
///
/// # Safety
/// As for [`cell`], and `w >= R`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dp_steady<T: Float, const PATH: bool>(
    d: &[T],
    tw: usize,
    w: usize,
    row: *mut f64,
    dirs: *mut u8,
    ds: usize,
    c: &mut [f64; R],
    lu: &mut [f64; R],
) {
    const { assert!(R == 8) };
    let [mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7] = *c;
    let [mut u0, mut u1, mut u2, mut u3, mut u4, mut u5, mut u6, mut u7] = *lu;
    // row r's cell at step t is d[r * tw + t - r]; likewise for dirs
    let dp: [*const T; R] = std::array::from_fn(|r| d.as_ptr().add(r * tw).sub(r));
    let dd: [*mut u8; R] = std::array::from_fn(|r| dirs.wrapping_add(r * ds).wrapping_sub(r));
    macro_rules! cell {
        ($t:expr, $r:literal, $c:ident, $u:ident, $up:expr) => {{
            let up = $up;
            let (v, dir) = relax::<PATH>(up, $u, $c, (*dp[$r].add($t)).to_f64());
            if PATH {
                *dd[$r].add($t) = dir;
            }
            $c = v;
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
        cell!(t, 0, c0, u0, *row.add(t));
        *row.add(t - (R - 1)) = c7; // row 7 is at offset t - 7
    }
    *c = [c0, c1, c2, c3, c4, c5, c6, c7];
    *lu = [u0, u1, u2, u3, u4, u5, u6, u7];
}

/// Advance the DP over `rows < R` rows one row at a time (last block).
/// Arguments as for [`dp_block`].
///
/// # Safety
/// As for [`cell`].
#[allow(clippy::too_many_arguments)]
unsafe fn dp_rows<T: Float, const PATH: bool>(
    d: &[T],
    tw: usize,
    w: usize,
    rows: usize,
    row: *mut f64,
    lin: &[f64; R + 1],
    dirs: *mut u8,
    ds: usize,
) -> [f64; R] {
    let mut right = [INF; R];
    for r in 0..rows {
        let dr = &d[r * tw..r * tw + w];
        let (mut left, mut diag) = (lin[r + 1], lin[r]);
        for (q, &x) in dr.iter().enumerate() {
            let up = *row.add(q);
            let (v, dir) = relax::<PATH>(up, diag, left, x.to_f64());
            if PATH {
                *dirs.add(r * ds + q) = dir;
            }
            *row.add(q) = v;
            (diag, left) = (up, v);
        }
        right[r] = left;
    }
    right
}

// ------------------------------------------------------------------ tiles

/// A rectangle of the DP: row blocks `rb0..rb1`, columns `c0..c1`.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub rb0: usize,
    pub rb1: usize,
    pub c0: usize,
    pub c1: usize,
}

/// Advance the DP over a rectangle.
///
/// * `left[k]` is the cost of cell `(r0 - 1 + k, c0 - 1)` for `r0 = rb0 * R`
///   (`k = 0` is the corner above-left of the rectangle), for every row of the
///   rectangle; out-of-band cells are `inf`, the virtual cell `(-1, -1)` is 0.
/// * `right[k]` receives the cost of cell `(r0 + k, c1 - 1)`.
/// * `row` is the shared row buffer (`row[j + 1]` is column `j`): columns
///   `c0..c1` hold the row above the rectangle on entry (`inf` where not yet
///   reached) and its last row on exit. Only indices `c0 + 1..=c1` are used.
/// * `dirs`, when `PATH`, receives every cell's decision.
///
/// # Safety
/// `row` must be valid for `band.nb + 1` values, with no concurrent access to
/// indices `c0 + 1..=c1`; `dirs` likewise for this rectangle's cells.
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_rect<T: Float, const PATH: bool>(
    a: &Rows<T>,
    b: &Cols<T>,
    metric: Metric,
    band: &Band,
    rect: Rect,
    left: &[f64],
    right: &mut [f64],
    row: *mut f64,
    dirs: Option<DirsView<'_>>,
    tile: &mut Vec<T>,
) {
    let Rect { rb0, rb1, c0, c1 } = rect;
    let cosine = metric == Metric::Cosine;
    let tw = (c1 - c0) + T::JB;
    tile.resize(R * tw, T::ZERO);
    let r0 = rb0 * R;
    for rb in rb0..rb1 {
        let (i0, rows, l, h) = band.block(rb);
        let (ls, he) = (l.max(c0), h.min(c1 - 1));
        let k = i0 - r0;
        if ls > he {
            right[k..k + rows].fill(INF);
            continue;
        }
        let w = he - ls + 1;
        // costs left of the block's columns: from the neighbouring rectangle
        // if the block starts at c0, else out of band
        let mut lin = [INF; R + 1];
        if ls == c0 {
            lin[..=rows].copy_from_slice(&left[k..=k + rows]);
        } else {
            lin[0] = *row.add(ls);
        }

        T::fill_tile(a, b, i0, ls, he, tile, tw, cosine);
        // the tile starts at the panel boundary below ls; view it from ls
        let d = &mut tile[ls % T::JB..];
        // kernel values -> distances; out-of-band cells -> inf
        for r in 0..rows {
            let (lo, hi) = band.row(i0 + r);
            let (lo, hi) = (lo.clamp(ls, he + 1), (hi + 1).clamp(ls, he + 1));
            let dr = &mut d[r * tw..r * tw + w];
            let (before, rest) = dr.split_at_mut(lo - ls);
            let (inside, after) = rest.split_at_mut(hi.max(lo) - lo);
            before.fill(T::INFINITY);
            after.fill(T::INFINITY);
            if cosine {
                inside.iter_mut().for_each(|x| *x = T::ONE - *x);
            } else {
                inside.iter_mut().for_each(|x| *x = x.sqrt());
            }
        }

        let (dp, ds) = match dirs {
            Some(dirs) => dirs.block_ptr(band, rb, ls),
            None => (std::ptr::null_mut(), 0),
        };
        let rp = row.add(ls + 1);
        let last = if rows == R {
            dp_block::<T, PATH>(d, tw, w, rp, &lin, dp, ds)
        } else {
            dp_rows::<T, PATH>(d, tw, w, rows, rp, &lin, dp, ds)
        };
        if he == c1 - 1 {
            right[k..k + rows].copy_from_slice(&last[..rows]);
        } else {
            right[k..k + rows].fill(INF);
        }
        // the block's last row, one column left of its range: out of band
        if ls > c0 {
            *row.add(ls) = INF;
        }
    }
}

/// Per-thread reusable buffers.
#[derive(Default)]
pub struct Scratch<T> {
    tile: Vec<T>,
    row: Vec<f64>,
    left: Vec<f64>,
    right: Vec<f64>,
}

impl<T> Scratch<T> {
    pub fn new() -> Self {
        Scratch { tile: Vec::new(), row: Vec::new(), left: Vec::new(), right: Vec::new() }
    }
}

/// Single-threaded DTW over the whole matrix as one rectangle.
fn dtw_sequential<T: Float, const PATH: bool>(
    a: &Rows<T>,
    b: &Cols<T>,
    metric: Metric,
    band: &Band,
    dirs: Option<DirsView<'_>>,
    s: &mut Scratch<T>,
) -> f64 {
    let (na, nb) = (band.na, band.nb);
    s.row.clear();
    s.row.resize(nb + 1, INF);
    s.left.clear();
    s.left.resize(na + 1, INF);
    s.left[0] = 0.0; // virtual cell (-1, -1)
    s.right.resize(na, INF);
    let rect = Rect { rb0: 0, rb1: band.blocks(), c0: 0, c1: nb };
    unsafe {
        run_rect::<T, PATH>(a, b, metric, band, rect, &s.left, &mut s.right, s.row.as_mut_ptr(), dirs, &mut s.tile)
    };
    s.row[nb]
}

/// DTW cost of one pair, single-threaded.
pub fn dtw_pair<T: Float>(a: &Rows<T>, b: &Cols<T>, metric: Metric, window: Option<usize>, s: &mut Scratch<T>) -> f64 {
    if a.nan || b.nan {
        return f64::NAN;
    }
    let band = Band::new(a.len, b.len, window);
    dtw_sequential::<T, false>(a, b, metric, &band, None, s)
}

/// DTW cost and (if `path`) warping path of one pair, using the current
/// rayon pool when `parallel` and the matrix is large enough.
pub fn dtw_pair_full<T: Float>(
    a: &Rows<T>,
    b: &Cols<T>,
    metric: Metric,
    window: Option<usize>,
    path: bool,
    parallel: bool,
) -> (f64, Option<crate::Path>) {
    if a.nan || b.nan {
        return (f64::NAN, path.then(Vec::new));
    }
    let band = Band::new(a.len, b.len, window);
    let geometry = crate::wavefront::Geometry::default();
    let parallel = parallel && band.cells() >= crate::wavefront::MIN_PARALLEL_CELLS;
    if path {
        let mut dirs = Dirs::new(&band);
        let view = Some(dirs.view());
        let cost = if parallel {
            crate::wavefront::dtw_parallel::<T, true>(a, b, metric, &band, view, geometry)
        } else {
            dtw_sequential::<T, true>(a, b, metric, &band, view, &mut Scratch::new())
        };
        (cost, Some(dirs.backtrack(&band)))
    } else if parallel {
        (crate::wavefront::dtw_parallel::<T, false>(a, b, metric, &band, None, geometry), None)
    } else {
        (dtw_sequential::<T, false>(a, b, metric, &band, None, &mut Scratch::new()), None)
    }
}

