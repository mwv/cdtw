//! Multi-threaded DTW for a single large pair.
//!
//! The DP matrix is cut into rectangles of `row_blocks` DP row blocks by
//! `cols` columns. A rectangle needs only the bottom row of the rectangle
//! above it and the right column of the rectangle to its left, so it can run
//! as soon as both are done: each rectangle counts its unfinished
//! predecessors, and whichever predecessor finishes last spawns it. Unlike a
//! diagonal-by-diagonal sweep there is no barrier per anti-diagonal.
//!
//! Rectangles exchange data through two shared buffers, each written by one
//! rectangle at a time:
//! * `row`: one cost row (`row[j + 1]` is column `j`). Rectangles in column
//!   chunk `bj` only touch that chunk's slice, which holds the bottom row of
//!   the latest finished rectangle in the chunk.
//! * `right`: for every column chunk, the cost in its last column of every row.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::float::Float;
use crate::kernel::{run_rect, Band, Cols, DirsView, Rect, Rows, R};
use crate::Metric;

/// Below this many band cells a pair runs single-threaded: the work is too
/// small to be worth splitting.
pub const MIN_PARALLEL_CELLS: usize = 1 << 18;

const INF: f64 = f64::INFINITY;

/// Rectangle size.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// DP row blocks (of `R` rows) per rectangle
    pub row_blocks: usize,
    /// columns per rectangle; a multiple of the tile width `T::JB`
    pub cols: usize,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry { row_blocks: 4, cols: 128 }
    }
}

/// A raw pointer that rectangles on different threads may use for their own
/// disjoint parts of a buffer.
#[derive(Clone, Copy)]
struct Shared(*mut f64);

// SAFETY: see the module docs; rectangles access disjoint index ranges, and
// a rectangle only starts after its predecessors (Acquire/Release on `deps`).
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

struct Ctx<'a, T: Float> {
    a: &'a Rows<'a, T>,
    b: &'a Cols<T>,
    metric: Metric,
    band: &'a Band,
    dirs: Option<DirsView<'a>>,
    g: Geometry,
    /// rectangles per column / per row
    nbi: usize,
    nbj: usize,
    row: Shared,
    right: Shared,
    /// unfinished predecessors of each rectangle
    deps: &'a [AtomicU8],
}

impl<'a, T: Float> Ctx<'a, T> {
    fn spawn<const PATH: bool>(&'a self, s: &rayon::Scope<'a>, bi: usize, bj: usize) {
        s.spawn(move |s| self.run::<PATH>(s, bi, bj));
    }

    fn run<const PATH: bool>(&'a self, s: &rayon::Scope<'a>, bi: usize, bj: usize) {
        let (na, nb) = (self.band.na, self.band.nb);
        let rb0 = bi * self.g.row_blocks;
        let rb1 = (rb0 + self.g.row_blocks).min(self.band.blocks());
        let (r0, r1) = (rb0 * R, (rb1 * R).min(na));
        let c0 = bj * self.g.cols;
        let c1 = (c0 + self.g.cols).min(nb);

        // costs in column c0 - 1 for rows r0 - 1..r1 (see run_rect)
        let mut left = vec![INF; r1 - r0 + 1];
        if bj == 0 {
            if r0 == 0 {
                left[0] = 0.0; // virtual cell (-1, -1)
            }
        } else {
            // written by the rectangles to the left and above-left, both done
            let col = self.right.0.wrapping_add((bj - 1) * na);
            for (k, x) in left.iter_mut().enumerate() {
                if r0 + k > 0 {
                    *x = unsafe { *col.add(r0 + k - 1) };
                }
            }
        }
        let right = unsafe { std::slice::from_raw_parts_mut(self.right.0.add(bj * na + r0), r1 - r0) };
        let rect = Rect { rb0, rb1, c0, c1 };
        let mut tile = Vec::new();
        unsafe {
            run_rect::<T, PATH>(self.a, self.b, self.metric, self.band, rect, &left, right, self.row.0, self.dirs, &mut tile)
        };

        // release the rectangles below and to the right
        for (ok, t) in [(bi + 1 < self.nbi, (bi + 1, bj)), (bj + 1 < self.nbj, (bi, bj + 1))] {
            if ok && self.deps[t.0 * self.nbj + t.1].fetch_sub(1, Ordering::AcqRel) == 1 {
                self.spawn::<PATH>(s, t.0, t.1);
            }
        }
    }
}

/// DTW cost of one pair on the current rayon pool. Fills `dirs` if `PATH`.
pub fn dtw_parallel<T: Float, const PATH: bool>(
    a: &Rows<T>,
    b: &Cols<T>,
    metric: Metric,
    band: &Band,
    dirs: Option<DirsView<'_>>,
    g: Geometry,
) -> f64 {
    assert!(g.row_blocks > 0 && g.cols > 0 && g.cols % T::JB == 0);
    let (na, nb) = (band.na, band.nb);
    let nbi = band.blocks().div_ceil(g.row_blocks);
    let nbj = nb.div_ceil(g.cols);
    let mut row = vec![INF; nb + 1];
    let mut right = vec![INF; nbj * na];
    let deps: Vec<AtomicU8> = (0..nbi * nbj)
        .map(|t| AtomicU8::new(u8::from(t / nbj > 0) + u8::from(t % nbj > 0)))
        .collect();
    let ctx = Ctx {
        a,
        b,
        metric,
        band,
        dirs,
        g,
        nbi,
        nbj,
        row: Shared(row.as_mut_ptr()),
        right: Shared(right.as_mut_ptr()),
        deps: &deps,
    };
    rayon::scope(|s| ctx.spawn::<PATH>(s, 0, 0));
    row[nb]
}
