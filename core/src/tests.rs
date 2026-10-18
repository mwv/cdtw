use super::*;

/// Deterministic xorshift generator; values in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn seq(&mut self, len: usize, dim: usize) -> Vec<f64> {
        (0..len * dim).map(|_| self.next()).collect()
    }
}

fn frame_dist(a: &[f64], b: &[f64], dim: usize, metric: Metric, i: usize, j: usize) -> f64 {
    let (x, y) = (&a[i * dim..(i + 1) * dim], &b[j * dim..(j + 1) * dim]);
    match metric {
        Metric::Euclidean => x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt(),
        Metric::Cosine => {
            let dot: f64 = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let nx = x.iter().map(|p| p * p).sum::<f64>().sqrt();
            let ny = y.iter().map(|q| q * q).sum::<f64>().sqrt();
            1.0 - dot / (nx * ny)
        }
    }
}

/// Textbook DTW over the full cost matrix (1-based, `h[0][0] = 0`), cells
/// outside the band at infinity.
fn naive_matrix(a: &[f64], na: usize, b: &[f64], nb: usize, dim: usize, metric: Metric, window: Option<usize>) -> Vec<Vec<f64>> {
    let r = window.map_or(na.max(nb), |w| w.max(na.abs_diff(nb)));
    let mut h = vec![vec![f64::INFINITY; nb + 1]; na + 1];
    h[0][0] = 0.0;
    for i in 1..=na {
        for j in 1..=nb {
            if (i as isize - j as isize).unsigned_abs() <= r {
                h[i][j] = h[i - 1][j].min(h[i][j - 1]).min(h[i - 1][j - 1]) + frame_dist(a, b, dim, metric, i - 1, j - 1);
            }
        }
    }
    h
}

fn naive(a: &[f64], na: usize, b: &[f64], nb: usize, dim: usize, metric: Metric, window: Option<usize>) -> f64 {
    naive_matrix(a, na, b, nb, dim, metric, window)[na][nb]
}

/// Backtrack through a [`naive_matrix`] with cdtw's tie-breaking: the
/// diagonal, then up (along `a`), and left only when strictly cheaper.
fn naive_path(h: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (h.len() - 1, h[0].len() - 1);
    let mut path = vec![(i - 1, j - 1)];
    while (i, j) != (1, 1) {
        if i == 1 {
            j -= 1;
        } else if j == 1 {
            i -= 1;
        } else {
            let (up, diag, left) = (h[i - 1][j], h[i - 1][j - 1], h[i][j - 1]);
            if left < up.min(diag) {
                j -= 1;
            } else if diag <= up {
                (i, j) = (i - 1, j - 1);
            } else {
                i -= 1;
            }
        }
        path.push((i - 1, j - 1));
    }
    path.reverse();
    path
}

fn assert_close(got: f64, want: f64, rtol: f64, ctx: &str) {
    assert!(
        (got - want).abs() <= rtol * want.abs().max(1.0),
        "{ctx}: got {got}, want {want}"
    );
}

fn opts(metric: Metric, window: Option<usize>) -> Options {
    Options { metric, window }
}

#[test]
fn matches_naive_across_block_boundaries() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    // lengths straddle the DP block (8) and the f64/f32 tile widths (8/16)
    let lens = [1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 33];
    for &dim in &[1, 3, 13] {
        for &na in &lens {
            for &nb in &lens {
                let a = rng.seq(na, dim);
                let b = rng.seq(nb, dim);
                for metric in [Metric::Euclidean, Metric::Cosine] {
                    for window in [None, Some(0), Some(1), Some(3), Some(100)] {
                        let want = naive(&a, na, &b, nb, dim, metric, window);
                        let ctx = format!("{na}x{nb} d{dim} {metric:?} {window:?}");
                        let got = dtw(Seq::new(&a, na, dim).unwrap(), Seq::new(&b, nb, dim).unwrap(), &opts(metric, window), Some(1)).unwrap();
                        assert_close(got, want, 1e-12, &ctx);

                        let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                        let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
                        let got32 = dtw(Seq::new(&a32, na, dim).unwrap(), Seq::new(&b32, nb, dim).unwrap(), &opts(metric, window), Some(1)).unwrap();
                        assert_close(got32, want, 1e-4, &format!("{ctx} f32"));
                    }
                }
            }
        }
    }
}

#[test]
fn identical_sequences_have_zero_euclidean_cost() {
    let mut rng = Rng(7);
    let a = rng.seq(50, 13);
    let s = Seq::new(&a, 50, 13).unwrap();
    assert_eq!(dtw(s, s, &Options::default(), Some(1)).unwrap(), 0.0);
}

#[test]
fn window_is_widened_to_length_difference() {
    let mut rng = Rng(11);
    let (a, b) = (rng.seq(5, 2), rng.seq(40, 2));
    let (sa, sb) = (Seq::new(&a, 5, 2).unwrap(), Seq::new(&b, 40, 2).unwrap());
    let got = dtw(sa, sb, &opts(Metric::Euclidean, Some(0)), Some(1)).unwrap();
    assert!(got.is_finite());
    assert_close(got, naive(&a, 5, &b, 40, 2, Metric::Euclidean, Some(35)), 1e-12, "widened");
}

#[test]
fn cosine_with_zero_frame_is_nan() {
    let a = [0.0, 0.0, 1.0, 2.0];
    let b = [1.0, 1.0, 2.0, 2.0];
    let (sa, sb) = (Seq::new(&a, 2, 2).unwrap(), Seq::new(&b, 2, 2).unwrap());
    let o = opts(Metric::Cosine, None);
    assert!(dtw(sa, sb, &o, Some(1)).unwrap().is_nan());
    assert!(dtw(sb, sa, &o, Some(1)).unwrap().is_nan());
}

#[test]
fn zero_features() {
    let e: [f64; 0] = [];
    let (sa, sb) = (Seq::new(&e, 3, 0).unwrap(), Seq::new(&e, 4, 0).unwrap());
    assert_eq!(dtw(sa, sb, &Options::default(), Some(1)).unwrap(), 0.0);
    assert!(dtw(sa, sb, &opts(Metric::Cosine, None), Some(1)).unwrap().is_nan());
}

#[test]
fn errors() {
    let x = [1.0; 6];
    assert!(Seq::new(&x, 4, 2).is_err());
    let (s32, s23, empty) = (Seq::new(&x, 3, 2).unwrap(), Seq::new(&x, 2, 3).unwrap(), Seq::new(&x[..0], 0, 2).unwrap());
    let o = Options::default();
    assert!(dtw(s32, s23, &o, Some(1)).is_err());
    assert!(dtw(s32, empty, &o, Some(1)).is_err());
    assert!(cdist(&[s32], &[s23], &o, None).is_err());
    assert!(pdist(&[s32, empty], &o, None).is_err());
    assert!(cdist(&[s32], &[s32], &o, Some(0)).is_err());
}

fn random_set(rng: &mut Rng, n: usize, dim: usize) -> Vec<(Vec<f64>, usize)> {
    (0..n)
        .map(|_| {
            let len = 1 + (rng.next().abs() * 30.0) as usize;
            (rng.seq(len, dim), len)
        })
        .collect()
}

fn seqs(set: &[(Vec<f64>, usize)], dim: usize) -> Vec<Seq<'_, f64>> {
    set.iter().map(|(d, n)| Seq::new(d, *n, dim).unwrap()).collect()
}

#[test]
fn cdist_and_pdist_match_pairwise_dtw() {
    let mut rng = Rng(42);
    let dim = 5;
    let (set_a, set_b) = (random_set(&mut rng, 13, dim), random_set(&mut rng, 7, dim));
    let (a, b) = (seqs(&set_a, dim), seqs(&set_b, dim));
    for metric in [Metric::Euclidean, Metric::Cosine] {
        for window in [None, Some(2)] {
            let o = opts(metric, window);
            for n_jobs in [None, Some(1), Some(3)] {
                let d = cdist(&a, &b, &o, n_jobs).unwrap();
                assert_eq!(d.len(), a.len() * b.len());
                for (i, &x) in a.iter().enumerate() {
                    for (j, &y) in b.iter().enumerate() {
                        assert_eq!(d[i * b.len() + j], dtw(x, y, &o, Some(1)).unwrap());
                    }
                }
                let p = pdist(&a, &o, n_jobs).unwrap();
                let mut k = 0;
                for i in 0..a.len() {
                    for j in i + 1..a.len() {
                        assert_eq!(p[k], dtw(a[i], a[j], &o, Some(1)).unwrap(), "pdist ({i},{j})");
                        k += 1;
                    }
                }
                assert_eq!(k, p.len());
            }
        }
    }
}

#[test]
fn empty_collections() {
    let o = Options::default();
    let none: [Seq<'_, f64>; 0] = [];
    let x = [1.0, 2.0];
    let one = [Seq::new(&x, 1, 2).unwrap()];
    assert!(cdist(&none, &one, &o, None).unwrap().is_empty());
    assert!(pdist(&none, &o, None).unwrap().is_empty());
    assert!(pdist(&one, &o, None).unwrap().is_empty());
}

// ------------------------------------------------------------------ paths

/// `path` is a warping path from corner to corner, inside the band, whose
/// summed frame distances equal `cost`.
#[allow(clippy::too_many_arguments)]
fn check_path(path: &[(usize, usize)], a: &[f64], na: usize, b: &[f64], nb: usize, dim: usize, o: &Options, cost: f64, rtol: f64, ctx: &str) {
    assert_eq!(path.first(), Some(&(0, 0)), "{ctx}: start");
    assert_eq!(path.last(), Some(&(na - 1, nb - 1)), "{ctx}: end");
    for w in path.windows(2) {
        let step = (w[1].0.checked_sub(w[0].0), w[1].1.checked_sub(w[0].1));
        assert!(matches!(step, (Some(0), Some(1)) | (Some(1), Some(0)) | (Some(1), Some(1))), "{ctx}: step {w:?}");
    }
    let r = o.window.map_or(usize::MAX, |w| w.max(na.abs_diff(nb)));
    assert!(path.iter().all(|&(i, j)| i.abs_diff(j) <= r), "{ctx}: outside band");
    let sum: f64 = path.iter().map(|&(i, j)| frame_dist(a, b, dim, o.metric, i, j)).sum();
    assert_close(sum, cost, rtol, &format!("{ctx}: path cost"));
}

#[test]
fn path_matches_naive() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    let lens = [1, 2, 7, 8, 9, 17, 33];
    for &dim in &[1, 5] {
        for &na in &lens {
            for &nb in &lens {
                let (a, b) = (rng.seq(na, dim), rng.seq(nb, dim));
                for metric in [Metric::Euclidean, Metric::Cosine] {
                    for window in [None, Some(0), Some(2)] {
                        let o = opts(metric, window);
                        let ctx = format!("{na}x{nb} d{dim} {metric:?} {window:?}");
                        let (sa, sb) = (Seq::new(&a, na, dim).unwrap(), Seq::new(&b, nb, dim).unwrap());
                        let (cost, path) = dtw_path(sa, sb, &o, Some(1)).unwrap();
                        assert_eq!(cost, dtw(sa, sb, &o, Some(1)).unwrap(), "{ctx}: cost differs from dtw");
                        assert_eq!(path, naive_path(&naive_matrix(&a, na, &b, nb, dim, metric, window)), "{ctx}");
                        check_path(&path, &a, na, &b, nb, dim, &o, cost, 1e-12, &ctx);

                        // f32 may take a different (near-)optimal path, but it must be valid
                        let (a32, b32): (Vec<f32>, Vec<f32>) =
                            (a.iter().map(|&x| x as f32).collect(), b.iter().map(|&x| x as f32).collect());
                        let (cost32, path32) =
                            dtw_path(Seq::new(&a32, na, dim).unwrap(), Seq::new(&b32, nb, dim).unwrap(), &o, Some(1)).unwrap();
                        check_path(&path32, &a, na, &b, nb, dim, &o, cost32, 1e-4, &format!("{ctx} f32"));
                    }
                }
            }
        }
    }
}

#[test]
fn path_edge_cases() {
    // zero features: all distances 0, ties everywhere -> diagonal first
    let e: [f64; 0] = [];
    let (cost, path) = dtw_path(Seq::new(&e, 3, 0).unwrap(), Seq::new(&e, 4, 0).unwrap(), &Options::default(), Some(1)).unwrap();
    assert_eq!(cost, 0.0);
    assert_eq!(path, [(0, 0), (0, 1), (1, 2), (2, 3)]);
    // undefined cost: no path
    let (a, b) = ([0.0, 0.0, 1.0, 2.0], [1.0, 1.0, 2.0, 2.0]);
    let (cost, path) = dtw_path(Seq::new(&a, 2, 2).unwrap(), Seq::new(&b, 2, 2).unwrap(), &opts(Metric::Cosine, None), Some(1)).unwrap();
    assert!(cost.is_nan() && path.is_empty());
    // single frames [1, 2] and [2, 2]
    let (cost, path) = dtw_path(Seq::new(&a[2..], 1, 2).unwrap(), Seq::new(&b[2..], 1, 2).unwrap(), &Options::default(), Some(1)).unwrap();
    assert_eq!((cost, path), (1.0, vec![(0, 0)]));
}

// ------------------------------------------------------------------ parallel

/// The rectangle-parallel DP gives bit-identical costs and paths to the
/// sequential one, with rectangles small enough that every input has many.
#[allow(clippy::too_many_arguments)]
fn check_parallel<T: Float>(a: &[T], na: usize, b: &[T], nb: usize, dim: usize, o: &Options, g: wavefront::Geometry, ctx: &str) {
    let (sa, sb) = (Seq::new(a, na, dim).unwrap(), Seq::new(b, nb, dim).unwrap());
    let rows = kernel::prepare_rows(&sa, o.metric);
    let cols = kernel::prepare_cols(&sb, o.metric);
    let band = kernel::Band::new(na, nb, o.window);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut dirs = kernel::Dirs::new(&band);
    let (cost_path, cost) = pool.install(|| {
        (
            wavefront::dtw_parallel::<T, true>(&rows, &cols, o.metric, &band, Some(dirs.view()), g),
            wavefront::dtw_parallel::<T, false>(&rows, &cols, o.metric, &band, None, g),
        )
    });
    let (want, want_path) = dtw_path(sa, sb, o, Some(1)).unwrap();
    assert_eq!(cost.to_bits(), want.to_bits(), "{ctx}: cost {cost} vs {want}");
    assert_eq!(cost_path.to_bits(), want.to_bits(), "{ctx}: cost with path");
    assert_eq!(dirs.backtrack(&band), want_path, "{ctx}: path");
}

#[test]
fn parallel_rectangles_match_sequential() {
    let mut rng = Rng(0x94D049BB133111EB);
    let geometries = [
        wavefront::Geometry { row_blocks: 1, cols: 16 },
        wavefront::Geometry { row_blocks: 2, cols: 32 },
    ];
    for (na, nb) in [(1, 1), (5, 40), (40, 5), (64, 64), (65, 130), (130, 65), (200, 203)] {
        for dim in [1, 3] {
            let (a, b) = (rng.seq(na, dim), rng.seq(nb, dim));
            let (a32, b32): (Vec<f32>, Vec<f32>) = (a.iter().map(|&x| x as f32).collect(), b.iter().map(|&x| x as f32).collect());
            for metric in [Metric::Euclidean, Metric::Cosine] {
                for window in [None, Some(0), Some(5), Some(30)] {
                    for g in geometries {
                        let o = opts(metric, window);
                        let ctx = format!("{na}x{nb} d{dim} {metric:?} {window:?} {g:?}");
                        check_parallel(&a, na, &b, nb, dim, &o, g, &ctx);
                        check_parallel(&a32, na, &b32, nb, dim, &o, g, &format!("{ctx} f32"));
                    }
                }
            }
        }
    }
}

#[test]
fn public_api_runs_large_pairs_in_parallel() {
    // above MIN_PARALLEL_CELLS, so n_jobs > 1 takes the parallel path
    let mut rng = Rng(5);
    let (na, nb, dim) = (1024, 1100, 2);
    assert!(na * nb >= wavefront::MIN_PARALLEL_CELLS);
    let (a, b) = (rng.seq(na, dim), rng.seq(nb, dim));
    let (sa, sb) = (Seq::new(&a, na, dim).unwrap(), Seq::new(&b, nb, dim).unwrap());
    for window in [None, Some(50)] {
        let o = opts(Metric::Euclidean, window);
        let seq = dtw(sa, sb, &o, Some(1)).unwrap();
        assert_eq!(dtw(sa, sb, &o, Some(4)).unwrap().to_bits(), seq.to_bits());
        assert_eq!(dtw(sa, sb, &o, None).unwrap().to_bits(), seq.to_bits());
        assert_eq!(dtw_path(sa, sb, &o, Some(4)).unwrap(), dtw_path(sa, sb, &o, Some(1)).unwrap());
    }
    assert!(dtw(sa, sb, &Options::default(), Some(0)).is_err());
}
