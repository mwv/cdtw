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

/// Textbook DTW over the full cost matrix, cells outside the band at infinity.
fn naive(a: &[f64], na: usize, b: &[f64], nb: usize, dim: usize, metric: Metric, window: Option<usize>) -> f64 {
    let r = window.map_or(na.max(nb), |w| w.max(na.abs_diff(nb)));
    let dist = |i: usize, j: usize| {
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
    };
    let mut h = vec![vec![f64::INFINITY; nb + 1]; na + 1];
    h[0][0] = 0.0;
    for i in 1..=na {
        for j in 1..=nb {
            if (i as isize - j as isize).unsigned_abs() <= r {
                h[i][j] = h[i - 1][j].min(h[i][j - 1]).min(h[i - 1][j - 1]) + dist(i - 1, j - 1);
            }
        }
    }
    h[na][nb]
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
                        let got = dtw(Seq::new(&a, na, dim).unwrap(), Seq::new(&b, nb, dim).unwrap(), &opts(metric, window)).unwrap();
                        assert_close(got, want, 1e-12, &ctx);

                        let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                        let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
                        let got32 = dtw(Seq::new(&a32, na, dim).unwrap(), Seq::new(&b32, nb, dim).unwrap(), &opts(metric, window)).unwrap();
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
    assert_eq!(dtw(s, s, &Options::default()).unwrap(), 0.0);
}

#[test]
fn window_is_widened_to_length_difference() {
    let mut rng = Rng(11);
    let (a, b) = (rng.seq(5, 2), rng.seq(40, 2));
    let (sa, sb) = (Seq::new(&a, 5, 2).unwrap(), Seq::new(&b, 40, 2).unwrap());
    let got = dtw(sa, sb, &opts(Metric::Euclidean, Some(0))).unwrap();
    assert!(got.is_finite());
    assert_close(got, naive(&a, 5, &b, 40, 2, Metric::Euclidean, Some(35)), 1e-12, "widened");
}

#[test]
fn cosine_with_zero_frame_is_nan() {
    let a = [0.0, 0.0, 1.0, 2.0];
    let b = [1.0, 1.0, 2.0, 2.0];
    let (sa, sb) = (Seq::new(&a, 2, 2).unwrap(), Seq::new(&b, 2, 2).unwrap());
    let o = opts(Metric::Cosine, None);
    assert!(dtw(sa, sb, &o).unwrap().is_nan());
    assert!(dtw(sb, sa, &o).unwrap().is_nan());
}

#[test]
fn zero_features() {
    let e: [f64; 0] = [];
    let (sa, sb) = (Seq::new(&e, 3, 0).unwrap(), Seq::new(&e, 4, 0).unwrap());
    assert_eq!(dtw(sa, sb, &Options::default()).unwrap(), 0.0);
    assert!(dtw(sa, sb, &opts(Metric::Cosine, None)).unwrap().is_nan());
}

#[test]
fn errors() {
    let x = [1.0; 6];
    assert!(Seq::new(&x, 4, 2).is_err());
    let (s32, s23, empty) = (Seq::new(&x, 3, 2).unwrap(), Seq::new(&x, 2, 3).unwrap(), Seq::new(&x[..0], 0, 2).unwrap());
    let o = Options::default();
    assert!(dtw(s32, s23, &o).is_err());
    assert!(dtw(s32, empty, &o).is_err());
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
                        assert_eq!(d[i * b.len() + j], dtw(x, y, &o).unwrap());
                    }
                }
                let p = pdist(&a, &o, n_jobs).unwrap();
                let mut k = 0;
                for i in 0..a.len() {
                    for j in i + 1..a.len() {
                        assert_eq!(p[k], dtw(a[i], a[j], &o).unwrap(), "pdist ({i},{j})");
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
