//! Tight loop over one DTW pair, for profilers (`samply record`, Instruments)
//! and assembly inspection without Python in the way.
//!
//!     cargo run --release -p cdtw-core --example profile -- \
//!         [len] [dim] [euclidean|cosine] [f32|f64] [window|none] [threads] [path]
//!
//! `threads` is 1 by default; 0 means all cores. A 7th argument `path` also
//! computes the warping path.

use std::time::Instant;

use cdtw_core::{dtw, dtw_path, Float, Metric, Options, Seq};

fn run<T: Float>(data: Vec<T>, n: usize, m: usize, dim: usize, opts: Options, n_jobs: Option<usize>, path: bool) {
    let (a, b) = data.split_at(n * dim);
    let (a, b) = (Seq::new(a, n, dim).unwrap(), Seq::new(b, m, dim).unwrap());
    let cells = (n * m) as f64;
    let iters = ((2e9 / (cells * dim.max(4) as f64)) as usize).max(3);
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..iters {
        acc += if path {
            let (cost, p) = dtw_path(a, b, &opts, n_jobs).unwrap();
            cost + p.len() as f64
        } else {
            dtw(a, b, &opts, n_jobs).unwrap()
        };
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "{n}x{m} d{dim} {:?} {} threads={} path={path}: {:.3} ms/pair, {:.1} Mcells/s (checksum {acc:.6})",
        opts.metric,
        std::any::type_name::<T>(),
        n_jobs.map_or("all".to_string(), |j| j.to_string()),
        per * 1e3,
        cells / per / 1e6
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| args.get(i).cloned().unwrap_or_else(|| d.to_string());
    let n: usize = arg(1, "1000").parse().unwrap();
    let dim: usize = arg(2, "39").parse().unwrap();
    let metric = match arg(3, "euclidean").as_str() {
        "cosine" => Metric::Cosine,
        _ => Metric::Euclidean,
    };
    let window = arg(5, "none").parse().ok();
    let n_jobs = match arg(6, "1").parse().unwrap() {
        0 => None,
        j => Some(j),
    };
    let path = arg(7, "") == "path";
    let m = n + n / 10;
    let mut state = 0x2545F4914F6CDD1Du64;
    let data: Vec<f64> = (0..(n + m) * dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    let opts = Options { metric, window };
    if arg(4, "f64") == "f32" {
        run(data.iter().map(|&x| x as f32).collect(), n, m, dim, opts, n_jobs, path)
    } else {
        run(data, n, m, dim, opts, n_jobs, path)
    }
}
