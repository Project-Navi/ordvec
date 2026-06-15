// Stage-1 SignBitmap scan-kernel A/B for BGE-style dims.
// Times `score_all_batched_flat` (the per-query dense Hamming scan, the
// stage-1 candidate-gen kernel) at a given dim. On origin/main, dim=768
// (qpv=12) takes the SCALAR fallback; on the avx512-tail branch it takes
// AVX-512 VPOPCNTDQ with a masked tail. Same public call, same inputs.
//
//   cargo run --release --example bge_kernel_bench -- <dim> <n> <batch> <reps>
use ordvec::SignBitmap;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dim: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let batch: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(40);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let corpus: Vec<f32> = (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut idx = SignBitmap::new(dim);
    idx.add(&corpus);
    let queries: Vec<f32> = (0..batch * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    // Warmup.
    for _ in 0..3 {
        std::hint::black_box(idx.score_all_batched_flat(&queries));
    }

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let s = idx.score_all_batched_flat(&queries);
        let us = t.elapsed().as_secs_f64() * 1e6 / batch as f64;
        std::hint::black_box(&s);
        samples.push(us);
    }
    let med = median(samples.clone());
    let p10 = {
        let mut v = samples.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 10]
    };
    println!(
        "dim={dim} n={n} batch={batch} reps={reps} qpv={} -> scan median {med:.2} us/query (p10 {p10:.2})",
        dim / 64,
    );
}
