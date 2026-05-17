//! Head-to-head benchmark: TurboQuant vs RankIndex / RankQuantIndex.
//!
//! Measures, on a single synthetic corpus:
//! - bytes per document
//! - encode throughput (vectors / second)
//! - single-query latency p50 / p99 (top-10)
//! - recall@10 against FP32 brute-force cosine ground truth
//!
//! Run with:
//!     cargo run --release --example bench_rank
//!     cargo run --release --example bench_rank -- --dim 1024 --n 100000 --queries 200
//!
//! Output is two human-readable tables followed by a JSON line for
//! downstream tooling.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use turbovec::{RankIndex, RankQuantIndex, TurboQuantIndex};

#[derive(Clone, Copy)]
struct Config {
    dim: usize,
    n: usize,
    n_queries: usize,
    k: usize,
    n_clusters: usize,
    latent_dim: usize,
    encode_threads_note: bool,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        dim: 1024,
        n: 50_000,
        n_queries: 200,
        k: 10,
        n_clusters: 200,
        latent_dim: 64,
        encode_threads_note: true,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dim" => cfg.dim = args.next().unwrap().parse().unwrap(),
            "--n" => cfg.n = args.next().unwrap().parse().unwrap(),
            "--queries" => cfg.n_queries = args.next().unwrap().parse().unwrap(),
            "--k" => cfg.k = args.next().unwrap().parse().unwrap(),
            "--clusters" => cfg.n_clusters = args.next().unwrap().parse().unwrap(),
            "--latent" => cfg.latent_dim = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    cfg
}

/// Sample a single standard-normal value.
fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.gen_range(1e-9..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Low-rank clustered corpus and matched-cluster queries.
///
/// Construction:
///   - Sample a `dim x latent_dim` projection `A` with N(0,1) entries.
///   - Sample `n_clusters` latent prototypes in `R^latent_dim`.
///   - Each corpus doc picks a random cluster, samples `z = proto + noise_doc`,
///     and embeds as `normalize(A @ z)`.
///   - Each query picks a random cluster, samples `z = proto + noise_q` (smaller
///     noise than the doc), and embeds the same way.
///
/// Returns `(corpus, queries, query_cluster_id)`. The cluster IDs are the
/// "loose" ground truth for Tier-B recall (set-overlap with cluster
/// members), and FP32 brute-force on the same data is the "tight"
/// ground truth.
fn make_clustered_corpus(cfg: &Config, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let d = cfg.dim;
    let l = cfg.latent_dim;
    // Projection A: dim x latent.
    let mut a = vec![0.0f32; d * l];
    for x in a.iter_mut() {
        *x = gauss(&mut rng);
    }
    // Cluster prototypes in latent space.
    let mut protos = vec![0.0f32; cfg.n_clusters * l];
    for x in protos.iter_mut() {
        *x = gauss(&mut rng);
    }
    let noise_doc = 0.3_f32;
    let noise_q = 0.1_f32;

    let make_embedding = |proto: &[f32], noise_scale: f32, rng: &mut ChaCha8Rng| -> Vec<f32> {
        let mut z = vec![0.0f32; l];
        for j in 0..l {
            z[j] = proto[j] + noise_scale * gauss(rng);
        }
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            let mut acc = 0.0f32;
            for j in 0..l {
                acc += a[i * l + j] * z[j];
            }
            out[i] = acc;
        }
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in out.iter_mut() {
                *x *= inv;
            }
        }
        out
    };

    let mut corpus = Vec::with_capacity(cfg.n * d);
    for _ in 0..cfg.n {
        let c = rng.gen_range(0..cfg.n_clusters);
        let proto = &protos[c * l..(c + 1) * l];
        corpus.extend_from_slice(&make_embedding(proto, noise_doc, &mut rng));
    }
    let mut queries = Vec::with_capacity(cfg.n_queries * d);
    let mut q_clusters = Vec::with_capacity(cfg.n_queries);
    for _ in 0..cfg.n_queries {
        let c = rng.gen_range(0..cfg.n_clusters);
        q_clusters.push(c);
        let proto = &protos[c * l..(c + 1) * l];
        queries.extend_from_slice(&make_embedding(proto, noise_q, &mut rng));
    }
    (corpus, queries, q_clusters)
}

/// FP32 brute-force top-k cosine ground truth.
/// Returns a Vec of top-k indices per query (length nq * k).
fn fp32_ground_truth(corpus: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<i64> {
    use rayon::prelude::*;
    let n = corpus.len() / dim;
    let nq = queries.len() / dim;
    let mut out = vec![-1i64; nq * k];
    out.par_chunks_mut(k)
        .zip(queries.par_chunks(dim))
        .for_each(|(out_slot, q)| {
            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|di| {
                    let doc = &corpus[di * dim..(di + 1) * dim];
                    let dot: f32 = q.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
                    (di, dot)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (slot, (di, _)) in scored.into_iter().take(k).enumerate() {
                out_slot[slot] = di as i64;
            }
        });
    out
}

fn recall_at_k(pred: &[i64], truth: &[i64], k: usize) -> f32 {
    use std::collections::HashSet;
    assert_eq!(pred.len(), truth.len());
    let nq = pred.len() / k;
    let mut hits = 0usize;
    let mut total = 0usize;
    for qi in 0..nq {
        let p: HashSet<i64> = pred[qi * k..(qi + 1) * k].iter().copied().collect();
        let t: HashSet<i64> = truth[qi * k..(qi + 1) * k].iter().copied().collect();
        hits += p.intersection(&t).count();
        total += k;
    }
    hits as f32 / total as f32
}

fn percentile_us(samples: &mut [u128], p: f32) -> f64 {
    samples.sort_unstable();
    let i = ((samples.len() as f32 - 1.0) * p).round() as usize;
    samples[i] as f64 / 1_000.0
}

#[derive(Debug, Clone)]
struct Row {
    name: String,
    bytes_per_vec: usize,
    total_mib: f64,
    encode_vecs_per_sec: f64,
    p50_ms: f64,
    p99_ms: f64,
    recall_at_10_vs_fp32: f32,
    /// Effective scan bandwidth at p50: bytes_per_vec * n / p50.
    gib_per_sec: f64,
    /// Per-coordinate p50 time in nanoseconds.
    ns_per_dim: f64,
}

fn finalise_row(
    name: String,
    bytes_per_vec: usize,
    total_mib: f64,
    encode_vps: f64,
    p50_ms: f64,
    p99_ms: f64,
    recall: f32,
    n: usize,
    dim: usize,
) -> Row {
    let p50_s = p50_ms / 1_000.0;
    let scanned_bytes = (bytes_per_vec as f64) * (n as f64);
    let gib_per_sec = if p50_s > 0.0 {
        scanned_bytes / p50_s / (1024.0 * 1024.0 * 1024.0)
    } else {
        f64::NAN
    };
    let ns_per_dim = if n > 0 {
        (p50_ms * 1_000_000.0) / ((n as f64) * (dim as f64))
    } else {
        f64::NAN
    };
    Row {
        name,
        bytes_per_vec,
        total_mib,
        encode_vecs_per_sec: encode_vps,
        p50_ms,
        p99_ms,
        recall_at_10_vs_fp32: recall,
        gib_per_sec,
        ns_per_dim,
    }
}

/// Runs the supplied search closure once per query and returns
/// (p50_ms, p99_ms). Performs a small untimed warmup before recording.
fn time_queries<F>(queries: &[f32], dim: usize, n_queries: usize, mut search_one: F) -> (f64, f64)
where
    F: FnMut(&[f32]),
{
    let warmup = 5.min(n_queries);
    for qi in 0..warmup {
        search_one(&queries[qi * dim..(qi + 1) * dim]);
    }
    let mut samples = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let t0 = Instant::now();
        search_one(q);
        samples.push(t0.elapsed().as_nanos());
    }
    let p50 = percentile_us(&mut samples.clone(), 0.50) / 1_000.0;
    let p99 = percentile_us(&mut samples, 0.99) / 1_000.0;
    (p50, p99)
}

/// Run the search closure once per query, collecting results.
fn collect_preds<F>(queries: &[f32], dim: usize, n_queries: usize, k: usize, mut search_one: F) -> Vec<i64>
where
    F: FnMut(&[f32]) -> Vec<i64>,
{
    let mut out = Vec::with_capacity(n_queries * k);
    for qi in 0..n_queries {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let idx = search_one(q);
        debug_assert_eq!(idx.len(), k);
        out.extend_from_slice(&idx);
    }
    out
}

fn bench_turboquant(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: usize,
) -> Row {
    let mut idx = TurboQuantIndex::new(cfg.dim, bits);
    let t0 = Instant::now();
    idx.add(corpus);
    idx.prepare();
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = cfg.dim * bits / 8;
    let total_mib = (bytes_per_vec * cfg.n) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = idx.search(q, cfg.k);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
        idx.search(q, cfg.k).indices
    });
    let recall = recall_at_k(&pred, truth, cfg.k);
    finalise_row(
        format!("TurboQuant b={bits}"),
        bytes_per_vec,
        total_mib,
        encode_vps,
        p50,
        p99,
        recall,
        cfg.n,
        cfg.dim,
    )
}

fn bench_rank_full(corpus: &[f32], queries: &[f32], truth: &[i64], cfg: &Config) -> Vec<Row> {
    let mut idx = RankIndex::new(cfg.dim);
    let t0 = Instant::now();
    idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = idx.bytes_per_vec();
    let total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let mut rows = Vec::new();
    for &(label, asym) in &[("RankIndex sym", false), ("RankIndex asym", true)] {
        let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
            let _ = if asym {
                idx.search_asymmetric(q, cfg.k)
            } else {
                idx.search(q, cfg.k)
            };
        });
        let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
            if asym {
                idx.search_asymmetric(q, cfg.k).indices
            } else {
                idx.search(q, cfg.k).indices
            }
        });
        let recall = recall_at_k(&pred, truth, cfg.k);
        rows.push(finalise_row(
            label.to_string(),
            bytes_per_vec,
            total_mib,
            encode_vps,
            p50,
            p99,
            recall,
            cfg.n,
            cfg.dim,
        ));
    }
    rows
}

fn bench_rankquant(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
) -> Vec<Row> {
    let mut idx = RankQuantIndex::new(cfg.dim, bits);
    let t0 = Instant::now();
    idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = idx.bytes_per_vec();
    let total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let mut rows = Vec::new();
    for &(label_suffix, asym) in &[("sym", false), ("asym", true)] {
        let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
            let _ = if asym {
                idx.search_asymmetric(q, cfg.k)
            } else {
                idx.search(q, cfg.k)
            };
        });
        let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
            if asym {
                idx.search_asymmetric(q, cfg.k).indices
            } else {
                idx.search(q, cfg.k).indices
            }
        });
        let recall = recall_at_k(&pred, truth, cfg.k);
        rows.push(finalise_row(
            format!("RankQuant b={bits} {label_suffix}"),
            bytes_per_vec,
            total_mib,
            encode_vps,
            p50,
            p99,
            recall,
            cfg.n,
            cfg.dim,
        ));
    }
    rows
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<22} {:>10} {:>10} {:>13} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "mode", "bytes/vec", "total MiB", "encode v/s", "p50 ms", "p99 ms", "GiB/s", "ns/dim", "R@10",
    );
    println!("{}", "-".repeat(110));
    for r in rows {
        println!(
            "{:<22} {:>10} {:>10.1} {:>13.0} {:>9.3} {:>9.3} {:>9.2} {:>9.3} {:>8.4}",
            r.name,
            r.bytes_per_vec,
            r.total_mib,
            r.encode_vecs_per_sec,
            r.p50_ms,
            r.p99_ms,
            r.gib_per_sec,
            r.ns_per_dim,
            r.recall_at_10_vs_fp32,
        );
    }
}

fn print_json(rows: &[Row], cfg: &Config) {
    print!("{{");
    print!("\"dim\":{},", cfg.dim);
    print!("\"n\":{},", cfg.n);
    print!("\"queries\":{},", cfg.n_queries);
    print!("\"k\":{},", cfg.k);
    print!("\"rows\":[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"bytes_per_vec\":{},\"total_mib\":{:.3},\"encode_vps\":{:.1},\"p50_ms\":{:.4},\"p99_ms\":{:.4},\"gib_per_sec\":{:.3},\"ns_per_dim\":{:.4},\"recall_at_10_vs_fp32\":{:.4}}}",
            r.name, r.bytes_per_vec, r.total_mib, r.encode_vecs_per_sec, r.p50_ms, r.p99_ms, r.gib_per_sec, r.ns_per_dim, r.recall_at_10_vs_fp32,
        );
    }
    println!("]}}");
}

fn main() {
    let cfg = parse_args();
    eprintln!(
        "bench_rank: dim={} n={} queries={} k={}",
        cfg.dim, cfg.n, cfg.n_queries, cfg.k,
    );
    if cfg.encode_threads_note {
        let threads = rayon::current_num_threads();
        eprintln!(
            "rayon threads = {threads} (encode + brute-force GT are parallelised)",
        );
    }

    let t0 = Instant::now();
    eprintln!(
        "generating low-rank clustered corpus (clusters={}, latent={}) ...",
        cfg.n_clusters, cfg.latent_dim,
    );
    let (corpus, queries, _q_clusters) = make_clustered_corpus(&cfg, 1);
    eprintln!("  done in {:.2}s", t0.elapsed().as_secs_f64());

    eprintln!("FP32 brute-force ground truth ...");
    let t0 = Instant::now();
    let truth = fp32_ground_truth(&corpus, &queries, cfg.dim, cfg.k);
    eprintln!("  done in {:.2}s", t0.elapsed().as_secs_f64());

    let mut all_rows = Vec::new();

    eprintln!("benching TurboQuant b=2 ...");
    all_rows.push(bench_turboquant(&corpus, &queries, &truth, &cfg, 2));
    eprintln!("benching TurboQuant b=4 ...");
    all_rows.push(bench_turboquant(&corpus, &queries, &truth, &cfg, 4));

    eprintln!("benching RankIndex (full u16) ...");
    all_rows.extend(bench_rank_full(&corpus, &queries, &truth, &cfg));

    eprintln!("benching RankQuant b=2 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 2));
    eprintln!("benching RankQuant b=4 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 4));
    eprintln!("benching RankQuant b=1 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 1));

    println!();
    print_table(&all_rows);
    println!();
    eprintln!("JSON:");
    print_json(&all_rows, &cfg);
}
