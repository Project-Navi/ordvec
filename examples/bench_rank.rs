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
use turbovec::rank_index::search_asymmetric_byte_lut;
use turbovec::{BitmapIndex, MultiBucketBitmapIndex, RankIndex, RankQuantIndex, TurboQuantIndex};

#[derive(Clone)]
struct Config {
    dim: usize,
    n: usize,
    n_queries: usize,
    k: usize,
    n_clusters: usize,
    latent_dim: usize,
    encode_threads_note: bool,
    /// Optional path to a NumPy .npy file holding the corpus as
    /// `(n, dim)` little-endian float32. When set, `--n` and `--dim`
    /// are overridden by the file's shape.
    corpus_npy: Option<String>,
    /// Optional path to a NumPy .npy file holding queries as
    /// `(n_q, dim)` little-endian float32.
    queries_npy: Option<String>,
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
        corpus_npy: None,
        queries_npy: None,
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
            "--corpus-npy" => cfg.corpus_npy = Some(args.next().unwrap()),
            "--queries-npy" => cfg.queries_npy = Some(args.next().unwrap()),
            other => panic!("unknown arg {other}"),
        }
    }
    cfg
}

/// Minimal NumPy v1 .npy reader for 2-D little-endian float32 arrays.
///
/// Returns `(flat_data_row_major, n, dim)`. Panics with a descriptive
/// message on any format deviation we don't support (non-f32 dtype,
/// fortran order, version != 1.x).
fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    assert!(bytes.len() >= 10, "npy too short");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a numpy file");
    let major = bytes[6];
    let minor = bytes[7];
    assert!(
        major == 1 || major == 2,
        "unsupported npy version {major}.{minor}",
    );
    let (header_len, header_start) = if major == 1 {
        let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hl, 10)
    } else {
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hl, 12)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("npy header not utf-8");
    // Header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (207695, 1024), }
    assert!(header.contains("'descr': '<f4'"), "expected <f4 dtype");
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order",
    );
    let shape_start = header.find("'shape':").expect("no shape in header");
    let after = &header[shape_start..];
    let open = after.find('(').unwrap();
    let close = after.find(')').unwrap();
    let dims: Vec<usize> = after[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    assert_eq!(dims.len(), 2, "expected 2-D array, got {} dims", dims.len());
    let n = dims[0];
    let dim = dims[1];
    let data_start = header_start + header_len;
    let n_floats = n * dim;
    assert_eq!(
        bytes.len() - data_start,
        n_floats * 4,
        "data length mismatch",
    );
    let mut out = vec![0.0f32; n_floats];
    for (i, chunk) in bytes[data_start..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    (out, n, dim)
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
    /// Effective single-query throughput: n / p50.
    docs_per_sec: f64,
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
    let docs_per_sec = if p50_s > 0.0 {
        (n as f64) / p50_s
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
        docs_per_sec,
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

/// Bench the byte-LUT alternative scoring path for the same
/// RankQuantIndex (asymmetric only, bits in {2, 4}). Returns one
/// row labelled `RankQuant b={bits} asym byte-LUT`.
fn bench_rankquant_byte_lut(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
) -> Row {
    let mut idx = RankQuantIndex::new(cfg.dim, bits);
    let t0 = Instant::now();
    idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = idx.bytes_per_vec();
    let total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = search_asymmetric_byte_lut(&idx, q, cfg.k);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
        search_asymmetric_byte_lut(&idx, q, cfg.k).indices
    });
    let recall = recall_at_k(&pred, truth, cfg.k);
    finalise_row(
        format!("RankQuant b={bits} asym byte-LUT"),
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

/// Bench the standalone top-bucket bitmap scan (no exact rerank).
/// `n_top` is the bitmap's set-bit count per doc (e.g., dim/4 for the
/// "top quarter" / b=2-equivalent operating point).
fn bench_bitmap(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    n_top: usize,
) -> Row {
    let mut idx = BitmapIndex::new(cfg.dim, n_top);
    let t0 = Instant::now();
    idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = idx.bytes_per_vec();
    let total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = idx.search(q, cfg.k);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
        idx.search(q, cfg.k).indices
    });
    let recall = recall_at_k(&pred, truth, cfg.k);
    finalise_row(
        format!("Bitmap n_top={n_top}"),
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

/// Two-stage: bitmap candidate generator (top M by overlap) then
/// exact RankQuant b=`bits` asymmetric rerank on the M candidates.
/// One row per (bits, M) pair.
///
/// `exact_rq_top` is the precomputed exact-RankQuant top-`k` indices
/// per query (length `nq * k`). When provided, the row's name is
/// suffixed with the candidate-recall: the fraction of exact-RQ
/// top-`k` doc IDs present in the bitmap's M-candidate set, averaged
/// over queries. This is the *ANN probe quality* metric, distinct
/// from the task R@10 (which compares against FP32 brute-force
/// ground truth).
fn bench_two_stage(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
    m: usize,
    n_top: usize,
    exact_rq_top: Option<&[i64]>,
) -> Row {
    let mut bitmap = BitmapIndex::new(cfg.dim, n_top);
    let mut rq = RankQuantIndex::new(cfg.dim, bits);
    let t0 = Instant::now();
    bitmap.add(corpus);
    rq.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = bitmap.bytes_per_vec() + rq.bytes_per_vec();
    let total_mib = (bitmap.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let two_stage = |q: &[f32]| -> Vec<i64> {
        // Stage 1: bitmap → top-M candidate indices.
        let cands = bitmap.top_m_candidates(q, m);
        // Stage 2: exact RankQuant scoring restricted to the candidate
        // subset — no per-query index rebuild, the existing rq.packed
        // buffer is reused; the helper gathers candidate bytes into a
        // small contiguous scan buffer and runs the AVX-512 kernel.
        let (_scores, global) = rq.search_asymmetric_subset(q, &cands, cfg.k);
        global
    };

    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = two_stage(q);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| two_stage(q));
    let recall = recall_at_k(&pred, truth, cfg.k);

    // Candidate-recall metric: for each query, how many of the
    // exact-RankQuant top-k indices are present in the bitmap's M-
    // candidate set? Reports the ANN probe quality.
    let cand_recall_label = if let Some(exact) = exact_rq_top {
        use std::collections::HashSet;
        let mut hits = 0usize;
        let mut total = 0usize;
        for qi in 0..cfg.n_queries {
            let q = &queries[qi * cfg.dim..(qi + 1) * cfg.dim];
            let cands = bitmap.top_m_candidates(q, m);
            let cand_set: HashSet<i64> = cands.iter().map(|&i| i as i64).collect();
            let exact_top: &[i64] = &exact[qi * cfg.k..(qi + 1) * cfg.k];
            for &di in exact_top {
                if di >= 0 && cand_set.contains(&di) {
                    hits += 1;
                }
                total += 1;
            }
        }
        let cr = hits as f32 / total.max(1) as f32;
        format!(" CR={cr:.3}")
    } else {
        String::new()
    };
    finalise_row(
        format!("TwoStage b={bits} M={m}{cand_recall_label}"),
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

/// Multi-bucket bitmap as a candidate generator: scores all docs by
/// the bilinear bucket-overlap with `weights`, takes top-M, then
/// reruns the exact RankQuant b=`bits` asymmetric kernel on those M.
/// Reports candidate-recall against the exact RankQuant top-k baseline.
fn bench_multi_bucket_two_stage(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
    m: usize,
    weight_label: &str,
    weight_filter: impl Fn(usize, usize, usize) -> bool + Sync + Send + Copy,
    exact_rq_top: Option<&[i64]>,
) -> Row {
    let mut mb = MultiBucketBitmapIndex::new(cfg.dim, bits);
    let mut rq = RankQuantIndex::new(cfg.dim, bits);
    let t0 = Instant::now();
    mb.add(corpus);
    rq.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = mb.bytes_per_vec() + rq.bytes_per_vec();
    let total_mib = (mb.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let nb = mb.n_buckets();
    let c = (nb as f32 - 1.0) / 2.0;
    let mut w = vec![0.0f32; nb * nb];
    for a in 0..nb {
        for b in 0..nb {
            if weight_filter(a, b, nb) {
                w[a * nb + b] = (a as f32 - c) * (b as f32 - c);
            }
        }
    }
    let w_ref = w.as_slice();

    let two_stage = |q: &[f32]| -> Vec<i64> {
        let q_bitmaps = mb.query_bitmaps_from_ranks(q);
        let cands = mb.top_m_bilinear(&q_bitmaps, w_ref, m);
        let (_, global) = rq.search_asymmetric_subset(q, &cands, cfg.k);
        global
    };

    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = two_stage(q);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| two_stage(q));
    let recall = recall_at_k(&pred, truth, cfg.k);

    let cand_recall_label = if let Some(exact) = exact_rq_top {
        use std::collections::HashSet;
        let mut hits = 0usize;
        let mut total = 0usize;
        for qi in 0..cfg.n_queries {
            let q = &queries[qi * cfg.dim..(qi + 1) * cfg.dim];
            let q_bitmaps = mb.query_bitmaps_from_ranks(q);
            let cands = mb.top_m_bilinear(&q_bitmaps, w_ref, m);
            let cand_set: HashSet<i64> = cands.iter().map(|&i| i as i64).collect();
            let exact_top: &[i64] = &exact[qi * cfg.k..(qi + 1) * cfg.k];
            for &di in exact_top {
                if di >= 0 && cand_set.contains(&di) {
                    hits += 1;
                }
                total += 1;
            }
        }
        let cr = hits as f32 / total.max(1) as f32;
        format!(" CR={cr:.3}")
    } else {
        String::new()
    };
    finalise_row(
        format!("MB b={bits} {weight_label} M={m}{cand_recall_label}"),
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
        "{:<26} {:>10} {:>10} {:>13} {:>9} {:>9} {:>8} {:>8} {:>14} {:>8}",
        "mode", "bytes/vec", "total MiB", "encode v/s", "p50 ms", "p99 ms", "GiB/s", "ns/dim", "Mdocs/s scan", "R@10",
    );
    println!("{}", "-".repeat(126));
    for r in rows {
        println!(
            "{:<26} {:>10} {:>10.1} {:>13.0} {:>9.3} {:>9.3} {:>8.2} {:>8.3} {:>14.2} {:>8.4}",
            r.name,
            r.bytes_per_vec,
            r.total_mib,
            r.encode_vecs_per_sec,
            r.p50_ms,
            r.p99_ms,
            r.gib_per_sec,
            r.ns_per_dim,
            r.docs_per_sec / 1_000_000.0,
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
            "{{\"name\":\"{}\",\"bytes_per_vec\":{},\"total_mib\":{:.3},\"encode_vps\":{:.1},\"p50_ms\":{:.4},\"p99_ms\":{:.4},\"gib_per_sec\":{:.3},\"ns_per_dim\":{:.4},\"docs_per_sec\":{:.1},\"recall_at_10_vs_fp32\":{:.4}}}",
            r.name, r.bytes_per_vec, r.total_mib, r.encode_vecs_per_sec, r.p50_ms, r.p99_ms, r.gib_per_sec, r.ns_per_dim, r.docs_per_sec, r.recall_at_10_vs_fp32,
        );
    }
    println!("]}}");
}

fn main() {
    let mut cfg = parse_args();

    // Bench environment capture — exposes the CPU features the kernels
    // actually compile against, the rayon thread pool, and rust-version
    // so a published table is reproducible.
    eprintln!(
        "target arch {} / opt-level 3 + lto (release profile)",
        std::env::consts::ARCH,
    );
    #[cfg(target_arch = "x86_64")]
    {
        let feats = [
            ("sse4.2", is_x86_feature_detected!("sse4.2")),
            ("avx2", is_x86_feature_detected!("avx2")),
            ("fma", is_x86_feature_detected!("fma")),
            ("avx512f", is_x86_feature_detected!("avx512f")),
            ("avx512bw", is_x86_feature_detected!("avx512bw")),
            ("avx512vl", is_x86_feature_detected!("avx512vl")),
        ];
        let on: Vec<&str> = feats.iter().filter(|(_, v)| *v).map(|(n, _)| *n).collect();
        eprintln!("x86_64 features detected: {}", on.join(", "));
    }
    if cfg.encode_threads_note {
        let threads = rayon::current_num_threads();
        eprintln!(
            "rayon threads = {threads} (encode + brute-force GT are parallelised; \
             per-query latency rows measure single-thread scan)",
        );
    }

    let (corpus, queries) = if let (Some(corpus_path), Some(queries_path)) =
        (cfg.corpus_npy.clone(), cfg.queries_npy.clone())
    {
        eprintln!("loading corpus {} ...", corpus_path);
        let t0 = Instant::now();
        let (corpus, n, dim) = load_npy_f32(&corpus_path);
        eprintln!("  loaded n={} dim={} in {:.2}s", n, dim, t0.elapsed().as_secs_f64());
        eprintln!("loading queries {} ...", queries_path);
        let t0 = Instant::now();
        let (queries, n_q, q_dim) = load_npy_f32(&queries_path);
        assert_eq!(q_dim, dim, "query dim {q_dim} != corpus dim {dim}");
        let n_q_take = cfg.n_queries.min(n_q);
        let queries: Vec<f32> = queries[..n_q_take * dim].to_vec();
        eprintln!(
            "  loaded n_q={} dim={} in {:.2}s (using first {} for the bench)",
            n_q,
            q_dim,
            t0.elapsed().as_secs_f64(),
            n_q_take,
        );
        cfg.dim = dim;
        cfg.n = n;
        cfg.n_queries = n_q_take;
        (corpus, queries)
    } else {
        let t0 = Instant::now();
        eprintln!(
            "generating low-rank clustered corpus (clusters={}, latent={}) ...",
            cfg.n_clusters, cfg.latent_dim,
        );
        let (corpus, queries, _q_clusters) = make_clustered_corpus(&cfg, 1);
        eprintln!("  done in {:.2}s", t0.elapsed().as_secs_f64());
        (corpus, queries)
    };

    eprintln!(
        "bench_rank: dim={} n={} queries={} k={}",
        cfg.dim, cfg.n, cfg.n_queries, cfg.k,
    );

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
    eprintln!("benching RankQuant b=2 byte-LUT ...");
    all_rows.push(bench_rankquant_byte_lut(&corpus, &queries, &truth, &cfg, 2));
    eprintln!("benching RankQuant b=4 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 4));
    eprintln!("benching RankQuant b=4 byte-LUT ...");
    all_rows.push(bench_rankquant_byte_lut(&corpus, &queries, &truth, &cfg, 4));
    eprintln!("benching RankQuant b=1 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 1));

    let n_top = cfg.dim / 4;
    eprintln!("benching Bitmap (n_top={n_top}, b=2-equivalent) ...");
    all_rows.push(bench_bitmap(&corpus, &queries, &truth, &cfg, n_top));

    // Precompute exact-RankQuant b=2 top-k per query so two-stage
    // rows can report candidate-recall (ANN probe quality, distinct
    // from task R@10).
    eprintln!("computing exact RankQuant b=2 top-k for candidate-recall metric ...");
    let mut rq_exact = RankQuantIndex::new(cfg.dim, 2);
    rq_exact.add(&corpus);
    let rq_top: Vec<i64> = collect_preds(&queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
        rq_exact.search_asymmetric(q, cfg.k).indices
    });

    for &m in &[100usize, 500, 1000, 5000] {
        eprintln!("benching TwoStage b=2 M={m} ...");
        all_rows.push(bench_two_stage(
            &corpus,
            &queries,
            &truth,
            &cfg,
            2,
            m,
            n_top,
            Some(&rq_top),
        ));
    }

    // Multi-bucket bitmap b=4 probe: tests the bilinear bucket-overlap
    // decomposition empirically as a candidate generator. Outer-product
    // weights make the score algebraically equal to symmetric RankQuant
    // (verified by tests/rank_index.rs::multi_bucket_bilinear_*).
    eprintln!("computing exact RankQuant b=4 top-k for candidate-recall metric ...");
    let mut rq_b4_exact = RankQuantIndex::new(cfg.dim, 4);
    rq_b4_exact.add(&corpus);
    let rq_b4_top: Vec<i64> =
        collect_preds(&queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
            rq_b4_exact.search_asymmetric(q, cfg.k).indices
        });

    // Three weight schemes: full 16x16, diagonal-only, top-heavy
    // (top 4 buckets only, 4x4 = 16 pair interactions).
    let weight_filters: &[(&str, &(dyn Fn(usize, usize, usize) -> bool + Sync))] = &[
        ("full16x16", &(|_a: usize, _b: usize, _nb: usize| true)),
        ("diag", &(|a: usize, b: usize, _nb: usize| a == b)),
        ("top4", &(|a: usize, b: usize, nb: usize| a + 4 >= nb && b + 4 >= nb)),
    ];
    for &m in &[100usize, 500, 1000] {
        for &(label, _) in weight_filters {
            eprintln!("benching MB b=4 {label} M={m} ...");
            // Re-construct the closure inline because the trait-object
            // form above is not Copy.
            let row = match label {
                "full16x16" => bench_multi_bucket_two_stage(
                    &corpus,
                    &queries,
                    &truth,
                    &cfg,
                    4,
                    m,
                    label,
                    |_a, _b, _nb| true,
                    Some(&rq_b4_top),
                ),
                "diag" => bench_multi_bucket_two_stage(
                    &corpus,
                    &queries,
                    &truth,
                    &cfg,
                    4,
                    m,
                    label,
                    |a, b, _nb| a == b,
                    Some(&rq_b4_top),
                ),
                "top4" => bench_multi_bucket_two_stage(
                    &corpus,
                    &queries,
                    &truth,
                    &cfg,
                    4,
                    m,
                    label,
                    |a, b, nb| a + 4 >= nb && b + 4 >= nb,
                    Some(&rq_b4_top),
                ),
                _ => unreachable!(),
            };
            all_rows.push(row);
        }
    }

    println!();
    print_table(&all_rows);
    println!();
    eprintln!("JSON:");
    print_json(&all_rows, &cfg);
}
