//! Head-to-head benchmark for the rank-mode index family:
//! Rank, RankQuant (b=1/2/4), Bitmap (single-stage and
//! two-stage candidate-gen + exact rerank), and SignBitmap.
//!
//! SELF-CONTAINED BY DEFAULT. The default run needs NO external corpus
//! file: it generates a seeded (seed = `CORPUS_SEED`) low-rank clustered
//! synthetic corpus in-process, so the headline numbers are regenerable
//! from a clean checkout with a single command:
//!
//!     cargo run --release --example bench_rank
//!
//! No system dependencies are required — ordvec links no BLAS.
//!
//! Measures, per index type:
//! - bytes per document and total index size
//! - encode throughput (vectors / second)
//! - single-query latency p50 / p99 (top-10) + derived scan bandwidth
//! - recall@10 against FP32 brute-force cosine ground truth
//! - candidate-recall (CR) for the two-stage modes (ANN probe quality)
//!
//! The bitmap rows also serve as an empirical complement to the Lean
//! constant-weight overlap model: they measure whether this synthetic or
//! user-supplied corpus behaves like the monotone-overlap regime assumed by the
//! theorem.
//!
//! DETERMINISM. The corpus, queries, and ground truth are fully seeded,
//! so every QUALITY column (R@10, CR, bytes/vec, total MiB, ns/dim) is
//! bit-identical across runs on the same machine. Only the wall-clock
//! THROUGHPUT/LATENCY columns (encode v/s, p50/p99 ms, GiB/s, Mdocs/s)
//! vary run-to-run as expected. A committed capture of one run lives at
//! `benchmarks/rank_modes_results.txt`.
//!
//! Larger sweeps / real public corpora:
//!     cargo run --release --example bench_rank -- --dim 1024 --n 100000 --queries 200
//!     # Point at a real public embedding corpus (no file required for
//!     # the default run). Both must be 2-D little-endian float32 .npy
//!     # (C order). For GloVe or OpenAI text-embedding-3 dumps:
//!     cargo run --release --example bench_rank -- \
//!         --corpus-npy /path/to/corpus.npy --queries-npy /path/to/queries.npy
//!
//! Output is a human-readable table followed by a JSON line for
//! downstream tooling.

use ordvec::search_asymmetric_byte_lut;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
// `RankQuantFastscan` is `#[doc(hidden)]` (optional b=2 scan path);
// the bench imports it to compare its throughput/recall against the
// production RankQuant b=2 asym kernel on identical data.
use ordvec::RankQuantFastscan;
use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};

/// Fixed RNG seed for the synthetic corpus + queries. Pinning this is
/// what makes the recall/CR columns reproducible run-to-run. Change it
/// only if you intend to regenerate the committed results artifact.
const CORPUS_SEED: u64 = 1;

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
    /// Optional path. When set, every bench function appends one JSONL
    /// row per query with its top-K doc ids, so downstream tooling can
    /// compute R@K for K' <= K without re-running the bench.
    /// Schema: {"qid_idx": int, "mode": str, "k": int, "doc_ids": [int]}
    /// where doc_ids of length k; -1 entries are sentinels for modes
    /// that cannot return k candidates (e.g. two-stage with M < k).
    dump_top_k_jsonl: Option<String>,
    /// Optional mode filter. When set, only rows whose row.name
    /// matches one of these tags are included. Supported tags:
    /// "bitmap" (single-query bitmap scan), "batched-two-stage"
    /// (multi-query batched candidate gen + rerank at the default M
    /// sweep), "batch-sweep" (varies --batch across {1,2,4,8,16} at
    /// M=500), "sign-headline" (sign-cosine vs rank-bitmap at matched
    /// storage), "storage-matched" (TwoStage b=1 rerank at 256 B/vec).
    /// Unset = run the full bench suite.
    mode: Option<String>,
    /// Batch size for batched scan modes. The batched kernel streams
    /// the doc bitmaps once and computes overlap scores against
    /// `batch` queries in parallel, amortising L3→core bandwidth.
    /// Default = 8.
    batch: usize,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        // Defaults chosen so the *self-contained synthetic* run is cheap
        // and reproducible: dim=256, n=30k, 200 queries finishes in well
        // under a minute on a laptop-class core, while still giving clean
        // recall separation between the rank-mode index types. Override
        // with --dim / --n / --queries for larger sweeps, or point at a
        // real public corpus with --corpus-npy / --queries-npy (see the
        // header comment for the GloVe / OpenAI .npy recipe).
        dim: 256,
        n: 30_000,
        n_queries: 200,
        k: 10,
        n_clusters: 200,
        latent_dim: 64,
        encode_threads_note: true,
        corpus_npy: None,
        queries_npy: None,
        dump_top_k_jsonl: None,
        mode: None,
        batch: 8,
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
            "--dump-top-k-jsonl" => cfg.dump_top_k_jsonl = Some(args.next().unwrap()),
            "--mode" => cfg.mode = Some(args.next().unwrap()),
            "--batch" => cfg.batch = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(cfg.batch >= 1, "--batch must be >= 1");
    cfg
}

/// Append per-query top-K JSONL rows to `path`. Each line is one query's
/// top-K for a single mode. `pred` must be a flat `n_queries * k` slice
/// of doc indices; modes that cannot return k candidates should pad with
/// -1 sentinels before calling.
fn dump_pred_jsonl(path: &str, mode: &str, n_queries: usize, k: usize, pred: &[i64]) {
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};
    debug_assert_eq!(
        pred.len(),
        n_queries * k,
        "pred buffer length must equal n_queries * k"
    );
    let f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("dump_pred_jsonl: open");
    let mut w = BufWriter::new(f);
    for qi in 0..n_queries {
        let row = &pred[qi * k..(qi + 1) * k];
        write!(
            &mut w,
            r#"{{"qid_idx":{qi},"mode":"{mode}","k":{k},"doc_ids":["#
        )
        .unwrap();
        for (i, &di) in row.iter().enumerate() {
            if i > 0 {
                w.write_all(b",").unwrap();
            }
            write!(&mut w, "{di}").unwrap();
        }
        writeln!(&mut w, "]}}").unwrap();
    }
}

/// Conditional wrapper: dumps pred only when `cfg.dump_top_k_jsonl` is set.
fn maybe_dump_pred(cfg: &Config, mode: &str, pred: &[i64]) {
    if let Some(ref path) = cfg.dump_top_k_jsonl {
        dump_pred_jsonl(path, mode, cfg.n_queries, cfg.k, pred);
    }
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
    let u1: f32 = rng.random_range(1e-9..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
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
        let c = rng.random_range(0..cfg.n_clusters);
        let proto = &protos[c * l..(c + 1) * l];
        corpus.extend_from_slice(&make_embedding(proto, noise_doc, &mut rng));
    }
    let mut queries = Vec::with_capacity(cfg.n_queries * d);
    let mut q_clusters = Vec::with_capacity(cfg.n_queries);
    for _ in 0..cfg.n_queries {
        let c = rng.random_range(0..cfg.n_clusters);
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

/// "Candidate ceiling" recall: given a system that returns `k_out`
/// candidates per query, what fraction of the FP32 top-`k_eval`
/// ground truth is contained in those candidates? Equivalently: the
/// upper bound on R@`k_eval` that a perfect reranker over the
/// system's top-`k_out` could deliver.
///
/// `pred` is shape `n_queries × k_out` (the system's candidate set).
/// `truth_topk_eval` is shape `n_queries × k_eval` (FP32 top-k_eval).
#[allow(dead_code)] // diagnostic utility — no current mode invokes it
fn ceiling_recall(
    pred: &[i64],
    k_out: usize,
    truth_topk_eval: &[i64],
    k_eval: usize,
    n_queries: usize,
) -> f32 {
    use std::collections::HashSet;
    let mut hits = 0usize;
    let mut total = 0usize;
    for qi in 0..n_queries {
        let pred_set: HashSet<i64> = pred[qi * k_out..(qi + 1) * k_out].iter().copied().collect();
        let truth_row = &truth_topk_eval[qi * k_eval..(qi + 1) * k_eval];
        for &di in truth_row {
            if di >= 0 && pred_set.contains(&di) {
                hits += 1;
            }
            total += 1;
        }
    }
    hits as f32 / total.max(1) as f32
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

#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
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
fn collect_preds<F>(
    queries: &[f32],
    dim: usize,
    n_queries: usize,
    k: usize,
    mut search_one: F,
) -> Vec<i64>
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

fn bench_rank_full(corpus: &[f32], queries: &[f32], truth: &[i64], cfg: &Config) -> Vec<Row> {
    let mut idx = Rank::new(cfg.dim);
    let t0 = Instant::now();
    idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = idx.bytes_per_vec();
    let total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    let mut rows = Vec::new();
    for &(label, asym) in &[("Rank sym", false), ("Rank asym", true)] {
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
        maybe_dump_pred(cfg, label, &pred);
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
/// RankQuant (asymmetric only, bits in {2, 4}). Returns one
/// row labelled `RankQuant b={bits} asym byte-LUT`.
fn bench_rankquant_byte_lut(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
) -> Row {
    let mut idx = RankQuant::new(cfg.dim, bits);
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
    let name = format!("RankQuant b={bits} asym byte-LUT");
    maybe_dump_pred(cfg, &name, &pred);
    finalise_row(
        name,
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
fn bench_bitmap(corpus: &[f32], queries: &[f32], truth: &[i64], cfg: &Config, n_top: usize) -> Row {
    let mut idx = Bitmap::new(cfg.dim, n_top);
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
    let name = format!("Bitmap n_top={n_top}");
    maybe_dump_pred(cfg, &name, &pred);
    finalise_row(
        name,
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
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
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
    let mut bitmap = Bitmap::new(cfg.dim, n_top);
    let mut rq = RankQuant::new(cfg.dim, bits);
    let t0 = Instant::now();
    bitmap.add(corpus);
    rq.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = bitmap.bytes_per_vec() + rq.bytes_per_vec();
    let total_mib = (bitmap.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;

    // When the caller asks for more results than the bitmap stage can
    // produce (cfg.k > m), the rerank only returns m results — pad the
    // remainder with -1 sentinels so downstream R@K computation sees a
    // uniform-length pred buffer. -1 never matches a real doc id, so
    // recall_at_k is unaffected by the padding entries (they correctly
    // surface the candidate-set ceiling at K > M).
    let effective_k = cfg.k.min(m);
    let two_stage = |q: &[f32]| -> Vec<i64> {
        // Stage 1: bitmap → top-M candidate indices.
        let cands = bitmap.top_m_candidates(q, m);
        // Stage 2: exact RankQuant scoring restricted to the candidate
        // subset — no per-query index rebuild, the existing rq.packed
        // buffer is reused; the helper gathers candidate bytes into a
        // small contiguous scan buffer and runs the AVX-512 kernel.
        let (_scores, mut global) = rq.search_asymmetric_subset(q, &cands, effective_k);
        global.resize(cfg.k, -1);
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
    // Dump uses the prefix-stable name (no CR suffix) so downstream
    // tooling can join across runs where CR varies by encoder/corpus.
    let dump_name = format!("TwoStage b={bits} M={m}");
    maybe_dump_pred(cfg, &dump_name, &pred);
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

/// Batched two-stage: streams the bitmap corpus once for groups of
/// `batch_size` queries, then runs the existing exact RankQuant
/// `search_asymmetric_subset` per-query rerank. Per-query effective
/// latency = batch wall time / batch_size.
///
/// Reported p50 / p99 are over **per-query effective samples** — each
/// query in a given batch shares that batch's wall time / batch_size,
/// so within-batch variance is zero and across-batch variance is
/// captured directly. This makes the row directly comparable to the
/// existing single-query TwoStage rows.
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
fn bench_two_stage_batched(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
    m: usize,
    n_top: usize,
    batch_size: usize,
    exact_rq_top: Option<&[i64]>,
) -> Row {
    let mut bitmap = Bitmap::new(cfg.dim, n_top);
    let mut rq = RankQuant::new(cfg.dim, bits);
    let t0 = Instant::now();
    bitmap.add(corpus);
    rq.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = bitmap.bytes_per_vec() + rq.bytes_per_vec();
    let total_mib = (bitmap.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;
    let effective_k = cfg.k.min(m);

    // Warmup: one full batch so the kernel allocations + first-touch
    // page faults don't pollute the first measured batch.
    let warm_n = batch_size.min(cfg.n_queries);
    if warm_n > 0 {
        let _ = bitmap.top_m_candidates_batched(&queries[..warm_n * cfg.dim], m);
    }

    let mut samples: Vec<u128> = Vec::with_capacity(cfg.n_queries);
    let mut pred: Vec<i64> = Vec::with_capacity(cfg.n_queries * cfg.k);

    let mut batch_start = 0usize;
    while batch_start < cfg.n_queries {
        let batch_end = (batch_start + batch_size).min(cfg.n_queries);
        let b = batch_end - batch_start;
        let batch_q = &queries[batch_start * cfg.dim..batch_end * cfg.dim];

        let t0 = Instant::now();
        let cands = bitmap.top_m_candidates_batched(batch_q, m);
        let mut batch_pred = Vec::with_capacity(b * cfg.k);
        for (i, cand_set) in cands.iter().enumerate() {
            let q = &batch_q[i * cfg.dim..(i + 1) * cfg.dim];
            let (_, mut global) = rq.search_asymmetric_subset(q, cand_set, effective_k);
            global.resize(cfg.k, -1);
            batch_pred.extend(global);
        }
        let elapsed_ns = t0.elapsed().as_nanos();
        let per_query_ns = elapsed_ns / b as u128;
        for _ in 0..b {
            samples.push(per_query_ns);
        }
        pred.extend(batch_pred);
        batch_start = batch_end;
    }

    let p50 = percentile_us(&mut samples.clone(), 0.50) / 1_000.0;
    let p99 = percentile_us(&mut samples, 0.99) / 1_000.0;
    let recall = recall_at_k(&pred, truth, cfg.k);

    // Candidate-recall vs the exact RankQuant top-k baseline. Mirrors
    // the single-query bench_two_stage path. Computed in a second
    // pass over the batched candidate generator so the timing loop
    // above isn't perturbed by the HashSet construction.
    let cand_recall_label = if let Some(exact) = exact_rq_top {
        use std::collections::HashSet;
        let mut hits = 0usize;
        let mut total = 0usize;
        let mut bs = 0usize;
        while bs < cfg.n_queries {
            let be = (bs + batch_size).min(cfg.n_queries);
            let bq = &queries[bs * cfg.dim..be * cfg.dim];
            let cands = bitmap.top_m_candidates_batched(bq, m);
            for (i, c) in cands.iter().enumerate() {
                let qi = bs + i;
                let cand_set: HashSet<i64> = c.iter().map(|&x| x as i64).collect();
                let exact_top: &[i64] = &exact[qi * cfg.k..(qi + 1) * cfg.k];
                for &di in exact_top {
                    if di >= 0 && cand_set.contains(&di) {
                        hits += 1;
                    }
                    total += 1;
                }
            }
            bs = be;
        }
        let cr = hits as f32 / total.max(1) as f32;
        format!(" CR={cr:.3}")
    } else {
        String::new()
    };

    let name = format!("TwoStage b={bits} M={m} B={batch_size}{cand_recall_label}");
    let dump_name = format!("TwoStage b={bits} M={m} B={batch_size}");
    maybe_dump_pred(cfg, &dump_name, &pred);
    finalise_row(
        name,
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

/// Sign-cosine bitmap probe only (data-independent threshold at zero).
/// 128 B/vec storage at D=1024 — same byte budget as the rank-bitmap
/// probe — but the threshold is `coord > 0` rather than `rank ≥ dim
/// − n_top`. Score = `dim − popcount(q XOR d)` = sign-agreement count.
fn bench_sign_bitmap(corpus: &[f32], queries: &[f32], truth: &[i64], cfg: &Config) -> Row {
    let mut idx = SignBitmap::new(cfg.dim);
    let t0 = Instant::now();
    idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = idx.bytes_per_vec();
    let total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;
    let probe = |q: &[f32]| -> Vec<i64> {
        let cands = idx.top_m_candidates(q, cfg.k);
        let mut out = vec![-1i64; cfg.k];
        for (i, &c) in cands.iter().take(cfg.k).enumerate() {
            out[i] = c as i64;
        }
        out
    };
    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = probe(q);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| probe(q));
    let recall = recall_at_k(&pred, truth, cfg.k);
    let name = "SignBitmap probe".to_string();
    maybe_dump_pred(cfg, &name, &pred);
    finalise_row(
        name,
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

/// Sign-cosine two-stage: SignBitmap candidate gen → exact RankQuant
/// b=`bits` rerank. Direct head-to-head with the rank-bitmap
/// two-stage at the same 384 B/vec storage (128 sign + 256 RQ b=2).
fn bench_sign_two_stage(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
    m: usize,
    exact_rq_top: Option<&[i64]>,
) -> Row {
    let mut sign = SignBitmap::new(cfg.dim);
    let mut rq = RankQuant::new(cfg.dim, bits);
    let t0 = Instant::now();
    sign.add(corpus);
    rq.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = sign.bytes_per_vec() + rq.bytes_per_vec();
    let total_mib = (sign.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;
    let effective_k = cfg.k.min(m);
    let two_stage = |q: &[f32]| -> Vec<i64> {
        let cands = sign.top_m_candidates(q, m);
        let (_, mut global) = rq.search_asymmetric_subset(q, &cands, effective_k);
        global.resize(cfg.k, -1);
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
            let cands = sign.top_m_candidates(q, m);
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
    let name = format!("SignTwoStage b={bits} M={m}{cand_recall_label}");
    let dump_name = format!("SignTwoStage b={bits} M={m}");
    maybe_dump_pred(cfg, &dump_name, &pred);
    finalise_row(
        name,
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

/// Batched sign-cosine two-stage. Mirrors `bench_two_stage_batched`:
/// chunks queries into CHUNK=8 groups, streams the sign bitmaps once
/// per chunk through the AVX-512 XOR-popcount kernel, then runs the
/// existing RankQuant subset rerank per query. Per-query effective
/// latency = batch wall time / batch_size.
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
fn bench_sign_two_stage_batched(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
    bits: u8,
    m: usize,
    batch_size: usize,
    exact_rq_top: Option<&[i64]>,
) -> Row {
    let mut sign = SignBitmap::new(cfg.dim);
    let mut rq = RankQuant::new(cfg.dim, bits);
    let t0 = Instant::now();
    sign.add(corpus);
    rq.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let bytes_per_vec = sign.bytes_per_vec() + rq.bytes_per_vec();
    let total_mib = (sign.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;
    let encode_vps = cfg.n as f64 / encode_secs;
    let effective_k = cfg.k.min(m);

    let warm_n = batch_size.min(cfg.n_queries);
    if warm_n > 0 {
        let _ = sign.top_m_candidates_batched(&queries[..warm_n * cfg.dim], m);
    }

    let mut samples: Vec<u128> = Vec::with_capacity(cfg.n_queries);
    let mut pred: Vec<i64> = Vec::with_capacity(cfg.n_queries * cfg.k);
    let mut batch_start = 0usize;
    while batch_start < cfg.n_queries {
        let batch_end = (batch_start + batch_size).min(cfg.n_queries);
        let b = batch_end - batch_start;
        let batch_q = &queries[batch_start * cfg.dim..batch_end * cfg.dim];
        let t0 = Instant::now();
        let cands = sign.top_m_candidates_batched(batch_q, m);
        let mut batch_pred = Vec::with_capacity(b * cfg.k);
        for (i, cand_set) in cands.iter().enumerate() {
            let q = &batch_q[i * cfg.dim..(i + 1) * cfg.dim];
            let (_, mut global) = rq.search_asymmetric_subset(q, cand_set, effective_k);
            global.resize(cfg.k, -1);
            batch_pred.extend(global);
        }
        let elapsed_ns = t0.elapsed().as_nanos();
        let per_query_ns = elapsed_ns / b as u128;
        for _ in 0..b {
            samples.push(per_query_ns);
        }
        pred.extend(batch_pred);
        batch_start = batch_end;
    }
    let p50 = percentile_us(&mut samples.clone(), 0.50) / 1_000.0;
    let p99 = percentile_us(&mut samples, 0.99) / 1_000.0;
    let recall = recall_at_k(&pred, truth, cfg.k);

    let cand_recall_label = if let Some(exact) = exact_rq_top {
        use std::collections::HashSet;
        let mut hits = 0usize;
        let mut total = 0usize;
        let mut bs = 0usize;
        while bs < cfg.n_queries {
            let be = (bs + batch_size).min(cfg.n_queries);
            let bq = &queries[bs * cfg.dim..be * cfg.dim];
            let cands = sign.top_m_candidates_batched(bq, m);
            for (i, c) in cands.iter().enumerate() {
                let qi = bs + i;
                let cand_set: HashSet<i64> = c.iter().map(|&x| x as i64).collect();
                let exact_top: &[i64] = &exact[qi * cfg.k..(qi + 1) * cfg.k];
                for &di in exact_top {
                    if di >= 0 && cand_set.contains(&di) {
                        hits += 1;
                    }
                    total += 1;
                }
            }
            bs = be;
        }
        let cr = hits as f32 / total.max(1) as f32;
        format!(" CR={cr:.3}")
    } else {
        String::new()
    };
    let name = format!("SignTwoStage b={bits} M={m} B={batch_size}{cand_recall_label}");
    let dump_name = format!("SignTwoStage b={bits} M={m} B={batch_size}");
    maybe_dump_pred(cfg, &dump_name, &pred);
    finalise_row(
        name,
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

/// Bench the FastScan b=2 path against the same corpus/queries the
/// regular RankQuant b=2 asym bench uses. Returns one row labelled
/// `RankQuant b=2 fastscan`. The companion `RankQuant b=2 asym` row
/// from `bench_rankquant` is the apples-to-apples baseline (same
/// recall; FastScan trades 2× storage for lower scan latency).
fn bench_rankquant_fastscan_b2(
    corpus: &[f32],
    queries: &[f32],
    truth: &[i64],
    cfg: &Config,
) -> Row {
    let dim = cfg.dim;
    let n = cfg.n;

    // Build the FastScan layout once via the type wrapper. The type's
    // add() encapsulates rank-transform + bucket + pack_fastscan_b2;
    // time the whole encode for the encode-throughput column.
    let t0 = Instant::now();
    let mut fs_idx = RankQuantFastscan::new(dim);
    fs_idx.add(corpus);
    let encode_secs = t0.elapsed().as_secs_f64();
    let encode_vps = n as f64 / encode_secs;

    // bytes/vec reports the FastScan-layout storage: the block-32
    // re-blocking gives dim/2 bytes per doc — 2× the standard b=2
    // packing (dim/4). This is the well-known FastScan space cost.
    let bytes_per_vec = fs_idx.byte_size() / n.max(1);
    let total_mib = fs_idx.byte_size() as f64 / 1024.0 / 1024.0;

    let (p50, p99) = time_queries(queries, cfg.dim, cfg.n_queries, |q| {
        let _ = fs_idx.search(q, cfg.k);
    });
    let pred = collect_preds(queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
        fs_idx.search(q, cfg.k).indices
    });
    let recall = recall_at_k(&pred, truth, cfg.k);
    maybe_dump_pred(cfg, "RankQuant b=2 fastscan", &pred);
    finalise_row(
        "RankQuant b=2 fastscan".to_string(),
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
    let mut idx = RankQuant::new(cfg.dim, bits);
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
        let name = format!("RankQuant b={bits} {label_suffix}");
        maybe_dump_pred(cfg, &name, &pred);
        rows.push(finalise_row(
            name,
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
        "{:<32} {:>10} {:>10} {:>13} {:>9} {:>9} {:>8} {:>8} {:>14} {:>8}",
        "mode",
        "bytes/vec",
        "total MiB",
        "encode v/s",
        "p50 ms",
        "p99 ms",
        "GiB/s",
        "ns/dim",
        "Mdocs/s scan",
        "R@10",
    );
    println!("{}", "-".repeat(132));
    for r in rows {
        println!(
            "{:<32} {:>10} {:>10.1} {:>13.0} {:>9.3} {:>9.3} {:>8.2} {:>8.3} {:>14.2} {:>8.4}",
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
        eprintln!(
            "  loaded n={} dim={} in {:.2}s",
            n,
            dim,
            t0.elapsed().as_secs_f64()
        );
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
        let (corpus, queries, _q_clusters) = make_clustered_corpus(&cfg, CORPUS_SEED);
        eprintln!(
            "  done in {:.2}s (seed={CORPUS_SEED}, self-contained)",
            t0.elapsed().as_secs_f64()
        );
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

    // --mode filter: when set, run only the selected micro-bench so an
    // experiment isolates one kernel without the 20+ row default suite.
    if let Some(mode) = cfg.mode.clone() {
        let n_top = cfg.dim / 4;
        match mode.as_str() {
            "bitmap" => {
                eprintln!("benching Bitmap (n_top={n_top}, b=2-equivalent) ...");
                all_rows.push(bench_bitmap(&corpus, &queries, &truth, &cfg, n_top));
            }
            "batched-two-stage" => {
                // Multi-query batched candidate gen → exact RQ rerank.
                // Sweeps M ∈ {100, 500, 1000, 5000} at the configured
                // --batch size, plus the single-query M=500 row as a
                // direct head-to-head sanity-check.
                eprintln!("computing exact RankQuant b=2 top-k for CR metric ...");
                let mut rq_exact = RankQuant::new(cfg.dim, 2);
                rq_exact.add(&corpus);
                let rq_top: Vec<i64> =
                    collect_preds(&queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
                        rq_exact.search_asymmetric(q, cfg.k).indices
                    });
                eprintln!("benching single-query TwoStage b=2 M=500 (baseline) ...");
                all_rows.push(bench_two_stage(
                    &corpus,
                    &queries,
                    &truth,
                    &cfg,
                    2,
                    500,
                    n_top,
                    Some(&rq_top),
                ));
                for &m in &[100usize, 500, 1000, 5000] {
                    eprintln!("benching TwoStage b=2 M={m} B={} (batched) ...", cfg.batch,);
                    all_rows.push(bench_two_stage_batched(
                        &corpus,
                        &queries,
                        &truth,
                        &cfg,
                        2,
                        m,
                        n_top,
                        cfg.batch,
                        Some(&rq_top),
                    ));
                }
            }
            "sign-headline" => {
                // Sign-cosine vs rank-bitmap, head-to-head at matched
                // storage. The substrate test prompted by Todd's
                // schema.org-typed result, validated at the Harrier
                // scale and recall regime.
                eprintln!("benching SignBitmap probe (128 B/vec, sign-cos) ...");
                all_rows.push(bench_sign_bitmap(&corpus, &queries, &truth, &cfg));

                let bitmap_n_top = cfg.dim / 4;
                eprintln!("benching rank-Bitmap probe (n_top={bitmap_n_top}, 128 B/vec) ...");
                all_rows.push(bench_bitmap(&corpus, &queries, &truth, &cfg, bitmap_n_top));

                eprintln!("computing exact RankQuant b=2 top-k for CR ...");
                let mut rq_exact = RankQuant::new(cfg.dim, 2);
                rq_exact.add(&corpus);
                let rq_top: Vec<i64> =
                    collect_preds(&queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
                        rq_exact.search_asymmetric(q, cfg.k).indices
                    });

                // SignTwoStage (sign + RQ rerank) batched B=8 at the
                // same M sweep as the rank two-stage baseline.
                for &m in &[100usize, 500, 1000, 5000] {
                    eprintln!("benching SignTwoStage b=2 M={m} B={} ...", cfg.batch);
                    all_rows.push(bench_sign_two_stage_batched(
                        &corpus,
                        &queries,
                        &truth,
                        &cfg,
                        2,
                        m,
                        cfg.batch,
                        Some(&rq_top),
                    ));
                    eprintln!("benching rank TwoStage b=2 M={m} B={} ...", cfg.batch);
                    all_rows.push(bench_two_stage_batched(
                        &corpus,
                        &queries,
                        &truth,
                        &cfg,
                        2,
                        m,
                        bitmap_n_top,
                        cfg.batch,
                        Some(&rq_top),
                    ));
                }
            }
            "storage-matched" => {
                // Storage-matched head-to-head: TwoStage with b=1
                // RankQuant rerank (128 B bitmap + 128 B RankQuant b=1 =
                // 256 B/vec). Also runs the 384 B/vec b=2 rerank rows
                // for the existing +50% storage Pareto.
                eprintln!("computing exact RankQuant b=2 top-k for CR ...");
                let mut rq_exact = RankQuant::new(cfg.dim, 2);
                rq_exact.add(&corpus);
                let rq_top: Vec<i64> =
                    collect_preds(&queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
                        rq_exact.search_asymmetric(q, cfg.k).indices
                    });
                for &m in &[100usize, 500, 1000, 5000] {
                    eprintln!("benching TwoStage b=1 batched (256 B/vec, MATCHED) M={m} ...",);
                    all_rows.push(bench_two_stage_batched(
                        &corpus,
                        &queries,
                        &truth,
                        &cfg,
                        1,
                        m,
                        n_top,
                        cfg.batch,
                        Some(&rq_top),
                    ));
                }
                for &m in &[500usize, 5000] {
                    eprintln!("benching TwoStage b=2 batched (384 B/vec, +50%) M={m} ...",);
                    all_rows.push(bench_two_stage_batched(
                        &corpus,
                        &queries,
                        &truth,
                        &cfg,
                        2,
                        m,
                        n_top,
                        cfg.batch,
                        Some(&rq_top),
                    ));
                }
            }
            "batch-sweep" => {
                // Vary batch ∈ {1, 2, 4, 8, 16} at fixed M=500 to map
                // the bandwidth-amortisation curve. B=1 cross-checks
                // against the single-query path (should be within
                // noise; small overhead expected from the extra
                // copy in top_m_candidates_batched).
                eprintln!("computing exact RankQuant b=2 top-k for CR metric ...");
                let mut rq_exact = RankQuant::new(cfg.dim, 2);
                rq_exact.add(&corpus);
                let rq_top: Vec<i64> =
                    collect_preds(&queries, cfg.dim, cfg.n_queries, cfg.k, |q| {
                        rq_exact.search_asymmetric(q, cfg.k).indices
                    });
                eprintln!("benching single-query TwoStage b=2 M=500 (baseline) ...");
                all_rows.push(bench_two_stage(
                    &corpus,
                    &queries,
                    &truth,
                    &cfg,
                    2,
                    500,
                    n_top,
                    Some(&rq_top),
                ));
                for &b in &[1usize, 2, 4, 8, 16] {
                    eprintln!("benching TwoStage b=2 M=500 B={b} ...");
                    all_rows.push(bench_two_stage_batched(
                        &corpus,
                        &queries,
                        &truth,
                        &cfg,
                        2,
                        500,
                        n_top,
                        b,
                        Some(&rq_top),
                    ));
                }
            }
            other => panic!(
                "unknown --mode '{other}' (expected: bitmap, batched-two-stage, \
                 batch-sweep, storage-matched, sign-headline)",
            ),
        }
        println!();
        print_table(&all_rows);
        println!();
        print_json(&all_rows, &cfg);
        return;
    }

    eprintln!("benching Rank (full u16) ...");
    all_rows.extend(bench_rank_full(&corpus, &queries, &truth, &cfg));

    eprintln!("benching RankQuant b=2 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 2));
    eprintln!("benching RankQuant b=2 byte-LUT ...");
    all_rows.push(bench_rankquant_byte_lut(&corpus, &queries, &truth, &cfg, 2));
    eprintln!("benching RankQuant b=2 FastScan (optional path) ...");
    all_rows.push(bench_rankquant_fastscan_b2(&corpus, &queries, &truth, &cfg));
    eprintln!("benching RankQuant b=4 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 4));
    eprintln!("benching RankQuant b=4 byte-LUT ...");
    all_rows.push(bench_rankquant_byte_lut(&corpus, &queries, &truth, &cfg, 4));
    eprintln!("benching RankQuant b=1 ...");
    all_rows.extend(bench_rankquant(&corpus, &queries, &truth, &cfg, 1));

    let n_top = cfg.dim / 4;
    eprintln!("benching Bitmap (n_top={n_top}, b=2-equivalent) ...");
    all_rows.push(bench_bitmap(&corpus, &queries, &truth, &cfg, n_top));

    // Sign-cosine probe (data-independent threshold at zero, dim/8
    // bytes/vec). Included in the default suite so the SignBitmap
    // substrate is represented head-to-head with the rank-bitmap probe.
    eprintln!("benching SignBitmap probe (sign-cosine, dim/8 B/vec) ...");
    all_rows.push(bench_sign_bitmap(&corpus, &queries, &truth, &cfg));

    // Precompute exact-RankQuant b=2 top-k per query so two-stage
    // rows can report candidate-recall (ANN probe quality, distinct
    // from task R@10).
    eprintln!("computing exact RankQuant b=2 top-k for candidate-recall metric ...");
    let mut rq_exact = RankQuant::new(cfg.dim, 2);
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

    // Sign-cosine two-stage (SignBitmap candidate-gen → exact RankQuant
    // b=2 rerank), at a representative mid M, so the sign substrate is
    // represented in a two-stage configuration alongside the rank
    // two-stage rows above.
    eprintln!("benching SignTwoStage b=2 M=500 ...");
    all_rows.push(bench_sign_two_stage(
        &corpus,
        &queries,
        &truth,
        &cfg,
        2,
        500,
        Some(&rq_top),
    ));

    println!();
    print_table(&all_rows);
    println!();
    eprintln!("JSON:");
    print_json(&all_rows, &cfg);
}
