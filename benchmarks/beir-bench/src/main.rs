//! All-Rust BEIR comparison harness.
//!
//! Measures ordvec's rank/sign methods against an exact inner-product baseline
//! (`flat`, identical math to FAISS `IndexFlatIP`) and a pure-Rust HNSW
//! (`hnsw_rs`, Malkov–Yashunin — the faithful portable stand-in for C++ hnswlib),
//! ALL in one process so the latency comparison is genuinely apples-to-apples:
//! same machine, same batch, same thread count, no Python/FFI boundary.
//!
//! Two knobs make the comparison fair and reveal the scaling story:
//!
//! `--threads N`: query latency is measured inside a rayon pool of exactly N
//! threads (index *build* still uses all cores). N=1 gives the single-thread
//! story; N>1 the throughput story. Batch is matched across every method.
//!
//! `--max-docs M`: truncate the corpus to its first M vectors. Sweeping M
//! produces the speedup-vs-corpus-size curve (brute force is O(n); ordvec
//! sign/rank candidate-gen is near-flat in n).
//!
//! Output: `<out>/<dataset>/timing.jsonl` gets one record per
//! (method, n_docs, threads) run, appended every invocation — the plotter
//! consumes this. A FULL-corpus run (`--max-docs` absent) additionally writes
//! `<method>.topk.jsonl` + `<method>.summary.json` for offline nDCG eval;
//! sub-sampled runs skip those (qrels-based nDCG is only valid on the full
//! corpus).
//!
//! Cache layout (one encoder per prepare run):
//!   <cache-dir>/<dataset>/<split>/encoder=<slug>/
//!     corpus.f32.npy  queries.f32.npy  corpus_ids.json  query_ids.json
//!     qrels.json  embeddings.manifest.json  ...

use ordvec::{Bitmap, CandidateBatch, RankQuant, SignBitmap, SubsetScratch};
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::time::Instant;

use hnsw_rs::prelude::*;

// HNSW hyper-parameters (faithful to the prior "hnswlib M=32" comparison).
const HNSW_M: usize = 32;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_MAX_LAYER: usize = 16;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct Config {
    cache_dir: String,
    dataset: String,
    split: String,
    top_k: usize,
    batch: usize,
    candidates: usize,
    methods: Vec<String>,
    out_dir: String,
    threads: usize,          // 0 = all cores
    max_docs: Option<usize>, // None = full corpus
    ef_search: usize,        // HNSW query-time recall/latency knob (default 128)
}

fn parse_args() -> Config {
    let mut cache_dir = String::from(".cache/ordvec-beir");
    let mut dataset = String::new();
    let mut split = String::from("test");
    let mut top_k = 100usize;
    let mut batch = 8usize;
    let mut candidates = 500usize;
    let mut methods = vec![
        "flat".to_string(),
        "hnsw".to_string(),
        "rq2".to_string(),
        "rq4".to_string(),
        "bitmap-rq2".to_string(),
        "sign-rq2".to_string(),
    ];
    let mut out_dir = String::from("results/beir");
    let mut threads = 0usize;
    let mut max_docs: Option<usize> = None;
    let mut ef_search = 128usize;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ef-search" => {
                ef_search = args
                    .next()
                    .expect("--ef-search requires a value")
                    .parse()
                    .expect("--ef-search must be an integer")
            }
            "--cache-dir" => cache_dir = args.next().expect("--cache-dir requires a value"),
            "--dataset" => dataset = args.next().expect("--dataset requires a value"),
            "--split" => split = args.next().expect("--split requires a value"),
            "--top-k" => {
                top_k = args
                    .next()
                    .expect("--top-k requires a value")
                    .parse()
                    .expect("--top-k must be an integer")
            }
            "--batch" => {
                batch = args
                    .next()
                    .expect("--batch requires a value")
                    .parse()
                    .expect("--batch must be an integer")
            }
            "--candidates" => {
                candidates = args
                    .next()
                    .expect("--candidates requires a value")
                    .parse()
                    .expect("--candidates must be an integer")
            }
            "--methods" => {
                methods = args
                    .next()
                    .expect("--methods requires a value")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--out-dir" => out_dir = args.next().expect("--out-dir requires a value"),
            "--threads" => {
                threads = args
                    .next()
                    .expect("--threads requires a value")
                    .parse()
                    .expect("--threads must be an integer")
            }
            "--max-docs" => {
                max_docs = Some(
                    args.next()
                        .expect("--max-docs requires a value")
                        .parse()
                        .expect("--max-docs must be an integer"),
                )
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    assert!(!dataset.is_empty(), "--dataset is required");
    assert!(batch >= 1, "--batch must be >= 1");
    assert!(top_k >= 1, "--top-k must be >= 1");
    assert!(candidates >= 1, "--candidates must be >= 1");
    // hnsw_rs requires ef_search >= the requested neighbour count (it internally
    // clamps ef = max(ef, knbn)). An --ef-search below --top-k would otherwise be
    // silently bumped, flattening an ef sweep at the low end. Clamp explicitly +
    // warn so the sweep stays meaningful and the recorded ef matches what ran.
    let ef_search = if ef_search < top_k {
        eprintln!(
            "warning: --ef-search {ef_search} < --top-k {top_k}; clamping ef_search to {top_k} \
             (hnsw_rs requires ef >= k)"
        );
        top_k
    } else {
        ef_search
    };

    Config {
        cache_dir,
        dataset,
        split,
        top_k,
        batch,
        candidates,
        methods,
        out_dir,
        threads,
        max_docs,
        ef_search,
    }
}

// ---------------------------------------------------------------------------
// NumPy v1/v2 reader (2-D LE f32, C-order)
// ---------------------------------------------------------------------------

fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    load_npy_f32_rows(path, None)
}

fn read_npy_header(f: &mut std::fs::File, path: &str) -> (String, usize) {
    use std::io::Read;

    let mut pre = [0u8; 12];
    f.read_exact(&mut pre[..10])
        .unwrap_or_else(|e| panic!("read npy magic {path}: {e}"));
    assert_eq!(&pre[..6], b"\x93NUMPY", "not a numpy file: {path}");
    let major = pre[6];
    let minor = pre[7];
    assert!(
        major == 1 || major == 2,
        "unsupported npy version {major}.{minor}: {path}",
    );
    let (header_len, data_start) = if major == 1 {
        (u16::from_le_bytes([pre[8], pre[9]]) as usize, 10usize)
    } else {
        f.read_exact(&mut pre[10..12])
            .unwrap_or_else(|e| panic!("read npy header length {path}: {e}"));
        (
            u32::from_le_bytes([pre[8], pre[9], pre[10], pre[11]]) as usize,
            12usize,
        )
    };
    let mut hb = vec![0u8; header_len];
    f.read_exact(&mut hb)
        .unwrap_or_else(|e| panic!("read npy header {path}: {e}"));
    let header =
        String::from_utf8(hb).unwrap_or_else(|e| panic!("npy header not utf-8 {path}: {e}"));
    (header, data_start + header_len)
}

fn npy_shape(header: &str, path: &str) -> Vec<usize> {
    let after = &header[header.find("'shape':").expect("no shape in npy header")..];
    let open = after.find('(').unwrap();
    let close = after.find(')').unwrap();
    let dims: Vec<usize> = after[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    assert!(!dims.is_empty(), "empty npy shape in {path}");
    dims
}

fn npy_payload_bytes(n: usize, dim: usize, path: &str) -> usize {
    n.checked_mul(dim)
        .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>()))
        .unwrap_or_else(|| panic!("npy payload too large: {path}"))
}

/// Read just the npy header and return the row count (dim 0). Cheap: no payload read.
fn npy_row_count(path: &str) -> usize {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open npy {path}: {e}"));
    let (header, _) = read_npy_header(&mut f, path);
    npy_shape(&header, path)
        .into_iter()
        .next()
        .expect("no row count in npy shape")
}

/// Read a 2-D LE-f32 C-order npy. When `max_rows` is `Some(m)`, only the first
/// `m` rows of the payload are read off disk (so `--max-docs` subsampling does
/// NOT pull the whole 36 GB corpus into RAM). The payload is parsed in parallel
/// directly into the output `Vec<f32>` — no intermediate full `Vec<u8>` copy, so
/// peak memory is ~1× the kept data, not 2× the whole file.
fn load_npy_f32_rows(path: &str, max_rows: Option<usize>) -> (Vec<f32>, usize, usize) {
    use std::io::Read;

    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open npy {path}: {e}"));
    let (header, data_start) = read_npy_header(&mut f, path);
    assert!(
        header.contains("'descr': '<f4'"),
        "expected <f4 dtype in {path}: {header}",
    );
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order in {path}",
    );
    let dims = npy_shape(&header, path);
    assert_eq!(dims.len(), 2, "expected 2-D array in {path}");
    let (n_full, dim) = (dims[0], dims[1]);
    let n = max_rows.map_or(n_full, |m| m.min(n_full));
    let full_payload_bytes = npy_payload_bytes(n_full, dim, path);
    let expected_len = (data_start as u64)
        .checked_add(full_payload_bytes as u64)
        .unwrap_or_else(|| panic!("npy file too large: {path}"));
    let actual_len = f
        .metadata()
        .unwrap_or_else(|e| panic!("stat npy {path}: {e}"))
        .len();
    assert_eq!(actual_len, expected_len, "data length mismatch in {path}");

    let n_floats = n
        .checked_mul(dim)
        .unwrap_or_else(|| panic!("npy payload too large: {path}"));
    let mut out = vec![0.0f32; n_floats];
    let read_bytes = npy_payload_bytes(n, dim, path);
    // SAFETY: `out` is fully initialized, `f32` has no invalid bit patterns, and
    // the byte slice covers exactly the initialized backing storage.
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), read_bytes) };
    f.read_exact(out_bytes)
        .unwrap_or_else(|e| panic!("read npy payload {path}: {e}"));

    #[cfg(target_endian = "big")]
    out.par_iter_mut()
        .for_each(|v| *v = f32::from_bits(v.to_bits().swap_bytes()));

    (out, n, dim)
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn load_json_string_array(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse json string array {path}: {e}"))
}

/// SHA-256 of a file, pure Rust (no shelling out — portable, incl. Windows /
/// minimal containers). Hex-encoded; matches the Python `hashlib` digest.
fn sha256_file(path: &str) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path} for sha256: {e}"));
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn rustc_version() -> String {
    if let Ok(out) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    "unknown".to_string()
}

fn detected_simd() -> Vec<String> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut v = Vec::new();
        if is_x86_feature_detected!("avx2") {
            v.push("avx2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            v.push("fma".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            v.push("avx512f".to_string());
        }
        v
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}

fn percentile_ms(samples: &[u128], p: f32) -> f64 {
    let mut s = samples.to_vec();
    s.sort_unstable();
    if s.is_empty() {
        return 0.0;
    }
    let i = ((s.len() as f32 - 1.0) * p).round() as usize;
    s[i] as f64 / 1_000_000.0
}

// ---------------------------------------------------------------------------
// Validate embeddings (dim == 1024, unit-norm rows)
// ---------------------------------------------------------------------------

fn validate_embeddings(data: &[f32], n: usize, dim: usize, label: &str) {
    assert_eq!(dim, 1024, "{label}: embedding_dim must be 1024, got {dim}");
    assert_eq!(dim % 16, 0, "{label}: dim must be divisible by 16");
    for (i, row) in data.chunks_exact(dim).enumerate() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "{label} row {i}: L2 norm {norm:.6} not ~1.0",
        );
    }
    eprintln!("  {label}: validated {n} rows (dim={dim}, L2-normalised)");
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn open_output(out_dir: &str, dataset: &str, slug: &str, ext: &str) -> BufWriter<std::fs::File> {
    let dir = format!("{out_dir}/{dataset}");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create_dir_all {dir}: {e}"));
    let path = format!("{dir}/{slug}.{ext}");
    let f = std::fs::File::create(&path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    BufWriter::new(f)
}

/// Append-only writer for the per-config timing record stream.
fn open_timing_appender(out_dir: &str, dataset: &str) -> BufWriter<std::fs::File> {
    let dir = format!("{out_dir}/{dataset}");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create_dir_all {dir}: {e}"));
    let path = format!("{dir}/timing.jsonl");
    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"));
    BufWriter::new(f)
}

/// Write one JSONL row per query (global doc indices; -1 = padding).
#[allow(clippy::too_many_arguments)]
fn write_topk_jsonl<W: Write>(
    writer: &mut W,
    dataset: &str,
    split: &str,
    method: &str,
    k: usize,
    query_ids: &[String],
    corpus_ids: &[String],
    indices: &[i64],
    scores: &[f32],
) {
    let nq = query_ids.len();
    let n_corpus = corpus_ids.len();
    for qi in 0..nq {
        let row_indices = &indices[qi * k..(qi + 1) * k];
        let mut doc_idxs: Vec<u64> = Vec::new();
        let mut doc_ids: Vec<&str> = Vec::new();
        let mut row_scores: Vec<f64> = Vec::new();
        for (j, &di) in row_indices.iter().enumerate() {
            if di < 0 {
                break; // sentinel marks the end of this query's results
            }
            let di_usize = di as usize;
            doc_idxs.push(di_usize as u64);
            doc_ids.push(if di_usize < n_corpus {
                corpus_ids[di_usize].as_str()
            } else {
                ""
            });
            let sc = scores.get(qi * k + j).copied().unwrap_or(0.0);
            row_scores.push(if sc.is_finite() { sc as f64 } else { 0.0 });
        }
        // serde_json guarantees valid JSON (escapes quotes/backslashes/unicode in
        // doc/query IDs), so downstream `json.loads` never trips.
        let row = serde_json::json!({
            "dataset": dataset,
            "split": split,
            "method": method,
            "qid_idx": qi,
            "qid": query_ids[qi],
            "k": k,
            "doc_idxs": doc_idxs,
            "doc_ids": doc_ids,
            "scores": row_scores,
        });
        writeln!(writer, "{row}").expect("write topk jsonl");
    }
}

/// A single benchmarked configuration's record — written both to the per-method
/// summary.json (full-corpus runs) and appended to timing.jsonl (every run).
struct Record<'a> {
    dataset: &'a str,
    split: &'a str,
    method: &'a str,
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    threads: usize,
    batch: usize,
    candidates: usize,
    bytes_per_vector: usize,
    index_total_mib: f64,
    build_seconds: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    qps: f64,
    simd: &'a [String],
    encoder_sha: &'a str,
}

fn write_record_json<W: Write + ?Sized>(w: &mut W, r: &Record) {
    let rec = serde_json::json!({
        "dataset": r.dataset,
        "split": r.split,
        "method": r.method,
        "dim": r.dim,
        "n_docs": r.n_docs,
        "n_queries": r.n_queries,
        "top_k": r.top_k,
        "threads": r.threads,
        "batch": r.batch,
        "candidates": r.candidates,
        "bytes_per_vector": r.bytes_per_vector,
        "index_total_mib": r.index_total_mib,
        "build_seconds": r.build_seconds,
        "query_latency_ms_p50": r.p50_ms,
        "query_latency_ms_p95": r.p95_ms,
        "query_latency_ms_p99": r.p99_ms,
        "queries_per_second": r.qps,
        "cpu_arch": std::env::consts::ARCH,
        "simd_detected": r.simd,
        "rustc": rustc_version(),
        "crate_version": env!("CARGO_PKG_VERSION"),
        "encoder_manifest_sha256": r.encoder_sha,
    });
    writeln!(w, "{rec}").expect("write record json");
}

// ---------------------------------------------------------------------------
// Timing driver
// ---------------------------------------------------------------------------

/// Flat `nq*top_k` (indices, scores) for one query batch (sentinel -1 padded).
type Preds = (Vec<i64>, Vec<f32>);
/// Per-query latency samples (ns) + optionally the collected predictions.
type TimedRun = (Vec<u128>, Option<Preds>);

/// Warm up, time (amortized per-query over each batch), and optionally collect
/// predictions. `search_batch(b_start, b_end)` returns flat `(b_end-b_start)*top_k`
/// indices (sentinel -1 padded) and matching scores. Runs inside the caller's
/// rayon pool so query parallelism is pinned to the configured thread count.
fn time_and_collect<F>(
    n_queries: usize,
    batch: usize,
    warmup: usize,
    collect: bool,
    mut search_batch: F,
) -> TimedRun
where
    F: FnMut(usize, usize) -> Preds,
{
    // Warmup.
    let w_end = (warmup.div_ceil(batch) * batch).min(n_queries);
    let mut b_start = 0usize;
    while b_start < w_end {
        let b_end = (b_start + batch).min(n_queries);
        let _ = search_batch(b_start, b_end);
        b_start = b_end;
    }

    // Timing.
    let mut samples = Vec::with_capacity(n_queries);
    let mut preds_i: Vec<i64> = Vec::new();
    let mut preds_s: Vec<f32> = Vec::new();
    b_start = 0;
    while b_start < n_queries {
        let b_end = (b_start + batch).min(n_queries);
        let b = b_end - b_start;
        let t0 = Instant::now();
        let (idx, sc) = search_batch(b_start, b_end);
        let per_query_ns = t0.elapsed().as_nanos() / b as u128;
        for _ in 0..b {
            samples.push(per_query_ns);
        }
        if collect {
            preds_i.extend_from_slice(&idx);
            preds_s.extend_from_slice(&sc);
        }
        b_start = b_end;
    }

    let preds = if collect {
        Some((preds_i, preds_s))
    } else {
        None
    };
    (samples, preds)
}

/// Finalize one method run: percentiles, optional topk/summary write, timing record.
#[allow(clippy::too_many_arguments)]
fn finalize(
    slug: &str,
    samples: &[u128],
    preds: Option<(Vec<i64>, Vec<f32>)>,
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    threads: usize,
    batch: usize,
    candidates: usize,
    bytes_per_vector: usize,
    index_total_mib: f64,
    build_seconds: f64,
    dataset: &str,
    split: &str,
    query_ids: &[String],
    corpus_ids: &[String],
    out_dir: &str,
    simd: &[String],
    encoder_sha: &str,
    timing_writer: &mut dyn Write,
) {
    let p50 = percentile_ms(samples, 0.50);
    let p95 = percentile_ms(samples, 0.95);
    let p99 = percentile_ms(samples, 0.99);
    let qps = 1_000.0 / p50.max(f64::EPSILON);

    let rec = Record {
        dataset,
        split,
        method: slug,
        dim,
        n_docs,
        n_queries,
        top_k,
        threads,
        batch,
        candidates,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        p50_ms: p50,
        p95_ms: p95,
        p99_ms: p99,
        qps,
        simd,
        encoder_sha,
    };
    // Always append to the timing stream.
    write_record_json(timing_writer, &rec);

    // Full-corpus runs (preds collected) also write topk + per-method summary.
    if let Some((pred_i, pred_s)) = preds {
        let mut jw = open_output(out_dir, dataset, slug, "topk.jsonl");
        write_topk_jsonl(
            &mut jw, dataset, split, slug, top_k, query_ids, corpus_ids, &pred_i, &pred_s,
        );
        jw.flush().expect("flush topk");
        let mut sw = open_output(out_dir, dataset, slug, "summary.json");
        write_record_json(&mut sw, &rec);
        sw.flush().expect("flush summary");
    }

    eprintln!(
        "  {slug} [n={n_docs} t={threads}]: p50={p50:.4}ms p95={p95:.4}ms p99={p99:.4}ms qps={qps:.1}"
    );
}

// ---------------------------------------------------------------------------
// Per-query top-k from raw scores (used by the flat baseline)
// ---------------------------------------------------------------------------

/// One chunk's contribution: `nq` rows, each a local top-k of (score, global_id).
type ChunkTopK = Vec<Vec<(f32, i64)>>;

/// Local top-k of a score row, returned as (score, global_id) sorted by score
/// desc, with `id_offset` added to the local column index.
fn local_topk(row: &[f32], id_offset: usize, top_k: usize) -> Vec<(f32, i64)> {
    let mut scored: Vec<(f32, i64)> = row
        .iter()
        .enumerate()
        .map(|(j, &s)| (s, (id_offset + j) as i64))
        .collect();
    let k = top_k.min(scored.len());
    // `k > 0` guards the `k - 1` index (top_k is asserted >= 1 at the CLI, but
    // keep this defensive so a zero can never underflow to usize::MAX here).
    if k > 0 && k < scored.len() {
        scored.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
    }
    scored
}

/// Exact inner-product top-k for a whole query batch against `corpus[..n_docs]`.
/// Same math as FAISS `IndexFlatIP`: scores = Q · Dᵀ via a blocked SIMD GEMM
/// (matrixmultiply), parallelized over doc-chunks on the current rayon pool so
/// the baseline both vectorizes and scales with the configured thread count.
fn flat_batch_topk(
    qbatch: &[f32],
    nq: usize,
    corpus: &[f32],
    n_docs: usize,
    dim: usize,
    top_k: usize,
) -> (Vec<i64>, Vec<f32>) {
    // ~2 chunks per thread (≥1024 docs each) for balance without tiny GEMMs.
    let nthreads = rayon::current_num_threads().max(1);
    let target_chunks = (nthreads * 2).max(1);
    let chunk_size = n_docs.div_ceil(target_chunks).max(1024);
    let n_chunks = n_docs.div_ceil(chunk_size).max(1);

    // Per chunk → nq rows of local top-k (global ids).
    let per_chunk: Vec<ChunkTopK> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n_docs);
            let cn = end - start;
            if cn == 0 {
                return vec![Vec::new(); nq];
            }
            // C(nq × cn) = Q(nq × dim) · Dᵀ_chunk : B element (k, j) is at
            // corpus[(start+j)*dim + k] → row-stride 1, col-stride dim.
            let mut cmat = vec![0.0f32; nq * cn];
            unsafe {
                matrixmultiply::sgemm(
                    nq,
                    dim,
                    cn,
                    1.0,
                    qbatch.as_ptr(),
                    dim as isize,
                    1,
                    corpus[start * dim..end * dim].as_ptr(),
                    1,
                    dim as isize,
                    0.0,
                    cmat.as_mut_ptr(),
                    cn as isize,
                    1,
                );
            }
            (0..nq)
                .map(|qi| local_topk(&cmat[qi * cn..(qi + 1) * cn], start, top_k))
                .collect()
        })
        .collect();

    // Merge chunk-local top-k into the global top-k per query.
    let mut idx = vec![-1i64; nq * top_k];
    let mut sc = vec![0.0f32; nq * top_k];
    for qi in 0..nq {
        let mut merged: Vec<(f32, i64)> = Vec::new();
        for chunk in &per_chunk {
            merged.extend_from_slice(&chunk[qi]);
        }
        let k = top_k.min(merged.len());
        if k < merged.len() {
            merged.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            merged.truncate(k);
        }
        merged.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for (j, &(s, i)) in merged.iter().take(top_k).enumerate() {
            idx[qi * top_k + j] = i;
            sc[qi * top_k + j] = s;
        }
    }
    (idx, sc)
}

/// Pad a per-query Vec<(idx, score)> ordering into flat `top_k` rows (-1 / 0.0).
fn pad_rows(rows: Vec<Vec<(i64, f32)>>, top_k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut idx = vec![-1i64; rows.len() * top_k];
    let mut sc = vec![0.0f32; rows.len() * top_k];
    for (qi, row) in rows.iter().enumerate() {
        for (j, &(i, s)) in row.iter().take(top_k).enumerate() {
            idx[qi * top_k + j] = i;
            sc[qi * top_k + j] = s;
        }
    }
    (idx, sc)
}

// ---------------------------------------------------------------------------
// Cache resolution
// ---------------------------------------------------------------------------

fn resolve_encoder_dir(cache_dir: &str, dataset: &str, split: &str) -> String {
    let parent = format!("{cache_dir}/{dataset}/{split}");
    let entries = std::fs::read_dir(&parent).unwrap_or_else(|e| panic!("read_dir {parent}: {e}"));
    let mut matches: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("encoder=") && e.path().is_dir())
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    assert!(!matches.is_empty(), "no encoder=* subdir under {parent}");
    assert!(
        matches.len() == 1,
        "multiple encoder=* dirs under {parent}: {matches:?} — one encoder per dataset/split",
    );
    matches.remove(0)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cfg = parse_args();

    let threads_resolved = if cfg.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        cfg.threads
    };
    // Per-config query pool: build still uses all cores (default global pool);
    // query latency is pinned to `threads_resolved` via pool.install(...).
    let query_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads_resolved)
        .build()
        .expect("build query thread pool");

    eprintln!(
        "beir-bench: dataset={} split={} top_k={} batch={} candidates={} threads={} (resolved {}) max_docs={:?} methods={:?}",
        cfg.dataset, cfg.split, cfg.top_k, cfg.batch, cfg.candidates, cfg.threads, threads_resolved, cfg.max_docs, cfg.methods,
    );

    let enc_dir = resolve_encoder_dir(&cfg.cache_dir, &cfg.dataset, &cfg.split);
    let manifest_path = format!("{enc_dir}/embeddings.manifest.json");
    let encoder_sha = sha256_file(&manifest_path);

    // Load ONLY the first n_docs rows when sub-sampling (--max-docs), so the scan
    // sweep never pulls the whole corpus off disk just to slice it. corpus_ids is
    // truncated to match; the full row count comes from the npy header.
    let n_corpus_full = npy_row_count(&format!("{enc_dir}/corpus.f32.npy"));
    let n_docs = cfg.max_docs.unwrap_or(n_corpus_full).min(n_corpus_full);
    let full_corpus = cfg.max_docs.is_none() || n_docs == n_corpus_full;
    let (corpus_vec, n_loaded, dim) =
        load_npy_f32_rows(&format!("{enc_dir}/corpus.f32.npy"), Some(n_docs));
    assert_eq!(n_loaded, n_docs, "corpus load row mismatch");
    let (queries, n_queries, q_dim) = load_npy_f32(&format!("{enc_dir}/queries.f32.npy"));
    assert_eq!(q_dim, dim, "query dim {q_dim} != corpus dim {dim}");
    validate_embeddings(&corpus_vec, n_docs, dim, "corpus");
    validate_embeddings(&queries, n_queries, q_dim, "queries");

    let corpus_ids_full = load_json_string_array(&format!("{enc_dir}/corpus_ids.json"));
    let query_ids = load_json_string_array(&format!("{enc_dir}/query_ids.json"));
    assert_eq!(
        corpus_ids_full.len(),
        n_corpus_full,
        "corpus_ids/embeddings mismatch"
    );
    assert_eq!(query_ids.len(), n_queries, "query_ids/embeddings mismatch");

    let corpus = &corpus_vec[..n_docs * dim];
    let corpus_ids = &corpus_ids_full[..n_docs];
    let write_topk = full_corpus; // qrels-based nDCG only valid on the full corpus

    let simd = detected_simd();
    eprintln!(
        "dim={dim} n_docs={n_docs}{} n_queries={n_queries} simd={simd:?}",
        if full_corpus {
            " (full)"
        } else {
            " (sub-sampled)"
        }
    );

    let mut timing_writer = open_timing_appender(&cfg.out_dir, &cfg.dataset);

    for method in &cfg.methods {
        eprintln!("\n--- {method} ---");
        match method.as_str() {
            "flat" => run_flat(
                corpus,
                &queries,
                dim,
                n_docs,
                n_queries,
                cfg.top_k,
                cfg.batch,
                threads_resolved,
                &query_pool,
                &cfg,
                corpus_ids,
                &query_ids,
                &simd,
                &encoder_sha,
                write_topk,
                &mut timing_writer,
            ),
            "hnsw" => run_hnsw(
                corpus,
                &queries,
                dim,
                n_docs,
                n_queries,
                cfg.top_k,
                cfg.batch,
                threads_resolved,
                &query_pool,
                &cfg,
                corpus_ids,
                &query_ids,
                &simd,
                &encoder_sha,
                write_topk,
                &mut timing_writer,
            ),
            "rq2" => run_rq(
                corpus,
                &queries,
                dim,
                n_docs,
                n_queries,
                cfg.top_k,
                cfg.batch,
                2,
                threads_resolved,
                &query_pool,
                &cfg,
                corpus_ids,
                &query_ids,
                &simd,
                &encoder_sha,
                write_topk,
                &mut timing_writer,
            ),
            "rq4" => run_rq(
                corpus,
                &queries,
                dim,
                n_docs,
                n_queries,
                cfg.top_k,
                cfg.batch,
                4,
                threads_resolved,
                &query_pool,
                &cfg,
                corpus_ids,
                &query_ids,
                &simd,
                &encoder_sha,
                write_topk,
                &mut timing_writer,
            ),
            "bitmap-rq2" => run_two_stage(
                TwoStage::Bitmap,
                corpus,
                &queries,
                dim,
                n_docs,
                n_queries,
                cfg.top_k,
                cfg.batch,
                cfg.candidates,
                threads_resolved,
                &query_pool,
                &cfg,
                corpus_ids,
                &query_ids,
                &simd,
                &encoder_sha,
                write_topk,
                &mut timing_writer,
            ),
            "sign-rq2" => run_two_stage(
                TwoStage::Sign,
                corpus,
                &queries,
                dim,
                n_docs,
                n_queries,
                cfg.top_k,
                cfg.batch,
                cfg.candidates,
                threads_resolved,
                &query_pool,
                &cfg,
                corpus_ids,
                &query_ids,
                &simd,
                &encoder_sha,
                write_topk,
                &mut timing_writer,
            ),
            "sign-rq2-threaded" => run_sign_threaded(
                cfg.candidates, corpus, &queries, dim, n_docs, n_queries, cfg.top_k, cfg.batch,
                threads_resolved, &query_pool, &cfg, corpus_ids, &query_ids, &simd, &encoder_sha,
                write_topk, &mut timing_writer,
            ),
            other => panic!(
                "unknown method '{other}'. Supported: flat, hnsw, rq2, rq4, bitmap-rq2, sign-rq2, sign-rq2-threaded"
            ),
        }
    }
    timing_writer.flush().expect("flush timing.jsonl");
    eprintln!(
        "\ndone. timing -> {}/{}/timing.jsonl",
        cfg.out_dir, cfg.dataset
    );
}

// ---------------------------------------------------------------------------
// Method: flat (exact inner product == FAISS IndexFlatIP math)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_flat(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    threads: usize,
    pool: &rayon::ThreadPool,
    cfg: &Config,
    corpus_ids: &[String],
    query_ids: &[String],
    simd: &[String],
    encoder_sha: &str,
    write_topk: bool,
    timing_writer: &mut dyn Write,
) {
    let bytes_per_vector = dim * 4;
    let index_total_mib = (n_docs * bytes_per_vector) as f64 / 1024.0 / 1024.0;
    let warmup = 5.min(n_queries);

    let (samples, preds) = pool.install(|| {
        time_and_collect(n_queries, batch, warmup, write_topk, |bs, be| {
            let qbatch = &queries[bs * dim..be * dim];
            flat_batch_topk(qbatch, be - bs, corpus, n_docs, dim, top_k)
        })
    });

    finalize(
        "flat",
        &samples,
        preds,
        dim,
        n_docs,
        n_queries,
        top_k,
        threads,
        batch,
        0,
        bytes_per_vector,
        index_total_mib,
        0.0,
        &cfg.dataset,
        &cfg.split,
        query_ids,
        corpus_ids,
        &cfg.out_dir,
        simd,
        encoder_sha,
        timing_writer,
    );
}

// ---------------------------------------------------------------------------
// Method: hnsw (pure-Rust HNSW, hnsw_rs; DistL2 ≡ max-dot on unit-norm vectors).
// Score is `-distance` (nearer = smaller L2 = higher score), so the eval ranks
// nearest-first; for unit vectors this is the identical ordering DistDot gives.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_hnsw(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    threads: usize,
    pool: &rayon::ThreadPool,
    cfg: &Config,
    corpus_ids: &[String],
    query_ids: &[String],
    simd: &[String],
    encoder_sha: &str,
    write_topk: bool,
    timing_writer: &mut dyn Write,
) {
    // ef in the slug so an ef-sweep does not overwrite topk/summary/timing rows
    // (each operating point on the recall/latency frontier is recorded distinctly).
    let slug = format!("hnsw_ef{}", cfg.ef_search);
    let slug = slug.as_str();
    eprintln!("  building HNSW M={HNSW_M} ef_c={HNSW_EF_CONSTRUCTION} ef_s={} ({n_docs} docs) ...", cfg.ef_search);
    // DistL2 (not DistDot): embeddings are unit-normalized, so min-L2 ≡ max-dot ≡
    // max-cosine — identical neighbors — but DistL2 avoids anndists' DistDot
    // `1-dot` distance assert, which panics on near-duplicate pairs whose float
    // dot rounds just past 1.0 (rare at 171K, frequent at ~1M).
    let hnsw: Hnsw<f32, DistL2> = Hnsw::new(
        HNSW_M,
        n_docs,
        HNSW_MAX_LAYER,
        HNSW_EF_CONSTRUCTION,
        DistL2 {},
    );
    // Insert (build uses all cores via the global pool).
    let doc_refs: Vec<(&[f32], usize)> = (0..n_docs)
        .map(|di| (&corpus[di * dim..(di + 1) * dim], di))
        .collect();
    let t0 = Instant::now();
    hnsw.parallel_insert_slice(&doc_refs);
    let build_seconds = t0.elapsed().as_secs_f64();
    eprintln!("  build done in {build_seconds:.2}s");

    // HNSW graph size is implementation-internal; report the stored-vector bytes
    // (full float) as the index footprint, matching the dense baseline accounting.
    let bytes_per_vector = dim * 4;
    let index_total_mib = (n_docs * bytes_per_vector) as f64 / 1024.0 / 1024.0;
    let warmup = 5.min(n_queries);

    // Pre-slice query rows so neither timing mode pays per-batch allocation.
    let query_rows: Vec<&[f32]> = (0..n_queries)
        .map(|qi| &queries[qi * dim..(qi + 1) * dim])
        .collect();

    let (samples, preds) = pool.install(|| {
        time_and_collect(n_queries, batch, warmup, write_topk, |bs, be| {
            let rows: Vec<Vec<(i64, f32)>> = if threads == 1 {
                // Single-thread: serial search per query.
                (bs..be)
                    .map(|qi| {
                        hnsw.search(query_rows[qi], top_k, cfg.ef_search)
                            .into_iter()
                            .map(|nb| (nb.d_id as i64, -nb.distance))
                            .collect()
                    })
                    .collect()
            } else {
                // Threaded: batched parallel search (rayon, this pool).
                let batch_slice: Vec<Vec<f32>> =
                    (bs..be).map(|qi| query_rows[qi].to_vec()).collect();
                hnsw.parallel_search(&batch_slice, top_k, cfg.ef_search)
                    .into_iter()
                    .map(|nbs| {
                        nbs.into_iter()
                            .map(|nb| (nb.d_id as i64, -nb.distance))
                            .collect()
                    })
                    .collect()
            };
            pad_rows(rows, top_k)
        })
    });

    finalize(
        slug,
        &samples,
        preds,
        dim,
        n_docs,
        n_queries,
        top_k,
        threads,
        batch,
        0,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        &cfg.dataset,
        &cfg.split,
        query_ids,
        corpus_ids,
        &cfg.out_dir,
        simd,
        encoder_sha,
        timing_writer,
    );
}

// ---------------------------------------------------------------------------
// Method: rq2 / rq4 (RankQuant full-scan asymmetric LUT)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_rq(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    bits: u8,
    threads: usize,
    pool: &rayon::ThreadPool,
    cfg: &Config,
    corpus_ids: &[String],
    query_ids: &[String],
    simd: &[String],
    encoder_sha: &str,
    write_topk: bool,
    timing_writer: &mut dyn Write,
) {
    let slug = format!("ordvec-rq{bits}");
    eprintln!("  building RankQuant b={bits} ({n_docs} docs) ...");
    let mut idx = RankQuant::new(dim, bits);
    let t0 = Instant::now();
    idx.add(corpus);
    let build_seconds = t0.elapsed().as_secs_f64();
    let bytes_per_vector = idx.bytes_per_vec();
    let index_total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;
    let warmup = 5.min(n_queries);

    let (samples, preds) = pool.install(|| {
        time_and_collect(n_queries, batch, warmup, write_topk, |bs, be| {
            let batch_q = &queries[bs * dim..be * dim];
            let res = idx.search_asymmetric(batch_q, top_k);
            (res.indices, res.scores)
        })
    });

    finalize(
        &slug,
        &samples,
        preds,
        dim,
        n_docs,
        n_queries,
        top_k,
        threads,
        batch,
        0,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        &cfg.dataset,
        &cfg.split,
        query_ids,
        corpus_ids,
        &cfg.out_dir,
        simd,
        encoder_sha,
        timing_writer,
    );
}

// ---------------------------------------------------------------------------
// Method: bitmap-rq2 / sign-rq2 (two-stage candidate-gen → rerank)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum TwoStage {
    Bitmap,
    Sign,
}

fn bitmap_vecs_to_csr(vecs: Vec<Vec<u32>>) -> (Vec<usize>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(vecs.len() + 1);
    let mut candidates = Vec::new();
    offsets.push(0usize);
    for row in &vecs {
        candidates.extend_from_slice(row);
        offsets.push(candidates.len());
    }
    (offsets, candidates)
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage(
    stage: TwoStage,
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    candidates: usize,
    threads: usize,
    pool: &rayon::ThreadPool,
    cfg: &Config,
    corpus_ids: &[String],
    query_ids: &[String],
    simd: &[String],
    encoder_sha: &str,
    write_topk: bool,
    timing_writer: &mut dyn Write,
) {
    let (slug, label) = match stage {
        TwoStage::Bitmap => ("ordvec-bitmap-rq2", "Bitmap"),
        TwoStage::Sign => ("ordvec-sign-rq2", "SignBitmap"),
    };
    eprintln!("  building {label} + RankQuant b=2 (m={candidates}, {n_docs} docs) ...");

    let n_top = dim / 4;
    let mut bitmap = Bitmap::new(dim, n_top);
    let mut sign = SignBitmap::new(dim);
    let mut rq = RankQuant::new(dim, 2);
    let t0 = Instant::now();
    match stage {
        TwoStage::Bitmap => bitmap.add(corpus),
        TwoStage::Sign => sign.add(corpus),
    }
    rq.add(corpus);
    let build_seconds = t0.elapsed().as_secs_f64();

    let stage1_bytes = match stage {
        TwoStage::Bitmap => bitmap.bytes_per_vec(),
        TwoStage::Sign => sign.bytes_per_vec(),
    };
    let stage1_size = match stage {
        TwoStage::Bitmap => bitmap.byte_size(),
        TwoStage::Sign => sign.byte_size(),
    };
    let bytes_per_vector = stage1_bytes + rq.bytes_per_vec();
    let index_total_mib = (stage1_size + rq.byte_size()) as f64 / 1024.0 / 1024.0;

    let out_k = top_k.min(candidates).min(n_docs);
    let warmup = 5.min(n_queries);

    let mut scratch = SubsetScratch::new();
    let mut out_scores_buf = vec![f32::NEG_INFINITY; batch * out_k];
    let mut out_indices_buf = vec![-1i64; batch * out_k];

    let (samples, preds) = pool.install(|| {
        time_and_collect(n_queries, batch, warmup, write_topk, |bs, be| {
            let batch_q = &queries[bs * dim..be * dim];
            let nq_batch = be - bs;
            let needed = nq_batch * out_k;
            if out_scores_buf.len() != needed {
                out_scores_buf.resize(needed, f32::NEG_INFINITY);
                out_indices_buf.resize(needed, -1);
            }

            // Stage 1: candidate generation → CSR (offsets, candidates).
            let (offsets, cand_flat) = match stage {
                TwoStage::Bitmap => {
                    let cand_vecs = bitmap.top_m_candidates_batched(batch_q, candidates);
                    bitmap_vecs_to_csr(cand_vecs)
                }
                TwoStage::Sign => {
                    let cb: CandidateBatch =
                        sign.top_m_candidates_batched_serial_csr(batch_q, candidates);
                    (cb.offsets, cb.candidates)
                }
            };

            // Stage 2: pooled subset rerank (allocation-free).
            // Rerank for `out_k` (= top_k capped by the candidate budget + corpus),
            // matching the `batch * out_k` buffers; passing `top_k` would mis-size the
            // buffers and panic the length assert when the budget is below `top_k`.
            rq.search_asymmetric_subset_batched_serial_into(
                batch_q,
                &offsets,
                &cand_flat,
                out_k,
                &mut scratch,
                &mut out_scores_buf,
                &mut out_indices_buf,
            );

            // Pad per-query results to `top_k`.
            let mut idx = vec![-1i64; nq_batch * top_k];
            let mut sc = vec![0.0f32; nq_batch * top_k];
            for qi in 0..nq_batch {
                let src_i = &out_indices_buf[qi * out_k..(qi + 1) * out_k];
                let src_s = &out_scores_buf[qi * out_k..(qi + 1) * out_k];
                let copy = src_i.len().min(top_k);
                idx[qi * top_k..qi * top_k + copy].copy_from_slice(&src_i[..copy]);
                sc[qi * top_k..qi * top_k + copy].copy_from_slice(&src_s[..copy]);
            }
            (idx, sc)
        })
    });

    finalize(
        slug,
        &samples,
        preds,
        dim,
        n_docs,
        n_queries,
        top_k,
        threads,
        batch,
        candidates,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        &cfg.dataset,
        &cfg.split,
        query_ids,
        corpus_ids,
        &cfg.out_dir,
        simd,
        encoder_sha,
        timing_writer,
    );
}

/// Deterministic EXACTLY-`m` selection over a `(count, id)` candidate pool, by
/// `(count desc, id asc)` -- mirrors `SignBitmap`'s `select_nth_unstable_by`
/// exact-`m_eff` tie-break. The `>= tau` threshold set is `>= m` (boundary ties
/// overshoot); this trims it to exactly `m` by keeping the highest-agreement
/// docs, tie-broken on smaller id. Output is sorted ascending by id so the serial
/// and threaded paths return byte-identical candidate sets.
fn select_exact_m(pool: &mut [(u32, u32)], m: usize, out: &mut Vec<u32>) {
    out.clear();
    let m_eff = m.min(pool.len());
    if m_eff == 0 {
        return;
    }
    let cmp = |a: &(u32, u32), b: &(u32, u32)| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1));
    pool.select_nth_unstable_by(m_eff - 1, cmp);
    out.extend(pool[..m_eff].iter().map(|&(_, id)| id));
    out.sort_unstable();
}

/// Within-query-threaded `sign->rq2` baseline: doc-major sign codes (`wpd`
/// u64/doc), per-doc agreement via hardware VPOPCNTDQ over all `dim` bits,
/// parallelized over doc-stripes (parallel agreement scan -> histogram -> global
/// top-M tau -> EXACTLY-m trim via [`select_exact_m`]) with the SAME fixed-`m`
/// budget + (count desc, id asc) tie-break as the serial `sign-rq2` baseline. It
/// is the SAME SignBitmap candidate set as `sign-rq2`, just computed with
/// within-query threads (the serial baseline scans single-threaded per query).
/// `threads=1` reproduces the serial sign scan.
#[allow(clippy::too_many_arguments)]
fn run_sign_threaded(
    m: usize,
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    threads: usize,
    pool: &rayon::ThreadPool,
    cfg: &Config,
    corpus_ids: &[String],
    query_ids: &[String],
    simd: &[String],
    encoder_sha: &str,
    write_topk: bool,
    timing_writer: &mut dyn Write,
) {
    let slug = "ordvec-sign-rq2-threaded";
    let wpd = dim.div_ceil(64);
    eprintln!(
        "  building doc-major SIGN codes + RankQuant b=2 (threaded, m={m}, {n_docs} docs) ..."
    );
    let mut rq = RankQuant::new(dim, 2);
    let t0 = Instant::now();
    let mut codes = vec![0u64; n_docs * wpd];
    // One mutable stripe per doc (`wpd` words) -> parallel sign-code build, matching
    // the adjacent `rq.add` which is already parallel over the corpus.
    {
        use rayon::prelude::*;
        codes.par_chunks_mut(wpd).enumerate().for_each(|(d, code)| {
            let row = &corpus[d * dim..(d + 1) * dim];
            for (j, &v) in row.iter().enumerate() {
                // `> 0.0` -- same threshold as core SignBitmap (zero/NaN with negatives).
                if v > 0.0 {
                    code[j >> 6] |= 1u64 << (j & 63);
                }
            }
        });
    }
    rq.add(corpus);
    let build_seconds = t0.elapsed().as_secs_f64();
    // doc-major sign code is dim bits/doc (= dim/8 bytes) -- identical substrate
    // size to the serial SignBitmap baseline.
    let bytes_per_vector = (wpd * 8) + rq.bytes_per_vec();
    let index_total_mib = ((codes.len() * 8) + rq.byte_size()) as f64 / 1024.0 / 1024.0;

    let out_k = top_k.min(m).min(n_docs);
    let warmup = 5.min(n_queries);
    let mut scratch = SubsetScratch::new();
    let mut out_scores = vec![f32::NEG_INFINITY; batch * out_k];
    let mut out_indices = vec![-1i64; batch * out_k];
    let mut cand: Vec<u32> = Vec::new();
    // Per-query scratch, allocated once so the timed path is allocation-free:
    // `agree` is the u32 per-doc agreement buffer, `hists_buf` holds one (dim+1)-bin
    // histogram per thread-stripe, `poolv` is the reused >= tau candidate pool.
    let mut agree = vec![0u32; n_docs];
    // checked_mul so a pathological threads/dim can't wrap usize into a too-small
    // buffer in release (matches the core crate's `util::checked_*` convention);
    // `sign_scan_topm_par` slices `hists_buf[..stripes * (dim + 1)]`.
    let hists_buf_len = threads
        .max(1)
        .checked_mul(wpd * 64 + 1)
        .expect("hists_buf length overflow");
    let mut hists_buf = vec![0u32; hists_buf_len];
    let mut poolv: Vec<(u32, u32)> = Vec::with_capacity(m * 2);

    let (samples, preds) = pool.install(|| {
        time_and_collect(n_queries, batch, warmup, write_topk, |bs, be| {
            let nq_batch = be - bs;
            let needed = nq_batch * out_k;
            if out_scores.len() != needed {
                out_scores.resize(needed, f32::NEG_INFINITY);
                out_indices.resize(needed, -1);
            }
            // Stage 1: per-query within-query-threaded sign scan -> exact-m CSR.
            let mut offsets = Vec::with_capacity(nq_batch + 1);
            let mut cand_flat: Vec<u32> = Vec::new();
            offsets.push(0usize);
            for qi in bs..be {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let qcode = build_query_sign(q, wpd);
                sign_scan_topm_par(
                    &codes,
                    wpd,
                    n_docs,
                    &qcode,
                    m,
                    threads,
                    &mut agree,
                    &mut hists_buf,
                    &mut poolv,
                    &mut cand,
                );
                cand_flat.extend_from_slice(&cand);
                offsets.push(cand_flat.len());
            }
            // Stage 2: pooled subset rerank. Rerank for `out_k` (= top_k capped by `m`
            // + corpus) to match the `batch * out_k` buffers; passing `top_k` would
            // mis-size them and panic the length assert when `m < top_k`.
            let batch_q = &queries[bs * dim..be * dim];
            rq.search_asymmetric_subset_batched_serial_into(
                batch_q,
                &offsets,
                &cand_flat,
                out_k,
                &mut scratch,
                &mut out_scores,
                &mut out_indices,
            );
            let mut idx = vec![-1i64; nq_batch * top_k];
            let mut sc = vec![0.0f32; nq_batch * top_k];
            for qi in 0..nq_batch {
                let si = &out_indices[qi * out_k..(qi + 1) * out_k];
                let ss = &out_scores[qi * out_k..(qi + 1) * out_k];
                let copy = si.len().min(top_k);
                idx[qi * top_k..qi * top_k + copy].copy_from_slice(&si[..copy]);
                sc[qi * top_k..qi * top_k + copy].copy_from_slice(&ss[..copy]);
            }
            (idx, sc)
        })
    });

    finalize(
        slug,
        &samples,
        preds,
        dim,
        n_docs,
        n_queries,
        top_k,
        threads,
        batch,
        m,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        &cfg.dataset,
        &cfg.split,
        query_ids,
        corpus_ids,
        &cfg.out_dir,
        simd,
        encoder_sha,
        timing_writer,
    );
}

/// Query sign bits, doc-major layout (bit `j` set iff `q[j] > 0.0` -- the SAME
/// threshold as core SignBitmap's `build_query_bitmap`, so zero/NaN group with
/// the negatives and this threaded baseline is candidate-faithful to `sign-rq2`).
fn build_query_sign(q: &[f32], wpd: usize) -> Vec<u64> {
    let mut c = vec![0u64; wpd];
    for (j, &v) in q.iter().enumerate() {
        if v > 0.0 {
            c[j >> 6] |= 1u64 << (j & 63);
        }
    }
    c
}

/// Single-pass parallel sign-agreement scan + EXACTLY-m top selection. Scans the
/// doc-major sign codes ONCE (hardware VPOPCNTDQ, bandwidth-bound -- the same
/// vectorized popcount the optimized scan uses, so the baseline is not unfairly
/// slow) into a per-doc agreement buffer, histograms to a global top-M `tau`,
/// then trims the `>= tau` set to exactly `m` via [`select_exact_m`].
/// `agree` (u32 per-doc agreement), `hists_buf` (one `dim+1`-bin histogram per
/// thread-stripe) and `poolv` (the `>= tau` candidate pool) are all caller-owned
/// reusable scratch, so the timed path performs no per-query allocation.
#[allow(clippy::too_many_arguments)]
fn sign_scan_topm_par(
    codes: &[u64],
    wpd: usize,
    n: usize,
    qcode: &[u64],
    m: usize,
    threads: usize,
    agree: &mut [u32],
    hists_buf: &mut [u32],
    poolv: &mut Vec<(u32, u32)>,
    out: &mut Vec<u32>,
) {
    use rayon::prelude::*;
    let dim = wpd * 64;
    let hlen = dim + 1;
    let t = threads.max(1).min(n.max(1));
    let chunk = n.div_ceil(t).max(1);
    let stripes = n.div_ceil(chunk);
    // Phase A: ONE parallel pass over the codes -> per-doc agreement. Stored as u32
    // so the `dim - hamming` count never truncates regardless of dim.
    agree[..n]
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, slot)| {
            let d0 = ci * chunk;
            #[cfg(target_arch = "x86_64")]
            {
                // Guard EVERY feature the kernel enables via `#[target_feature]`
                // (`avx512f` + `avx512vpopcntdq`) -- detecting only vpopcntdq would
                // call into an under-verified target. Mirrors the core crate's
                // dispatch (e.g. `lib.rs` / `multi_bucket.rs`).
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512vpopcntdq")
                {
                    unsafe { scan_agree_avx512(codes, wpd, d0, qcode, slot) };
                    return;
                }
            }
            for (li, a) in slot.iter_mut().enumerate() {
                let base = (d0 + li) * wpd;
                let mut ham = 0u32;
                for w in 0..wpd {
                    ham += (codes[base + w] ^ qcode[w]).count_ones();
                }
                *a = dim as u32 - ham;
            }
        });
    // Parallel per-stripe histogram into the reused `hists_buf` (stripe `ci` owns
    // `hists_buf[ci*hlen .. (ci+1)*hlen]`); zeroed per query but never reallocated.
    let used = stripes * hlen;
    hists_buf[..used].fill(0);
    hists_buf[..used]
        .par_chunks_mut(hlen)
        .zip(agree[..n].par_chunks(chunk))
        .for_each(|(h, slot)| {
            for &a in slot {
                h[a as usize] += 1;
            }
        });
    // Global top-M threshold tau: walk agreement high->low, summing the per-stripe
    // histogram columns until the cumulative count reaches m (no merge buffer).
    let mut cum = 0u64;
    let mut tau = 0u32;
    'tau: for c in (0..=dim).rev() {
        for s in 0..stripes {
            cum += hists_buf[s * hlen + c] as u64;
        }
        if cum >= m as u64 {
            tau = c as u32;
            break 'tau;
        }
    }
    // Phase B: parallel extract (agreement, id) for agreement >= tau into the reused
    // `poolv` (clear() keeps the capacity; `par_extend` stays parallel), then trim to
    // EXACTLY m via `select_exact_m` -- same fixed budget + (count desc, id asc)
    // tie-break as the serial sign baseline, so both rerank identical candidate sets.
    // Extract order is irrelevant: select_exact_m imposes a strict total order.
    poolv.clear();
    poolv.par_extend(
        agree[..n]
            .par_chunks(chunk)
            .enumerate()
            .flat_map_iter(|(ci, slot)| {
                let d0 = ci * chunk;
                slot.iter().enumerate().filter_map(move |(li, &a)| {
                    if a >= tau {
                        Some((a, (d0 + li) as u32))
                    } else {
                        None
                    }
                })
            }),
    );
    select_exact_m(poolv, m, out);
}

/// Hardware VPOPCNTDQ sign-agreement scan for docs `[d0, d0+slot.len())`: the same
/// vectorized popcount the optimized scan uses, so the baseline is not unfairly
/// slow. Fills `slot` with `agreement = dim - hamming`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn scan_agree_avx512(codes: &[u64], wpd: usize, d0: usize, qcode: &[u64], slot: &mut [u32]) {
    use std::arch::x86_64::*;
    let dim = (wpd * 64) as u32;
    let cp = codes.as_ptr();
    let qp = qcode.as_ptr();
    for (li, a) in slot.iter_mut().enumerate() {
        let base = (d0 + li) * wpd;
        let mut acc = _mm512_setzero_si512();
        let mut w = 0usize;
        while w + 8 <= wpd {
            let c = _mm512_loadu_si512(cp.add(base + w) as *const __m512i);
            let q = _mm512_loadu_si512(qp.add(w) as *const __m512i);
            let pc = _mm512_popcnt_epi64(_mm512_xor_si512(c, q));
            acc = _mm512_add_epi64(acc, pc);
            w += 8;
        }
        let mut ham = _mm512_reduce_add_epi64(acc) as u32;
        while w < wpd {
            ham += (*cp.add(base + w) ^ *qp.add(w)).count_ones();
            w += 1;
        }
        *a = dim - ham;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_npy_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ordvec-beir-bench-{name}-{}-{}.npy",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    fn write_v1_npy(path: &std::path::Path, rows: usize, dim: usize, values: &[f32]) {
        assert_eq!(values.len(), rows * dim);
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {dim}), }}");
        let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
        header.extend(std::iter::repeat_n(' ', padding));
        header.push('\n');

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn npy_row_count_uses_shared_version_guard() {
        let path = temp_npy_path("v3");
        std::fs::write(&path, b"\x93NUMPY\x03\x00\x00\x00").unwrap();
        let result = std::panic::catch_unwind(|| npy_row_count(path.to_str().unwrap()));
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn load_npy_f32_rows_reads_only_requested_prefix() {
        let path = temp_npy_path("prefix");
        write_v1_npy(&path, 3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(npy_row_count(path.to_str().unwrap()), 3);
        let (values, rows, dim) = load_npy_f32_rows(path.to_str().unwrap(), Some(2));

        let _ = std::fs::remove_file(&path);
        assert_eq!((rows, dim), (2, 2));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn load_npy_f32_rows_rejects_trailing_payload_bytes() {
        let path = temp_npy_path("trailing");
        write_v1_npy(&path, 1, 1, &[1.0]);
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&[0]).unwrap();

        let result = std::panic::catch_unwind(|| load_npy_f32_rows(path.to_str().unwrap(), None));
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }
}
