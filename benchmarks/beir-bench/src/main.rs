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
const HNSW_EF_SEARCH: usize = 128;
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

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
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
    }
}

// ---------------------------------------------------------------------------
// NumPy v1/v2 reader (2-D LE f32, C-order)
// ---------------------------------------------------------------------------

fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read npy {path}: {e}"));
    assert!(bytes.len() >= 10, "npy file too short: {path}");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a numpy file: {path}");
    let major = bytes[6];
    let minor = bytes[7];
    assert!(
        major == 1 || major == 2,
        "unsupported npy version {major}.{minor}: {path}",
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
    assert!(
        header.contains("'descr': '<f4'"),
        "expected <f4 dtype in {path}: {header}",
    );
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order in {path}",
    );
    let shape_start = header.find("'shape':").expect("no shape in npy header");
    let after = &header[shape_start..];
    let open = after.find('(').unwrap();
    let close = after.find(')').unwrap();
    let dims: Vec<usize> = after[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    assert_eq!(dims.len(), 2, "expected 2-D array in {path}");
    let n = dims[0];
    let dim = dims[1];
    let data_start = header_start + header_len;
    let n_floats = n * dim;
    assert_eq!(
        bytes.len() - data_start,
        n_floats * 4,
        "data length mismatch in {path}",
    );
    let mut out = vec![0.0f32; n_floats];
    for (i, chunk) in bytes[data_start..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    (out, n, dim)
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn load_json_string_array(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut out = Vec::new();
    let mut in_str = false;
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_str {
                out.push(cur.clone());
                cur.clear();
                in_str = false;
            } else {
                in_str = true;
            }
        } else if in_str {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    match next {
                        '"' => cur.push('"'),
                        '\\' => cur.push('\\'),
                        'n' => cur.push('\n'),
                        't' => cur.push('\t'),
                        other => {
                            cur.push('\\');
                            cur.push(other);
                        }
                    }
                }
            } else {
                cur.push(c);
            }
        }
    }
    out
}

/// sha256 of a file via system sha256sum / shasum / openssl. Panics with a clear
/// reason; never emits a non-SHA value that merely looks like one.
fn sha256_file(path: &str) -> String {
    if !std::path::Path::new(path).is_file() {
        panic!("cannot compute SHA-256: file does not exist: {path}");
    }
    let mut tool_ran = false;
    for (cmd, args) in &[
        ("sha256sum", vec![path]),
        ("shasum", vec!["-a", "256", path]),
        ("openssl", vec!["dgst", "-sha256", path]),
    ] {
        if let Ok(out) = std::process::Command::new(cmd).args(args).output() {
            tool_ran = true;
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                for token in s.split_whitespace() {
                    if token.len() == 64 && token.chars().all(|c| c.is_ascii_hexdigit()) {
                        return token.to_string();
                    }
                }
            }
        }
    }
    if tool_ran {
        panic!("a SHA-256 tool ran but produced no digest for {path}");
    }
    panic!(
        "no SHA-256 tool available (tried sha256sum / shasum -a 256 / openssl) — \
         cannot compute the encoder-manifest digest for {path}"
    );
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
        let mut doc_idxs_str = String::from("[");
        let mut doc_ids_str = String::from("[");
        let mut scores_str = String::from("[");
        let mut first = true;
        for (j, &di) in row_indices.iter().enumerate() {
            if di < 0 {
                break;
            }
            let di_usize = di as usize;
            if !first {
                doc_idxs_str.push(',');
                doc_ids_str.push(',');
                scores_str.push(',');
            }
            first = false;
            doc_idxs_str.push_str(&di_usize.to_string());
            let doc_id = if di_usize < n_corpus {
                corpus_ids[di_usize].as_str()
            } else {
                ""
            };
            doc_ids_str.push('"');
            doc_ids_str.push_str(doc_id);
            doc_ids_str.push('"');
            let sc = scores.get(qi * k + j).copied().unwrap_or(0.0);
            if sc.is_finite() {
                scores_str.push_str(&sc.to_string());
            } else {
                scores_str.push_str("0.0");
            }
        }
        doc_idxs_str.push(']');
        doc_ids_str.push(']');
        scores_str.push(']');

        writeln!(
            writer,
            r#"{{"dataset":"{dataset}","split":"{split}","method":"{method}","qid_idx":{qi},"qid":"{qid}","k":{k},"doc_idxs":{doc_idxs_str},"doc_ids":{doc_ids_str},"scores":{scores_str}}}"#,
            qid = query_ids[qi],
        )
        .expect("write topk jsonl");
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
    let simd_arr: String = {
        let parts: Vec<String> = r.simd.iter().map(|s| format!("\"{s}\"")).collect();
        format!("[{}]", parts.join(","))
    };
    writeln!(
        w,
        r#"{{"dataset":"{ds}","split":"{sp}","method":"{m}","dim":{dim},"n_docs":{nd},"n_queries":{nq},"top_k":{tk},"threads":{th},"batch":{b},"candidates":{c},"bytes_per_vector":{bpv},"index_total_mib":{imib:.3},"build_seconds":{bs:.4},"query_latency_ms_p50":{p50:.5},"query_latency_ms_p95":{p95:.5},"query_latency_ms_p99":{p99:.5},"queries_per_second":{qps:.2},"cpu_arch":"{arch}","simd_detected":{simd},"rustc":"{rustc}","crate_version":"{cv}","encoder_manifest_sha256":"{sha}"}}"#,
        ds = r.dataset,
        sp = r.split,
        m = r.method,
        dim = r.dim,
        nd = r.n_docs,
        nq = r.n_queries,
        tk = r.top_k,
        th = r.threads,
        b = r.batch,
        c = r.candidates,
        bpv = r.bytes_per_vector,
        imib = r.index_total_mib,
        bs = r.build_seconds,
        p50 = r.p50_ms,
        p95 = r.p95_ms,
        p99 = r.p99_ms,
        qps = r.qps,
        arch = std::env::consts::ARCH,
        simd = simd_arr,
        rustc = rustc_version(),
        cv = env!("CARGO_PKG_VERSION"),
        sha = r.encoder_sha,
    )
    .expect("write record json");
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
    if k < scored.len() {
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

    let (corpus_full, n_corpus_full, dim) = load_npy_f32(&format!("{enc_dir}/corpus.f32.npy"));
    let (queries, n_queries, q_dim) = load_npy_f32(&format!("{enc_dir}/queries.f32.npy"));
    assert_eq!(q_dim, dim, "query dim {q_dim} != corpus dim {dim}");
    validate_embeddings(&corpus_full, n_corpus_full, dim, "corpus");
    validate_embeddings(&queries, n_queries, q_dim, "queries");

    let corpus_ids_full = load_json_string_array(&format!("{enc_dir}/corpus_ids.json"));
    let query_ids = load_json_string_array(&format!("{enc_dir}/query_ids.json"));
    assert_eq!(
        corpus_ids_full.len(),
        n_corpus_full,
        "corpus_ids/embeddings mismatch"
    );
    assert_eq!(query_ids.len(), n_queries, "query_ids/embeddings mismatch");

    // Sub-sample the corpus for the scaling sweep (latency-only; no nDCG).
    let n_docs = cfg.max_docs.unwrap_or(n_corpus_full).min(n_corpus_full);
    let full_corpus = cfg.max_docs.is_none() || n_docs == n_corpus_full;
    let corpus = &corpus_full[..n_docs * dim];
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
            other => panic!(
                "unknown method '{other}'. Supported: flat, hnsw, rq2, rq4, bitmap-rq2, sign-rq2"
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
// Method: hnsw (pure-Rust HNSW, hnsw_rs; DistDot on unit-norm vectors)
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
    let slug = "hnsw";
    eprintln!("  building HNSW M={HNSW_M} ef_c={HNSW_EF_CONSTRUCTION} ({n_docs} docs) ...");
    let hnsw: Hnsw<f32, DistDot> = Hnsw::new(
        HNSW_M,
        n_docs,
        HNSW_MAX_LAYER,
        HNSW_EF_CONSTRUCTION,
        DistDot {},
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
                        hnsw.search(query_rows[qi], top_k, HNSW_EF_SEARCH)
                            .into_iter()
                            .map(|nb| (nb.d_id as i64, 1.0 - nb.distance))
                            .collect()
                    })
                    .collect()
            } else {
                // Threaded: batched parallel search (rayon, this pool).
                let batch_slice: Vec<Vec<f32>> =
                    (bs..be).map(|qi| query_rows[qi].to_vec()).collect();
                hnsw.parallel_search(&batch_slice, top_k, HNSW_EF_SEARCH)
                    .into_iter()
                    .map(|nbs| {
                        nbs.into_iter()
                            .map(|nb| (nb.d_id as i64, 1.0 - nb.distance))
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
            rq.search_asymmetric_subset_batched_serial_into(
                batch_q,
                &offsets,
                &cand_flat,
                top_k,
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
