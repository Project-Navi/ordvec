//! Focused benchmark + integration example for the **caller-owned serial**
//! two-stage path (the integration contract for DBs / runtimes that own their
//! own parallelism). SYNTHETIC corpus — these numbers are a
//! relative decomposition of the serial path on random data, NOT a retrieval-
//! quality or real-corpus claim, and the dim=1024 result is its own mechanism
//! (do not conflate it with the SignBitmap AVX-tail dim=768 result).
//!
//! It decomposes the cost into four separately-timed phases at the Harrier-1024
//! shape and prints a headline "batched `_into` vs single-query loop" rerank
//! speedup — the per-query-overhead reduction the caller-owned API exists for:
//!   1. stage-1 candidate generation  (top_m_candidates_batched_serial_csr)
//!   2. single-query subset rerank loop  (search_asymmetric_subset, baseline)
//!   3. batched rerank `_into`  (warmed SubsetScratch, caller-owned buffers)
//!   4. full two-stage serial  (1 + 3 end to end)
//!
//!   cargo run --release --example two_stage_bench -- [--dim N] [--n N]
//!       [--queries N] [--m N] [--k N] [--bits {1,2,4}] [--reps N]

use ordvec::{RankQuant, SignBitmap, SubsetScratch};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    // Harrier-1024 defaults; all overridable.
    let mut dim = 1024usize;
    let mut n = 50_000usize;
    let mut nq = 200usize;
    let mut m = 256usize;
    let mut k = 10usize;
    let mut bits = 2u8;
    let mut reps = 20usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut val = || args.next().expect("flag needs a value").parse().unwrap();
        match flag.as_str() {
            "--dim" => dim = val(),
            "--n" => n = val(),
            "--queries" => nq = val(),
            "--m" => m = val(),
            "--k" => k = val(),
            "--bits" => bits = args.next().unwrap().parse().unwrap(),
            "--reps" => reps = val(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    assert!(nq > 0 && n > 0 && reps > 0, "n, queries, reps must be > 0");

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let corpus: Vec<f32> = (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut sign = SignBitmap::new(dim);
    sign.add(&corpus);
    let mut rq = RankQuant::new(dim, bits);
    rq.add(&corpus);
    let queries: Vec<f32> = (0..nq * dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    drop(corpus);

    let out_k = k.min(rq.len());
    // Caller-owned output buffers, allocated ONCE and reused across batches —
    // rectangular nq*out_k, sentinel-padded for underfull rows.
    let mut out_scores = vec![f32::NEG_INFINITY; nq * out_k];
    let mut out_indices = vec![-1i64; nq * out_k];
    let mut scratch = SubsetScratch::new();

    // Warm: build the candidate batch once and warm the scratch to this shape.
    let cb = sign.top_m_candidates_batched_serial_csr(&queries, m);
    rq.search_asymmetric_subset_batched_serial_into(
        &queries,
        &cb.offsets,
        &cb.candidates,
        k,
        &mut scratch,
        &mut out_scores,
        &mut out_indices,
    );
    let total_candidates = cb.candidates.len();

    // Phase 1 — stage-1 candidate generation (serial CSR).
    let p1 = median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                let c = sign.top_m_candidates_batched_serial_csr(&queries, m);
                std::hint::black_box(&c);
                t.elapsed().as_secs_f64()
            })
            .collect(),
    );

    // Phase 2 — single-query subset rerank loop (the per-query baseline).
    let p2 = median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                for qi in 0..nq {
                    let row = &cb.candidates[cb.offsets[qi]..cb.offsets[qi + 1]];
                    let r = rq.search_asymmetric_subset(&queries[qi * dim..(qi + 1) * dim], row, k);
                    std::hint::black_box(&r);
                }
                t.elapsed().as_secs_f64()
            })
            .collect(),
    );

    // Phase 3 — batched `_into` (warmed scratch + reused caller buffers).
    let p3 = median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                rq.search_asymmetric_subset_batched_serial_into(
                    &queries,
                    &cb.offsets,
                    &cb.candidates,
                    k,
                    &mut scratch,
                    &mut out_scores,
                    &mut out_indices,
                );
                t.elapsed().as_secs_f64()
            })
            .collect(),
    );

    // Phase 4 — full two-stage serial (stage-1 gen + batched rerank).
    let p4 = median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                let c = sign.top_m_candidates_batched_serial_csr(&queries, m);
                rq.search_asymmetric_subset_batched_serial_into(
                    &queries,
                    &c.offsets,
                    &c.candidates,
                    k,
                    &mut scratch,
                    &mut out_scores,
                    &mut out_indices,
                );
                t.elapsed().as_secs_f64()
            })
            .collect(),
    );

    let row = |label: &str, secs: f64| {
        println!(
            "  {label:<34} {:>9.3} ms   {:>10.2} q/s   {:>9.2} us/query",
            secs * 1e3,
            nq as f64 / secs,
            secs / nq as f64 * 1e6,
        );
    };
    println!("caller-owned serial two-stage (SYNTHETIC corpus)");
    println!(
        "  dim={dim} n={n} queries={nq} m={m} k={k} bits={bits} out_k={out_k} \
         candidates={total_candidates} reps={reps}"
    );
    println!(
        "  (dim % 64 == {}: AVX-512 tier eligible when supported)",
        dim % 64
    );
    row("1. stage-1 candidate gen (CSR)", p1);
    row("2. single-query rerank loop", p2);
    row("3. batched rerank _into", p3);
    row("4. full two-stage (1+3)", p4);
    println!(
        "  rerank speedup (batched _into vs single-query loop): {:.2}x",
        p2 / p3
    );
}
