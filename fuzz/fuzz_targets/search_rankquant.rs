//! libFuzzer target for the RankQuant ingest + asymmetric-search hot path:
//! `add` (rank_transform -> pack) then `search_asymmetric` (the runtime-
//! dispatched scalar / AVX2 / AVX-512 scan kernels -> TopK). The four `load_*`
//! targets cover deserialization; this exercises the *compute* surface they
//! feed, through the public API (the SIMD kernels themselves are `pub(crate)`).
//!
//! `dim` is fixed at 64 — divisible by `8 / bits` and `2^bits` for every
//! bits in {1,2,4} and by the AVX-512 64-code unroll — so `RankQuant::new` is
//! infallible and the highest SIMD tier the host supports is reached. The
//! fuzzer shapes the doc count, the embedding/query values, and `k` (k == 0
//! included — the empty-`TopK` edge). Embedding values are mapped to finite
//! f32: the public API rejects NaN / ±Inf by contract, so raw float bit
//! patterns would only re-exercise that guard, not the kernels.
//!
//! Contract: no panic, abort, or out-of-bounds access on any input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ordvec::RankQuant;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    const DIM: usize = 64;
    let bits: u8 = match data[0] % 3 {
        0 => 1,
        1 => 2,
        _ => 4,
    };
    let n = (data[1] as usize % 16) + 1; // 1..=16 docs
    let k = data[2] as usize % (n + 1); // 0..=n

    let payload = &data[3..];
    let total = (n + 1) * DIM;
    let floats: Vec<f32> = (0..total)
        .map(|i| {
            if payload.is_empty() {
                0.0
            } else {
                payload[i % payload.len()] as f32 - 128.0
            }
        })
        .collect();
    let (vecs, query) = floats.split_at(n * DIM);

    let mut idx = RankQuant::new(DIM, bits);
    idx.add(vecs);
    let _ = idx.search_asymmetric(query, k);
});
