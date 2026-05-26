//! libFuzzer target for the FastScan b=2 compute path (`RankQuantFastscan`):
//! `add` (rank_transform -> bucket -> block-32 re-pack via `pack_fastscan_b2`)
//! then `search` (`search_asymmetric_fastscan_b2` -> the scalar / AVX-512
//! VPSHUFB-LUT kernel -> TopK). This is the one `unsafe`-heavy scan path the
//! `search_rankquant` target does NOT reach: `RankQuant::search_asymmetric`
//! dispatches the single-rate kernels, never the FastScan block-32 kernel.
//!
//! `dim` is fixed at 64 — `RankQuantFastscan::new` requires `dim % 4 == 0`
//! (b=2 constant composition) and `dim <= u16::MAX`; 64 also gives a
//! `dim / 2 = 32`-pair inner loop. The fuzzer shapes the doc count (crossing
//! the 32-doc block boundary so tail-padding blocks are exercised), the
//! embedding/query values, and `k` (including `k == 0`). Values map to finite
//! f32: the public API rejects NaN / ±Inf by contract, so raw float bit
//! patterns would only re-exercise that guard, not the kernel.
//!
//! On CI runners without AVX-512 this drives the scalar reference kernel
//! (`scan_b2_fastscan_scalar`); under Intel SDE it drives the AVX-512 kernel.
//!
//! Contract: no panic, abort, or out-of-bounds access on any input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ordvec::RankQuantFastscan;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    // dim % 4 == 0 and dim <= u16::MAX (RankQuantFastscan::new contract).
    const DIM: usize = 64;
    // 1..=100 docs — crosses the 32-doc block boundary (1..=4 blocks) so the
    // tail-padding path (`n % 32 != 0`) is exercised.
    let n = (data[0] as usize % 100) + 1;
    let k = data[1] as usize % (n + 1); // 0..=n

    let payload = &data[2..];
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

    let mut idx = RankQuantFastscan::new(DIM);
    idx.add(vecs);
    let _ = idx.search(query, k);
});
