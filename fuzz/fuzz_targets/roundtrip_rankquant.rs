//! libFuzzer target for the RankQuant write -> load round-trip. The `load_*`
//! targets feed arbitrary bytes straight into the loaders; this instead builds
//! a real index from fuzzer-shaped data, persists it via the public `write`,
//! and requires `load` to accept the result and preserve shape — the type-level
//! round-trip guarantee (`write_rankquant` is `pub(crate)`, reachable only here
//! via `RankQuant::write`). It exercises the write path the loader targets
//! cannot reach.
//!
//! Contract: `write` then `load` must succeed and round-trip; failing to reload
//! self-produced output is a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ordvec::RankQuant;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    const DIM: usize = 64;
    let bits: u8 = match data[0] % 3 {
        0 => 1,
        1 => 2,
        _ => 4,
    };
    let n = (data[1] as usize % 16) + 1;

    let payload = &data[2..];
    let vecs: Vec<f32> = (0..n * DIM)
        .map(|i| {
            if payload.is_empty() {
                0.0
            } else {
                payload[i % payload.len()] as f32 - 128.0
            }
        })
        .collect();

    let mut idx = RankQuant::new(DIM, bits);
    idx.add(&vecs);

    let tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    if idx.write(tmp.path()).is_err() {
        return;
    }
    let reloaded = RankQuant::load(tmp.path()).expect("write output must reload (round-trip)");
    assert_eq!(reloaded.dim(), idx.dim());
    assert_eq!(reloaded.len(), idx.len());
});
