//! libFuzzer target for the RankQuant write -> load round-trip. The `load_*`
//! targets feed arbitrary bytes straight into the loaders; this instead builds
//! a real index from fuzzer-shaped data, persists it via the public `write`,
//! and requires `load` to accept the result and preserve shape — the type-level
//! round-trip guarantee (`write_rankquant` is `pub(crate)`, reachable only here
//! via `RankQuant::write`). It exercises the write path the loader targets
//! cannot reach.
//!
//! Contract: `write` of a validly-built index, then `load`, must succeed and
//! round-trip; a write failure OR a failure to reload self-produced output is a
//! crash. The index is written to a fresh path inside a tempdir — not an
//! already-open `NamedTempFile` handle, which a reopen-by-path write can fail to
//! overwrite on some platforms.
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

    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let path = dir.path().join("roundtrip.tvrq");
    idx.write(&path).expect("write of a validly-built index must succeed");
    let reloaded = RankQuant::load(&path).expect("write output must reload (round-trip)");
    assert_eq!(reloaded.dim(), idx.dim());
    assert_eq!(reloaded.len(), idx.len());
});
