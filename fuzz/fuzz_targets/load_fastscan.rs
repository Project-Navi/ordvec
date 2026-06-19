//! libFuzzer target for the `.ovfs` / `OVFS` loader (the FastScan b=2
//! persistence format — new in the ordvec format, no legacy `TV*` magic),
//! driven through the public `ordvec::RankQuantFastscan::load_from_bytes`
//! entry point.
//!
//! The low-level `rank_io::load_fastscan` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through
//! `RankQuantFastscan::load_from_bytes` — which runs that exact loader (the
//! full public in-memory load path).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or `Err(..)`;
//! if it returns `Ok`, at least one safe search must also complete without
//! panic, abort, or read out of bounds. libFuzzer treats any panic/abort as a
//! crash.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(index) = ordvec::RankQuantFastscan::load_from_bytes(data) {
        let query = vec![0.0f32; index.dim()];
        let _ = index.search(&query, 1);
    }
});
