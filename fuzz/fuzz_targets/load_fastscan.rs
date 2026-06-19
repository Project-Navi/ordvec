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
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or `Err(..)` —
//! never panic, abort, or read out of bounds. libFuzzer treats any panic/abort
//! as a crash, so simply letting the result drop is the assertion.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The only thing under test: arbitrary bytes -> Ok | Err, no panic.
    let _ = ordvec::RankQuantFastscan::load_from_bytes(data);
});
