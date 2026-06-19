//! libFuzzer target for the `.ovrq` / `OVRQ` loader (which also accepts the
//! legacy `.tvrq` / `TVRQ` magic), driven through the public
//! `ordvec::RankQuant::load_from_bytes` entry point.
//!
//! The low-level `rank_io::load_rankquant` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through
//! `RankQuant::load_from_bytes` — which runs that exact loader and then the
//! type's post-load checks (the full public in-memory load path).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion. This loader has the densest header validation
//! (`bits ∈ {1,2,4}`, `dim % 2^bits == 0`, `dim % (8/bits) == 0`,
//! checked-mul payload sizing) — exactly the surface fuzzing exercises.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ordvec::RankQuant::load_from_bytes(data);
});
