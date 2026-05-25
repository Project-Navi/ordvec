//! libFuzzer target for the `.tvrq` / `TVRQ` loader, driven through the
//! public `ordvec::RankQuant::load` entry point.
//!
//! The low-level `rank_io::load_rankquant` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `RankQuant::load` —
//! which runs that exact loader and then the type's post-load checks (the
//! full public load path). `load` takes a `&Path`, so each iteration writes
//! the arbitrary input to a unique temp file (auto-cleaned by `tempfile`).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion. This loader has the densest header validation
//! (`bits ∈ {1,2,4}`, `dim % 2^bits == 0`, `dim % (8/bits) == 0`,
//! checked-mul payload sizing) — exactly the surface fuzzing exercises.

#![no_main]

use std::io::Write;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    if tmp.write_all(data).is_err() {
        return;
    }
    if tmp.flush().is_err() {
        return;
    }

    let _ = ordvec::RankQuant::load(tmp.path());
});
