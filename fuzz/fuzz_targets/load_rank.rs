//! libFuzzer target for the `.tvr` / `TVR1` loader, driven through the
//! public `ordvec::Rank::load` entry point.
//!
//! The low-level `rank_io::load_rank` parser is crate-internal (`pub(crate)`),
//! so the fuzzer exercises it through `Rank::load` — which runs that exact
//! loader and then the type's post-load length check (the full public load
//! path). `load` takes a `&Path`, so each iteration writes the arbitrary
//! input to a unique temp file (auto-cleaned by `tempfile`).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion.

#![no_main]

use std::io::Write;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // `NamedTempFile` gives a unique path per run and removes the file on
    // drop, so a multi-million-run campaign does not leak into $TMPDIR.
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        // A transient temp-file failure is environmental, not a loader
        // bug — skip this input rather than crash the fuzzer.
        Err(_) => return,
    };
    if tmp.write_all(data).is_err() {
        return;
    }
    if tmp.flush().is_err() {
        return;
    }

    // The only thing under test: arbitrary bytes -> Ok | Err, no panic.
    let _ = ordvec::Rank::load(tmp.path());
});
