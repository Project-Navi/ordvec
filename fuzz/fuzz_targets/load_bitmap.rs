//! libFuzzer target for the `.tvbm` / `TVBM` loader, driven through the
//! public `ordvec::Bitmap::load` entry point.
//!
//! The low-level `rank_io::load_bitmap` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `Bitmap::load` — which
//! runs that exact loader and then the type's post-load checks (the full
//! public load path). `load` takes a `&Path`, so each iteration writes the
//! arbitrary input to a unique temp file (auto-cleaned by `tempfile`).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion.

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

    let _ = ordvec::Bitmap::load(tmp.path());
});
