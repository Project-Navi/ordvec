//! libFuzzer target for the `.tvsb` / `TVSB` loader, driven through the
//! public `ordvec::SignBitmap::load` entry point.
//!
//! The low-level `rank_io::load_sign_bitmap` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `SignBitmap::load` —
//! which runs that exact loader and then the type's post-load checks (the
//! full public load path). `load` takes a `&Path`, so each iteration writes
//! the arbitrary input to a unique temp file (auto-cleaned by `tempfile`).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion. The `.tvsb` dim validation path differs from the
//! other three (`MAX_SIGN_BITMAP_DIM`, multiple-of-64), so it gets its
//! own target rather than riding on `load_bitmap`.

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

    let _ = ordvec::SignBitmap::load(tmp.path());
});
