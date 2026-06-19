//! libFuzzer target for the `.ovbm` / `OVBM` loader (which also accepts the
//! legacy `.tvbm` / `TVBM` magic), driven through the public
//! `ordvec::Bitmap::load_from_bytes` entry point.
//!
//! The low-level `rank_io::load_bitmap` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through
//! `Bitmap::load_from_bytes` — which runs that exact loader and then the
//! type's post-load checks (the full public in-memory load path).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ordvec::Bitmap::load_from_bytes(data);
});
