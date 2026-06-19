//! libFuzzer target for the `.ovsb` / `OVSB` loader (which also accepts the
//! legacy `.tvsb` / `TVSB` magic), driven through the public
//! `ordvec::SignBitmap::load_from_bytes` entry point.
//!
//! The low-level `rank_io::load_sign_bitmap` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through
//! `SignBitmap::load_from_bytes` — which runs that exact loader and then the
//! type's post-load checks (the full public in-memory load path).
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion. The `.ovsb` dim validation path differs from the
//! other three (`MAX_SIGN_BITMAP_DIM`, multiple-of-64), so it gets its
//! own target rather than riding on `load_bitmap`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ordvec::SignBitmap::load_from_bytes(data);
});
