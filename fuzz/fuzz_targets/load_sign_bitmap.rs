//! libFuzzer target for the `.tvsb` / `TVSB` loader, driven through the
//! public `ordvec::SignBitmap::load` entry point.
//!
//! The low-level `rank_io::load_sign_bitmap` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `SignBitmap::load` —
//! which runs that exact loader and then the type's post-load checks (the full
//! public load path). `load` takes a `&Path`, and the only public load entry
//! points are path-based (there is no public `&[u8]`/`Read` loader — issue
//! #6), so a shared process-local scratch file (see [`scratch`]) feeds the
//! loader the fuzz bytes without the per-iteration `mkstemp`/`unlink` churn a
//! fresh `NamedTempFile` each run would incur.
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion. The `.tvsb` dim validation path differs from the
//! other three (`MAX_SIGN_BITMAP_DIM`, multiple-of-64), so it gets its
//! own target rather than riding on `load_bitmap`.

#![no_main]

use libfuzzer_sys::fuzz_target;

mod scratch;

fuzz_target!(|data: &[u8]| {
    scratch::with_scratch_file(data, |path| {
        let _ = ordvec::SignBitmap::load(path);
    });
});
