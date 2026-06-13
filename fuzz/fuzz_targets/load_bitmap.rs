//! libFuzzer target for the `.tvbm` / `TVBM` loader, driven through the
//! public `ordvec::Bitmap::load` entry point.
//!
//! The low-level `rank_io::load_bitmap` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `Bitmap::load` — which
//! runs that exact loader and then the type's post-load checks (the full
//! public load path). `load` takes a `&Path`, and the only public load entry
//! points are path-based (there is no public `&[u8]`/`Read` loader — issue
//! #6), so a shared process-local scratch file (see [`scratch`]) feeds the
//! loader the fuzz bytes without the per-iteration `mkstemp`/`unlink` churn a
//! fresh `NamedTempFile` each run would incur.
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion.

#![no_main]

use libfuzzer_sys::fuzz_target;

mod scratch;

fuzz_target!(|data: &[u8]| {
    scratch::with_scratch_file(data, |path| {
        let _ = ordvec::Bitmap::load(path);
    });
});
