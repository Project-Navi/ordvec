//! libFuzzer target for the `.tvrq` / `TVRQ` loader, driven through the
//! public `ordvec::RankQuant::load` entry point.
//!
//! The low-level `rank_io::load_rankquant` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `RankQuant::load` —
//! which runs that exact loader and then the type's post-load checks (the
//! full public load path). `load` takes a `&Path`, and the only public load
//! entry points are path-based (there is no public `&[u8]`/`Read` loader —
//! issue #6), so a shared process-local scratch file (see [`scratch`]) feeds
//! the loader the fuzz bytes without the per-iteration `mkstemp`/`unlink`
//! churn a fresh `NamedTempFile` each run would incur.
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or
//! `Err(..)` — never panic, abort, or read out of bounds. libFuzzer
//! treats any panic/abort as a crash, so simply letting the result drop
//! is the assertion. This loader has the densest header validation
//! (`bits ∈ {1,2,4}`, `dim % 2^bits == 0`, `dim % (8/bits) == 0`,
//! checked-mul payload sizing) — exactly the surface fuzzing exercises.

#![no_main]

use libfuzzer_sys::fuzz_target;

mod scratch;

fuzz_target!(|data: &[u8]| {
    scratch::with_scratch_file(data, |path| {
        let _ = ordvec::RankQuant::load(path);
    });
});
