//! libFuzzer target for [`ordvec::rank_io::load_rankquant`] (the `.tvrq`
//! / `TVRQ` loader for [`ordvec::RankQuantIndex`]).
//!
//! `rank_io` is `pub` in the crate root, so we drive the loader directly
//! rather than through `RankQuantIndex::load` — same loader code, one
//! fewer wrapper. The loader takes a `&Path`, so each iteration writes
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

    let _ = ordvec::rank_io::load_rankquant(tmp.path());
});
