//! libFuzzer target for [`ordvec::rank_io::load_bitmap`] (the `.tvbm` /
//! `TVBM` loader for [`ordvec::BitmapIndex`]).
//!
//! `rank_io` is `pub` in the crate root, so we drive the loader directly
//! rather than through `BitmapIndex::load` — same loader code, one fewer
//! wrapper. The loader takes a `&Path`, so each iteration writes the
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

    let _ = ordvec::rank_io::load_bitmap(tmp.path());
});
