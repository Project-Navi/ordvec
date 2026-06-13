//! Shared per-process scratch temp file for the `.tvr` / `.tvrq` / `.tvbm` /
//! `.tvsb` loader fuzz targets.
//!
//! # Why this exists (issue #6)
//!
//! The four `load_*` targets exercise the production loaders, but the only
//! public load entry points (`Rank::load` / `RankQuant::load` / `Bitmap::load`
//! / `SignBitmap::load`, and `probe_index_metadata`) take a `&Path` and open a
//! real file — the low-level `rank_io::load_*` parsers (which operate on a
//! generic `R: Read + Seek` and would accept a `Cursor<&[u8]>`) are
//! `pub(crate)` and unreachable from this external fuzz crate. A *true*
//! zero-temp-file in-memory driver therefore needs a new **public core API**
//! (e.g. `Type::load_from_bytes(&[u8])` or a `pub` `Read + Seek` loader),
//! which is out of scope for a fuzz-only change.
//!
//! What this *does* remove is the avoidable per-iteration filesystem churn the
//! issue calls out: instead of allocating a fresh `NamedTempFile` (an
//! `mkstemp` + `open`) and unlinking it on drop every single iteration, each
//! fuzzer process creates **one** temp file and rewrites it in place. The
//! loader still runs its exact real path (`File::open` + `metadata().len()` +
//! parse) on the precise fuzz bytes, so the loader code path and the
//! corpus/format contract are unchanged.
//!
//! # Determinism & truncation
//!
//! Reusing a file means a shorter iteration must not inherit trailing bytes
//! from a longer previous one — that would change the bytes the loader sees and
//! break determinism (a fresh `NamedTempFile` gave clean truncation for free).
//! So each call rewinds, writes exactly `data`, and `set_len`s the file to
//! `data.len()`. Identical input therefore always yields identical loader input
//! and behaviour.
//!
//! # Parallel-fuzz safety
//!
//! Each libFuzzer job is a separate process and gets its own unique temp path
//! (`NamedTempFile::new`); iterations within a process are sequential, so
//! reusing one path per process is race-free. The file is auto-removed when the
//! process exits (`NamedTempFile` drop), so a multi-million-run campaign does
//! not leak into `$TMPDIR`.

use std::cell::RefCell;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use tempfile::NamedTempFile;

thread_local! {
    /// One temp file per fuzzer process, reused across iterations. `None` until
    /// the first successful create.
    static SCRATCH: RefCell<Option<NamedTempFile>> = const { RefCell::new(None) };
}

/// Write `data` to a process-local scratch file and invoke `run` with its path.
///
/// On any transient temp-file/IO error (create, seek, write, truncate, flush)
/// the iteration is skipped without calling `run` — such failures are
/// environmental, not loader bugs, so they must not be reported as crashes.
pub fn with_scratch_file<F>(data: &[u8], run: F)
where
    F: FnOnce(&Path),
{
    SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match NamedTempFile::new() {
                Ok(t) => *slot = Some(t),
                Err(_) => return,
            }
        }
        let tmp = slot.as_mut().expect("scratch temp file initialized above");

        // Overwrite the file with exactly `data`: rewind, write, then truncate
        // to the new length so stale bytes from a longer previous iteration
        // cannot leak in.
        let file = tmp.as_file_mut();
        if file.seek(SeekFrom::Start(0)).is_err()
            || file.write_all(data).is_err()
            || file.set_len(data.len() as u64).is_err()
            || file.flush().is_err()
        {
            return;
        }

        run(tmp.path());
    });
}
