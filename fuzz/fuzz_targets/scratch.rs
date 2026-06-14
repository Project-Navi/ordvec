//! Shared per-worker scratch temp file for the `.tvr` / `.tvrq` / `.tvbm` /
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
//! tracked separately and out of scope for a fuzz-only change.
//!
//! What this *does* remove is the avoidable per-iteration filesystem churn the
//! issue calls out: instead of allocating a fresh `NamedTempFile` (an
//! `mkstemp` + `open`) and unlinking it on drop every single iteration, each
//! fuzzer worker creates **one** temp file and rewrites it in place. The loader
//! still runs its exact real path (`File::open` + `metadata().len()` + parse) on
//! the precise fuzz bytes, so the loader code path and the corpus/format
//! contract are unchanged.
//!
//! # Storage scope (per worker thread)
//!
//! The scratch file lives in a [`thread_local!`], i.e. one file **per worker
//! thread** — not a single shared file across threads. libFuzzer drives each
//! fuzz target from a single thread, and fork mode runs each job in its own
//! process (hence its own thread-local), so in practice this is one file per
//! fuzzer worker, never shared between concurrent workers — so reuse is
//! race-free. The file is auto-removed when the thread/process exits
//! (`NamedTempFile` drop), so a multi-million-run campaign does not leak into
//! `$TMPDIR`.
//!
//! # Determinism & truncation
//!
//! Reusing a file means a shorter iteration must not inherit trailing bytes from
//! a longer previous one — that would change the bytes the loader sees and break
//! determinism (a fresh `NamedTempFile` gave clean truncation for free). Each
//! call rewinds, writes exactly `data`, and truncates to `data.len()` **only
//! when the new input is shorter** than the previous one (a longer-or-equal
//! write already overwrites the old length). Identical input therefore always
//! yields identical loader input and behaviour.

use std::cell::RefCell;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

/// One reused temp file plus the byte length last written to it, so we only
/// `set_len` (truncate) when the next input is strictly shorter.
struct Scratch {
    file: NamedTempFile,
    len: usize,
}

thread_local! {
    /// One temp file per worker thread, reused across iterations. `None` until
    /// the first successful create, and reset to `None` after any IO error so a
    /// broken descriptor is discarded and recreated on the next iteration.
    static SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

/// Write `data` to a per-worker scratch file and invoke `run` with its path.
///
/// On any transient temp-file/IO error (create, seek, write, truncate, flush)
/// the scratch state is reset and the iteration is skipped without calling
/// `run` — such failures are environmental, not loader bugs, so they must not be
/// reported as crashes.
///
/// The thread-local `RefCell` borrow is released **before** `run` is invoked, so
/// a `run` that (directly or indirectly) re-enters `with_scratch_file` will not
/// trip a `RefCell` double-borrow panic — the helper is *borrow-safe*. It is
/// **not** safe for genuinely nested use, however: there is one scratch file per
/// worker thread, so a nested call would rewrite the same file the outer call is
/// still pointing `run` at, clobbering its bytes. The loader fuzz targets never
/// nest (one synchronous call per iteration), so this is a forward-looking
/// caveat, not a current bug.
pub fn with_scratch_file<F>(data: &[u8], run: F)
where
    F: FnOnce(&Path),
{
    // Prepare the file under the borrow and hand back an owned path; the borrow
    // is dropped when this closure returns, before `run` is called below.
    let path: Option<PathBuf> = SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match NamedTempFile::new() {
                Ok(file) => *slot = Some(Scratch { file, len: 0 }),
                Err(_) => return None,
            }
        }
        let scratch = slot.as_mut().expect("scratch temp file initialized above");

        // Overwrite the file with exactly `data`: rewind, write, then truncate
        // to the new length only when it shrank (a longer/equal write already
        // overwrites the old bytes) so stale trailing bytes from a longer
        // previous iteration cannot leak in.
        let ok = {
            let file = scratch.file.as_file_mut();
            file.seek(SeekFrom::Start(0)).is_ok()
                && file.write_all(data).is_ok()
                && (data.len() >= scratch.len || file.set_len(data.len() as u64).is_ok())
                && file.flush().is_ok()
        };
        if ok {
            scratch.len = data.len();
            Some(scratch.file.path().to_path_buf())
        } else {
            // Discard the possibly-broken descriptor; the next call recreates it.
            *slot = None;
            None
        }
    });

    if let Some(path) = path {
        run(&path);
    }
}
