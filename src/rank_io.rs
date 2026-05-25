//! Read/write ordinal/sign index files.
//!
//! Four formats live here, each self-describing via a 4-byte magic:
//! * `.tvr`  — [`Rank`](crate::Rank) — magic `TVR1`
//! * `.tvrq` — [`RankQuant`](crate::RankQuant) — magic `TVRQ`
//! * `.tvbm` — [`Bitmap`](crate::Bitmap) — magic `TVBM`
//! * `.tvsb` — [`SignBitmap`](crate::SignBitmap) — magic `TVSB`
//!
//! All formats are little-endian. Headers are small fixed-size structs
//! followed by a single contiguous payload (the rank / packed / bitmap
//! bytes). No norms, no codebooks, no rotation matrices — these are the
//! deterministic-encode index types so the on-disk format is exactly the
//! in-memory buffer plus enough header to rehydrate the type parameters.
//!
//! Each format is a minimal fixed-size header followed by a contiguous
//! payload. ID-map wrappers (analogous to `.tvim`) are an obvious
//! follow-up but not in this v1.
//!
//! # Safety against malformed files
//!
//! All loaders validate header fields *before* allocating the payload
//! buffer:
//! * `dim` and `n_vectors` are bounded by [`MAX_DIM`] and [`MAX_VECTORS`].
//! * `bits` is checked against `{1, 2, 4}` before any multiplication.
//! * Total payload size is computed via [`usize::checked_mul`] and
//!   rejected if it overflows or exceeds the 128 GiB `MAX_PAYLOAD` cap.
//!   (`MAX_DIM * MAX_VECTORS` alone is ~8 TiB, so `MAX_PAYLOAD` is the
//!   binding byte ceiling, not the `dim` / `n_vectors` caps.)
//! * The declared payload must match the file's remaining bytes
//!   *exactly* — a structurally-valid file with trailing bytes is
//!   rejected (v1 formats have no footer or reserved trailing section).
//! * Per-index invariants (e.g., `dim % (1 << bits) == 0` for RankQuant)
//!   are returned as `Err(InvalidData)`, never `assert!`'d.
//!
//! Any malformed input returns `io::Error` rather than panicking.
//!
//! # Persistence API & round-trip contract
//!
//! The supported persistence API is the index types' `write()` / `load()`
//! methods: [`Rank`](crate::Rank) / [`RankQuant`](crate::RankQuant) /
//! [`Bitmap`](crate::Bitmap) / [`SignBitmap`](crate::SignBitmap). The
//! `write_*` / `load_*` format helpers in this module are **crate-internal**
//! (`pub(crate)`); only the `MAX_*` capacity constants are public.
//!
//! Round-trip is a guarantee of the **index types**: each constructor
//! validates its parameters (matching the loaders' `dim` / `n_top` / `bits` /
//! divisibility bounds), `add` caps `n_vectors` at [`MAX_VECTORS`], and the
//! types emit only loader-valid data — so anything `T::write` produces,
//! `T::load` reloads. The raw `write_*` helpers are trusted serializers: they
//! assume loader-valid inputs (which only the index types construct) and do
//! *not* re-validate `dim` / `n_vectors` / structure / data semantics. The
//! 128 GiB `MAX_PAYLOAD` cap is the one loader bound they also enforce
//! (checked before `File::create`, so a rejected oversized write never
//! truncates an existing file) — defense-in-depth, and belt-and-braces now
//! that the helpers are no longer reachable with arbitrary external input.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

const TVR_MAGIC: &[u8; 4] = b"TVR1";
const TVRQ_MAGIC: &[u8; 4] = b"TVRQ";
const TVBM_MAGIC: &[u8; 4] = b"TVBM";
const TVSB_MAGIC: &[u8; 4] = b"TVSB";
const VERSION: u8 = 1;

/// Largest accepted `dim` from a loaded file. Matches `u16::MAX` so the
/// rank transform's `u16` invariant in [`crate::Rank`] is honoured.
pub const MAX_DIM: usize = u16::MAX as usize;
/// Largest accepted `dim` for sign-bitmap files. The rank-storage
/// invariant (`u16` ranks) does not apply here, so the cap is the
/// on-disk u32 header field clamped to a safe multiple of 64. Set to
/// `1 << 24 = 16_777_216` — comfortably above any realistic embedding
/// dimensionality while bounded well within usize math.
pub const MAX_SIGN_BITMAP_DIM: usize = 1 << 24;
/// Largest accepted `n_vectors` — a document *count* cap. 64 M docs at
/// `dim=u16::MAX` (128 KiB / vec for u16 ranks) tops out at ~8 TiB, well
/// past any sane on-disk index. Chosen to fail loud before allocation
/// panics. The total *byte payload* is bounded independently by the 128 GiB
/// `MAX_PAYLOAD` cap (see `check_payload_bytes`), which both the load and
/// write paths enforce — so an index whose `dim * n_vectors` payload exceeds
/// it cannot be persisted even when `n_vectors` is within this cap.
pub const MAX_VECTORS: usize = 64 * 1024 * 1024;

fn invalid<S: Into<String>>(msg: S) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Allocate a zeroed `Vec<u8>` of `n` bytes using *fallible* allocation.
///
/// `vec![0u8; n]` aborts the process on allocation failure (the abort is
/// not a recoverable `io::Error`). Sizes here are derived from
/// attacker-influenced headers, so reserve via `try_reserve_exact` and only
/// then `resize` — an OOM becomes `InvalidData` instead of `SIGABRT`.
fn try_alloc_zeroed(n: usize) -> io::Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(n)
        .map_err(|_| invalid("payload allocation too large"))?;
    buf.resize(n, 0);
    Ok(buf)
}

/// Read `n` little-endian `W`-byte elements directly into a fallibly
/// pre-reserved Vec, so an oversized/under-memory load returns an
/// io::Error instead of aborting (and avoids the 2x byte-buffer + typed-Vec peak).
///
/// Building the typed `Vec` via `bytes.chunks_exact(W).map(..).collect()`
/// uses an infallible allocation — an OOM there is a `SIGABRT`, not a
/// recoverable error — and holds both the byte buffer and the typed `Vec`
/// live at once (2x peak). Reserving fallibly and reading element-by-element
/// removes both problems. The size guards ([`check_payload_bytes`],
/// [`check_payload_matches_file`]) run *before* this call; `read_le_vec`
/// reserves the same element count those guards validated.
fn read_le_vec<R: Read, T, const W: usize>(
    r: &mut R,
    n: usize,
    parse: impl Fn([u8; W]) -> T,
) -> io::Result<Vec<T>> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| invalid("payload allocation too large"))?;
    let mut buf = [0u8; W];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        v.push(parse(buf));
    }
    Ok(v)
}

/// Reject a declared payload whose size does not *exactly* match the
/// file's remaining bytes.
///
/// `reader` is positioned just past the header; `file_len` is the file's
/// total length. The payload is the sole, final section of every v1
/// format (no footer, no appended sections), so the declared payload
/// must consume the rest of the file exactly:
/// * `payload > remaining` catches a forged "tiny header claims
///   gigabytes" before any allocation — the primary size defense, with
///   [`try_alloc_zeroed`] as defense-in-depth.
/// * `payload < remaining` rejects a structurally-valid file with
///   trailing bytes (corruption, or a record smuggling extra data past
///   a smaller declared payload). A canonical index file has no slack;
///   a future format that reserves a trailing section will carry a new
///   magic/version and its own loader.
///
/// `stream_position` gives the bytes already consumed without manual
/// offset accounting.
fn check_payload_matches_file<R: Seek>(
    reader: &mut R,
    file_len: u64,
    payload_bytes: usize,
) -> io::Result<()> {
    let pos = reader.stream_position()?;
    let remaining = file_len.saturating_sub(pos);
    if payload_bytes as u64 != remaining {
        return Err(invalid(format!(
            "declared payload ({payload_bytes} B) does not match remaining \
             file size ({remaining} B): truncated, forged, or trailing bytes"
        )));
    }
    Ok(())
}

fn check_dim(dim: usize) -> io::Result<()> {
    if !(2..=MAX_DIM).contains(&dim) {
        return Err(invalid(format!("dim {dim} out of range [2, {MAX_DIM}]")));
    }
    Ok(())
}

/// Dimension check for `.tvsb` sign-bitmap files.
///
/// The `u16::MAX` ceiling in [`check_dim`] exists to honour
/// [`crate::Rank`]'s `u16` rank-storage invariant. Sign bitmaps
/// have no such constraint — `dim` is just a bit count — so this check
/// uses [`MAX_SIGN_BITMAP_DIM`] instead. Without it, any
/// `SignBitmap::new(d)` with `d > u16::MAX` could be written but
/// would fail on load, breaking roundtrip persistence.
fn check_sign_bitmap_dim(dim: usize) -> io::Result<()> {
    if !(64..=MAX_SIGN_BITMAP_DIM).contains(&dim) {
        return Err(invalid(format!(
            "TVSB dim {dim} out of range [64, {MAX_SIGN_BITMAP_DIM}]"
        )));
    }
    if !dim.is_multiple_of(64) {
        return Err(invalid(format!("TVSB dim {dim} is not a multiple of 64")));
    }
    Ok(())
}

fn check_n_vectors(n_vectors: usize) -> io::Result<()> {
    if n_vectors > MAX_VECTORS {
        return Err(invalid(format!(
            "n_vectors {n_vectors} exceeds MAX_VECTORS={MAX_VECTORS}"
        )));
    }
    Ok(())
}

fn check_payload_bytes(payload_bytes: usize) -> io::Result<()> {
    // 128 GiB hard cap — refuses absurd allocations from a corrupt
    // header even if dim and n_vectors individually pass. Called on BOTH
    // paths: on load (against a possibly-forged header) and on write as
    // defense-in-depth (the catastrophic unloadable-file case is an oversized
    // payload; checking before File::create also avoids truncating an existing
    // file). This is the only loader bound the raw `write_*` share — full
    // round-trip is a type-level guarantee (see module docs). Typed `u64` (not
    // `usize`) so the literal doesn't overflow const-eval on 32-bit targets
    // (wasm32, armv7), where `usize::MAX` (~4 GiB) is already the ceiling and
    // the widened comparison simply never trips.
    const MAX_PAYLOAD: u64 = 128 * 1024 * 1024 * 1024;
    if payload_bytes as u64 > MAX_PAYLOAD {
        return Err(invalid(format!(
            "payload {payload_bytes} B exceeds MAX_PAYLOAD={MAX_PAYLOAD}"
        )));
    }
    Ok(())
}

// -------------------------------------------------------------------
// Rank: u16 ranks per coordinate.
// Header: magic(4) | version(1) | dim(u32 LE) | n_vectors(u32 LE)  = 13 B
// Payload: n_vectors * dim * 2 bytes (u16 LE ranks).
// -------------------------------------------------------------------

pub(crate) fn write_rank(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    ranks: &[u16],
) -> io::Result<()> {
    // Enforce the loaders' MAX_PAYLOAD cap *before* File::create so a rejected
    // oversized write never truncates an existing file. Defense-in-depth; the
    // round-trip guarantee is type-level (see module docs). Mirrors load_rank.
    let payload_bytes = n_vectors
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(ranks.len(), payload_bytes / 2);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVR_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    for &r in ranks {
        f.write_all(&r.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub(crate) fn load_rank(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<u16>)> {
    let file = File::open(path)?;
    // Propagate a metadata() failure instead of swallowing it as `0`. With a
    // bogus `file_len == 0`, `check_payload_matches_file` would false-reject
    // every non-empty index (its `remaining` saturates to 0, never equal to a
    // positive `payload_bytes`) and, for an empty corpus, pass while skipping
    // the trailing-byte check. Both are wrong on a metadata race (NFS/procfs).
    let file_len = file.metadata()?.len();
    let mut f = BufReader::new(file);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVR_MAGIC {
        return Err(invalid("not a TVR1 file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVR1 version: {}", ver[0])));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_dim(dim)?;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = n_vectors
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * dim * 2`, so the u16 element count is
    // `payload_bytes / 2`. Read directly into a fallibly reserved Vec<u16>
    // instead of allocating a byte buffer and `.collect()`-ing it — the old
    // intermediate was an infallible (abort-on-OOM) allocation that also
    // doubled peak memory.
    let ranks = read_le_vec(&mut f, payload_bytes / 2, u16::from_le_bytes)?;
    // Each stored document must be a *permutation* of `[0, dim)`, not merely
    // bounded by `dim`. The scoring math (`rank_norm`) assumes a permutation:
    // a non-permutation row (all-zero, or with repeats) passes a bound check
    // but silently corrupts the Spearman score. Verify each row is a bijection
    // of `[0, dim)` at the loader boundary so a forged or corrupted file fails
    // loud instead of returning a wrong-but-not-crashing result.
    //
    // O(dim) per row with a reusable O(dim) stamp buffer: `seen[v]` holds the
    // 1-based index of the row that last wrote `v`, so `v` is a duplicate
    // within the current row iff `seen[v] == stamp` — no per-row re-zeroing.
    // `n_vectors <= MAX_VECTORS` (<< u32::MAX), so the stamp never overflows or
    // collides with the `0 = unseen` sentinel. A row of exactly `dim` values
    // that are all `< dim` and all distinct is necessarily a permutation
    // (pigeonhole), so the bound + duplicate checks together prove it.
    let mut seen = vec![0u32; dim];
    for (row_idx, row) in ranks.chunks_exact(dim).enumerate() {
        let stamp = row_idx as u32 + 1;
        for &r in row {
            let ri = r as usize;
            if ri >= dim {
                return Err(invalid(format!(
                    "TVR1 rank value {r} >= dim ({dim}); ranks must be a permutation of [0, dim)"
                )));
            }
            if seen[ri] == stamp {
                return Err(invalid(format!(
                    "TVR1 row {row_idx} is not a permutation of [0, dim): value {r} repeats"
                )));
            }
            seen[ri] = stamp;
        }
    }
    Ok((dim, n_vectors, ranks))
}

// -------------------------------------------------------------------
// RankQuant: B-bit packed bucket vectors.
// Header: magic(4) | version(1) | bits(u8) | dim(u32 LE) | n_vectors(u32 LE) = 14 B
// Payload: n_vectors * dim * bits / 8 packed bytes.
// -------------------------------------------------------------------

pub(crate) fn write_rankquant(
    path: impl AsRef<Path>,
    bits: u8,
    dim: usize,
    n_vectors: usize,
    packed: &[u8],
) -> io::Result<()> {
    // Enforce the loaders' MAX_PAYLOAD cap *before* File::create (defense-in-
    // depth; a rejected write must not truncate an existing file). Mirrors
    // load_rankquant: checked multiply before the /8 divide.
    let payload_bytes = n_vectors
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(bits as usize))
        .map(|x| x / 8)
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(packed.len(), payload_bytes);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVRQ_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&[bits])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    f.write_all(packed)?;
    f.flush()?;
    Ok(())
}

pub(crate) fn load_rankquant(path: impl AsRef<Path>) -> io::Result<(u8, usize, usize, Vec<u8>)> {
    let file = File::open(path)?;
    // Propagate a metadata() failure instead of swallowing it as `0`. With a
    // bogus `file_len == 0`, `check_payload_matches_file` would false-reject
    // every non-empty index (its `remaining` saturates to 0, never equal to a
    // positive `payload_bytes`) and, for an empty corpus, pass while skipping
    // the trailing-byte check. Both are wrong on a metadata race (NFS/procfs).
    let file_len = file.metadata()?.len();
    let mut f = BufReader::new(file);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVRQ_MAGIC {
        return Err(invalid("not a TVRQ file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVRQ version: {}", ver[0])));
    }
    let mut bits_buf = [0u8; 1];
    f.read_exact(&mut bits_buf)?;
    let bits = bits_buf[0];
    if !matches!(bits, 1 | 2 | 4) {
        return Err(invalid(format!(
            "unsupported TVRQ bits: {bits} (expected 1, 2, or 4)"
        )));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_dim(dim)?;
    // Constant-composition invariants (documented at module level and
    // enforced by `RankQuant::new`): `dim` must be a multiple of
    // both `2^bits` (one bucket-rank slot per code value) and the
    // codes-per-byte packing factor `8 / bits`. Without these, a forged
    // header with an indivisible `dim` would yield a packed buffer the
    // bucket-rank decoder cannot interpret. `bits ∈ {1,2,4}` is already
    // validated above, so neither divisor is zero.
    let n_buckets = 1usize << bits;
    if !dim.is_multiple_of(n_buckets) {
        return Err(invalid(format!(
            "TVRQ dim {dim} is not a multiple of 2^bits = {n_buckets}; \
             constant-composition invariant violated"
        )));
    }
    let codes_per_byte = (8 / bits) as usize;
    if !dim.is_multiple_of(codes_per_byte) {
        return Err(invalid(format!(
            "TVRQ dim {dim} is not a multiple of codes_per_byte = {codes_per_byte}"
        )));
    }
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = n_vectors
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(bits as usize))
        .map(|x| x / 8)
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, file_len, payload_bytes)?;
    let mut packed = try_alloc_zeroed(payload_bytes)?;
    f.read_exact(&mut packed)?;
    // Constant-composition invariant: every document must place exactly
    // `dim / 2^bits` coordinates in each of the `2^bits` buckets. The
    // analytical `rankquant_norm` depends on this exact composition, so a
    // forged buffer with valid shape and in-range codes but skewed bucket
    // counts would silently corrupt every score. Histogram the unpacked codes
    // per row (MSB-first packing, matching `rank::pack_buckets`) and reject any
    // document whose composition is not uniform.
    let bytes_per_row = dim / codes_per_byte;
    let expected_per_bucket = dim / n_buckets;
    let mask = (1u8 << bits) - 1;
    let bits_u = bits as usize;
    for (row_idx, row) in packed.chunks_exact(bytes_per_row).enumerate() {
        let mut hist = [0usize; 16]; // n_buckets <= 2^4 = 16
        for &byte in row {
            for slot in 0..codes_per_byte {
                let shift = (codes_per_byte - 1 - slot) * bits_u;
                hist[((byte >> shift) & mask) as usize] += 1;
            }
        }
        for (bucket, &count) in hist[..n_buckets].iter().enumerate() {
            if count != expected_per_bucket {
                return Err(invalid(format!(
                    "TVRQ row {row_idx} violates constant composition: bucket {bucket} \
                     has {count} codes, expected {expected_per_bucket} (= dim / 2^bits)"
                )));
            }
        }
    }
    Ok((bits, dim, n_vectors, packed))
}

// -------------------------------------------------------------------
// Bitmap: top-n_top bitmap per document.
// Header: magic(4) | version(1) | dim(u32 LE) | n_top(u32 LE) | n_vectors(u32 LE) = 17 B
// Payload: n_vectors * dim / 8 bytes (qwords as u64 LE).
// -------------------------------------------------------------------

pub(crate) fn write_bitmap(
    path: impl AsRef<Path>,
    dim: usize,
    n_top: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let qpv = dim / 64;
    // Enforce the loaders' MAX_PAYLOAD cap *before* File::create (defense-in-
    // depth; a rejected write must not truncate an existing file). Mirrors
    // load_bitmap.
    let payload_bytes = n_vectors
        .checked_mul(qpv)
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(bitmaps.len(), payload_bytes / 8);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVBM_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_top as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    for &w in bitmaps {
        f.write_all(&w.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub(crate) fn load_bitmap(path: impl AsRef<Path>) -> io::Result<(usize, usize, usize, Vec<u64>)> {
    let file = File::open(path)?;
    // Propagate a metadata() failure instead of swallowing it as `0`. With a
    // bogus `file_len == 0`, `check_payload_matches_file` would false-reject
    // every non-empty index (its `remaining` saturates to 0, never equal to a
    // positive `payload_bytes`) and, for an empty corpus, pass while skipping
    // the trailing-byte check. Both are wrong on a metadata race (NFS/procfs).
    let file_len = file.metadata()?.len();
    let mut f = BufReader::new(file);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVBM_MAGIC {
        return Err(invalid("not a TVBM file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVBM version: {}", ver[0])));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_dim(dim)?;
    if !dim.is_multiple_of(64) {
        return Err(invalid(format!("TVBM dim {dim} is not a multiple of 64")));
    }
    let mut top_buf = [0u8; 4];
    f.read_exact(&mut top_buf)?;
    let n_top = u32::from_le_bytes(top_buf) as usize;
    if n_top == 0 || n_top >= dim {
        return Err(invalid(format!(
            "TVBM n_top {n_top} must satisfy 0 < n_top < dim ({dim})"
        )));
    }
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let qpv = dim / 64;
    let payload_bytes = n_vectors
        .checked_mul(qpv)
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * qpv * 8`, so the u64 element count is
    // `payload_bytes / 8`. Read directly into a fallibly reserved Vec<u64>
    // rather than allocating a byte buffer and `.collect()`-ing it.
    let bitmaps = read_le_vec(&mut f, payload_bytes / 8, u64::from_le_bytes)?;
    // Constant-composition invariant: every document bitmap must have exactly
    // `n_top` bits set (it flags the document's top `n_top` coordinates). The
    // hypergeometric null model and the documented `[0, n_top]` score range
    // both assume this, so a forged row with valid shape but a different
    // popcount would break both. Verify per-row popcount at the boundary.
    for (row_idx, row) in bitmaps.chunks_exact(qpv).enumerate() {
        let pop: u32 = row.iter().map(|w| w.count_ones()).sum();
        if pop as usize != n_top {
            return Err(invalid(format!(
                "TVBM row {row_idx} has {pop} bits set, expected n_top = {n_top}"
            )));
        }
    }
    Ok((dim, n_top, n_vectors, bitmaps))
}

/// Persist a [`crate::SignBitmap`] payload to a `.tvsb` file.
///
/// On-disk layout (little-endian throughout):
///
/// | offset | bytes | field                       |
/// |-------:|:-----:|-----------------------------|
/// | 0      | 4     | magic = `TVSB`              |
/// | 4      | 1     | version = 1                 |
/// | 5      | 4     | `dim` (u32)                 |
/// | 9      | 4     | `n_vectors` (u32)           |
/// | 13     | …     | `n_vectors * dim/64` u64s   |
///
/// 13-byte header — one u32 shorter than `TVBM` because SignBitmap
/// has no `n_top` parameter (the threshold is fixed at zero).
pub(crate) fn write_sign_bitmap(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let qpv = dim / 64;
    // Enforce the loaders' MAX_PAYLOAD cap *before* File::create (defense-in-
    // depth; a rejected write must not truncate an existing file). Mirrors
    // load_sign_bitmap.
    let payload_bytes = n_vectors
        .checked_mul(qpv)
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(bitmaps.len(), payload_bytes / 8);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVSB_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    for &w in bitmaps {
        f.write_all(&w.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// Load a `.tvsb` file written by `write_sign_bitmap`.
///
/// Validates magic, version, dim (must be in
/// `[64, MAX_SIGN_BITMAP_DIM]` and a multiple of 64), and `n_vectors`
/// (≤ `MAX_VECTORS`). Payload size is computed with `checked_mul` and
/// rejected if it overflows or exceeds the 128 GiB hard cap from
/// `check_payload_bytes`. Any malformed input returns
/// `io::Error::InvalidData`.
///
/// Dim validation deliberately does NOT use `check_dim`: that helper
/// caps at `u16::MAX` to honour [`crate::Rank`]'s `u16` rank
/// invariant, which sign bitmaps do not share. Sharing it would reject
/// valid `SignBitmap::new(d)` instances for any `d > 65535`,
/// breaking the constructor↔loader roundtrip.
pub(crate) fn load_sign_bitmap(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<u64>)> {
    let file = File::open(path)?;
    // Propagate a metadata() failure instead of swallowing it as `0`. With a
    // bogus `file_len == 0`, `check_payload_matches_file` would false-reject
    // every non-empty index (its `remaining` saturates to 0, never equal to a
    // positive `payload_bytes`) and, for an empty corpus, pass while skipping
    // the trailing-byte check. Both are wrong on a metadata race (NFS/procfs).
    let file_len = file.metadata()?.len();
    let mut f = BufReader::new(file);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVSB_MAGIC {
        return Err(invalid("not a TVSB file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVSB version: {}", ver[0])));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_sign_bitmap_dim(dim)?;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let qpv = dim / 64;
    let payload_bytes = n_vectors
        .checked_mul(qpv)
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * qpv * 8`, so the u64 element count is
    // `payload_bytes / 8`. Read directly into a fallibly reserved Vec<u64>
    // rather than allocating a byte buffer and `.collect()`-ing it.
    let bitmaps = read_le_vec(&mut f, payload_bytes / 8, u64::from_le_bytes)?;
    // No per-row composition invariant exists for sign bitmaps: a document is
    // `bit j = (coord_j > 0)`, so *any* bit pattern is a valid document (unlike
    // Rank's permutation or Bitmap/RankQuant's constant-composition rules). The
    // structural validation above (magic, version, dim, n_vectors, payload
    // length) is therefore complete for this format — nothing further to verify.
    Ok((dim, n_vectors, bitmaps))
}

#[cfg(test)]
mod tests {
    use super::{
        load_rank, load_rankquant, write_bitmap, write_rank, write_rankquant, write_sign_bitmap,
    };
    use crate::Rank;
    use std::io::Write;
    use std::path::PathBuf;

    /// Write `bytes` to a uniquely-named temp file and return its path.
    fn forge(suffix: &str, bytes: &[u8]) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!(
            "rank_io_test_{}_{}_{}",
            std::process::id(),
            nonce,
            suffix
        ));
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    }

    // -------------------------------------------------------------------
    // Loader semantic-validation red-team (TV-DESER-004 / 005). Moved here
    // from tests/redteam_delta.rs when the rank_io read/write helpers became
    // crate-internal (`pub(crate)`): they exercise the loaders directly, so
    // they live with the code now that the public persistence surface is the
    // index types' write()/load(). Both loaders must return Err, never panic.
    // -------------------------------------------------------------------

    /// TV-DESER-004: `load_rankquant` must reject `dim % (1 << bits) != 0`
    /// (bits=2, dim=6 → 6 % 4 = 2). Empty payload isolates the gate.
    #[test]
    fn tvdeser004_load_rankquant_rejects_dim_not_multiple_of_2pow_bits() {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVRQ");
        v.push(1); // version
        v.push(2); // bits = 2 → n_buckets = 4
        v.extend_from_slice(&6u32.to_le_bytes()); // dim = 6 (not a multiple of 4)
        v.extend_from_slice(&0u32.to_le_bytes()); // n_vectors = 0
        let path = forge("tvrq_dim6_bits2.tvrq", &v);
        let result = std::panic::catch_unwind(|| load_rankquant(&path));
        std::fs::remove_file(&path).ok();
        let result = result.expect("load_rankquant panicked on bits=2 dim=6");
        assert!(
            result.is_err(),
            "load_rankquant accepted bits=2 dim=6 (dim % 4 != 0); expected Err"
        );
    }

    /// TV-DESER-004 (other side): bits=4, dim=4 → 4 % 16 != 0.
    #[test]
    fn tvdeser004_load_rankquant_rejects_dim_smaller_than_buckets() {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVRQ");
        v.push(1);
        v.push(4); // bits = 4 → n_buckets = 16
        v.extend_from_slice(&4u32.to_le_bytes()); // dim = 4 (not a multiple of 16)
        v.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("tvrq_dim4_bits4.tvrq", &v);
        let result = std::panic::catch_unwind(|| load_rankquant(&path));
        std::fs::remove_file(&path).ok();
        let result = result.expect("load_rankquant panicked on bits=4 dim=4");
        assert!(
            result.is_err(),
            "load_rankquant accepted bits=4 dim=4 (dim % 16 != 0); expected Err"
        );
    }

    /// TV-DESER-004 happy path: bits=2, dim=8 (both invariants hold, empty
    /// corpus) must still load — the divisibility gate must not over-reject.
    #[test]
    fn tvdeser004_load_rankquant_accepts_valid_dim() {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVRQ");
        v.push(1);
        v.push(2);
        v.extend_from_slice(&8u32.to_le_bytes()); // dim = 8 (8 % 4 == 0)
        v.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("tvrq_dim8_bits2.tvrq", &v);
        let result = load_rankquant(&path);
        std::fs::remove_file(&path).ok();
        let (bits, dim, n, packed) = result.expect("valid TVRQ should load");
        assert_eq!(bits, 2);
        assert_eq!(dim, 8);
        assert_eq!(n, 0);
        assert!(packed.is_empty());
    }

    /// TV-DESER-005: `load_rank` must reject any rank value `>= dim`
    /// (ranks=[60000,1,2,3], dim=4). The payload length matches the header,
    /// so the loader reaches the per-value permutation scan.
    #[test]
    fn tvdeser005_load_rank_rejects_rank_value_ge_dim() {
        let ranks: [u16; 4] = [60000, 1, 2, 3];
        let mut v = Vec::new();
        v.extend_from_slice(b"TVR1");
        v.push(1);
        v.extend_from_slice(&4u32.to_le_bytes()); // dim
        v.extend_from_slice(&1u32.to_le_bytes()); // n_vectors
        for &r in &ranks {
            v.extend_from_slice(&r.to_le_bytes());
        }
        let path = forge("tvr_rank_ge_dim.tvr", &v);
        let result = std::panic::catch_unwind(|| load_rank(&path));
        std::fs::remove_file(&path).ok();
        let result = result.expect("load_rank panicked on rank >= dim");
        assert!(
            result.is_err(),
            "load_rank accepted ranks=[60000,1,2,3] with dim=4 (60000 >= dim); expected Err"
        );
    }

    /// TV-DESER-005 happy path: a true permutation of `0..dim` must load and
    /// round-trip its bytes — the value scan must not over-reject.
    #[test]
    fn tvdeser005_load_rank_accepts_valid_permutation() {
        let ranks: [u16; 8] = [0, 1, 2, 3, 3, 2, 1, 0];
        let mut v = Vec::new();
        v.extend_from_slice(b"TVR1");
        v.push(1);
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        for &r in &ranks {
            v.extend_from_slice(&r.to_le_bytes());
        }
        let path = forge("tvr_valid_perm.tvr", &v);
        let result = load_rank(&path);
        std::fs::remove_file(&path).ok();
        let (d, n, loaded) = result.expect("valid TVR1 should load");
        assert_eq!(d, 4);
        assert_eq!(n, 2);
        assert_eq!(loaded, ranks.to_vec());
    }

    // -------------------------------------------------------------------
    // Write-side payload guard (Codex stop-review). `write_*` enforce the
    // same MAX_PAYLOAD cap the loaders do, *before* File::create, so an
    // oversized write fails loud (InvalidData) without truncating an existing
    // file. Oversized dims pair with empty payload slices: the cap check
    // fires before the length assert, so no terabyte allocation occurs (and
    // on 32-bit targets the size product overflows usize first — still a
    // clean InvalidData error, never a panic).
    // -------------------------------------------------------------------
    #[test]
    fn writers_reject_oversized_payload_without_truncating() {
        use std::io::ErrorKind;
        let tmp_dir = std::env::temp_dir();
        // Per-run nonce (not just the pid, which the OS reuses) so a leftover
        // file from a prior aborted run can't make the `!exists()` checks fail.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = |s: &str| {
            tmp_dir.join(format!(
                "rank_io_write_guard_{}_{}_{}.bin",
                std::process::id(),
                nonce,
                s
            ))
        };

        // dim and n_vectors each individually pass the loaders' caps, but
        // their byte payload blows past MAX_PAYLOAD (128 GiB).
        let big_dim = u16::MAX as usize; // 65535 == MAX_DIM
        let big_n = 64 * 1024 * 1024; // == MAX_VECTORS; 65535 * 64Mi * 2 ≈ 8 TiB

        let pr = path("rank");
        let e = write_rank(&pr, big_dim, big_n, &[]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert!(
            !pr.exists(),
            "write_rank created a file despite rejecting the payload"
        );

        let prq = path("rankquant");
        let e = write_rankquant(&prq, 4, big_dim, big_n, &[]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert!(
            !prq.exists(),
            "write_rankquant created a file despite rejecting the payload"
        );

        // Bitmap/SignBitmap dims must be a multiple of 64 and within MAX_DIM
        // (65535); 32768/64 = 512 qwords/doc → 512 * 8 * 64Mi = 256 GiB > 128
        // GiB, so the payload guard fires on a loader-valid dim.
        let bm_dim = 32768;
        let pbm = path("bitmap");
        let e = write_bitmap(&pbm, bm_dim, 1, big_n, &[]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert!(
            !pbm.exists(),
            "write_bitmap created a file despite rejecting the payload"
        );

        let psb = path("sign_bitmap");
        let e = write_sign_bitmap(&psb, bm_dim, big_n, &[]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert!(
            !psb.exists(),
            "write_sign_bitmap created a file despite rejecting the payload"
        );

        // No-truncation: a rejected oversized write leaves an existing valid
        // file untouched (the cap check precedes File::create).
        let keep = path("rank_existing");
        {
            let mut idx = Rank::new(8);
            idx.add(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            idx.write(&keep).unwrap();
        }
        let before = std::fs::read(&keep).unwrap();
        let e = write_rank(&keep, big_dim, big_n, &[]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        let after = std::fs::read(&keep).unwrap();
        assert_eq!(
            before, after,
            "rejected oversized write altered an existing file"
        );
        let (_dim, n, _ranks) = load_rank(&keep).unwrap();
        assert_eq!(
            n, 1,
            "existing index no longer loads after a rejected write"
        );

        for p in [pr, prq, pbm, psb, keep] {
            let _ = std::fs::remove_file(p);
        }
    }
}
