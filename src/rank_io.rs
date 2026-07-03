//! Read/write ordinal/sign index files.
//!
//! Five formats live here, each self-describing via a 4-byte magic. Files
//! written by this crate use the **`.ov*` / `OV*`** magics (the ordvec format);
//! the legacy turbovec-era **`.tv*` / `TV*`** magics are still accepted on load
//! for backward compatibility, but are never written:
//! * `.ovr`  (legacy `.tvr`)  — [`Rank`](crate::Rank) — magic `OVR1` (also reads `TVR1`)
//! * `.ovrq` (legacy `.tvrq`) — [`RankQuant`](crate::RankQuant) — magic `OVRQ` (also reads `TVRQ`)
//! * `.ovbm` (legacy `.tvbm`) — [`Bitmap`](crate::Bitmap) — magic `OVBM` (also reads `TVBM`)
//! * `.ovsb` (legacy `.tvsb`) — [`SignBitmap`](crate::SignBitmap) — magic `OVSB` (also reads `TVSB`)
//! * `.ovfs` — [`RankQuantFastscan`](crate::RankQuantFastscan) — magic `OVFS`
//!   (new in the ordvec format; no legacy counterpart)
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
//! * `dim` and `n_vectors` are bounded by [`MAX_DIM`] (or
//!   [`MAX_SIGN_BITMAP_DIM`] for sign bitmaps) and [`MAX_VECTORS`].
//! * `bits` is checked against `{1, 2, 4}` before any multiplication.
//! * Total payload size is computed via [`usize::checked_mul`] and
//!   rejected if it overflows or exceeds the 128 GiB `MAX_PAYLOAD` cap.
//!   (`MAX_DIM * MAX_VECTORS * 2` bytes alone is ~8 TiB, so `MAX_PAYLOAD`
//!   is the binding byte ceiling, not the `dim` / `n_vectors` caps.)
//! * The declared payload must match the file's remaining bytes
//!   *exactly* — a structurally-valid file with trailing bytes is
//!   rejected (v1 formats have no footer or reserved trailing section).
//! * Per-index invariants (e.g., `dim % (1 << bits) == 0` for RankQuant)
//!   are returned as `Err(InvalidData)`, never `assert!`'d.
//! * FastScan `.ovfs` payloads are decoded far enough to reject invalid
//!   nibbles, non-canonical tail padding, and rows that violate b=2 constant
//!   composition before any search path can observe the bytes.
//!
//! Any malformed input returns `io::Error` rather than panicking.
//!
//! # Persistence API & round-trip contract
//!
//! The supported persistence API is the index types' path, stream, and byte
//! loaders/writers: `write(path)`, `write_to(writer)`, `load(path)`,
//! `read_from(reader)`, and `load_from_bytes(bytes)` on
//! [`Rank`](crate::Rank), [`RankQuant`](crate::RankQuant),
//! [`Bitmap`](crate::Bitmap), [`SignBitmap`](crate::SignBitmap), and
//! [`RankQuantFastscan`](crate::RankQuantFastscan) (the last via the `.ovfs`
//! format). `read_from` parses from the reader's current position through EOF
//! and rejects trailing bytes; callers embedding an index inside a larger
//! container should pass a length-bounded reader such as `Cursor<&[u8]>`.
//!
//! The `write_*` / `load_*` format helpers in this module are
//! **crate-internal** (`pub(crate)`); only the `MAX_*` capacity constants are
//! public.
//!
//! Round-trip is a guarantee of the **index types**: each constructor
//! validates its parameters (matching the loaders' `dim` / `n_top` / `bits` /
//! divisibility bounds), `add` caps `n_vectors` at [`MAX_VECTORS`], and the
//! types emit only loader-valid data — so anything `T::write` produces,
//! `T::load` reloads. The raw `write_*` helpers are trusted serializers for
//! the private in-memory buffers; they still enforce the same header, length,
//! and size-cap guards as the loaders (and `.ovfs` also revalidates its public
//! payload bytes) before `File::create`, so a rejected write never truncates an
//! existing file.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::format::{
    self, PersistedFormat, ProbeCoverage, OVBM_MAGIC, OVFS_MAGIC, OVRQ_MAGIC, OVR_MAGIC,
    OVSB_MAGIC, TVBM_MAGIC, TVRQ_MAGIC, TVR_MAGIC, TVSB_MAGIC,
};

const VERSION: u8 = 1;

/// Persisted index family identified from an on-disk ordvec index header.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexKind {
    Rank,
    RankQuant,
    Bitmap,
    SignBitmap,
    RankQuantFastscan,
}

/// Format-specific parameters declared by an on-disk ordvec index header.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexParams {
    Rank,
    RankQuant { bits: u8 },
    Bitmap { n_top: usize },
    SignBitmap,
    RankQuantFastscan { bits: u8 },
}

/// Header-derived metadata for a persisted ordvec index.
///
/// [`probe_index_metadata`] validates the fixed header, declared dimensions,
/// version, payload byte count, and exact file length, but deliberately does
/// not allocate or inspect the payload rows. Full row-invariant validation
/// remains the job of the index loaders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMetadata {
    pub kind: IndexKind,
    pub format_version: u8,
    pub dim: usize,
    pub vector_count: usize,
    pub bytes_per_vec: usize,
    pub params: IndexParams,
    pub file_size_bytes: u64,
}

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
    label: &str,
    file_len: u64,
    payload_bytes: usize,
) -> io::Result<()> {
    let pos = reader.stream_position()?;
    let remaining = file_len.saturating_sub(pos);
    let payload_bytes = payload_bytes as u64;
    if payload_bytes > remaining {
        return Err(invalid(format!(
            "{label} payload truncated: header declares {payload_bytes} B but file has {remaining} B remaining"
        )));
    }
    if payload_bytes < remaining {
        return Err(invalid(format!(
            "{label} payload has trailing bytes: header declares {payload_bytes} B but file has {remaining} B remaining"
        )));
    }
    Ok(())
}

fn stream_len_from_current<R: Seek>(reader: &mut R) -> io::Result<u64> {
    let start = reader.stream_position()?;
    let end = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(start))?;
    Ok(end)
}

fn check_dim(dim: usize) -> io::Result<()> {
    if !(2..=MAX_DIM).contains(&dim) {
        return Err(invalid(format!("dim {dim} out of range [2, {MAX_DIM}]")));
    }
    Ok(())
}

/// Dimension check for `.ovsb` sign-bitmap files.
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
            "OVSB dim {dim} out of range [64, {MAX_SIGN_BITMAP_DIM}]"
        )));
    }
    if !dim.is_multiple_of(64) {
        return Err(invalid(format!("OVSB dim {dim} is not a multiple of 64")));
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

fn truncated_field(label: &str, field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("{label} header truncated while reading {field}"),
    )
}

fn read_exact_field<R: Read, const N: usize>(
    reader: &mut R,
    label: &str,
    field: &str,
) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    reader
        .read_exact(&mut buf)
        .map_err(|err| match err.kind() {
            io::ErrorKind::UnexpectedEof => truncated_field(label, field),
            _ => err,
        })?;
    Ok(buf)
}

fn read_u8_field<R: Read>(reader: &mut R, label: &str, field: &str) -> io::Result<u8> {
    Ok(read_exact_field::<_, 1>(reader, label, field)?[0])
}

fn read_u32_le<R: Read>(reader: &mut R, label: &str, field: &str) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact_field::<_, 4>(
        reader, label, field,
    )?))
}

fn read_version<R: Read>(reader: &mut R, label: &str) -> io::Result<u8> {
    let ver = read_u8_field(reader, label, "version")?;
    if ver != VERSION {
        return Err(invalid(format!("unsupported {label} version: {ver}")));
    }
    Ok(ver)
}

fn read_magic<R: Read>(reader: &mut R, label: &str) -> io::Result<[u8; 4]> {
    read_exact_field(reader, label, "magic")
}

fn rank_payload_bytes(dim: usize, vector_count: usize) -> io::Result<usize> {
    vector_count
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| invalid("OVR1 payload size overflows usize"))
}

fn rankquant_bytes_per_vec(dim: usize, bits: u8) -> io::Result<usize> {
    dim.checked_mul(bits as usize)
        .map(|x| x / 8)
        .ok_or_else(|| invalid("OVRQ bytes_per_vec overflows usize"))
}

fn rankquant_payload_bytes(dim: usize, vector_count: usize, bits: u8) -> io::Result<usize> {
    let bytes_per_vec = rankquant_bytes_per_vec(dim, bits)?;
    vector_count
        .checked_mul(bytes_per_vec)
        .ok_or_else(|| invalid("OVRQ payload size overflows usize"))
}

fn bitmap_payload_bytes(dim: usize, vector_count: usize, label: &str) -> io::Result<usize> {
    let qpv = dim / 64;
    vector_count
        .checked_mul(qpv)
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| invalid(format!("{label} payload size overflows usize")))
}

/// Probe an ordvec index file's fixed header and declared byte shape.
///
/// This is the allocation-resistant metadata path used by external manifest
/// verification. It reads only the magic/version/parameter header plus file
/// metadata. It validates the same header domains as the full loaders and
/// requires the declared payload length to exactly match the remaining file
/// length, but it does not read or validate row payload invariants such as Rank
/// permutations, RankQuant constant composition, or Bitmap popcounts.
pub fn probe_index_metadata(path: impl AsRef<Path>) -> io::Result<IndexMetadata> {
    let file = File::open(path)?;
    let file_size_bytes = file.metadata()?.len();
    let mut f = BufReader::new(file);
    let magic = read_magic(&mut f, "ordvec index")?;
    let spec = format::lookup_magic(&magic).ok_or_else(|| invalid("unknown ordvec index magic"))?;
    match spec.format {
        PersistedFormat::Rank => probe_rank_metadata(&mut f, file_size_bytes),
        PersistedFormat::RankQuant => probe_rankquant_metadata(&mut f, file_size_bytes),
        PersistedFormat::Bitmap => probe_bitmap_metadata(&mut f, file_size_bytes),
        PersistedFormat::SignBitmap => probe_sign_bitmap_metadata(&mut f, file_size_bytes),
        PersistedFormat::RankQuantFastscan => match spec.probe {
            ProbeCoverage::Covered => unreachable!("FastScan probe is not wired yet"),
            ProbeCoverage::NotCovered { reason, .. } => Err(invalid(reason)),
        },
    }
}

fn probe_rank_metadata<R: Read + Seek>(
    reader: &mut R,
    file_size_bytes: u64,
) -> io::Result<IndexMetadata> {
    let format_version = read_version(reader, "OVR1")?;
    let dim = read_u32_le(reader, "OVR1", "dim")? as usize;
    check_dim(dim)?;
    let vector_count = read_u32_le(reader, "OVR1", "n_vectors")? as usize;
    check_n_vectors(vector_count)?;
    let bytes_per_vec = rank_payload_bytes(dim, 1)?;
    let payload_bytes = rank_payload_bytes(dim, vector_count)?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(reader, "OVR1", file_size_bytes, payload_bytes)?;
    Ok(IndexMetadata {
        kind: IndexKind::Rank,
        format_version,
        dim,
        vector_count,
        bytes_per_vec,
        params: IndexParams::Rank,
        file_size_bytes,
    })
}

fn probe_rankquant_metadata<R: Read + Seek>(
    reader: &mut R,
    file_size_bytes: u64,
) -> io::Result<IndexMetadata> {
    let format_version = read_version(reader, "OVRQ")?;
    let bits = read_u8_field(reader, "OVRQ", "bits")?;
    if !matches!(bits, 1 | 2 | 4) {
        return Err(invalid(format!(
            "unsupported OVRQ bits: {bits} (expected 1, 2, or 4)"
        )));
    }
    let dim = read_u32_le(reader, "OVRQ", "dim")? as usize;
    check_dim(dim)?;
    let n_buckets = 1usize << bits;
    if !dim.is_multiple_of(n_buckets) {
        return Err(invalid(format!(
            "OVRQ dim {dim} is not a multiple of 2^bits = {n_buckets}; \
             constant-composition invariant violated"
        )));
    }
    let codes_per_byte = (8 / bits) as usize;
    if !dim.is_multiple_of(codes_per_byte) {
        return Err(invalid(format!(
            "OVRQ dim {dim} is not a multiple of codes_per_byte = {codes_per_byte}"
        )));
    }
    let vector_count = read_u32_le(reader, "OVRQ", "n_vectors")? as usize;
    check_n_vectors(vector_count)?;
    let payload_bytes = rankquant_payload_bytes(dim, vector_count, bits)?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(reader, "OVRQ", file_size_bytes, payload_bytes)?;
    let bytes_per_vec = rankquant_bytes_per_vec(dim, bits)?;
    Ok(IndexMetadata {
        kind: IndexKind::RankQuant,
        format_version,
        dim,
        vector_count,
        bytes_per_vec,
        params: IndexParams::RankQuant { bits },
        file_size_bytes,
    })
}

fn probe_bitmap_metadata<R: Read + Seek>(
    reader: &mut R,
    file_size_bytes: u64,
) -> io::Result<IndexMetadata> {
    let format_version = read_version(reader, "OVBM")?;
    let dim = read_u32_le(reader, "OVBM", "dim")? as usize;
    check_dim(dim)?;
    if !dim.is_multiple_of(64) {
        return Err(invalid(format!("OVBM dim {dim} is not a multiple of 64")));
    }
    let n_top = read_u32_le(reader, "OVBM", "n_top")? as usize;
    if n_top == 0 || n_top >= dim {
        return Err(invalid(format!(
            "OVBM n_top {n_top} must satisfy 0 < n_top < dim ({dim})"
        )));
    }
    let vector_count = read_u32_le(reader, "OVBM", "n_vectors")? as usize;
    check_n_vectors(vector_count)?;
    let payload_bytes = bitmap_payload_bytes(dim, vector_count, "OVBM")?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(reader, "OVBM", file_size_bytes, payload_bytes)?;
    Ok(IndexMetadata {
        kind: IndexKind::Bitmap,
        format_version,
        dim,
        vector_count,
        bytes_per_vec: dim / 8,
        params: IndexParams::Bitmap { n_top },
        file_size_bytes,
    })
}

fn probe_sign_bitmap_metadata<R: Read + Seek>(
    reader: &mut R,
    file_size_bytes: u64,
) -> io::Result<IndexMetadata> {
    let format_version = read_version(reader, "OVSB")?;
    let dim = read_u32_le(reader, "OVSB", "dim")? as usize;
    check_sign_bitmap_dim(dim)?;
    let vector_count = read_u32_le(reader, "OVSB", "n_vectors")? as usize;
    check_n_vectors(vector_count)?;
    let payload_bytes = bitmap_payload_bytes(dim, vector_count, "OVSB")?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(reader, "OVSB", file_size_bytes, payload_bytes)?;
    Ok(IndexMetadata {
        kind: IndexKind::SignBitmap,
        format_version,
        dim,
        vector_count,
        bytes_per_vec: dim / 8,
        params: IndexParams::SignBitmap,
        file_size_bytes,
    })
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
    check_rank_write(dim, n_vectors, ranks)?;
    write_rank_to_checked(File::create(path)?, dim, n_vectors, ranks)
}

pub(crate) fn write_rank_to<W: Write>(
    writer: W,
    dim: usize,
    n_vectors: usize,
    ranks: &[u16],
) -> io::Result<()> {
    check_rank_write(dim, n_vectors, ranks)?;
    write_rank_to_checked(writer, dim, n_vectors, ranks)
}

fn check_rank_write(dim: usize, n_vectors: usize, ranks: &[u16]) -> io::Result<()> {
    let payload_bytes = rank_payload_bytes(dim, n_vectors)?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(ranks.len(), payload_bytes / 2);
    Ok(())
}

fn write_rank_to_checked<W: Write>(
    writer: W,
    dim: usize,
    n_vectors: usize,
    ranks: &[u16],
) -> io::Result<()> {
    let mut f = BufWriter::new(writer);
    f.write_all(OVR_MAGIC)?;
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
    load_rank_from_stream(&mut f, file_len)
}

pub(crate) fn load_rank_from<R: Read + Seek>(reader: R) -> io::Result<(usize, usize, Vec<u16>)> {
    let mut reader = reader;
    let file_len = stream_len_from_current(&mut reader)?;
    load_rank_from_stream(&mut reader, file_len)
}

fn load_rank_from_stream<R: Read + Seek>(
    mut f: &mut R,
    file_len: u64,
) -> io::Result<(usize, usize, Vec<u16>)> {
    let magic = read_magic(f, "OVR1")?;
    if &magic != OVR_MAGIC && &magic != TVR_MAGIC {
        return Err(invalid("not an OVR1/TVR1 (Rank) file: wrong magic"));
    }
    read_version(&mut f, "OVR1")?;
    let dim = read_u32_le(&mut f, "OVR1", "dim")? as usize;
    check_dim(dim)?;
    let n_vectors = read_u32_le(&mut f, "OVR1", "n_vectors")? as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = rank_payload_bytes(dim, n_vectors)?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, "OVR1", file_len, payload_bytes)?;
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
                    "OVR1 rank value {r} >= dim ({dim}); ranks must be a permutation of [0, dim)"
                )));
            }
            if seen[ri] == stamp {
                return Err(invalid(format!(
                    "OVR1 row {row_idx} is not a permutation of [0, dim): value {r} repeats"
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
    check_rankquant_write(bits, dim, n_vectors, packed)?;
    write_rankquant_to_checked(File::create(path)?, bits, dim, n_vectors, packed)
}

pub(crate) fn write_rankquant_to<W: Write>(
    writer: W,
    bits: u8,
    dim: usize,
    n_vectors: usize,
    packed: &[u8],
) -> io::Result<()> {
    check_rankquant_write(bits, dim, n_vectors, packed)?;
    write_rankquant_to_checked(writer, bits, dim, n_vectors, packed)
}

fn check_rankquant_write(bits: u8, dim: usize, n_vectors: usize, packed: &[u8]) -> io::Result<()> {
    let payload_bytes = rankquant_payload_bytes(dim, n_vectors, bits)?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(packed.len(), payload_bytes);
    Ok(())
}

fn write_rankquant_to_checked<W: Write>(
    writer: W,
    bits: u8,
    dim: usize,
    n_vectors: usize,
    packed: &[u8],
) -> io::Result<()> {
    let mut f = BufWriter::new(writer);
    f.write_all(OVRQ_MAGIC)?;
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
    load_rankquant_from_stream(&mut f, file_len)
}

pub(crate) fn load_rankquant_from<R: Read + Seek>(
    reader: R,
) -> io::Result<(u8, usize, usize, Vec<u8>)> {
    let mut reader = reader;
    let file_len = stream_len_from_current(&mut reader)?;
    load_rankquant_from_stream(&mut reader, file_len)
}

fn load_rankquant_from_stream<R: Read + Seek>(
    mut f: &mut R,
    file_len: u64,
) -> io::Result<(u8, usize, usize, Vec<u8>)> {
    let magic = read_magic(f, "OVRQ")?;
    if &magic != OVRQ_MAGIC && &magic != TVRQ_MAGIC {
        return Err(invalid("not an OVRQ/TVRQ (RankQuant) file: wrong magic"));
    }
    read_version(&mut f, "OVRQ")?;
    let bits = read_u8_field(&mut f, "OVRQ", "bits")?;
    if !matches!(bits, 1 | 2 | 4) {
        return Err(invalid(format!(
            "unsupported OVRQ bits: {bits} (expected 1, 2, or 4)"
        )));
    }
    let dim = read_u32_le(&mut f, "OVRQ", "dim")? as usize;
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
            "OVRQ dim {dim} is not a multiple of 2^bits = {n_buckets}; \
             constant-composition invariant violated"
        )));
    }
    let codes_per_byte = (8 / bits) as usize;
    if !dim.is_multiple_of(codes_per_byte) {
        return Err(invalid(format!(
            "OVRQ dim {dim} is not a multiple of codes_per_byte = {codes_per_byte}"
        )));
    }
    let n_vectors = read_u32_le(&mut f, "OVRQ", "n_vectors")? as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = rankquant_payload_bytes(dim, n_vectors, bits)?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, "OVRQ", file_len, payload_bytes)?;
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
    // Per-byte bucket-count LUT: byte value -> how many of its packed codes
    // land in each bucket. Replaces the per-code shift/mask loop (dim ops
    // per row) with bytes_per_row table lookups, and rows check in parallel
    // (they are independent). `find_first` preserves the serial contract of
    // reporting the lowest offending row.
    let mut lut = [[0u8; 16]; 256];
    for (byte, counts) in lut.iter_mut().enumerate() {
        for slot in 0..codes_per_byte {
            let shift = (codes_per_byte - 1 - slot) * bits_u;
            counts[((byte as u8 >> shift) & mask) as usize] += 1;
        }
    }
    let row_is_valid = |row: &[u8]| {
        let mut hist = [0u16; 16];
        for &byte in row {
            let counts = &lut[byte as usize];
            for bucket in 0..n_buckets {
                hist[bucket] += u16::from(counts[bucket]);
            }
        }
        hist[..n_buckets]
            .iter()
            .all(|&count| count as usize == expected_per_bucket)
    };
    use rayon::prelude::*;
    let first_bad = (0..n_vectors).into_par_iter().find_first(|&row_idx| {
        !row_is_valid(&packed[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row])
    });
    if let Some(row_idx) = first_bad {
        // Rerun the scalar histogram on the offending row for the exact
        // bucket/count in the error message.
        let row = &packed[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
        let mut hist = [0usize; 16];
        for &byte in row {
            for slot in 0..codes_per_byte {
                let shift = (codes_per_byte - 1 - slot) * bits_u;
                hist[((byte >> shift) & mask) as usize] += 1;
            }
        }
        for (bucket, &count) in hist[..n_buckets].iter().enumerate() {
            if count != expected_per_bucket {
                return Err(invalid(format!(
                    "OVRQ row {row_idx} violates constant composition: bucket {bucket} \
                     has {count} codes, expected {expected_per_bucket} (= dim / 2^bits)"
                )));
            }
        }
        unreachable!("row {row_idx} failed the LUT check but passed the scalar recheck");
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
    // Enforce the loaders' MAX_PAYLOAD cap *before* File::create (defense-in-
    // depth; a rejected write must not truncate an existing file). Mirrors
    // load_bitmap.
    check_bitmap_write(dim, n_vectors, bitmaps)?;
    write_bitmap_to_checked(File::create(path)?, dim, n_top, n_vectors, bitmaps)
}

pub(crate) fn write_bitmap_to<W: Write>(
    writer: W,
    dim: usize,
    n_top: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    check_bitmap_write(dim, n_vectors, bitmaps)?;
    write_bitmap_to_checked(writer, dim, n_top, n_vectors, bitmaps)
}

fn check_bitmap_write(dim: usize, n_vectors: usize, bitmaps: &[u64]) -> io::Result<()> {
    let payload_bytes = bitmap_payload_bytes(dim, n_vectors, "OVBM")?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(bitmaps.len(), payload_bytes / 8);
    Ok(())
}

fn write_bitmap_to_checked<W: Write>(
    writer: W,
    dim: usize,
    n_top: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let mut f = BufWriter::new(writer);
    f.write_all(OVBM_MAGIC)?;
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
    load_bitmap_from_stream(&mut f, file_len)
}

pub(crate) fn load_bitmap_from<R: Read + Seek>(
    reader: R,
) -> io::Result<(usize, usize, usize, Vec<u64>)> {
    let mut reader = reader;
    let file_len = stream_len_from_current(&mut reader)?;
    load_bitmap_from_stream(&mut reader, file_len)
}

fn load_bitmap_from_stream<R: Read + Seek>(
    mut f: &mut R,
    file_len: u64,
) -> io::Result<(usize, usize, usize, Vec<u64>)> {
    let magic = read_magic(f, "OVBM")?;
    if &magic != OVBM_MAGIC && &magic != TVBM_MAGIC {
        return Err(invalid("not an OVBM/TVBM (Bitmap) file: wrong magic"));
    }
    read_version(&mut f, "OVBM")?;
    let dim = read_u32_le(&mut f, "OVBM", "dim")? as usize;
    check_dim(dim)?;
    if !dim.is_multiple_of(64) {
        return Err(invalid(format!("OVBM dim {dim} is not a multiple of 64")));
    }
    let n_top = read_u32_le(&mut f, "OVBM", "n_top")? as usize;
    if n_top == 0 || n_top >= dim {
        return Err(invalid(format!(
            "OVBM n_top {n_top} must satisfy 0 < n_top < dim ({dim})"
        )));
    }
    let n_vectors = read_u32_le(&mut f, "OVBM", "n_vectors")? as usize;
    check_n_vectors(n_vectors)?;
    let qpv = dim / 64;
    let payload_bytes = bitmap_payload_bytes(dim, n_vectors, "OVBM")?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, "OVBM", file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * qpv * 8`, so the u64 element count is
    // `payload_bytes / 8`. Read directly into a fallibly reserved Vec<u64>
    // rather than allocating a byte buffer and `.collect()`-ing it.
    let bitmaps = read_le_vec(&mut f, payload_bytes / 8, u64::from_le_bytes)?;
    // Constant-composition invariant: every document bitmap must have exactly
    // `n_top` bits set (it flags the document's top `n_top` coordinates). The
    // idealized uniform constant-weight hypergeometric null model and the
    // documented `[0, n_top]` score range both assume this, so a forged row
    // with valid shape but a different popcount would break both. Verify
    // per-row popcount at the boundary.
    for (row_idx, row) in bitmaps.chunks_exact(qpv).enumerate() {
        let pop: u32 = row.iter().map(|w| w.count_ones()).sum();
        if pop as usize != n_top {
            return Err(invalid(format!(
                "OVBM row {row_idx} has {pop} bits set, expected n_top = {n_top}"
            )));
        }
    }
    Ok((dim, n_top, n_vectors, bitmaps))
}

/// Persist a [`crate::SignBitmap`] payload to a `.ovsb` file.
///
/// On-disk layout (little-endian throughout):
///
/// | offset | bytes | field                       |
/// |-------:|:-----:|-----------------------------|
/// | 0      | 4     | magic = `OVSB`              |
/// | 4      | 1     | version = 1                 |
/// | 5      | 4     | `dim` (u32)                 |
/// | 9      | 4     | `n_vectors` (u32)           |
/// | 13     | …     | `n_vectors * dim/64` u64s   |
///
/// 13-byte header — one u32 shorter than `OVBM` because SignBitmap
/// has no `n_top` parameter (the threshold is fixed at zero).
pub(crate) fn write_sign_bitmap(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    // Enforce the loaders' MAX_PAYLOAD cap *before* File::create (defense-in-
    // depth; a rejected write must not truncate an existing file). Mirrors
    // load_sign_bitmap.
    check_sign_bitmap_write(dim, n_vectors, bitmaps)?;
    write_sign_bitmap_to_checked(File::create(path)?, dim, n_vectors, bitmaps)
}

pub(crate) fn write_sign_bitmap_to<W: Write>(
    writer: W,
    dim: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    check_sign_bitmap_write(dim, n_vectors, bitmaps)?;
    write_sign_bitmap_to_checked(writer, dim, n_vectors, bitmaps)
}

fn check_sign_bitmap_write(dim: usize, n_vectors: usize, bitmaps: &[u64]) -> io::Result<()> {
    let payload_bytes = bitmap_payload_bytes(dim, n_vectors, "OVSB")?;
    check_payload_bytes(payload_bytes)?;
    assert_eq!(bitmaps.len(), payload_bytes / 8);
    Ok(())
}

fn write_sign_bitmap_to_checked<W: Write>(
    writer: W,
    dim: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let mut f = BufWriter::new(writer);
    f.write_all(OVSB_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    for &w in bitmaps {
        f.write_all(&w.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// Load a `.ovsb` file written by `write_sign_bitmap`.
///
/// Validates magic, version, dim (must be in
/// `[64, MAX_SIGN_BITMAP_DIM]` and a multiple of 64), and `n_vectors`
/// (≤ `MAX_VECTORS`). Payload size is computed with `checked_mul` and
/// rejected if it overflows or exceeds the 128 GiB hard cap from
/// `check_payload_bytes`. Malformed input returns `io::Error`; structurally
/// invalid fields use `InvalidData`, while truncated headers surface
/// `UnexpectedEof` with field context.
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
    load_sign_bitmap_from_stream(&mut f, file_len)
}

pub(crate) fn load_sign_bitmap_from<R: Read + Seek>(
    reader: R,
) -> io::Result<(usize, usize, Vec<u64>)> {
    let mut reader = reader;
    let file_len = stream_len_from_current(&mut reader)?;
    load_sign_bitmap_from_stream(&mut reader, file_len)
}

fn load_sign_bitmap_from_stream<R: Read + Seek>(
    mut f: &mut R,
    file_len: u64,
) -> io::Result<(usize, usize, Vec<u64>)> {
    let magic = read_magic(f, "OVSB")?;
    if &magic != OVSB_MAGIC && &magic != TVSB_MAGIC {
        return Err(invalid("not an OVSB/TVSB (SignBitmap) file: wrong magic"));
    }
    read_version(&mut f, "OVSB")?;
    let dim = read_u32_le(&mut f, "OVSB", "dim")? as usize;
    check_sign_bitmap_dim(dim)?;
    let n_vectors = read_u32_le(&mut f, "OVSB", "n_vectors")? as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = bitmap_payload_bytes(dim, n_vectors, "OVSB")?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, "OVSB", file_len, payload_bytes)?;
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

// -------------------------------------------------------------------
// RankQuantFastscan: b=2 block-32 FastScan layout.
// Header: magic(4) | version(1) | dim(u32 LE) | n_vectors(u32 LE) = 13 B
// Payload: n_blocks * (dim/2) * 32 bytes, n_blocks = ceil(n_vectors / 32).
// New ordvec format (no legacy TV* counterpart).
// -------------------------------------------------------------------

fn fastscan_payload_bytes(dim: usize, vector_count: usize) -> io::Result<usize> {
    // FastScan b=2 packs 32 docs per block; each block holds `pairs * 32` bytes
    // (`pairs = dim / 2`). `dim % 4 == 0` is enforced by the loader / constructor
    // before this is called, so `dim / 2` is exact. An empty corpus has zero
    // blocks and zero payload.
    let n_blocks = vector_count.div_ceil(32);
    let pairs = dim / 2;
    n_blocks
        .checked_mul(pairs)
        .and_then(|x| x.checked_mul(32))
        .ok_or_else(|| invalid("OVFS payload size overflows usize"))
}

fn validate_fastscan_payload(dim: usize, n_vectors: usize, packed_fs: &[u8]) -> io::Result<()> {
    if n_vectors == 0 {
        if packed_fs.is_empty() {
            return Ok(());
        }
        return Err(invalid(format!(
            "OVFS payload is {} bytes but empty index implies 0",
            packed_fs.len()
        )));
    }

    let pairs = dim / 2;
    let n_blocks = n_vectors.div_ceil(32);
    let bytes_per_block = pairs * 32;
    let expected_per_bucket = dim / 4;

    for block in 0..n_blocks {
        let doc_base = block * 32;
        let docs_in_block = (n_vectors - doc_base).min(32);
        let block_offset = block * bytes_per_block;

        for lane in 0..docs_in_block {
            let doc = doc_base + lane;
            let mut bucket_counts = [0usize; 4];
            for pair in 0..pairs {
                let offset = block_offset + pair * 32 + lane;
                let byte = packed_fs[offset];
                if byte & 0xf0 != 0 {
                    return Err(invalid(format!(
                        "OVFS payload byte at block {block}, pair {pair}, lane {lane} \
                         (document {doc}) has invalid FastScan nibble 0x{byte:02x}"
                    )));
                }
                bucket_counts[((byte >> 2) & 0x03) as usize] += 1;
                bucket_counts[(byte & 0x03) as usize] += 1;
            }
            if bucket_counts != [expected_per_bucket; 4] {
                return Err(invalid(format!(
                    "OVFS document {doc} violates b=2 constant composition: \
                     counts={bucket_counts:?}, expected {expected_per_bucket} per bucket"
                )));
            }
        }

        for lane in docs_in_block..32 {
            for pair in 0..pairs {
                let offset = block_offset + pair * 32 + lane;
                let byte = packed_fs[offset];
                if byte != 0 {
                    return Err(invalid(format!(
                        "OVFS tail padding byte at block {block}, pair {pair}, lane {lane} \
                         must be zero, got 0x{byte:02x}"
                    )));
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn write_fastscan(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    packed_fs: &[u8],
) -> io::Result<()> {
    check_fastscan_write(dim, n_vectors, packed_fs)?;
    write_fastscan_to_checked(File::create(path)?, dim, n_vectors, packed_fs)
}

pub(crate) fn write_fastscan_to<W: Write>(
    writer: W,
    dim: usize,
    n_vectors: usize,
    packed_fs: &[u8],
) -> io::Result<()> {
    check_fastscan_write(dim, n_vectors, packed_fs)?;
    write_fastscan_to_checked(writer, dim, n_vectors, packed_fs)
}

fn check_fastscan_write(dim: usize, n_vectors: usize, packed_fs: &[u8]) -> io::Result<()> {
    // Validate every header parameter *before* File::create, so a now-public
    // persistence API never (a) silently truncates `dim`/`n_vectors` through the
    // `as u32` casts below, (b) writes a corrupt/oversized file (the loaders'
    // MAX_PAYLOAD cap; a rejected write never truncates an existing file), or
    // (c) panics from a `Result`-returning fn. Mirrors load_fastscan's contract.
    check_dim(dim)?;
    if !dim.is_multiple_of(4) {
        return Err(invalid(format!(
            "OVFS dim {dim} is not a multiple of 4 (FastScan b=2 constant composition)"
        )));
    }
    check_n_vectors(n_vectors)?;
    let payload_bytes = fastscan_payload_bytes(dim, n_vectors)?;
    check_payload_bytes(payload_bytes)?;
    if packed_fs.len() != payload_bytes {
        return Err(invalid(format!(
            "OVFS packed buffer is {} bytes but dim={dim}/n_vectors={n_vectors} implies {payload_bytes}",
            packed_fs.len()
        )));
    }
    validate_fastscan_payload(dim, n_vectors, packed_fs)?;
    Ok(())
}

fn write_fastscan_to_checked<W: Write>(
    writer: W,
    dim: usize,
    n_vectors: usize,
    packed_fs: &[u8],
) -> io::Result<()> {
    let mut f = BufWriter::new(writer);
    f.write_all(OVFS_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    f.write_all(packed_fs)?;
    f.flush()?;
    Ok(())
}

pub(crate) fn load_fastscan(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<u8>)> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut f = BufReader::new(file);
    load_fastscan_from_stream(&mut f, file_len)
}

pub(crate) fn load_fastscan_from<R: Read + Seek>(reader: R) -> io::Result<(usize, usize, Vec<u8>)> {
    let mut reader = reader;
    let file_len = stream_len_from_current(&mut reader)?;
    load_fastscan_from_stream(&mut reader, file_len)
}

fn load_fastscan_from_stream<R: Read + Seek>(
    mut f: &mut R,
    file_len: u64,
) -> io::Result<(usize, usize, Vec<u8>)> {
    let magic = read_magic(f, "OVFS")?;
    // OVFS is new in the ordvec format: there is no legacy TV* fastscan magic.
    if &magic != OVFS_MAGIC {
        return Err(invalid("not an OVFS (RankQuantFastscan) file: wrong magic"));
    }
    read_version(&mut f, "OVFS")?;
    let dim = read_u32_le(&mut f, "OVFS", "dim")? as usize;
    check_dim(dim)?;
    // FastScan b=2 requires `dim % 4 == 0` (mirrors `RankQuantFastscan::new` /
    // `RankQuant::new(dim, 2)`: constant composition, exact analytical norm).
    // `dim % 4 == 0` subsumes the pair-encoding's `dim % 2 == 0`.
    if !dim.is_multiple_of(4) {
        return Err(invalid(format!(
            "OVFS dim {dim} is not a multiple of 4 (b=2 constant composition)"
        )));
    }
    let n_vectors = read_u32_le(&mut f, "OVFS", "n_vectors")? as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = fastscan_payload_bytes(dim, n_vectors)?;
    check_payload_bytes(payload_bytes)?;
    check_payload_matches_file(&mut f, "OVFS", file_len, payload_bytes)?;
    let mut packed_fs = try_alloc_zeroed(payload_bytes)?;
    f.read_exact(&mut packed_fs)?;
    validate_fastscan_payload(dim, n_vectors, &packed_fs)?;
    Ok((dim, n_vectors, packed_fs))
}

#[cfg(test)]
mod tests {
    use super::{
        load_bitmap, load_rank, load_rankquant, probe_index_metadata, write_bitmap, write_rank,
        write_rankquant, write_sign_bitmap, IndexKind, IndexParams, MAX_DIM, MAX_VECTORS, VERSION,
    };
    use crate::{Bitmap, Rank, RankQuant, SignBitmap};
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

    fn temp_index_path(suffix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!(
            "rank_io_probe_{}_{}_{}",
            std::process::id(),
            nonce,
            suffix
        ));
        p
    }

    fn assert_err_contains<T>(result: std::io::Result<T>, expected: &str) {
        let Err(err) = result else {
            panic!("expected error containing {expected:?}, got Ok(_)");
        };
        let text = err.to_string();
        assert!(
            text.contains(expected),
            "expected error containing {expected:?}, got {text:?}"
        );
    }

    fn rank_header(dim: u32, n_vectors: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVR1");
        v.push(VERSION);
        v.extend_from_slice(&dim.to_le_bytes());
        v.extend_from_slice(&n_vectors.to_le_bytes());
        v
    }

    fn rankquant_header(bits: u8, dim: u32, n_vectors: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVRQ");
        v.push(VERSION);
        v.push(bits);
        v.extend_from_slice(&dim.to_le_bytes());
        v.extend_from_slice(&n_vectors.to_le_bytes());
        v
    }

    fn bitmap_header(dim: u32, n_top: u32, n_vectors: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVBM");
        v.push(VERSION);
        v.extend_from_slice(&dim.to_le_bytes());
        v.extend_from_slice(&n_top.to_le_bytes());
        v.extend_from_slice(&n_vectors.to_le_bytes());
        v
    }

    fn sign_bitmap_header(dim: u32, n_vectors: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"TVSB");
        v.push(VERSION);
        v.extend_from_slice(&dim.to_le_bytes());
        v.extend_from_slice(&n_vectors.to_le_bytes());
        v
    }

    #[test]
    fn probe_metadata_matches_full_loaders_on_generated_fixtures() {
        let mut paths = Vec::new();

        let rank_path = temp_index_path("rank.tvr");
        let mut rank = Rank::new(8);
        rank.add(&[
            1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 6.0, 5.0, 8.0, 6.0, 7.0, 5.0, 1.0, 2.0, 3.0, 4.0,
        ]);
        rank.write(&rank_path).unwrap();
        let meta = probe_index_metadata(&rank_path).unwrap();
        let loaded = Rank::load(&rank_path).unwrap();
        assert_eq!(meta.kind, IndexKind::Rank);
        assert_eq!(meta.params, IndexParams::Rank);
        assert_eq!(meta.format_version, VERSION);
        assert_eq!(meta.dim, loaded.dim());
        assert_eq!(meta.vector_count, loaded.len());
        assert_eq!(meta.bytes_per_vec, loaded.bytes_per_vec());
        assert_eq!(
            meta.file_size_bytes,
            std::fs::metadata(&rank_path).unwrap().len()
        );
        paths.push(rank_path);

        let quant_path = temp_index_path("rankquant.tvrq");
        let mut quant = RankQuant::new(16, 2);
        let quant_docs: Vec<f32> = (0..32).map(|i| i as f32 - 11.0).collect();
        quant.add(&quant_docs);
        quant.write(&quant_path).unwrap();
        let meta = probe_index_metadata(&quant_path).unwrap();
        let loaded = RankQuant::load(&quant_path).unwrap();
        assert_eq!(meta.kind, IndexKind::RankQuant);
        assert_eq!(
            meta.params,
            IndexParams::RankQuant {
                bits: loaded.bits()
            }
        );
        assert_eq!(meta.format_version, VERSION);
        assert_eq!(meta.dim, loaded.dim());
        assert_eq!(meta.vector_count, loaded.len());
        assert_eq!(meta.bytes_per_vec, loaded.bytes_per_vec());
        assert_eq!(
            meta.file_size_bytes,
            std::fs::metadata(&quant_path).unwrap().len()
        );
        paths.push(quant_path);

        let bitmap_path = temp_index_path("bitmap.tvbm");
        let mut bitmap = Bitmap::new(64, 16);
        let bitmap_docs: Vec<f32> = (0..128).map(|i| ((i * 17) % 31) as f32).collect();
        bitmap.add(&bitmap_docs);
        bitmap.write(&bitmap_path).unwrap();
        let meta = probe_index_metadata(&bitmap_path).unwrap();
        let loaded = Bitmap::load(&bitmap_path).unwrap();
        assert_eq!(meta.kind, IndexKind::Bitmap);
        assert_eq!(
            meta.params,
            IndexParams::Bitmap {
                n_top: loaded.n_top()
            }
        );
        assert_eq!(meta.format_version, VERSION);
        assert_eq!(meta.dim, loaded.dim());
        assert_eq!(meta.vector_count, loaded.len());
        assert_eq!(meta.bytes_per_vec, loaded.bytes_per_vec());
        assert_eq!(
            meta.file_size_bytes,
            std::fs::metadata(&bitmap_path).unwrap().len()
        );
        paths.push(bitmap_path);

        let sign_path = temp_index_path("sign_bitmap.tvsb");
        let mut sign = SignBitmap::new(64);
        let sign_docs: Vec<f32> = (0usize..128)
            .map(|i| if i.is_multiple_of(3) { 1.0 } else { -1.0 })
            .collect();
        sign.add(&sign_docs);
        sign.write(&sign_path).unwrap();
        let meta = probe_index_metadata(&sign_path).unwrap();
        let loaded = SignBitmap::load(&sign_path).unwrap();
        assert_eq!(meta.kind, IndexKind::SignBitmap);
        assert_eq!(meta.params, IndexParams::SignBitmap);
        assert_eq!(meta.format_version, VERSION);
        assert_eq!(meta.dim, loaded.dim());
        assert_eq!(meta.vector_count, loaded.len());
        assert_eq!(meta.bytes_per_vec, loaded.bytes_per_vec());
        assert_eq!(
            meta.file_size_bytes,
            std::fs::metadata(&sign_path).unwrap().len()
        );
        paths.push(sign_path);

        for path in paths {
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn probe_rejects_header_and_length_errors_without_payload_allocation() {
        let wrong_magic = forge("wrong_magic", b"NOPE");
        let err = probe_index_metadata(&wrong_magic).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(&wrong_magic).ok();

        let bad_version = forge("bad_version", b"TVR1\x09");
        let err = probe_index_metadata(&bad_version).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(&bad_version).ok();

        let truncated = forge("truncated_header", b"TVR1\x01");
        let err = probe_index_metadata(&truncated).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        assert!(
            err.to_string()
                .contains("OVR1 header truncated while reading dim"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&truncated).ok();

        let length_mismatch = forge("length_mismatch", &rank_header(8, 1));
        assert_err_contains(
            probe_index_metadata(&length_mismatch),
            "OVR1 payload truncated",
        );
        std::fs::remove_file(&length_mismatch).ok();

        let mut huge_declared = Vec::new();
        huge_declared.extend_from_slice(b"TVR1");
        huge_declared.push(VERSION);
        huge_declared.extend_from_slice(&(MAX_DIM as u32).to_le_bytes());
        huge_declared.extend_from_slice(&(MAX_VECTORS as u32).to_le_bytes());
        let huge_declared = forge("huge_declared", &huge_declared);
        let err = probe_index_metadata(&huge_declared).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("MAX_PAYLOAD"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&huge_declared).ok();
    }

    #[test]
    fn probe_reports_header_field_context_for_truncated_headers() {
        let cases: [(&str, Vec<u8>, &str); 5] = [
            (
                "short_magic",
                b"TV".to_vec(),
                "ordvec index header truncated while reading magic",
            ),
            (
                "rank_version",
                b"TVR1".to_vec(),
                "OVR1 header truncated while reading version",
            ),
            (
                "rankquant_bits",
                b"TVRQ\x01".to_vec(),
                "OVRQ header truncated while reading bits",
            ),
            (
                "bitmap_n_top",
                {
                    let mut v = Vec::new();
                    v.extend_from_slice(b"TVBM");
                    v.push(VERSION);
                    v.extend_from_slice(&64u32.to_le_bytes());
                    v
                },
                "OVBM header truncated while reading n_top",
            ),
            (
                "sign_n_vectors",
                {
                    let mut v = Vec::new();
                    v.extend_from_slice(b"TVSB");
                    v.push(VERSION);
                    v.extend_from_slice(&64u32.to_le_bytes());
                    v
                },
                "OVSB header truncated while reading n_vectors",
            ),
        ];
        for (suffix, bytes, expected) in cases {
            let path = forge(suffix, &bytes);
            assert_err_contains(probe_index_metadata(&path), expected);
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn probe_reports_distinct_payload_truncation_and_trailing_bytes_for_all_formats() {
        let cases: [(&str, Vec<u8>, Vec<u8>, &str); 4] = [
            ("rank", rank_header(8, 1), rank_header(8, 0), "OVR1"),
            (
                "rankquant",
                rankquant_header(2, 8, 1),
                rankquant_header(2, 8, 0),
                "OVRQ",
            ),
            (
                "bitmap",
                bitmap_header(64, 16, 1),
                bitmap_header(64, 16, 0),
                "OVBM",
            ),
            (
                "sign_bitmap",
                sign_bitmap_header(64, 1),
                sign_bitmap_header(64, 0),
                "OVSB",
            ),
        ];

        for (suffix, truncated_header, mut trailing_bytes, label) in cases {
            let truncated = forge(&format!("{suffix}_truncated"), &truncated_header);
            assert_err_contains(
                probe_index_metadata(&truncated),
                &format!("{label} payload truncated"),
            );
            std::fs::remove_file(&truncated).ok();

            trailing_bytes.push(0);
            let trailing = forge(&format!("{suffix}_trailing"), &trailing_bytes);
            assert_err_contains(
                probe_index_metadata(&trailing),
                &format!("{label} payload has trailing bytes"),
            );
            std::fs::remove_file(&trailing).ok();
        }
    }

    #[test]
    fn probe_rejects_format_specific_header_errors() {
        let mut bad_bits = Vec::new();
        bad_bits.extend_from_slice(b"TVRQ");
        bad_bits.push(VERSION);
        bad_bits.push(3);
        bad_bits.extend_from_slice(&8u32.to_le_bytes());
        bad_bits.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("probe_bad_bits.tvrq", &bad_bits);
        assert_eq!(
            probe_index_metadata(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();

        let mut bad_rq_dim = Vec::new();
        bad_rq_dim.extend_from_slice(b"TVRQ");
        bad_rq_dim.push(VERSION);
        bad_rq_dim.push(4);
        bad_rq_dim.extend_from_slice(&8u32.to_le_bytes());
        bad_rq_dim.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("probe_bad_rq_dim.tvrq", &bad_rq_dim);
        assert_eq!(
            probe_index_metadata(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();

        let mut bad_bitmap_dim = Vec::new();
        bad_bitmap_dim.extend_from_slice(b"TVBM");
        bad_bitmap_dim.push(VERSION);
        bad_bitmap_dim.extend_from_slice(&100u32.to_le_bytes());
        bad_bitmap_dim.extend_from_slice(&10u32.to_le_bytes());
        bad_bitmap_dim.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("probe_bad_bitmap_dim.tvbm", &bad_bitmap_dim);
        assert_eq!(
            probe_index_metadata(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();

        let mut bad_n_top = Vec::new();
        bad_n_top.extend_from_slice(b"TVBM");
        bad_n_top.push(VERSION);
        bad_n_top.extend_from_slice(&64u32.to_le_bytes());
        bad_n_top.extend_from_slice(&64u32.to_le_bytes());
        bad_n_top.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("probe_bad_n_top.tvbm", &bad_n_top);
        assert_eq!(
            probe_index_metadata(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();

        let mut bad_sign_dim = Vec::new();
        bad_sign_dim.extend_from_slice(b"TVSB");
        bad_sign_dim.push(VERSION);
        bad_sign_dim.extend_from_slice(&32u32.to_le_bytes());
        bad_sign_dim.extend_from_slice(&0u32.to_le_bytes());
        let path = forge("probe_bad_sign_dim.tvsb", &bad_sign_dim);
        assert_eq!(
            probe_index_metadata(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn probe_does_not_validate_payload_row_invariants() {
        let mut forged = Vec::new();
        forged.extend_from_slice(b"TVBM");
        forged.push(VERSION);
        forged.extend_from_slice(&64u32.to_le_bytes());
        forged.extend_from_slice(&16u32.to_le_bytes());
        forged.extend_from_slice(&1u32.to_le_bytes());
        forged.extend_from_slice(&0u64.to_le_bytes());
        let path = forge("bad_bitmap_payload.tvbm", &forged);

        let meta = probe_index_metadata(&path).expect("probe reads only metadata");
        assert_eq!(meta.kind, IndexKind::Bitmap);
        assert_eq!(meta.dim, 64);
        assert_eq!(meta.vector_count, 1);

        let load_err = load_bitmap(&path).unwrap_err();
        assert_eq!(load_err.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
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

    // OVFS (FastScan) write path: valid round-trip, and fail-loud (io::Error, not
    // a panic) on invalid `dim`/`n_vectors`/payload — the now-public persistence
    // API must never abort the caller or silently truncate the header.
    #[test]
    fn write_fastscan_validates_and_never_panics() {
        use super::{load_fastscan, write_fastscan};
        // dim=8 (multiple of 4), 4 vectors -> ceil(4/32)*(8/2)*32 = 128-byte payload.
        let (dim, n) = (8usize, 4usize);
        let mut payload = vec![0u8; 128];
        for lane in 0..n {
            payload[lane] = 0x00;
            payload[32 + lane] = 0x05;
            payload[64 + lane] = 0x0a;
            payload[96 + lane] = 0x0f;
        }
        let p = temp_index_path("ovfs_ok");
        write_fastscan(&p, dim, n, &payload).unwrap();
        let (ld, ln, lbytes) = load_fastscan(&p).unwrap();
        assert_eq!((ld, ln), (dim, n));
        assert_eq!(lbytes, payload, "OVFS round-trip altered the payload");
        let _ = std::fs::remove_file(&p);

        // dim not a multiple of 4 -> rejected before File::create (no panic, no file).
        let p2 = temp_index_path("ovfs_baddim");
        let e = write_fastscan(&p2, 6, n, &payload).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(!p2.exists(), "rejected write must not create a file");

        // packed buffer inconsistent with dim/n_vectors -> rejected, not panic.
        let p3 = temp_index_path("ovfs_badlen");
        let e = write_fastscan(&p3, dim, n, &payload[..100]).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(!p3.exists(), "rejected write must not create a file");

        // A byte that is not a real FastScan nibble is rejected on write, before
        // a file can be created for the safe load/search APIs to observe.
        let p4 = temp_index_path("ovfs_badnibble");
        let mut invalid_payload = payload.clone();
        invalid_payload[32] = 0x10;
        let e = write_fastscan(&p4, dim, n, &invalid_payload).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            e.to_string().contains("invalid FastScan nibble"),
            "unexpected error: {e}"
        );
        assert!(!p4.exists(), "rejected write must not create a file");
    }

    // Probing a valid `.ovfs` file returns a specific, actionable error — NOT the
    // generic "unknown ordvec index magic" (which would be misleading, since the
    // magic *is* recognized). Metadata-probe support for OVFS is deferred to #232;
    // this pins the registry-backed deferral contract so it can't silently
    // regress to the generic case.
    #[test]
    fn probe_rejects_ovfs_with_specific_unsupported_error() {
        use super::{probe_index_metadata, write_fastscan};
        let (dim, n) = (8usize, 4usize);
        let mut payload = vec![0u8; 128];
        for lane in 0..n {
            payload[lane] = 0x00;
            payload[32 + lane] = 0x05;
            payload[64 + lane] = 0x0a;
            payload[96 + lane] = 0x0f;
        }
        let p = temp_index_path("ovfs_probe");
        write_fastscan(&p, dim, n, &payload).unwrap();
        let err = probe_index_metadata(&p);
        assert_err_contains(
            err,
            "OVFS (RankQuantFastscan) metadata probing is not supported",
        );
        // It must NOT be reported as an unknown magic.
        let again = probe_index_metadata(&p).unwrap_err().to_string();
        assert!(
            !again.contains("unknown ordvec index magic"),
            "OVFS is a recognized magic; probe must not report it as unknown, got {again:?}"
        );
        let _ = std::fs::remove_file(&p);
    }
}
