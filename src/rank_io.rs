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
//! * `dim` and `n_vectors` are bounded by [`MAX_DIM`] and [`MAX_VECTORS`]
//!   (chosen so a worst-case index fits in 128 GiB).
//! * `bits` is checked against `{1, 2, 4}` before any multiplication.
//! * Total payload size is computed via [`usize::checked_mul`] and
//!   rejected if it overflows.
//! * Per-index invariants (e.g., `dim % (1 << bits) == 0` for RankQuant)
//!   are returned as `Err(InvalidData)`, never `assert!`'d.
//!
//! Any malformed input returns `io::Error` rather than panicking.

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
/// Largest accepted `n_vectors` from a loaded file. 64 M docs at
/// `dim=u16::MAX` (128 KiB / vec for u16 ranks) tops out at ~8 TiB,
/// well past any sane on-disk index. Chosen to fail loud before
/// allocation panics.
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
/// [`check_payload_fits_file`]) run *before* this call; `read_le_vec`
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

/// Reject a declared payload that cannot fit in the file's remaining bytes.
///
/// `reader` is positioned just past the header; `file_len` is the file's
/// total length. A file cannot contain more payload than its own length,
/// so this catches a forged "tiny header claims gigabytes" before any
/// allocation — the primary defense, with [`try_alloc_zeroed`] as
/// defense-in-depth. `stream_position` gives the bytes already consumed
/// without manual offset accounting.
fn check_payload_fits_file<R: Seek>(
    reader: &mut R,
    file_len: u64,
    payload_bytes: usize,
) -> io::Result<()> {
    let pos = reader.stream_position()?;
    let remaining = file_len.saturating_sub(pos);
    if payload_bytes as u64 > remaining {
        return Err(invalid(
            "declared payload exceeds remaining file size (truncated or forged header)",
        ));
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
    // header even if dim and n_vectors individually pass.
    const MAX_PAYLOAD: usize = 128 * 1024 * 1024 * 1024;
    if payload_bytes > MAX_PAYLOAD {
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

pub fn write_rank(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    ranks: &[u16],
) -> io::Result<()> {
    assert_eq!(ranks.len(), n_vectors * dim);
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

pub fn load_rank(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<u16>)> {
    let file = File::open(path)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
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
    check_payload_fits_file(&mut f, file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * dim * 2`, so the u16 element count is
    // `payload_bytes / 2`. Read directly into a fallibly reserved Vec<u16>
    // instead of allocating a byte buffer and `.collect()`-ing it — the old
    // intermediate was an infallible (abort-on-OOM) allocation that also
    // doubled peak memory.
    let ranks = read_le_vec(&mut f, payload_bytes / 2, u16::from_le_bytes)?;
    // Every stored rank must be a valid coordinate index in `[0, dim)`.
    // An out-of-range value is not an OOB read here (it indexes a
    // per-query LUT sized to `dim` downstream) but silently corrupts the
    // Spearman score, so reject it at the loader boundary rather than
    // surfacing as a wrong-but-not-crashing result.
    if ranks.iter().any(|&r| (r as usize) >= dim) {
        return Err(invalid(format!(
            "TVR1 rank value >= dim ({dim}); ranks must be a permutation of [0, dim)"
        )));
    }
    Ok((dim, n_vectors, ranks))
}

// -------------------------------------------------------------------
// RankQuant: B-bit packed bucket vectors.
// Header: magic(4) | version(1) | bits(u8) | dim(u32 LE) | n_vectors(u32 LE) = 14 B
// Payload: n_vectors * dim * bits / 8 packed bytes.
// -------------------------------------------------------------------

pub fn write_rankquant(
    path: impl AsRef<Path>,
    bits: u8,
    dim: usize,
    n_vectors: usize,
    packed: &[u8],
) -> io::Result<()> {
    let expected = n_vectors * dim * bits as usize / 8;
    assert_eq!(packed.len(), expected);
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

pub fn load_rankquant(path: impl AsRef<Path>) -> io::Result<(u8, usize, usize, Vec<u8>)> {
    let file = File::open(path)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
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
    check_payload_fits_file(&mut f, file_len, payload_bytes)?;
    let mut packed = try_alloc_zeroed(payload_bytes)?;
    f.read_exact(&mut packed)?;
    Ok((bits, dim, n_vectors, packed))
}

// -------------------------------------------------------------------
// Bitmap: top-n_top bitmap per document.
// Header: magic(4) | version(1) | dim(u32 LE) | n_top(u32 LE) | n_vectors(u32 LE) = 17 B
// Payload: n_vectors * dim / 8 bytes (qwords as u64 LE).
// -------------------------------------------------------------------

pub fn write_bitmap(
    path: impl AsRef<Path>,
    dim: usize,
    n_top: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let qpv = dim / 64;
    assert_eq!(bitmaps.len(), n_vectors * qpv);
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

pub fn load_bitmap(path: impl AsRef<Path>) -> io::Result<(usize, usize, usize, Vec<u64>)> {
    let file = File::open(path)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
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
    check_payload_fits_file(&mut f, file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * qpv * 8`, so the u64 element count is
    // `payload_bytes / 8`. Read directly into a fallibly reserved Vec<u64>
    // rather than allocating a byte buffer and `.collect()`-ing it.
    let bitmaps = read_le_vec(&mut f, payload_bytes / 8, u64::from_le_bytes)?;
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
pub fn write_sign_bitmap(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let qpv = dim / 64;
    assert_eq!(bitmaps.len(), n_vectors * qpv);
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

/// Load a `.tvsb` file written by [`write_sign_bitmap`].
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
pub fn load_sign_bitmap(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<u64>)> {
    let file = File::open(path)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
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
    check_payload_fits_file(&mut f, file_len, payload_bytes)?;
    // `payload_bytes == n_vectors * qpv * 8`, so the u64 element count is
    // `payload_bytes / 8`. Read directly into a fallibly reserved Vec<u64>
    // rather than allocating a byte buffer and `.collect()`-ing it.
    let bitmaps = read_le_vec(&mut f, payload_bytes / 8, u64::from_le_bytes)?;
    Ok((dim, n_vectors, bitmaps))
}
