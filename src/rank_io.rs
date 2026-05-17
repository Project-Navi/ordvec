//! Read/write rank-mode index files.
//!
//! Three formats live here, each self-describing via a 4-byte magic:
//! * `.tvr`  — [`RankIndex`](crate::RankIndex)        — magic `TVR1`
//! * `.tvrq` — [`RankQuantIndex`](crate::RankQuantIndex) — magic `TVRQ`
//! * `.tvbm` — [`BitmapIndex`](crate::BitmapIndex)    — magic `TVBM`
//!
//! All formats are little-endian. Headers are small fixed-size structs
//! followed by a single contiguous payload (the rank / packed / bitmap
//! bytes). No norms, no codebooks, no rotation matrices — these are the
//! deterministic-encode index types so the on-disk format is exactly the
//! in-memory buffer plus enough header to rehydrate the type parameters.
//!
//! The shape mirrors [`crate::io`] for `TurboQuantIndex`. ID-map wrappers
//! (analogous to `.tvim`) are an obvious follow-up but not in this v1.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const TVR_MAGIC: &[u8; 4] = b"TVR1";
const TVRQ_MAGIC: &[u8; 4] = b"TVRQ";
const TVBM_MAGIC: &[u8; 4] = b"TVBM";
const VERSION: u8 = 1;

// -------------------------------------------------------------------
// RankIndex: u16 ranks per coordinate.
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
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVR_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVR1 file: wrong magic",
        ));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported TVR1 version: {}", ver[0]),
        ));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    let mut bytes = vec![0u8; n_vectors * dim * 2];
    f.read_exact(&mut bytes)?;
    let ranks: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    Ok((dim, n_vectors, ranks))
}

// -------------------------------------------------------------------
// RankQuantIndex: B-bit packed bucket vectors.
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
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVRQ_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVRQ file: wrong magic",
        ));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported TVRQ version: {}", ver[0]),
        ));
    }
    let mut bits_buf = [0u8; 1];
    f.read_exact(&mut bits_buf)?;
    let bits = bits_buf[0];
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    let packed_bytes = n_vectors * dim * bits as usize / 8;
    let mut packed = vec![0u8; packed_bytes];
    f.read_exact(&mut packed)?;
    Ok((bits, dim, n_vectors, packed))
}

// -------------------------------------------------------------------
// BitmapIndex: top-n_top bitmap per document.
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

pub fn load_bitmap(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u64>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVBM_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVBM file: wrong magic",
        ));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported TVBM version: {}", ver[0]),
        ));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    let mut top_buf = [0u8; 4];
    f.read_exact(&mut top_buf)?;
    let n_top = u32::from_le_bytes(top_buf) as usize;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    let qpv = dim / 64;
    let mut bytes = vec![0u8; n_vectors * qpv * 8];
    f.read_exact(&mut bytes)?;
    let bitmaps: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();
    Ok((dim, n_top, n_vectors, bitmaps))
}
