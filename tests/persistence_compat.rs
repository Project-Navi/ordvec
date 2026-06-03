//! Persisted byte-format compatibility fixtures.
//!
//! These are deliberately tiny committed byte expectations, not round trips
//! that only prove the current writer can feed the current loader.

use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ordvec::{probe_index_metadata, Bitmap, IndexKind, IndexParams, Rank, RankQuant, SignBitmap};

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
}

impl AsRef<Path> for TempFile {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Deref for TempFile {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn tmp(name: &str) -> TempFile {
    let nonce = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ordvec_persistence_compat_{}_{}_{}.bin",
        name,
        std::process::id(),
        nonce
    ));
    TempFile { path }
}

fn write_bytes(path: &Path, bytes: &[u8]) {
    std::fs::File::create(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}

fn assert_metadata(
    path: &Path,
    kind: IndexKind,
    dim: usize,
    vector_count: usize,
    bytes_per_vec: usize,
    params: IndexParams,
    file_size_bytes: u64,
) {
    let meta = probe_index_metadata(path).unwrap();
    assert_eq!(meta.kind, kind);
    assert_eq!(meta.format_version, 1);
    assert_eq!(meta.dim, dim);
    assert_eq!(meta.vector_count, vector_count);
    assert_eq!(meta.bytes_per_vec, bytes_per_vec);
    assert_eq!(meta.params, params);
    assert_eq!(meta.file_size_bytes, file_size_bytes);
}

fn assert_rejects_version_and_trailing_bytes<T>(
    name: &str,
    expected: &[u8],
    load: impl Fn(&Path) -> std::io::Result<T>,
) {
    assert!(
        expected.len() > 4,
        "expected fixture must include a version byte at index 4"
    );
    let path = tmp(name);

    let mut unsupported_version = expected.to_vec();
    unsupported_version[4] = 2;
    write_bytes(&path, &unsupported_version);
    assert!(probe_index_metadata(&path).is_err());
    assert!(load(&path).is_err());

    let mut trailing = expected.to_vec();
    trailing.push(0);
    write_bytes(&path, &trailing);
    assert!(probe_index_metadata(&path).is_err());
    assert!(load(&path).is_err());
}

#[test]
fn rank_v1_fixture_bytes_are_stable() {
    let expected = [
        b'T', b'V', b'R', b'1', 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0,
    ];
    let path = tmp("rank");

    let mut index = Rank::new(4);
    index.add(&[0.0, 1.0, 2.0, 3.0]);
    index.write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), expected);
    assert_metadata(
        &path,
        IndexKind::Rank,
        4,
        1,
        8,
        IndexParams::Rank,
        expected.len() as u64,
    );

    write_bytes(&path, &expected);
    let loaded = Rank::load(&path).unwrap();
    assert_eq!(loaded.dim(), 4);
    assert_eq!(loaded.len(), 1);

    assert_rejects_version_and_trailing_bytes("rank", &expected, |path| Rank::load(path));
}

#[test]
fn rankquant_v1_fixture_bytes_are_stable() {
    let expected = [
        b'T', b'V', b'R', b'Q', 1, 2, 8, 0, 0, 0, 1, 0, 0, 0, 0x05, 0xaf,
    ];
    let path = tmp("rankquant");

    let mut index = RankQuant::new(8, 2);
    index.add(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    index.write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), expected);
    assert_metadata(
        &path,
        IndexKind::RankQuant,
        8,
        1,
        2,
        IndexParams::RankQuant { bits: 2 },
        expected.len() as u64,
    );

    write_bytes(&path, &expected);
    let loaded = RankQuant::load(&path).unwrap();
    assert_eq!(loaded.dim(), 8);
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.bits(), 2);

    assert_rejects_version_and_trailing_bytes("rankquant", &expected, |path| RankQuant::load(path));
}

#[test]
fn bitmap_v1_fixture_bytes_are_stable() {
    let expected = [
        b'T', b'V', b'B', b'M', 1, 64, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xc0,
    ];
    let path = tmp("bitmap");

    let mut index = Bitmap::new(64, 2);
    let vector: Vec<f32> = (0..64).map(|value| value as f32).collect();
    index.add(&vector);
    index.write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), expected);
    assert_metadata(
        &path,
        IndexKind::Bitmap,
        64,
        1,
        8,
        IndexParams::Bitmap { n_top: 2 },
        expected.len() as u64,
    );

    write_bytes(&path, &expected);
    let loaded = Bitmap::load(&path).unwrap();
    assert_eq!(loaded.dim(), 64);
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.n_top(), 2);

    assert_rejects_version_and_trailing_bytes("bitmap", &expected, |path| Bitmap::load(path));
}

#[test]
fn sign_bitmap_v1_fixture_bytes_are_stable() {
    let expected = [
        b'T', b'V', b'S', b'B', 1, 64, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0x80,
    ];
    let path = tmp("sign_bitmap");

    let mut index = SignBitmap::new(64);
    let mut vector = vec![-1.0; 64];
    vector[0] = 1.0;
    vector[63] = 1.0;
    index.add(&vector);
    index.write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), expected);
    assert_metadata(
        &path,
        IndexKind::SignBitmap,
        64,
        1,
        8,
        IndexParams::SignBitmap,
        expected.len() as u64,
    );

    write_bytes(&path, &expected);
    let loaded = SignBitmap::load(&path).unwrap();
    assert_eq!(loaded.dim(), 64);
    assert_eq!(loaded.len(), 1);

    assert_rejects_version_and_trailing_bytes("sign_bitmap", &expected, |path| {
        SignBitmap::load(path)
    });
}
