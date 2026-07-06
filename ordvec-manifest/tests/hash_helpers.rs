use std::io::{Cursor, ErrorKind};

use ordvec_manifest::{sha256_bytes, sha256_file, sha256_file_bounded, sha256_reader};

const CONTENT: &[u8] = b"ordinal geometry is rank, not distance";

#[test]
fn sha256_helpers_agree_on_identical_content() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("content.bin");
    std::fs::write(&path, CONTENT).expect("write content");

    let from_file = sha256_file(&path).expect("sha256_file");
    let from_bytes = sha256_bytes(CONTENT);
    let from_reader =
        sha256_reader(Cursor::new(CONTENT), CONTENT.len() as u64).expect("sha256_reader");
    let from_bounded_file = sha256_file_bounded(
        &path,
        CONTENT.len() as u64,
        "artifact_file_too_large",
        "test artifact",
    )
    .expect("sha256_file_bounded");

    assert_eq!(from_file.sha256, from_bytes.sha256);
    assert_eq!(from_file.sha256, from_reader.sha256);
    assert_eq!(from_file.sha256, from_bounded_file.sha256);
    assert_eq!(from_file.size_bytes, CONTENT.len() as u64);
    assert_eq!(from_bytes.size_bytes, CONTENT.len() as u64);
    assert_eq!(from_reader.size_bytes, CONTENT.len() as u64);
}

#[test]
fn sha256_reader_accepts_input_of_exactly_max_bytes() {
    let hash = sha256_reader(Cursor::new(CONTENT), CONTENT.len() as u64)
        .expect("input at the bound must hash");
    assert_eq!(hash.size_bytes, CONTENT.len() as u64);
}

#[test]
fn sha256_reader_rejects_input_larger_than_max_bytes() {
    let err = sha256_reader(Cursor::new(CONTENT), CONTENT.len() as u64 - 1)
        .expect_err("input past the bound must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("exceeds"));
}

#[test]
fn sha256_reader_handles_empty_input() {
    let hash = sha256_reader(Cursor::new(&[][..]), 0).expect("empty input hashes under a 0 bound");
    assert_eq!(hash.size_bytes, 0);
    assert_eq!(hash.sha256, sha256_bytes(&[]).sha256);
}
