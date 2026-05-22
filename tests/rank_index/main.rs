//! Integration tests for the rank-cosine index family.
//!
//! Three substrate types and their kernels:
//!
//! 1. Scalar correctness — each kernel agrees with a hand-written
//!    reference implementation on the same inputs (top-k indices
//!    match; scores agree within tolerance).
//! 2. Recall parity — on a synthetic corpus with planted nearest
//!    neighbours, the indices retrieve the planted neighbour at
//!    sub-percent error.
//! 3. Loader robustness — malformed serialisation files surface as
//!    `Err`, never panic.
//!
//! The file split mirrors `ordvec::rank_index` (`index.rs`,
//! `quant.rs`, `bitmap.rs`, `multi_bucket.rs`). Shared corpus +
//! reference helpers live here; loader fuzz lives here because it
//! crosses all three types in a single hermetic test.

use std::io::Write;

use ordvec::rank::{bucket_centre, bucket_ranks, rank_norm, rank_transform, rankquant_norm};
use ordvec::{BitmapIndex, RankIndex, RankQuantIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod bitmap;
mod fastscan;
mod index;
// `MultiBucketBitmapIndex` is gated behind the `experimental` feature.
#[cfg(feature = "experimental")]
mod multi_bucket;
mod quant;

pub const D: usize = 128;
pub const N: usize = 256;

pub fn make_corpus(seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut v = vec![0.0f32; N * D];
    for x in v.iter_mut() {
        *x = rng.gen_range(-1.0..1.0);
    }
    v
}

/// Reference scalar rank-cosine between two raw float vectors.
pub fn ref_rank_cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len();
    let ra = rank_transform(a);
    let rb = rank_transform(b);
    let mean = (d as f32 - 1.0) / 2.0;
    let mut acc = 0.0f32;
    for i in 0..d {
        acc += (ra[i] as f32 - mean) * (rb[i] as f32 - mean);
    }
    let norm = rank_norm(d);
    acc / (norm * norm)
}

/// Reference scalar asymmetric (float query vs rank doc).
pub fn ref_asymmetric(q: &[f32], doc: &[f32]) -> f32 {
    let d = q.len();
    let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q_unit: Vec<f32> = q.iter().map(|x| x / q_norm).collect();
    let r = rank_transform(doc);
    let mean = (d as f32 - 1.0) / 2.0;
    let norm = rank_norm(d);
    let mut acc = 0.0f32;
    for i in 0..d {
        acc += q_unit[i] * (r[i] as f32 - mean);
    }
    acc / norm
}

/// Reference scalar B-bit asymmetric (float query vs bucketed doc).
pub fn ref_rankquant_asymmetric(q: &[f32], doc: &[f32], bits: u8) -> f32 {
    let d = q.len();
    let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q_unit: Vec<f32> = q.iter().map(|x| x / q_norm).collect();
    let r = rank_transform(doc);
    let b = bucket_ranks(&r, bits);
    let norm = rankquant_norm(d, bits);
    let mut acc = 0.0f32;
    for i in 0..d {
        acc += q_unit[i] * bucket_centre(b[i], bits);
    }
    acc / norm
}

/// Loader robustness fuzz: feed malformed bytes (wrong magic, wrong
/// version, oversized dim/n_vectors, mismatched payload length,
/// invalid bits value, dim that violates constant-composition
/// invariants) and assert each loader returns `Err` rather than
/// panicking. Codex stop-time-review gate.
#[test]
fn rank_io_loaders_reject_malformed_files_without_panicking() {
    let tmp_dir = std::env::temp_dir();
    let make_file = |suffix: &str, bytes: &[u8]| -> std::path::PathBuf {
        let p = tmp_dir.join(format!(
            "rank_io_fuzz_{}_{}.bin",
            suffix,
            std::process::id()
        ));
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    };

    // Cases (suffix, bytes, which loaders should err) ----------
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("garbage_16", vec![0xAB; 16]),
        ("garbage_4k", vec![0xCC; 4096]),
        // Wrong magic for every type.
        (
            "wrong_magic",
            b"XXXX\x01".iter().chain([0u8; 8].iter()).copied().collect(),
        ),
        // TVR1 with overflowing dim/n_vectors.
        ("tvr_oversize", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVR1");
            v.push(1);
            v.extend_from_slice(&u32::MAX.to_le_bytes()); // dim
            v.extend_from_slice(&u32::MAX.to_le_bytes()); // n_vectors
            v
        }),
        // TVRQ with absurd bits value.
        ("tvrq_bad_bits", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVRQ");
            v.push(1);
            v.push(255); // bits
            v.extend_from_slice(&1024u32.to_le_bytes());
            v.extend_from_slice(&10u32.to_le_bytes());
            v
        }),
        // TVRQ with dim that violates 2^bits divisibility.
        ("tvrq_bad_dim", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVRQ");
            v.push(1);
            v.push(2); // bits=2 → n_buckets=4
            v.extend_from_slice(&13u32.to_le_bytes()); // dim=13 not /4
            v.extend_from_slice(&0u32.to_le_bytes());
            v
        }),
        // TVRQ with overflowing payload size.
        ("tvrq_oversize", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVRQ");
            v.push(1);
            v.push(4);
            v.extend_from_slice(&u32::MAX.to_le_bytes());
            v.extend_from_slice(&u32::MAX.to_le_bytes());
            v
        }),
        // TVBM with dim that isn't a multiple of 64.
        ("tvbm_dim_not_64", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVBM");
            v.push(1);
            v.extend_from_slice(&100u32.to_le_bytes()); // dim
            v.extend_from_slice(&25u32.to_le_bytes()); // n_top
            v.extend_from_slice(&5u32.to_le_bytes()); // n_vectors
            v
        }),
        // TVBM with n_top >= dim.
        ("tvbm_bad_n_top", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVBM");
            v.push(1);
            v.extend_from_slice(&128u32.to_le_bytes());
            v.extend_from_slice(&128u32.to_le_bytes()); // n_top == dim
            v.extend_from_slice(&5u32.to_le_bytes());
            v
        }),
        // TVBM with overflowing payload.
        ("tvbm_oversize", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVBM");
            v.push(1);
            v.extend_from_slice(&u32::MAX.to_le_bytes());
            v.extend_from_slice(&1u32.to_le_bytes());
            v.extend_from_slice(&u32::MAX.to_le_bytes());
            v
        }),
        // TVR1 with truncated payload (header claims a payload bigger
        // than what's on disk → read_exact returns UnexpectedEof, not
        // a panic).
        ("tvr_truncated", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVR1");
            v.push(1);
            v.extend_from_slice(&64u32.to_le_bytes()); // dim
            v.extend_from_slice(&100u32.to_le_bytes()); // n_vectors
                                                        // Header says 100 * 64 * 2 = 12800 payload bytes; provide
                                                        // only 100.
            v.extend(std::iter::repeat_n(0u8, 100));
            v
        }),
    ];

    let mut paths = Vec::new();
    for (label, bytes) in &cases {
        let p = make_file(label, bytes);
        // Each loader must return an error (any kind), not panic.
        // Use catch_unwind to enforce the no-panic contract: if any
        // loader panics on a malformed input, the test fails.
        let p1 = p.clone();
        let r1 = std::panic::catch_unwind(|| RankIndex::load(&p1));
        assert!(r1.is_ok(), "RankIndex::load panicked on {label}");
        assert!(r1.unwrap().is_err(), "RankIndex::load accepted {label}");

        let p2 = p.clone();
        let r2 = std::panic::catch_unwind(|| RankQuantIndex::load(&p2));
        assert!(r2.is_ok(), "RankQuantIndex::load panicked on {label}");
        assert!(
            r2.unwrap().is_err(),
            "RankQuantIndex::load accepted {label}"
        );

        let p3 = p.clone();
        let r3 = std::panic::catch_unwind(|| BitmapIndex::load(&p3));
        assert!(r3.is_ok(), "BitmapIndex::load panicked on {label}");
        assert!(r3.unwrap().is_err(), "BitmapIndex::load accepted {label}");

        paths.push(p);
    }

    // Cleanup.
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}
