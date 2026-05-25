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
//! The file split mirrors `ordvec::index` (`rank.rs`,
//! `quant.rs`, `bitmap.rs`, `multi_bucket.rs`). Shared corpus +
//! reference helpers live here; loader fuzz lives here because it
//! crosses all four loader types (rank, rankquant, bitmap, sign
//! bitmap) in a single hermetic test.

use std::io::Write;

use ordvec::rank::{bucket_centre, bucket_ranks, rank_norm, rank_transform, rankquant_norm};
use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod bitmap;
mod fastscan;
mod rank;
// `MultiBucketBitmap` is gated behind the `experimental` feature.
mod finite;
mod loader_validation;
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
        // Wrong magic for every loader (TVR1 / TVRQ / TVBM / TVSB).
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
            // Header claims 100 * 64 * 2 = 12800 payload bytes but only 100
            // are provided, so the loader hits UnexpectedEof, not a panic.
            v.extend_from_slice(&64u32.to_le_bytes()); // dim
            v.extend_from_slice(&100u32.to_le_bytes()); // n_vectors
            v.extend(std::iter::repeat_n(0u8, 100));
            v
        }),
        // TVSB with dim that isn't a multiple of 64 (sign bitmaps pack
        // 64 coordinates per u64 qword, so the loader rejects it).
        ("tvsb_dim_not_64", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVSB");
            v.push(1);
            v.extend_from_slice(&100u32.to_le_bytes()); // dim, not /64
            v.extend_from_slice(&5u32.to_le_bytes()); // n_vectors
            v
        }),
        // TVSB with overflowing payload size.
        ("tvsb_oversize", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVSB");
            v.push(1);
            v.extend_from_slice(&u32::MAX.to_le_bytes()); // dim
            v.extend_from_slice(&u32::MAX.to_le_bytes()); // n_vectors
            v
        }),
        // TVSB with truncated payload: header declares 8 docs * 128/64 =
        // 16 qwords = 128 payload bytes but the file ends right after the
        // header, so read_exact yields UnexpectedEof rather than a panic.
        ("tvsb_truncated", {
            let mut v = Vec::new();
            v.extend_from_slice(b"TVSB");
            v.push(1);
            v.extend_from_slice(&128u32.to_le_bytes()); // dim (valid)
            v.extend_from_slice(&8u32.to_le_bytes()); // n_vectors
                                                      // No payload bytes provided.
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
        let r1 = std::panic::catch_unwind(|| Rank::load(&p1));
        assert!(r1.is_ok(), "Rank::load panicked on {label}");
        assert!(r1.unwrap().is_err(), "Rank::load accepted {label}");

        let p2 = p.clone();
        let r2 = std::panic::catch_unwind(|| RankQuant::load(&p2));
        assert!(r2.is_ok(), "RankQuant::load panicked on {label}");
        assert!(r2.unwrap().is_err(), "RankQuant::load accepted {label}");

        let p3 = p.clone();
        let r3 = std::panic::catch_unwind(|| Bitmap::load(&p3));
        assert!(r3.is_ok(), "Bitmap::load panicked on {label}");
        assert!(r3.unwrap().is_err(), "Bitmap::load accepted {label}");

        let p4 = p.clone();
        let r4 = std::panic::catch_unwind(|| SignBitmap::load(&p4));
        assert!(r4.is_ok(), "SignBitmap::load panicked on {label}");
        assert!(r4.unwrap().is_err(), "SignBitmap::load accepted {label}");

        paths.push(p);
    }

    // Cleanup.
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

/// Write-side payload guard (Codex stop-review): `write_*` must enforce the
/// same `MAX_PAYLOAD` (128 GiB) cap the loaders do, so the library never
/// emits a file it would refuse to load — and the check must run *before*
/// `File::create`, so a rejected oversized write cannot truncate an existing
/// valid file. Oversized dims are paired with empty payload slices: the cap
/// check fires before the length assert, so no terabyte allocation occurs
/// (and on 32-bit targets the size product overflows `usize` first — still a
/// clean `InvalidData` error, never a panic).
#[test]
fn rank_io_writers_reject_oversized_payload_without_truncating() {
    use std::io::ErrorKind;
    let tmp_dir = std::env::temp_dir();
    let path = |s: &str| {
        tmp_dir.join(format!(
            "rank_io_write_guard_{}_{}.bin",
            s,
            std::process::id()
        ))
    };

    // dim and n_vectors each individually pass the loaders' dim/n_vectors
    // caps, but their byte payload blows past MAX_PAYLOAD (128 GiB).
    let big_dim = u16::MAX as usize; // 65535 == MAX_DIM
    let big_n = 64 * 1024 * 1024; // == MAX_VECTORS; 65535 * 64Mi * 2 ≈ 8 TiB

    let pr = path("rank");
    let e = ordvec::rank_io::write_rank(&pr, big_dim, big_n, &[]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert!(
        !pr.exists(),
        "write_rank created a file despite rejecting the payload"
    );

    let prq = path("rankquant");
    let e = ordvec::rank_io::write_rankquant(&prq, 4, big_dim, big_n, &[]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert!(
        !prq.exists(),
        "write_rankquant created a file despite rejecting the payload"
    );

    // Bitmap/SignBitmap dims must be multiples of 64; 65536/64 = 1024
    // qwords/doc → 1024 * 8 * 64Mi = 512 GiB > 128 GiB.
    let bm_dim = 65536;
    let pbm = path("bitmap");
    let e = ordvec::rank_io::write_bitmap(&pbm, bm_dim, 1, big_n, &[]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert!(
        !pbm.exists(),
        "write_bitmap created a file despite rejecting the payload"
    );

    let psb = path("sign_bitmap");
    let e = ordvec::rank_io::write_sign_bitmap(&psb, bm_dim, big_n, &[]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert!(
        !psb.exists(),
        "write_sign_bitmap created a file despite rejecting the payload"
    );

    // No-truncation guarantee: a rejected oversized write must leave an
    // existing valid file at the same path untouched (the cap check precedes
    // File::create).
    let keep = path("rank_existing");
    {
        let mut idx = Rank::new(8);
        idx.add(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        idx.write(&keep).unwrap();
    }
    let before = std::fs::read(&keep).unwrap();
    let e = ordvec::rank_io::write_rank(&keep, big_dim, big_n, &[]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    let after = std::fs::read(&keep).unwrap();
    assert_eq!(
        before, after,
        "rejected oversized write altered an existing file"
    );
    let (_dim, n, _ranks) = ordvec::rank_io::load_rank(&keep).unwrap();
    assert_eq!(
        n, 1,
        "existing index no longer loads after a rejected write"
    );

    for p in [pr, prq, pbm, psb, keep] {
        let _ = std::fs::remove_file(p);
    }
}
