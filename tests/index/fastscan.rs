//! RankQuantFastscan (FastScan b=2 block-32 scan) integration
//! tests.
//!
//! FastScan is an optional b=2 scan path that wraps a FastScan-specific
//! kernel not shared with the other three index types, so its coverage
//! is repeated here rather than inherited. Two functional checks
//! (top-10 parity with the exact RankQuant b=2 kernel; k==0 short-
//! circuit) plus the audit-coverage suite (empty corpus, empty query,
//! k>N, thread-safe, dim-boundary matrix, second-add panic, metadata
//! roundtrip) carried over from the author's earlier rank-modes
//! development.

use std::io::Cursor;
use std::sync::Arc;
use std::thread;

use ordvec::{RankQuant, RankQuantFastscan};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{make_corpus, D, N};

#[test]
fn fastscan_b2_top10_matches_avx512_kernel() {
    // FastScan re-blocked layout must return the same top-10 indices
    // as the production scan_b2 asym kernel (modulo ties on the
    // boundary). Scores will differ by the LUT quantization error and
    // by the centre-drop residual, but the *ordering* of strong hits
    // should be preserved at 8-bit precision over D=128, N=100.
    const FD: usize = 128;
    const FN: usize = 100;
    let mut rng = ChaCha8Rng::seed_from_u64(31337);

    let docs: Vec<f32> = (0..FN * FD).map(|_| rng.random_range(-1.0..1.0)).collect();
    let queries: Vec<f32> = (0..3 * FD).map(|_| rng.random_range(-1.0..1.0)).collect();

    // Reference: the production RankQuant asym kernel.
    let mut idx = RankQuant::new(FD, 2);
    idx.add(&docs);
    let ref_res = idx.search_asymmetric(&queries, 10);

    // FastScan via the type wrapper (encapsulates rank-transform +
    // bucket + pack_fastscan_b2 + scan dispatch).
    let mut fs_idx = RankQuantFastscan::new(FD);
    fs_idx.add(&docs);
    let fs_res = fs_idx.search(&queries, 10);

    // Compare top-10 as sets per query. At 8-bit LUT precision the
    // intersection should be >= 9 (allow one boundary flip from
    // quantization).
    for q in 0..3 {
        let r_set: std::collections::HashSet<i64> = ref_res.indices[q * 10..(q + 1) * 10]
            .iter()
            .copied()
            .collect();
        let f_set: std::collections::HashSet<i64> = fs_res.indices[q * 10..(q + 1) * 10]
            .iter()
            .copied()
            .collect();
        let inter = r_set.intersection(&f_set).count();
        assert!(
            inter >= 9,
            "query {q}: FastScan top-10 differs by >1 from AVX-512 kernel: \
             ref={:?} fastscan={:?}",
            ref_res.indices[q * 10..(q + 1) * 10].to_vec(),
            fs_res.indices[q * 10..(q + 1) * 10].to_vec(),
        );
    }
}

#[test]
fn fastscan_handles_k_zero() {
    // par_chunks_mut(0) panics — FastScan's search must short-circuit
    // k == 0 to an empty-shape SearchResults instead of entering the
    // parallel scan (Codex stop-hook regression, source c4fd4d6).
    let corpus = make_corpus(250);
    let mut rng = ChaCha8Rng::seed_from_u64(251);
    let queries: Vec<f32> = (0..(2 * D)).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut fs = RankQuantFastscan::new(D);
    fs.add(&corpus);
    let r = fs.search(&queries, 0);
    assert_eq!(r.k, 0, "result.k must equal caller's k");
    assert!(r.scores.is_empty(), "scores must be empty for k=0");
    assert!(r.indices.is_empty(), "indices must be empty for k=0");
}

#[test]
fn fastscan_search_on_empty_corpus_returns_sentinel() {
    let fs = RankQuantFastscan::new(D);
    let q: Vec<f32> = vec![0.5; D];
    let r = fs.search(&q, 10);
    assert_eq!(r.nq, 1);
    // k is clamped to n_vectors (== 0) before sizing the buffers, so an
    // empty corpus collapses to k == 0 with empty result buffers —
    // matching the sibling search methods' empty-corpus behaviour.
    assert_eq!(r.k, 0, "empty corpus clamps k to n_vectors (0)");
    assert!(
        r.indices.iter().all(|&i| i == -1),
        "FastScan empty corpus: indices must all be sentinel -1"
    );
}

#[test]
fn fastscan_search_on_empty_query_returns_empty_results() {
    let corpus = make_corpus(260);
    let mut fs = RankQuantFastscan::new(D);
    fs.add(&corpus);
    let r = fs.search(&[], 10);
    assert_eq!(r.nq, 0);
    assert!(r.scores.is_empty());
    assert!(r.indices.is_empty());
}

#[test]
fn fastscan_handles_k_greater_than_n_vectors() {
    const N_SMALL: usize = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(261);
    let corpus: Vec<f32> = (0..(N_SMALL * D))
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let mut fs = RankQuantFastscan::new(D);
    fs.add(&corpus);
    let query: Vec<f32> = corpus[0..D].to_vec();
    let k = 20usize;
    let r = fs.search(&query, k);
    // k is clamped to n_vectors before sizing the result buffers, so a
    // k > N request returns exactly N_SMALL slots (all valid hits),
    // matching `RankQuant::search_asymmetric`'s clamp discipline.
    assert_eq!(r.k, N_SMALL, "k clamps to n_vectors when k > N");
    for slot in 0..N_SMALL {
        assert!(
            r.indices[slot] >= 0 && (r.indices[slot] as usize) < N_SMALL,
            "FastScan k>N: slot {slot} index {} invalid",
            r.indices[slot]
        );
    }
}

#[test]
fn fastscan_search_is_thread_safe() {
    let corpus = make_corpus(262);
    let mut rng = ChaCha8Rng::seed_from_u64(263);
    let queries: Vec<f32> = (0..(4 * D)).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut fs = RankQuantFastscan::new(D);
    fs.add(&corpus);
    let fs = Arc::new(fs);
    let ref_indices = fs.search(&queries, 10).indices;

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let fs = Arc::clone(&fs);
            let queries = queries.clone();
            let expected = ref_indices.clone();
            thread::spawn(move || {
                for _ in 0..8 {
                    assert_eq!(fs.search(&queries, 10).indices, expected);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn fastscan_dim_boundary_matrix() {
    // Smaller corpora than the RankQuant b=2 dim sweep — FastScan's
    // 2x storage and per-doc re-blocking make wide-dim tests expensive.
    // Covers a small dim and a mid-dim (1024 = Harrier production
    // target).
    for &dim in &[64usize, 1024] {
        const N_SMALL: usize = 16;
        let mut rng = ChaCha8Rng::seed_from_u64(270 + dim as u64);
        let corpus: Vec<f32> = (0..(N_SMALL * dim))
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut fs = RankQuantFastscan::new(dim);
        fs.add(&corpus);
        let q: Vec<f32> = corpus[0..dim].to_vec();
        let r = fs.search(&q, 5);
        assert_eq!(r.k, 5);
        for &idx in &r.indices {
            assert!(
                idx >= 0 && (idx as usize) < N_SMALL,
                "FastScan dim={dim}: invalid index {idx}"
            );
        }
        // Self-query: top-1 should be doc 0.
        assert_eq!(
            r.indices[0], 0,
            "FastScan dim={dim}: top-1 of self-query should be 0"
        );
    }
}

#[test]
#[should_panic(expected = "incremental add()")]
fn fastscan_second_add_panics_per_v1_contract() {
    // v1 limitation: the block-32 layout doesn't compose with
    // incremental extend. This test pins the panic contract so future
    // loosening of this limit doesn't silently corrupt the addressing
    // scheme.
    let corpus = make_corpus(280);
    let mut fs = RankQuantFastscan::new(D);
    fs.add(&corpus);
    fs.add(&corpus); // <- must panic with "incremental add()" in message
}

#[test]
fn fastscan_construct_then_metadata_roundtrips() {
    // Sanity-pin the type's read-only accessors after add().
    let corpus = make_corpus(281);
    let mut fs = RankQuantFastscan::new(D);
    assert!(fs.is_empty());
    assert_eq!(fs.len(), 0);
    assert_eq!(fs.dim(), D);
    assert_eq!(fs.bytes_per_vec(), D / 2);
    assert_eq!(fs.byte_size(), 0);

    fs.add(&corpus);
    assert!(!fs.is_empty());
    assert_eq!(fs.len(), N);
    assert_eq!(fs.dim(), D);
    assert_eq!(fs.bytes_per_vec(), D / 2);
    // byte_size includes per-block tail padding when N % 32 != 0;
    // here N = 256 = 8 * 32 so no padding overhead.
    assert_eq!(fs.byte_size(), N * (D / 2));
}

// ---------------------------------------------------------------------------
// Constructor domain: `RankQuantFastscan::new` must accept exactly the same
// `dim` domain as `RankQuant::new(dim, 2)` — `dim % 4 == 0` (b=2 constant
// composition) and `dim <= u16::MAX` (the u16 rank-transform invariant).
// Without the tighter guard, a too-loose `dim` constructs successfully but
// then either skews the analytical norm (dim % 4 != 0) or panics on the
// first `add()` inside `rank_transform` (dim > u16::MAX) — a latent bug the
// constructor should reject up front.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "divisible by 4")]
fn fastscan_new_rejects_dim_2_not_multiple_of_4() {
    // dim = 2 passes the old `dim % 2 == 0` guard but violates b=2's
    // constant-composition (4 buckets can't each hold dim/4 = 0.5 ranks).
    let _ = RankQuantFastscan::new(2);
}

#[test]
#[should_panic(expected = "divisible by 4")]
fn fastscan_new_rejects_dim_6_not_multiple_of_4() {
    // dim = 6 is even but 6 % 4 == 2: buckets would be [2, 2, 1, 1], skewing
    // the analytical rankquant_norm. `RankQuant::new(6, 2)` rejects it too.
    let _ = RankQuantFastscan::new(6);
}

#[test]
#[should_panic(expected = "fit in u16")]
fn fastscan_new_rejects_dim_above_u16_max() {
    // 65_536 satisfies `% 4 == 0` but exceeds u16::MAX, so it must be caught
    // by the u16 bound — not deferred to a panic on the first add().
    let _ = RankQuantFastscan::new(65_536);
}

// ---------------------------------------------------------------------
// Persistence: `.ovfs` (magic `OVFS`) write/load round-trip + validation.
// ---------------------------------------------------------------------

fn fs_tmp(name: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ordvec_fastscan_{}_{}_{}.ovfs",
        name,
        std::process::id(),
        nonce
    ))
}

fn forge_ovfs(dim: usize, n_vectors: usize, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"OVFS");
    bytes.push(1);
    bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_vectors as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn valid_dim4_n1_payload() -> Vec<u8> {
    let mut payload = vec![0u8; 64];
    // Buckets [0, 1, 2, 3] -> one coordinate in each b=2 bucket.
    payload[0] = 0x01;
    payload[32] = 0x0b;
    payload
}

fn assert_fastscan_loaders_reject(bytes: &[u8], expected: &str) {
    let path = fs_tmp("malformed_payload");
    std::fs::write(&path, bytes).unwrap();
    let path_err = RankQuantFastscan::load(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(
        path_err.to_string().contains(expected),
        "path loader returned unexpected error: {path_err}"
    );

    let reader_err = RankQuantFastscan::read_from(Cursor::new(bytes.to_vec())).unwrap_err();
    assert!(
        reader_err.to_string().contains(expected),
        "reader loader returned unexpected error: {reader_err}"
    );

    let bytes_err = RankQuantFastscan::load_from_bytes(bytes).unwrap_err();
    assert!(
        bytes_err.to_string().contains(expected),
        "byte-slice loader returned unexpected error: {bytes_err}"
    );
}

#[test]
fn fastscan_write_load_roundtrip_searches_identically() {
    const FD: usize = 128;
    const FN: usize = 200;
    let mut rng = ChaCha8Rng::seed_from_u64(909090);
    let docs: Vec<f32> = (0..FN * FD).map(|_| rng.random_range(-1.0..1.0)).collect();
    let queries: Vec<f32> = (0..4 * FD).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut idx = RankQuantFastscan::new(FD);
    idx.add(&docs);
    let before = idx.search(&queries, 10);

    let path = fs_tmp("roundtrip");
    idx.write(&path).unwrap();
    let loaded = RankQuantFastscan::load(&path).unwrap();
    std::fs::remove_file(&path).ok();

    // Reloaded index reports the same shape and scans byte-identically: the
    // packed buffer is the same, so scores/indices match exactly (no recompute).
    assert_eq!(loaded.dim(), FD);
    assert_eq!(loaded.len(), FN);
    assert_eq!(loaded.byte_size(), idx.byte_size());
    let after = loaded.search(&queries, 10);
    assert_eq!(after.indices, before.indices, "reloaded indices must match");
    assert_eq!(after.scores, before.scores, "reloaded scores must match");
}

#[test]
fn fastscan_stream_persistence_roundtrips() {
    const FD: usize = 128;
    const FN: usize = 96;
    const PREFIX: &[u8] = b"container-prefix";

    let mut rng = ChaCha8Rng::seed_from_u64(909091);
    let docs: Vec<f32> = (0..FN * FD).map(|_| rng.random_range(-1.0..1.0)).collect();
    let queries: Vec<f32> = (0..2 * FD).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut idx = RankQuantFastscan::new(FD);
    idx.add(&docs);
    let before = idx.search(&queries, 10);

    let mut bytes = Vec::new();
    idx.write_to(&mut bytes).unwrap();
    assert_eq!(&bytes[..4], b"OVFS");

    let path = fs_tmp("stream_bytes");
    idx.write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    std::fs::remove_file(&path).ok();

    let from_bytes = RankQuantFastscan::load_from_bytes(&bytes).unwrap();
    let mut prefixed = PREFIX.to_vec();
    prefixed.extend_from_slice(&bytes);
    let mut cursor = std::io::Cursor::new(prefixed);
    cursor.set_position(PREFIX.len() as u64);
    let from_reader = RankQuantFastscan::read_from(cursor).unwrap();

    for loaded in [from_bytes, from_reader] {
        assert_eq!(loaded.dim(), FD);
        assert_eq!(loaded.len(), FN);
        assert_eq!(loaded.byte_size(), idx.byte_size());
        let after = loaded.search(&queries, 10);
        assert_eq!(after.indices, before.indices);
        assert_eq!(after.scores, before.scores);
    }
}

#[test]
fn fastscan_reader_does_not_buffer_past_reported_trailing_bytes() {
    const FD: usize = 128;
    const FN: usize = 96;

    let mut rng = ChaCha8Rng::seed_from_u64(909092);
    let docs: Vec<f32> = (0..FN * FD).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut idx = RankQuantFastscan::new(FD);
    idx.add(&docs);

    let mut bytes = Vec::new();
    idx.write_to(&mut bytes).unwrap();
    bytes.extend_from_slice(b"next-record");

    let mut cursor = Cursor::new(bytes);
    let Err(err) = RankQuantFastscan::read_from(&mut cursor) else {
        panic!("FastScan reader accepted trailing bytes");
    };
    assert!(
        err.to_string().contains("OVFS payload has trailing bytes"),
        "unexpected error: {err}"
    );
    assert_eq!(
        cursor.position(),
        13,
        "FastScan reader should stop after header"
    );
}

#[test]
fn fastscan_empty_index_roundtrips() {
    let idx = RankQuantFastscan::new(64); // never add()-ed → 0 vectors, empty payload
    let path = fs_tmp("empty");
    idx.write(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let loaded = RankQuantFastscan::load(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(bytes.len(), 13, "empty .ovfs is header-only (no payload)");
    assert_eq!(&bytes[0..4], b"OVFS", "magic is OVFS");
    assert_eq!(loaded.dim(), 64);
    assert_eq!(loaded.len(), 0);
    assert!(loaded.is_empty());
}

#[test]
fn fastscan_written_file_starts_with_ovfs_magic() {
    let mut idx = RankQuantFastscan::new(64);
    idx.add(&vec![0.5f32; 64 * 40]);
    let path = fs_tmp("magic");
    idx.write(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(&bytes[0..4], b"OVFS");
}

#[test]
fn fastscan_load_rejects_wrong_magic() {
    let mut idx = RankQuantFastscan::new(64);
    idx.add(&vec![0.25f32; 64 * 40]);
    let path = fs_tmp("badmagic");
    idx.write(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0..4].copy_from_slice(b"OVRQ"); // a different (valid) ordvec magic
    std::fs::write(&path, &bytes).unwrap();
    let err = match RankQuantFastscan::load(&path) {
        Ok(_) => panic!("expected load error, got Ok"),
        Err(e) => e,
    };
    std::fs::remove_file(&path).ok();
    assert!(err.to_string().contains("OVFS"), "got: {err}");
}

#[test]
fn fastscan_load_rejects_trailing_bytes() {
    let mut idx = RankQuantFastscan::new(64);
    idx.add(&vec![-0.3f32; 64 * 40]);
    let path = fs_tmp("trailing");
    idx.write(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(0xAB); // one trailing byte past the declared payload
    std::fs::write(&path, &bytes).unwrap();
    let err = match RankQuantFastscan::load(&path) {
        Ok(_) => panic!("expected load error, got Ok"),
        Err(e) => e,
    };
    std::fs::remove_file(&path).ok();
    // A structurally-valid file with trailing bytes is rejected.
    assert!(!err.to_string().is_empty());
}

#[test]
fn fastscan_load_rejects_dim_not_multiple_of_4() {
    // Forge a header with dim = 66 (even but % 4 == 2) and zero payload.
    let path = fs_tmp("baddim");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"OVFS");
    bytes.push(1); // version
    bytes.extend_from_slice(&66u32.to_le_bytes()); // dim = 66
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_vectors = 0
    std::fs::write(&path, &bytes).unwrap();
    let err = match RankQuantFastscan::load(&path) {
        Ok(_) => panic!("expected load error, got Ok"),
        Err(e) => e,
    };
    std::fs::remove_file(&path).ok();
    assert!(err.to_string().contains("multiple of 4"), "got: {err}");
}

#[test]
fn fastscan_load_rejects_invalid_payload_nibble_on_all_public_loaders() {
    let mut payload = valid_dim4_n1_payload();
    payload[32] = 0x10;
    let bytes = forge_ovfs(4, 1, &payload);
    assert_fastscan_loaders_reject(&bytes, "invalid FastScan nibble");
}

#[test]
fn fastscan_load_rejects_nonzero_tail_padding_on_all_public_loaders() {
    let mut payload = valid_dim4_n1_payload();
    payload[1] = 0x01;
    let bytes = forge_ovfs(4, 1, &payload);
    assert_fastscan_loaders_reject(&bytes, "tail padding byte");
}

#[test]
fn fastscan_load_rejects_constant_composition_violation_on_all_public_loaders() {
    let mut payload = valid_dim4_n1_payload();
    payload[0] = 0x00;
    let bytes = forge_ovfs(4, 1, &payload);
    assert_fastscan_loaders_reject(&bytes, "constant composition");
}
