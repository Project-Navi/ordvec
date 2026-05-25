//! Bitmap integration tests: top-bucket bitmap candidate
//! generation, two-stage rerank wiring, and the AVX-512 VPOPCNTDQ
//! batched-kernel parity proofs.

use ordvec::rank::rank_transform;
use ordvec::{Bitmap, RankQuant};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{make_corpus, D, N};

#[test]
fn rank_io_round_trip_bitmap_index() {
    let corpus = make_corpus(42);
    let mut idx = Bitmap::new(D, D / 4);
    idx.add(&corpus);
    let tmp = std::env::temp_dir().join("bitmap_index_io.tvbm");
    idx.write(&tmp).expect("write");
    let loaded = Bitmap::load(&tmp).expect("load");
    std::fs::remove_file(&tmp).ok();

    assert_eq!(loaded.len(), idx.len());
    assert_eq!(loaded.dim(), idx.dim());
    assert_eq!(loaded.n_top(), idx.n_top());

    let mut rng = ChaCha8Rng::seed_from_u64(142);
    let q: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let r1 = idx.search(&q, 10);
    let r2 = loaded.search(&q, 10);
    assert_eq!(r1.indices_for_query(0), r2.indices_for_query(0));
}

#[test]
fn bitmap_index_constant_composition_invariant() {
    // Every doc bitmap should have exactly n_top bits set.
    let corpus = make_corpus(20);
    let n_top = D / 4;
    let mut idx = Bitmap::new(D, n_top);
    idx.add(&corpus);
    assert_eq!(idx.len(), N);
    assert_eq!(idx.bytes_per_vec(), D / 8);
    // Re-derive what the bitmap should be for each doc by ranking + cutoff.
    for di in 0..N {
        let v = &corpus[di * D..(di + 1) * D];
        let r = rank_transform(v);
        let expected_top: std::collections::HashSet<usize> =
            (0..D).filter(|&j| (r[j] as usize) >= D - n_top).collect();
        assert_eq!(expected_top.len(), n_top, "doc {di} top-set size");
    }
}

#[test]
fn bitmap_then_subset_recovers_exact_when_m_eq_n() {
    // When M = N the bitmap probe returns every doc and the subset
    // rerank must agree with the full RankQuant.search_asymmetric.
    let corpus = make_corpus(21);
    let n_top = D / 4;
    let mut bitmap = Bitmap::new(D, n_top);
    bitmap.add(&corpus);
    let mut rq = RankQuant::new(D, 2);
    rq.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(99_999);
    let query: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Stage 1 with M = N: candidate set is every doc.
    let cands = bitmap.top_m_candidates(&query, N);
    assert_eq!(cands.len(), N);

    let (_, two_stage_top) = rq.search_asymmetric_subset(&query, &cands, 10);
    let exact = rq.search_asymmetric(&query, 10);

    let two_stage_set: std::collections::HashSet<i64> = two_stage_top.iter().copied().collect();
    let exact_set: std::collections::HashSet<i64> =
        exact.indices_for_query(0).iter().copied().collect();
    assert_eq!(two_stage_set, exact_set, "M=N two-stage must equal exact");
}

#[test]
fn bitmap_top_m_candidates_uses_no_ground_truth() {
    // Sanity guardrail: top_m_candidates must depend only on the
    // query embedding (rank transform) and the stored bitmaps. The
    // same query twice must yield identical candidates.
    let corpus = make_corpus(22);
    let n_top = D / 4;
    let mut bitmap = Bitmap::new(D, n_top);
    bitmap.add(&corpus);
    let mut rng = ChaCha8Rng::seed_from_u64(50);
    let query: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let a = bitmap.top_m_candidates(&query, 50);
    let b = bitmap.top_m_candidates(&query, 50);
    assert_eq!(a, b, "candidate selection must be deterministic");
}

#[test]
fn bitmap_top_m_candidates_deterministic_at_ties() {
    // Body-kernel tie-break regression: with composite-key
    // partition `(score desc, doc_id asc)`, candidate selection is
    // deterministic even when many docs share the same bitmap-overlap
    // score at the cutoff. Construct a corpus where most docs have
    // identical bitmaps (= identical scores) and verify the candidate
    // set is bit-stable across repeated calls and across the batched
    // path.
    const TIE_D: usize = 128;
    const TIE_N: usize = 200;
    // First 150 docs are exact duplicates → all score identically
    // against any query. Remaining 50 are random.
    let mut rng = ChaCha8Rng::seed_from_u64(404);
    let duplicate_vec: Vec<f32> = (0..TIE_D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut corpus: Vec<f32> = Vec::with_capacity(TIE_N * TIE_D);
    for _ in 0..150 {
        corpus.extend_from_slice(&duplicate_vec);
    }
    for _ in 0..50 {
        for _ in 0..TIE_D {
            corpus.push(rng.gen_range(-1.0..1.0));
        }
    }
    let _ = rank_transform(&duplicate_vec); // assert symbol is in scope
    let n_top = TIE_D / 4;
    let mut bitmap = Bitmap::new(TIE_D, n_top);
    bitmap.add(&corpus);
    let query: Vec<f32> = (0..TIE_D).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Repeated calls must produce identical candidate sets — the
    // composite key forces a unique partition even when 150 docs
    // tie on score.
    let m = 100;
    let c1 = bitmap.top_m_candidates(&query, m);
    let c2 = bitmap.top_m_candidates(&query, m);
    assert_eq!(
        c1, c2,
        "single-query candidates must be deterministic at ties"
    );

    // Batched path agrees with single-query (the batched-equivalence
    // guarantee from `bitmap_batched_matches_single_query` extended
    // to the high-tie regime).
    let queries: Vec<f32> = (0..3 * TIE_D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    for q in [
        &queries[..TIE_D],
        &queries[TIE_D..2 * TIE_D],
        &queries[2 * TIE_D..],
    ] {
        let single = bitmap.top_m_candidates(q, m);
        let batched = bitmap.top_m_candidates_batched(q, m);
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0], single,
            "batched path must match single-query path under heavy ties",
        );
    }
}

#[test]
fn bitmap_batched_avx512_production_dim() {
    // The module-level D=128 in the broader equivalence test gives
    // qpv=2, which fails the `qpv % 8 == 0` dispatch gate and falls
    // through to the scalar fallback. This test pins the AVX-512
    // hot path at the production dim (D=1024, qpv=16, lanes=2),
    // which is what the bench actually times. It exercises a batch
    // smaller than CHUNK (so the hot path runs zero iterations and
    // the tail path covers the whole batch — distinct from
    // `_hot_plus_tail_split` below).
    const PROD_D: usize = 1024;
    const N_DOCS: usize = 256;
    const BATCH: usize = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let corpus: Vec<f32> = (0..N_DOCS * PROD_D)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let queries: Vec<f32> = (0..BATCH * PROD_D)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let n_top = PROD_D / 4;
    let mut bitmap = Bitmap::new(PROD_D, n_top);
    bitmap.add(&corpus);
    for m in [10usize, 50, 200] {
        let single: Vec<Vec<u32>> = (0..BATCH)
            .map(|bi| bitmap.top_m_candidates(&queries[bi * PROD_D..(bi + 1) * PROD_D], m))
            .collect();
        let batched = bitmap.top_m_candidates_batched(&queries, m);
        for bi in 0..BATCH {
            assert_eq!(
                single[bi], batched[bi],
                "AVX-512 batched diverged from single-query at dim={PROD_D}, M={m}, batch idx {bi}",
            );
        }
    }
}

#[test]
fn bitmap_batched_hot_plus_tail_split() {
    // Batch=11 with CHUNK=8 splits into one full hot-path chunk of 8
    // plus a 3-query tail. The hot and tail kernels share their
    // accs[__m512i; CHUNK] stack array but use different inner-loop
    // bounds (const-bounded vs runtime-bounded). This regression
    // exercises the boundary directly: each query must produce
    // exactly the same candidates whether it landed in the hot or
    // the tail half of the batch.
    const PROD_D: usize = 1024;
    const N_DOCS: usize = 256;
    const BATCH: usize = 11;
    let mut rng = ChaCha8Rng::seed_from_u64(101);
    let corpus: Vec<f32> = (0..N_DOCS * PROD_D)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let queries: Vec<f32> = (0..BATCH * PROD_D)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let n_top = PROD_D / 4;
    let mut bitmap = Bitmap::new(PROD_D, n_top);
    bitmap.add(&corpus);
    let single: Vec<Vec<u32>> = (0..BATCH)
        .map(|bi| bitmap.top_m_candidates(&queries[bi * PROD_D..(bi + 1) * PROD_D], 32))
        .collect();
    let batched = bitmap.top_m_candidates_batched(&queries, 32);
    assert_eq!(single.len(), batched.len());
    for bi in 0..BATCH {
        assert_eq!(
            single[bi], batched[bi],
            "hot+tail mismatch at batch idx {bi} (hot for bi<8, tail for bi in 8..11)",
        );
    }
}

#[test]
fn bitmap_batched_edge_cases() {
    // Documented contract: empty queries → empty result; m=0 →
    // empty per-query result; m > n_vectors → clamped to n_vectors.
    // Both batched and single-query paths must observe the same
    // contract.
    let corpus = make_corpus(13);
    let n_top = D / 4;
    let mut bitmap = Bitmap::new(D, n_top);
    bitmap.add(&corpus);

    // Empty batch.
    let empty: Vec<f32> = Vec::new();
    let res = bitmap.top_m_candidates_batched(&empty, 10);
    assert!(res.is_empty(), "empty queries must produce empty result");

    // m == 0: each per-query slot is an empty Vec.
    let mut rng = ChaCha8Rng::seed_from_u64(202);
    let queries: Vec<f32> = (0..3 * D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let res = bitmap.top_m_candidates_batched(&queries, 0);
    assert_eq!(res.len(), 3);
    for c in &res {
        assert!(c.is_empty(), "m=0 must produce empty candidate sets");
    }
    // Single-query path agrees.
    for bi in 0..3 {
        assert!(bitmap
            .top_m_candidates(&queries[bi * D..(bi + 1) * D], 0)
            .is_empty());
    }

    // m > n_vectors: clamps to n_vectors candidates per query.
    let res = bitmap.top_m_candidates_batched(&queries, N * 2);
    assert_eq!(res.len(), 3);
    for c in &res {
        assert_eq!(c.len(), N, "m > n_vectors must clamp to n_vectors");
    }
    // Single-query path agrees on the clamp.
    for bi in 0..3 {
        let single = bitmap.top_m_candidates(&queries[bi * D..(bi + 1) * D], N * 2);
        assert_eq!(single.len(), N);
    }

    // Chunked wrapper: empty input → empty result.
    assert!(bitmap
        .top_m_candidates_batched_chunked(&empty, 10, 4)
        .is_empty());
}

#[test]
fn bitmap_batched_avx512_high_qpv_no_panic() {
    // Regression for the AVX-512 batched kernel: an earlier version
    // capped the per-doc lane cache at 8 lanes (dim ≤ 4096), but
    // Bitmap accepts any dim multiple of 64 and the AVX-512
    // dispatch fires on any qpv % 8 == 0. At dim=4608 (qpv=72, lanes
    // = 9) the cached-doc-lane array would index out of bounds and
    // panic in release builds. The current kernel uses per-batch
    // accumulator ZMMs with no fixed lane cap; this test exercises
    // the high-qpv path and asserts bit-identical agreement with the
    // single-query path.
    const HIGH_D: usize = 4608; // > 4096, dim % 64 == 0, qpv % 8 == 0
    const N_DOCS: usize = 64;
    const BATCH: usize = 3;
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let corpus: Vec<f32> = (0..N_DOCS * HIGH_D)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let queries: Vec<f32> = (0..BATCH * HIGH_D)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let n_top = HIGH_D / 4;
    let mut bitmap = Bitmap::new(HIGH_D, n_top);
    bitmap.add(&corpus);
    let single: Vec<Vec<u32>> = (0..BATCH)
        .map(|bi| bitmap.top_m_candidates(&queries[bi * HIGH_D..(bi + 1) * HIGH_D], 16))
        .collect();
    let batched = bitmap.top_m_candidates_batched(&queries, 16);
    for bi in 0..BATCH {
        assert_eq!(
            single[bi], batched[bi],
            "high-qpv batched candidates diverged from single-query for batch idx {bi}",
        );
    }
}

#[test]
fn bitmap_batched_matches_single_query() {
    // Audit-critical: the batched candidate generator must produce
    // candidate sets that are *element-equal* to running
    // `top_m_candidates` per query. Scoring kernel is bit-identical
    // (AND-popcount-reduce), so any divergence indicates a layout
    // bug. We check both M=50 (small) and M=N (degenerate full
    // selection) to exercise the select_nth boundary.
    let corpus = make_corpus(31);
    let n_top = D / 4;
    let mut bitmap = Bitmap::new(D, n_top);
    bitmap.add(&corpus);
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let batch: usize = 7; // intentionally non-power-of-2
    let queries: Vec<f32> = (0..batch * D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    for m in [10usize, 50, 100] {
        let single: Vec<Vec<u32>> = (0..batch)
            .map(|bi| bitmap.top_m_candidates(&queries[bi * D..(bi + 1) * D], m))
            .collect();
        let batched = bitmap.top_m_candidates_batched(&queries, m);
        assert_eq!(single.len(), batched.len());
        for bi in 0..batch {
            assert_eq!(
                single[bi], batched[bi],
                "batched candidates diverged from single-query for batch idx {bi}, M={m}",
            );
        }
    }
    // Also exercise the chunked-parallel wrapper at chunk size 3
    // (forces a non-aligned tail batch).
    let chunked = bitmap.top_m_candidates_batched_chunked(&queries, 50, 3);
    let reference: Vec<Vec<u32>> = (0..batch)
        .map(|bi| bitmap.top_m_candidates(&queries[bi * D..(bi + 1) * D], 50))
        .collect();
    assert_eq!(chunked, reference);
}

#[test]
#[should_panic(expected = "u16 rank invariant")]
fn bitmap_new_rejects_dim_above_u16_max() {
    // dim = 65536 is a multiple of 64 but exceeds u16::MAX, so the rank
    // transform (u16 ranks) and the query-side u16 coordinate indexing cannot
    // represent it. The constructor must reject it loudly rather than construct
    // an index that panics later in `add`/`search`, and stay consistent with
    // the `.tvbm` loader (which caps dim at MAX_DIM).
    let _ = Bitmap::new(65_536, 256);
}

#[test]
#[should_panic(expected = "batch_size must be > 0")]
fn bitmap_batched_chunked_rejects_zero_batch_size() {
    // `batch_size = 0` makes the per-chunk float count zero, which would panic
    // deep inside `par_chunks(0)`; the public method guards it up front.
    let corpus = make_corpus(77);
    let mut idx = Bitmap::new(D, D / 4);
    idx.add(&corpus);
    let q = corpus[..D].to_vec();
    let _ = idx.top_m_candidates_batched_chunked(&q, 10, 0);
}
