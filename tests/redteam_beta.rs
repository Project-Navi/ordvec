//! Red-team hardening tests for the rank-mode substrate.
//!
//! Each test pins a robustness fix to a concrete failure mode. All of
//! them exercise the rank-mode substrate (`Rank`, `RankQuant`,
//! `SignBitmap`, and the byte-LUT bench helper) that lives in
//! `ordvec`.
//!
//! - **RT-2 / RT-3**: the AVX-512 / AVX2 asymmetric kernels carry lane
//!   invariants (`dim % 64` for AVX-512, `dim % 16` for AVX2 b=2,
//!   `dim % 8` for AVX2 b=4). For constructor-valid dims that violate
//!   the SIMD invariant (e.g. 48, 80, 20) the kernels silently dropped
//!   the trailing chunk in *release* builds and returned wrong top-k.
//!   The dispatch must only reach a kernel whose invariant holds, and
//!   fall back to the scalar path otherwise. We assert parity against
//!   the scalar `search_asymmetric_byte_lut` reference for b in {2, 4}.
//! - **#4**: `search_asymmetric_subset` with an out-of-range candidate
//!   id used to panic with a cryptic slice-range message. It must now
//!   fail an explicit bounds `assert!` with a clear message.
//! - **P-H**: passing `k = usize::MAX` used to attempt `vec![_; nq*k]`
//!   and abort with `capacity overflow`. `k` must be clamped to
//!   `n_vectors` before any allocation.
//! - **P-I**: a b=1 `RankQuant` must run `search_asymmetric`
//!   end-to-end (routed away from the {2,4}-only byte-LUT path) and
//!   match a scalar reference.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use ordvec::rank::{bucket_centre, bucket_ranks, rank_transform, rankquant_norm};
use ordvec::search_asymmetric_byte_lut;
use ordvec::{Rank, RankQuant, SearchResults, SignBitmap};

fn make_corpus(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Scalar reference for asymmetric RankQuant scoring of one query
/// against one bucket-packed doc. Mirrors `scan_via_lut_scalar`:
/// `inv_norm * Σ_d q_unit[d] * bucket_centre(doc_bucket[d])`.
fn ref_rankquant_asymmetric(query: &[f32], doc: &[f32], bits: u8) -> f32 {
    let dim = query.len();
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q_unit: Vec<f32> = if norm <= 1e-12 {
        vec![0.0; dim]
    } else {
        query.iter().map(|&x| x / norm).collect()
    };
    let doc_ranks = rank_transform(doc);
    let doc_buckets = bucket_ranks(&doc_ranks, bits);
    let inv_norm = 1.0_f32 / rankquant_norm(dim, bits);
    let mut acc = 0.0f32;
    for d in 0..dim {
        acc += q_unit[d] * bucket_centre(doc_buckets[d], bits);
    }
    acc * inv_norm
}

// -------------------------------------------------------------------
// RT-2 / RT-3: SIMD dispatch must never call a kernel whose lane
// invariant is unmet. Parity is checked against the scalar byte-LUT
// reference for the {2, 4}-bit widths the byte-LUT path supports.
//
// The (dim, bits) grid is chosen to straddle every dispatch tier:
//   - 48 b2 : dim % 64 != 0 but dim % 16 == 0  (AVX2 b2 ok, AVX-512 no)
//   - 80 b4 : dim % 64 != 0 but dim % 8  == 0  (AVX2 b4 ok, AVX-512 no)
//   - 20 b2 : dim % 16 != 0 and dim % 64 != 0  (scalar fallback only)
//   -  4 b2 : tiny, scalar fallback only
//   - 64 b2 : dim % 64 == 0  (AVX-512 happy path — must NOT regress)
//   - 128 b4: dim % 64 == 0  (AVX-512 happy path — must NOT regress)
//   - 768 b4: production-scale AVX-512 happy path
// -------------------------------------------------------------------

fn assert_asym_matches_byte_lut(dim: usize, bits: u8, seed: u64) {
    let n = 64;
    let corpus = make_corpus(seed, n, dim);
    let mut idx = RankQuant::new(dim, bits);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(7));
    let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

    let k = 10;
    let prod = idx.search_asymmetric(&query, k);
    let reference = search_asymmetric_byte_lut(&idx, &query, k);

    let prod_idx = prod.indices_for_query(0);
    let ref_idx = reference.indices_for_query(0);

    // This dispatch-grid red-team check uses set equality because random
    // near-ties can sit inside the scalar/SIMD tolerance. Exact score-tie
    // ordering is pinned by tests/determinism_contract.rs.
    let prod_set: std::collections::HashSet<i64> = prod_idx.iter().copied().collect();
    let ref_set: std::collections::HashSet<i64> = ref_idx.iter().copied().collect();
    assert_eq!(
        prod_set, ref_set,
        "dim={dim} b={bits}: search_asymmetric top-{k} set diverged from scalar byte-LUT reference",
    );

    // Scores at matching docs must agree within fp tolerance.
    let prod_scores = prod.scores_for_query(0);
    for slot in 0..k.min(n) {
        let di = prod_idx[slot];
        if di < 0 {
            continue;
        }
        let ri = ref_idx.iter().position(|&x| x == di).unwrap();
        let sp = prod_scores[slot];
        let sr = reference.scores_for_query(0)[ri];
        assert!(
            (sp - sr).abs() < 1e-3,
            "dim={dim} b={bits} doc {di}: score {sp} vs reference {sr}",
        );
    }
}

#[test]
fn rt2_asym_b2_dim48_matches_scalar() {
    assert_asym_matches_byte_lut(48, 2, 101);
}

#[test]
fn rt2_asym_b4_dim80_matches_scalar() {
    assert_asym_matches_byte_lut(80, 4, 102);
}

#[test]
fn rt2_asym_b2_dim20_matches_scalar() {
    assert_asym_matches_byte_lut(20, 2, 103);
}

#[test]
fn rt2_asym_b2_dim4_matches_scalar() {
    assert_asym_matches_byte_lut(4, 2, 104);
}

#[test]
fn rt2_asym_b2_dim64_happy_path_matches_scalar() {
    assert_asym_matches_byte_lut(64, 2, 105);
}

#[test]
fn rt2_asym_b4_dim128_happy_path_matches_scalar() {
    assert_asym_matches_byte_lut(128, 4, 106);
}

#[test]
fn rt2_asym_b4_dim768_happy_path_matches_scalar() {
    assert_asym_matches_byte_lut(768, 4, 107);
}

// -------------------------------------------------------------------
// #4: search_asymmetric_subset must reject out-of-range candidate ids
// with a clear assert, not a cryptic slice-range panic.
// -------------------------------------------------------------------

#[test]
#[should_panic(expected = "candidate id out of range")]
fn subset_rejects_out_of_range_candidate() {
    let dim = 64;
    let n = 32;
    let corpus = make_corpus(201, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(202);
    let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

    // n_vectors == 32, so id 999 is out of range.
    let candidates: Vec<u32> = vec![0, 1, 999];
    let _ = idx.search_asymmetric_subset(&query, &candidates, 3);
}

#[test]
fn subset_accepts_in_range_candidates() {
    // Boundary: the largest valid id is n_vectors - 1.
    let dim = 64;
    let n = 32;
    let corpus = make_corpus(203, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(204);
    let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

    let candidates: Vec<u32> = vec![0, 5, (n - 1) as u32];
    let (scores, global) = idx.search_asymmetric_subset(&query, &candidates, 3);
    assert_eq!(scores.len(), 3);
    assert_eq!(global.len(), 3);
    // Every returned global id must be one of the candidates.
    for &g in &global {
        assert!(g < 0 || candidates.contains(&(g as u32)));
    }
}

// -------------------------------------------------------------------
// P-H: k = usize::MAX must clamp to n_vectors, not abort with
// capacity overflow on `vec![_; nq * k]`.
// -------------------------------------------------------------------

#[test]
fn rankquant_search_huge_k_clamps() {
    let dim = 64;
    let n = 16;
    let corpus = make_corpus(301, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(302, 1, dim);

    let res = idx.search(&query, usize::MAX);
    let returned: usize = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert!(returned <= n, "search returned more than n_vectors results");
    assert_eq!(returned, n, "all n docs should be returned for huge k");
}

#[test]
fn rankquant_search_asymmetric_huge_k_clamps() {
    let dim = 64;
    let n = 16;
    let corpus = make_corpus(303, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(304, 1, dim);

    let res = idx.search_asymmetric(&query, usize::MAX);
    let returned: usize = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert!(returned <= n);
    assert_eq!(returned, n);
}

#[test]
fn rank_index_search_huge_k_clamps() {
    let dim = 64;
    let n = 16;
    let corpus = make_corpus(305, n, dim);
    let mut idx = Rank::new(dim);
    idx.add(&corpus);
    let query = make_corpus(306, 1, dim);

    let res = idx.search(&query, usize::MAX);
    let returned: usize = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert!(returned <= n);
    assert_eq!(returned, n);
}

#[test]
fn rank_index_search_asymmetric_huge_k_clamps() {
    let dim = 64;
    let n = 16;
    let corpus = make_corpus(307, n, dim);
    let mut idx = Rank::new(dim);
    idx.add(&corpus);
    let query = make_corpus(308, 1, dim);

    let res = idx.search_asymmetric(&query, usize::MAX);
    let returned: usize = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert!(returned <= n);
    assert_eq!(returned, n);
}

#[test]
fn sign_bitmap_top_m_huge_m_clamps() {
    let dim = 64;
    let n = 16;
    let corpus = make_corpus(309, n, dim);
    let mut idx = SignBitmap::new(dim);
    idx.add(&corpus);
    let query = make_corpus(310, 1, dim);

    let cands = idx.top_m_candidates(&query, usize::MAX);
    assert!(cands.len() <= n);
    assert_eq!(cands.len(), n);

    let batched = idx.top_m_candidates_batched(&query, usize::MAX);
    assert_eq!(batched.len(), 1);
    assert!(batched[0].len() <= n);
    assert_eq!(batched[0].len(), n);
}

// -------------------------------------------------------------------
// P-J: search_asymmetric_byte_lut with k = usize::MAX must clamp to
// n_vectors and NOT abort with capacity overflow. Covers Finding 2
// (the byte-LUT method only clamped `k_eff`, leaving the `nq * k`
// allocation and `par_chunks_mut(k)` sized by the raw `usize::MAX`)
// and the byte-LUT half of Finding 1 (the `result_buffer_len` guard on
// `nq * k`). The clamped `k` must flow into the returned `k` field so
// the per-query accessors slice consistently.
// -------------------------------------------------------------------

#[test]
fn byte_lut_huge_k_clamps_no_overflow() {
    let dim = 64;
    let n = 16;
    let corpus = make_corpus(501, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let query = make_corpus(502, 1, dim);

    // Single query, k = usize::MAX. Before the fix this aborted the
    // process with `capacity overflow` while sizing `vec![_; 1 * MAX]`.
    let res: SearchResults = search_asymmetric_byte_lut(&idx, &query, usize::MAX);
    assert_eq!(res.nq, 1);
    // The reported k must be clamped to n_vectors so the buffer and the
    // accessor stride agree.
    assert_eq!(res.k, n, "byte-LUT k must clamp to n_vectors");
    let returned: usize = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert!(
        returned <= n,
        "byte-LUT search returned more than n_vectors results",
    );
    assert_eq!(returned, n, "all n docs should be returned for huge k");
}

#[test]
fn byte_lut_huge_k_multi_query_clamps_no_overflow() {
    // Multi-query exercises the `nq * k` result-buffer axis (Finding 1):
    // with the raw `usize::MAX` the product `nq * k` overflows usize and
    // would silently wrap to a too-small Vec; `result_buffer_len` turns
    // that into a loud panic, and the k-clamp keeps the real product at
    // `nq * n_vectors`.
    let dim = 64;
    let n = 16;
    let nq = 3;
    let corpus = make_corpus(503, n, dim);
    let mut idx = RankQuant::new(dim, 2);
    idx.add(&corpus);
    let queries = make_corpus(504, nq, dim);

    let res: SearchResults = search_asymmetric_byte_lut(&idx, &queries, usize::MAX);
    assert_eq!(res.nq, nq);
    assert_eq!(res.k, n);
    for qi in 0..nq {
        let returned: usize = res
            .indices_for_query(qi)
            .iter()
            .filter(|&&i| i >= 0)
            .count();
        assert_eq!(returned, n, "query {qi}: all n docs should be returned");
    }
}

// -------------------------------------------------------------------
// P-I: a b=1 index must run search_asymmetric end-to-end and match a
// scalar reference (must NOT route into the {2,4}-only byte-LUT path).
// -------------------------------------------------------------------

#[test]
fn rankquant_b1_asymmetric_works_and_matches_reference() {
    let dim = 64;
    let n = 64;
    let corpus = make_corpus(401, n, dim);
    let mut idx = RankQuant::new(dim, 1);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(402);
    let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

    let k = 10;
    let res = idx.search_asymmetric(&query, k);

    let ref_scores: Vec<f32> = (0..n)
        .map(|di| ref_rankquant_asymmetric(&query, &corpus[di * dim..(di + 1) * dim], 1))
        .collect();

    // Top-k set parity against the scalar reference.
    let mut ref_sorted: Vec<(usize, f32)> = ref_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    ref_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_ref: std::collections::HashSet<usize> = ref_sorted[..k].iter().map(|x| x.0).collect();
    let top_idx: std::collections::HashSet<usize> = res
        .indices_for_query(0)
        .iter()
        .filter(|&&i| i >= 0)
        .map(|&i| i as usize)
        .collect();
    assert_eq!(top_idx, top_ref, "b=1 asymmetric top-{k} set mismatch");

    // Scores agree within fp tolerance.
    let prod_scores = res.scores_for_query(0);
    let prod_idx = res.indices_for_query(0);
    for slot in 0..k {
        let di = prod_idx[slot];
        if di < 0 {
            continue;
        }
        let sp = prod_scores[slot];
        let sr = ref_scores[di as usize];
        assert!(
            (sp - sr).abs() < 1e-4,
            "b=1 slot {slot} doc {di}: {sp} vs {sr}",
        );
    }
}
