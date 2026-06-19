//! Capability-gated `b=8` RankQuant integration tests (#221).
//!
//! `b=8` is a stable/core evidence-refinement width, not experimental:
//!
//! - code generation, pair-evidence, and asymmetric (float-query) scoring
//!   work at **any** dim;
//! - symmetric scoring (and the symmetric analytical norm) require
//!   `dim % 256 == 0` (equal bucket occupancy), so a non-`256`-aligned
//!   `b=8` index is `AsymmetricOnly` and its `search` panics with an exact,
//!   directing message.
//!
//! These tests pin the maintainer's capability matrix plus a brute-force
//! parity check of the scalar `b=8` asymmetric path against a naive
//! reference.

use ordvec::rank::{bucket_centre, bucket_ranks, rank_transform, rankquant_norm};
use ordvec::{RankQuant, RankQuantCapability};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Naive reference for `b=8` asymmetric scoring of one float query against
/// one float doc: L2-normalise the query, rank-transform + bucket the doc to
/// `b=8` codes, score `Σ_d q_unit[d] * bucket_centre(code[d]) / norm`. This
/// mirrors `ref_rankquant_asymmetric` in the shared helpers but is duplicated
/// here so the b=8 module is self-contained.
fn ref_b8_asymmetric(q: &[f32], doc: &[f32]) -> f32 {
    let d = q.len();
    let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q_unit: Vec<f32> = q.iter().map(|x| x / q_norm).collect();
    let r = rank_transform(doc);
    let codes = bucket_ranks(&r, 8);
    // Exact L2 norm of this doc's centred bucket vector. For b=8 the bucket
    // occupancy is uniform only when `dim % 256 == 0`; at other dims (e.g. 384)
    // the closed-form `rankquant_norm` mis-scales the absolute score, so the
    // reference — like production's `asymmetric_norm` — sums the realised
    // squared centres (f64-accumulated, matching `rankquant_eval_norm`). The
    // ranks are a permutation of `0..d` for every doc, so this equals the
    // closed form exactly at 256-aligned dims.
    let norm = {
        let acc: f64 = codes
            .iter()
            .map(|&c| {
                let cc = bucket_centre(c, 8) as f64;
                cc * cc
            })
            .sum();
        acc.sqrt() as f32
    };
    let mut acc = 0.0f32;
    for i in 0..d {
        acc += q_unit[i] * bucket_centre(codes[i], 8);
    }
    acc / norm
}

fn random_corpus(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

// ---------------------------------------------------------------------
// Capability reporting.
// ---------------------------------------------------------------------

#[test]
fn b8_new_asymmetric_384_is_asymmetric_only() {
    let idx = RankQuant::new_asymmetric(384, 8);
    assert_eq!(idx.capability(), RankQuantCapability::AsymmetricOnly);
    assert!(!idx.symmetric_supported());
    assert_eq!(idx.bits(), 8);
    assert_eq!(idx.dim(), 384);
    // b=8 stores one byte per coordinate.
    assert_eq!(idx.bytes_per_vec(), 384);
}

#[test]
fn b8_new_1024_is_symmetric_and_asymmetric() {
    let idx = RankQuant::new(1024, 8);
    assert_eq!(
        idx.capability(),
        RankQuantCapability::SymmetricAndAsymmetric
    );
    assert!(idx.symmetric_supported());
    assert_eq!(idx.bits(), 8);
}

#[test]
fn b8_new_asymmetric_256_aligned_upgrades_to_full() {
    // new_asymmetric on a 256-aligned dim should NOT withhold symmetric
    // scoring — there is no reason to, the analytical norm is exact.
    let idx = RankQuant::new_asymmetric(768, 8);
    assert_eq!(
        idx.capability(),
        RankQuantCapability::SymmetricAndAsymmetric
    );
    assert!(idx.symmetric_supported());
}

#[test]
fn b124_constructors_are_always_full_capability() {
    for &(dim, bits) in &[(384usize, 4u8), (384, 2), (256, 1), (1024, 4)] {
        let a = RankQuant::new(dim, bits);
        assert_eq!(a.capability(), RankQuantCapability::SymmetricAndAsymmetric);
        assert!(a.symmetric_supported());
        // new_asymmetric for b ∈ {1,2,4} is never less capable than new.
        let b = RankQuant::new_asymmetric(dim, bits);
        assert_eq!(b.capability(), RankQuantCapability::SymmetricAndAsymmetric);
        assert!(b.symmetric_supported());
    }
}

// ---------------------------------------------------------------------
// new() fail-loud for non-256-aligned b=8.
// ---------------------------------------------------------------------

#[test]
fn b8_new_panics_for_non_256_aligned_dim_directing_to_new_asymmetric() {
    let res = std::panic::catch_unwind(|| RankQuant::new(384, 8));
    assert!(res.is_err(), "new(384, 8) must panic (384 % 256 != 0)");
    let payload = match res {
        Ok(_) => panic!("panic payload present"),
        Err(payload) => payload,
    };
    let msg = *payload
        .downcast::<String>()
        .expect("panic payload should be a String");
    assert!(
        msg.contains("dim % 256 == 0"),
        "panic should explain the 256-alignment requirement: {msg}"
    );
    assert!(
        msg.contains("new_asymmetric"),
        "panic should direct to new_asymmetric: {msg}"
    );
}

// ---------------------------------------------------------------------
// dim=384 b=8: code-gen passes, asymmetric passes, symmetric REJECTS.
// ---------------------------------------------------------------------

#[test]
fn b8_384_code_gen_and_asymmetric_work() {
    let dim = 384;
    let n = 50;
    let corpus = random_corpus(8384, n, dim);
    let mut idx = RankQuant::new_asymmetric(dim, 8);
    // add() runs the rank → bucket → pack pipeline (the code-gen path).
    idx.add(&corpus);
    assert_eq!(idx.len(), n);
    assert_eq!(idx.byte_size(), n * dim); // one byte per coord per doc

    // Asymmetric scoring works at this non-256-aligned dim.
    let query = random_corpus(8385, 1, dim);
    let res = idx.search_asymmetric(&query, 10);
    assert_eq!(res.nq, 1);
    assert_eq!(res.k, 10);
    for slot in 0..10 {
        assert!(res.scores_for_query(0)[slot].is_finite());
        let id = res.indices_for_query(0)[slot];
        assert!(id >= 0 && (id as usize) < n);
    }
}

#[test]
fn b8_384_symmetric_search_rejects_with_exact_message() {
    let dim = 384;
    let mut idx = RankQuant::new_asymmetric(dim, 8);
    idx.add(&random_corpus(8386, 8, dim));
    let query = random_corpus(8387, 1, dim);

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = idx.search(&query, 5);
    }));
    assert!(
        res.is_err(),
        "symmetric search on AsymmetricOnly must panic"
    );
    let msg = *res
        .unwrap_err()
        .downcast::<String>()
        .expect("panic payload should be a String");
    // The EXACT wording shape from the spec.
    let expected = format!(
        "RankQuant b=8 symmetric scoring requires dim % 256 == 0; dim={dim} supports asymmetric/evidence APIs only."
    );
    assert_eq!(msg, expected, "symmetric-gating message must match exactly");
}

// ---------------------------------------------------------------------
// dim=768/1024/1536 b=8: full path incl. symmetric passes.
// ---------------------------------------------------------------------

#[test]
fn b8_aligned_dims_full_path_including_symmetric() {
    for &dim in &[768usize, 1024, 1536] {
        let n = 40;
        let corpus = random_corpus(9000 + dim as u64, n, dim);
        // Both constructors should yield a full-capability instance here.
        let mut idx = RankQuant::new(dim, 8);
        assert!(
            idx.symmetric_supported(),
            "dim={dim} should be symmetric-capable"
        );
        idx.add(&corpus);

        let queries = random_corpus(9500 + dim as u64, 3, dim);

        // Symmetric path runs without panicking and returns well-formed,
        // descending, in-range results.
        let sym = idx.search(&queries, 10);
        assert_eq!(sym.nq, 3);
        assert_eq!(sym.k, 10);
        for qi in 0..3 {
            let scores = sym.scores_for_query(qi);
            let ids = sym.indices_for_query(qi);
            for slot in 0..10 {
                assert!(scores[slot].is_finite(), "dim={dim} non-finite sym score");
                assert!(ids[slot] >= 0 && (ids[slot] as usize) < n);
            }
            for slot in 1..10 {
                assert!(
                    scores[slot].total_cmp(&scores[slot - 1]).is_le(),
                    "dim={dim} symmetric results not sorted descending"
                );
            }
        }

        // Asymmetric path runs too.
        let asym = idx.search_asymmetric(&queries, 10);
        assert_eq!(asym.nq, 3);
        assert_eq!(asym.k, 10);
    }
}

// ---------------------------------------------------------------------
// dim=384 b=4 UNCHANGED (sanity that the b=8 work didn't disturb b=4).
// ---------------------------------------------------------------------

#[test]
fn b4_384_unchanged_full_capability_and_search() {
    let dim = 384;
    let n = 40;
    let corpus = random_corpus(4384, n, dim);
    let mut idx = RankQuant::new(dim, 4);
    assert_eq!(
        idx.capability(),
        RankQuantCapability::SymmetricAndAsymmetric
    );
    assert!(idx.symmetric_supported());
    idx.add(&corpus);
    let queries = random_corpus(4385, 3, dim);
    let sym = idx.search(&queries, 10);
    assert_eq!(sym.k, 10);
    let asym = idx.search_asymmetric(&queries, 10);
    assert_eq!(asym.k, 10);
}

// ---------------------------------------------------------------------
// Brute-force parity: b=8 asymmetric scores match a naive reference.
// ---------------------------------------------------------------------

#[test]
fn b8_asymmetric_matches_naive_reference_any_dim() {
    // Cover both an asymmetric-only (384) and a full-capability (768) dim;
    // the asymmetric scalar path is identical for both.
    for &dim in &[384usize, 768] {
        let n = 60;
        let corpus = random_corpus(7000 + dim as u64, n, dim);
        let mut idx = RankQuant::new_asymmetric(dim, 8);
        idx.add(&corpus);

        let mut rng = ChaCha8Rng::seed_from_u64(7777 + dim as u64);
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let res = idx.search_asymmetric(&query, 10);

        let ref_scores: Vec<f32> = (0..n)
            .map(|di| ref_b8_asymmetric(&query, &corpus[di * dim..(di + 1) * dim]))
            .collect();

        // Every returned score must agree with the reference at its doc id.
        for slot in 0..10 {
            let di = res.indices_for_query(0)[slot] as usize;
            let got = res.scores_for_query(0)[slot];
            let want = ref_scores[di];
            assert!(
                (got - want).abs() < 1e-4,
                "dim={dim} slot {slot} doc {di}: {got} vs ref {want}"
            );
        }

        // And the returned top-10 set must equal the reference top-10 set.
        let mut ref_sorted: Vec<(usize, f32)> = ref_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        ref_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_ref: std::collections::HashSet<usize> =
            ref_sorted[..10].iter().map(|x| x.0).collect();
        let top_got: std::collections::HashSet<usize> = res
            .indices_for_query(0)
            .iter()
            .map(|&i| i as usize)
            .collect();
        assert_eq!(top_got, top_ref, "dim={dim} b=8 top-10 set mismatch");
    }
}

// ---------------------------------------------------------------------
// Optimized (AVX-512 gather) b=8 asymmetric path is parity-correct vs the
// naive reference across the headline embedding dims.
//
// On an AVX-512 host `search_asymmetric` dispatches the b=8 score to the
// `vgatherdps` kernel; on every other host it takes the scalar LUT path.
// Either way the returned top-k scores must agree with the naive per-doc
// reference within the crate's 1e-4 cross-backend score tolerance, and the
// returned top-k *set* must equal the reference top-k set. This is the
// end-to-end parity gate for the optimized kernel at dims 384/768/1024/1536.
// ---------------------------------------------------------------------

#[test]
fn b8_asymmetric_optimized_path_parity_headline_dims() {
    for &dim in &[384usize, 768, 1024, 1536] {
        let n = 200;
        let corpus = random_corpus(6000 + dim as u64, n, dim);
        let mut idx = RankQuant::new_asymmetric(dim, 8);
        idx.add(&corpus);

        let mut rng = ChaCha8Rng::seed_from_u64(6666 + dim as u64);
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

        let k = 25;
        let res = idx.search_asymmetric(&query, k);

        // Naive scalar reference score per doc.
        let ref_scores: Vec<f32> = (0..n)
            .map(|di| ref_b8_asymmetric(&query, &corpus[di * dim..(di + 1) * dim]))
            .collect();

        // (a) every returned score agrees with the reference at its doc id.
        for slot in 0..k {
            let di = res.indices_for_query(0)[slot] as usize;
            let got = res.scores_for_query(0)[slot];
            let want = ref_scores[di];
            assert!(
                (got - want).abs() < 1e-4,
                "dim={dim} slot {slot} doc {di}: optimized {got} vs ref {want}"
            );
        }

        // (b) the returned top-k *set* equals the reference top-k set.
        let mut ref_sorted: Vec<(usize, f32)> = ref_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        ref_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_ref: std::collections::HashSet<usize> =
            ref_sorted[..k].iter().map(|x| x.0).collect();
        let top_got: std::collections::HashSet<usize> = res
            .indices_for_query(0)
            .iter()
            .map(|&i| i as usize)
            .collect();
        assert_eq!(
            top_got, top_ref,
            "dim={dim} optimized b=8 top-{k} set mismatch vs reference"
        );
    }
}

// The optimized b=8 path must also be parity-correct through the subset
// rerank entry point (`search_asymmetric_subset`), which gathers candidate
// bytes into a scratch buffer and runs the same gather kernel.
#[test]
fn b8_asymmetric_subset_optimized_path_parity() {
    let dim = 768;
    let n = 300;
    let corpus = random_corpus(6321, n, dim);
    let mut idx = RankQuant::new_asymmetric(dim, 8);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(6322);
    let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

    // An arbitrary, intentionally-unsorted candidate subset.
    let candidates: Vec<u32> = (0..n as u32).rev().step_by(3).collect();
    let k = 10;
    let (scores, indices) = idx.search_asymmetric_subset(&query, &candidates, k);

    for slot in 0..k {
        let di = indices[slot] as usize;
        let want = ref_b8_asymmetric(&query, &corpus[di * dim..(di + 1) * dim]);
        assert!(
            (scores[slot] - want).abs() < 1e-4,
            "subset slot {slot} doc {di}: optimized {} vs ref {want}",
            scores[slot]
        );
    }
}

// The b=8 routing also runs through the *batched* two-stage rerank entry point
// (`search_asymmetric_subset_batched_serial`), which packs each query's
// candidate row into a reused `SubsetScratch` and scans it with the same b=8
// gather kernel. Cover both a non-256-aligned dim (384, exercising the
// empirical asymmetric norm) and an aligned dim (768), with two queries that
// have distinct candidate rows (exercising the CSR offsets and scratch reuse
// across rows). Every returned score must match the per-doc naive reference.
#[test]
fn b8_asymmetric_subset_batched_serial_path_parity() {
    for &dim in &[384usize, 768] {
        let n = 256;
        let corpus = random_corpus(8100 + dim as u64, n, dim);
        let mut idx = RankQuant::new_asymmetric(dim, 8);
        idx.add(&corpus);

        let mut rng = ChaCha8Rng::seed_from_u64(8200 + dim as u64);
        let q0: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let q1: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut queries = q0.clone();
        queries.extend_from_slice(&q1);

        // Two distinct, intentionally-unsorted candidate rows in CSR layout.
        let cand0: Vec<u32> = (0..n as u32).rev().step_by(3).collect();
        let cand1: Vec<u32> = (0..n as u32).step_by(5).collect();
        let mut candidates = cand0.clone();
        candidates.extend_from_slice(&cand1);
        let candidate_offsets = [0usize, cand0.len(), cand0.len() + cand1.len()];

        let k = 10;
        let res = idx.search_asymmetric_subset_batched_serial(
            &queries,
            &candidate_offsets,
            &candidates,
            k,
        );

        for (qi, q) in [&q0, &q1].into_iter().enumerate() {
            let got_scores = res.scores_for_query(qi);
            let got_indices = res.indices_for_query(qi);
            for slot in 0..k {
                let di = got_indices[slot];
                if di < 0 {
                    continue; // fewer candidates than k in this row
                }
                let di = di as usize;
                let want = ref_b8_asymmetric(q, &corpus[di * dim..(di + 1) * dim]);
                assert!(
                    (got_scores[slot] - want).abs() < 1e-4,
                    "dim={dim} q{qi} slot {slot} doc {di}: batched {} vs ref {want}",
                    got_scores[slot]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// validate_params: b=8 is code-valid at any dim; b ∈ {1,2,4} unchanged.
// ---------------------------------------------------------------------

#[test]
fn validate_params_b8_any_dim_but_b124_still_require_alignment() {
    // b=8 accepts any dim >= 2 (no dim % 256 requirement).
    assert!(RankQuant::validate_params(384, 8).is_ok());
    assert!(RankQuant::validate_params(2, 8).is_ok());
    assert!(RankQuant::validate_params(1000, 8).is_ok());
    assert!(
        RankQuant::validate_params(1, 8).is_err(),
        "dim < 2 rejected"
    );

    // b ∈ {1,2,4} keep their 2^bits divisibility requirement.
    assert!(RankQuant::validate_params(6, 2).is_err(), "6 % 4 != 0");
    assert!(RankQuant::validate_params(8, 2).is_ok());
    assert!(RankQuant::validate_params(384, 4).is_ok());
    // b=3 is still not a packable width.
    assert!(RankQuant::validate_params(384, 3).is_err());
}

// ---------------------------------------------------------------------
// Symmetric b=8 (256-aligned) matches a naive symmetric reference.
// ---------------------------------------------------------------------

#[test]
fn b8_symmetric_matches_naive_reference_aligned_dim() {
    let dim = 512; // 256-aligned → exact analytical norm
    let n = 40;
    let corpus = random_corpus(5512, n, dim);
    let mut idx = RankQuant::new(dim, 8);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(5513);
    let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let res = idx.search(&query, 10);

    // Naive symmetric reference: bucket query + doc to b=8, dot the centred
    // bucket vectors, divide by norm^2.
    let norm = rankquant_norm(dim, 8);
    let inv_norm_sq = 1.0f32 / (norm * norm);
    let q_codes = bucket_ranks(&rank_transform(&query), 8);
    let ref_scores: Vec<f32> = (0..n)
        .map(|di| {
            let doc = &corpus[di * dim..(di + 1) * dim];
            let d_codes = bucket_ranks(&rank_transform(doc), 8);
            let acc: f32 = q_codes
                .iter()
                .zip(&d_codes)
                .map(|(&qc, &dc)| bucket_centre(qc, 8) * bucket_centre(dc, 8))
                .sum();
            acc * inv_norm_sq
        })
        .collect();

    for slot in 0..10 {
        let di = res.indices_for_query(0)[slot] as usize;
        let got = res.scores_for_query(0)[slot];
        assert!(
            (got - ref_scores[di]).abs() < 1e-4,
            "b=8 symmetric slot {slot} doc {di}: {got} vs ref {}",
            ref_scores[di]
        );
    }
}

#[test]
fn rankquant_eval_search_supports_b8_at_any_dim() {
    // The eval/empirical path (check_eval_bits widened to 1..=8) accepts b=8 even
    // at a non-256-aligned dim, where the analytical symmetric norm is
    // unavailable — it computes the norm empirically. Returns ranked results
    // without panicking.
    //
    // This is a *distinct* surface from the analytical-norm `RankQuant::search`,
    // whose b=8 symmetric scoring is gated to `dim % 256 == 0`. There is no
    // contradiction: the eval path's empirical norm is exact under any bucket
    // occupancy, which is precisely why it is unbound by the 256 gate.
    let dim = 384usize; // not a multiple of 256
    let n = 32usize;
    let nq = 2usize;
    let corpus: Vec<f32> = (0..n * dim)
        .map(|i| ((i * 7 % 101) as f32) - 50.0)
        .collect();
    let queries: Vec<f32> = (0..nq * dim)
        .map(|i| ((i * 13 % 97) as f32) - 48.0)
        .collect();
    let res = ordvec::rankquant_eval_search(&corpus, &queries, dim, 8, 5);
    assert_eq!(res.k, 5);
    assert_eq!(res.nq, nq);
    for &id in &res.indices {
        assert!(
            id >= 0 && (id as usize) < n,
            "eval-search id out of range: {id}"
        );
    }
}
