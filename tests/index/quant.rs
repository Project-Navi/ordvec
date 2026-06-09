//! RankQuant (B-bit bucket-packed) integration tests.

use ordvec::rank::{bucket_centre, rank_to_bucket, rank_transform};
use ordvec::{rankquant_eval_search, RankQuant, SearchResults};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{make_corpus, ref_rankquant_asymmetric, D, N};

fn ref_rankquant_eval_norm(dim: usize, bits: u8) -> f32 {
    let mut acc = 0.0f32;
    for rank in 0..dim {
        let b = rank_to_bucket(rank as u16, dim, bits);
        let c = bucket_centre(b, bits);
        acc += c * c;
    }
    acc.sqrt()
}

fn ref_rankquant_eval_symmetric(a: &[f32], b: &[f32], bits: u8) -> f32 {
    let dim = a.len();
    let ra = rank_transform(a);
    let rb = rank_transform(b);
    let norm = ref_rankquant_eval_norm(dim, bits);
    let inv_norm_sq = 1.0f32 / (norm * norm);
    let mut acc = 0.0f32;
    for d in 0..dim {
        let ba = rank_to_bucket(ra[d], dim, bits);
        let bb = rank_to_bucket(rb[d], dim, bits);
        acc += bucket_centre(ba, bits) * bucket_centre(bb, bits);
    }
    acc * inv_norm_sq
}

fn assert_rankquant_result_shape_and_order(
    label: &str,
    res: &SearchResults,
    nq: usize,
    k_eff: usize,
    n_vectors: usize,
) {
    assert_eq!(res.nq, nq, "{label}: wrong query count");
    assert_eq!(res.k, k_eff, "{label}: wrong effective k");
    assert_eq!(res.scores.len(), nq * k_eff, "{label}: wrong score length");
    assert_eq!(res.indices.len(), nq * k_eff, "{label}: wrong index length");

    for qi in 0..nq {
        let scores = res.scores_for_query(qi);
        let ids = res.indices_for_query(qi);
        for slot in 0..k_eff {
            let score = scores[slot];
            let id = ids[slot];
            assert!(
                score.is_finite(),
                "{label}: non-finite score at query {qi} slot {slot}",
            );
            assert!(id >= 0, "{label}: negative id at query {qi} slot {slot}");
            assert!(
                (id as usize) < n_vectors,
                "{label}: id {id} out of range for n={n_vectors}",
            );
        }
        for slot in 1..k_eff {
            let prev = (scores[slot - 1], ids[slot - 1]);
            let cur = (scores[slot], ids[slot]);
            assert!(
                cur.0.total_cmp(&prev.0).is_le(),
                "{label}: row {qi} not sorted at slots {} and {slot}",
                slot - 1,
            );
        }
    }
}

#[test]
fn rankquant_asymmetric_matches_reference_b2() {
    rankquant_asymmetric_matches_reference(2);
}

#[test]
fn rankquant_asymmetric_matches_reference_b4() {
    rankquant_asymmetric_matches_reference(4);
}

#[test]
fn rankquant_asymmetric_matches_reference_b1() {
    rankquant_asymmetric_matches_reference(1);
}

#[test]
fn rankquant_eval_search_matches_rankquant_search_for_packed_widths() {
    let corpus = make_corpus(71);
    let mut rng = ChaCha8Rng::seed_from_u64(72);
    let nq = 5;
    let queries: Vec<f32> = (0..nq * D).map(|_| rng.random_range(-1.0..1.0)).collect();

    for bits in [1u8, 2, 4] {
        let mut idx = RankQuant::new(D, bits);
        idx.add(&corpus);

        let packed = idx.search(&queries, 12);
        let eval = rankquant_eval_search(&corpus, &queries, D, bits, 12);

        assert_eq!(eval.nq, packed.nq);
        assert_eq!(eval.k, packed.k);
        assert_eq!(
            eval.indices, packed.indices,
            "eval search top-k diverged from RankQuant::search at bits={bits}",
        );
        for (slot, (&a, &b)) in eval.scores.iter().zip(&packed.scores).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "bits={bits} slot {slot}: eval score {a} vs packed score {b}",
            );
        }
    }
}

#[test]
fn rankquant_eval_search_b3_matches_scalar_reference() {
    let corpus = make_corpus(73);
    let mut rng = ChaCha8Rng::seed_from_u64(74);
    let nq = 4;
    let queries: Vec<f32> = (0..nq * D).map(|_| rng.random_range(-1.0..1.0)).collect();
    let res = rankquant_eval_search(&corpus, &queries, D, 3, 10);

    assert_eq!(res.nq, nq);
    assert_eq!(res.k, 10);
    for qi in 0..nq {
        let q = &queries[qi * D..(qi + 1) * D];
        let mut reference: Vec<(f32, i64)> = (0..N)
            .map(|di| {
                (
                    ref_rankquant_eval_symmetric(q, &corpus[di * D..(di + 1) * D], 3),
                    di as i64,
                )
            })
            .collect();
        reference.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let ref_top = &reference[..10];
        let ref_ids: Vec<i64> = ref_top.iter().map(|&(_, di)| di).collect();
        assert_eq!(
            res.indices_for_query(qi),
            ref_ids.as_slice(),
            "b=3 eval top-k ids diverged for query {qi}",
        );
        for (slot, &(s_ref, _)) in ref_top.iter().enumerate() {
            let s = res.scores_for_query(qi)[slot];
            assert!(
                (s - s_ref).abs() < 1e-6,
                "query {qi} slot {slot}: b=3 eval score {s} vs reference {s_ref}",
            );
        }
    }
}

#[test]
fn rankquant_eval_search_empty_queries_does_not_transform_corpus() {
    let corpus = make_corpus(75);
    let queries: Vec<f32> = Vec::new();
    let res = rankquant_eval_search(&corpus, &queries, D, 3, 10);

    assert_eq!(res.nq, 0);
    assert_eq!(res.k, 10);
    assert!(res.scores.is_empty());
    assert!(res.indices.is_empty());
}

#[test]
fn rankquant_hotpath_search_shapes_cover_empty_queries_and_index() {
    let dim = 64;
    let one_query = vec![0.0; dim];
    for bits in [1u8, 2, 4] {
        let empty = RankQuant::new(dim, bits);
        let res = empty.search(&one_query, usize::MAX);
        assert_rankquant_result_shape_and_order(
            &format!("empty search bits={bits}"),
            &res,
            1,
            0,
            0,
        );
        let res = empty.search_asymmetric(&one_query, usize::MAX);
        assert_rankquant_result_shape_and_order(
            &format!("empty asymmetric bits={bits}"),
            &res,
            1,
            0,
            0,
        );

        let mut idx = RankQuant::new(dim, bits);
        let docs: Vec<f32> = (0..3 * dim).map(|i| (i % 7) as f32 - 3.0).collect();
        idx.add(&docs);

        let res = idx.search(&[], 2);
        assert_rankquant_result_shape_and_order(
            &format!("empty queries search bits={bits}"),
            &res,
            0,
            2,
            3,
        );
        let res = idx.search_asymmetric(&[], 2);
        assert_rankquant_result_shape_and_order(
            &format!("empty queries asymmetric bits={bits}"),
            &res,
            0,
            2,
            3,
        );

        let queries: Vec<f32> = (0..2 * dim).map(|i| (i % 5) as f32 - 2.0).collect();
        let res = idx.search(&queries, usize::MAX);
        assert_rankquant_result_shape_and_order(
            &format!("huge k search bits={bits}"),
            &res,
            2,
            3,
            3,
        );
        let res = idx.search_asymmetric(&queries, usize::MAX);
        assert_rankquant_result_shape_and_order(
            &format!("huge k asymmetric bits={bits}"),
            &res,
            2,
            3,
            3,
        );
    }
}

#[test]
fn rankquant_hotpath_search_ties_break_by_doc_id() {
    let dim = 64;
    let n = 6;
    let docs = vec![0.0; n * dim];
    let query = vec![0.0; dim];

    for bits in [1u8, 2, 4] {
        let mut idx = RankQuant::new(dim, bits);
        idx.add(&docs);

        let res = idx.search(&query, n);
        assert_rankquant_result_shape_and_order(&format!("tie search bits={bits}"), &res, 1, n, n);
        assert_eq!(res.indices_for_query(0), &[0, 1, 2, 3, 4, 5]);

        let res = idx.search_asymmetric(&query, n);
        assert_rankquant_result_shape_and_order(
            &format!("tie asymmetric bits={bits}"),
            &res,
            1,
            n,
            n,
        );
        assert_eq!(res.indices_for_query(0), &[0, 1, 2, 3, 4, 5]);
    }
}

#[test]
fn rankquant_hotpath_search_constructor_valid_dims_keep_shapes() {
    for &(dim, bits) in &[(8usize, 1u8), (20, 2), (36, 2), (48, 4), (80, 4)] {
        let n = 5;
        let nq = 3;
        let mut rng = ChaCha8Rng::seed_from_u64(1_500 + dim as u64 * 8 + bits as u64);
        let docs: Vec<f32> = (0..n * dim).map(|_| rng.random_range(-2.0..2.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.random_range(-2.0..2.0)).collect();

        let mut idx = RankQuant::new(dim, bits);
        idx.add(&docs);
        assert_eq!(idx.len(), n);
        assert_eq!(idx.dim(), dim);
        assert_eq!(idx.bits(), bits);
        assert_eq!(idx.byte_size(), n * idx.bytes_per_vec());

        let res = idx.search(&queries, usize::MAX);
        assert_rankquant_result_shape_and_order(
            &format!("dim={dim} bits={bits} search"),
            &res,
            nq,
            n,
            n,
        );
        let res = idx.search_asymmetric(&queries, usize::MAX);
        assert_rankquant_result_shape_and_order(
            &format!("dim={dim} bits={bits} asymmetric"),
            &res,
            nq,
            n,
            n,
        );
    }
}

#[test]
fn rankquant_constructor_still_rejects_b3() {
    let err = std::panic::catch_unwind(|| RankQuant::new(D, 3));
    assert!(
        err.is_err(),
        "RankQuant::new must keep the packed-width domain"
    );
}

fn rankquant_asymmetric_matches_reference(bits: u8) {
    let corpus = make_corpus(3 + bits as u64);
    let mut idx = RankQuant::new(D, bits);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(200 + bits as u64);
    let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();

    let res = idx.search_asymmetric(&query, 10);

    let ref_scores: Vec<f32> = (0..N)
        .map(|di| {
            let doc = &corpus[di * D..(di + 1) * D];
            ref_rankquant_asymmetric(&query, doc, bits)
        })
        .collect();

    // Compare per-doc scores: every returned score must agree with the
    // reference at the corresponding index.
    for slot in 0..10 {
        let di = res.indices_for_query(0)[slot] as usize;
        let s_idx = res.scores_for_query(0)[slot];
        let s_ref = ref_scores[di];
        assert!(
            (s_idx - s_ref).abs() < 1e-4,
            "B={bits} slot {slot} doc {di}: {s_idx} vs {s_ref}",
        );
    }

    // This random reference check uses set equality to avoid overfitting a
    // near-tolerance boundary. Exact score-tie ordering is pinned by
    // tests/determinism_contract.rs.
    let mut ref_sorted: Vec<(usize, f32)> = ref_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    ref_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_ref: std::collections::HashSet<usize> = ref_sorted[..10].iter().map(|x| x.0).collect();
    let top_idx: std::collections::HashSet<usize> = res
        .indices_for_query(0)
        .iter()
        .map(|&i| i as usize)
        .collect();
    assert_eq!(top_idx, top_ref, "B={bits} top-10 set mismatch",);
}

#[test]
fn rankquant_b2_recovers_planted_neighbour_in_top_10() {
    // Plant a deliberately near-duplicate of one corpus doc and check
    // that asymmetric RankQuant-2 finds it in the top-10 across a
    // batch of queries.
    let mut corpus = make_corpus(42);
    let mut idx = RankQuant::new(D, 2);

    // For each of 50 queries, pick a corpus doc and add small noise to
    // produce the query. Top-1 should be the corpus doc itself.
    let n_q = 50;
    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    let mut queries = Vec::with_capacity(n_q * D);
    let mut planted = Vec::with_capacity(n_q);
    for _ in 0..n_q {
        let target = rng.random_range(0..N);
        planted.push(target);
        let src = &corpus[target * D..(target + 1) * D];
        for &v in src.iter() {
            queries.push(v + rng.random_range(-0.05..0.05));
        }
    }
    // Re-encode the corpus *after* sampling targets so the targets are
    // present in the index.
    let _ = &mut corpus;
    idx.add(&corpus);

    let res = idx.search_asymmetric(&queries, 10);

    let mut hits = 0;
    for (qi, &target) in planted.iter().enumerate() {
        let top: Vec<usize> = res
            .indices_for_query(qi)
            .iter()
            .map(|&i| i as usize)
            .collect();
        if top.contains(&target) {
            hits += 1;
        }
    }
    let recall = hits as f32 / n_q as f32;
    assert!(
        recall >= 0.95,
        "RankQuant-2 recall@10 too low: {recall} (expected >= 0.95)",
    );
}

#[test]
fn rankquant_swap_remove_keeps_state_consistent() {
    let corpus = make_corpus(11);
    let mut idx = RankQuant::new(D, 2);
    idx.add(&corpus);
    assert_eq!(idx.len(), N);
    let bpv = idx.bytes_per_vec();
    let moved = idx.swap_remove(0);
    assert_eq!(moved, N - 1);
    assert_eq!(idx.len(), N - 1);
    assert_eq!(idx.byte_size(), (N - 1) * bpv);
}

#[test]
fn rank_io_round_trip_rankquant_index() {
    let corpus = make_corpus(41);
    let mut idx = RankQuant::new(D, 2);
    idx.add(&corpus);
    let tmp = std::env::temp_dir().join("rankquant_index_io.tvrq");
    idx.write(&tmp).expect("write");
    let loaded = RankQuant::load(&tmp).expect("load");
    std::fs::remove_file(&tmp).ok();

    assert_eq!(loaded.len(), idx.len());
    assert_eq!(loaded.dim(), idx.dim());
    assert_eq!(loaded.bits(), idx.bits());

    let mut rng = ChaCha8Rng::seed_from_u64(141);
    let q: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
    let r1 = idx.search_asymmetric(&q, 10);
    let r2 = loaded.search_asymmetric(&q, 10);
    assert_eq!(r1.indices_for_query(0), r2.indices_for_query(0));
}

#[test]
fn rankquant_asymmetric_correct_on_simd_invalid_dims() {
    // Constructor-valid dims that are NOT multiples of the SIMD lane
    // widths, so `select_simd_tier` must route them away from a kernel
    // that would otherwise drop the trailing chunk. (48,4)/(80,2) fall
    // to AVX2; (20,2)/(36,2) fall all the way to the scalar LUT
    // (dim % 64 != 0 and dim % 16 != 0). Each must still agree with the
    // scalar reference — this is the regression guard for the dispatch
    // logic that exists precisely to protect these dimensions.
    for &(dim, bits) in &[(48usize, 4u8), (80, 2), (20, 2), (36, 2)] {
        let n = 40usize;
        let mut rng = ChaCha8Rng::seed_from_u64(900 + dim as u64 * 8 + bits as u64);
        let corpus: Vec<f32> = (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut idx = RankQuant::new(dim, bits);
        idx.add(&corpus);

        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let res = idx.search_asymmetric(&query, 10);

        let ref_scores: Vec<f32> = (0..n)
            .map(|di| ref_rankquant_asymmetric(&query, &corpus[di * dim..(di + 1) * dim], bits))
            .collect();
        for slot in 0..10 {
            let di = res.indices_for_query(0)[slot] as usize;
            let s = res.scores_for_query(0)[slot];
            assert!(
                (s - ref_scores[di]).abs() < 1e-4,
                "dim={dim} bits={bits} slot {slot} doc {di}: {s} vs {}",
                ref_scores[di],
            );
        }
    }
}
