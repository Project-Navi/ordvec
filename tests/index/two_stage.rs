use ordvec::{
    validate_candidate_ids, validate_flat_vectors_len, Bitmap, OrdvecError, RankQuant, SignBitmap,
    TwoStageCandidatePolicy,
};

use crate::{make_corpus, D, N};

fn corpus_after_swap_remove(corpus: &[f32], rows: usize, remove: usize) -> Vec<f32> {
    let mut expected = corpus[..rows * D].to_vec();
    let last = rows - 1;
    if remove != last {
        let src = last * D..(last + 1) * D;
        expected.copy_within(src, remove * D);
    }
    expected.truncate(last * D);
    expected
}

fn build_two_stage(bits: u8) -> (SignBitmap, RankQuant, Vec<f32>) {
    let corpus = make_corpus(15_001);
    let mut sign = SignBitmap::new(D);
    let mut rankquant = RankQuant::new(D, bits);
    sign.add(&corpus);
    rankquant.add(&corpus);
    assert_two_stage_invariants(&sign, &rankquant);
    (sign, rankquant, corpus)
}

fn assert_two_stage_invariants(sign: &SignBitmap, rankquant: &RankQuant) {
    assert_eq!(sign.dim(), rankquant.dim());
    assert_eq!(sign.len(), rankquant.len());
    assert_eq!(sign.dim(), D);
    assert_eq!(sign.len(), N);
}

fn assert_score_then_id_order(scores: &[f32], ids: &[i64]) {
    for slot in 1..scores.len() {
        let prev = (scores[slot - 1], ids[slot - 1]);
        let cur = (scores[slot], ids[slot]);
        let score_order = cur.0.total_cmp(&prev.0);
        assert!(
            score_order.is_lt() || (score_order.is_eq() && cur.1 >= prev.1),
            "results violate score-desc/doc-id-asc order at slots {} and {slot}",
            slot - 1,
        );
    }
}

#[test]
fn core_validation_helpers_report_errors_without_panicking() {
    assert!(RankQuant::validate_params(D, 2).is_ok());
    assert!(RankQuant::validate_params(D, 3).is_err());
    assert!(RankQuant::validate_params(1, 2).is_err());
    assert!(RankQuant::validate_params(D + 1, 2).is_err());

    assert!(Bitmap::validate_params(D, D / 4).is_ok());
    assert!(Bitmap::validate_params(D + 1, D / 4).is_err());
    assert!(Bitmap::validate_params(D, 0).is_err());
    assert!(Bitmap::validate_params(D, D).is_err());

    assert!(SignBitmap::validate_dim(D).is_ok());
    assert!(SignBitmap::validate_dim(0).is_err());
    assert!(SignBitmap::validate_dim(D + 1).is_err());

    assert_eq!(validate_flat_vectors_len(D * 3, D).unwrap(), 3);
    assert!(validate_flat_vectors_len(D * 3 + 1, D).is_err());
    assert!(validate_candidate_ids(&[0, 7, (N - 1) as u32], N).is_ok());
    assert!(validate_candidate_ids(&[N as u32], N).is_err());
}

#[test]
fn two_stage_candidate_policy_is_overflow_safe_and_clamped() {
    let default = TwoStageCandidatePolicy::default();
    assert_eq!(default.candidate_count(0, N), 0);
    assert_eq!(default.candidate_count(1, N), 256.min(N));
    assert_eq!(default.candidate_count(10, N), N);

    let capped = TwoStageCandidatePolicy {
        min_candidates: 4,
        k_multiplier: usize::MAX,
        max_candidates: Some(37),
    };
    assert_eq!(capped.candidate_count(usize::MAX, N), 37.min(N));
    assert_eq!(capped.candidate_count(10, 9), 9);
}

#[test]
fn sign_bitmap_swap_remove_cases_match_rebuilt_probe() {
    let corpus = make_corpus(15_010);
    let query = &make_corpus(15_011)[..D];

    for (rows, remove) in [(1usize, 0usize), (8, 0), (8, 3), (8, 7)] {
        let mut index = SignBitmap::new(D);
        index.add(&corpus[..rows * D]);
        let moved = index.swap_remove(remove);

        let expected_corpus = corpus_after_swap_remove(&corpus, rows, remove);
        let mut rebuilt = SignBitmap::new(D);
        rebuilt.add(&expected_corpus);

        assert_eq!(moved, rows - 1);
        assert_eq!(index.len(), rows - 1);
        assert_eq!(index.byte_size(), (rows - 1) * index.bytes_per_vec());
        assert_eq!(index.score_all(query), rebuilt.score_all(query));
        assert_eq!(
            index.top_m_candidates(query, rows),
            rebuilt.top_m_candidates(query, rows),
            "remove={remove} rows={rows}"
        );
    }
}

#[test]
fn sign_bitmap_write_then_load_after_swap_remove_preserves_probe() {
    let corpus = make_corpus(15_012);
    let mut index = SignBitmap::new(D);
    index.add(&corpus);
    index.swap_remove(17);

    let tmp = std::env::temp_dir().join(format!(
        "ordvec_sign_bitmap_after_swap_remove_{}.tvsb",
        std::process::id()
    ));
    index.write(&tmp).expect("write");
    let loaded = SignBitmap::load(&tmp).expect("load");
    std::fs::remove_file(&tmp).ok();

    let query = &make_corpus(15_013)[..D];
    assert_eq!(loaded.len(), index.len());
    assert_eq!(loaded.score_all(query), index.score_all(query));
    assert_eq!(
        loaded.top_m_candidates(query, 32),
        index.top_m_candidates(query, 32)
    );
}

#[test]
fn sign_rankquant_pipeline_handles_edge_candidate_and_k_shapes() {
    let (sign, rankquant, _corpus) = build_two_stage(2);
    assert_two_stage_invariants(&sign, &rankquant);
    let query = &make_corpus(15_002)[..D];

    let empty_candidates = sign.top_m_candidates(query, 0);
    assert!(empty_candidates.is_empty());
    let (scores, ids) = rankquant.search_asymmetric_subset(query, &empty_candidates, 8);
    assert!(scores.is_empty());
    assert!(ids.is_empty());

    let all_candidates = sign.top_m_candidates(query, N + 32);
    assert_eq!(all_candidates.len(), N);
    assert!(all_candidates.iter().all(|&id| (id as usize) < N));

    let (scores, ids) = rankquant.search_asymmetric_subset(query, &all_candidates, 0);
    assert!(scores.is_empty());
    assert!(ids.is_empty());

    let duplicate = all_candidates[0];
    let duplicate_candidates = vec![duplicate, duplicate, duplicate];
    let (scores, ids) = rankquant.search_asymmetric_subset(query, &duplicate_candidates, 8);
    assert_eq!(scores.len(), 3);
    assert_eq!(ids, vec![i64::from(duplicate); 3]);
    assert!(scores.iter().all(|score| score.is_finite()));

    let shortlist = &all_candidates[..3];
    let (scores, ids) = rankquant.search_asymmetric_subset(query, shortlist, 32);
    assert_eq!(scores.len(), shortlist.len());
    assert_eq!(ids.len(), shortlist.len());
    assert!(ids.iter().all(|&id| shortlist.contains(&(id as u32))));
    assert_score_then_id_order(&scores, &ids);
}

#[test]
fn rankquant_sign_probe_helper_matches_manual_candidates() {
    let (sign, rankquant, _corpus) = build_two_stage(2);
    let query = &make_corpus(15_004)[..D];
    let policy = TwoStageCandidatePolicy {
        min_candidates: 0,
        k_multiplier: 4,
        max_candidates: Some(37),
    };
    let k = 5;
    let candidates = sign.top_m_candidates(query, policy.candidate_count(k, rankquant.len()));
    let manual = rankquant.search_asymmetric_subset(query, &candidates, k);
    let helper = rankquant
        .try_search_with_sign_probe_with_policy(&sign, query, k, policy)
        .unwrap();

    assert_eq!(helper.1, manual.1);
    assert_eq!(helper.0.len(), manual.0.len());
    for (helper, manual) in helper.0.iter().zip(manual.0.iter()) {
        assert!((helper - manual).abs() <= 1e-6);
    }

    let default_try = rankquant
        .try_search_with_sign_probe(&sign, query, k)
        .unwrap();
    let default_panic = rankquant.search_with_sign_probe(&sign, query, k);
    assert_eq!(default_try.1, default_panic.1);
}

#[test]
fn rankquant_sign_probe_helper_validates_probe_and_query_shape() {
    let (sign, rankquant, _corpus) = build_two_stage(2);
    let query = &make_corpus(15_005)[..D];
    let wrong_dim = SignBitmap::new(D * 2);
    assert!(rankquant
        .try_search_with_sign_probe(&wrong_dim, query, 5)
        .is_err());

    let mut short_probe = SignBitmap::new(D);
    short_probe.add(&make_corpus(15_006)[..(N - 1) * D]);
    assert!(rankquant
        .try_search_with_sign_probe(&short_probe, query, 5)
        .is_err());

    assert!(matches!(
        rankquant
        .try_search_with_sign_probe(&sign, &query[..D - 1], 5)
        .unwrap_err(),
        OrdvecError::InvalidVectorLength {
            name: "query",
            len,
            expected: D,
        } if len == D - 1
    ));

    let mut bad_query = query.to_vec();
    bad_query[0] = f32::NAN;
    assert!(rankquant
        .try_search_with_sign_probe(&sign, &bad_query, 5)
        .is_err());
}

#[test]
fn sign_rankquant_full_candidate_set_matches_full_rankquant_search() {
    let (sign, rankquant, _corpus) = build_two_stage(4);
    assert_two_stage_invariants(&sign, &rankquant);
    let query = &make_corpus(15_003)[..D];
    let candidates = sign.top_m_candidates(query, usize::MAX);
    assert_eq!(candidates.len(), N);

    let full = rankquant.search_asymmetric(query, 16);
    let (subset_scores, subset_ids) = rankquant.search_asymmetric_subset(query, &candidates, 16);

    assert_eq!(subset_ids, full.indices_for_query(0));
    assert_eq!(subset_scores.len(), full.scores_for_query(0).len());
    assert_score_then_id_order(&subset_scores, &subset_ids);
    for (subset, full) in subset_scores.iter().zip(full.scores_for_query(0)) {
        assert!(
            (subset - full).abs() <= 1e-6,
            "subset score {subset} diverged from full score {full}"
        );
    }
}

#[test]
fn sign_rankquant_subset_orders_visible_ties_after_centre_offset() {
    let dim = 128usize;
    let n_vectors = 5usize;
    let bits = 4u8;
    let payload = [
        158u8, 158, 158, 158, 158, 158, 158, 158, 158, 158, 137, 10, 10,
    ];
    let floats: Vec<f32> = (0..((n_vectors + 1) * dim))
        .map(|i| payload[i % payload.len()] as f32 - 128.0)
        .collect();
    let (corpus, query) = floats.split_at(n_vectors * dim);

    let mut sign = SignBitmap::new(dim);
    let mut rankquant = RankQuant::new(dim, bits);
    sign.add(corpus);
    rankquant.add(corpus);

    let candidates = sign.top_m_candidates(query, n_vectors);
    assert_eq!(candidates.len(), n_vectors);

    let (scores, ids) = rankquant.search_asymmetric_subset(query, &candidates, n_vectors + 1);

    assert_eq!(scores.len(), n_vectors);
    assert_eq!(ids.len(), n_vectors);
    assert!(scores.iter().all(|score| score.is_finite()));
    assert_score_then_id_order(&scores, &ids);
}

#[test]
fn serial_csr_matches_looped_single_query_and_invariants() {
    let corpus = make_corpus(20_001);
    let mut sign = SignBitmap::new(D);
    sign.add(&corpus);
    let nq = 5usize;
    let queries = make_corpus(99)[..nq * D].to_vec();
    let m = 12usize;

    let cb = sign.top_m_candidates_batched_serial_csr(&queries, m);

    // CSR invariants.
    assert_eq!(cb.offsets.len(), nq + 1);
    assert_eq!(cb.offsets[0], 0);
    assert_eq!(*cb.offsets.last().unwrap(), cb.candidates.len());
    assert!(cb.offsets.windows(2).all(|w| w[1] >= w[0]));
    assert_eq!(cb.query_count(), nq);

    // Row-for-row parity with looped single-query top_m_candidates.
    for qi in 0..nq {
        let q = &queries[qi * D..(qi + 1) * D];
        let expected = sign.top_m_candidates(q, m);
        assert_eq!(
            cb.candidates_for_query(qi).unwrap(),
            &expected[..],
            "row {qi}"
        );
        assert_eq!(expected.len(), m.min(N));
    }
}

#[test]
fn serial_csr_edges() {
    let corpus = make_corpus(20_002);
    let mut sign = SignBitmap::new(D);
    sign.add(&corpus);
    // nq == 0
    let cb0 = sign.top_m_candidates_batched_serial_csr(&[], 8);
    assert_eq!(cb0.query_count(), 0);
    assert!(cb0.is_empty());
    assert_eq!(cb0.offsets, vec![0]);
    // m == 0 → every row empty, but 2 queries → not empty
    let q2 = make_corpus(7)[..2 * D].to_vec();
    let cb = sign.top_m_candidates_batched_serial_csr(&q2, 0);
    assert_eq!(cb.query_count(), 2);
    assert!(cb.has_no_candidates());
    assert_eq!(cb.offsets, vec![0, 0, 0]);
    // m > n clamps to n
    let cb_big = sign.top_m_candidates_batched_serial_csr(&q2, N + 100);
    assert_eq!(cb_big.candidates_for_query(0).unwrap().len(), N);
}

#[test]
#[should_panic]
fn serial_csr_rejects_ragged_queries() {
    let mut sign = SignBitmap::new(D);
    sign.add(&make_corpus(20_003));
    let ragged = vec![0.0f32; D + 1]; // not a multiple of D
    let _ = sign.top_m_candidates_batched_serial_csr(&ragged, 4);
}
