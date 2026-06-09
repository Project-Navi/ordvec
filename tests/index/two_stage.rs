use ordvec::{RankQuant, SignBitmap};

use crate::{make_corpus, D, N};

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
