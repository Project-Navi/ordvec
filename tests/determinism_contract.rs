use ordvec::{search_asymmetric_byte_lut, Bitmap, Rank, RankQuant, SignBitmap};

fn repeated_docs(n: usize, dim: usize, value: f32) -> Vec<f32> {
    vec![value; n * dim]
}

fn assert_ids(actual: &[i64], expected: &[i64]) {
    assert_eq!(actual, expected, "ids {actual:?} != expected {expected:?}");
}

fn assert_u32_ids(actual: &[u32], expected: &[u32]) {
    assert_eq!(actual, expected, "ids {actual:?} != expected {expected:?}");
}

#[test]
fn full_search_ties_return_lowest_row_ids() {
    const DIM: usize = 64;
    const N: usize = 8;
    let docs = repeated_docs(N, DIM, 1.0);
    let query = vec![1.0; DIM];
    let zero_query = vec![0.0; DIM];

    let mut rank = Rank::new(DIM);
    rank.add(&docs);
    assert_ids(rank.search(&query, 4).indices_for_query(0), &[0, 1, 2, 3]);
    let rank_asym = rank.search_asymmetric(&zero_query, 4);
    assert_ids(rank_asym.indices_for_query(0), &[0, 1, 2, 3]);
    assert!(rank_asym.scores_for_query(0).iter().all(|&s| s == 0.0));

    let mut rankquant = RankQuant::new(DIM, 2);
    rankquant.add(&docs);
    assert_ids(
        rankquant.search(&query, 4).indices_for_query(0),
        &[0, 1, 2, 3],
    );
    let rq_asym = rankquant.search_asymmetric(&zero_query, 4);
    assert_ids(rq_asym.indices_for_query(0), &[0, 1, 2, 3]);
    assert!(rq_asym.scores_for_query(0).iter().all(|&s| s == 0.0));

    let mut bitmap = Bitmap::new(DIM, DIM / 4);
    bitmap.add(&docs);
    let bitmap_hits = bitmap.search(&query, 4);
    assert_ids(bitmap_hits.indices_for_query(0), &[0, 1, 2, 3]);
    let bitmap_score = bitmap_hits.scores_for_query(0)[0];
    assert!(bitmap_hits
        .scores_for_query(0)
        .iter()
        .all(|&s| s == bitmap_score));
}

#[test]
fn rankquant_dispatch_matches_scalar_reference_on_ordered_ties() {
    for &dim in &[20usize, 64] {
        let docs = repeated_docs(8, dim, 1.0);
        let query = vec![0.0; dim];
        let mut index = RankQuant::new(dim, 2);
        index.add(&docs);

        let production = index.search_asymmetric(&query, 6);
        let scalar = search_asymmetric_byte_lut(&index, &query, 6);

        assert_ids(production.indices_for_query(0), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(production.indices, scalar.indices, "dim={dim}");
        assert_eq!(production.scores, scalar.scores, "dim={dim}");
    }
}

#[test]
fn rankquant_subset_ties_use_global_row_ids() {
    const DIM: usize = 64;
    let docs = repeated_docs(12, DIM, 1.0);
    let query = vec![0.0; DIM];
    let mut index = RankQuant::new(DIM, 2);
    index.add(&docs);

    let (scores, ids) = index.search_asymmetric_subset(&query, &[9, 3, 7, 1], 2);
    assert_eq!(scores, vec![0.0, 0.0]);
    assert_ids(&ids, &[1, 3]);

    let (duplicate_scores, duplicate_ids) = index.search_asymmetric_subset(&query, &[7, 8, 7], 2);
    assert_eq!(duplicate_scores, vec![0.0, 0.0]);
    assert_ids(&duplicate_ids, &[7, 7]);
}

#[test]
fn batched_subset_rerank_ties_use_global_row_ids_and_keep_duplicates() {
    const DIM: usize = 64;
    let docs = repeated_docs(12, DIM, 1.0);
    let query = vec![0.0; DIM];
    let mut index = RankQuant::new(DIM, 2);
    index.add(&docs);

    // Single tied-score row routed through the batched `_into` path. All scores
    // are equal (zero query over constant-composition docs), so the order is
    // purely (score desc, global row-id asc). Duplicate candidate ids must NOT
    // be collapsed — each is scored independently.
    let candidate_offsets = [0usize, 4];
    let candidates = [7u32, 7, 3, 7];
    let k = 4usize;
    let out_k = k.min(12);
    let mut scratch = ordvec::SubsetScratch::new();
    let mut scores = vec![0.0f32; out_k];
    let mut indices = vec![0i64; out_k];
    index.search_asymmetric_subset_batched_serial_into(
        &query,
        &candidate_offsets,
        &candidates,
        k,
        &mut scratch,
        &mut scores,
        &mut indices,
    );
    // (score desc, global id asc): id 3 sorts first, then the three duplicate 7s.
    assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0]);
    assert_ids(&indices, &[3, 7, 7, 7]);

    // The batched `_into` row must agree byte-for-byte with the single-query
    // `search_asymmetric_subset` reference on the same tied/duplicate row.
    let (ref_scores, ref_ids) = index.search_asymmetric_subset(&query, &candidates, k);
    assert_eq!(scores, ref_scores);
    assert_ids(&indices, &ref_ids);
}

#[test]
fn candidate_prefilters_preserve_order_across_single_and_batched_paths() {
    const DIM: usize = 64;
    const N: usize = 10;
    let docs = repeated_docs(N, DIM, 1.0);
    let query = vec![1.0; DIM];
    let queries = [query.clone(), query.clone()].concat();

    let mut bitmap = Bitmap::new(DIM, DIM / 4);
    bitmap.add(&docs);
    let bitmap_expected = vec![0, 1, 2, 3, 4];
    assert_u32_ids(&bitmap.top_m_candidates(&query, 5), &bitmap_expected);
    for row in bitmap.top_m_candidates_batched(&queries, 5) {
        assert_u32_ids(&row, &bitmap_expected);
    }

    let mut sign = SignBitmap::new(DIM);
    sign.add(&docs);
    let sign_expected = vec![0, 1, 2, 3, 4];
    assert_u32_ids(&sign.top_m_candidates(&query, 5), &sign_expected);
    for row in sign.top_m_candidates_batched(&queries, 5) {
        assert_u32_ids(&row, &sign_expected);
    }
}

#[test]
fn empty_and_zero_k_result_shapes_are_empty() {
    const DIM: usize = 64;
    let query = vec![1.0; DIM];

    let rank = Rank::new(DIM);
    let rank_empty = rank.search(&query, 10);
    assert_eq!(rank_empty.k, 0);
    assert!(rank_empty.scores.is_empty());
    assert!(rank_empty.indices.is_empty());

    let rankquant = RankQuant::new(DIM, 2);
    let rq_empty = rankquant.search_asymmetric(&query, 10);
    assert_eq!(rq_empty.k, 0);
    assert!(rq_empty.scores.is_empty());
    assert!(rq_empty.indices.is_empty());

    let bitmap = Bitmap::new(DIM, DIM / 4);
    let bitmap_empty = bitmap.search(&query, 10);
    assert_eq!(bitmap_empty.k, 0);
    assert!(bitmap_empty.scores.is_empty());
    assert!(bitmap_empty.indices.is_empty());

    let sign = SignBitmap::new(DIM);
    assert!(sign.top_m_candidates(&query, 10).is_empty());

    let mut nonempty = RankQuant::new(DIM, 2);
    nonempty.add(&repeated_docs(2, DIM, 1.0));
    let zero_k = nonempty.search_asymmetric(&query, 0);
    assert_eq!(zero_k.k, 0);
    assert!(zero_k.scores.is_empty());
    assert!(zero_k.indices.is_empty());
}
