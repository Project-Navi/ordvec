//! libFuzzer target for the common two-stage retrieval path:
//! `SignBitmap::top_m_candidates` candidate generation followed by
//! `RankQuant::search_asymmetric_subset` reranking over the same document
//! rows. This is the shape used by downstream retrieval products before they
//! apply host-owned metadata filters or sparse/lexical fusion.
//!
//! The fuzzer builds both indexes over one generated finite corpus, varies
//! `m` and `k` across empty, oversized, and normal cases, and feeds duplicate
//! candidate IDs into the subset path. When sign candidate generation returns
//! the full corpus (`m >= n`), the target also checks that subset reranking
//! agrees with a full RankQuant search.
//!
//! Contract: no panic, abort, or out-of-bounds access on any in-range candidate
//! input, and full-corpus candidate reranking must match full RankQuant search.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ordvec::{RankQuant, SignBitmap};

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    const DIM: usize = 64;
    let bits: u8 = match data[0] % 3 {
        0 => 1,
        1 => 2,
        _ => 4,
    };
    let n = data[1] as usize % 17; // 0..=16 docs.
    let m = match data[2] % 4 {
        0 => 0,
        1 => 1,
        2 => n,
        _ => n.saturating_add((data[2] as usize % 8) + 1),
    };
    let k = match data[3] % 4 {
        0 => 0,
        1 => 1,
        2 => m.saturating_add(1),
        _ => data[3] as usize % (n + 8),
    };

    let payload = &data[5..];
    let total = (n + 1) * DIM;
    let floats: Vec<f32> = (0..total)
        .map(|i| {
            if payload.is_empty() {
                0.0
            } else {
                payload[i % payload.len()] as f32 - 128.0
            }
        })
        .collect();
    let (vecs, query) = floats.split_at(n * DIM);

    let mut sign = SignBitmap::new(DIM);
    let mut rankquant = RankQuant::new(DIM, bits);
    sign.add(vecs);
    rankquant.add(vecs);

    let candidates = sign.top_m_candidates(query, m);
    assert_eq!(candidates.len(), m.min(n));
    assert!(candidates.iter().all(|&id| (id as usize) < n));

    let mut subset_candidates = candidates.clone();
    if n > 0 {
        match data[4] % 4 {
            0 => subset_candidates.clear(),
            1 => {
                let id = subset_candidates.first().copied().unwrap_or(0);
                subset_candidates.push(id);
            }
            2 if subset_candidates.is_empty() => subset_candidates.push(0),
            _ => {}
        }
    }

    let (scores, ids) = rankquant.search_asymmetric_subset(query, &subset_candidates, k);
    let k_eff = k.min(subset_candidates.len());
    assert_eq!(scores.len(), k_eff);
    assert_eq!(ids.len(), k_eff);
    assert!(scores.iter().all(|score| score.is_finite()));
    assert!(scores.windows(2).all(|pair| pair[0] >= pair[1]));
    for &id in &ids {
        assert!(id >= 0);
        assert!(subset_candidates.contains(&(id as u32)));
    }

    if n > 0 && candidates.len() == n {
        let full = rankquant.search_asymmetric(query, k);
        let (subset_scores, subset_ids) =
            rankquant.search_asymmetric_subset(query, &candidates, k);
        assert_eq!(subset_ids.len(), full.indices_for_query(0).len());
        assert_eq!(subset_scores.len(), full.scores_for_query(0).len());
        let mut subset_scores_sorted = subset_scores;
        let mut full_scores_sorted = full.scores_for_query(0).to_vec();
        subset_scores_sorted.sort_by(|left, right| left.total_cmp(right));
        full_scores_sorted.sort_by(|left, right| left.total_cmp(right));
        for (subset, full) in subset_scores_sorted.iter().zip(&full_scores_sorted) {
            assert!((subset - full).abs() <= 1e-6);
        }
    }
});
