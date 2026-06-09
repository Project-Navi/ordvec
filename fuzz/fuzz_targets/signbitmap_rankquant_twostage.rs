//! libFuzzer target for the common two-stage retrieval path:
//! `SignBitmap::top_m_candidates` candidate generation followed by
//! `RankQuant::search_asymmetric_subset` reranking over the same document
//! rows. This is the shape used by downstream retrieval products before they
//! apply host-owned metadata filters or sparse/lexical fusion.
//!
//! The fuzzer builds both indexes over one generated finite corpus, derives a
//! bounded structured shape for `(dim, bits, n_vectors, m, k)`, and feeds
//! duplicate candidate IDs into the subset path while preserving the public
//! subset API's corpus-sized candidate-budget contract. When sign candidate
//! generation returns the full corpus (`m >= n`), the target also checks that
//! subset reranking agrees with a full RankQuant search.
//!
//! Contract: no panic, abort, or out-of-bounds access on any bounded in-range
//! candidate input, subset reranking must preserve score-descending/doc-ID-
//! ascending ordering, and full-corpus candidate reranking must match full
//! RankQuant search.
#![no_main]

use libfuzzer_sys::{
    arbitrary::{Arbitrary, Result, Unstructured},
    fuzz_target,
};
use ordvec::{RankQuant, SignBitmap};

#[derive(Debug)]
struct TwoStageInput {
    dim: usize,
    bits: u8,
    n_vectors: usize,
    m: usize,
    k: usize,
    duplicate_case: u8,
    payload: Vec<u8>,
}

fn assert_rankquant_order(label: &str, scores: &[f32], ids: &[i64]) {
    assert_eq!(scores.len(), ids.len(), "{label}: score/id length mismatch");
    for slot in 1..scores.len() {
        let prev = (scores[slot - 1], ids[slot - 1]);
        let cur = (scores[slot], ids[slot]);
        let score_order = cur.0.total_cmp(&prev.0);
        assert!(
            score_order.is_lt() || score_order.is_eq(),
            "{label}: violates score-desc order at slots {} and {slot}: prev={prev:?} cur={cur:?}",
            slot - 1,
        );
        assert!(
            cur.0 != prev.0 || cur.1 >= prev.1,
            "{label}: violates id-asc tie order at slots {} and {slot}: prev={prev:?} cur={cur:?}",
            slot - 1,
        );
    }
}

impl<'a> Arbitrary<'a> for TwoStageInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dim = *u.choose(&[64usize, 128, 256, 512])?;
        let bits = *u.choose(&[1u8, 2, 4])?;
        let n_vectors = usize::from(u.int_in_range(0..=16u8)?);
        let m = match u.int_in_range(0..=3u8)? {
            0 => 0,
            1 => 1,
            2 => n_vectors,
            _ => n_vectors.saturating_add(usize::from(u.int_in_range(1..=8u8)?)),
        };
        let k = match u.int_in_range(0..=3u8)? {
            0 => 0,
            1 => 1,
            2 => m.saturating_add(1),
            _ => usize::from(u.int_in_range(0..=24u8)?),
        };
        let duplicate_case = u.int_in_range(0..=3u8)?;
        let payload_len = u.int_in_range(0..=u.len().min(1024))?;
        let payload = u.bytes(payload_len)?.to_vec();
        Ok(Self {
            dim,
            bits,
            n_vectors,
            m,
            k,
            duplicate_case,
            payload,
        })
    }
}

fuzz_target!(|input: TwoStageInput| {
    let total = (input.n_vectors + 1) * input.dim;
    let floats: Vec<f32> = (0..total)
        .map(|i| {
            if input.payload.is_empty() {
                0.0
            } else {
                input.payload[i % input.payload.len()] as f32 - 128.0
            }
        })
        .collect();
    let (vecs, query) = floats.split_at(input.n_vectors * input.dim);

    let mut sign = SignBitmap::new(input.dim);
    let mut rankquant = RankQuant::new(input.dim, input.bits);
    sign.add(vecs);
    rankquant.add(vecs);
    assert_eq!(sign.dim(), rankquant.dim());
    assert_eq!(sign.len(), rankquant.len());
    assert_eq!(sign.dim(), input.dim);
    assert_eq!(sign.len(), input.n_vectors);

    let candidates = sign.top_m_candidates(query, input.m);
    assert_eq!(candidates.len(), input.m.min(input.n_vectors));
    assert!(candidates.iter().all(|&id| (id as usize) < input.n_vectors));

    let mut subset_candidates = candidates.clone();
    if input.n_vectors > 0 {
        match input.duplicate_case {
            0 => subset_candidates.clear(),
            1 => {
                let id = subset_candidates.first().copied().unwrap_or(0);
                if subset_candidates.len() < input.n_vectors {
                    subset_candidates.push(id);
                }
            }
            2 if subset_candidates.is_empty() => subset_candidates.push(0),
            _ => {}
        }
    }
    assert!(subset_candidates.len() <= input.n_vectors);

    let (scores, ids) =
        rankquant.search_asymmetric_subset(query, &subset_candidates, input.k);
    let k_eff = input.k.min(subset_candidates.len());
    assert_eq!(scores.len(), k_eff);
    assert_eq!(ids.len(), k_eff);
    assert!(scores.iter().all(|score| score.is_finite()));
    assert_rankquant_order("subset rerank", &scores, &ids);
    for &id in &ids {
        assert!(id >= 0);
        assert!(subset_candidates.contains(&(id as u32)));
    }

    if input.n_vectors > 0 && candidates.len() == input.n_vectors {
        let full = rankquant.search_asymmetric(query, input.k);
        let (subset_scores, subset_ids) =
            rankquant.search_asymmetric_subset(query, &candidates, input.k);
        assert_eq!(subset_ids, full.indices_for_query(0));
        let full_scores = full.scores_for_query(0);
        assert_eq!(subset_scores.len(), full_scores.len());
        for (subset, full) in subset_scores.iter().zip(full_scores) {
            assert!((subset - full).abs() <= 1e-6);
        }
    }
});
