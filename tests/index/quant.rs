//! RankQuant (B-bit bucket-packed) integration tests.

use ordvec::RankQuant;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{make_corpus, ref_rankquant_asymmetric, D, N};

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

fn rankquant_asymmetric_matches_reference(bits: u8) {
    let corpus = make_corpus(3 + bits as u64);
    let mut idx = RankQuant::new(D, bits);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(200 + bits as u64);
    let query: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();

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

    // And the top-10 set must match (we allow tied scores to permute
    // within ties — same set, possibly different order).
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
        let target = rng.gen_range(0..N);
        planted.push(target);
        let src = &corpus[target * D..(target + 1) * D];
        for &v in src.iter() {
            queries.push(v + rng.gen_range(-0.05..0.05));
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
    let q: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let r1 = idx.search_asymmetric(&q, 10);
    let r2 = loaded.search_asymmetric(&q, 10);
    assert_eq!(r1.indices_for_query(0), r2.indices_for_query(0));
}
