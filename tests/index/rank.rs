//! Rank (full-precision u16 ranks) integration tests.

use ordvec::Rank;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{make_corpus, ref_asymmetric, ref_rank_cosine, D, N};

#[test]
fn rank_index_symmetric_matches_reference() {
    let corpus = make_corpus(1);
    let mut idx = Rank::new(D);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();

    let res = idx.search(&query, 10);
    assert_eq!(res.nq, 1);
    assert_eq!(res.k, 10);

    // Reference: brute-force rank-cosine, then top-10.
    let mut ref_scores: Vec<(usize, f32)> = (0..N)
        .map(|di| {
            let doc = &corpus[di * D..(di + 1) * D];
            (di, ref_rank_cosine(&query, doc))
        })
        .collect();
    ref_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_ref: Vec<usize> = ref_scores[..10].iter().map(|x| x.0).collect();
    let top_idx: Vec<usize> = res
        .indices_for_query(0)
        .iter()
        .map(|&i| i as usize)
        .collect();
    assert_eq!(top_idx, top_ref, "top-10 indices must match reference");

    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for slot in 0..10 {
        let s_idx = res.scores_for_query(0)[slot];
        let s_ref = ref_scores[slot].1;
        assert!(
            (s_idx - s_ref).abs() < 1e-4,
            "score mismatch at slot {slot}: {s_idx} vs {s_ref}",
        );
    }
}

#[test]
fn rank_index_asymmetric_matches_reference() {
    let corpus = make_corpus(2);
    let mut idx = Rank::new(D);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(100);
    let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();

    let res = idx.search_asymmetric(&query, 10);

    let mut ref_scores: Vec<(usize, f32)> = (0..N)
        .map(|di| {
            let doc = &corpus[di * D..(di + 1) * D];
            (di, ref_asymmetric(&query, doc))
        })
        .collect();
    ref_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_ref: Vec<usize> = ref_scores[..10].iter().map(|x| x.0).collect();
    let top_idx: Vec<usize> = res
        .indices_for_query(0)
        .iter()
        .map(|&i| i as usize)
        .collect();
    assert_eq!(top_idx, top_ref);

    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for slot in 0..10 {
        let s_idx = res.scores_for_query(0)[slot];
        let s_ref = ref_scores[slot].1;
        assert!(
            (s_idx - s_ref).abs() < 1e-4,
            "score mismatch at slot {slot}: {s_idx} vs {s_ref}",
        );
    }
}

#[test]
fn rank_index_recall_at_10_matches_fp32() {
    // Top-10 from Rank.search should match the brute-force FP32
    // cosine baseline on >= 80% of queries (rank-cosine is *not*
    // identical to FP32 cosine, so we don't demand 100% — but on
    // smooth random data overlap should be high).
    let corpus = make_corpus(7);
    let mut idx = Rank::new(D);
    idx.add(&corpus);

    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let mut queries = Vec::with_capacity(20 * D);
    for _ in 0..(20 * D) {
        queries.push(rng.random_range(-1.0..1.0));
    }
    let res = idx.search(&queries, 10);

    // Reference top-10 by raw FP32 cosine.
    let fp32_top = |query: &[f32]| -> std::collections::HashSet<usize> {
        let mut scored: Vec<(usize, f32)> = (0..N)
            .map(|di| {
                let doc = &corpus[di * D..(di + 1) * D];
                let dot: f32 = query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
                let qn: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let dn: f32 = doc.iter().map(|x| x * x).sum::<f32>().sqrt();
                (di, dot / (qn * dn))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored[..10].iter().map(|x| x.0).collect()
    };

    let mut total_overlap = 0usize;
    for qi in 0..20 {
        let q = &queries[qi * D..(qi + 1) * D];
        let rank_top: std::collections::HashSet<usize> = res
            .indices_for_query(qi)
            .iter()
            .map(|&i| i as usize)
            .collect();
        let cos_top = fp32_top(q);
        total_overlap += rank_top.intersection(&cos_top).count();
    }
    let avg_overlap = total_overlap as f32 / (20.0 * 10.0);
    // Rank-cosine vs FP32 cosine on smooth random vectors at D=128 —
    // we expect >= 70% top-10 set overlap.
    assert!(
        avg_overlap >= 0.7,
        "rank vs FP32 top-10 overlap too low: {avg_overlap}",
    );
}

#[test]
fn rank_index_swap_remove_keeps_state_consistent() {
    let corpus = make_corpus(12);
    let mut idx = Rank::new(D);
    idx.add(&corpus);
    assert_eq!(idx.len(), N);
    let moved = idx.swap_remove(0);
    assert_eq!(moved, N - 1);
    assert_eq!(idx.len(), N - 1);
    assert_eq!(idx.byte_size(), (N - 1) * D * 2);
}

#[test]
fn rank_io_round_trip_rank_index() {
    let corpus = make_corpus(40);
    let mut idx = Rank::new(D);
    idx.add(&corpus);
    let tmp = std::env::temp_dir().join("rank_index_io.tvr");
    idx.write(&tmp).expect("write");
    let loaded = Rank::load(&tmp).expect("load");
    std::fs::remove_file(&tmp).ok();

    assert_eq!(loaded.len(), idx.len());
    assert_eq!(loaded.dim(), idx.dim());

    let mut rng = ChaCha8Rng::seed_from_u64(140);
    let q: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
    let r1 = idx.search(&q, 10);
    let r2 = loaded.search(&q, 10);
    assert_eq!(r1.indices_for_query(0), r2.indices_for_query(0));
}
