//! MultiBucketBitmap integration tests — the bilinear bucket-overlap
//! identity that underwrites the bucket decomposition.

use ordvec::rank::{bucket_centre, bucket_ranks, rank_transform};
use ordvec::MultiBucketBitmap;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{make_corpus, D, N};

/// The load-bearing structural test for the bilinear bucket-overlap
/// decomposition. With outer-product weights `W[a,b] = (a-c)(b-c)`,
/// the bilinear score `Σ_{a,b} W[a,b] · |Q_a ∩ D_b|` is algebraically
/// identical to the symmetric RankQuant per-coord score
/// `Σ_j (q_bucket[j] - c)(d_bucket[j] - c)`. This must hold to
/// floating-point epsilon, not approximately.
fn multi_bucket_bilinear_equals_symmetric_rankquant_inner(bits: u8) {
    let corpus = make_corpus(30 + bits as u64);
    let mut mb = MultiBucketBitmap::new(D, bits);
    mb.add(&corpus);
    let w = mb.outer_product_weights();

    let mut rng = ChaCha8Rng::seed_from_u64(700 + bits as u64);
    let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
    let q_bitmaps = mb.query_bitmaps_from_ranks(&query);

    let q_ranks = rank_transform(&query);
    let q_buckets = bucket_ranks(&q_ranks, bits);

    // Per-coordinate scalar reference: the symmetric RankQuant
    // un-normalised score for each doc.
    for di in 0..N {
        let doc = &corpus[di * D..(di + 1) * D];
        let d_ranks = rank_transform(doc);
        let d_buckets = bucket_ranks(&d_ranks, bits);
        let mut scalar = 0.0f32;
        for j in 0..D {
            scalar += bucket_centre(q_buckets[j], bits) * bucket_centre(d_buckets[j], bits);
        }

        let bilinear = mb.bilinear_score(&q_bitmaps, &w, di);
        let rel_err = (scalar - bilinear).abs() / scalar.abs().max(1.0);
        assert!(
            rel_err < 1e-4,
            "B={bits} doc {di}: scalar {scalar}, bilinear {bilinear}, rel_err {rel_err}",
        );
    }
}

#[test]
fn multi_bucket_bilinear_equals_symmetric_rankquant_b2() {
    multi_bucket_bilinear_equals_symmetric_rankquant_inner(2);
}

#[test]
fn multi_bucket_bilinear_equals_symmetric_rankquant_b4() {
    multi_bucket_bilinear_equals_symmetric_rankquant_inner(4);
}

#[test]
fn multi_bucket_storage_matches_formula() {
    let mb2 = MultiBucketBitmap::new(D, 2);
    let mb4 = MultiBucketBitmap::new(D, 4);
    assert_eq!(mb2.bytes_per_vec(), D * 4 / 8); // 4 buckets, dim/8 B per bitmap
    assert_eq!(mb4.bytes_per_vec(), D * 16 / 8); // 16 buckets
    assert_eq!(mb2.bytes_per_vec(), D / 2);
    assert_eq!(mb4.bytes_per_vec(), D * 2);
}

/// `top_m_bilinear` is the candidate-generation primitive — it must return the
/// same top-`m` doc IDs (descending score) a brute-force scan of `bilinear_score`
/// over every doc would, and clamp `m` to the corpus size. Also covers the
/// `m == 0` early-return path.
#[test]
fn multi_bucket_top_m_bilinear_matches_bruteforce() {
    let corpus = make_corpus(42);
    let mut mb = MultiBucketBitmap::new(D, 2);
    mb.add(&corpus);
    let w = mb.outer_product_weights();

    let mut rng = ChaCha8Rng::seed_from_u64(43);
    let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
    let q_bitmaps = mb.query_bitmaps_from_ranks(&query);

    // top_m_bilinear scores with the same kernel a brute-force scan would.
    let score = |di: u32| mb.bilinear_score(&q_bitmaps, &w, di as usize);

    let m = 10;
    let got = mb.top_m_bilinear(&q_bitmaps, &w, m);
    assert_eq!(got.len(), m);

    // Tie-robust top-m correctness: every kept doc scores at least as high as
    // every dropped doc. Bilinear scores are quantised onto the weight grid, so
    // boundary ties are real — a set-equality check against a full sort would be
    // fragile, but this boundary property holds regardless of tie-breaking.
    let kept: std::collections::HashSet<u32> = got.iter().copied().collect();
    let min_kept = got
        .iter()
        .map(|&di| score(di))
        .fold(f32::INFINITY, f32::min);
    for di in 0..N as u32 {
        if !kept.contains(&di) {
            assert!(
                score(di) <= min_kept,
                "doc {di} (score {}) was dropped but outscores the kept minimum {min_kept}",
                score(di),
            );
        }
    }

    // Result is ordered by descending score.
    let got_scores: Vec<f32> = got.iter().map(|&di| score(di)).collect();
    for pair in got_scores.windows(2) {
        assert!(pair[0] >= pair[1], "top_m_bilinear not in descending order");
    }

    // m == 0 → empty (the early-return path); m > n_vectors clamps to n_vectors.
    assert!(mb.top_m_bilinear(&q_bitmaps, &w, 0).is_empty());
    assert_eq!(mb.top_m_bilinear(&q_bitmaps, &w, N + 100).len(), N);
}

/// A truncated (diagonal-only) weight matrix exercises the `weight == 0` skip
/// branch in `bilinear_score`: off-diagonal terms are zero and must be skipped,
/// leaving `Σ_a |Q_a ∩ D_a|` — the count of coordinates the query and doc place
/// in the same bucket. (Outer-product weights are never zero, so this path needs
/// a custom matrix — and diagonal/banded weights are the documented real use.)
#[test]
fn multi_bucket_bilinear_diagonal_weights_skip_zeros() {
    let corpus = make_corpus(11);
    let mut mb = MultiBucketBitmap::new(D, 2);
    mb.add(&corpus);
    let nb = mb.n_buckets();
    let mut w = vec![0.0f32; nb * nb];
    for a in 0..nb {
        w[a * nb + a] = 1.0;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
    let q_bitmaps = mb.query_bitmaps_from_ranks(&query);

    let q_buckets = bucket_ranks(&rank_transform(&query), 2);
    for di in 0..std::cmp::min(8, N) {
        let doc = &corpus[di * D..(di + 1) * D];
        let d_buckets = bucket_ranks(&rank_transform(doc), 2);
        // Exact integer count (≤ D), representable in f32 with weight 1.0.
        let same = (0..D).filter(|&j| q_buckets[j] == d_buckets[j]).count() as f32;
        let got = mb.bilinear_score(&q_bitmaps, &w, di);
        assert_eq!(
            got, same,
            "diagonal bilinear doc {di}: got {got}, want {same}"
        );
    }
}

/// Exercise the index accessors before and after `add`.
#[test]
fn multi_bucket_accessors() {
    let mut mb = MultiBucketBitmap::new(D, 2);
    assert_eq!(mb.dim(), D);
    assert_eq!(mb.bits(), 2);
    assert_eq!(mb.n_buckets(), 4);
    assert_eq!(mb.len(), 0);
    assert!(mb.is_empty());
    assert_eq!(mb.byte_size(), 0);

    mb.add(&make_corpus(7));
    assert_eq!(mb.len(), N);
    assert!(!mb.is_empty());
    // byte_size = n_vectors * bytes_per_vec (bitmaps are u64 words).
    assert_eq!(mb.byte_size(), N * mb.bytes_per_vec());
}
