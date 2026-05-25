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
