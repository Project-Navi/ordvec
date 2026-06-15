//! Cross-API integration: the codes [`BucketCode::from_vector`] produces are
//! exactly what the stateless dense-code contingency surface
//! ([`Contingency::new`], issue #219) consumes.
//!
//! `from_vector` output is a valid, balanced contingency input: every code is a
//! valid bucket id `< nb`, and a self-pair (`Contingency::new(codes, codes, nb)`)
//! has full diagonal agreement (`diagonal_agreement() == dim`). This pins the
//! `#220 ⇄ #219` acceptance property directly against the real `Contingency`.

#![cfg(feature = "experimental")]

use ordvec::bucket_code::BucketCode;
use ordvec::Contingency;

#[test]
fn from_vector_codes_feed_contingency_self_pair_full_diagonal() {
    let dim = 1024usize;
    let bits = 2u8;
    let nb = 1usize << bits; // 4

    // An arbitrary finite embedding.
    let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 7.0).sin()).collect();
    let code = BucketCode::from_vector(dim, bits, &v).unwrap();
    let codes = code.codes();

    // Every code is a valid bucket id < nb — the range invariant
    // `Contingency::new` requires before it indexes its nb × nb table.
    assert!(codes.iter().all(|&c| (c as usize) < nb));
    assert_eq!(codes.len(), dim);

    // Self-pair: `Contingency::new(codes, codes, nb)` puts every coordinate on
    // the diagonal, so `diagonal_agreement() == dim`.
    let cont = Contingency::new(codes, codes, nb)
        .expect("from_vector codes must be a valid contingency input");
    assert_eq!(
        cont.diagonal_agreement() as usize,
        dim,
        "self-pair must agree on every coordinate"
    );
}

#[test]
fn from_vector_codes_are_in_range_for_all_supported_bits() {
    let dim = 256usize;
    let v: Vec<f32> = (0..dim).map(|i| ((i * 31 + 7) % 97) as f32).collect();
    for bits in [1u8, 2, 4] {
        let nb = 1usize << bits;
        let code = BucketCode::from_vector(dim, bits, &v).unwrap();
        // Cross-pair against a reversed copy: still a valid `Contingency` input
        // (every code < nb), so `Contingency::new` never hits its range guard.
        let mut rev: Vec<u8> = code.codes().to_vec();
        rev.reverse();
        let cont = Contingency::new(code.codes(), &rev, nb)
            .expect("from_vector codes must be a valid contingency input");
        // Diagonal agreement is bounded by dim and well-defined (sanity).
        assert!((cont.diagonal_agreement() as usize) <= dim);
    }
}
