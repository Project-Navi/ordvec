//! Cross-API integration: the codes [`BucketCode::from_vector`] produces are
//! exactly what the stateless dense-code contingency surface (`Contingency::new`,
//! issue #219) consumes.
//!
//! `Contingency` lands in ordvec via the sibling #219 PR (it is not yet in this
//! branch's tree). Rather than depend on an unmerged module, this test pins the
//! *contract* `Contingency::new` enforces — every code is a valid bucket id
//! `< nb`, and `Contingency::new(codes, codes, nb)` over a self-pair has full
//! diagonal agreement (`diagonal_agreement() == dim`, off-diagonal cells empty).
//! It reproduces the exact `O(dim)` histogram `Contingency::new` builds, so when
//! #219 merges, swapping this reference for the real `Contingency::new` is a
//! mechanical change. The acceptance property (#220 ⇄ #219) is the same:
//! `from_vector` output is a valid, balanced contingency input.

#![cfg(feature = "experimental")]

use ordvec::bucket_code::BucketCode;

/// Reference re-implementation of `Contingency::new`'s histogram pass and the
/// `diagonal_agreement` projection (verbatim algebra from #219's
/// `contingency.rs`). Returns `Err(bad_code)` on the first out-of-range code —
/// exactly the rejection `Contingency::new` performs — else the trace of the
/// `nb × nb` table.
fn contingency_diagonal_agreement(q: &[u8], d: &[u8], nb: usize) -> Result<u32, u8> {
    assert_eq!(q.len(), d.len(), "query and doc must share dim");
    assert!(nb > 0, "nb must be > 0");
    let cap = nb as u32;
    let mut counts = vec![0u32; nb * nb];
    for (&qb, &db) in q.iter().zip(d.iter()) {
        if qb as u32 >= cap {
            return Err(qb);
        }
        if db as u32 >= cap {
            return Err(db);
        }
        counts[qb as usize * nb + db as usize] += 1;
    }
    Ok((0..nb).map(|b| counts[b * nb + b]).sum())
}

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

    // Self-pair: Contingency::new(codes, codes, nb) puts every coordinate on the
    // diagonal, so diagonal_agreement == dim and nothing falls off-diagonal.
    let diag = contingency_diagonal_agreement(codes, codes, nb).unwrap();
    assert_eq!(
        diag as usize, dim,
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
        // Cross-pair against a reversed copy: still a valid Contingency input
        // (every code < nb), so the histogram build never hits the range guard.
        let mut rev: Vec<u8> = code.codes().to_vec();
        rev.reverse();
        let diag = contingency_diagonal_agreement(code.codes(), &rev, nb)
            .expect("from_vector codes must be valid contingency input");
        // Diagonal agreement is bounded by dim and well-defined (sanity).
        assert!((diag as usize) <= dim);
    }
}
