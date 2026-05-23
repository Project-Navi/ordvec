//! Smoke tests confirming that the pre-0.2 `*Index` type aliases still
//! compile and are reachable as `ordvec::*Index`. These names were
//! deprecated in 0.2.0 (the OrdVec ontology rebrand); they will be
//! removed in a future release.

#![allow(deprecated)]

#[test]
fn deprecated_index_aliases_compile() {
    // Rank(dim)
    let _ = ordvec::RankIndex::new(64);
    // RankQuant(dim, bits)
    let _ = ordvec::RankQuantIndex::new(64, 2);
    // Bitmap(dim, n_top)
    let _ = ordvec::BitmapIndex::new(64, 16);
    // SignBitmap(dim)
    let _ = ordvec::SignBitmapIndex::new(64);
}

#[cfg(feature = "experimental")]
#[test]
fn deprecated_experimental_alias_compiles() {
    // MultiBucketBitmap(dim, bits)
    let _ = ordvec::MultiBucketBitmapIndex::new(64, 2);
}
