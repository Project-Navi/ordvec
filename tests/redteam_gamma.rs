//! Red-team / robustness tests for the rank-mode substrate.
//!
//! Only the `rank.rs` domain-guard cases are kept here. The earlier
//! rank-modes red-team suite also covered a quantized-index
//! constructor, id-map `add_with_ids*`, and a core `io` read path —
//! none of which exist in `ordvec` (it carries only the ordinal/sign
//! substrate), so those cases are intentionally dropped.
//!
//! * `P-B` / `P-E` — `rank::rank_to_bucket` overflow on `bits >= 32` and
//!   div-by-zero on `d == 0`.
//! * `P-D` — `rank::rankquant_bytes_per_vec` div-by-zero on `bits == 0`.

use ordvec::rank::{rank_to_bucket, rankquant_bytes_per_vec};

// ---------------------------------------------------------------------------
// P-B / P-E — rank_to_bucket overflow / div-by-zero. Must panic in release too.
// ---------------------------------------------------------------------------

#[test]
#[should_panic]
fn rank_to_bucket_large_bits_panics() {
    // Signature is `rank_to_bucket(rank, d, bits)`, so this is rank=3, d=8,
    // bits=200 — the `bits` value is what's under test. `bits >= 32` makes
    // `1u32 << bits` overflow (silently-wrong bucket in release), so the
    // function guards with `assert!(bits <= 8, "bits too large")` (b=8 is the
    // widest RankQuant width whose codes still fit a u8). bits=200 trips that
    // guard; the panic must fire in release as well as debug.
    let _ = rank_to_bucket(3, 8, 200);
}

#[test]
#[should_panic]
fn rank_to_bucket_zero_d_panics() {
    let _ = rank_to_bucket(3, 0, 2);
}

// ---------------------------------------------------------------------------
// P-D — rankquant_bytes_per_vec div-by-zero on bits == 0.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "bits must be 1,2,4")]
fn rankquant_bytes_per_vec_zero_bits_panics() {
    let _ = rankquant_bytes_per_vec(64, 0);
}
