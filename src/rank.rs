//! Rank-cosine primitives: dimension-wise rank transform, mean-centred
//! bucket quantisation, packed-bit encode/decode, and analytical norms.
//!
//! These primitives back two index modes:
//! - [`crate::RankIndex`] stores full-precision rank vectors (`u16` per
//!   coordinate) and supports both symmetric (rank-vs-rank) and
//!   asymmetric (FP32-vs-rank) scoring.
//! - [`crate::RankQuantIndex`] buckets each rank into one of `1 << B`
//!   equal-width bins on `[0, D)` and packs `B` bits per coordinate,
//!   giving compact `D * B / 8` byte storage per document while
//!   preserving the rank order the encoder induced.
//!
//! No training, no rotation, no centroid lookup: the bucket index *is*
//! the value, and rank-vector norms are analytical (a permutation of
//! `{0..D-1}` has fixed L2 norm after mean-centring).
//!
//! See `tests/rank.rs` for the round-trip and norm-invariant tests.

use ordered_float::OrderedFloat;

/// Compute the dimension-wise rank transform of a single vector.
///
/// `out[k]` is the rank of `v[k]` among `v[0..d]`, with ties broken by
/// the index (stable sort). Output values are in `[0, d)`. Equivalent
/// to NumPy's `np.argsort(np.argsort(v))` for a vector of length `d`.
///
/// `d` must fit in `u16` (`d <= 65_535`); panics otherwise.
pub fn rank_transform(v: &[f32]) -> Vec<u16> {
    let d = v.len();
    assert!(d <= u16::MAX as usize, "dim must fit in u16");
    let mut order: Vec<u16> = (0..d as u16).collect();
    order.sort_by_key(|&i| OrderedFloat(v[i as usize]));
    let mut ranks = vec![0u16; d];
    for (rank, &orig_idx) in order.iter().enumerate() {
        ranks[orig_idx as usize] = rank as u16;
    }
    ranks
}

/// In-place variant: write the rank transform of `v` into `out`.
///
/// Allocates a `Vec<u16>` for the auxiliary argsort buffer.
pub fn rank_transform_into(v: &[f32], out: &mut [u16]) {
    let d = v.len();
    assert_eq!(d, out.len(), "out must have the same length as v");
    assert!(d <= u16::MAX as usize, "dim must fit in u16");
    let mut order: Vec<u16> = (0..d as u16).collect();
    order.sort_by_key(|&i| OrderedFloat(v[i as usize]));
    for (rank, &orig_idx) in order.iter().enumerate() {
        out[orig_idx as usize] = rank as u16;
    }
}

/// Bucket a single rank into one of `1 << bits` equal-width bins on
/// `[0, d)`. Returns a value in `[0, 1 << bits)`.
#[inline]
pub fn rank_to_bucket(rank: u16, d: usize, bits: u8) -> u8 {
    // `bits` is a `u8`, so a caller could pass e.g. 8 or 255. `1u32 << bits`
    // overflows for `bits >= 32` (in release that silently wraps and yields a
    // wrong bucket; in debug it panics inconsistently), and the result must
    // also fit in the returned `u8`, so cap at 7. `d == 0` would divide by
    // zero. Guard both up front so the failure is loud in every build.
    assert!(bits <= 7, "bits too large");
    assert!(d > 0, "d must be positive");
    let n_buckets = 1u32 << bits;
    let b = (rank as u32 * n_buckets) / (d as u32);
    b.min(n_buckets - 1) as u8
}

/// Bucket every entry of a full rank vector.
pub fn bucket_ranks(ranks: &[u16], bits: u8) -> Vec<u8> {
    let d = ranks.len();
    ranks.iter().map(|&r| rank_to_bucket(r, d, bits)).collect()
}

/// Pack a slice of bucket indices (each in `[0, 1 << bits)`) into a
/// dense byte stream.
///
/// Layout: the bucket with index 0 occupies the most-significant bits
/// of the first byte. Requires `bits ∈ {1, 2, 4}` and `d`'s length to
/// be a multiple of `8 / bits`.
pub fn pack_buckets(buckets: &[u8], bits: u8) -> Vec<u8> {
    assert!(matches!(bits, 1 | 2 | 4), "bits must be 1, 2, or 4");
    let codes_per_byte = (8 / bits) as usize;
    let d = buckets.len();
    assert_eq!(
        d % codes_per_byte,
        0,
        "d ({d}) must be a multiple of codes_per_byte ({codes_per_byte}) for bits = {bits}",
    );
    let n_bytes = d / codes_per_byte;
    let mut out = vec![0u8; n_bytes];
    let mask = (1u8 << bits) - 1;
    let bits_u = bits as usize;
    for (i, &b) in buckets.iter().enumerate() {
        let byte_idx = i / codes_per_byte;
        let pos = i % codes_per_byte;
        let shift = (codes_per_byte - 1 - pos) * bits_u;
        out[byte_idx] |= (b & mask) << shift;
    }
    out
}

/// Unpack a stream of `bits`-bit bucket indices into a `Vec<u8>`.
///
/// Inverse of [`pack_buckets`].
pub fn unpack_buckets(packed: &[u8], d: usize, bits: u8) -> Vec<u8> {
    assert!(matches!(bits, 1 | 2 | 4), "bits must be 1, 2, or 4");
    let codes_per_byte = (8 / bits) as usize;
    assert_eq!(packed.len() * codes_per_byte, d);
    let mask = (1u8 << bits) - 1;
    let bits_u = bits as usize;
    let mut out = vec![0u8; d];
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for i in 0..d {
        let byte_idx = i / codes_per_byte;
        let pos = i % codes_per_byte;
        let shift = (codes_per_byte - 1 - pos) * bits_u;
        out[i] = (packed[byte_idx] >> shift) & mask;
    }
    out
}

/// Number of bytes per packed RankQuant document at dimension `d` and
/// bit width `bits ∈ {1, 2, 4}`.
#[inline]
pub fn rankquant_bytes_per_vec(d: usize, bits: u8) -> usize {
    // Guard the same domain as the sibling pack/unpack helpers: `bits == 0`
    // would divide by zero computing `codes_per_byte`, and only 1/2/4 give an
    // integral codes-per-byte.
    assert!(matches!(bits, 1 | 2 | 4), "bits must be 1,2,4");
    let codes_per_byte = (8 / bits) as usize;
    d / codes_per_byte
}

/// Mean-centred value of a bucket index for a `bits`-bit RankQuant
/// scheme.
///
/// With `2^B` equal-width bins on the rank axis the bucket centres
/// after mean-centring are `b - (2^B - 1) / 2`, giving the symmetric
/// pattern `..., -1.5, -0.5, +0.5, +1.5, ...` for `B = 2`.
#[inline]
pub fn bucket_centre(bucket: u8, bits: u8) -> f32 {
    let n = (1u32 << bits) as f32;
    bucket as f32 - (n - 1.0) / 2.0
}

/// L2 norm of a mean-centred rank vector of length `d`.
///
/// A rank vector is a permutation of `{0, ..., d - 1}`, so its mean is
/// `(d - 1) / 2` and the centred coordinates have variance
/// `(d^2 - 1) / 12`. The L2 norm is therefore
/// `sqrt(d * (d^2 - 1) / 12)`.
#[inline]
pub fn rank_norm(d: usize) -> f32 {
    let d = d as f64;
    (d * (d * d - 1.0) / 12.0).sqrt() as f32
}

/// L2 norm of a mean-centred B-bit RankQuant vector of length `d`,
/// assuming each bucket receives exactly `d / 2^B` coordinates (true by
/// construction when the source is a permutation rank vector and
/// `d % 2^B == 0`).
///
/// The mean-centred bucket index has variance `(2^(2B) - 1) / 12`, so
/// the per-vector L2 norm is `sqrt(d * (2^(2B) - 1) / 12)`.
#[inline]
pub fn rankquant_norm(d: usize, bits: u8) -> f32 {
    let n = (1u32 << bits) as f64;
    let var = (n * n - 1.0) / 12.0;
    ((d as f64) * var).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_transform_matches_numpy_argsort_argsort() {
        // [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
        // argsort = [1, 3, 6, 0, 2, 4, 7, 5]
        // argsort(argsort) = [3, 0, 4, 1, 5, 7, 2, 6]
        let v = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
        let r = rank_transform(&v);
        assert_eq!(r, vec![3, 0, 4, 1, 5, 7, 2, 6]);
    }

    #[test]
    fn rank_transform_is_permutation() {
        let v: Vec<f32> = (0..256).map(|i| (i as f32 * 7.0).sin()).collect();
        let r = rank_transform(&v);
        let mut sorted = r.clone();
        sorted.sort();
        let expected: Vec<u16> = (0..256u16).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn ties_broken_by_index() {
        let v = [1.0_f32, 1.0, 1.0, 1.0];
        let r = rank_transform(&v);
        assert_eq!(r, vec![0, 1, 2, 3]);
    }

    #[test]
    fn rank_to_bucket_partitions_uniformly() {
        let d = 1024;
        let bits = 2u8;
        let mut counts = [0usize; 4];
        for rank in 0..d as u16 {
            let b = rank_to_bucket(rank, d, bits);
            counts[b as usize] += 1;
        }
        for c in counts {
            assert_eq!(c, d / 4);
        }
    }

    #[test]
    fn pack_unpack_round_trip_bits2() {
        let buckets: Vec<u8> = (0..16).map(|i| (i % 4) as u8).collect();
        let packed = pack_buckets(&buckets, 2);
        assert_eq!(packed.len(), 4);
        let unpacked = unpack_buckets(&packed, 16, 2);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn pack_unpack_round_trip_bits1() {
        let buckets: Vec<u8> = (0..16).map(|i| (i % 2) as u8).collect();
        let packed = pack_buckets(&buckets, 1);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_buckets(&packed, 16, 1);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn pack_unpack_round_trip_bits4() {
        let buckets: Vec<u8> = (0..16).map(|i| (i % 16) as u8).collect();
        let packed = pack_buckets(&buckets, 4);
        assert_eq!(packed.len(), 8);
        let unpacked = unpack_buckets(&packed, 16, 4);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn bucket_centres_are_symmetric_around_zero() {
        // For B = 2: bucket values are {-1.5, -0.5, +0.5, +1.5}.
        let centres: Vec<f32> = (0..4u8).map(|b| bucket_centre(b, 2)).collect();
        assert_eq!(centres, vec![-1.5, -0.5, 0.5, 1.5]);
        let sum: f32 = centres.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn rank_norm_matches_direct_computation() {
        let d = 1024usize;
        let analytical = rank_norm(d);
        let direct: f32 = {
            let mean = (d as f32 - 1.0) / 2.0;
            let ss: f32 = (0..d)
                .map(|i| {
                    let c = i as f32 - mean;
                    c * c
                })
                .sum();
            ss.sqrt()
        };
        assert!(
            (analytical - direct).abs() / direct < 1e-5,
            "analytical {analytical}, direct {direct}"
        );
    }

    #[test]
    fn rankquant_norm_matches_direct_computation() {
        let d = 1024usize;
        let bits = 2u8;
        let analytical = rankquant_norm(d, bits);
        // Build the bucketed vector of an ideal permutation and measure
        // its mean-centred L2 norm directly.
        let ranks: Vec<u16> = (0..d as u16).collect();
        let buckets = bucket_ranks(&ranks, bits);
        let centred: Vec<f32> = buckets.iter().map(|&b| bucket_centre(b, bits)).collect();
        let direct: f32 = centred.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (analytical - direct).abs() / direct < 1e-5,
            "analytical {analytical}, direct {direct}"
        );
    }
}
