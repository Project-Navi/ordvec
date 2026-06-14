//! Index-free, fixed-composition ordinal **bucket codes** (issue #220).
//!
//! This module exposes the reusable bucket-code surface that underpins the
//! RankQuant family, lifted out of any retrieval index. It lets a caller
//! derive and validate the per-coordinate bucket codes of a vector (or of an
//! already-computed rank permutation) **without building a corpus, a packed
//! payload, or a search structure**. The output is a plain `Vec<u8>` of bucket
//! ids in `[0, buckets)`.
//!
//! Three types model the contract:
//!
//! - [`CompositionSpec`] — a *fixed-composition* parameterisation
//!   (`dim`, `buckets`) with `dim % buckets == 0`, so every bucket receives
//!   exactly `dim / buckets` coordinates. It owns the code-validation rules:
//!   length, range, and per-bucket occupancy.
//! - [`RankQuantSpec`] — the RankQuant-shaped specialisation: `buckets`
//!   derived as `1 << bits` for `bits ∈ {1, 2, 4}`, matching the crate's
//!   [`crate::RankQuant`] bit-width domain.
//! - [`BucketCode`] — a single validated code vector against a
//!   [`CompositionSpec`], built from raw codes, from a rank permutation
//!   ([`BucketCode::from_ranks`]), or directly from a float vector
//!   ([`BucketCode::from_vector`]).
//!
//! The codes [`BucketCode::from_vector`] produces are exactly the bucket ids
//! the crate's rank primitives ([`crate::rank::rank_transform`] +
//! [`crate::rank::rank_to_bucket`]) assign, so they can be fed straight into
//! the stateless dense-code contingency surface (`Contingency::new`, issue
//! #219) without any further transform.
//!
//! Ported to reach behavioural parity with the `ordgraph` bucket-code
//! prototype; the rank math is *not* re-implemented here — it delegates to the
//! crate's shared [`crate::rank`] primitives.

use std::error::Error;
use std::fmt;

use crate::rank::{rank_to_bucket, rank_transform};

/// Fixed-composition bucket-code parameters.
///
/// A spec fixes the code length (`dim`) and the bucket count (`buckets`) and
/// requires `dim % buckets == 0`, so a well-formed code places exactly
/// `dim / buckets` coordinates in every bucket. This *constant-composition*
/// invariant is what makes the codes interchangeable across documents and is
/// the property [`Self::validate_codes`] enforces.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompositionSpec {
    dim: usize,
    buckets: usize,
    expected_per_bucket: usize,
}

impl CompositionSpec {
    /// Build a fixed-composition spec for `dim` coordinates over `buckets`
    /// buckets.
    ///
    /// # Errors
    /// - [`CompositionViolation::InvalidSpec`] if `dim == 0` or `buckets < 2`.
    /// - [`CompositionViolation::NonUniformSpec`] if `dim` is not divisible by
    ///   `buckets` (the constant-composition invariant cannot hold).
    pub fn new(dim: usize, buckets: usize) -> Result<Self, CompositionViolation> {
        if dim == 0 {
            return Err(CompositionViolation::InvalidSpec("dim must be > 0"));
        }
        if buckets < 2 {
            return Err(CompositionViolation::InvalidSpec("buckets must be >= 2"));
        }
        // Codes are stored as `u8`, so a bucket id must fit in `0..=255`. Reject
        // `buckets > 256` here rather than letting `from_ranks` silently truncate
        // a computed id via `as u8` and fail later with a misleading
        // `WrongBucketCount`.
        if buckets > u8::MAX as usize + 1 {
            return Err(CompositionViolation::InvalidSpec(
                "buckets must be <= 256 (codes are stored as u8)",
            ));
        }
        if !dim.is_multiple_of(buckets) {
            return Err(CompositionViolation::NonUniformSpec { dim, buckets });
        }
        Ok(Self {
            dim,
            buckets,
            expected_per_bucket: dim / buckets,
        })
    }

    /// Build the spec implied by a RankQuant `(dim, bits)` pairing, where
    /// `buckets == 1 << bits`. A convenience wrapper over
    /// [`RankQuantSpec::new`] for callers that only need the composition.
    pub fn rank_quant(dim: usize, bits: u8) -> Result<Self, CompositionViolation> {
        RankQuantSpec::new(dim, bits).map(|spec| spec.composition)
    }

    /// Code length the spec validates against.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of buckets `buckets`.
    pub fn buckets(&self) -> usize {
        self.buckets
    }

    /// Coordinates a well-formed code places in each bucket: `dim / buckets`.
    pub fn expected_per_bucket(&self) -> usize {
        self.expected_per_bucket
    }

    /// Per-bucket occupancy histogram of `codes`.
    ///
    /// One `O(dim)` pass tallying how many coordinates land in each bucket.
    ///
    /// # Errors
    /// - [`CompositionViolation::WrongLength`] if `codes.len() != dim`.
    /// - [`CompositionViolation::BucketOutOfRange`] on the first code `>= buckets`.
    pub fn histogram(&self, codes: &[u8]) -> Result<Vec<usize>, CompositionViolation> {
        if codes.len() != self.dim {
            return Err(CompositionViolation::WrongLength {
                expected: self.dim,
                actual: codes.len(),
            });
        }
        let mut hist = vec![0usize; self.buckets];
        for (coordinate, &bucket) in codes.iter().enumerate() {
            let bucket = bucket as usize;
            if bucket >= self.buckets {
                return Err(CompositionViolation::BucketOutOfRange {
                    coordinate,
                    bucket,
                    buckets: self.buckets,
                });
            }
            hist[bucket] += 1;
        }
        Ok(hist)
    }

    /// Validate that `codes` is a well-formed fixed-composition code: correct
    /// length, every code in range, and every bucket holding exactly
    /// `expected_per_bucket` coordinates.
    ///
    /// # Errors
    /// - the [`Self::histogram`] errors (wrong length, out-of-range code), plus
    /// - [`CompositionViolation::WrongBucketCount`] on the first bucket whose
    ///   occupancy differs from `expected_per_bucket`.
    pub fn validate_codes(&self, codes: &[u8]) -> Result<(), CompositionViolation> {
        let hist = self.histogram(codes)?;
        for (bucket, &count) in hist.iter().enumerate() {
            if count != self.expected_per_bucket {
                return Err(CompositionViolation::WrongBucketCount {
                    bucket,
                    expected: self.expected_per_bucket,
                    actual: count,
                });
            }
        }
        Ok(())
    }
}

/// RankQuant-shaped fixed-composition code parameters.
///
/// Specialises [`CompositionSpec`] to the crate's RankQuant bit-width domain:
/// the bucket count is `1 << bits` for `bits ∈ {1, 2, 4}`, and `dim` is capped
/// at `u16::MAX` to mirror the crate-wide rank invariant (a rank vector is a
/// permutation of `[0, dim)` stored as `u16`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RankQuantSpec {
    bits: u8,
    composition: CompositionSpec,
}

impl RankQuantSpec {
    /// Build a RankQuant spec for `dim` coordinates at `bits` bits/coordinate.
    ///
    /// # Errors
    /// - [`CompositionViolation::InvalidBits`] if `bits ∉ {1, 2, 4}`. This is
    ///   the crate's [`crate::RankQuant`] bit-width domain — the reference
    ///   prototype also accepted `8`, but ordvec's packed format and analytical
    ///   norm are defined only for `{1, 2, 4}`, so 8-bit is rejected here.
    /// - [`CompositionViolation::DimTooLarge`] if `dim > u16::MAX`.
    /// - the [`CompositionSpec::new`] errors (non-divisible `dim`).
    pub fn new(dim: usize, bits: u8) -> Result<Self, CompositionViolation> {
        if !matches!(bits, 1 | 2 | 4) {
            return Err(CompositionViolation::InvalidBits { bits });
        }
        if dim > u16::MAX as usize {
            return Err(CompositionViolation::DimTooLarge {
                dim,
                max: u16::MAX as usize,
            });
        }
        let buckets = 1usize << bits;
        Ok(Self {
            bits,
            composition: CompositionSpec::new(dim, buckets)?,
        })
    }

    /// Bits per coordinate (`1`, `2`, or `4`).
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The underlying fixed-composition spec (`buckets == 1 << bits`).
    pub fn composition(&self) -> &CompositionSpec {
        &self.composition
    }

    /// Consume the spec, yielding the owned [`CompositionSpec`].
    pub fn into_composition(self) -> CompositionSpec {
        self.composition
    }
}

/// A single validated, fixed-composition ordinal bucket code.
///
/// Wraps a `Vec<u8>` of bucket ids together with the [`CompositionSpec`] it
/// satisfies. Every constructor validates the composition invariant up front,
/// so a constructed `BucketCode` is always well-formed: its [`Self::codes`]
/// are in range and balanced across buckets, and can be handed directly to the
/// dense-code contingency surface (`Contingency::new`, issue #219).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BucketCode {
    spec: CompositionSpec,
    codes: Vec<u8>,
}

impl BucketCode {
    /// Wrap pre-computed `codes` against `spec`, validating the composition.
    ///
    /// # Errors
    /// The [`CompositionSpec::validate_codes`] errors (wrong length,
    /// out-of-range code, wrong per-bucket occupancy).
    pub fn new(spec: CompositionSpec, codes: Vec<u8>) -> Result<Self, CompositionViolation> {
        spec.validate_codes(&codes)?;
        Ok(Self { spec, codes })
    }

    /// Derive a bucket code from an explicit rank permutation.
    ///
    /// `ranks` must be a permutation of `[0, dim)` (every value distinct and
    /// `< dim`); each rank maps to bucket `rank * buckets / dim`. This is the
    /// rank-vector entry point: the caller already holds ranks (e.g. from
    /// [`crate::rank::rank_transform`]) and wants the bucketed codes.
    ///
    /// # Errors
    /// - the [`CompositionSpec::new`] errors (bad `dim`/`buckets`).
    /// - [`CompositionViolation::WrongLength`] if `ranks.len() != dim`.
    /// - [`CompositionViolation::RankOutOfRange`] on the first `rank >= dim`.
    /// - [`CompositionViolation::DuplicateRank`] on the first repeated rank
    ///   (ranks must be a permutation).
    pub fn from_ranks(
        dim: usize,
        buckets: usize,
        ranks: &[usize],
    ) -> Result<Self, CompositionViolation> {
        let spec = CompositionSpec::new(dim, buckets)?;
        if ranks.len() != dim {
            return Err(CompositionViolation::WrongLength {
                expected: dim,
                actual: ranks.len(),
            });
        }

        let mut seen = vec![false; dim];
        let mut codes = Vec::with_capacity(dim);
        for (coordinate, &rank) in ranks.iter().enumerate() {
            if rank >= dim {
                return Err(CompositionViolation::RankOutOfRange {
                    coordinate,
                    rank,
                    dim,
                });
            }
            if seen[rank] {
                return Err(CompositionViolation::DuplicateRank { rank });
            }
            seen[rank] = true;
            // `rank < dim` and `dim % buckets == 0`, so `rank * buckets / dim`
            // lands in `[0, buckets)` and `buckets <= 256` (enforced in
            // `CompositionSpec::new`), so the result fits a `u8`. Compute the
            // product in `u64`: `rank * buckets` can exceed `usize::MAX` on
            // 32-bit / wasm32 targets for large `dim`.
            codes.push(((rank as u64 * buckets as u64) / dim as u64) as u8);
        }
        Self::new(spec, codes)
    }

    /// Derive a bucket code directly from a float vector.
    ///
    /// Computes the dimension-wise rank transform of `vector`
    /// ([`crate::rank::rank_transform`]) and buckets each rank via the crate's
    /// shared [`crate::rank::rank_to_bucket`] against the RankQuant spec for
    /// `(dim, bits)`. The resulting codes are bit-identical to what
    /// [`crate::RankQuant`] would pack for the same vector, so they feed the
    /// dense-code contingency surface (`Contingency::new`, #219) unchanged.
    ///
    /// `vector` must have length `dim` and contain only finite values; both are
    /// validated here so a malformed vector returns an error rather than
    /// panicking inside the rank primitives.
    ///
    /// # Errors
    /// - the [`RankQuantSpec::new`] errors (`bits ∉ {1, 2, 4}`, `dim` too large
    ///   or non-divisible).
    /// - [`CompositionViolation::WrongLength`] if `vector.len() != dim`.
    /// - [`CompositionViolation::NonFiniteValue`] on the first non-finite
    ///   coordinate.
    pub fn from_vector(dim: usize, bits: u8, vector: &[f32]) -> Result<Self, CompositionViolation> {
        let spec = RankQuantSpec::new(dim, bits)?;
        if vector.len() != dim {
            return Err(CompositionViolation::WrongLength {
                expected: dim,
                actual: vector.len(),
            });
        }
        // Validate finiteness up front: `rank_transform` *asserts* finiteness
        // and panics otherwise. Returning a clean error keeps the bucket-code
        // surface fail-soft on malformed input (its whole contract is
        // validation), matching the rest of this module.
        if let Some(coordinate) = vector.iter().position(|x| !x.is_finite()) {
            return Err(CompositionViolation::NonFiniteValue { coordinate });
        }
        let ranks = rank_transform(vector);
        let codes: Vec<u8> = ranks
            .iter()
            .map(|&rank| rank_to_bucket(rank, dim, bits))
            .collect();
        // The codes come straight from `rank_to_bucket` over a permutation, so
        // they already satisfy the composition invariant; route through the
        // validating constructor anyway so the guarantee is enforced in one
        // place (and any future drift in the primitives is caught).
        Self::new(spec.into_composition(), codes)
    }

    /// The composition spec these codes satisfy.
    pub fn spec(&self) -> &CompositionSpec {
        &self.spec
    }

    /// The validated bucket ids, each in `[0, buckets)`.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Top-bucket membership bitmap: `true` where the code is the highest
    /// bucket (`buckets - 1`). This is the constant-weight top-bucket indicator
    /// the [`crate::Bitmap`] candidate score is built on.
    pub fn top_bitmap(&self) -> Vec<bool> {
        let top = self.spec.buckets - 1;
        self.codes
            .iter()
            .map(|&bucket| bucket as usize == top)
            .collect()
    }
}

/// A violation of the fixed-composition bucket-code contract.
///
/// A stable, structured error type for the bucket-code surface. Distinct from
/// the crate's [`crate::OrdvecError`] (which models index/search parameter and
/// candidate errors): this enum carries the composition-specific detail
/// (duplicate ranks, per-bucket occupancy mismatches) the reference prototype's
/// tests assert on, which the flat `OrdvecError` variants cannot express
/// without losing those values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompositionViolation {
    /// A structural spec parameter was invalid (`dim == 0`, `buckets < 2`).
    InvalidSpec(&'static str),
    /// `bits` was outside the supported RankQuant set `{1, 2, 4}`.
    InvalidBits {
        /// The rejected bit width.
        bits: u8,
    },
    /// `dim` exceeded the `u16` rank-domain cap.
    DimTooLarge {
        /// The rejected dimension.
        dim: usize,
        /// The maximum supported dimension (`u16::MAX`).
        max: usize,
    },
    /// `dim` was not divisible by `buckets`, so no constant composition exists.
    NonUniformSpec {
        /// The dimension.
        dim: usize,
        /// The bucket count.
        buckets: usize,
    },
    /// A code or rank slice had the wrong length.
    WrongLength {
        /// The expected length (`dim`).
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },
    /// A code was `>= buckets`.
    BucketOutOfRange {
        /// The offending coordinate index.
        coordinate: usize,
        /// The out-of-range bucket id.
        bucket: usize,
        /// The bucket count (codes must be `< buckets`).
        buckets: usize,
    },
    /// A bucket's occupancy differed from `expected_per_bucket`.
    WrongBucketCount {
        /// The bucket whose occupancy was wrong.
        bucket: usize,
        /// The required per-bucket occupancy.
        expected: usize,
        /// The observed occupancy.
        actual: usize,
    },
    /// A rank was `>= dim`.
    RankOutOfRange {
        /// The offending coordinate index.
        coordinate: usize,
        /// The out-of-range rank.
        rank: usize,
        /// The dimension (ranks must be `< dim`).
        dim: usize,
    },
    /// A rank appeared more than once (ranks must be a permutation).
    DuplicateRank {
        /// The repeated rank.
        rank: usize,
    },
    /// A vector coordinate was non-finite (`NaN` or `±Inf`).
    NonFiniteValue {
        /// The offending coordinate index.
        coordinate: usize,
    },
}

impl fmt::Display for CompositionViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec(message) => write!(f, "{message}"),
            Self::InvalidBits { bits } => {
                write!(f, "bits {bits} is invalid; expected one of 1, 2, 4")
            }
            Self::DimTooLarge { dim, max } => write!(f, "dim {dim} exceeds maximum {max}"),
            Self::NonUniformSpec { dim, buckets } => {
                write!(f, "dim {dim} is not divisible by buckets {buckets}")
            }
            Self::WrongLength { expected, actual } => {
                write!(f, "code length {actual} does not match dim {expected}")
            }
            Self::BucketOutOfRange {
                coordinate,
                bucket,
                buckets,
            } => write!(
                f,
                "coordinate {coordinate} has bucket {bucket}, expected < {buckets}"
            ),
            Self::WrongBucketCount {
                bucket,
                expected,
                actual,
            } => write!(
                f,
                "bucket {bucket} has {actual} coordinates, expected {expected}"
            ),
            Self::RankOutOfRange {
                coordinate,
                rank,
                dim,
            } => write!(
                f,
                "coordinate {coordinate} has rank {rank}, expected < {dim}"
            ),
            Self::DuplicateRank { rank } => write!(f, "rank {rank} appears more than once"),
            Self::NonFiniteValue { coordinate } => {
                write!(f, "coordinate {coordinate} is non-finite (NaN or ±Inf)")
            }
        }
    }
}

impl Error for CompositionViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ordgraph bucket-code parity gate -------------------------------
    // Every assertion value below is reproduced verbatim from the reference
    // `code.rs` #[cfg(test)] module.

    #[test]
    fn from_ranks_builds_uniform_bucket_code() {
        let code = BucketCode::from_ranks(8, 4, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();

        assert_eq!(code.spec().expected_per_bucket(), 2);
        assert_eq!(code.codes(), &[0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn rejects_non_uniform_bucket_counts() {
        let spec = CompositionSpec::new(8, 4).unwrap();
        let err = BucketCode::new(spec, vec![0, 0, 0, 1, 2, 2, 3, 3]).unwrap_err();

        assert_eq!(
            err,
            CompositionViolation::WrongBucketCount {
                bucket: 0,
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn rejects_duplicate_ranks() {
        let err = BucketCode::from_ranks(4, 2, &[0, 1, 1, 3]).unwrap_err();

        assert_eq!(err, CompositionViolation::DuplicateRank { rank: 1 });
    }

    #[test]
    fn rankquant_spec_rejects_unsupported_bits_and_large_dims() {
        assert_eq!(
            RankQuantSpec::new(8, 3).unwrap_err(),
            CompositionViolation::InvalidBits { bits: 3 }
        );
        // `from_vector` surfaces the same unsupported-bits rejection.
        assert_eq!(
            BucketCode::from_vector(8, 3, &[0.0f32; 8]).unwrap_err(),
            CompositionViolation::InvalidBits { bits: 3 }
        );
        assert_eq!(
            RankQuantSpec::new(u16::MAX as usize + 1, 2).unwrap_err(),
            CompositionViolation::DimTooLarge {
                dim: u16::MAX as usize + 1,
                max: u16::MAX as usize,
            }
        );
    }

    #[test]
    fn composition_spec_rejects_more_than_256_buckets() {
        // Codes are u8: a bucket id must fit 0..=255.
        assert_eq!(
            CompositionSpec::new(512, 257).unwrap_err(),
            CompositionViolation::InvalidSpec("buckets must be <= 256 (codes are stored as u8)")
        );
        // 256 is the boundary and is accepted (dim a multiple of it).
        assert!(CompositionSpec::new(512, 256).is_ok());
    }

    #[test]
    fn rankquant_spec_rejects_non_divisible_dims() {
        assert_eq!(
            RankQuantSpec::new(10, 2).unwrap_err(),
            CompositionViolation::NonUniformSpec {
                dim: 10,
                buckets: 4,
            }
        );
    }

    // ---- ordvec-specific validation surface -----------------------------

    #[test]
    fn validate_codes_rejects_wrong_length() {
        let spec = CompositionSpec::new(8, 4).unwrap();
        assert_eq!(
            spec.validate_codes(&[0, 0, 1, 1, 2, 2, 3]).unwrap_err(),
            CompositionViolation::WrongLength {
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn validate_codes_rejects_out_of_range_code() {
        let spec = CompositionSpec::new(8, 4).unwrap();
        // coordinate 7 holds bucket 4, which is >= buckets (4).
        assert_eq!(
            spec.validate_codes(&[0, 0, 1, 1, 2, 2, 3, 4]).unwrap_err(),
            CompositionViolation::BucketOutOfRange {
                coordinate: 7,
                bucket: 4,
                buckets: 4,
            }
        );
    }

    #[test]
    fn composition_spec_rejects_zero_dim_and_small_buckets() {
        assert_eq!(
            CompositionSpec::new(0, 4).unwrap_err(),
            CompositionViolation::InvalidSpec("dim must be > 0")
        );
        assert_eq!(
            CompositionSpec::new(8, 1).unwrap_err(),
            CompositionViolation::InvalidSpec("buckets must be >= 2")
        );
    }

    #[test]
    fn rank_quant_helper_matches_rankquant_spec_composition() {
        let from_helper = CompositionSpec::rank_quant(16, 2).unwrap();
        let from_spec = RankQuantSpec::new(16, 2).unwrap().into_composition();
        assert_eq!(from_helper, from_spec);
        assert_eq!(from_helper.buckets(), 4);
        assert_eq!(from_helper.expected_per_bucket(), 4);
    }

    #[test]
    fn from_ranks_rejects_rank_out_of_range() {
        // rank 4 at coordinate 0 is >= dim (4).
        assert_eq!(
            BucketCode::from_ranks(4, 2, &[4, 1, 2, 3]).unwrap_err(),
            CompositionViolation::RankOutOfRange {
                coordinate: 0,
                rank: 4,
                dim: 4,
            }
        );
    }

    #[test]
    fn histogram_counts_each_bucket() {
        let spec = CompositionSpec::new(8, 4).unwrap();
        assert_eq!(
            spec.histogram(&[0, 0, 1, 1, 2, 2, 3, 3]).unwrap(),
            vec![2, 2, 2, 2]
        );
    }

    #[test]
    fn top_bitmap_marks_only_the_top_bucket() {
        let code = BucketCode::from_ranks(8, 4, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(
            code.top_bitmap(),
            vec![false, false, false, false, false, false, true, true]
        );
    }

    // ---- from_vector: ordvec primitive integration ----------------------

    #[test]
    fn from_vector_matches_from_ranks_for_sorted_input() {
        // A strictly increasing vector has ranks [0, 1, ..., dim-1], so the
        // codes must match the from_ranks path on that identity permutation.
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let code = BucketCode::from_vector(8, 2, &v).unwrap();
        assert_eq!(code.codes(), &[0, 0, 1, 1, 2, 2, 3, 3]);

        let via_ranks = BucketCode::from_ranks(8, 4, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(code.codes(), via_ranks.codes());
    }

    #[test]
    fn from_vector_buckets_are_balanced() {
        // Arbitrary finite vector: the codes must still satisfy the
        // constant-composition invariant (dim / buckets per bucket).
        let v = [3.0f32, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
        let code = BucketCode::from_vector(8, 2, &v).unwrap();
        assert_eq!(code.spec().validate_codes(code.codes()), Ok(()));
        assert_eq!(
            code.spec().histogram(code.codes()).unwrap(),
            vec![2, 2, 2, 2]
        );
    }

    #[test]
    fn from_vector_rejects_wrong_length() {
        assert_eq!(
            BucketCode::from_vector(8, 2, &[0.0, 1.0, 2.0]).unwrap_err(),
            CompositionViolation::WrongLength {
                expected: 8,
                actual: 3,
            }
        );
    }

    #[test]
    fn from_vector_rejects_non_finite() {
        let v = [0.0f32, 1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(
            BucketCode::from_vector(8, 2, &v).unwrap_err(),
            CompositionViolation::NonFiniteValue { coordinate: 2 }
        );
    }

    #[test]
    fn from_vector_rejects_invalid_bits() {
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(
            BucketCode::from_vector(8, 3, &v).unwrap_err(),
            CompositionViolation::InvalidBits { bits: 3 }
        );
    }

    #[test]
    fn display_is_stable_for_each_variant() {
        // The error type is part of the public surface; spot-check the
        // human-readable rendering does not panic and carries the detail.
        let cases = [
            CompositionViolation::InvalidSpec("dim must be > 0"),
            CompositionViolation::InvalidBits { bits: 3 },
            CompositionViolation::DimTooLarge {
                dim: 70000,
                max: 65535,
            },
            CompositionViolation::NonUniformSpec {
                dim: 10,
                buckets: 4,
            },
            CompositionViolation::WrongLength {
                expected: 8,
                actual: 7,
            },
            CompositionViolation::BucketOutOfRange {
                coordinate: 7,
                bucket: 4,
                buckets: 4,
            },
            CompositionViolation::WrongBucketCount {
                bucket: 0,
                expected: 2,
                actual: 3,
            },
            CompositionViolation::RankOutOfRange {
                coordinate: 0,
                rank: 4,
                dim: 4,
            },
            CompositionViolation::DuplicateRank { rank: 1 },
            CompositionViolation::NonFiniteValue { coordinate: 2 },
        ];
        for case in cases {
            assert!(!case.to_string().is_empty());
        }
    }
}
