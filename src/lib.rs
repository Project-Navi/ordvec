//! Training-free ordinal & sign quantization for vector retrieval.
//!
//! `ordvec` is a training-free ordinal/sign retrieval substrate. It was
//! developed using the early
//! [turbovec](https://github.com/RyanCodrai/turbovec) project context as a
//! rapid-development scaffold, with thanks to that lineage; ordvec's
//! implementation history and active upstream live in this repository. It
//! carries no system dependencies — no BLAS, no `ndarray`, no `faer` — and
//! needs no training, rotation, or codebook. Norms are analytical.
//!
//! Four substrate families, all data-oblivious:
//!
//! - [`Rank`] stores full-precision rank vectors (`u16` per
//!   coordinate, `2 * dim` bytes per document).
//! - [`RankQuant`] buckets each rank into `1 << bits` equal-width
//!   bins and packs `bits` bits per coordinate (`dim * bits / 8` bytes
//!   per document). `bits ∈ {1, 2, 4}` are the stable retrieval widths;
//!   `b = 8` is a capability-gated evidence/refinement width — asymmetric
//!   scoring and code/projection generation at any dim, *analytical-norm*
//!   symmetric scoring (via [`RankQuant::search`]) only when
//!   `dim % 256 == 0` (see [`RankQuant::new_asymmetric`]). The standalone
//!   [`rankquant_eval_search`] computes its norm *empirically*, so it scores
//!   any `bits ∈ 1..=8` at any dim (including `b = 8` off the 256 grid) and
//!   carries no such restriction.
//! - [`Bitmap`] stores a top-bucket bitmap per document (one bit
//!   per coordinate) and scores via `popcount(Q AND D)`.
//! - [`SignBitmap`] stores a sign bitmap per document (one bit per
//!   coordinate, set when the coordinate is positive) for sign-cosine
//!   candidate generation.
//!
//! For b=2 specifically, [`RankQuantFastscan`] is a specialized companion to
//! [`RankQuant`] — a block-32 FastScan kernel (nibble LUT; AVX-512 → scalar
//! dispatch) for absolute-minimum stage-1 scan latency, trading 2× the
//! b=2 storage and 8-bit LUT scoring noise. Reach for it only when scan latency
//! is the binding constraint.
//!
//! These four families are the headline retrieval surface, with
//! [`RankQuantFastscan`] as the specialized b=2 latency companion above. The
//! `experimental`
//! `MultiBucketBitmap` indexed contingency / projection API is a niche
//! research/analysis substrate for the bilinear bucket-overlap decomposition —
//! it is **not** a default single-score retrieval path and was never
//! kernel-optimized for that role. For primary nearest-neighbour retrieval use
//! [`RankQuant`], [`Bitmap`], or the two-stage candidate-generation → rerank
//! flow instead.
//!
//! The [`Bitmap`] candidate score is the implementation surface with the
//! strongest formal story: in the companion Lean formalization, literal
//! constant-weight bitmap overlap is the query-preserving quotient statistic,
//! an overlap threshold is Bayes-optimal under an explicit finite
//! monotone-overlap signal model, and the idealized uniform constant-weight null
//! calibrates that threshold by the hypergeometric upper tail. This is a finite
//! in-model theorem, not a claim that real encoders automatically satisfy the
//! quotient, symmetry, or null assumptions.
//!
//! ```
//! use ordvec::RankQuant;
//!
//! let documents = [
//!     8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
//!     1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
//!     8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0,
//! ];
//! let query = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
//!
//! let mut index = RankQuant::new(8, 1);
//! index.add(&documents);
//! let results = index.search_asymmetric(&query, 1);
//! assert_eq!(results.indices_for_query(0)[0], 0);
//! ```

// Every unsafe operation in the crate must sit inside an explicit `unsafe {}`
// block rather than leaning on an enclosing `unsafe fn`. This keeps the unsafe
// surface of the SIMD kernels (fastscan / bitmap / sign_bitmap / quant_kernels,
// plus the NEON popcount in util) visible to every future edit
// (THREAT_MODEL.md, THREAT-SIMD-001).
#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt;

mod bitmap;
/// Index-free, fixed-composition ordinal bucket codes (issue #220).
#[cfg(feature = "experimental")]
pub mod bucket_code;
/// Constant-weight bitmap overlap + the finite constant-weight null (issue #222).
#[cfg(feature = "experimental")]
pub mod const_weight_bitmap;
#[cfg(feature = "experimental")]
mod contingency;
mod fastscan;
#[doc(hidden)]
pub mod format;
#[cfg(feature = "experimental")]
mod multi_bucket;
mod quant;
mod quant_kernels;
/// Rank math primitives and the [`Rank`] index type.
pub mod rank;
pub mod rank_io;
pub mod sign_bitmap;
mod util;

pub use bitmap::Bitmap;
#[doc(hidden)]
pub use format::{
    FfiLoadSupport, FormatSpec, ManifestCoverage, PersistedFormat, ProbeCoverage, FORMATS,
};
pub use quant::SubsetScratch;
pub use quant::{rankquant_eval_search, RankQuant, RankQuantCapability, TwoStageCandidatePolicy};
pub use rank::Rank;
pub use rank_io::{probe_index_metadata, IndexKind, IndexMetadata, IndexParams};
pub use sign_bitmap::CandidateBatch;
pub use sign_bitmap::SignBitmap;

// Bench-only scoring reference for `examples/bench_rank` and parity tests.
// Gated off the default public API surface; production callers use
// `RankQuant::search_asymmetric`.
#[cfg(feature = "bench-utils")]
#[doc(hidden)]
pub use quant::search_asymmetric_byte_lut;

// `subset_rerank_uses_simd` is a test-only dispatch probe used by the crate's
// own SIMD-parity tests. Gated behind the non-default `test-utils` feature and
// excluded from semver guarantees — not a supported downstream API.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use quant::subset_rerank_uses_simd;

// `MultiBucketBitmap` underwrites the bilinear bucket-overlap decomposition.
//
// **`MultiBucketBitmap` is NOT the default retrieval surface.** It is a
// research/analysis primitive for the full bilinear `nb × nb` weight-matrix
// decomposition, not the constant-weight top-bucket theorem surface implemented
// by [`Bitmap`]. Its per-document storage is 2–4× larger than the corresponding
// `RankQuant` encoding; the full outer-product path does not outperform the
// equivalent per-coord scalar form and exists to expose the decomposition
// empirically and serve as a reference for truncated weight matrices.
//
// `MultiBucketBitmap` is gated behind the **non-default `experimental` cargo
// feature**, is excluded from semver guarantees, and may change or be removed
// without a major-version bump. It is not part of the stable public surface.
#[cfg(feature = "experimental")]
pub use multi_bucket::MultiBucketBitmap;

// `Contingency` / `Projection` are intended-to-stabilize stateless dense-code
// contingency-table analysis APIs added in this release (issue #219): the full
// `nb × nb` bucket-overlap table for two `&[u8]` code slices, plus its named
// projections (diagonal agreement, band agreement, top-bucket overlap, L1
// distance, etc.). This is a research/analysis primitive — it is *not* a
// retrieval index and is never wired into any search path.
//
// They remain behind the same non-default `experimental` feature as
// `MultiBucketBitmap`, so they are not yet part of the patch-stable default
// Rust surface. The stateless dense API is the intended long-term surface, but
// graduating it to a stable feature is a later compatibility decision.
#[cfg(feature = "experimental")]
pub use contingency::{Contingency, Projection};

// Index-free, fixed-composition ordinal bucket codes (issue #220). The reusable
// bucket-code surface — derive/validate per-coordinate bucket codes from a
// vector or a rank permutation with no retrieval index — behind the
// `experimental` feature. Whether it graduates to the stable surface is a
// deliberate later decision.
#[cfg(feature = "experimental")]
pub use bucket_code::{BucketCode, CompositionSpec, CompositionViolation, RankQuantSpec};

// Constant-weight bitmap overlap + the finite constant-weight null (issue #222).
// The ordinal-kernel evidence surface built on the #220 bucket codes: the
// top-bucket / top-group constant-weight bitmaps, their popcount overlap (routed
// through the crate's shared `util::and_popcount` primitive), and the idealized
// uniform constant-weight null that turns an observed overlap into an exact
// finite tail probability. Behind the `experimental` feature; whether it
// graduates to the stable surface is a deliberate later decision.
#[cfg(feature = "experimental")]
pub use const_weight_bitmap::{
    choose, top_group_overlap_vector, BitmapNull, ConstantWeightBitmap, PackedConstantWeightBitmap,
};

// `RankQuantFastscan` is a specialized b=2 FastScan scan path (block-32 nibble
// LUT, AVX-512 → scalar dispatch) for absolute-minimum stage-1 scan
// latency, at the cost of 2× the `RankQuant` b=2 storage and 8-bit LUT scoring
// noise. It is a stable, documented public type, but a *specialized* one — the
// headline retrieval surface is still `RankQuant` / `Bitmap` / two-stage; reach
// for FastScan only when scan latency at b=2 is the binding constraint.
pub use fastscan::RankQuantFastscan;

/// Whether the AVX-512 VPOPCNTDQ bitmap/sign scan kernels are active on this
/// CPU. `#[doc(hidden)]` — a diagnostic for tests and downstream probes, not a
/// stability surface.
///
/// The scan dispatch ([`SignBitmap`] and [`Bitmap`]) consults this and
/// **nothing else** — it takes no dimension. So once VPOPCNTDQ is present,
/// *every* `dim` (a multiple of 64) runs the kernel, including dims whose
/// 64-bit word count is not a multiple of 8 (e.g. 384, 768): those are handled
/// by a masked tail, not by falling back to the scalar path.
#[doc(hidden)]
#[must_use]
pub fn avx512vpop_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// Pre-0.2 names (the `Index` suffix was dropped in the OrdVec ontology
// rebrand). Retained as deprecated type aliases for back-compat; remove
// in a future release. `pub type` (rather than `pub use … as`) causes
// the `#[deprecated]` to actually warn at use sites.
#[deprecated(since = "0.2.0", note = "renamed to `Rank`")]
pub type RankIndex = Rank;
#[deprecated(since = "0.2.0", note = "renamed to `RankQuant`")]
pub type RankQuantIndex = RankQuant;
#[deprecated(since = "0.2.0", note = "renamed to `Bitmap`")]
pub type BitmapIndex = Bitmap;
#[deprecated(since = "0.2.0", note = "renamed to `SignBitmap`")]
pub type SignBitmapIndex = SignBitmap;
#[cfg(feature = "experimental")]
#[deprecated(since = "0.2.0", note = "renamed to `MultiBucketBitmap`")]
pub type MultiBucketBitmapIndex = MultiBucketBitmap;
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "renamed to `RankQuantFastscan`")]
pub type RankQuantFastscanIndex = RankQuantFastscan;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrdvecError {
    InvalidParameter {
        name: &'static str,
        message: String,
    },
    InvalidLength {
        name: &'static str,
        len: usize,
        dim: usize,
    },
    InvalidVectorLength {
        name: &'static str,
        len: usize,
        expected: usize,
    },
    CandidateIdOutOfRange {
        id: u32,
        n_vectors: usize,
    },
}

impl fmt::Display for OrdvecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { name, message } => {
                write!(f, "invalid {name}: {message}")
            }
            Self::InvalidLength { name, len, dim } => {
                write!(f, "{name} length {len} must be a multiple of dim {dim}")
            }
            Self::InvalidVectorLength {
                name,
                len,
                expected,
            } => {
                write!(f, "{name} length {len} must equal dim {expected}")
            }
            Self::CandidateIdOutOfRange { id, n_vectors } => {
                write!(
                    f,
                    "candidate id {id} out of range for n_vectors {n_vectors}"
                )
            }
        }
    }
}

impl std::error::Error for OrdvecError {}

pub fn validate_flat_vectors_len(len: usize, dim: usize) -> Result<usize, OrdvecError> {
    if dim == 0 {
        return Err(OrdvecError::InvalidParameter {
            name: "dim",
            message: "must be > 0".to_string(),
        });
    }
    if !len.is_multiple_of(dim) {
        return Err(OrdvecError::InvalidLength {
            name: "vectors",
            len,
            dim,
        });
    }
    Ok(len / dim)
}

pub fn validate_candidate_ids(candidates: &[u32], n_vectors: usize) -> Result<(), OrdvecError> {
    if let Some(&id) = candidates.iter().find(|&&id| (id as usize) >= n_vectors) {
        return Err(OrdvecError::CandidateIdOutOfRange { id, n_vectors });
    }
    Ok(())
}

/// Top-k search results, laid out as `nq` contiguous blocks of `k`.
///
/// `scores` and `indices` are flat row-major buffers of length `nq * k`;
/// block `qi` is `[qi * k, (qi + 1) * k)`. Use [`Self::scores_for_query`]
/// / [`Self::indices_for_query`] to slice a single query's results.
///
/// The fields are `pub` deliberately: callers (notably the Python binding)
/// move the buffers out by value for a zero-copy hand-off into the host array
/// type. Prefer the slice accessors above for read-only per-query access —
/// exposing the flat buffers as the stable representation is the trade-off for
/// that zero-copy interop.
#[must_use = "search runs the full scan to produce these results; dropping them discards that work"]
pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: usize,
    pub k: usize,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize, serde::Serialize)]
struct SearchResultsSerdeRepr {
    scores: Vec<f32>,
    indices: Vec<i64>,
    nq: u64,
    k: u64,
}

#[cfg(feature = "serde")]
impl serde::Serialize for SearchResults {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct as _;

        let mut state = serializer.serialize_struct("SearchResults", 4)?;
        state.serialize_field("scores", &self.scores)?;
        state.serialize_field("indices", &self.indices)?;
        state.serialize_field("nq", &(self.nq as u64))?;
        state.serialize_field("k", &(self.k as u64))?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SearchResults {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = <SearchResultsSerdeRepr as serde::Deserialize>::deserialize(deserializer)?;
        SearchResults::from_serde_repr(repr).map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for SearchResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchResults")
            .field("nq", &self.nq)
            .field("k", &self.k)
            .field("scores_len", &self.scores.len())
            .field("indices_len", &self.indices.len())
            .finish()
    }
}

impl SearchResults {
    #[cfg(feature = "serde")]
    fn from_serde_repr(repr: SearchResultsSerdeRepr) -> Result<Self, String> {
        let nq = usize::try_from(repr.nq)
            .map_err(|_| "SearchResults nq does not fit usize".to_string())?;
        let k = usize::try_from(repr.k)
            .map_err(|_| "SearchResults k does not fit usize".to_string())?;
        let expected_len = nq
            .checked_mul(k)
            .ok_or_else(|| "SearchResults nq * k overflows usize".to_string())?;
        if repr.scores.len() != expected_len {
            return Err(format!(
                "SearchResults scores length {} does not match nq * k {}",
                repr.scores.len(),
                expected_len
            ));
        }
        if repr.indices.len() != expected_len {
            return Err(format!(
                "SearchResults indices length {} does not match nq * k {}",
                repr.indices.len(),
                expected_len
            ));
        }
        if repr.scores.len() != repr.indices.len() {
            return Err(format!(
                "SearchResults scores length {} does not match indices length {}",
                repr.scores.len(),
                repr.indices.len()
            ));
        }
        Ok(Self {
            scores: repr.scores,
            indices: repr.indices,
            nq,
            k,
        })
    }

    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

#[cfg(all(test, feature = "serde"))]
mod search_results_serde_tests {
    use super::{SearchResults, SearchResultsSerdeRepr};
    use serde::de::{self, DeserializeSeed, IntoDeserializer, SeqAccess, Visitor};
    use serde::ser::{Impossible, SerializeSeq, SerializeStruct};
    use serde::Deserialize as _;
    use serde::Serialize as _;
    use std::fmt;

    fn repr(scores: Vec<f32>, indices: Vec<i64>, nq: u64, k: u64) -> SearchResultsSerdeRepr {
        SearchResultsSerdeRepr {
            scores,
            indices,
            nq,
            k,
        }
    }

    #[derive(Debug)]
    struct TestSerError(String);

    impl fmt::Display for TestSerError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for TestSerError {}

    impl serde::ser::Error for TestSerError {
        fn custom<T: fmt::Display>(msg: T) -> Self {
            Self(msg.to_string())
        }
    }

    macro_rules! unsupported_serializer_common {
        () => {
            fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported bool"))
            }
            fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported i8"))
            }
            fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported i16"))
            }
            fn serialize_i32(self, _v: i32) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported i32"))
            }
            fn serialize_i128(self, _v: i128) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported i128"))
            }
            fn serialize_u128(self, _v: u128) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported u128"))
            }
            fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported f64"))
            }
            fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported char"))
            }
            fn serialize_str(self, _v: &str) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported str"))
            }
            fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported bytes"))
            }
            fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported none"))
            }
            fn serialize_some<T: ?Sized + serde::Serialize>(
                self,
                _value: &T,
            ) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported some"))
            }
            fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported unit"))
            }
            fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported unit struct"))
            }
            fn serialize_unit_variant(
                self,
                _name: &'static str,
                _variant_index: u32,
                _variant: &'static str,
            ) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported unit variant"))
            }
            fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(
                self,
                _name: &'static str,
                _value: &T,
            ) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported newtype struct"))
            }
            fn serialize_newtype_variant<T: ?Sized + serde::Serialize>(
                self,
                _name: &'static str,
                _variant_index: u32,
                _variant: &'static str,
                _value: &T,
            ) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported newtype variant"))
            }
            fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
                Err(serde::ser::Error::custom("unsupported tuple"))
            }
            fn serialize_tuple_struct(
                self,
                _name: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeTupleStruct, Self::Error> {
                Err(serde::ser::Error::custom("unsupported tuple struct"))
            }
            fn serialize_tuple_variant(
                self,
                _name: &'static str,
                _variant_index: u32,
                _variant: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeTupleVariant, Self::Error> {
                Err(serde::ser::Error::custom("unsupported tuple variant"))
            }
            fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
                Err(serde::ser::Error::custom("unsupported map"))
            }
            fn serialize_struct_variant(
                self,
                _name: &'static str,
                _variant_index: u32,
                _variant: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeStructVariant, Self::Error> {
                Err(serde::ser::Error::custom("unsupported struct variant"))
            }
        };
    }

    macro_rules! unsupported_i64_serializer {
        () => {
            fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported i64"))
            }
        };
    }

    macro_rules! unsupported_f32_serializer {
        () => {
            fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported f32"))
            }
        };
    }

    macro_rules! unsupported_unsigned_serializers {
        () => {
            fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported u8"))
            }
            fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported u16"))
            }
            fn serialize_u32(self, _v: u32) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported u32"))
            }
            fn serialize_u64(self, _v: u64) -> Result<Self::Ok, Self::Error> {
                Err(serde::ser::Error::custom("unsupported u64"))
            }
        };
    }

    macro_rules! unsupported_seq_serializer {
        () => {
            fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
                Err(serde::ser::Error::custom("unsupported seq"))
            }
        };
    }

    macro_rules! unsupported_struct_serializer {
        () => {
            fn serialize_struct(
                self,
                _name: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeStruct, Self::Error> {
                Err(serde::ser::Error::custom("unsupported struct"))
            }
        };
    }

    struct SearchResultsSerializer;

    impl serde::Serializer for SearchResultsSerializer {
        type Ok = SearchResultsSerdeRepr;
        type Error = TestSerError;
        type SerializeSeq = Impossible<Self::Ok, Self::Error>;
        type SerializeTuple = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
        type SerializeMap = Impossible<Self::Ok, Self::Error>;
        type SerializeStruct = SearchResultsStructSerializer;
        type SerializeStructVariant = Impossible<Self::Ok, Self::Error>;

        fn serialize_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStruct, Self::Error> {
            Ok(SearchResultsStructSerializer::default())
        }

        unsupported_serializer_common! {}
        unsupported_i64_serializer! {}
        unsupported_f32_serializer! {}
        unsupported_unsigned_serializers! {}
        unsupported_seq_serializer! {}
    }

    #[derive(Default)]
    struct SearchResultsStructSerializer {
        scores: Option<Vec<f32>>,
        indices: Option<Vec<i64>>,
        nq: Option<u64>,
        k: Option<u64>,
    }

    impl SerializeStruct for SearchResultsStructSerializer {
        type Ok = SearchResultsSerdeRepr;
        type Error = TestSerError;

        fn serialize_field<T: ?Sized + serde::Serialize>(
            &mut self,
            key: &'static str,
            value: &T,
        ) -> Result<(), Self::Error> {
            match key {
                "scores" => self.scores = Some(value.serialize(F32VecSerializer)?),
                "indices" => self.indices = Some(value.serialize(I64VecSerializer)?),
                "nq" => self.nq = Some(value.serialize(U64Serializer)?),
                "k" => self.k = Some(value.serialize(U64Serializer)?),
                _ => {
                    return Err(serde::ser::Error::custom(format!(
                        "unexpected SearchResults field {key}"
                    )));
                }
            }
            Ok(())
        }

        fn end(self) -> Result<Self::Ok, Self::Error> {
            Ok(SearchResultsSerdeRepr {
                scores: self
                    .scores
                    .ok_or_else(|| serde::ser::Error::custom("missing scores"))?,
                indices: self
                    .indices
                    .ok_or_else(|| serde::ser::Error::custom("missing indices"))?,
                nq: self
                    .nq
                    .ok_or_else(|| serde::ser::Error::custom("missing nq"))?,
                k: self
                    .k
                    .ok_or_else(|| serde::ser::Error::custom("missing k"))?,
            })
        }
    }

    struct F32VecSerializer;
    struct I64VecSerializer;
    struct U64Serializer;
    struct F32ValueSerializer;
    struct I64ValueSerializer;

    impl serde::Serializer for F32VecSerializer {
        type Ok = Vec<f32>;
        type Error = TestSerError;
        type SerializeSeq = F32SeqSerializer;
        type SerializeTuple = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
        type SerializeMap = Impossible<Self::Ok, Self::Error>;
        type SerializeStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeStructVariant = Impossible<Self::Ok, Self::Error>;

        fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
            Ok(F32SeqSerializer {
                values: Vec::with_capacity(len.unwrap_or(0)),
            })
        }

        unsupported_serializer_common! {}
        unsupported_i64_serializer! {}
        unsupported_f32_serializer! {}
        unsupported_unsigned_serializers! {}
        unsupported_struct_serializer! {}
    }

    impl serde::Serializer for I64VecSerializer {
        type Ok = Vec<i64>;
        type Error = TestSerError;
        type SerializeSeq = I64SeqSerializer;
        type SerializeTuple = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
        type SerializeMap = Impossible<Self::Ok, Self::Error>;
        type SerializeStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeStructVariant = Impossible<Self::Ok, Self::Error>;

        fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
            Ok(I64SeqSerializer {
                values: Vec::with_capacity(len.unwrap_or(0)),
            })
        }

        unsupported_serializer_common! {}
        unsupported_i64_serializer! {}
        unsupported_f32_serializer! {}
        unsupported_unsigned_serializers! {}
        unsupported_struct_serializer! {}
    }

    struct F32SeqSerializer {
        values: Vec<f32>,
    }

    impl SerializeSeq for F32SeqSerializer {
        type Ok = Vec<f32>;
        type Error = TestSerError;

        fn serialize_element<T: ?Sized + serde::Serialize>(
            &mut self,
            value: &T,
        ) -> Result<(), Self::Error> {
            self.values.push(value.serialize(F32ValueSerializer)?);
            Ok(())
        }

        fn end(self) -> Result<Self::Ok, Self::Error> {
            Ok(self.values)
        }
    }

    struct I64SeqSerializer {
        values: Vec<i64>,
    }

    impl SerializeSeq for I64SeqSerializer {
        type Ok = Vec<i64>;
        type Error = TestSerError;

        fn serialize_element<T: ?Sized + serde::Serialize>(
            &mut self,
            value: &T,
        ) -> Result<(), Self::Error> {
            self.values.push(value.serialize(I64ValueSerializer)?);
            Ok(())
        }

        fn end(self) -> Result<Self::Ok, Self::Error> {
            Ok(self.values)
        }
    }

    impl serde::Serializer for F32ValueSerializer {
        type Ok = f32;
        type Error = TestSerError;
        type SerializeSeq = Impossible<Self::Ok, Self::Error>;
        type SerializeTuple = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
        type SerializeMap = Impossible<Self::Ok, Self::Error>;
        type SerializeStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeStructVariant = Impossible<Self::Ok, Self::Error>;

        fn serialize_f32(self, value: f32) -> Result<Self::Ok, Self::Error> {
            Ok(value)
        }

        unsupported_serializer_common! {}
        unsupported_i64_serializer! {}
        unsupported_unsigned_serializers! {}
        unsupported_seq_serializer! {}
        unsupported_struct_serializer! {}
    }

    impl serde::Serializer for I64ValueSerializer {
        type Ok = i64;
        type Error = TestSerError;
        type SerializeSeq = Impossible<Self::Ok, Self::Error>;
        type SerializeTuple = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
        type SerializeMap = Impossible<Self::Ok, Self::Error>;
        type SerializeStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeStructVariant = Impossible<Self::Ok, Self::Error>;

        fn serialize_i64(self, value: i64) -> Result<Self::Ok, Self::Error> {
            Ok(value)
        }

        unsupported_serializer_common! {}
        unsupported_f32_serializer! {}
        unsupported_unsigned_serializers! {}
        unsupported_seq_serializer! {}
        unsupported_struct_serializer! {}
    }

    impl serde::Serializer for U64Serializer {
        type Ok = u64;
        type Error = TestSerError;
        type SerializeSeq = Impossible<Self::Ok, Self::Error>;
        type SerializeTuple = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
        type SerializeMap = Impossible<Self::Ok, Self::Error>;
        type SerializeStruct = Impossible<Self::Ok, Self::Error>;
        type SerializeStructVariant = Impossible<Self::Ok, Self::Error>;

        fn serialize_u8(self, value: u8) -> Result<Self::Ok, Self::Error> {
            Ok(u64::from(value))
        }

        fn serialize_u16(self, value: u16) -> Result<Self::Ok, Self::Error> {
            Ok(u64::from(value))
        }

        fn serialize_u32(self, value: u32) -> Result<Self::Ok, Self::Error> {
            Ok(u64::from(value))
        }

        fn serialize_u64(self, value: u64) -> Result<Self::Ok, Self::Error> {
            Ok(value)
        }

        unsupported_serializer_common! {}
        unsupported_i64_serializer! {}
        unsupported_f32_serializer! {}
        unsupported_seq_serializer! {}
        unsupported_struct_serializer! {}
    }

    enum TestValue {
        F32s(Vec<f32>),
        I64s(Vec<i64>),
        U64(u64),
    }

    struct VecSeq<T> {
        iter: std::vec::IntoIter<T>,
    }

    impl<'de, T> SeqAccess<'de> for VecSeq<T>
    where
        T: IntoDeserializer<'de, de::value::Error>,
    {
        type Error = de::value::Error;

        fn next_element_seed<S>(&mut self, seed: S) -> Result<Option<S::Value>, Self::Error>
        where
            S: DeserializeSeed<'de>,
        {
            self.iter
                .next()
                .map(|value| seed.deserialize(value.into_deserializer()))
                .transpose()
        }
    }

    impl<'de> serde::Deserializer<'de> for TestValue {
        type Error = de::value::Error;

        fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            match self {
                Self::F32s(values) => visitor.visit_seq(VecSeq {
                    iter: values.into_iter(),
                }),
                Self::I64s(values) => visitor.visit_seq(VecSeq {
                    iter: values.into_iter(),
                }),
                Self::U64(value) => visitor.visit_u64(value),
            }
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf option unit unit_struct newtype_struct seq tuple
            tuple_struct map struct enum identifier ignored_any
        }
    }

    struct ReprDeserializer {
        values: std::vec::IntoIter<TestValue>,
    }

    impl ReprDeserializer {
        fn new(repr: SearchResultsSerdeRepr) -> Self {
            Self {
                values: vec![
                    TestValue::F32s(repr.scores),
                    TestValue::I64s(repr.indices),
                    TestValue::U64(repr.nq),
                    TestValue::U64(repr.k),
                ]
                .into_iter(),
            }
        }
    }

    impl<'de> SeqAccess<'de> for ReprDeserializer {
        type Error = de::value::Error;

        fn next_element_seed<S>(&mut self, seed: S) -> Result<Option<S::Value>, Self::Error>
        where
            S: DeserializeSeed<'de>,
        {
            self.values
                .next()
                .map(|value| seed.deserialize(value))
                .transpose()
        }
    }

    impl<'de> serde::Deserializer<'de> for ReprDeserializer {
        type Error = de::value::Error;

        fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            visitor.visit_seq(self)
        }

        fn deserialize_struct<V>(
            self,
            _name: &'static str,
            _fields: &'static [&'static str],
            visitor: V,
        ) -> Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            visitor.visit_seq(self)
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf option unit unit_struct newtype_struct seq tuple
            tuple_struct map enum identifier ignored_any
        }
    }

    fn deserialize(repr: SearchResultsSerdeRepr) -> Result<SearchResults, de::value::Error> {
        SearchResults::deserialize(ReprDeserializer::new(repr))
    }

    fn roundtrip(results: &SearchResults) -> SearchResults {
        let encoded = results
            .serialize(SearchResultsSerializer)
            .expect("serialize SearchResults");
        deserialize(encoded).expect("deserialize SearchResults")
    }

    #[test]
    fn search_results_roundtrips_through_in_memory_serde_format() {
        let results = SearchResults {
            scores: vec![1.0, 0.25, -0.5, 0.0],
            indices: vec![10, 2, -1, 7],
            nq: 2,
            k: 2,
        };

        let decoded = roundtrip(&results);
        assert_eq!(decoded.scores, results.scores);
        assert_eq!(decoded.indices, results.indices);
        assert_eq!(decoded.nq, results.nq);
        assert_eq!(decoded.k, results.k);
    }

    #[test]
    fn search_results_deserialize_accepts_valid_shape() {
        let results = deserialize(repr(vec![1.0, 0.5], vec![7, 3], 1, 2)).unwrap();
        assert_eq!(results.scores_for_query(0), &[1.0, 0.5]);
        assert_eq!(results.indices_for_query(0), &[7, 3]);
    }

    #[test]
    fn search_results_deserialize_rejects_invalid_lengths() {
        let invalid = [
            repr(vec![1.0], vec![7], 1, 2),
            repr(vec![1.0, 0.5], vec![7], 1, 2),
            repr(vec![1.0], vec![7, 3], 1, 2),
        ];
        for repr in invalid {
            assert!(deserialize(repr).is_err());
        }
    }

    #[test]
    fn search_results_deserialize_rejects_shape_overflow() {
        let repr = repr(Vec::new(), Vec::new(), u64::MAX, 2);
        assert!(deserialize(repr).is_err());
    }
}
