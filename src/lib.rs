//! Training-free ordinal & sign quantization for vector retrieval.
//!
//! `ordvec` is a training-free ordinal/sign retrieval
//! substrate, developed within the
//! [turbovec](https://github.com/RyanCodrai/turbovec) project (MIT, by
//! Ryan Codrai) and factored out here as a standalone crate. It carries
//! no system dependencies — no BLAS, no `ndarray`, no `faer` — and needs
//! no training, rotation, or codebook. Norms are analytical.
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
//! The [`Bitmap`] candidate score is the implementation surface with the
//! strongest formal story: in the companion Lean formalization, literal
//! constant-weight bitmap overlap is the query-preserving quotient statistic,
//! an overlap threshold is Bayes-optimal under an explicit finite
//! monotone-overlap signal model, and the idealized uniform constant-weight null
//! calibrates that threshold by the hypergeometric upper tail. This is a finite
//! in-model theorem, not a claim that real encoders automatically satisfy the
//! quotient, symmetry, or null assumptions.
//!
//! ```no_run
//! use ordvec::{Rank, RankQuant};
//!
//! let mut idx = RankQuant::new(1024, 2);
//! let docs: Vec<f32> = vec![0.0; 1024 * 10_000];
//! idx.add(&docs);
//!
//! let queries: Vec<f32> = vec![0.0; 1024 * 4];
//! let res = idx.search_asymmetric(&queries, 10);
//! assert_eq!(res.k, 10);
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
#[cfg(feature = "experimental")]
mod contingency;
mod fastscan;
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
pub use quant::SubsetScratch;
pub use quant::{rankquant_eval_search, RankQuant, RankQuantCapability, TwoStageCandidatePolicy};
pub use rank::Rank;
pub use rank_io::{probe_index_metadata, IndexKind, IndexMetadata, IndexParams};
pub use sign_bitmap::CandidateBatch;
pub use sign_bitmap::SignBitmap;

// `search_asymmetric_byte_lut` is a bench-only scoring reference: it
// panics on b=1 and exists so `examples/bench_rank` can compare the
// byte-LUT path against the production AVX kernels on the same data.
// Re-exported `#[doc(hidden)]` — reachable for the example and the
// red-team parity tests, but not part of the headline API. Production
// callers use `RankQuant::search_asymmetric`, whose dispatch routes
// every supported bit width to a non-panicking kernel.
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

// `Contingency` / `Projection` are the **stable** stateless dense-code
// contingency-table surface added in this release (issue #219): the full
// `nb × nb` bucket-overlap table for two `&[u8]` code slices, plus its named
// projections (diagonal agreement, band agreement, top-bucket overlap, L1
// distance, etc.). This is a research/analysis primitive — it is *not* a
// retrieval index and is never wired into any search path.
//
// Although `Contingency` and `Projection` are gated behind the same
// `experimental` feature as `MultiBucketBitmap` (they complement the bilinear
// decomposition that surface exposes), they are the **stable** side of the
// `experimental` gate: the stateless dense API is the intended long-term
// surface and is covered by semver guarantees from this release forward.
// `MultiBucketBitmap` is the unstable counterpart — see the note above.
#[cfg(feature = "experimental")]
pub use contingency::{Contingency, Projection};

// Index-free, fixed-composition ordinal bucket codes (issue #220). The reusable
// bucket-code surface — derive/validate per-coordinate bucket codes from a
// vector or a rank permutation with no retrieval index — behind the
// `experimental` feature. Whether it graduates to the stable surface is a
// deliberate later decision.
#[cfg(feature = "experimental")]
pub use bucket_code::{BucketCode, CompositionSpec, CompositionViolation, RankQuantSpec};

// `RankQuantFastscan` is an optional FastScan b=2 scan path. It is
// re-exported `#[doc(hidden)]` at the crate root — reachable as
// `ordvec::RankQuantFastscan` for callers who opt in, but not
// advertised alongside the headline index types above.
#[doc(hidden)]
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

impl SearchResults {
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}
