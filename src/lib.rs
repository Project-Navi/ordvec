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
//!   per document).
//! - [`Bitmap`] stores a top-bucket bitmap per document (one bit
//!   per coordinate) and scores via `popcount(Q AND D)`.
//! - [`SignBitmap`] stores a sign bitmap per document (one bit per
//!   coordinate, set when the coordinate is positive) for sign-cosine
//!   candidate generation.
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

mod bitmap;
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
pub use quant::RankQuant;
pub use rank::Rank;
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

// `MultiBucketBitmap` underwrites the bilinear bucket-overlap
// decomposition but is not stable public API. It is reachable only with
// the `experimental` feature; the default surface excludes it.
#[cfg(feature = "experimental")]
pub use multi_bucket::MultiBucketBitmap;

// `RankQuantFastscan` is an optional FastScan b=2 scan path. It is
// re-exported `#[doc(hidden)]` at the crate root — reachable as
// `ordvec::RankQuantFastscan` for callers who opt in, but not
// advertised alongside the headline index types above.
#[doc(hidden)]
pub use fastscan::RankQuantFastscan;

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

/// Top-k search results, laid out as `nq` contiguous blocks of `k`.
///
/// `scores` and `indices` are flat row-major buffers of length `nq * k`;
/// block `qi` is `[qi * k, (qi + 1) * k)`. Use [`Self::scores_for_query`]
/// / [`Self::indices_for_query`] to slice a single query's results.
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
