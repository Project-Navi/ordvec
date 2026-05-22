//! Training-free ordinal & sign quantization for vector retrieval.
//!
//! `ordvec` is the ordinal/sign retrieval substrate extracted from
//! [turbovec](https://github.com/RyanCodrai/turbovec) (MIT). It carries
//! no system dependencies — no BLAS, no `ndarray`, no `faer` — and needs
//! no training, rotation, or codebook. Norms are analytical.
//!
//! Three substrate families, all data-oblivious:
//!
//! - [`RankIndex`] stores full-precision rank vectors (`u16` per
//!   coordinate, `2 * dim` bytes per document).
//! - [`RankQuantIndex`] buckets each rank into `1 << bits` equal-width
//!   bins and packs `bits` bits per coordinate (`dim * bits / 8` bytes
//!   per document).
//! - [`BitmapIndex`] stores a top-bucket bitmap per document (one bit
//!   per coordinate) and scores via `popcount(Q AND D)`.
//! - [`SignBitmapIndex`] stores a sign bitmap per document (one bit per
//!   coordinate, set when the coordinate is positive) for sign-cosine
//!   candidate generation.
//!
//! ```no_run
//! use ordvec::{RankIndex, RankQuantIndex};
//!
//! let mut idx = RankQuantIndex::new(1024, 2);
//! let docs: Vec<f32> = vec![0.0; 1024 * 10_000];
//! idx.add(&docs);
//!
//! let queries: Vec<f32> = vec![0.0; 1024 * 4];
//! let res = idx.search_asymmetric(&queries, 10);
//! assert_eq!(res.k, 10);
//! ```

pub mod rank;
pub mod rank_index;
pub mod rank_io;
pub mod sign_bitmap;

pub use rank_index::{BitmapIndex, RankIndex, RankQuantIndex};
pub use sign_bitmap::SignBitmapIndex;

// `MultiBucketBitmapIndex` underwrites the bilinear bucket-overlap
// decomposition but is not stable public API. It is reachable only with
// the `experimental` feature; the default surface excludes it.
#[cfg(feature = "experimental")]
pub use rank_index::MultiBucketBitmapIndex;

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
