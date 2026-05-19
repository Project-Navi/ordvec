//! Rank-cosine index modes.
//!
//! Two index types and two scoring modes each:
//!
//! - [`RankIndex`] stores full-precision rank vectors (`u16` per
//!   coordinate, `2 * dim` bytes per document).
//! - [`RankQuantIndex`] buckets each rank into `1 << bits` equal-width
//!   bins on `[0, dim)` and packs `bits` bits per coordinate
//!   (`dim * bits / 8` bytes per document).
//!
//! Both expose:
//!
//! - `search` — *symmetric* rank-cosine: the query is rank-transformed,
//!   mean-centred, and scored against the stored ranks/buckets. This
//!   reproduces Spearman correlation on the underlying coordinates and
//!   matches the symmetric variant of the paper.
//! - `search_asymmetric` — *asymmetric* rank-cosine: the query stays as
//!   raw L2-normalised floats and is scored against the stored
//!   ranks/buckets via a per-query D-by-`1<<bits` lookup table. The
//!   document side carries no magnitudes; the query encoder runs
//!   exactly once at query time.
//!
//! No training, no rotation, no codebook. Norms are analytical.
//!
//! A third type — [`BitmapIndex`] — stores only a *top-bucket bitmap*
//! per document (one bit per coordinate, set when that coordinate's
//! rank is in the top `n_top` of the document). For `dim=1024,
//! n_top=256` (equivalent to RankQuant b=2's top bucket) that's
//! `128 B/doc`, half of RankQuant b=2 storage. Scoring is
//! `popcount(Q_bitmap AND D_bitmap)`: a coarsened rank-overlap that
//! exploits the constant-composition prior (every doc has exactly
//! `n_top` bits set) and runs as a streaming AND + popcount over 16
//! qwords per doc at `D=1024`.
//!
//! ```no_run
//! use turbovec::{RankIndex, RankQuantIndex};
//!
//! let mut idx = RankQuantIndex::new(1024, 2);
//! let docs: Vec<f32> = vec![0.0; 1024 * 10_000];
//! idx.add(&docs);
//!
//! let queries: Vec<f32> = vec![0.0; 1024 * 4];
//! let res = idx.search_asymmetric(&queries, 10);
//! assert_eq!(res.k, 10);
//! ```
//!
//! # Module layout
//!
//! The implementation is split across sibling modules for compile-unit
//! locality and to keep individual files under the project's 800-line
//! guideline. The split is internal — all `pub` items are re-exported
//! here so `use turbovec::rank_index::{RankIndex, RankQuantIndex,
//! BitmapIndex, MultiBucketBitmapIndex, search_asymmetric_byte_lut}`
//! continues to resolve unchanged.

mod bitmap;
mod index;
mod multi_bucket;
mod quant;
mod quant_kernels;
mod util;

pub use bitmap::BitmapIndex;
pub use index::RankIndex;
pub use multi_bucket::MultiBucketBitmapIndex;
pub use quant::{search_asymmetric_byte_lut, RankQuantIndex};
