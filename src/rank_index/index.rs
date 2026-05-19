//! Full-precision rank-cosine index ([`RankIndex`]).
//!
//! `u16` per coordinate; storage is `2 * dim` bytes per document.
//! Symmetric and asymmetric search paths share the rank-transform
//! pipeline from [`crate::rank`] and the [`TopK`](super::util::TopK)
//! collector from [`super::util`].

use rayon::prelude::*;

use super::util::{l2_normalise, TopK};
use crate::rank::{rank_norm, rank_transform, rank_transform_into};
use crate::SearchResults;

/// Full-precision rank-cosine index.
///
/// Stores each document as a `u16` rank vector of length `dim`. Storage
/// is `2 * dim` bytes per document. Norms are not stored — a permutation
/// of `{0, ..., dim - 1}` has fixed analytical L2 norm
/// `sqrt(dim * (dim^2 - 1) / 12)` after mean-centring.
///
/// Use this mode as the parity / upper-bound reference. For deployment
/// at compact byte budgets, prefer [`super::RankQuantIndex`].
pub struct RankIndex {
    dim: usize,
    n_vectors: usize,
    /// Row-major `n_vectors * dim` rank values in `[0, dim)`.
    ranks: Vec<u16>,
}

impl RankIndex {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2, "dim must be >= 2");
        assert!(dim <= u16::MAX as usize, "dim must fit in u16");
        Self {
            dim,
            n_vectors: 0,
            ranks: Vec::new(),
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim",
        );
        let start = self.ranks.len();
        self.ranks.resize(start + n * self.dim, 0);
        let dim = self.dim;
        self.ranks[start..]
            .par_chunks_mut(dim)
            .zip(vectors.par_chunks(dim))
            .for_each(|(out, v)| rank_transform_into(v, out));
        self.n_vectors += n;
    }

    /// Symmetric rank-cosine search: rank-transform the query, then
    /// score by Spearman correlation against each stored rank vector.
    ///
    /// Score is `sum_d (q_rank[d] - mean) * (d_rank[d] - mean)`. The
    /// constant `1 / (norm * norm)` is omitted because it does not
    /// affect top-`k` ordering, but the *displayed* score reflects the
    /// normalised cosine in `[-1, 1]` by dividing by the fixed
    /// analytical norm pair.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        let k_eff = k.min(self.n_vectors);
        if k_eff == 0 {
            return SearchResults {
                scores: vec![0.0; nq * k],
                indices: vec![-1; nq * k],
                nq,
                k,
            };
        }
        let dim = self.dim;
        let mean_2x = (dim as i32) - 1; // 2 * mean = D - 1; use to avoid f32 in the inner loop
        let n = self.n_vectors;
        let norm = rank_norm(dim);
        let inv_norm_sq = 1.0_f32 / (norm * norm);

        let mut scores_flat = vec![0.0f32; nq * k];
        let mut indices_flat = vec![-1i64; nq * k];

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                let q_ranks = rank_transform(q);
                let mut top = TopK::new(k_eff);
                for di in 0..n {
                    let doc = &self.ranks[di * dim..(di + 1) * dim];
                    // sum_d (2*q[d] - (D-1)) * (2*doc[d] - (D-1)) / 4
                    let mut acc: i64 = 0;
                    for d in 0..dim {
                        let qc = 2 * (q_ranks[d] as i32) - mean_2x;
                        let dc = 2 * (doc[d] as i32) - mean_2x;
                        acc += (qc as i64) * (dc as i64);
                    }
                    let s = (acc as f32) * 0.25 * inv_norm_sq;
                    top.maybe_insert(s, di);
                }
                top.finalize_into(out_scores, out_indices);
            });

        SearchResults {
            scores: scores_flat,
            indices: indices_flat,
            nq,
            k,
        }
    }

    /// Asymmetric rank-cosine search: queries stay as raw L2-normalised
    /// floats, documents are stored as ranks.
    ///
    /// Score is `sum_d q_unit[d] * (d_rank[d] - (D - 1) / 2) / norm`.
    /// The per-query constant `((D - 1) / 2) * sum_d q[d]` is folded out
    /// (it shifts every doc's score identically and does not change
    /// top-`k` ordering); the displayed score is the cosine on the
    /// mean-centred rank vector.
    pub fn search_asymmetric(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        let k_eff = k.min(self.n_vectors);
        if k_eff == 0 {
            return SearchResults {
                scores: vec![0.0; nq * k],
                indices: vec![-1; nq * k],
                nq,
                k,
            };
        }
        let dim = self.dim;
        let n = self.n_vectors;
        let norm = rank_norm(dim);
        let inv_norm = 1.0_f32 / norm;
        let mean = (dim as f32 - 1.0) / 2.0;

        let mut scores_flat = vec![0.0f32; nq * k];
        let mut indices_flat = vec![-1i64; nq * k];

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                // L2-normalise the query so the displayed score is a
                // cosine on the centred rank vector.
                let q_unit = l2_normalise(q);
                let q_sum: f32 = q_unit.iter().sum();
                let mut top = TopK::new(k_eff);
                for di in 0..n {
                    let doc = &self.ranks[di * dim..(di + 1) * dim];
                    let mut acc = 0.0f32;
                    for d in 0..dim {
                        acc += q_unit[d] * doc[d] as f32;
                    }
                    // <q, doc - mean> = <q, doc> - mean * <q, 1>
                    let s = (acc - mean * q_sum) * inv_norm;
                    top.maybe_insert(s, di);
                }
                top.finalize_into(out_scores, out_indices);
            });

        SearchResults {
            scores: scores_flat,
            indices: indices_flat,
            nq,
            k,
        }
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bytes_per_vec(&self) -> usize {
        self.dim * 2
    }
    /// Total bytes held by the rank buffer (excludes Vec overhead).
    pub fn byte_size(&self) -> usize {
        self.ranks.len() * std::mem::size_of::<u16>()
    }

    /// Remove a vector in O(1) by swapping with the last (mirrors
    /// `TurboQuantIndex::swap_remove`).
    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(idx < self.n_vectors, "index out of bounds");
        let last = self.n_vectors - 1;
        let dim = self.dim;
        if idx != last {
            for d in 0..dim {
                self.ranks[idx * dim + d] = self.ranks[last * dim + d];
            }
        }
        self.ranks.truncate(last * dim);
        self.n_vectors -= 1;
        last
    }

    /// Persist to a `.tvr` file. Format: 13-byte header + u16 ranks LE.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_rank(path, self.dim, self.n_vectors, &self.ranks)
    }

    /// Load from a `.tvr` file produced by [`Self::write`].
    ///
    /// Returns `io::Error` (kind `InvalidData`) on any structural
    /// inconsistency between the header and the payload (`load_rank`
    /// validates `dim ∈ [2, MAX_DIM]`, bounds `n_vectors`, and uses
    /// `checked_mul` for the payload size). Additional invariants
    /// specific to `RankIndex` are checked here.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (dim, n_vectors, ranks) = crate::rank_io::load_rank(path)?;
        if ranks.len() != n_vectors.saturating_mul(dim) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "TVR1 payload length does not match dim * n_vectors",
            ));
        }
        Ok(Self { dim, n_vectors, ranks })
    }
}
