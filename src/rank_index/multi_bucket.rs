//! MultiBucketBitmapIndex: `2^bits` bitmaps per document, one per bucket.
//!
//! Represents the constant-composition bucket assignment of each
//! document explicitly as a set of `2^bits` disjoint bitmaps over the
//! `dim` coordinates. The bilinear bucket-overlap score
//!
//! ```text
//! score(q, d) = Σ_{a, b} W[a, b] · |Q_a ∩ D_b|
//! ```
//!
//! for arbitrary weights `W[2^bits][2^bits]` is the formal object the
//! scoring decomposes into. For **outer-product weights**
//! `W[a, b] = (a − c)(b − c)` with `c = (2^bits − 1) / 2` this is
//! algebraically identical to the symmetric RankQuant per-coord score
//! `Σ_j (q_bucket[j] − c)(d_bucket[j] − c)` — a rank-1 weight matrix
//! just rearranges the same sum.
//!
//! Storage: `dim × 2^bits / 8` bytes per document
//! (b=2: 512 B/doc at D=1024 = matches RankQuant b=2;
//!  b=4: 2048 B/doc at D=1024 = 4× RankQuant b=4).
//!
//! The full 16×16 (b=4) probe is *not* a faster scoring kernel — it
//! uses the same FLOP count as the per-coord scalar form, rearranged
//! as 256 popcount-AND ops per doc. Its purpose is to expose the
//! bilinear decomposition empirically and serve as the reference for
//! **truncated** weight matrices (top-k buckets only, diagonal-only,
//! banded) which are the principled candidate-generation primitives.

use rayon::prelude::*;

use crate::rank::{rank_to_bucket, rank_transform};

/// Multi-bucket bitmap index over a constant-composition partition.
pub struct MultiBucketBitmapIndex {
    dim: usize,
    bits: u8,
    n_buckets: usize,
    qwords_per_bitmap: usize,
    n_vectors: usize,
    /// Row-major: doc-major outer, then bucket-major inner.
    /// Layout: bitmaps[di * (n_buckets * qpb) + bi * qpb + word_idx].
    bitmaps: Vec<u64>,
}

impl MultiBucketBitmapIndex {
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(matches!(bits, 1 | 2 | 4), "bits must be 1, 2, or 4");
        assert_eq!(dim % 64, 0, "dim must be a multiple of 64");
        let n_buckets = 1usize << bits;
        let qpb = dim / 64;
        assert_eq!(
            dim % n_buckets,
            0,
            "dim must be a multiple of 2^bits for constant-composition",
        );
        Self {
            dim,
            bits,
            n_buckets,
            qwords_per_bitmap: qpb,
            n_vectors: 0,
            bitmaps: Vec::new(),
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        let per_doc = nb * qpb;
        let start = self.bitmaps.len();
        self.bitmaps.resize(start + n * per_doc, 0u64);
        let dim = self.dim;
        let bits = self.bits;
        self.bitmaps[start..]
            .par_chunks_mut(per_doc)
            .zip(vectors.par_chunks(dim))
            .for_each(|(out, v)| {
                let ranks = rank_transform(v);
                for j in 0..dim {
                    let b = rank_to_bucket(ranks[j], dim, bits) as usize;
                    out[b * qpb + j / 64] |= 1u64 << (j % 64);
                }
            });
        self.n_vectors += n;
    }

    /// Bucket a query's rank-transformed coordinates into bitmaps,
    /// matching the document encoding. Used for symmetric bilinear
    /// scoring and bucket-overlap probes.
    pub fn query_bitmaps_from_ranks(&self, q: &[f32]) -> Vec<u64> {
        assert_eq!(q.len(), self.dim);
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        let bits = self.bits;
        let dim = self.dim;
        let ranks = rank_transform(q);
        let mut out = vec![0u64; nb * qpb];
        for j in 0..dim {
            let b = rank_to_bucket(ranks[j], dim, bits) as usize;
            out[b * qpb + j / 64] |= 1u64 << (j % 64);
        }
        out
    }

    /// Outer-product weight matrix `W[a, b] = (a − c) (b − c)` where
    /// `c = (2^bits − 1) / 2`. This is the weight that makes the
    /// bilinear bucket-overlap score equal the symmetric RankQuant
    /// per-coord score.
    pub fn outer_product_weights(&self) -> Vec<f32> {
        let nb = self.n_buckets;
        let c = (nb as f32 - 1.0) / 2.0;
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                w[a * nb + b] = (a as f32 - c) * (b as f32 - c);
            }
        }
        w
    }

    /// Compute the bilinear bucket-overlap score
    ///   `Σ_{a, b} W[a, b] · |Q_a ∩ D_b|`
    /// for a single (query, doc) pair. Scales nothing — caller
    /// applies any normalisation.
    pub fn bilinear_score(&self, q_bitmaps: &[u64], w: &[f32], doc_idx: usize) -> f32 {
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        debug_assert_eq!(q_bitmaps.len(), nb * qpb);
        debug_assert_eq!(w.len(), nb * nb);
        let doc_base = doc_idx * nb * qpb;
        let mut acc = 0.0f32;
        for a in 0..nb {
            for b in 0..nb {
                let weight = w[a * nb + b];
                if weight == 0.0 {
                    continue;
                }
                let q_off = a * qpb;
                let d_off = doc_base + b * qpb;
                let mut overlap: u32 = 0;
                for k in 0..qpb {
                    overlap += (q_bitmaps[q_off + k] & self.bitmaps[d_off + k]).count_ones();
                }
                acc += weight * (overlap as f32);
            }
        }
        acc
    }

    /// Single-query candidate generation: returns the top-`m` doc IDs
    /// by bilinear bucket-overlap score against the query's bucket
    /// bitmaps under weight matrix `w`. Uses scan-then-select_nth so
    /// large M doesn't pay an O(N·M) TopK tax.
    pub fn top_m_bilinear(&self, q_bitmaps: &[u64], w: &[f32], m: usize) -> Vec<u32> {
        let m_eff = m.min(self.n_vectors);
        if m_eff == 0 {
            return Vec::new();
        }
        let n = self.n_vectors;
        let mut scores = vec![0.0f32; n];
        scores
            .par_iter_mut()
            .enumerate()
            .for_each(|(di, s)| {
                *s = self.bilinear_score(q_bitmaps, w, di);
            });
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.select_nth_unstable_by(m_eff - 1, |&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut head = idx[..m_eff].to_vec();
        head.sort_unstable_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        head
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
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }
    pub fn bytes_per_vec(&self) -> usize {
        self.qwords_per_bitmap * self.n_buckets * 8
    }
    pub fn byte_size(&self) -> usize {
        self.bitmaps.len() * std::mem::size_of::<u64>()
    }
}
