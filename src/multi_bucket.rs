//! MultiBucketBitmap: `2^bits` bitmaps per document, one per bucket.
//!
//! Represents the constant-composition bucket assignment of each
//! document explicitly as a set of `2^bits` disjoint bitmaps over the
//! `dim` coordinates. The bilinear bucket-overlap score
//!
//! ```text
//! score(q, d) = Σ_{a, b} W[a, b] · |Q_a ∩ D_b|
//! ```
//!
//! for arbitrary weights `W[2^bits][2^bits]` is the algebraic object the
//! scoring decomposes into. For **outer-product weights**
//! `W[a, b] = (a − c)(b − c)` with `c = (2^bits − 1) / 2` this is
//! algebraically identical to the symmetric RankQuant per-coord score
//! `Σ_j (q_bucket[j] − c)(d_bucket[j] − c)` — a rank-1 weight matrix
//! just rearranges the same sum.
//!
//! Storage: `dim × 2^bits / 8` bytes per document
//! (b=2: 512 B/doc at D=1024 = 2× RankQuant b=2;
//!  b=4: 2048 B/doc at D=1024 = 4× RankQuant b=4).
//!
//! The full 16×16 (b=4) probe is *not* a faster scoring kernel — it
//! uses the same FLOP count as the per-coord scalar form, rearranged
//! as 256 popcount-AND ops per doc. Its purpose is to expose the
//! bilinear decomposition empirically and serve as the reference for
//! **truncated** weight matrices (top-k buckets only, diagonal-only,
//! banded) which are the principled candidate-generation primitives.
//!
//! The companion Lean theorem currently attaches to the literal top-bucket
//! constant-weight [`crate::Bitmap`] overlap statistic and its threshold tail,
//! not to arbitrary bilinear weights over all buckets.

use rayon::prelude::*;

use crate::rank::{rank_to_bucket, rank_transform};

/// Multi-bucket bitmap index over a constant-composition partition.
pub struct MultiBucketBitmap {
    dim: usize,
    bits: u8,
    n_buckets: usize,
    qwords_per_bitmap: usize,
    n_vectors: usize,
    /// Row-major: doc-major outer, then bucket-major inner.
    /// Layout: bitmaps[di * (n_buckets * qpb) + bi * qpb + word_idx].
    bitmaps: Vec<u64>,
}

impl MultiBucketBitmap {
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(matches!(bits, 1 | 2 | 4), "bits must be 1, 2, or 4");
        // dim=0 satisfies `% 64` and `% n_buckets` divisibility but
        // produces qwords_per_bitmap=0, deferring a div-by-zero into
        // `add` (n = vectors.len() / dim). Reject at construction,
        // mirroring SignBitmap::new.
        assert!(dim > 0, "dim must be > 0");
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
        crate::util::assert_all_finite(vectors);
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
        crate::util::assert_all_finite(q);
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

    /// Unit-diagonal weight matrix: `W[a, a] = 1`, off-diagonal `0`. The
    /// bilinear score reduces to `Σ_a |Q_a ∩ D_a|` — the same-bucket
    /// agreement count. This is the cheapest truncation: `nb` popcount-AND
    /// passes per doc instead of the full `nb²`. It is a pure overlap signal
    /// (no bucket-magnitude weighting), so it is closest in spirit to a
    /// multi-level [`crate::Bitmap`].
    pub fn diagonal_weights(&self) -> Vec<f32> {
        let nb = self.n_buckets;
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            w[a * nb + a] = 1.0;
        }
        w
    }

    /// Outer-product weights `(a − c)(b − c)` restricted to the band
    /// `|a − b| <= half_width` (off-band entries zeroed). `half_width = 0`
    /// keeps only the magnitude-weighted diagonal; `half_width >= nb − 1`
    /// recovers the full [`Self::outer_product_weights`] matrix. Sweeping
    /// `half_width` interpolates candidate-gen cost (non-zero band entries ⇒
    /// popcount-AND passes per doc) between the diagonal and the exact
    /// bilinear probe, tracing the recall/latency frontier.
    pub fn banded_weights(&self, half_width: usize) -> Vec<f32> {
        let nb = self.n_buckets;
        let c = (nb as f32 - 1.0) / 2.0;
        let mut w = vec![0.0f32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                if a.abs_diff(b) <= half_width {
                    w[a * nb + b] = (a as f32 - c) * (b as f32 - c);
                }
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
        assert!(
            doc_idx < self.n_vectors,
            "bilinear_score: doc_idx {doc_idx} out of range (n_vectors {})",
            self.n_vectors,
        );
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
        scores.par_iter_mut().enumerate().for_each(|(di, s)| {
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

    // -----------------------------------------------------------------
    // Indexed contingency / projection surface (issue #219, API 2 of 2).
    //
    // `bilinear_score` / `top_m_bilinear` recompute one weighted `Σ W·C`
    // per call and rescan the doc's bitmaps once per *projection*. The
    // methods below decouple the accumulation from the projection: the
    // `nb × nb` contingency table `C[a, b] = |Q_a ∩ D_b|` is built **once**
    // per (query, doc) over a single pass of the doc's bitmaps, then any
    // number of weight matrices are applied as cheap `nb²` dot products
    // over the cached integer table. This is the in-index analogue of the
    // stateless [`crate::Contingency`] (API 1) — same `nb × nb` object,
    // but accumulated straight from the bitmap rows rather than from raw
    // `&[u8]` codes.
    //
    // The hot inner accumulation (the `nb × nb` popcount-AND over `qpb`
    // u64 words) dispatches at runtime to an AVX-512 VPOPCNTDQ kernel,
    // mirroring [`crate::Bitmap`]'s masked-tail popcount-AND scan, with a
    // portable scalar fallback on every other target / unsupported CPU.
    // -----------------------------------------------------------------

    /// Full `nb × nb` contingency table `C[a, b] = |Q_a ∩ D_b|` for one
    /// document, accumulated in a single pass over the doc's bitmaps.
    ///
    /// Returns a row-major `nb × nb` `Vec<u32>` where `out[a * nb + b]` is
    /// the count of coordinates the query placed in bucket `a` and the doc
    /// placed in bucket `b`. This is the indexed twin of
    /// [`crate::Contingency::counts`]: the same table, built from the
    /// bitmap rows instead of `&[u8]` codes. A caller can then apply any
    /// projection (e.g. via [`crate::Contingency`]'s projections, or a raw
    /// `nb²` weighted sum) without rescanning the bitmaps.
    ///
    /// # Panics
    /// Panics if `doc_idx >= len()`. `q_bitmaps` must be `nb * qpb` words
    /// (as produced by [`Self::query_bitmaps_from_ranks`]).
    pub fn contingency_row(&self, q_bitmaps: &[u64], doc_idx: usize) -> Vec<u32> {
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        assert!(
            doc_idx < self.n_vectors,
            "contingency_row: doc_idx {doc_idx} out of range (n_vectors {})",
            self.n_vectors,
        );
        assert_eq!(
            q_bitmaps.len(),
            nb * qpb,
            "contingency_row: q_bitmaps must be nb * qpb words",
        );
        let doc_base = doc_idx * nb * qpb;
        let doc = &self.bitmaps[doc_base..doc_base + nb * qpb];
        let mut table = vec![0u32; nb * nb];
        contingency_accumulate(q_bitmaps, doc, nb, qpb, &mut table);
        table
    }

    /// Diagonal-only fast path: the `nb` cells `C[a, a] = |Q_a ∩ D_a|` for
    /// one document, in a single pass over the doc's bitmaps.
    ///
    /// This is the common cheap projection — same-bucket agreement per
    /// bucket — and costs `nb` popcount-AND passes per doc instead of the
    /// full `nb²`. `out[a]` is `|Q_a ∩ D_a|`. Summing the result is the
    /// [`Self::diagonal_weights`] bilinear score.
    ///
    /// # Panics
    /// Panics if `doc_idx >= len()` or `q_bitmaps.len() != nb * qpb`.
    pub fn diagonal_overlap_row(&self, q_bitmaps: &[u64], doc_idx: usize) -> Vec<u32> {
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        assert!(
            doc_idx < self.n_vectors,
            "diagonal_overlap_row: doc_idx {doc_idx} out of range (n_vectors {})",
            self.n_vectors,
        );
        assert_eq!(
            q_bitmaps.len(),
            nb * qpb,
            "diagonal_overlap_row: q_bitmaps must be nb * qpb words",
        );
        let doc_base = doc_idx * nb * qpb;
        let doc = &self.bitmaps[doc_base..doc_base + nb * qpb];
        let mut diag = vec![0u32; nb];
        diagonal_accumulate(q_bitmaps, doc, nb, qpb, &mut diag);
        diag
    }

    /// Batched indexed projection: for every document, build its `nb × nb`
    /// contingency table **once**, then apply *all* of `weights` to that
    /// single cached table, returning `docs × projections` scores.
    ///
    /// `weights[p]` is the row-major `nb × nb` weight matrix for projection
    /// `p`. The returned `out[di][p]` is `Σ_{a, b} weights[p][a*nb+b] ·
    /// C_di[a, b]`. Each document's bitmaps are streamed once regardless of
    /// how many projections are requested — the projections share the
    /// accumulated table rather than each rescanning the corpus (which is
    /// what `bilinear_score` would do per projection).
    ///
    /// Runs across documents in parallel via rayon. The per-doc table is a
    /// small `nb²` integer buffer; the projection dot products are cheap
    /// relative to the popcount-AND accumulation that dominates.
    ///
    /// # Panics
    /// Panics if `q_bitmaps.len() != nb * qpb` or if any `weights[p].len()
    /// != nb * nb`.
    #[must_use = "this scans every doc's bitmaps to build contingency tables; dropping the result discards that work"]
    pub fn project_all_batched(&self, q_bitmaps: &[u64], weights: &[&[f32]]) -> Vec<Vec<f32>> {
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        assert_eq!(
            q_bitmaps.len(),
            nb * qpb,
            "project_all_batched: q_bitmaps must be nb * qpb words",
        );
        for (p, w) in weights.iter().enumerate() {
            assert_eq!(
                w.len(),
                nb * nb,
                "project_all_batched: weights[{p}] must be an nb * nb matrix",
            );
        }
        let n = self.n_vectors;
        let n_proj = weights.len();
        if n == 0 || n_proj == 0 {
            return vec![Vec::new(); n];
        }
        let per_doc = nb * qpb;
        let bitmaps = &self.bitmaps;
        (0..n)
            .into_par_iter()
            .map(|di| {
                let doc = &bitmaps[di * per_doc..(di + 1) * per_doc];
                // One accumulation pass over this doc's bitmaps. nb <= 16 ⇒
                // nb*nb <= 256, so a stack table avoids a per-doc heap
                // allocation inside the parallel map (allocator contention).
                let mut table = [0u32; 256];
                let table = &mut table[..nb * nb];
                contingency_accumulate(q_bitmaps, doc, nb, qpb, table);
                project_table(table, weights)
            })
            .collect()
    }

    /// `#[doc(hidden)]` bench-only twin of [`Self::diagonal_overlap_row`] that
    /// forces the **portable scalar** diagonal kernel, bypassing the runtime
    /// AVX-512 dispatch. It exists so `examples/bench_contingency` can time the
    /// scalar and SIMD diagonal paths against each other on the same index
    /// (mirroring the `#[doc(hidden)]` `search_asymmetric_byte_lut` bench
    /// reference at the crate root). Not part of the stable API — production
    /// callers use [`Self::diagonal_overlap_row`], which dispatches to the
    /// fastest available kernel.
    ///
    /// # Panics
    /// Panics if `doc_idx >= len()` or `q_bitmaps.len() != nb * qpb`.
    #[doc(hidden)]
    pub fn diagonal_overlap_row_scalar(&self, q_bitmaps: &[u64], doc_idx: usize) -> Vec<u32> {
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        assert!(
            doc_idx < self.n_vectors,
            "diagonal_overlap_row_scalar: doc_idx {doc_idx} out of range (n_vectors {})",
            self.n_vectors,
        );
        assert_eq!(
            q_bitmaps.len(),
            nb * qpb,
            "diagonal_overlap_row_scalar: q_bitmaps must be nb * qpb words",
        );
        let doc_base = doc_idx * nb * qpb;
        let doc = &self.bitmaps[doc_base..doc_base + nb * qpb];
        let mut diag = vec![0u32; nb];
        diagonal_accumulate_scalar(q_bitmaps, doc, nb, qpb, &mut diag);
        diag
    }

    /// `#[doc(hidden)]` bench-only twin of [`Self::project_all_batched`] that
    /// forces the **portable scalar** contingency-accumulation kernel,
    /// bypassing the runtime AVX-512 dispatch. Lets
    /// `examples/bench_contingency` time the scalar and SIMD batched-projection
    /// paths on the same index. Not part of the stable API.
    ///
    /// # Panics
    /// Panics if `q_bitmaps.len() != nb * qpb` or if any `weights[p].len()
    /// != nb * nb`.
    #[doc(hidden)]
    #[must_use = "this scans every doc's bitmaps to build contingency tables; dropping the result discards that work"]
    pub fn project_all_batched_scalar(
        &self,
        q_bitmaps: &[u64],
        weights: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        let qpb = self.qwords_per_bitmap;
        let nb = self.n_buckets;
        assert_eq!(
            q_bitmaps.len(),
            nb * qpb,
            "project_all_batched_scalar: q_bitmaps must be nb * qpb words",
        );
        for (p, w) in weights.iter().enumerate() {
            assert_eq!(
                w.len(),
                nb * nb,
                "project_all_batched_scalar: weights[{p}] must be an nb * nb matrix",
            );
        }
        let n = self.n_vectors;
        let n_proj = weights.len();
        if n == 0 || n_proj == 0 {
            return vec![Vec::new(); n];
        }
        let per_doc = nb * qpb;
        let bitmaps = &self.bitmaps;
        (0..n)
            .into_par_iter()
            .map(|di| {
                let doc = &bitmaps[di * per_doc..(di + 1) * per_doc];
                // Stack table (nb*nb <= 256): no per-doc heap alloc in the
                // parallel map.
                let mut table = [0u32; 256];
                let table = &mut table[..nb * nb];
                contingency_accumulate_scalar(q_bitmaps, doc, nb, qpb, table);
                project_table(table, weights)
            })
            .collect()
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

// =====================================================================
// Contingency-accumulation kernels.
//
// `q_bitmaps` and `doc` are both row-major `nb * qpb` u64 layouts: bucket
// `a`'s bitmap occupies words `[a * qpb, (a + 1) * qpb)`. The full table
// cell is
//   C[a, b] = Σ_{w < qpb} popcount(Q_a[w] & D_b[w]),
// the popcount over the bitwise-AND of bucket `a`'s query bitmap and
// bucket `b`'s doc bitmap — the exact popcount-AND shape of
// [`crate::Bitmap`]'s overlap kernel, here run for every (a, b) pair.
//
// Each path dispatches to an AVX-512 VPOPCNTDQ kernel when the host
// supports it (mirroring the masked-tail discipline in `index/bitmap.rs`),
// otherwise the portable scalar reference. Counts are exact integers, so
// the SIMD and scalar paths are required to agree bit-for-bit (the parity
// test enforces this against API 1's dense `Contingency`).
// =====================================================================

/// Build the full row-major `nb × nb` contingency table for one
/// (query, doc) pair into `table` (length `nb * nb`).
///
/// Dispatches to the AVX-512 VPOPCNTDQ kernel when available, else the
/// scalar reference. Both paths require `q_bitmaps.len() == doc.len() ==
/// nb * qpb` and `table.len() == nb * nb`.
fn contingency_accumulate(
    q_bitmaps: &[u64],
    doc: &[u64],
    nb: usize,
    qpb: usize,
    table: &mut [u32],
) {
    debug_assert_eq!(q_bitmaps.len(), nb * qpb);
    debug_assert_eq!(doc.len(), nb * qpb);
    debug_assert_eq!(table.len(), nb * nb);

    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop =
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            contingency_accumulate_avx512vpop(q_bitmaps, doc, nb, qpb, table);
            return;
        }
    }
    contingency_accumulate_scalar(q_bitmaps, doc, nb, qpb, table);
}

/// Apply every weight matrix in `weights` to a single cached `nb × nb`
/// contingency `table`, returning one `f32` projection score per matrix.
/// Each score is `Σ_{cell} weight[cell] · table[cell]` — the cheap dot
/// product the batched path shares across projections after one accumulation
/// pass. Caller guarantees every `weights[p].len() == table.len()`.
fn project_table(table: &[u32], weights: &[&[f32]]) -> Vec<f32> {
    weights
        .iter()
        .map(|w| {
            w.iter()
                .zip(table.iter())
                .map(|(&weight, &c)| weight * c as f32)
                .sum()
        })
        .collect()
}

/// Portable scalar reference for the full `nb × nb` table. One
/// popcount-AND reduction per (a, b) cell over the `qpb` u64 words.
fn contingency_accumulate_scalar(
    q_bitmaps: &[u64],
    doc: &[u64],
    nb: usize,
    qpb: usize,
    table: &mut [u32],
) {
    for a in 0..nb {
        let q_off = a * qpb;
        let row = a * nb;
        for b in 0..nb {
            let d_off = b * qpb;
            let mut overlap: u32 = 0;
            for w in 0..qpb {
                overlap += (q_bitmaps[q_off + w] & doc[d_off + w]).count_ones();
            }
            table[row + b] = overlap;
        }
    }
}

/// Build only the `nb` diagonal cells `C[a, a] = |Q_a ∩ D_a|` into `diag`
/// (length `nb`). The common cheap projection.
///
/// Dispatches to the AVX-512 VPOPCNTDQ diagonal kernel when available,
/// else the scalar reference.
fn diagonal_accumulate(q_bitmaps: &[u64], doc: &[u64], nb: usize, qpb: usize, diag: &mut [u32]) {
    debug_assert_eq!(q_bitmaps.len(), nb * qpb);
    debug_assert_eq!(doc.len(), nb * qpb);
    debug_assert_eq!(diag.len(), nb);

    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop =
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            diagonal_accumulate_avx512vpop(q_bitmaps, doc, nb, qpb, diag);
            return;
        }
    }
    diagonal_accumulate_scalar(q_bitmaps, doc, nb, qpb, diag);
}

/// Portable scalar reference for the `nb` diagonal cells.
fn diagonal_accumulate_scalar(
    q_bitmaps: &[u64],
    doc: &[u64],
    nb: usize,
    qpb: usize,
    diag: &mut [u32],
) {
    // `a` is load-bearing: it indexes the bucket bitmap offset (`a * qpb`)
    // as well as `diag[a]`.
    #[allow(clippy::needless_range_loop)]
    for a in 0..nb {
        let off = a * qpb;
        let mut overlap: u32 = 0;
        for w in 0..qpb {
            overlap += (q_bitmaps[off + w] & doc[off + w]).count_ones();
        }
        diag[a] = overlap;
    }
}

/// AVX-512 VPOPCNTDQ full-table contingency accumulation.
///
/// Tiles the `qpb`-word bucket bitmaps into 512-bit (8×u64) ZMM blocks
/// with a `bzhi`-style masked tail for the final `qpb % 8` words, exactly
/// as [`crate::Bitmap`]'s scan kernel does. For each block, every query
/// bucket `a`'s lane and every doc bucket `b`'s lane is AND-ed,
/// `_mm512_popcnt_epi64`-counted, and reduced into the `(a, b)` cell — the
/// masked-tail popcount-AND, run over all `nb²` bucket pairs.
///
/// # Safety
/// Requires the `avx512f` and `avx512vpopcntdq` target features, confirmed
/// by the `#[target_feature]` gate plus the caller's runtime
/// `is_x86_feature_detected!`. `q_bitmaps.len() == doc.len() == nb * qpb`
/// and `table.len() == nb * nb` are caller contracts (asserted at the
/// public entry points); the tail load uses a `(1 << rem) - 1` word mask
/// so no lane past `qpb` is read for either operand.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn contingency_accumulate_avx512vpop(
    q_bitmaps: &[u64],
    doc: &[u64],
    nb: usize,
    qpb: usize,
    table: &mut [u32],
) {
    use std::arch::x86_64::*;
    // SAFETY: `q_bitmaps.len() == doc.len() == nb * qpb` and `table.len()
    // == nb * nb` (caller contract). Each ZMM load reads 8 u64 words at a
    // word offset `< qpb` within bucket `a`/`b`'s `qpb`-word region; the
    // final partial block uses `_mm512_maskz_loadu_epi64` with a
    // `(1 << rem) - 1` mask so lanes past `qpb` read as zero and are never
    // dereferenced. AVX-512 F/VPOPCNTDQ are confirmed by `#[target_feature]`
    // plus the runtime detection in the caller.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        let full_blocks = qpb / 8;
        let rem = qpb % 8;
        let tail_mask: __mmask8 = if rem == 0 { 0 } else { (1u8 << rem) - 1 };
        let q_ptr = q_bitmaps.as_ptr();
        let d_ptr = doc.as_ptr();

        for a in 0..nb {
            let q_base = a * qpb;
            let row = a * nb;
            for b in 0..nb {
                let d_base = b * qpb;
                let mut acc = _mm512_setzero_si512();
                // Whole 512-bit blocks.
                for blk in 0..full_blocks {
                    let w = blk * 8;
                    let q_zmm = _mm512_loadu_si512(q_ptr.add(q_base + w) as *const __m512i);
                    let d_zmm = _mm512_loadu_si512(d_ptr.add(d_base + w) as *const __m512i);
                    let and_zmm = _mm512_and_si512(q_zmm, d_zmm);
                    let pop_zmm = _mm512_popcnt_epi64(and_zmm);
                    acc = _mm512_add_epi64(acc, pop_zmm);
                }
                // Masked tail: only the low `rem` words are loaded; lanes
                // past `qpb` read as zero and contribute no popcount.
                if rem != 0 {
                    let w = full_blocks * 8;
                    let q_zmm =
                        _mm512_maskz_loadu_epi64(tail_mask, q_ptr.add(q_base + w) as *const i64);
                    let d_zmm =
                        _mm512_maskz_loadu_epi64(tail_mask, d_ptr.add(d_base + w) as *const i64);
                    let and_zmm = _mm512_and_si512(q_zmm, d_zmm);
                    let pop_zmm = _mm512_popcnt_epi64(and_zmm);
                    acc = _mm512_add_epi64(acc, pop_zmm);
                }
                let sum: i64 = _mm512_reduce_add_epi64(acc);
                table[row + b] = sum as u32;
            }
        }
    }
}

/// AVX-512 VPOPCNTDQ diagonal-only accumulation: the `nb` cells
/// `C[a, a]`, each a masked-tail popcount-AND over bucket `a`'s query and
/// doc bitmaps. Same masked-tail discipline as the full kernel, restricted
/// to the diagonal.
///
/// # Safety
/// Same contract as [`contingency_accumulate_avx512vpop`]:
/// `q_bitmaps.len() == doc.len() == nb * qpb`, `diag.len() == nb`, the
/// target features confirmed by the gate plus runtime detection, and the
/// masked tail bounding every load to the live `qpb` words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn diagonal_accumulate_avx512vpop(
    q_bitmaps: &[u64],
    doc: &[u64],
    nb: usize,
    qpb: usize,
    diag: &mut [u32],
) {
    use std::arch::x86_64::*;
    // SAFETY: identical bounding to `contingency_accumulate_avx512vpop`,
    // restricted to `b == a`: every load is at a word offset `< qpb` within
    // bucket `a`'s `qpb`-word region, with the final partial block masked to
    // the low `rem` words. `diag.len() == nb` bounds every `diag[a]` write.
    // AVX-512 F/VPOPCNTDQ confirmed by `#[target_feature]` + runtime detection.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        let full_blocks = qpb / 8;
        let rem = qpb % 8;
        let tail_mask: __mmask8 = if rem == 0 { 0 } else { (1u8 << rem) - 1 };
        let q_ptr = q_bitmaps.as_ptr();
        let d_ptr = doc.as_ptr();

        // `a` indexes the bucket bitmap offset (`a * qpb`) and `diag[a]`.
        #[allow(clippy::needless_range_loop)]
        for a in 0..nb {
            let base = a * qpb;
            let mut acc = _mm512_setzero_si512();
            for blk in 0..full_blocks {
                let w = blk * 8;
                let q_zmm = _mm512_loadu_si512(q_ptr.add(base + w) as *const __m512i);
                let d_zmm = _mm512_loadu_si512(d_ptr.add(base + w) as *const __m512i);
                let and_zmm = _mm512_and_si512(q_zmm, d_zmm);
                let pop_zmm = _mm512_popcnt_epi64(and_zmm);
                acc = _mm512_add_epi64(acc, pop_zmm);
            }
            if rem != 0 {
                let w = full_blocks * 8;
                let q_zmm = _mm512_maskz_loadu_epi64(tail_mask, q_ptr.add(base + w) as *const i64);
                let d_zmm = _mm512_maskz_loadu_epi64(tail_mask, d_ptr.add(base + w) as *const i64);
                let and_zmm = _mm512_and_si512(q_zmm, d_zmm);
                let pop_zmm = _mm512_popcnt_epi64(and_zmm);
                acc = _mm512_add_epi64(acc, pop_zmm);
            }
            let sum: i64 = _mm512_reduce_add_epi64(acc);
            diag[a] = sum as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Contingency;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Reconstruct the per-coordinate bucket codes a doc was encoded with
    /// by reading back the index's stored bitmaps. Used to feed API 1's
    /// dense `Contingency` the *same* bucket assignment the index holds.
    fn doc_codes(idx: &MultiBucketBitmap, doc_idx: usize) -> Vec<u8> {
        let nb = idx.n_buckets();
        let qpb = idx.qwords_per_bitmap;
        let dim = idx.dim();
        let base = doc_idx * nb * qpb;
        let mut codes = vec![0u8; dim];
        for b in 0..nb {
            let off = base + b * qpb;
            // `j` is load-bearing: it indexes the bit position (`j/64`,
            // `j%64`) as well as `codes[j]`.
            #[allow(clippy::needless_range_loop)]
            for j in 0..dim {
                if (idx.bitmaps[off + j / 64] >> (j % 64)) & 1 == 1 {
                    codes[j] = b as u8;
                }
            }
        }
        codes
    }

    /// Reconstruct the query bucket codes from the query bitmaps.
    fn query_codes(q_bitmaps: &[u64], nb: usize, qpb: usize, dim: usize) -> Vec<u8> {
        let mut codes = vec![0u8; dim];
        for b in 0..nb {
            let off = b * qpb;
            // `j` is load-bearing: bit position and `codes[j]` index.
            #[allow(clippy::needless_range_loop)]
            for j in 0..dim {
                if (q_bitmaps[off + j / 64] >> (j % 64)) & 1 == 1 {
                    codes[j] = b as u8;
                }
            }
        }
        codes
    }

    /// Independent pure-scalar reference for the full table, not routed
    /// through the dispatch (so the dispatched path can't mask its own bug).
    fn reference_table(q_bitmaps: &[u64], doc: &[u64], nb: usize, qpb: usize) -> Vec<u32> {
        let mut table = vec![0u32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                let mut overlap = 0u32;
                for w in 0..qpb {
                    overlap += (q_bitmaps[a * qpb + w] & doc[b * qpb + w]).count_ones();
                }
                table[a * nb + b] = overlap;
            }
        }
        table
    }

    fn make_corpus(seed: u64, n: usize, dim: usize) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    /// CORRECTNESS GATE (issue #219): the dispatched contingency path
    /// (AVX-512 on this host) and the diagonal fast path must produce
    /// BIT-IDENTICAL results to (a) an independent scalar reference, the
    /// forced scalar kernel, and (c) a freshly-built dense `Contingency`
    /// (API 1) over the same query/doc bucket codes. Counts are exact
    /// integers — equality is exact, not tolerance-based.
    ///
    /// Dims chosen to exercise the masked tail: 384 (qpb=6, all-tail, no
    /// full ZMM), 768 (qpb=12, one full ZMM + 4-word tail), 1024 (qpb=16,
    /// two full ZMMs, no tail).
    #[test]
    fn parity_scalar_simd_diagonal_dense() {
        for &dim in &[384usize, 768, 1024] {
            for bits in [2u8, 4u8] {
                let nb = 1usize << bits;
                let qpb = dim / 64;
                let n = 24;
                let corpus = make_corpus(0xC0FFEE ^ dim as u64 ^ (bits as u64) << 32, n, dim);
                let mut idx = MultiBucketBitmap::new(dim, bits);
                idx.add(&corpus);

                let mut rng = ChaCha8Rng::seed_from_u64(0x5EED ^ dim as u64);
                let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
                let q_bitmaps = idx.query_bitmaps_from_ranks(&query);
                let q_codes = query_codes(&q_bitmaps, nb, qpb, dim);

                for di in 0..n {
                    let doc_base = di * nb * qpb;
                    let doc = &idx.bitmaps[doc_base..doc_base + nb * qpb];

                    // (a) independent scalar reference table.
                    let want = reference_table(&q_bitmaps, doc, nb, qpb);

                    // forced scalar kernel.
                    let mut scalar = vec![0u32; nb * nb];
                    contingency_accumulate_scalar(&q_bitmaps, doc, nb, qpb, &mut scalar);
                    assert_eq!(
                        scalar, want,
                        "scalar kernel != reference (dim={dim}, bits={bits}, doc={di})",
                    );

                    // dispatched path (AVX-512 on this host).
                    let dispatched = idx.contingency_row(&q_bitmaps, di);
                    assert_eq!(
                        dispatched, want,
                        "dispatched (SIMD) != reference (dim={dim}, bits={bits}, doc={di})",
                    );

                    // diagonal fast path == diagonal of the full table.
                    let diag = idx.diagonal_overlap_row(&q_bitmaps, di);
                    let want_diag: Vec<u32> = (0..nb).map(|a| want[a * nb + a]).collect();
                    assert_eq!(
                        diag, want_diag,
                        "diagonal fast path != table diagonal (dim={dim}, bits={bits}, doc={di})",
                    );

                    // dense `Contingency` (API 1) over the same codes.
                    let d_codes = doc_codes(&idx, di);
                    let dense = Contingency::new(&q_codes, &d_codes, nb).unwrap();
                    assert_eq!(
                        dispatched,
                        dense.counts(),
                        "dispatched (SIMD) != dense Contingency (dim={dim}, bits={bits}, doc={di})",
                    );
                    // And the dense diagonal trace matches the fast-path sum.
                    assert_eq!(
                        diag.iter().sum::<u32>(),
                        dense.diagonal_agreement(),
                        "diagonal sum != dense diagonal_agreement (dim={dim}, bits={bits}, doc={di})",
                    );
                }
            }
        }
    }

    /// `project_all_batched` applies every projection to each doc's table
    /// built once, and agrees with per-doc `bilinear_score` (which rescans
    /// per projection) for the same weight matrices.
    #[test]
    fn project_all_batched_matches_bilinear_score() {
        let dim = 768;
        let bits = 4u8;
        let n = 40;
        let corpus = make_corpus(0xBA7C4, n, dim);
        let mut idx = MultiBucketBitmap::new(dim, bits);
        idx.add(&corpus);

        let mut rng = ChaCha8Rng::seed_from_u64(0xD0C5);
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let q_bitmaps = idx.query_bitmaps_from_ranks(&query);

        let outer = idx.outer_product_weights();
        let diagonal = idx.diagonal_weights();
        let banded = idx.banded_weights(1);
        let weights: Vec<&[f32]> = vec![&outer, &diagonal, &banded];

        let batched = idx.project_all_batched(&q_bitmaps, &weights);
        assert_eq!(batched.len(), n);
        for (di, row) in batched.iter().enumerate() {
            assert_eq!(row.len(), 3);
            // bilinear_score rescans the doc per weight matrix — the
            // batched path must reproduce it exactly (same integer table,
            // same weighted sum, same f32 accumulation order per matrix).
            for (p, w) in weights.iter().enumerate() {
                let want = idx.bilinear_score(&q_bitmaps, w, di);
                assert_eq!(
                    row[p], want,
                    "project_all_batched[{di}][{p}] != bilinear_score",
                );
            }
        }
    }

    /// Diagonal fast path sums to the `diagonal_weights` bilinear score.
    #[test]
    fn diagonal_row_sums_to_diagonal_bilinear_score() {
        let dim = 384;
        let bits = 2u8;
        let n = 16;
        let corpus = make_corpus(0xD1A6, n, dim);
        let mut idx = MultiBucketBitmap::new(dim, bits);
        idx.add(&corpus);
        let mut rng = ChaCha8Rng::seed_from_u64(0xF00D);
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let q_bitmaps = idx.query_bitmaps_from_ranks(&query);
        let w = idx.diagonal_weights();
        for di in 0..n {
            let diag_sum: u32 = idx.diagonal_overlap_row(&q_bitmaps, di).iter().sum();
            assert_eq!(diag_sum as f32, idx.bilinear_score(&q_bitmaps, &w, di));
        }
    }

    #[test]
    fn empty_index_project_all_batched_is_empty() {
        let idx = MultiBucketBitmap::new(256, 2);
        let q_bitmaps = idx.query_bitmaps_from_ranks(&vec![0.0f32; 256]);
        let diag = idx.diagonal_weights();
        let weights: Vec<&[f32]> = vec![&diag];
        assert!(idx.project_all_batched(&q_bitmaps, &weights).is_empty());
    }

    #[test]
    fn project_all_batched_no_weights_yields_empty_rows() {
        let dim = 256;
        let mut idx = MultiBucketBitmap::new(dim, 2);
        idx.add(&make_corpus(1, 5, dim));
        let q_bitmaps = idx.query_bitmaps_from_ranks(&vec![0.5f32; dim]);
        let weights: Vec<&[f32]> = Vec::new();
        let out = idx.project_all_batched(&q_bitmaps, &weights);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|row| row.is_empty()));
    }
}
