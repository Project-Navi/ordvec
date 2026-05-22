//! BitmapIndex: top-bucket bitmap per document (constant-composition).
//!
//! Each document is encoded as a `dim`-bit bitmap where bit j is set
//! iff the document's rank vector places coordinate j in the top
//! `n_top` coordinates (i.e., the top bucket under a coarse
//! equal-width partition of the rank axis).
//!
//! For `dim=1024, n_top=256` this is 128 B/doc — half of RankQuant b=2's
//! storage — and equivalent to the top bucket of RankQuant b=2.
//!
//! Score = `popcount(Q_bitmap AND D_bitmap) ∈ [0, n_top]`. The null
//! distribution for a random doc is hypergeometric
//! `H(N=dim, K=n_top, n=n_top)` with mean `n_top² / dim`, which lets
//! callers compute a closed-form p-value for the overlap — a property
//! magnitude quantizers don't have.
//!
//! Intended primary use: candidate generator for two-stage retrieval
//! (bitmap probe → top-M candidates → exact RankQuant rerank).

use rayon::prelude::*;

use super::util::{result_buffer_len, TopK};
use crate::rank::rank_transform;
use crate::SearchResults;

/// Top-bucket bitmap index for constant-composition coarse scoring.
pub struct BitmapIndex {
    dim: usize,
    n_top: usize,
    qwords_per_vec: usize,
    n_vectors: usize,
    /// Row-major `n_vectors * qwords_per_vec` u64 lanes.
    bitmaps: Vec<u64>,
}

impl BitmapIndex {
    pub fn new(dim: usize, n_top: usize) -> Self {
        assert_eq!(dim % 64, 0, "dim must be a multiple of 64");
        assert!(n_top > 0 && n_top < dim, "0 < n_top < dim");
        Self {
            dim,
            n_top,
            qwords_per_vec: dim / 64,
            n_vectors: 0,
            bitmaps: Vec::new(),
        }
    }

    /// Add documents. Each vector is rank-transformed; bit j of the
    /// document's bitmap is set iff coordinate j has rank ≥
    /// `dim - n_top` (equivalently: it is among the `n_top` largest
    /// coordinates of the document).
    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);
        let qpv = self.qwords_per_vec;
        let cutoff = (self.dim - self.n_top) as u16;
        let start = self.bitmaps.len();
        self.bitmaps.resize(start + n * qpv, 0u64);
        let dim = self.dim;
        self.bitmaps[start..]
            .par_chunks_mut(qpv)
            .zip(vectors.par_chunks(dim))
            .for_each(|(out, v)| {
                let ranks = rank_transform(v);
                for j in 0..dim {
                    if ranks[j] >= cutoff {
                        out[j / 64] |= 1u64 << (j % 64);
                    }
                }
            });
        self.n_vectors += n;
    }

    /// Build the query-side bitmap from the *FP32 query directly*
    /// (top `n_top` coordinates by value). This preserves the
    /// "rich query, cheap docs" asymmetry of the rank-cosine paper:
    /// the query side never gets rank-quantised.
    pub fn build_query_bitmap_fp32(&self, q: &[f32]) -> Vec<u64> {
        assert_eq!(q.len(), self.dim);
        // Index the dim sorted by |q[j]| desc; alternative: by q[j] desc.
        // We use raw value desc so the top bits flag where the query
        // points positively, matching the doc-side semantics.
        let mut idx: Vec<u16> = (0..self.dim as u16).collect();
        idx.sort_by(|&a, &b| {
            q[b as usize]
                .partial_cmp(&q[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut bm = vec![0u64; self.qwords_per_vec];
        for &j in &idx[..self.n_top] {
            bm[j as usize / 64] |= 1u64 << (j as usize % 64);
        }
        bm
    }

    /// Search: returns the top-`k` documents by popcount-overlap with
    /// the query's top-bucket bitmap. Scores are `popcount(Q AND D)`
    /// reported as f32 (in `[0, n_top]`).
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        // Clamp the user `k` to n_vectors BEFORE it sizes any
        // allocation. `vec![_; nq * k]` with an unclamped `k` (e.g.
        // usize::MAX) overflows Vec capacity and aborts. There can
        // never be more than n_vectors results, so the clamp is also
        // semantically correct — and it keeps the reported `k`, the
        // row stride (`par_chunks(k)`), and `k_eff` mutually
        // consistent.
        let k = k.min(self.n_vectors);
        let k_eff = k;
        let buf_len = result_buffer_len(nq, k);
        let mut scores_flat = vec![f32::NEG_INFINITY; buf_len];
        let mut indices_flat = vec![-1i64; buf_len];
        if k_eff == 0 {
            return SearchResults {
                scores: scores_flat,
                indices: indices_flat,
                nq,
                k,
            };
        }
        let dim = self.dim;
        let qpv = self.qwords_per_vec;
        let n = self.n_vectors;
        let bitmaps = &self.bitmaps;

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                let qb = self.build_query_bitmap_fp32(q);
                let mut top = TopK::new(k_eff);
                bitmap_scan(bitmaps, n, qpv, &qb, &mut top);
                top.finalize_into(out_scores, out_indices);
            });

        SearchResults {
            scores: scores_flat,
            indices: indices_flat,
            nq,
            k,
        }
    }

    /// Return the top-`m` candidate document indices for a single
    /// query, ordered by bitmap-overlap desc. Helper for two-stage
    /// retrieval.
    ///
    /// For large `m` this would exhibit O(N · m) behaviour through a
    /// streaming top-k buffer (each replacement triggers a linear
    /// recompute_min). Instead we scan once into a contiguous
    /// `Vec<u32>` of all N scores and `select_nth_unstable` the
    /// top-`m`: O(N + m log m). The 828 KiB temp at N=207k is
    /// cheap relative to the cost it saves at M ≥ 1000.
    pub fn top_m_candidates(&self, q: &[f32], m: usize) -> Vec<u32> {
        let m_eff = m.min(self.n_vectors);
        if m_eff == 0 {
            return Vec::new();
        }
        let qb = self.build_query_bitmap_fp32(q);
        let mut scores = vec![0u32; self.n_vectors];
        bitmap_scan_collect(
            &self.bitmaps,
            self.n_vectors,
            self.qwords_per_vec,
            &qb,
            &mut scores,
        );
        let mut idx: Vec<u32> = (0..self.n_vectors as u32).collect();
        // Composite-key partition on `(score desc, doc_id asc)` so
        // boundary ties at the m_eff cutoff produce a deterministic
        // survivor *set*, not just a deterministic post-partition
        // sort. Body bitmap scores have wider spread than a coarse
        // summary tier would, so boundary ties are rare — but the
        // structural nondeterminism is identical, hence the composite
        // key.
        let cmp = |a: &u32, b: &u32| {
            scores[*b as usize]
                .cmp(&scores[*a as usize])
                .then_with(|| a.cmp(b))
        };
        idx.select_nth_unstable_by(m_eff - 1, cmp);
        let mut head = idx[..m_eff].to_vec();
        head.sort_unstable_by(cmp);
        head
    }

    /// Batched variant: stream the bitmap corpus **once** and compute
    /// top-`m` candidate sets for `batch` queries in parallel. The
    /// per-query bandwidth drops by ~`batch`× because the doc stream
    /// is amortised, while compute per doc scales linearly in
    /// `batch` (AND-popcount-reduce is cheap relative to the L3→core
    /// bandwidth that dominates single-query scans).
    ///
    /// `queries` is a flat `batch * dim` f32 slice. Returns
    /// `Vec<Vec<u32>>` with one top-`m` set per query, sorted by
    /// overlap descending.
    pub fn top_m_candidates_batched(&self, queries: &[f32], m: usize) -> Vec<Vec<u32>> {
        let dim = self.dim;
        let batch = queries.len() / dim;
        assert_eq!(queries.len(), batch * dim);
        let m_eff = m.min(self.n_vectors);
        if batch == 0 || m_eff == 0 {
            return vec![Vec::new(); batch];
        }
        let n = self.n_vectors;
        let qpv = self.qwords_per_vec;

        // Build all query bitmaps up front. select_nth on the value-
        // sorted indices is asymptotically O(D + n_top log n_top), but
        // for D=1024 the existing full-sort path is fine — the wall
        // is dominated by the doc scan below.
        let mut q_batch = vec![0u64; batch * qpv];
        for bi in 0..batch {
            let qb = self.build_query_bitmap_fp32(&queries[bi * dim..(bi + 1) * dim]);
            q_batch[bi * qpv..(bi + 1) * qpv].copy_from_slice(&qb);
        }

        // One doc-scan pass writes `batch * n` u32 scores, layout
        // scores[bi * n + di]. At B=8, N=207k this is 6.6 MB — fits
        // in L2 per worker if rayon dispatches batch-sized chunks
        // to separate workers.
        let mut scores = vec![0u32; batch * n];
        bitmap_scan_collect_batched(&self.bitmaps, n, qpv, &q_batch, batch, &mut scores);

        // Per-query select_nth on contiguous score slices, in
        // parallel across queries. The slice borrows are disjoint.
        // Composite-key `(score desc, doc_id asc)` makes the
        // partition deterministic at boundary ties — see the
        // matching comment in `top_m_candidates`.
        let n_eff = n;
        scores
            .par_chunks(n_eff)
            .map(|q_scores| {
                let mut idx: Vec<u32> = (0..n_eff as u32).collect();
                let cmp = |a: &u32, b: &u32| {
                    q_scores[*b as usize]
                        .cmp(&q_scores[*a as usize])
                        .then_with(|| a.cmp(b))
                };
                idx.select_nth_unstable_by(m_eff - 1, cmp);
                let mut head = idx[..m_eff].to_vec();
                head.sort_unstable_by(cmp);
                head
            })
            .collect()
    }

    /// Convenience wrapper: chunks `queries` into groups of
    /// `batch_size` and runs each chunk through
    /// [`Self::top_m_candidates_batched`] in parallel via rayon. Use
    /// when the full query workload is larger than one batch fits
    /// efficiently in L2/L3.
    pub fn top_m_candidates_batched_chunked(
        &self,
        queries: &[f32],
        m: usize,
        batch_size: usize,
    ) -> Vec<Vec<u32>> {
        let dim = self.dim;
        let n_queries = queries.len() / dim;
        assert_eq!(queries.len(), n_queries * dim);
        if n_queries == 0 {
            return Vec::new();
        }
        let chunk_floats = batch_size * dim;
        queries
            .par_chunks(chunk_floats)
            .flat_map_iter(|chunk| self.top_m_candidates_batched(chunk, m).into_iter())
            .collect()
    }

    /// Compute bitmap-overlap scores for a sorted subset of doc IDs.
    /// `doc_ids` MUST be in ascending order (callers ensure this so
    /// the body scan is sequential over nearby bitmap rows — random-
    /// access gather would defeat the cache locality argument).
    /// `out` is filled in the same order as `doc_ids`.
    ///
    /// Public surface to support staged-pipeline callers that need to
    /// rescore a small survivor set under the exact body overlap.
    pub fn body_overlap_scores_subset(&self, q_bitmap: &[u64], doc_ids: &[u32], out: &mut [u32]) {
        let qpv = self.qwords_per_vec;
        assert_eq!(q_bitmap.len(), qpv);
        assert_eq!(out.len(), doc_ids.len());
        // CRITICAL: bound-check every doc_id BEFORE dispatch. The
        // AVX-512 kernel forwards `di` straight into
        // `bitmaps.as_ptr().add(di * qpv)` + `_mm512_loadu_si512`,
        // which is a raw load with no bounds check — an out-of-range
        // id reads past the heap allocation (silent garbage in
        // release, SEGV on a large id). The scalar fallback would
        // panic on the slice index, but only after the SIMD path has
        // already corrupted; assert here so both paths are covered.
        assert!(
            doc_ids.iter().all(|&di| (di as usize) < self.n_vectors),
            "body_overlap_scores_subset: doc_id out of range [0, {})",
            self.n_vectors,
        );
        debug_assert!(
            doc_ids.windows(2).all(|w| w[0] <= w[1]),
            "body_overlap_scores_subset: doc_ids must be sorted ascending",
        );

        #[cfg(target_arch = "x86_64")]
        let use_avx512vpop = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && qpv.is_multiple_of(8);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx512vpop = false;

        if use_avx512vpop {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                body_overlap_scores_subset_avx512vpop(&self.bitmaps, qpv, q_bitmap, doc_ids, out);
                return;
            }
        }
        for (i, &di) in doc_ids.iter().enumerate() {
            let off = (di as usize) * qpv;
            let doc = &self.bitmaps[off..off + qpv];
            let mut acc: u32 = 0;
            for w in 0..qpv {
                acc += (doc[w] & q_bitmap[w]).count_ones();
            }
            out[i] = acc;
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
    pub fn n_top(&self) -> usize {
        self.n_top
    }
    pub fn bytes_per_vec(&self) -> usize {
        self.qwords_per_vec * 8
    }
    pub fn byte_size(&self) -> usize {
        self.bitmaps.len() * std::mem::size_of::<u64>()
    }

    /// Persist to a `.tvbm` file. Format: 17-byte header + u64 bitmaps LE.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_bitmap(path, self.dim, self.n_top, self.n_vectors, &self.bitmaps)
    }

    /// Load from a `.tvbm` file produced by [`Self::write`].
    ///
    /// Returns `io::Error::InvalidData` on any constructor-invariant
    /// violation (`load_bitmap` already validates dim/n_top/n_vectors;
    /// this method only verifies the payload length matches the
    /// expected `n_vectors * dim / 64` u64 lanes).
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (dim, n_top, n_vectors, bitmaps) = crate::rank_io::load_bitmap(path)?;
        let qpv = dim / 64;
        let expected = n_vectors.saturating_mul(qpv);
        if bitmaps.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TVBM payload length {} does not match expected {expected} u64 lanes",
                    bitmaps.len(),
                ),
            ));
        }
        Ok(Self {
            dim,
            n_top,
            qwords_per_vec: qpv,
            n_vectors,
            bitmaps,
        })
    }
}

/// Streaming bitmap scan: AND-popcount each doc bitmap against the
/// query bitmap. Uses runtime feature detection for AVX-512 VPOPCNTDQ
/// (one VPOPCNTQ over 8 u64 lanes), otherwise falls back to scalar
/// `u64::count_ones()` which Zen 5 retires at 1/cycle.
fn bitmap_scan(bitmaps: &[u64], n: usize, qpv: usize, q: &[u64], top: &mut TopK) {
    debug_assert_eq!(q.len(), qpv);

    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vpopcntdq")
        && qpv.is_multiple_of(8);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            bitmap_scan_avx512vpop(bitmaps, n, qpv, q, top);
            return;
        }
    }
    bitmap_scan_scalar(bitmaps, n, qpv, q, top);
}

fn bitmap_scan_scalar(bitmaps: &[u64], n: usize, qpv: usize, q: &[u64], top: &mut TopK) {
    for di in 0..n {
        let doc = &bitmaps[di * qpv..(di + 1) * qpv];
        let mut acc: u32 = 0;
        for w in 0..qpv {
            acc += (doc[w] & q[w]).count_ones();
        }
        top.maybe_insert(acc as f32, di);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn bitmap_scan_avx512vpop(bitmaps: &[u64], n: usize, qpv: usize, q: &[u64], top: &mut TopK) {
    use std::arch::x86_64::*;
    debug_assert_eq!(qpv % 8, 0, "AVX-512 bitmap scan needs qpv % 8 == 0");
    let lanes = qpv / 8;
    let mut q_zmms: Vec<__m512i> = Vec::with_capacity(lanes);
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for l in 0..lanes {
        q_zmms.push(_mm512_loadu_si512(q.as_ptr().add(l * 8) as *const __m512i));
    }
    for di in 0..n {
        let doc_ptr = bitmaps.as_ptr().add(di * qpv) as *const __m512i;
        let mut acc_zmm = _mm512_setzero_si512();
        #[allow(clippy::needless_range_loop)]
        // indexed access is clearer / matches the kernel layout
        for l in 0..lanes {
            let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
            let and_zmm = _mm512_and_si512(d_zmm, q_zmms[l]);
            let pop_zmm = _mm512_popcnt_epi64(and_zmm);
            acc_zmm = _mm512_add_epi64(acc_zmm, pop_zmm);
        }
        let acc_sum: i64 = _mm512_reduce_add_epi64(acc_zmm);
        top.maybe_insert(acc_sum as f32, di);
    }
}

/// Scan all N docs and write the raw popcount-overlap score into
/// `scores[di]`. No top-k maintenance, no allocation per doc, no
/// O(N · k) tax — used by [`BitmapIndex::top_m_candidates`] for large
/// M where the streaming top-k path would dominate.
fn bitmap_scan_collect(bitmaps: &[u64], n: usize, qpv: usize, q: &[u64], scores: &mut [u32]) {
    debug_assert_eq!(scores.len(), n);
    debug_assert_eq!(q.len(), qpv);

    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vpopcntdq")
        && qpv.is_multiple_of(8);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            bitmap_scan_collect_avx512vpop(bitmaps, n, qpv, q, scores);
            return;
        }
    }
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for di in 0..n {
        let doc = &bitmaps[di * qpv..(di + 1) * qpv];
        let mut acc: u32 = 0;
        for w in 0..qpv {
            acc += (doc[w] & q[w]).count_ones();
        }
        scores[di] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn bitmap_scan_collect_avx512vpop(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q: &[u64],
    scores: &mut [u32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(qpv % 8, 0);
    let lanes = qpv / 8;
    let mut q_zmms: Vec<__m512i> = Vec::with_capacity(lanes);
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for l in 0..lanes {
        q_zmms.push(_mm512_loadu_si512(q.as_ptr().add(l * 8) as *const __m512i));
    }
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for di in 0..n {
        let doc_ptr = bitmaps.as_ptr().add(di * qpv) as *const __m512i;
        let mut acc_zmm = _mm512_setzero_si512();
        for l in 0..lanes {
            let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
            let and_zmm = _mm512_and_si512(d_zmm, q_zmms[l]);
            let pop_zmm = _mm512_popcnt_epi64(and_zmm);
            acc_zmm = _mm512_add_epi64(acc_zmm, pop_zmm);
        }
        let acc_sum: i64 = _mm512_reduce_add_epi64(acc_zmm);
        scores[di] = acc_sum as u32;
    }
}

// -------------------------------------------------------------------
// Batched bitmap scan: process B queries against the same doc stream.
//
// The single-query path streams ~26 MB of bitmap data for a single
// 200-query bench-batch at D=1024, N=207k — 5.2 GB of bandwidth total.
// The batched path loads each doc once and computes B overlap scores
// against B pre-loaded query bitmaps, amortising the bitmap stream
// across the batch. Total bandwidth scales as N·qpv·8 + B·qpv·8
// (queries) + B·N·4 (scores out) — the N·qpv·8 term is shared.
//
// At qpv=16 (D=1024), each doc is 2 ZMMs. For B queries we keep
// B*lanes (=B*2) query ZMMs preloaded and run B AND-popcount-reduce
// cycles per doc. Compute per doc grows linearly in B; the doc load
// is paid once.
// -------------------------------------------------------------------

/// Scalar fallback for the batched scan. Used when AVX-512 VPOPCNTDQ
/// is unavailable or when `qpv % 8 != 0`.
fn bitmap_scan_collect_batched_scalar(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q_batch: &[u64],
    batch: usize,
    scores: &mut [u32],
) {
    debug_assert_eq!(q_batch.len(), batch * qpv);
    debug_assert_eq!(scores.len(), batch * n);
    for di in 0..n {
        let doc = &bitmaps[di * qpv..(di + 1) * qpv];
        for bi in 0..batch {
            let q = &q_batch[bi * qpv..(bi + 1) * qpv];
            let mut acc: u32 = 0;
            for w in 0..qpv {
                acc += (doc[w] & q[w]).count_ones();
            }
            scores[bi * n + di] = acc;
        }
    }
}

// Chunk size for the AVX-512 batched kernel: number of queries the
// inner loop accumulates against a single doc-lane load. Chosen at 8
// because (a) the stack-resident `accs: [__m512i; CHUNK]` array
// reliably promotes to 8 ZMM registers under LLVM, (b) at CHUNK=8 on
// Zen 5 (32 ZMM regs total) we have 8 accs + lanes doc/query temps
// + spillover headroom, and (c) empirical sweeps show CHUNK=8 sits
// at the bandwidth/register-pressure inflection. Larger `batch` is
// processed in multiple CHUNK-sized passes through the bitmap stream
// — each pass amortises the doc load across CHUNK queries.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const BATCHED_AVX512_CHUNK: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn bitmap_scan_collect_batched_avx512vpop(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q_batch: &[u64],
    batch: usize,
    scores: &mut [u32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(qpv % 8, 0);
    debug_assert_eq!(q_batch.len(), batch * qpv);
    debug_assert_eq!(scores.len(), batch * n);
    let lanes = qpv / 8;
    const CHUNK: usize = BATCHED_AVX512_CHUNK;

    // Pre-load all batch * lanes query ZMMs once. For typical
    // (batch=8, lanes=2) this is 16 __m512i of register-equivalent
    // state, which fits in the 32-ZMM file alongside the per-chunk
    // accs and doc lane temps.
    let mut q_zmms: Vec<__m512i> = Vec::with_capacity(batch * lanes);
    for bi in 0..batch {
        for l in 0..lanes {
            q_zmms.push(_mm512_loadu_si512(
                q_batch.as_ptr().add(bi * qpv + l * 8) as *const __m512i
            ));
        }
    }

    // Hot path: process whole CHUNK-sized groups. The inner `for bi
    // in 0..CHUNK` is bounded by a *const*, so LLVM unrolls it and
    // promotes the `accs: [__m512i; CHUNK]` stack array to ZMM
    // registers — that's the property that keeps the kernel
    // competitive with the single-query AVX-512 path on a per-query
    // basis, plus the bandwidth amortisation. A runtime-bounded
    // `0..chunk` loop would force `accs[bi]` to spill to stack
    // memory and double per-doc latency.
    let mut chunk_start = 0usize;
    while chunk_start + CHUNK <= batch {
        for di in 0..n {
            let mut accs: [__m512i; CHUNK] = [_mm512_setzero_si512(); CHUNK];
            let doc_ptr = bitmaps.as_ptr().add(di * qpv) as *const __m512i;
            for l in 0..lanes {
                let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                for bi in 0..CHUNK {
                    let q_zmm = q_zmms[(chunk_start + bi) * lanes + l];
                    let and_zmm = _mm512_and_si512(d_zmm, q_zmm);
                    let pop_zmm = _mm512_popcnt_epi64(and_zmm);
                    accs[bi] = _mm512_add_epi64(accs[bi], pop_zmm);
                }
            }
            for bi in 0..CHUNK {
                let acc_sum: i64 = _mm512_reduce_add_epi64(accs[bi]);
                scores[(chunk_start + bi) * n + di] = acc_sum as u32;
            }
        }
        chunk_start += CHUNK;
    }
    // Tail path: any remaining `batch % CHUNK` queries. Slower per
    // doc (runtime-bounded inner loop, accs[bi] may spill) but the
    // tail runs once per kernel call, not once per doc — total cost
    // is at most CHUNK-1 queries of slower scan, dominated by the
    // hot path for any batch > 1.
    let tail = batch - chunk_start;
    if tail > 0 {
        for di in 0..n {
            let mut accs: [__m512i; CHUNK] = [_mm512_setzero_si512(); CHUNK];
            let doc_ptr = bitmaps.as_ptr().add(di * qpv) as *const __m512i;
            for l in 0..lanes {
                let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                for bi in 0..tail {
                    let q_zmm = q_zmms[(chunk_start + bi) * lanes + l];
                    let and_zmm = _mm512_and_si512(d_zmm, q_zmm);
                    let pop_zmm = _mm512_popcnt_epi64(and_zmm);
                    accs[bi] = _mm512_add_epi64(accs[bi], pop_zmm);
                }
            }
            for bi in 0..tail {
                let acc_sum: i64 = _mm512_reduce_add_epi64(accs[bi]);
                scores[(chunk_start + bi) * n + di] = acc_sum as u32;
            }
        }
    }
}

/// Batched bitmap scan: writes `scores[bi * n + di]` = popcount overlap
/// for query `bi` against doc `di`, for all `bi ∈ [0, batch)` and
/// `di ∈ [0, n)`. Dispatches to the AVX-512 VPOPCNTDQ kernel when
/// available (qpv % 8 == 0), else falls back to scalar.
fn bitmap_scan_collect_batched(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q_batch: &[u64],
    batch: usize,
    scores: &mut [u32],
) {
    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vpopcntdq")
        && qpv.is_multiple_of(8);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            bitmap_scan_collect_batched_avx512vpop(bitmaps, n, qpv, q_batch, batch, scores);
            return;
        }
    }
    bitmap_scan_collect_batched_scalar(bitmaps, n, qpv, q_batch, batch, scores);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn body_overlap_scores_subset_avx512vpop(
    bitmaps: &[u64],
    qpv: usize,
    q_bitmap: &[u64],
    doc_ids: &[u32],
    out: &mut [u32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(qpv % 8, 0);
    let lanes = qpv / 8;
    let mut q_zmms: Vec<__m512i> = Vec::with_capacity(lanes);
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for l in 0..lanes {
        q_zmms.push(_mm512_loadu_si512(
            q_bitmap.as_ptr().add(l * 8) as *const __m512i
        ));
    }
    for (i, &di) in doc_ids.iter().enumerate() {
        let doc_ptr = bitmaps.as_ptr().add((di as usize) * qpv) as *const __m512i;
        let mut acc_zmm = _mm512_setzero_si512();
        #[allow(clippy::needless_range_loop)]
        // indexed access is clearer / matches the kernel layout
        for l in 0..lanes {
            let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
            let and_zmm = _mm512_and_si512(d_zmm, q_zmms[l]);
            let pop_zmm = _mm512_popcnt_epi64(and_zmm);
            acc_zmm = _mm512_add_epi64(acc_zmm, pop_zmm);
        }
        let acc_sum: i64 = _mm512_reduce_add_epi64(acc_zmm);
        out[i] = acc_sum as u32;
    }
}
