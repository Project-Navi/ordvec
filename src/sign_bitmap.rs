//! Sign-cosine bitmap retrieval substrate.
//!
//! 1-bit-per-coord quantization at the **data-independent threshold
//! of zero**: bit j of doc d is set iff `d.embedding[j] > 0`. Storage
//! is `dim/8` bytes per doc (128 B at D=1024).
//!
//! This is the **SimHash family** primitive (Charikar 2002) applied to
//! native embedding coords rather than random projections. For
//! contrastively-trained embeddings (e.g. BGE or OpenAI ada), the
//! native coord axes already carry semantically-aligned signal, so the
//! sign pattern alone preserves much of the angular structure that cosine
//! ranking depends on — which is what lets a `dim/8`-byte sign code serve
//! as a useful candidate-generation substrate.
//!
//! Score: `agreement(q, d) = dim - popcount(q ^ d)`. The kernel
//! computes the per-doc Hamming distance via popcount(XOR); the
//! candidate selector takes top-M docs by **lowest** Hamming
//! (= **highest** agreement).
//!
//! This is a separate sign-agreement primitive. It is not a constant-weight
//! bitmap space and is not covered by [`crate::Bitmap`]'s hypergeometric
//! overlap-tail theorem.
//!
//! Kernel architecture mirrors [`crate::Bitmap`] (single-query
//! and CHUNK=8 batched hot+tail paths under AVX-512 VPOPCNTDQ). The
//! only material difference is `_mm512_xor_si512` in place of
//! `_mm512_and_si512` and an ascending tie-broken composite-key
//! selection on Hamming distance.
//!
//! # Dimensions and the AVX-512 kernel
//!
//! `dim` must be a multiple of 64. On a host with AVX-512 VPOPCNTDQ **every**
//! such `dim` runs the vectorized scan: the kernel processes whole 512-bit
//! (8 × u64) groups, then handles any trailing `(dim / 64) % 8` words with a
//! single masked load (`_mm512_maskz_loadu_epi64`). Dimensions whose 64-bit
//! word count is a multiple of 8 — 512, 1024, 1536, … — have no tail; others
//! (e.g. **384, 768**, the common BGE/MiniLM widths) pay **one extra masked
//! chunk** — a few percent, so 768 ≈ 1024 — instead of falling back to the
//! scalar path. See [`crate::avx512vpop_supported`].

use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::OrdvecError;

/// Candidate sets for a query batch in CSR (compressed-sparse-row) form, as
/// produced by [`SignBitmap::top_m_candidates_batched_serial_csr`].
///
/// Invariants (guaranteed and tested):
/// - `offsets.len() == query_count() + 1`
/// - `offsets[0] == 0`
/// - `offsets` is monotonic non-decreasing
/// - `*offsets.last().unwrap() == candidates.len()`
/// - row `i` is `candidates[offsets[i]..offsets[i + 1]]`
///
/// Fields are `pub` for zero-copy hand-off (same precedent as
/// [`crate::SearchResults`]); the invariants above are part of the stable API.
#[derive(Clone, Debug)]
#[must_use = "candidate generation scans the corpus; dropping the result discards that work"]
pub struct CandidateBatch {
    pub candidates: Vec<u32>,
    pub offsets: Vec<usize>,
}

impl CandidateBatch {
    /// Number of queries in the batch (`offsets.len() - 1`).
    pub fn query_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    /// Candidate row for query `qi`, or `None` if `qi >= query_count()`.
    pub fn candidates_for_query(&self, qi: usize) -> Option<&[u32]> {
        let start = *self.offsets.get(qi)?;
        let end = *self.offsets.get(qi + 1)?;
        Some(&self.candidates[start..end])
    }
    /// `true` iff there are **no queries** (`query_count() == 0`) — NOT iff
    /// there are no candidates. A 3-query batch with zero candidates per query
    /// is not empty.
    pub fn is_empty(&self) -> bool {
        self.query_count() == 0
    }
    /// `true` iff there are no candidates across all queries.
    pub fn has_no_candidates(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// Index storing a 1-bit sign-cosine fingerprint per document.
///
/// Storage: `dim / 8` bytes per doc. Dim must be a multiple of 64
/// (so the u64-packed layout has no straddling tail bits — same
/// invariant as [`crate::Bitmap`]).
pub struct SignBitmap {
    dim: usize,
    qwords_per_vec: usize,
    n_vectors: usize,
    /// Row-major `n_vectors * qwords_per_vec` u64s. Bit j of doc di
    /// is at `bitmaps[di*qpv + j/64] >> (j%64) & 1`.
    bitmaps: Vec<u64>,
}

impl std::fmt::Debug for SignBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignBitmap")
            .field("dim", &self.dim)
            .field("n_vectors", &self.n_vectors)
            .field("bytes_per_vector", &self.bytes_per_vec())
            .finish()
    }
}

impl SignBitmap {
    pub fn validate_dim(dim: usize) -> Result<(), OrdvecError> {
        if dim == 0 {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: "must be > 0".to_string(),
            });
        }
        if !dim.is_multiple_of(64) {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: "must be a multiple of 64".to_string(),
            });
        }
        if dim > crate::rank_io::MAX_SIGN_BITMAP_DIM {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: format!(
                    "must be <= MAX_SIGN_BITMAP_DIM (= {})",
                    crate::rank_io::MAX_SIGN_BITMAP_DIM
                ),
            });
        }
        Ok(())
    }

    /// Build an empty index for `dim`-dimensional embeddings.
    ///
    /// `dim` must be a multiple of 64 in
    /// `[64, crate::rank_io::MAX_SIGN_BITMAP_DIM]`. `dim = 0` is
    /// rejected because it would create an index whose
    /// `qwords_per_vec = 0`, dividing by zero inside [`Self::add`].
    /// The upper bound matches the loader so any index built here
    /// can be persisted via [`Self::write`] and reloaded via
    /// [`Self::load`] — without it, `new` could produce indices the
    /// loader refuses to round-trip (the issue Codex caught after the
    /// first `.ovsb` revision used [`crate::rank_io::MAX_DIM`]'s
    /// rank-storage `u16::MAX` cap, which doesn't apply to sign
    /// bitmaps).
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert_eq!(dim % 64, 0, "dim must be a multiple of 64");
        assert!(
            dim <= crate::rank_io::MAX_SIGN_BITMAP_DIM,
            "dim must be <= MAX_SIGN_BITMAP_DIM (= {})",
            crate::rank_io::MAX_SIGN_BITMAP_DIM,
        );
        Self {
            dim,
            qwords_per_vec: dim / 64,
            n_vectors: 0,
            bitmaps: Vec::new(),
        }
    }

    /// Add documents. Each doc is sign-quantized at threshold zero:
    /// bit j is set iff `vectors[di*dim + j] > 0.0`. The sign of
    /// exactly zero (rare in practice for trained embeddings) is
    /// treated as negative (bit unset).
    ///
    /// # Panics
    /// Panics if the index would grow beyond `rank_io::MAX_VECTORS` documents
    /// — the supported capacity. Candidate APIs materialise document IDs as
    /// `u32`; `MAX_VECTORS` sits well below `u32::MAX` and matches the on-disk
    /// loader's `n_vectors` ceiling. (Bounds the count, not the byte payload —
    /// see the loaders' separate `MAX_PAYLOAD` cap.) Also panics if the
    /// resulting row-major buffer length would overflow `usize` (reachable only
    /// on 32-bit targets — see `util::checked_new_count`).
    pub fn add(&mut self, vectors: &[f32]) {
        crate::util::assert_all_finite(vectors);
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);
        let new_n = crate::util::checked_new_count(self.n_vectors, n, self.qwords_per_vec);
        let qpv = self.qwords_per_vec;
        let dim = self.dim;
        let start = self.bitmaps.len();
        self.bitmaps.resize(start + n * qpv, 0u64);
        self.bitmaps[start..]
            .par_chunks_mut(qpv)
            .zip(vectors.par_chunks(dim))
            .for_each(|(out, v)| {
                for j in 0..dim {
                    if v[j] > 0.0 {
                        out[j / 64] |= 1u64 << (j % 64);
                    }
                }
            });
        self.n_vectors = new_n;
    }

    /// Build the query-side sign bitmap. Same threshold semantics as
    /// [`Self::add`]: bit j set iff `q[j] > 0.0`.
    pub fn build_query_bitmap(&self, q: &[f32]) -> Vec<u64> {
        assert_eq!(q.len(), self.dim);
        crate::util::assert_all_finite(q);
        let mut bm = vec![0u64; self.qwords_per_vec];
        for j in 0..self.dim {
            if q[j] > 0.0 {
                bm[j / 64] |= 1u64 << (j % 64);
            }
        }
        bm
    }

    /// Return the top-`m` candidate doc IDs ranked by **highest
    /// sign agreement** (equivalently: lowest Hamming distance) with
    /// `q`. Selection uses the composite key
    /// `(hamming ascending, doc_id ascending)` so boundary ties at
    /// `m_eff` produce a deterministic survivor set across runs and
    /// SIMD dispatch paths — same audit discipline as
    /// [`crate::Bitmap::top_m_candidates`].
    #[must_use = "this scans the corpus to generate candidates; dropping the result discards that work"]
    /// Streamed exact top-m selection shared by [`Self::top_m_candidates`]
    /// and [`Self::top_m_candidates_batched_serial_csr`]: the corpus is
    /// scanned once per call in L2-sized doc blocks, each hot block is
    /// scored against every query (in small query tiles), and per-query
    /// bounded min-m collectors keyed by `(hamming, doc_id)` select exactly
    /// the lexicographic top-m — bit-identical to a full sort, independent
    /// of processing order. Serial by contract: no rayon.
    fn top_m_candidates_streamed(&self, queries: &[f32], m_eff: usize) -> Vec<Vec<u32>> {
        const TILE_QUERIES: usize = 32;
        const BLOCK_BYTES: usize = 256 * 1024;

        let dim = self.dim;
        debug_assert!(
            queries.len().is_multiple_of(dim),
            "queries buffer must be a whole number of rows"
        );
        let nq = queries.len() / dim;
        let qpv = self.qwords_per_vec;
        let n = self.n_vectors;
        debug_assert!(m_eff >= 1 && m_eff <= n);

        let mut q_bitmaps = vec![0u64; nq * qpv];
        for qi in 0..nq {
            let qb = self.build_query_bitmap(&queries[qi * dim..(qi + 1) * dim]);
            q_bitmaps[qi * qpv..(qi + 1) * qpv].copy_from_slice(&qb);
        }

        let block_docs = (BLOCK_BYTES / (qpv * 8)).max(64).min(n);
        let tile = TILE_QUERIES.min(nq);
        let mut block_scores = vec![0u32; tile * block_docs];
        // Max-heap keeps the current worst kept key at the top, so the
        // retained set is always the m lexicographically smallest
        // (hamming, doc_id) keys seen so far.
        // Selection state is O(nq * m_eff) on top of the CSR output — an
        // explicit checked bound (32-bit/wasm32 targets can overflow the
        // multiplication) with a clear message, per the crate's
        // checked-allocation discipline. Exact per-heap reservation of
        // m_eff + 1 is deliberate: gradual growth would double-allocate to
        // the next power of two (~2x m_eff peak per query); callers with
        // extreme nq * m_eff should tile the query batch (as OrdinalDB's
        // chunk scheduler does).
        let selection_cells = nq.checked_mul(m_eff).unwrap_or_else(|| {
            panic!("selection state nq ({nq}) * m ({m_eff}) overflows usize; tile the query batch")
        });
        let _ = selection_cells;
        let mut heaps: Vec<BinaryHeap<(u32, u32)>> = (0..nq)
            .map(|_| BinaryHeap::with_capacity(m_eff + 1))
            .collect();
        // Cached copy of each full heap's worst kept hamming. Doc ids visit
        // each heap strictly ascending (d ascends within a row, blocks
        // ascend), so a candidate tying the worst hamming always loses the
        // (hamming, doc_id) tie-break — once full, the boundary test
        // reduces to one u32 compare against this register. u32::MAX while
        // filling (hamming <= dim can never reach it).
        let mut worst_bounds = vec![u32::MAX; nq];

        let mut block_start = 0usize;
        while block_start < n {
            let bn = block_docs.min(n - block_start);
            let block = &self.bitmaps[block_start * qpv..(block_start + bn) * qpv];
            let mut tile_start = 0usize;
            while tile_start < nq {
                let tq = tile.min(nq - tile_start);
                let qb_tile = &q_bitmaps[tile_start * qpv..(tile_start + tq) * qpv];
                let scores = &mut block_scores[..tq * bn];
                sign_scan_collect_batched(block, bn, qpv, qb_tile, tq, scores);
                for ti in 0..tq {
                    let heap = &mut heaps[tile_start + ti];
                    let worst = &mut worst_bounds[tile_start + ti];
                    let row = &scores[ti * bn..(ti + 1) * bn];
                    for (d, &hamming) in row.iter().enumerate() {
                        if hamming >= *worst {
                            continue;
                        }
                        heap.push((hamming, (block_start + d) as u32));
                        if heap.len() > m_eff {
                            heap.pop();
                        }
                        if heap.len() == m_eff {
                            *worst = heap.peek().expect("full collector").0;
                        }
                    }
                }
                tile_start += tq;
            }
            block_start += bn;
        }

        heaps
            .into_iter()
            .map(|heap| {
                let mut kept = heap.into_vec();
                kept.sort_unstable();
                kept.into_iter().map(|(_, doc)| doc).collect()
            })
            .collect()
    }

    pub fn top_m_candidates(&self, q: &[f32], m: usize) -> Vec<u32> {
        assert_eq!(q.len(), self.dim);
        crate::util::assert_all_finite(q);
        let m_eff = m.min(self.n_vectors);
        if m_eff == 0 {
            return Vec::new();
        }
        // Single-query stays on the dense partition path: with one query
        // there is no scan to share, and select_nth_unstable_by (O(n)
        // average) measurably beats an O(n log m) bounded heap for m in the
        // hundreds at small/medium n (audit: +50-90% regression otherwise).
        let qb = self.build_query_bitmap(q);
        let mut scores = vec![0u32; self.n_vectors]; // Hamming distance per doc
        sign_scan_collect(
            &self.bitmaps,
            self.n_vectors,
            self.qwords_per_vec,
            &qb,
            &mut scores,
        );
        let mut idx: Vec<u32> = (0..self.n_vectors as u32).collect();
        // Ascending Hamming = best candidates first. Composite key
        // ensures deterministic partition at boundary ties.
        let cmp = |a: &u32, b: &u32| {
            scores[*a as usize]
                .cmp(&scores[*b as usize])
                .then_with(|| a.cmp(b))
        };
        idx.select_nth_unstable_by(m_eff - 1, cmp);
        let mut head = idx[..m_eff].to_vec();
        head.sort_unstable_by(cmp);
        head
    }

    /// Batched variant: stream the sign bitmaps **once** and produce
    /// top-`m` candidate sets for `batch` queries in parallel. Mirrors
    /// [`crate::Bitmap::top_m_candidates_batched`] in kernel
    /// shape (CHUNK=8 hot + tail) and tie-break semantics.
    #[must_use = "this scans the corpus per query to generate candidates; dropping the result discards that work"]
    pub fn top_m_candidates_batched(&self, queries: &[f32], m: usize) -> Vec<Vec<u32>> {
        let dim = self.dim;
        let batch = queries.len() / dim;
        assert_eq!(queries.len(), batch * dim);
        crate::util::assert_all_finite(queries);
        let m_eff = m.min(self.n_vectors);
        if batch == 0 || m_eff == 0 {
            return vec![Vec::new(); batch];
        }
        let n = self.n_vectors;
        let qpv = self.qwords_per_vec;

        // `batch * qpv` and `batch * n` (below) are checked: on a 32-bit target
        // (wasm32) a moderate corpus and large query batch can overflow `usize`,
        // silently under-sizing these buffers and then indexing out of bounds.
        let q_batch_len = batch
            .checked_mul(qpv)
            .expect("batched query-bitmap buffer length (batch * qpv) overflows usize");
        let mut q_batch = vec![0u64; q_batch_len];
        for bi in 0..batch {
            let qb = self.build_query_bitmap(&queries[bi * dim..(bi + 1) * dim]);
            q_batch[bi * qpv..(bi + 1) * qpv].copy_from_slice(&qb);
        }

        let scores_len = batch
            .checked_mul(n)
            .expect("batched candidate score buffer length (batch * n) overflows usize");
        let mut scores = vec![0u32; scores_len];
        sign_scan_collect_batched(&self.bitmaps, n, qpv, &q_batch, batch, &mut scores);

        let n_eff = n;
        scores
            .par_chunks(n_eff)
            .map(|q_scores| {
                let mut idx: Vec<u32> = (0..n_eff as u32).collect();
                let cmp = |a: &u32, b: &u32| {
                    q_scores[*a as usize]
                        .cmp(&q_scores[*b as usize])
                        .then_with(|| a.cmp(b))
                };
                idx.select_nth_unstable_by(m_eff - 1, cmp);
                let mut head = idx[..m_eff].to_vec();
                head.sort_unstable_by(cmp);
                head
            })
            .collect()
    }

    /// Serial (NO rayon) CSR candidate generation for a query batch. Returns a
    /// [`CandidateBatch`]; row `qi` is the top-`m` candidate doc ids for query
    /// `qi`, ordered `(hamming ascending, doc_id ascending)`, of length
    /// `m.min(self.len())`.
    ///
    /// This is the caller-owned integration primitive: it never enters rayon,
    /// so a caller (e.g. a database) parallelises across queries with its own
    /// pool. (The existing [`Self::top_m_candidates_batched`] remains the
    /// internally-parallel standalone convenience.)
    ///
    /// The internals stream the corpus **once per call** in L2-sized doc
    /// blocks, scoring every query of the call against each hot block and
    /// selecting per-query top-m with bounded `(hamming, doc_id)` collectors
    /// — per-query corpus traffic drops by the call's query count relative
    /// to the historical per-query rescan. The CSR output contract is
    /// unchanged and bit-identical to the previous implementation.
    ///
    /// "Serial" scopes the scan and selection: no rayon is entered for the
    /// candidate work, so callers own that parallelism. Input finite-
    /// validation MAY briefly use the global rayon pool for large query
    /// buffers (order-independent boolean reduction; deterministic).
    ///
    /// # Example
    /// ```no_run
    /// use ordvec::SignBitmap;
    /// # let (dim, m) = (1024usize, 256usize);
    /// let sign = SignBitmap::new(dim);
    /// # let queries = vec![0.0f32; dim * 64];
    /// let cb = sign.top_m_candidates_batched_serial_csr(&queries, m);
    /// // CSR: query qi's candidate row is
    /// // `cb.candidates[cb.offsets[qi]..cb.offsets[qi + 1]]`. Pass `cb.offsets`
    /// // and `cb.candidates` straight into
    /// // `RankQuant::search_asymmetric_subset_batched_serial_into`.
    /// let _row0 = &cb.candidates[cb.offsets[0]..cb.offsets[1]];
    /// ```
    #[must_use = "this scans the corpus per query to generate candidates; dropping the result discards that work"]
    pub fn top_m_candidates_batched_serial_csr(&self, queries: &[f32], m: usize) -> CandidateBatch {
        let dim = self.dim;
        assert!(
            queries.len().is_multiple_of(dim),
            "queries length {} must be a multiple of dim {dim}",
            queries.len()
        );
        crate::util::assert_all_finite(queries);
        let nq = queries.len() / dim;
        let m_eff = m.min(self.n_vectors);
        let mut offsets = Vec::with_capacity(nq + 1);
        offsets.push(0usize);
        let mut candidates = Vec::with_capacity(nq.checked_mul(m_eff).unwrap_or_else(|| {
            panic!("CSR output nq ({nq}) * m ({m_eff}) overflows usize; tile the query batch")
        }));
        if nq == 0 || m_eff == 0 {
            offsets.extend(std::iter::repeat_n(0usize, nq));
            return CandidateBatch {
                candidates,
                offsets,
            };
        }
        for row in self.top_m_candidates_streamed(queries, m_eff) {
            candidates.extend_from_slice(&row);
            offsets.push(candidates.len());
        }
        CandidateBatch {
            candidates,
            offsets,
        }
    }

    /// Score every indexed document against one query and return dense
    /// sign-agreement counts aligned by document id.
    ///
    /// `scores[di] = dim - popcount(q_bits ^ doc_bits[di])`, so higher is
    /// better. This is a full-corpus scoring primitive, not a retrieval helper:
    /// it performs no top-k selection and no sorting.
    #[must_use = "this scans the corpus to score every document; dropping the result discards that work"]
    pub fn score_all(&self, q: &[f32]) -> Vec<u32> {
        let qb = self.build_query_bitmap(q);
        let mut scores = vec![0u32; self.n_vectors]; // Hamming distance first.
        sign_scan_collect(
            &self.bitmaps,
            self.n_vectors,
            self.qwords_per_vec,
            &qb,
            &mut scores,
        );
        let dim = u32::try_from(self.dim).expect("sign bitmap dim fits u32");
        scores.par_iter_mut().for_each(|h| *h = dim - *h);
        scores
    }

    /// Batched dense scoring. Returns a flat row-major buffer of full-corpus
    /// sign-agreement scores of length `batch * len(index)`, with columns
    /// aligned by document id and no sorting.
    #[must_use = "this scans the corpus to score every document per query; dropping the result discards that work"]
    pub fn score_all_batched_flat(&self, queries: &[f32]) -> Vec<u32> {
        let dim = self.dim;
        let batch = queries.len() / dim;
        assert_eq!(queries.len(), batch * dim);
        if batch == 0 {
            return Vec::new();
        }
        let n = self.n_vectors;
        let qpv = self.qwords_per_vec;

        let q_batch_len = batch
            .checked_mul(qpv)
            .expect("batched query-bitmap buffer length (batch * qpv) overflows usize");
        let mut q_batch = vec![0u64; q_batch_len];
        for bi in 0..batch {
            let qb = self.build_query_bitmap(&queries[bi * dim..(bi + 1) * dim]);
            q_batch[bi * qpv..(bi + 1) * qpv].copy_from_slice(&qb);
        }

        if n == 0 {
            return Vec::new();
        }

        let scores_len = batch
            .checked_mul(n)
            .expect("batched dense score buffer length (batch * n) overflows usize");
        let mut scores = vec![0u32; scores_len]; // Hamming distance first.
        sign_scan_collect_batched(&self.bitmaps, n, qpv, &q_batch, batch, &mut scores);

        let dim = u32::try_from(dim).expect("sign bitmap dim fits u32");
        scores
            .par_chunks_mut(n)
            .for_each(|row| row.iter_mut().for_each(|h| *h = dim - *h));
        scores
    }

    /// Batched dense scoring. Returns one full-corpus sign-agreement row per
    /// query, with columns aligned by document id and no sorting.
    #[must_use = "this scans the corpus to score every document per query; dropping the result discards that work"]
    pub fn score_all_batched(&self, queries: &[f32]) -> Vec<Vec<u32>> {
        let dim = self.dim;
        let batch = queries.len() / dim;
        assert_eq!(queries.len(), batch * dim);
        let n = self.n_vectors;
        let flat = self.score_all_batched_flat(queries);
        if n == 0 {
            return vec![Vec::new(); batch];
        }
        flat.chunks(n).map(|row| row.to_vec()).collect()
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
        self.qwords_per_vec * 8
    }
    pub fn byte_size(&self) -> usize {
        self.bitmaps.len() * std::mem::size_of::<u64>()
    }

    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(idx < self.n_vectors, "index out of bounds");
        let last = self.n_vectors - 1;
        let qpv = self.qwords_per_vec;
        if idx != last {
            let src = last * qpv;
            let dst = idx * qpv;
            self.bitmaps.copy_within(src..src + qpv, dst);
        }
        self.bitmaps.truncate(last * qpv);
        self.n_vectors -= 1;
        last
    }

    /// Persist to a `.ovsb` file. Format: 13-byte header + LE u64 bitmaps.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_sign_bitmap(path, self.dim, self.n_vectors, &self.bitmaps)
    }

    /// Persist to any byte writer using the `.ovsb` format.
    pub fn write_to<W: std::io::Write>(&self, writer: W) -> std::io::Result<()> {
        crate::rank_io::write_sign_bitmap_to(writer, self.dim, self.n_vectors, &self.bitmaps)
    }

    /// Load from a `.ovsb` file produced by [`Self::write`].
    ///
    /// Legacy `.tvsb` files (magic `TVSB`) written by older versions of this
    /// crate are also accepted; newly written files use the `OVSB` magic.
    ///
    /// Returns `io::Error::InvalidData` on any constructor-invariant
    /// violation. `load_sign_bitmap` already validates dim and n_vectors;
    /// this method only verifies the payload length matches the
    /// expected `n_vectors * dim / 64` u64 lanes.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (dim, n_vectors, bitmaps) = crate::rank_io::load_sign_bitmap(path)?;
        Self::from_persisted_parts(dim, n_vectors, bitmaps)
    }

    /// Load a `.ovsb`/legacy `.tvsb` index from any reader that can seek.
    ///
    /// The reader is parsed from its current position through EOF; any trailing
    /// bytes after the declared payload are rejected.
    pub fn read_from<R: std::io::Read + std::io::Seek>(reader: R) -> std::io::Result<Self> {
        let (dim, n_vectors, bitmaps) = crate::rank_io::load_sign_bitmap_from(reader)?;
        Self::from_persisted_parts(dim, n_vectors, bitmaps)
    }

    /// Load a `.ovsb`/legacy `.tvsb` index from an in-memory byte slice.
    pub fn load_from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::read_from(std::io::Cursor::new(bytes))
    }

    fn from_persisted_parts(
        dim: usize,
        n_vectors: usize,
        bitmaps: Vec<u64>,
    ) -> std::io::Result<Self> {
        let qpv = dim / 64;
        // `checked_mul` (not `saturating`): on a 32-bit target `n_vectors * qpv`
        // can overflow `usize`; treat overflow as malformed rather than letting
        // a saturated `usize::MAX` pass as a plausible length.
        let expected = n_vectors.checked_mul(qpv).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "OVSB n_vectors * dim/64 overflows usize",
            )
        })?;
        if bitmaps.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "OVSB payload length {} does not match expected {expected} u64 lanes",
                    bitmaps.len(),
                ),
            ));
        }
        Ok(Self {
            dim,
            qwords_per_vec: qpv,
            n_vectors,
            bitmaps,
        })
    }
}

// -------------------------------------------------------------------
// Scan kernels: XOR-popcount, write Hamming distance per doc.
//
// Identical shape to `bitmap_scan_collect{,_batched}` in index/bitmap.rs,
// but with `_mm512_xor_si512` in place of `_mm512_and_si512`. The
// kernel structure (lane preload, hot+tail CHUNK=8 in the batched
// variant, const-bounded inner loop for accumulator register
// promotion) is preserved exactly so the batched bandwidth-
// amortisation property carries over.
// -------------------------------------------------------------------

fn sign_scan_collect(bitmaps: &[u64], n: usize, qpv: usize, q: &[u64], scores: &mut [u32]) {
    debug_assert_eq!(scores.len(), n);
    debug_assert_eq!(q.len(), qpv);

    let use_avx512vpop = crate::avx512vpop_supported();

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sign_scan_collect_avx512vpop(bitmaps, n, qpv, q, scores);
            return;
        }
    }
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for di in 0..n {
        let doc = &bitmaps[di * qpv..(di + 1) * qpv];
        scores[di] = crate::util::xor_popcount(doc, q);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn sign_scan_collect_avx512vpop(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q: &[u64],
    scores: &mut [u32],
) {
    use std::arch::x86_64::*;
    // SAFETY: mirrors `bitmap_scan_collect_avx512vpop`. The caller
    // (`sign_scan_collect`) guarantees `q.len() == qpv` and
    // `bitmaps.len() == n * qpv`. Full 8-word groups use `loadu`; the trailing
    // `rem = qpv % 8` words use `maskz_loadu`, which only accesses the `rem`
    // valid low lanes (fault-suppressed), so loads never over-read the qpv
    // slice or the buffer end. AVX-512 F/VPOPCNTDQ confirmed by
    // `#[target_feature]` + runtime detection. The explicit block is required
    // by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        debug_assert!(qpv > 0);
        let lanes = qpv / 8;
        let rem = qpv % 8;
        let tail_mask: __mmask8 = if rem != 0 { (1u8 << rem) - 1 } else { 0 };
        let mut q_zmms: Vec<__m512i> = Vec::with_capacity(lanes);
        #[allow(clippy::needless_range_loop)]
        // indexed access is clearer / matches the kernel layout
        for l in 0..lanes {
            q_zmms.push(_mm512_loadu_si512(q.as_ptr().add(l * 8) as *const __m512i));
        }
        // Trailing `rem` query words, masked (high lanes read as 0).
        let q_tail = if rem != 0 {
            _mm512_maskz_loadu_epi64(tail_mask, q.as_ptr().add(lanes * 8) as *const i64)
        } else {
            _mm512_setzero_si512()
        };
        #[allow(clippy::needless_range_loop)]
        // indexed access is clearer / matches the kernel layout
        for di in 0..n {
            let doc_base = bitmaps.as_ptr().add(di * qpv);
            let doc_ptr = doc_base as *const __m512i;
            let mut acc_zmm = _mm512_setzero_si512();
            for l in 0..lanes {
                let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                let xor_zmm = _mm512_xor_si512(d_zmm, q_zmms[l]);
                let pop_zmm = _mm512_popcnt_epi64(xor_zmm);
                acc_zmm = _mm512_add_epi64(acc_zmm, pop_zmm);
            }
            if rem != 0 {
                // Masked tail: masked-off lanes are not loaded and XOR/popcnt to
                // 0, so they leave the Hamming sum unchanged.
                let d_tail =
                    _mm512_maskz_loadu_epi64(tail_mask, doc_base.add(lanes * 8) as *const i64);
                let xor_zmm = _mm512_xor_si512(d_tail, q_tail);
                acc_zmm = _mm512_add_epi64(acc_zmm, _mm512_popcnt_epi64(xor_zmm));
            }
            let acc_sum: i64 = _mm512_reduce_add_epi64(acc_zmm);
            scores[di] = acc_sum as u32;
        }
    }
}

// -------------------------------------------------------------------
// Batched variant — CHUNK=8 hot + tail, same shape as
// `bitmap_scan_collect_batched_avx512vpop` in index/bitmap.rs.
// -------------------------------------------------------------------

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const BATCHED_AVX512_CHUNK: usize = 8;

fn sign_scan_collect_batched(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q_batch: &[u64],
    batch: usize,
    scores: &mut [u32],
) {
    let use_avx512vpop = crate::avx512vpop_supported();

    if use_avx512vpop {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sign_scan_collect_batched_avx512vpop(bitmaps, n, qpv, q_batch, batch, scores);
            return;
        }
    }
    // Portable fallback (NEON on aarch64, scalar elsewhere).
    for di in 0..n {
        let doc = &bitmaps[di * qpv..(di + 1) * qpv];
        for bi in 0..batch {
            let q = &q_batch[bi * qpv..(bi + 1) * qpv];
            scores[bi * n + di] = crate::util::xor_popcount(doc, q);
        }
    }
}

/// Fold eight u64-lane accumulators into one vector holding their eight
/// horizontal sums, in accumulator order: an unpack/permute/shuffle tree
/// (25 vector ops) replacing eight serial `_mm512_reduce_add_epi64`
/// expansions on the per-doc hot path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn hsum8_epi64_avx512(accs: &[std::arch::x86_64::__m512i; 8]) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    {
        // L1: pairwise lane sums, interleaved per source:
        // s01 = [a0p01, a1p01, a0p23, a1p23, a0p45, a1p45, a0p67, a1p67]
        let s01 = _mm512_add_epi64(
            _mm512_unpacklo_epi64(accs[0], accs[1]),
            _mm512_unpackhi_epi64(accs[0], accs[1]),
        );
        let s23 = _mm512_add_epi64(
            _mm512_unpacklo_epi64(accs[2], accs[3]),
            _mm512_unpackhi_epi64(accs[2], accs[3]),
        );
        let s45 = _mm512_add_epi64(
            _mm512_unpacklo_epi64(accs[4], accs[5]),
            _mm512_unpackhi_epi64(accs[4], accs[5]),
        );
        let s67 = _mm512_add_epi64(
            _mm512_unpacklo_epi64(accs[6], accs[7]),
            _mm512_unpackhi_epi64(accs[6], accs[7]),
        );
        // L2: gather even/odd u64s across pair vectors:
        // e01_23 = [a0p01, a0p23, a0p45, a0p67, a2p01, a2p23, a2p45, a2p67]
        let even_idx = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let odd_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        let e02 = _mm512_permutex2var_epi64(s01, even_idx, s23);
        let o13 = _mm512_permutex2var_epi64(s01, odd_idx, s23);
        let e46 = _mm512_permutex2var_epi64(s45, even_idx, s67);
        let o57 = _mm512_permutex2var_epi64(s45, odd_idx, s67);
        // L3: pairwise again ->
        // w1 = [a0p0123, a1p0123, a0p4567, a1p4567, a2p0123, a3p0123, a2p4567, a3p4567]
        let w1 = _mm512_add_epi64(
            _mm512_unpacklo_epi64(e02, o13),
            _mm512_unpackhi_epi64(e02, o13),
        );
        let w2 = _mm512_add_epi64(
            _mm512_unpacklo_epi64(e46, o57),
            _mm512_unpackhi_epi64(e46, o57),
        );
        // L4: fold 128-bit blocks: w1 blocks B0=[a0p0123,a1p0123]
        // B1=[a0p4567,a1p4567] B2=[a2..],B3 -> sums = B0+B1, B2+B3.
        let t = _mm512_shuffle_i64x2(w1, w2, 0b10_00_10_00);
        let u = _mm512_shuffle_i64x2(w1, w2, 0b11_01_11_01);
        _mm512_add_epi64(t, u)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn sign_scan_collect_batched_avx512vpop(
    bitmaps: &[u64],
    n: usize,
    qpv: usize,
    q_batch: &[u64],
    batch: usize,
    scores: &mut [u32],
) {
    use std::arch::x86_64::*;
    // SAFETY: mirrors `bitmap_scan_collect_batched_avx512vpop`. The caller
    // (`sign_scan_collect_batched`) guarantees `q_batch.len() == batch * qpv`,
    // `bitmaps.len() == n * qpv`, and `scores.len() == batch * n`. Full 8-word
    // groups use `loadu`; the trailing `rem = qpv % 8` words use `maskz_loadu`,
    // which only accesses the `rem` valid low lanes (fault-suppressed), so loads
    // never over-read a per-vector slice or the buffer end. AVX-512 F/VPOPCNTDQ
    // confirmed by `#[target_feature]` + runtime detection. The explicit block
    // is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        debug_assert!(qpv > 0);
        debug_assert_eq!(q_batch.len(), batch * qpv);
        debug_assert_eq!(scores.len(), batch * n);
        let lanes = qpv / 8;
        let rem = qpv % 8;
        let tail_mask: __mmask8 = if rem != 0 { (1u8 << rem) - 1 } else { 0 };
        const CHUNK: usize = BATCHED_AVX512_CHUNK;

        let mut q_zmms: Vec<__m512i> = Vec::with_capacity(batch * lanes);
        for bi in 0..batch {
            for l in 0..lanes {
                q_zmms.push(_mm512_loadu_si512(
                    q_batch.as_ptr().add(bi * qpv + l * 8) as *const __m512i
                ));
            }
        }
        // Per-query masked tail (trailing `rem` words); empty when qpv % 8 == 0.
        let mut q_tails: Vec<__m512i> = Vec::with_capacity(if rem != 0 { batch } else { 0 });
        if rem != 0 {
            for bi in 0..batch {
                q_tails.push(_mm512_maskz_loadu_epi64(
                    tail_mask,
                    q_batch.as_ptr().add(bi * qpv + lanes * 8) as *const i64,
                ));
            }
        }

        // Hot path: CHUNK-sized groups; const-bounded inner bi loop so
        // LLVM unrolls and promotes the accs array to ZMM registers.
        let mut chunk_start = 0usize;
        while chunk_start + CHUNK <= batch {
            for di in 0..n {
                let mut accs: [__m512i; CHUNK] = [_mm512_setzero_si512(); CHUNK];
                let doc_base = bitmaps.as_ptr().add(di * qpv);
                let doc_ptr = doc_base as *const __m512i;
                for l in 0..lanes {
                    let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                    for bi in 0..CHUNK {
                        let q_zmm = q_zmms[(chunk_start + bi) * lanes + l];
                        let xor_zmm = _mm512_xor_si512(d_zmm, q_zmm);
                        let pop_zmm = _mm512_popcnt_epi64(xor_zmm);
                        accs[bi] = _mm512_add_epi64(accs[bi], pop_zmm);
                    }
                }
                if rem != 0 {
                    let d_tail =
                        _mm512_maskz_loadu_epi64(tail_mask, doc_base.add(lanes * 8) as *const i64);
                    for bi in 0..CHUNK {
                        let xor_zmm = _mm512_xor_si512(d_tail, q_tails[chunk_start + bi]);
                        accs[bi] = _mm512_add_epi64(accs[bi], _mm512_popcnt_epi64(xor_zmm));
                    }
                }
                let sums = hsum8_epi64_avx512(&accs);
                let mut sums_arr = [0u64; CHUNK];
                _mm512_storeu_si512(sums_arr.as_mut_ptr() as *mut __m512i, sums);
                for bi in 0..CHUNK {
                    scores[(chunk_start + bi) * n + di] = sums_arr[bi] as u32;
                }
            }
            chunk_start += CHUNK;
        }
        // Tail over the query batch.
        let tail = batch - chunk_start;
        if tail > 0 {
            for di in 0..n {
                let mut accs: [__m512i; CHUNK] = [_mm512_setzero_si512(); CHUNK];
                let doc_base = bitmaps.as_ptr().add(di * qpv);
                let doc_ptr = doc_base as *const __m512i;
                for l in 0..lanes {
                    let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                    for bi in 0..tail {
                        let q_zmm = q_zmms[(chunk_start + bi) * lanes + l];
                        let xor_zmm = _mm512_xor_si512(d_zmm, q_zmm);
                        let pop_zmm = _mm512_popcnt_epi64(xor_zmm);
                        accs[bi] = _mm512_add_epi64(accs[bi], pop_zmm);
                    }
                }
                if rem != 0 {
                    let d_tail =
                        _mm512_maskz_loadu_epi64(tail_mask, doc_base.add(lanes * 8) as *const i64);
                    for bi in 0..tail {
                        let xor_zmm = _mm512_xor_si512(d_tail, q_tails[chunk_start + bi]);
                        accs[bi] = _mm512_add_epi64(accs[bi], _mm512_popcnt_epi64(xor_zmm));
                    }
                }
                for bi in 0..tail {
                    let acc_sum: i64 = _mm512_reduce_add_epi64(accs[bi]);
                    scores[(chunk_start + bi) * n + di] = acc_sum as u32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const D: usize = 256;

    fn make_corpus(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n * D).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    fn scalar_hamming(q: &[u64], d: &[u64]) -> u32 {
        q.iter()
            .zip(d.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Returns `true` if the host supports AVX-512 VPOPCNTDQ and the test should
    /// proceed. When AVX-512 is absent:
    ///
    /// - If `ORDVEC_REQUIRE_AVX512` is set to `"1"` or `"true"` (used by the
    ///   Intel SDE CI job), this panics so the job fails loudly instead of
    ///   silently treating a skipped test as green coverage.
    /// - Otherwise it emits a skip notice to stderr and returns `false`; the
    ///   caller should return immediately.
    fn require_avx512_or_skip(test_name: &str) -> bool {
        if crate::avx512vpop_supported() {
            return true;
        }
        let required = std::env::var("ORDVEC_REQUIRE_AVX512")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        if required {
            panic!(
                "SKIP {test_name}: host lacks AVX-512 VPOPCNTDQ but \
                 ORDVEC_REQUIRE_AVX512 is set — AVX-512 kernels are not enforced"
            );
        }
        eprintln!(
            "SKIP {test_name}: host lacks AVX-512 VPOPCNTDQ; \
             set ORDVEC_REQUIRE_AVX512=1 to enforce"
        );
        false
    }

    #[test]
    fn candidate_batch_helpers() {
        use super::CandidateBatch;
        let cb = CandidateBatch {
            candidates: vec![5, 6, 7, 2],
            offsets: vec![0, 2, 2, 4], // q0=[5,6], q1=[], q2=[7,2]
        };
        assert_eq!(cb.query_count(), 3);
        assert!(!cb.is_empty());
        assert!(!cb.has_no_candidates());
        assert_eq!(cb.candidates_for_query(0), Some(&[5u32, 6][..]));
        assert_eq!(cb.candidates_for_query(1), Some(&[][..]));
        assert_eq!(cb.candidates_for_query(2), Some(&[7u32, 2][..]));
        assert_eq!(cb.candidates_for_query(3), None);

        let empty = CandidateBatch {
            candidates: vec![],
            offsets: vec![0],
        };
        assert_eq!(empty.query_count(), 0);
        assert!(empty.is_empty());
        assert!(empty.has_no_candidates());

        // 2 queries, zero candidates each → NOT empty, but has_no_candidates.
        let no_cands = CandidateBatch {
            candidates: vec![],
            offsets: vec![0, 0, 0],
        };
        assert_eq!(no_cands.query_count(), 2);
        assert!(!no_cands.is_empty());
        assert!(no_cands.has_no_candidates());
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn new_rejects_dim_zero() {
        // Regression for the Codex stop-time finding: dim=0 used to
        // pass the `dim % 64 == 0` check, then `add()` would divide
        // by zero on `vectors.len() / self.dim`. The explicit
        // `assert!(dim > 0)` in `new` rejects the bad input upfront
        // with a clear message.
        let _ = SignBitmap::new(0);
    }

    #[test]
    fn sign_encoding_threshold_at_zero() {
        let mut idx = SignBitmap::new(D);
        // First doc: alternating signs (j even → positive, j odd → negative)
        let mut v: Vec<f32> = (0..D)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        // Force one zero — sign(0) is treated as negative (bit unset).
        v[0] = 0.0;
        idx.add(&v);
        let bm = idx.build_query_bitmap(&v);
        // Bit 0 must be UNSET (we used 0.0 which is "not > 0").
        assert_eq!(bm[0] & 1, 0, "zero must be encoded as bit-unset");
        // Bit 2 must be SET (we used 1.0).
        assert_eq!((bm[0] >> 2) & 1, 1, "positive must be encoded as bit-set");
        // Bit 1 must be UNSET (we used -1.0).
        assert_eq!((bm[0] >> 1) & 1, 0, "negative must be encoded as bit-unset");
    }

    #[test]
    fn top_m_returns_ascending_hamming() {
        let n = 100;
        let corpus = make_corpus(7, n);
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
        let candidates = idx.top_m_candidates(&query, 10);
        assert_eq!(candidates.len(), 10);
        // Recompute Hamming distance for each returned candidate and
        // verify they're in ascending order.
        let qbm = idx.build_query_bitmap(&query);
        let mut last_h: u32 = 0;
        for &di in &candidates {
            let off = (di as usize) * idx.qwords_per_vec;
            let dbm = &idx.bitmaps[off..off + idx.qwords_per_vec];
            let h = scalar_hamming(&qbm, dbm);
            assert!(
                h >= last_h,
                "top_m_candidates must be sorted ascending by Hamming",
            );
            last_h = h;
        }
    }

    #[test]
    fn batched_matches_single_query() {
        let n = 200;
        let corpus = make_corpus(13, n);
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let batch: usize = 5;
        let queries: Vec<f32> = (0..batch * D)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        for m in [10usize, 30, 100] {
            let single: Vec<Vec<u32>> = (0..batch)
                .map(|bi| idx.top_m_candidates(&queries[bi * D..(bi + 1) * D], m))
                .collect();
            let batched = idx.top_m_candidates_batched(&queries, m);
            assert_eq!(single.len(), batched.len());
            for bi in 0..batch {
                assert_eq!(
                    single[bi], batched[bi],
                    "batched diverged from single-query at batch idx {bi}, M={m}",
                );
            }
        }
    }

    #[test]
    fn score_all_returns_sign_agreement_by_doc_id() {
        let n = 37;
        let corpus = make_corpus(27, n);
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut rng = ChaCha8Rng::seed_from_u64(28);
        let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();

        let scores = idx.score_all(&query);
        assert_eq!(scores.len(), n);
        let qbm = idx.build_query_bitmap(&query);
        for (di, &score) in scores.iter().enumerate() {
            let off = di * idx.qwords_per_vec;
            let dbm = &idx.bitmaps[off..off + idx.qwords_per_vec];
            assert_eq!(
                score,
                D as u32 - scalar_hamming(&qbm, dbm),
                "score_all must return sign agreement for doc {di}",
            );
        }
    }

    #[test]
    fn score_all_batched_matches_single_query() {
        let n = 75;
        let corpus = make_corpus(29, n);
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut rng = ChaCha8Rng::seed_from_u64(30);
        let batch = 6;
        let queries: Vec<f32> = (0..batch * D)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let batched = idx.score_all_batched(&queries);
        assert_eq!(batched.len(), batch);
        for bi in 0..batch {
            assert_eq!(
                batched[bi],
                idx.score_all(&queries[bi * D..(bi + 1) * D]),
                "batched dense scoring diverged at batch idx {bi}",
            );
        }
    }

    #[test]
    fn score_all_batched_flat_matches_single_query() {
        let n = 75;
        let corpus = make_corpus(31, n);
        let mut idx = SignBitmap::new(D);
        idx.add(&corpus);
        let mut rng = ChaCha8Rng::seed_from_u64(32);
        let batch = 6;
        let queries: Vec<f32> = (0..batch * D)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let batched = idx.score_all_batched_flat(&queries);
        assert_eq!(batched.len(), batch * n);
        for bi in 0..batch {
            assert_eq!(
                &batched[bi * n..(bi + 1) * n],
                idx.score_all(&queries[bi * D..(bi + 1) * D]),
                "flat batched dense scoring diverged at batch idx {bi}",
            );
        }
    }

    #[test]
    fn score_all_empty_shapes() {
        let idx = SignBitmap::new(D);
        let query = vec![1.0f32; D];
        assert!(idx.score_all(&query).is_empty());

        let queries = vec![1.0f32; 2 * D];
        assert!(idx.score_all_batched_flat(&queries).is_empty());
        assert_eq!(idx.score_all_batched(&queries), vec![Vec::<u32>::new(); 2]);

        let empty_queries: Vec<f32> = Vec::new();
        assert!(idx.score_all_batched_flat(&empty_queries).is_empty());
        assert!(idx.score_all_batched(&empty_queries).is_empty());

        let mut idx = SignBitmap::new(D);
        idx.add(&make_corpus(33, 5));
        assert!(idx.score_all_batched_flat(&empty_queries).is_empty());
        assert!(idx.score_all_batched(&empty_queries).is_empty());
    }

    #[test]
    fn large_dim_above_u16_max_roundtrips() {
        // Regression for the Codex stop-time finding: SignBitmap::new
        // accepts dim > u16::MAX (65535) as a positive multiple of 64,
        // but the first revision of `load_sign_bitmap` reused the
        // Rank-specific `check_dim` helper whose u16::MAX cap
        // rejected any such file. The dedicated `check_sign_bitmap_dim`
        // aligns the constructor and loader invariants.
        const BIG_D: usize = 65_536; // u16::MAX + 1 — the smallest dim above the old cap
        let n = 4;
        let mut rng = ChaCha8Rng::seed_from_u64(41);
        let corpus: Vec<f32> = (0..n * BIG_D)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut original = SignBitmap::new(BIG_D);
        original.add(&corpus);

        let tmp = std::env::temp_dir().join("ordvec_sign_bitmap_large_dim.tvsb");
        original
            .write(&tmp)
            .expect("write must accept dim > u16::MAX");
        let loaded = SignBitmap::load(&tmp).expect("load must accept dim > u16::MAX");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(loaded.dim(), BIG_D);
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.bitmaps, original.bitmaps);
    }

    #[test]
    fn write_then_load_roundtrips() {
        let n = 64;
        let corpus = make_corpus(17, n);
        let mut original = SignBitmap::new(D);
        original.add(&corpus);

        let tmp = std::env::temp_dir().join("ordvec_sign_bitmap_roundtrip.tvsb");
        original.write(&tmp).expect("write should succeed");
        let loaded = SignBitmap::load(&tmp).expect("load should succeed");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(loaded.dim(), original.dim());
        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.bitmaps, original.bitmaps);

        // Sanity: same query produces same top-M.
        let mut rng = ChaCha8Rng::seed_from_u64(23);
        let query: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0..1.0)).collect();
        let orig_top = original.top_m_candidates(&query, 10);
        let loaded_top = loaded.top_m_candidates(&query, 10);
        assert_eq!(orig_top, loaded_top);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let tmp = std::env::temp_dir().join("ordvec_sign_bitmap_bad_magic.tvsb");
        std::fs::write(&tmp, b"BAD!\x01\x00\x00\x01\x00\x00\x00\x00\x00").expect("write tmp");
        // SignBitmap implements a params-only Debug that intentionally avoids
        // dumping packed buffers, so keep this explicit match for the error arm;
        // use a match to inspect the Err arm instead.
        match SignBitmap::load(&tmp) {
            Ok(_) => {
                std::fs::remove_file(&tmp).ok();
                panic!("load must reject a file with the wrong magic");
            }
            Err(e) => {
                std::fs::remove_file(&tmp).ok();
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
            }
        }
    }

    #[test]
    fn avx512_path_matches_scalar_at_production_dim() {
        if !require_avx512_or_skip("avx512_path_matches_scalar_at_production_dim") {
            return;
        }
        const PROD_D: usize = 1024;
        let n = 256;
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let corpus: Vec<f32> = (0..n * PROD_D)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut idx = SignBitmap::new(PROD_D);
        idx.add(&corpus);
        let queries: Vec<f32> = (0..3 * PROD_D)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        // Batched (AVX-512 dispatched at qpv=16) must agree with scalar
        // reference computed via simple Hamming.
        let batched = idx.top_m_candidates_batched(&queries, 32);
        for bi in 0..3 {
            let qbm = idx.build_query_bitmap(&queries[bi * PROD_D..(bi + 1) * PROD_D]);
            let mut all: Vec<(u32, u32)> = (0..n as u32)
                .map(|di| {
                    let off = (di as usize) * idx.qwords_per_vec;
                    let dbm = &idx.bitmaps[off..off + idx.qwords_per_vec];
                    (scalar_hamming(&qbm, dbm), di)
                })
                .collect();
            all.sort_by_key(|&(h, did)| (h, did));
            let reference: Vec<u32> = all.iter().take(32).map(|&(_, did)| did).collect();
            assert_eq!(
                batched[bi], reference,
                "AVX-512 batched diverged from scalar at batch idx {bi}",
            );
        }
    }

    #[test]
    fn avx512_path_matches_scalar_across_residues_and_common_dims() {
        if !require_avx512_or_skip("avx512_path_matches_scalar_across_residues_and_common_dims") {
            return;
        }
        // Covers every qpv tail residue (qpv % 8 ∈ 0..=7), the lanes==0 all-tail
        // cases (qpv < 8: 64/384/448), and the common embedding dims
        // 384/512/768/1024/1536. The AVX-512 path (masked tail for non-multiples
        // of 8) must stay byte-identical to a scalar Hamming reference.
        for &dim in &[
            64usize, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1536,
        ] {
            let n = 200usize;
            let m = 32usize;
            let nq = 5usize;
            let mut rng = ChaCha8Rng::seed_from_u64(7000 + dim as u64);
            let corpus: Vec<f32> = (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let mut idx = SignBitmap::new(dim);
            idx.add(&corpus);
            let queries: Vec<f32> = (0..nq * dim).map(|_| rng.random_range(-1.0..1.0)).collect();

            let qpv = idx.qwords_per_vec;
            let dimu = dim as u32;
            let batched = idx.top_m_candidates_batched(&queries, m);
            let scores_flat = idx.score_all_batched_flat(&queries);
            for qi in 0..nq {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let qbm = idx.build_query_bitmap(q);
                let single_scores = idx.score_all(q);
                let mut ref_pairs: Vec<(u32, u32)> = Vec::with_capacity(n);
                for di in 0..n {
                    let off = di * qpv;
                    let ham = scalar_hamming(&qbm, &idx.bitmaps[off..off + qpv]);
                    let agree = dimu - ham;
                    assert_eq!(
                        single_scores[di], agree,
                        "score_all dim={dim} qi={qi} di={di}"
                    );
                    assert_eq!(
                        scores_flat[qi * n + di],
                        agree,
                        "score_all_batched_flat dim={dim} qi={qi} di={di}"
                    );
                    ref_pairs.push((ham, di as u32));
                }
                ref_pairs.sort_by_key(|&(h, did)| (h, did));
                let reference: Vec<u32> = ref_pairs.iter().take(m).map(|&(_, did)| did).collect();
                assert_eq!(
                    idx.top_m_candidates(q, m),
                    reference,
                    "single dim={dim} qi={qi}"
                );
                assert_eq!(batched[qi], reference, "batched dim={dim} qi={qi}");
            }
        }
    }
}
