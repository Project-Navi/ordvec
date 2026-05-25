//! Sign-cosine bitmap retrieval substrate.
//!
//! 1-bit-per-coord quantization at the **data-independent threshold
//! of zero**: bit j of doc d is set iff `d.embedding[j] > 0`. Storage
//! is `dim/8` bytes per doc (128 B at D=1024).
//!
//! This is the **SimHash family** primitive (Charikar 2002) applied to
//! native embedding coords rather than random projections. For
//! contrastively-trained embeddings (e.g. BGE or OpenAI ada), the
//! native coord axes already carry semantically-aligned
//! signal — making direct sign quantization competitive with, and
//! sometimes superior to, learned hash codes or rank-thresholded
//! bitmaps at the same byte budget.
//!
//! Score: `agreement(q, d) = dim - popcount(q ^ d)`. The kernel
//! computes the per-doc Hamming distance via popcount(XOR); the
//! candidate selector takes top-M docs by **lowest** Hamming
//! (= **highest** agreement).
//!
//! Kernel architecture mirrors [`crate::Bitmap`] (single-query
//! and CHUNK=8 batched hot+tail paths under AVX-512 VPOPCNTDQ). The
//! only material difference is `_mm512_xor_si512` in place of
//! `_mm512_and_si512` and an ascending tie-broken composite-key
//! selection on Hamming distance.

use rayon::prelude::*;

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

impl SignBitmap {
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
    /// first `.tvsb` revision used [`crate::rank_io::MAX_DIM`]'s
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
    pub fn add(&mut self, vectors: &[f32]) {
        crate::util::assert_all_finite(vectors);
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim);
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
        self.n_vectors += n;
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
    pub fn top_m_candidates(&self, q: &[f32], m: usize) -> Vec<u32> {
        let m_eff = m.min(self.n_vectors);
        if m_eff == 0 {
            return Vec::new();
        }
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

    /// Persist to a `.tvsb` file. Format: 13-byte header + LE u64 bitmaps.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_sign_bitmap(path, self.dim, self.n_vectors, &self.bitmaps)
    }

    /// Load from a `.tvsb` file produced by [`Self::write`].
    ///
    /// Returns `io::Error::InvalidData` on any constructor-invariant
    /// violation. `load_sign_bitmap` already validates dim and n_vectors;
    /// this method only verifies the payload length matches the
    /// expected `n_vectors * dim / 64` u64 lanes.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (dim, n_vectors, bitmaps) = crate::rank_io::load_sign_bitmap(path)?;
        let qpv = dim / 64;
        let expected = n_vectors.saturating_mul(qpv);
        if bitmaps.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TVSB payload length {} does not match expected {expected} u64 lanes",
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

    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vpopcntdq")
        && qpv.is_multiple_of(8);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

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
            let xor_zmm = _mm512_xor_si512(d_zmm, q_zmms[l]);
            let pop_zmm = _mm512_popcnt_epi64(xor_zmm);
            acc_zmm = _mm512_add_epi64(acc_zmm, pop_zmm);
        }
        let acc_sum: i64 = _mm512_reduce_add_epi64(acc_zmm);
        scores[di] = acc_sum as u32;
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
    #[cfg(target_arch = "x86_64")]
    let use_avx512vpop = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vpopcntdq")
        && qpv.is_multiple_of(8);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512vpop = false;

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
    debug_assert_eq!(qpv % 8, 0);
    debug_assert_eq!(q_batch.len(), batch * qpv);
    debug_assert_eq!(scores.len(), batch * n);
    let lanes = qpv / 8;
    const CHUNK: usize = BATCHED_AVX512_CHUNK;

    let mut q_zmms: Vec<__m512i> = Vec::with_capacity(batch * lanes);
    for bi in 0..batch {
        for l in 0..lanes {
            q_zmms.push(_mm512_loadu_si512(
                q_batch.as_ptr().add(bi * qpv + l * 8) as *const __m512i
            ));
        }
    }

    // Hot path: CHUNK-sized groups; const-bounded inner bi loop so
    // LLVM unrolls and promotes the accs array to ZMM registers.
    let mut chunk_start = 0usize;
    while chunk_start + CHUNK <= batch {
        for di in 0..n {
            let mut accs: [__m512i; CHUNK] = [_mm512_setzero_si512(); CHUNK];
            let doc_ptr = bitmaps.as_ptr().add(di * qpv) as *const __m512i;
            for l in 0..lanes {
                let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                for bi in 0..CHUNK {
                    let q_zmm = q_zmms[(chunk_start + bi) * lanes + l];
                    let xor_zmm = _mm512_xor_si512(d_zmm, q_zmm);
                    let pop_zmm = _mm512_popcnt_epi64(xor_zmm);
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
    // Tail.
    let tail = batch - chunk_start;
    if tail > 0 {
        for di in 0..n {
            let mut accs: [__m512i; CHUNK] = [_mm512_setzero_si512(); CHUNK];
            let doc_ptr = bitmaps.as_ptr().add(di * qpv) as *const __m512i;
            for l in 0..lanes {
                let d_zmm = _mm512_loadu_si512(doc_ptr.add(l));
                for bi in 0..tail {
                    let q_zmm = q_zmms[(chunk_start + bi) * lanes + l];
                    let xor_zmm = _mm512_xor_si512(d_zmm, q_zmm);
                    let pop_zmm = _mm512_popcnt_epi64(xor_zmm);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const D: usize = 256;

    fn make_corpus(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n * D).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn scalar_hamming(q: &[u64], d: &[u64]) -> u32 {
        q.iter()
            .zip(d.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
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
        let query: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
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
        let queries: Vec<f32> = (0..batch * D).map(|_| rng.gen_range(-1.0..1.0)).collect();
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
        let corpus: Vec<f32> = (0..n * BIG_D).map(|_| rng.gen_range(-1.0..1.0)).collect();
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
        let query: Vec<f32> = (0..D).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let orig_top = original.top_m_candidates(&query, 10);
        let loaded_top = loaded.top_m_candidates(&query, 10);
        assert_eq!(orig_top, loaded_top);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let tmp = std::env::temp_dir().join("ordvec_sign_bitmap_bad_magic.tvsb");
        std::fs::write(&tmp, b"BAD!\x01\x00\x00\x01\x00\x00\x00\x00\x00").expect("write tmp");
        // SignBitmap does not derive Debug (matches the convention of
        // other rank-mode types), so unwrap_err / expect_err do not apply;
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
        const PROD_D: usize = 1024;
        let n = 256;
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let corpus: Vec<f32> = (0..n * PROD_D).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut idx = SignBitmap::new(PROD_D);
        idx.add(&corpus);
        let queries: Vec<f32> = (0..3 * PROD_D).map(|_| rng.gen_range(-1.0..1.0)).collect();
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
}
