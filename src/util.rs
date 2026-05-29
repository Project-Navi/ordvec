//! Shared internals for the ordvec index family.
//!
//! - [`TopK`] is the running top-`k` collector used by every search
//!   path (full ranks, bucketed ranks, bitmap overlap).
//! - [`l2_normalise`] returns the unit-norm copy of a query vector for
//!   the asymmetric scoring path.
//! - The checked-allocation guards (`result_buffer_len`, `checked_new_len`),
//!   the finite-input assert (`assert_all_finite`), and the portable AND/XOR
//!   popcount reductions (`and_popcount` / `xor_popcount`) round out the
//!   shared helpers.
//!
//! These items are all `pub(crate)` so they are reachable from the sibling
//! index modules (`rank`, `quant`, `bitmap`, `multi_bucket`, `fastscan`)
//! but not from outside the crate.

/// Compare finite `f32` values, using the coordinate index as a deterministic
/// tiebreaker.
#[inline]
pub(crate) fn cmp_finite_f32_then_index(
    lhs_value: f32,
    lhs_index: usize,
    rhs_value: f32,
    rhs_index: usize,
) -> std::cmp::Ordering {
    if lhs_value == rhs_value {
        lhs_index.cmp(&rhs_index)
    } else {
        lhs_value.total_cmp(&rhs_value)
    }
}

/// Result-buffer length `nq * k`, panicking loudly on usize overflow
/// instead of silently wrapping to a too-small allocation.
///
/// `k` is already clamped to `n_vectors` at every call site (a single
/// query can never return more than the corpus size), so this guards
/// the *remaining* axis: a huge query count `nq`, or a modest `nq * k`
/// on a 32-bit target. Without the check the wrapped product would size
/// a too-small `Vec`, and `par_chunks_mut(k)` would then silently drop
/// the trailing queries' results. An explicit panic turns that data-
/// corruption path into a loud, debuggable abort.
#[inline]
pub(crate) fn result_buffer_len(nq: usize, k: usize) -> usize {
    nq.checked_mul(k)
        .expect("search result buffer length (nq * k) overflows usize")
}

/// Validate that an `add` would not grow an index past
/// `rank_io::MAX_VECTORS`, **and** that the resulting row-major buffer of
/// `new_n * elems_per_vec` elements still fits `usize`. Returns the new length.
///
/// The on-disk loaders cap `n_vectors` at `MAX_VECTORS` (64 Mi); the four
/// in-memory growth paths (`Rank` / `RankQuant` / `Bitmap` / `SignBitmap`
/// `add`) share this guard so the in-memory count never exceeds the loaders'
/// `n_vectors` ceiling. Candidate APIs also materialise document IDs as
/// `u32`, and `MAX_VECTORS` sits well below `u32::MAX`, so every emitted ID
/// stays representable.
///
/// The buffer-length check (`elems_per_vec` is `dim` for `Rank`, packed
/// bytes/vec for `RankQuant`, or qwords/vec for the bitmaps) matters on 32-bit
/// targets (wasm32, armv7): there `MAX_VECTORS` (2^26) times a large `dim` (up
/// to 2^16) overflows `usize`, which would wrap the `resize` length in `add`.
/// Both checks fail loud (panic) — matching `add`'s other contract asserts and
/// the crate's checked-allocation discipline (cf. [`result_buffer_len`], the
/// loaders) — rather than silently wrapping into a truncated ID space or
/// buffer (issue #25). The *count* cap is the `u32` / round-trip contract; the
/// byte payload is bounded separately by the loaders' `MAX_PAYLOAD` cap.
#[inline]
pub(crate) fn checked_new_len(current: usize, adding: usize, elems_per_vec: usize) -> usize {
    let new_n = current
        .checked_add(adding)
        .expect("ordvec: n_vectors overflows usize");
    assert!(
        new_n <= crate::rank_io::MAX_VECTORS,
        "ordvec: index would exceed MAX_VECTORS ({}); had {current}, adding {adding}",
        crate::rank_io::MAX_VECTORS,
    );
    new_n
        .checked_mul(elems_per_vec)
        .expect("ordvec: index buffer length (n_vectors * elems_per_vec) overflows usize");
    new_n
}

/// Unit-L2 copy of `v`, used by the asymmetric scoring path.
///
/// **Degenerate queries are intentional, not errors.** A query with L2 norm
/// `≤ 1e-12` (the all-zero vector, or one numerically indistinguishable from
/// it) has no direction, so its unit copy is the zero vector. The asymmetric
/// score is then `0` for every document: they all tie, and the returned top-k
/// is an arbitrary — though deterministic, via the `(score, doc_id)`
/// tie-break — prefix of the corpus. This is the correct outcome for a
/// retrieval substrate (a directionless query has no nearest neighbour), and
/// it is deliberately *silent*: the input is finite and valid, so it is not
/// rejected the way NaN/±Inf are by [`assert_all_finite`]. Callers that treat
/// an all-zero query as an upstream bug should check `‖q‖` before searching.
pub(crate) fn l2_normalise(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        vec![0.0; v.len()]
    } else {
        let inv = 1.0 / norm;
        v.iter().map(|&x| x * inv).collect()
    }
}

/// Assert that every element of `v` is finite (no `NaN`, no `±Inf`).
///
/// ordvec's public `add` / `search` entry points reject non-finite
/// inputs fail-fast: the rank transform's ordering and the
/// constant-composition invariants are only well-defined on finite
/// embeddings (a `NaN` would otherwise sort nondeterministically across
/// the rank and bitmap query paths). The Python FFI is expected to
/// validate separately; this is the Rust-side backstop.
#[inline]
pub(crate) fn assert_all_finite(v: &[f32]) {
    assert!(
        v.iter().all(|x| x.is_finite()),
        "ordvec: input contains non-finite (NaN or ±Inf) values; embeddings must be finite"
    );
}

// ---------------------------------------------------------------------
// Portable per-row popcount reductions for the bitmap / sign-bitmap scan
// fallbacks. On x86_64 these are the scalar path — the AVX-512 VPOPCNTDQ
// kernels are the fast path and call `std::arch` directly. On aarch64 they
// use NEON (VCNT over a `uint8x16_t`), giving the bitmap/sign scans SIMD
// acceleration on Graviton / Apple silicon / Axion, which previously fell
// through to scalar `u64::count_ones()`. The result is an exact integer, so
// every path (scalar, NEON, AVX-512) returns a bit-identical count —
// popcount has no summation-order sensitivity, so there is no cross-CPU
// score drift to reconcile (unlike the float kernels).
// ---------------------------------------------------------------------

/// Sum of `popcount(doc[w] & q[w])` over two equal-length `u64` rows —
/// bitmap top-bucket overlap.
#[inline]
pub(crate) fn and_popcount(doc: &[u64], q: &[u64]) -> u32 {
    // Hard assert (not debug_assert): these are pub(crate) "safe" fns whose
    // SIMD paths read `q` at offsets up to `doc.len()`, so a length mismatch
    // would be a release-mode OOB read (the scalar path would silently
    // truncate instead — the paths must not diverge). All callers pass equal
    // `qpv` rows; this turns any future misuse into a clean panic, matching the
    // crate's hard-assert-before-SIMD pattern (see `body_overlap_scores_subset`).
    assert_eq!(
        doc.len(),
        q.len(),
        "popcount: doc and query bitmap rows must be equal length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline ABI, so these
        // intrinsics are unconditionally available — no runtime detection.
        unsafe { and_popcount_neon(doc, q) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        and_popcount_simd128(doc, q)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        and_popcount_scalar(doc, q)
    }
}

/// Sum of `popcount(doc[w] ^ q[w])` over two equal-length `u64` rows —
/// sign-bitmap Hamming distance.
#[inline]
pub(crate) fn xor_popcount(doc: &[u64], q: &[u64]) -> u32 {
    // Hard assert (not debug_assert): these are pub(crate) "safe" fns whose
    // SIMD paths read `q` at offsets up to `doc.len()`, so a length mismatch
    // would be a release-mode OOB read (the scalar path would silently
    // truncate instead — the paths must not diverge). All callers pass equal
    // `qpv` rows; this turns any future misuse into a clean panic, matching the
    // crate's hard-assert-before-SIMD pattern (see `body_overlap_scores_subset`).
    assert_eq!(
        doc.len(),
        q.len(),
        "popcount: doc and query bitmap rows must be equal length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: as above — NEON is baseline on aarch64.
        unsafe { xor_popcount_neon(doc, q) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        xor_popcount_simd128(doc, q)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        xor_popcount_scalar(doc, q)
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline]
fn and_popcount_scalar(doc: &[u64], q: &[u64]) -> u32 {
    doc.iter().zip(q).map(|(d, qq)| (d & qq).count_ones()).sum()
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline]
fn xor_popcount_scalar(doc: &[u64], q: &[u64]) -> u32 {
    doc.iter().zip(q).map(|(d, qq)| (d ^ qq).count_ones()).sum()
}

/// NEON AND-popcount: 16 bytes (2×`u64`) per `vcntq_u8`, horizontally
/// summed per 16-byte block (≤ 16×8 = 128, within the `u8` reduce) and
/// accumulated into a `u32`. A trailing odd `u64` (e.g. `dim = 192` →
/// `qpv = 3`) is handled scalar.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn and_popcount_neon(doc: &[u64], q: &[u64]) -> u32 {
    use std::arch::aarch64::*;
    // SAFETY: NEON is part of the aarch64 baseline ABI — these intrinsics
    // are unconditionally available on aarch64. The `vld1q_u8` loads read
    // 16 bytes starting at `dptr/qptr + w*8`; `w + 2 <= qpv` guarantees
    // both offsets are within the slice (each u64 is 8 bytes, so 2×u64 = 16
    // bytes). The trailing scalar path reads `doc[w]`/`q[w]` with a safe
    // slice index. The explicit block is required by
    // `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        let qpv = doc.len();
        let dptr = doc.as_ptr() as *const u8;
        let qptr = q.as_ptr() as *const u8;
        let mut acc = 0u32;
        let mut w = 0usize;
        while w + 2 <= qpv {
            let dv = vld1q_u8(dptr.add(w * 8));
            let qv = vld1q_u8(qptr.add(w * 8));
            acc += vaddvq_u8(vcntq_u8(vandq_u8(dv, qv))) as u32;
            w += 2;
        }
        if w < qpv {
            acc += (doc[w] & q[w]).count_ones();
        }
        acc
    }
}

/// NEON XOR-popcount (sign-bitmap Hamming); see [`and_popcount_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn xor_popcount_neon(doc: &[u64], q: &[u64]) -> u32 {
    use std::arch::aarch64::*;
    // SAFETY: same contract as `and_popcount_neon` — NEON baseline ABI,
    // `vld1q_u8` loads bounded by `w + 2 <= qpv`, trailing word via safe
    // index. The explicit block is required by
    // `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        let qpv = doc.len();
        let dptr = doc.as_ptr() as *const u8;
        let qptr = q.as_ptr() as *const u8;
        let mut acc = 0u32;
        let mut w = 0usize;
        while w + 2 <= qpv {
            let dv = vld1q_u8(dptr.add(w * 8));
            let qv = vld1q_u8(qptr.add(w * 8));
            acc += vaddvq_u8(vcntq_u8(veorq_u8(dv, qv))) as u32;
            w += 2;
        }
        if w < qpv {
            acc += (doc[w] ^ q[w]).count_ones();
        }
        acc
    }
}

/// WASM `simd128` AND-popcount: 16 bytes (2×`u64`) per `u8x16_popcnt`,
/// pairwise-reduced (≤ 16×8 = 128) to a `u32` per block, accumulated
/// across blocks; a trailing odd `u64` is handled scalar. Compile-time
/// gated — `simd128` has no runtime detection on wasm, so this path is
/// active only when built with `-C target-feature=+simd128`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn and_popcount_simd128(doc: &[u64], q: &[u64]) -> u32 {
    use std::arch::wasm32::*;
    let qpv = doc.len();
    let dptr = doc.as_ptr() as *const u8;
    let qptr = q.as_ptr() as *const u8;
    let mut acc = 0u32;
    let mut w = 0usize;
    while w + 2 <= qpv {
        // SAFETY: w + 2 <= qpv, so the 16-byte load is in-bounds for both
        // rows; `v128_load` is unaligned-safe.
        let dv = unsafe { v128_load(dptr.add(w * 8) as *const v128) };
        let qv = unsafe { v128_load(qptr.add(w * 8) as *const v128) };
        let pc = u8x16_popcnt(v128_and(dv, qv));
        let s16 = u16x8_extadd_pairwise_u8x16(pc);
        let s32 = u32x4_extadd_pairwise_u16x8(s16);
        acc += u32x4_extract_lane::<0>(s32)
            + u32x4_extract_lane::<1>(s32)
            + u32x4_extract_lane::<2>(s32)
            + u32x4_extract_lane::<3>(s32);
        w += 2;
    }
    if w < qpv {
        acc += (doc[w] & q[w]).count_ones();
    }
    acc
}

/// WASM `simd128` XOR-popcount (sign-bitmap Hamming); see
/// [`and_popcount_simd128`].
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn xor_popcount_simd128(doc: &[u64], q: &[u64]) -> u32 {
    use std::arch::wasm32::*;
    let qpv = doc.len();
    let dptr = doc.as_ptr() as *const u8;
    let qptr = q.as_ptr() as *const u8;
    let mut acc = 0u32;
    let mut w = 0usize;
    while w + 2 <= qpv {
        // SAFETY: see `and_popcount_simd128`.
        let dv = unsafe { v128_load(dptr.add(w * 8) as *const v128) };
        let qv = unsafe { v128_load(qptr.add(w * 8) as *const v128) };
        let pc = u8x16_popcnt(v128_xor(dv, qv));
        let s16 = u16x8_extadd_pairwise_u8x16(pc);
        let s32 = u32x4_extadd_pairwise_u16x8(s16);
        acc += u32x4_extract_lane::<0>(s32)
            + u32x4_extract_lane::<1>(s32)
            + u32x4_extract_lane::<2>(s32)
            + u32x4_extract_lane::<3>(s32);
        w += 2;
    }
    if w < qpv {
        acc += (doc[w] ^ q[w]).count_ones();
    }
    acc
}

/// Running top-`k` collector.
///
/// Maintains an unsorted array of the best `k` (score, index) pairs
/// seen so far and the slot of the current *worst* kept entry.
/// `maybe_insert` is O(k) worst-case (k-element scan after each
/// replacement) and the common path — entry worse than the current
/// worst kept — is O(1). No allocation per document, no full-N
/// partial sort.
///
/// **Tie-break (deterministic across CPUs).** Ranking is by the
/// composite key `(score desc, doc_id asc)`: on equal scores the
/// LOWER doc_id wins, both for eviction and in the final order. SIMD
/// vs scalar f32 summation-order differences can flip genuine
/// near-ties between hosts; the composite key removes that
/// nondeterminism and matches the candidate-gen paths
/// (`top_m_candidates`) which already partition on `(score, doc_id)`.
/// The "worst kept" entry — the one evicted first — is therefore the
/// one with the lowest score and, among equal-score entries, the
/// HIGHEST doc_id.
pub(crate) struct TopK {
    k: usize,
    scores: Vec<f32>,
    indices: Vec<i64>,
    filled: usize,
    /// Slot holding the worst kept entry under `(score asc, doc_id
    /// desc)` — the next to be evicted.
    worst_pos: usize,
    /// Score of the worst kept entry.
    worst_val: f32,
    /// doc_id of the worst kept entry (used to break score ties:
    /// among equal scores the higher doc_id is worse to keep).
    worst_idx: i64,
}

impl TopK {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            scores: vec![f32::NEG_INFINITY; k],
            indices: vec![-1; k],
            filled: 0,
            worst_pos: 0,
            worst_val: f32::INFINITY,
            worst_idx: i64::MAX,
        }
    }

    #[inline]
    pub(crate) fn maybe_insert(&mut self, score: f32, idx: usize) {
        // Convert the doc_id to its i64 storage form once, up front. doc_ids
        // are `< n_vectors ≤ MAX_VECTORS` (2^26) by the `add` cap, so this
        // never fails in practice; the checked conversion makes the "a doc_id
        // must fit i64" contract explicit rather than letting a pathological
        // `idx` near `usize::MAX` wrap to `-1` and collide with the empty-slot
        // sentinel (`indices` is pre-filled with `-1`). Hard, not debug: a
        // silent collision would corrupt results in release. `try_from` also
        // stays clippy-clean on 32-bit, where `idx <= i64::MAX as usize` would
        // be an always-true `absurd_extreme_comparison`.
        let id = i64::try_from(idx).expect("ordvec: doc_id exceeds i64::MAX");
        if self.filled < self.k {
            self.scores[self.filled] = score;
            self.indices[self.filled] = id;
            self.filled += 1;
            if self.filled == self.k {
                self.recompute_worst();
            }
        } else {
            // Replace the worst kept entry iff the incoming `(score, id)` is
            // strictly better to keep under the `(score desc, doc_id asc)`
            // order: a higher score, or an equal score with a lower doc_id.
            // doc_ids are unique per scan, so this is a total order — the
            // greedy eviction keeps exactly the top-k set under the composite
            // key.
            let better = score > self.worst_val || (score == self.worst_val && id < self.worst_idx);
            if better {
                self.scores[self.worst_pos] = score;
                self.indices[self.worst_pos] = id;
                self.recompute_worst();
            }
        }
    }

    /// Locate the worst kept entry under `(score asc, doc_id desc)`:
    /// lowest score, and among equal scores the highest doc_id. That
    /// is the entry a strictly-better incoming candidate evicts.
    fn recompute_worst(&mut self) {
        let mut wv = f32::INFINITY;
        let mut wi = i64::MIN;
        let mut wp = 0;
        for i in 0..self.filled {
            let s = self.scores[i];
            let id = self.indices[i];
            if s < wv || (s == wv && id > wi) {
                wv = s;
                wi = id;
                wp = i;
            }
        }
        self.worst_val = wv;
        self.worst_idx = wi;
        self.worst_pos = wp;
    }

    /// Drain into `out_scores` / `out_indices` sorted by the composite
    /// key `(score desc, doc_id asc)`. `out_scores.len()` is the
    /// user-requested `k`; positions beyond `self.filled` are left as
    /// sentinels.
    pub(crate) fn finalize_into(&self, out_scores: &mut [f32], out_indices: &mut [i64]) {
        debug_assert_eq!(out_scores.len(), out_indices.len());
        for s in out_scores.iter_mut() {
            *s = f32::NEG_INFINITY;
        }
        for i in out_indices.iter_mut() {
            *i = -1;
        }
        let mut pairs: Vec<(f32, i64)> = self
            .scores
            .iter()
            .zip(self.indices.iter())
            .take(self.filled)
            .map(|(&s, &i)| (s, i))
            .collect();
        // Composite key: score descending, then doc_id ascending. The
        // doc_id tie-break makes the final order deterministic when
        // scores are equal.
        pairs.sort_unstable_by(|a, b| {
            // `total_cmp` is a true total order (IEEE-754 `totalOrder`), so the
            // sort stays well-defined even if a non-finite score ever slipped
            // past the finite-input guards — `partial_cmp(..).unwrap_or(Equal)`
            // is not a total order and can mis-sort around NaN. For the finite
            // scores we actually have, the two agree. doc_id ascending breaks
            // score ties (unchanged).
            b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
        });
        for (slot, (s, i)) in pairs.into_iter().enumerate() {
            if slot >= out_scores.len() {
                break;
            }
            out_scores[slot] = s;
            out_indices[slot] = i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{and_popcount, checked_new_len, xor_popcount, TopK};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn naive_and(d: &[u64], q: &[u64]) -> u32 {
        d.iter().zip(q).map(|(a, b)| (a & b).count_ones()).sum()
    }
    fn naive_xor(d: &[u64], q: &[u64]) -> u32 {
        d.iter().zip(q).map(|(a, b)| (a ^ b).count_ones()).sum()
    }

    /// The portable popcount helpers must agree with a naive reference on
    /// every target. This is the runtime correctness gate for the SIMD
    /// kernels: it exercises whichever path is active — scalar on x86_64,
    /// NEON on aarch64 (via the ARM CI runner), simd128 on wasm
    /// (`-C target-feature=+simd128`, via the wasm CI lane). The `qpv`
    /// sweep covers odd lengths (the scalar tail after the 2×u64 SIMD
    /// stride) and exact multiples of the stride.
    #[test]
    fn popcount_helpers_match_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
        for qpv in [1usize, 2, 3, 4, 7, 8, 15, 16, 17, 31] {
            for _ in 0..64 {
                let d: Vec<u64> = (0..qpv).map(|_| rng.random()).collect();
                let q: Vec<u64> = (0..qpv).map(|_| rng.random()).collect();
                assert_eq!(and_popcount(&d, &q), naive_and(&d, &q), "AND qpv={qpv}");
                assert_eq!(xor_popcount(&d, &q), naive_xor(&d, &q), "XOR qpv={qpv}");
            }
        }
    }

    #[test]
    fn topk_zero_k_is_inert() {
        // k == 0 arises when an empty index clamps `k = min(requested, n) = 0`.
        // `maybe_insert` must be a no-op and `finalize_into` must emit nothing —
        // never panic or index out of bounds on the zero-length slots.
        let mut top = TopK::new(0);
        top.maybe_insert(1.0, 0);
        top.maybe_insert(f32::NEG_INFINITY, 7);
        let mut scores: [f32; 0] = [];
        let mut indices: [i64; 0] = [];
        top.finalize_into(&mut scores, &mut indices);
        assert!(scores.is_empty() && indices.is_empty());
    }

    #[test]
    fn checked_new_len_accepts_up_to_max() {
        use crate::rank_io::MAX_VECTORS;
        // Exactly MAX_VECTORS is allowed — the loaders accept the same ceiling,
        // so a freshly grown index stays write/load round-trippable. (elems=1
        // isolates the count cap from the buffer-size check.)
        assert_eq!(checked_new_len(0, MAX_VECTORS, 1), MAX_VECTORS);
        assert_eq!(checked_new_len(MAX_VECTORS - 1, 1, 1), MAX_VECTORS);
        // An empty add never trips the guard.
        assert_eq!(checked_new_len(MAX_VECTORS, 0, 1), MAX_VECTORS);
        // MAX_VECTORS * 4096 = 2^38 fits usize on 64-bit; on 32-bit it overflows,
        // which the guard correctly panics on (see
        // `checked_new_len_rejects_buffer_overflow`). Gate the success assertion
        // to 64-bit so the suite stays portable (wasm32 / armv7).
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(checked_new_len(0, MAX_VECTORS, 4096), MAX_VECTORS);
        }
    }

    #[test]
    #[should_panic(expected = "MAX_VECTORS")]
    fn checked_new_len_rejects_one_past_max() {
        use crate::rank_io::MAX_VECTORS;
        // One past the loader ceiling must fail loud rather than build an index
        // that write/load would refuse to round-trip.
        let _ = checked_new_len(MAX_VECTORS, 1, 1);
    }

    #[test]
    #[should_panic(expected = "n_vectors overflows usize")]
    fn checked_new_len_rejects_usize_overflow() {
        // The running count itself must not wrap before the cap is checked.
        let _ = checked_new_len(usize::MAX, 1, 1);
    }

    #[test]
    #[should_panic(expected = "buffer length")]
    fn checked_new_len_rejects_buffer_overflow() {
        // Count is within MAX_VECTORS, but new_n * elems_per_vec overflows
        // usize — the 32-bit (wasm32) hazard the `resize` in `add` would hit.
        let _ = checked_new_len(0, 2, usize::MAX);
    }
}
