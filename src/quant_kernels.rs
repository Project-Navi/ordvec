//! RankQuant scoring kernels.
//!
//! Three families, one entry point per family:
//!
//! - Scalar LUT scan ([`scan_via_lut_scalar`] + `scan_b{1,2,4}_to_topk`)
//!   — portable fallback used by the symmetric path and by the
//!   asymmetric path on non-x86 / older x86 CPUs.
//! - AVX2 + FMA ([`scan_b{2,4}_asym_avx2`]) — centre-drop math, raw
//!   codes in the hot loop, per-query offset reapplied at finalize.
//! - AVX-512 + AVX-512DQ ([`scan_b{2,4}_asym_avx512`]) — 4-way unrolled
//!   with 4 independent accumulators to break the FMA dep chain.
//!
//! All kernels feed into a [`TopK`](crate::util::TopK) collector
//! supplied by the caller. The caller is responsible for runtime
//! CPU-feature detection before reaching for any `unsafe` SIMD kernel.

use crate::rank::bucket_centre;
use crate::util::TopK;

/// Build the per-coord, per-bucket LUT for this query and dispatch
/// the matching `scan_b{N}_to_topk` for the configured `bits`.
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
pub(crate) fn scan_via_lut_scalar(
    packed: &[u8],
    n: usize,
    dim: usize,
    bits: u8,
    n_buckets: usize,
    q_unit: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let mut lut = Vec::new();
    scan_via_lut_scalar_with_lut(
        packed, n, dim, bits, n_buckets, q_unit, scale, top, &mut lut,
    );
}

pub(crate) fn build_asym_lut_into(
    lut: &mut Vec<f32>,
    dim: usize,
    bits: u8,
    n_buckets: usize,
    q_unit: &[f32],
) {
    assert_eq!(q_unit.len(), dim);
    lut.resize(dim * n_buckets, 0.0);
    for (&qd, row) in q_unit.iter().zip(lut.chunks_exact_mut(n_buckets)) {
        for (b, slot) in row.iter_mut().enumerate() {
            *slot = qd * bucket_centre(b as u8, bits);
        }
    }
}

/// Same scalar LUT scan as [`scan_via_lut_scalar`], but the caller supplies the
/// LUT buffer so hot paths can reuse capacity after warmup.
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
pub(crate) fn scan_via_lut_scalar_with_lut(
    packed: &[u8],
    n: usize,
    dim: usize,
    bits: u8,
    n_buckets: usize,
    q_unit: &[f32],
    scale: f32,
    top: &mut TopK,
    lut: &mut Vec<f32>,
) {
    build_asym_lut_into(lut, dim, bits, n_buckets, q_unit);
    match bits {
        1 => scan_b1_to_topk(packed, n, dim, lut, scale, top),
        2 => scan_b2_to_topk(packed, n, dim, lut, scale, top),
        4 => scan_b4_to_topk(packed, n, dim, lut, scale, top),
        8 => scan_b8_to_topk(packed, n, dim, lut, scale, top),
        _ => unreachable!("bits validated in new()"),
    }
}

// -------------------------------------------------------------------
// LUT-based scan kernels. Shared by symmetric and asymmetric paths —
// only the LUT construction differs.
//
// LUT layout: `lut[d * n_buckets + b]` is the contribution of bucket
// `b` at coordinate `d`. The kernel sums `lut[d][doc_bucket[d]]`
// across `d` and emits one (score * scale, doc_idx) into TopK.
// -------------------------------------------------------------------

/// 1-bit scan. 8 codes per byte; n_buckets = 2.
pub(crate) fn scan_b1_to_topk(
    packed: &[u8],
    n: usize,
    dim: usize,
    lut: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let bytes_per_vec = dim / 8;
    for di in 0..n {
        let doc = &packed[di * bytes_per_vec..(di + 1) * bytes_per_vec];
        let mut acc = 0.0f32;
        for (g, &byte) in doc.iter().enumerate() {
            let base = (g * 8) * 2;
            acc += lut[base + (((byte >> 7) & 1) as usize)];
            acc += lut[base + 2 + (((byte >> 6) & 1) as usize)];
            acc += lut[base + 4 + (((byte >> 5) & 1) as usize)];
            acc += lut[base + 6 + (((byte >> 4) & 1) as usize)];
            acc += lut[base + 8 + (((byte >> 3) & 1) as usize)];
            acc += lut[base + 10 + (((byte >> 2) & 1) as usize)];
            acc += lut[base + 12 + (((byte >> 1) & 1) as usize)];
            acc += lut[base + 14 + ((byte & 1) as usize)];
        }
        top.maybe_insert(acc * scale, di);
    }
}

/// 2-bit scan. 4 codes per byte; n_buckets = 4.
pub(crate) fn scan_b2_to_topk(
    packed: &[u8],
    n: usize,
    dim: usize,
    lut: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let bytes_per_vec = dim / 4;
    for di in 0..n {
        let doc = &packed[di * bytes_per_vec..(di + 1) * bytes_per_vec];
        let mut acc = 0.0f32;
        for (g, &byte) in doc.iter().enumerate() {
            let base = (g * 4) * 4;
            acc += lut[base + (((byte >> 6) & 3) as usize)];
            acc += lut[base + 4 + (((byte >> 4) & 3) as usize)];
            acc += lut[base + 8 + (((byte >> 2) & 3) as usize)];
            acc += lut[base + 12 + ((byte & 3) as usize)];
        }
        top.maybe_insert(acc * scale, di);
    }
}

/// 4-bit scan. 2 codes per byte; n_buckets = 16.
pub(crate) fn scan_b4_to_topk(
    packed: &[u8],
    n: usize,
    dim: usize,
    lut: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let bytes_per_vec = dim / 2;
    for di in 0..n {
        let doc = &packed[di * bytes_per_vec..(di + 1) * bytes_per_vec];
        let mut acc = 0.0f32;
        for (g, &byte) in doc.iter().enumerate() {
            let base = (g * 2) * 16;
            acc += lut[base + (((byte >> 4) & 0xF) as usize)];
            acc += lut[base + 16 + ((byte & 0xF) as usize)];
        }
        top.maybe_insert(acc * scale, di);
    }
}

/// Build the `dim * 256` per-coordinate asymmetric LUT for `b=8`:
/// `lut[d * 256 + code] = q_unit[d] * bucket_centre(code, 8)`. This is the
/// shared input to both the scalar [`scan_b8_to_topk`] reference and the
/// AVX-512 [`scan_b8_asym_avx512_gather`] kernel, so they score-parity.
///
/// `bucket_centre(code, 8) = code - 127.5`, so each row is the query
/// coordinate scaled across the 256 centred bucket values.
pub(crate) fn build_b8_asym_lut_into(lut: &mut Vec<f32>, q_unit: &[f32]) {
    let dim = q_unit.len();
    lut.resize(dim * 256, 0.0);
    for (&qd, row) in q_unit.iter().zip(lut.chunks_exact_mut(256)) {
        for (code, slot) in row.iter_mut().enumerate() {
            *slot = qd * bucket_centre(code as u8, 8);
        }
    }
}

/// 8-bit scan. 1 code per byte; n_buckets = 256. The degenerate
/// one-code-per-byte case: `doc[d]` is the code at coordinate `d`, so the
/// inner loop is a single LUT lookup per byte against the `dim * 256`
/// per-coord LUT. Used by both the symmetric path (`bucket_centre` LUT)
/// and the asymmetric scalar LUT path (`q_unit[d] * bucket_centre(b)`).
///
/// This is also the **portable scalar reference** for the `b=8` asymmetric
/// gather: it sums in strict coordinate order, one lookup + add per byte,
/// so it is the bit-exact baseline the AVX-512 gather kernel is parity-
/// tested against (within the crate's 1e-4 cross-backend tolerance).
pub(crate) fn scan_b8_to_topk(
    packed: &[u8],
    n: usize,
    dim: usize,
    lut: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let bytes_per_vec = dim; // 1 byte per coordinate
    for di in 0..n {
        let doc = &packed[di * bytes_per_vec..(di + 1) * bytes_per_vec];
        let mut acc = 0.0f32;
        for (d, &code) in doc.iter().enumerate() {
            // LUT row `d` has 256 entries (one per code value); the code is
            // already the bucket index for b=8.
            acc += lut[d * 256 + code as usize];
        }
        top.maybe_insert(acc * scale, di);
    }
}

// -------------------------------------------------------------------
// AVX2 + FMA kernels for the asymmetric path.
//
// The key trick: bucket_centre(b) = b - (n_buckets - 1) / 2 is one
// SIMD subtraction, so we never touch a per-coord LUT. For each chunk
// of doc bytes we (a) broadcast the packed bytes into a YMM,
// (b) shift each lane to align its 2- or 4-bit field,
// (c) mask to the bucket index,
// (d) convert to f32 and subtract the centre offset,
// (e) FMA against the corresponding query slice.
//
// Caller must verify `is_x86_feature_detected!("avx2,fma")` once.
// -------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn scan_b2_asym_avx2(
    packed: &[u8],
    n: usize,
    dim: usize,
    q: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    use std::arch::x86_64::*;

    // SAFETY: a `pub(crate) unsafe fn` reachable only via `quant.rs`'s
    // runtime-detected dispatch, which upholds the invariants the raw
    // `packed.as_ptr().add(di * bytes_per_vec)` doc reads and `q.as_ptr().add(..)`
    // query loads below depend on:
    //   * `packed.len() == n * bytes_per_vec` (all packed codes present),
    //   * `q.len() >= dim` (per-chunk query loads stay in bounds),
    //   * `dim % K == 0` (asserted immediately below).
    // `RankQuant::{new,add}` pack exactly `bytes_per_vec` bytes/doc and
    // `load_rankquant` re-validates the shape, so this holds on every path here.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        // Hard backstop: the dispatch in `quant.rs` must only route here
        // when `dim % 16 == 0`. Kept as a real `assert!` (not debug-only)
        // so a mis-dispatch fails loudly in release instead of silently
        // dropping the trailing chunk and returning wrong top-k.
        assert_eq!(dim % 16, 0, "b=2 AVX2 path needs dim % 16 == 0");
        let bytes_per_vec = dim / 4;
        // For each chunk of 4 doc bytes we extract 16 codes (top byte first,
        // most-significant 2 bits first within a byte). Shift amounts:
        //   chunk u32 = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3,
        //   code k = (chunk >> ((15 - k) * 2)) & 3, for k in 0..16.
        //
        // Centre-drop: score(d) = Σ q[j]·(code[j] - 1.5)
        //                       = Σ q[j]·code[j] - 1.5·Σ q[j]
        // The second term is per-query constant and is added back to the
        // TopK scores at finalize time. The hot loop only does the raw
        // dot product against unsigned code values.
        let shifts_hi = _mm256_setr_epi32(30, 28, 26, 24, 22, 20, 18, 16);
        let shifts_lo = _mm256_setr_epi32(14, 12, 10, 8, 6, 4, 2, 0);
        let mask3 = _mm256_set1_epi32(3);

        let bytes_per_chunk = 4usize;
        let chunks_per_vec = bytes_per_vec / bytes_per_chunk;

        for di in 0..n {
            let doc = packed.as_ptr().add(di * bytes_per_vec);
            let mut acc_hi = _mm256_setzero_ps();
            let mut acc_lo = _mm256_setzero_ps();

            for c in 0..chunks_per_vec {
                let chunk_ptr = doc.add(c * bytes_per_chunk);
                let b0 = *chunk_ptr as u32;
                let b1 = *chunk_ptr.add(1) as u32;
                let b2 = *chunk_ptr.add(2) as u32;
                let b3 = *chunk_ptr.add(3) as u32;
                let chunk = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
                let broadcast = _mm256_set1_epi32(chunk as i32);

                let codes_hi = _mm256_and_si256(_mm256_srlv_epi32(broadcast, shifts_hi), mask3);
                let codes_lo = _mm256_and_si256(_mm256_srlv_epi32(broadcast, shifts_lo), mask3);

                let codes_f_hi = _mm256_cvtepi32_ps(codes_hi);
                let codes_f_lo = _mm256_cvtepi32_ps(codes_lo);

                let d_base = c * 16;
                let q_hi = _mm256_loadu_ps(q.as_ptr().add(d_base));
                let q_lo = _mm256_loadu_ps(q.as_ptr().add(d_base + 8));

                acc_hi = _mm256_fmadd_ps(codes_f_hi, q_hi, acc_hi);
                acc_lo = _mm256_fmadd_ps(codes_f_lo, q_lo, acc_lo);
            }

            let total = _mm256_add_ps(acc_hi, acc_lo);
            let raw = horizontal_sum_avx2(total);
            top.maybe_insert(raw * scale, di);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn scan_b4_asym_avx2(
    packed: &[u8],
    n: usize,
    dim: usize,
    q: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    use std::arch::x86_64::*;

    // SAFETY: a `pub(crate) unsafe fn` reachable only via `quant.rs`'s
    // runtime-detected dispatch, which upholds the invariants the raw
    // `packed.as_ptr().add(di * bytes_per_vec)` doc reads and `q.as_ptr().add(..)`
    // query loads below depend on:
    //   * `packed.len() == n * bytes_per_vec` (all packed codes present),
    //   * `q.len() >= dim` (per-chunk query loads stay in bounds),
    //   * `dim % K == 0` (asserted immediately below).
    // `RankQuant::{new,add}` pack exactly `bytes_per_vec` bytes/doc and
    // `load_rankquant` re-validates the shape, so this holds on every path here.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        // Hard backstop (see `scan_b2_asym_avx2`): mis-dispatch must fail
        // loudly in release, not silently drop the trailing chunk.
        assert_eq!(dim % 8, 0, "b=4 AVX2 path needs dim % 8 == 0");
        let bytes_per_vec = dim / 2;
        // For each chunk of 4 doc bytes we extract 8 codes (one nibble each).
        //   chunk u32 = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3,
        //   code k = (chunk >> ((7 - k) * 4)) & 0xF, for k in 0..8.
        // Centre-drop: -7.5·Σq[j] is added back to the TopK scores at
        // finalize time; the hot loop scores raw nibble values.
        let shifts = _mm256_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let mask_f = _mm256_set1_epi32(0xF);

        let bytes_per_chunk = 4usize;
        let chunks_per_vec = bytes_per_vec / bytes_per_chunk;

        for di in 0..n {
            let doc = packed.as_ptr().add(di * bytes_per_vec);
            let mut acc = _mm256_setzero_ps();

            for c in 0..chunks_per_vec {
                let chunk_ptr = doc.add(c * bytes_per_chunk);
                let b0 = *chunk_ptr as u32;
                let b1 = *chunk_ptr.add(1) as u32;
                let b2 = *chunk_ptr.add(2) as u32;
                let b3 = *chunk_ptr.add(3) as u32;
                let chunk = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
                let broadcast = _mm256_set1_epi32(chunk as i32);

                let codes = _mm256_and_si256(_mm256_srlv_epi32(broadcast, shifts), mask_f);
                let codes_f = _mm256_cvtepi32_ps(codes);

                let d_base = c * 8;
                let q_vec = _mm256_loadu_ps(q.as_ptr().add(d_base));

                acc = _mm256_fmadd_ps(codes_f, q_vec, acc);
            }

            let raw = horizontal_sum_avx2(acc);
            top.maybe_insert(raw * scale, di);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: called only from `scan_b{2,4}_asym_avx2` which are themselves
    // guarded by the AVX2+FMA `#[target_feature]` and the caller's runtime
    // detection — the intrinsics here are always feature-available.
    // All intrinsics in this body (extract/cast/add/cvt) are safe under the
    // `avx2,fma` `#[target_feature]` gate; no explicit `unsafe {}` block is
    // needed or permitted (`unused_unsafe` under `-D warnings`).
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// -------------------------------------------------------------------
// AVX-512 + FMA kernels for the asymmetric path.
//
// Same broadcast → variable-shift → mask → cvt → FMA shape as the
// AVX2 kernels, with the centre-drop applied at finalize time.
// Doubled lane count: B=2 fits 16 codes in one __m512 (no hi/lo
// split), B=4 packs 16 codes per chunk by blending two 4-byte
// chunks into a single __m512i.
//
// Caller must verify `is_x86_feature_detected!("avx512f")` (and dq
// for the 32-bit lane ops) once.
// -------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
pub(crate) unsafe fn scan_b2_asym_avx512(
    packed: &[u8],
    n: usize,
    dim: usize,
    q: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    use std::arch::x86_64::*;

    // SAFETY: a `pub(crate) unsafe fn` reachable only via `quant.rs`'s
    // runtime-detected dispatch, which upholds the invariants the raw
    // `packed.as_ptr().add(di * bytes_per_vec)` doc reads and `q.as_ptr().add(..)`
    // query loads below depend on:
    //   * `packed.len() == n * bytes_per_vec` (all packed codes present),
    //   * `q.len() >= dim` (per-chunk query loads stay in bounds),
    //   * `dim % K == 0` (asserted immediately below).
    // `RankQuant::{new,add}` pack exactly `bytes_per_vec` bytes/doc and
    // `load_rankquant` re-validates the shape, so this holds on every path here.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        // Hard backstop (see `scan_b2_asym_avx2`): mis-dispatch must fail
        // loudly in release, not silently drop the trailing 64-code block.
        assert_eq!(
            dim % 64,
            0,
            "b=2 AVX-512 path needs dim % 64 == 0 for 4-way unroll"
        );
        let bytes_per_vec = dim / 4;
        let shifts = _mm512_setr_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
        let mask3 = _mm512_set1_epi32(3);

        let bytes_per_chunk = 4usize;
        let chunks_per_vec = bytes_per_vec / bytes_per_chunk;
        // Process 4 chunks per outer iteration with 4 independent
        // accumulators. Breaks the FMA dependency chain so the two Zen 5
        // FMA ports can both fire each cycle instead of waiting on a
        // single-acc dep chain.
        let outer_iters = chunks_per_vec / 4;
        debug_assert_eq!(chunks_per_vec % 4, 0);

        for di in 0..n {
            let doc = packed.as_ptr().add(di * bytes_per_vec);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            for outer in 0..outer_iters {
                let c0 = outer * 4;
                let c1 = c0 + 1;
                let c2 = c0 + 2;
                let c3 = c0 + 3;

                macro_rules! step {
                    ($c:expr, $acc:expr) => {{
                        let chunk_ptr = doc.add($c * bytes_per_chunk);
                        let b0 = *chunk_ptr as u32;
                        let b1 = *chunk_ptr.add(1) as u32;
                        let b2 = *chunk_ptr.add(2) as u32;
                        let b3 = *chunk_ptr.add(3) as u32;
                        let chunk = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
                        let broadcast = _mm512_set1_epi32(chunk as i32);
                        let codes = _mm512_and_si512(_mm512_srlv_epi32(broadcast, shifts), mask3);
                        let codes_f = _mm512_cvtepi32_ps(codes);
                        let d_base = $c * 16;
                        let q_vec = _mm512_loadu_ps(q.as_ptr().add(d_base));
                        $acc = _mm512_fmadd_ps(codes_f, q_vec, $acc);
                    }};
                }
                step!(c0, acc0);
                step!(c1, acc1);
                step!(c2, acc2);
                step!(c3, acc3);
            }

            let s01 = _mm512_add_ps(acc0, acc1);
            let s23 = _mm512_add_ps(acc2, acc3);
            let total = _mm512_add_ps(s01, s23);
            let raw = _mm512_reduce_add_ps(total);
            top.maybe_insert(raw * scale, di);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
pub(crate) unsafe fn scan_b4_asym_avx512(
    packed: &[u8],
    n: usize,
    dim: usize,
    q: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    use std::arch::x86_64::*;

    // SAFETY: a `pub(crate) unsafe fn` reachable only via `quant.rs`'s
    // runtime-detected dispatch, which upholds the invariants the raw
    // `packed.as_ptr().add(di * bytes_per_vec)` doc reads and `q.as_ptr().add(..)`
    // query loads below depend on:
    //   * `packed.len() == n * bytes_per_vec` (all packed codes present),
    //   * `q.len() >= dim` (per-chunk query loads stay in bounds),
    //   * `dim % K == 0` (asserted immediately below).
    // `RankQuant::{new,add}` pack exactly `bytes_per_vec` bytes/doc and
    // `load_rankquant` re-validates the shape, so this holds on every path here.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        // Hard backstop (see `scan_b2_asym_avx2`): mis-dispatch must fail
        // loudly in release, not silently drop the trailing 64-code block.
        assert_eq!(
            dim % 64,
            0,
            "b=4 AVX-512 path needs dim % 64 == 0 for 4-way unroll"
        );
        let bytes_per_vec = dim / 2;
        let shifts = _mm512_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0, 28, 24, 20, 16, 12, 8, 4, 0);
        let mask_f = _mm512_set1_epi32(0xF);

        let bytes_per_chunk = 8usize;
        let chunks_per_vec = bytes_per_vec / bytes_per_chunk;
        let outer_iters = chunks_per_vec / 4;
        debug_assert_eq!(chunks_per_vec % 4, 0);

        for di in 0..n {
            let doc = packed.as_ptr().add(di * bytes_per_vec);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            for outer in 0..outer_iters {
                macro_rules! step {
                    ($c:expr, $acc:expr) => {{
                        let chunk_ptr = doc.add($c * bytes_per_chunk);
                        let lo0 = *chunk_ptr as u32;
                        let lo1 = *chunk_ptr.add(1) as u32;
                        let lo2 = *chunk_ptr.add(2) as u32;
                        let lo3 = *chunk_ptr.add(3) as u32;
                        let hi0 = *chunk_ptr.add(4) as u32;
                        let hi1 = *chunk_ptr.add(5) as u32;
                        let hi2 = *chunk_ptr.add(6) as u32;
                        let hi3 = *chunk_ptr.add(7) as u32;
                        let chunk_lo = (lo0 << 24) | (lo1 << 16) | (lo2 << 8) | lo3;
                        let chunk_hi = (hi0 << 24) | (hi1 << 16) | (hi2 << 8) | hi3;
                        let lo_zmm = _mm512_set1_epi32(chunk_lo as i32);
                        let hi_zmm = _mm512_set1_epi32(chunk_hi as i32);
                        // Blend mask 0xFF00 (bits 8-15 set): _mm512_mask_blend_epi32
                        // takes lane i from `hi_zmm` where bit i is set, else from
                        // `lo_zmm` — so lanes 0-7 <- chunk_lo, lanes 8-15 <- chunk_hi.
                        // Pairs with `shifts` = [28,24,20,16,12,8,4,0] x2: lanes 0-7
                        // extract chunk_lo's 8 nibbles (codes 0-7), lanes 8-15 extract
                        // chunk_hi's (codes 8-15), most-significant nibble first.
                        let combined = _mm512_mask_blend_epi32(0xFF00u16, lo_zmm, hi_zmm);
                        let codes = _mm512_and_si512(_mm512_srlv_epi32(combined, shifts), mask_f);
                        let codes_f = _mm512_cvtepi32_ps(codes);
                        let d_base = $c * 16;
                        let q_vec = _mm512_loadu_ps(q.as_ptr().add(d_base));
                        $acc = _mm512_fmadd_ps(codes_f, q_vec, $acc);
                    }};
                }
                let c0 = outer * 4;
                step!(c0, acc0);
                step!(c0 + 1, acc1);
                step!(c0 + 2, acc2);
                step!(c0 + 3, acc3);
            }

            let s01 = _mm512_add_ps(acc0, acc1);
            let s23 = _mm512_add_ps(acc2, acc3);
            let total = _mm512_add_ps(s01, s23);
            let raw = _mm512_reduce_add_ps(total);
            top.maybe_insert(raw * scale, di);
        }
    }
}

/// Single entry point for the `b=8` asymmetric scan.
///
/// Builds the shared `dim * 256` per-coordinate LUT once
/// ([`build_b8_asym_lut_into`]), then dispatches to the AVX-512 gather kernel
/// ([`scan_b8_asym_avx512_gather`]) when `avx512f` + `avx512bw` are detected at
/// runtime and `dim % 16 == 0`, falling back to the portable scalar reference
/// ([`scan_b8_to_topk`]) on every other target / CPU / dim. Centralising
/// the dispatch here keeps the `unsafe` SIMD reach in one place and out of
/// `quant.rs`.
pub(crate) fn scan_b8_asym(
    packed: &[u8],
    n: usize,
    dim: usize,
    q_unit: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let mut lut = Vec::new();
    scan_b8_asym_with_lut(packed, n, dim, q_unit, scale, top, &mut lut);
}

pub(crate) fn scan_b8_asym_with_lut(
    packed: &[u8],
    n: usize,
    dim: usize,
    q_unit: &[f32],
    scale: f32,
    top: &mut TopK,
    lut: &mut Vec<f32>,
) {
    assert_eq!(q_unit.len(), dim);
    build_b8_asym_lut_into(lut, q_unit);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && dim.is_multiple_of(16)
        {
            // SAFETY: `avx512f`+`avx512bw` are confirmed by the runtime detection above
            // and `dim % 16 == 0` satisfies the kernel's lane invariant;
            // `packed.len() == n * dim` and `lut.len() == dim * 256` hold by
            // construction (b=8 packs one byte/coord; the LUT is built just
            // above). The explicit block is required by
            // `#![deny(unsafe_op_in_unsafe_fn)]`.
            unsafe {
                scan_b8_asym_avx512_gather(packed, n, dim, lut, scale, top);
            }
            return;
        }
    }
    scan_b8_to_topk(packed, n, dim, lut, scale, top);
}

// -------------------------------------------------------------------
// AVX-512 gather kernel for the b=8 asymmetric path.
//
// Unlike b ∈ {2, 4} — whose tiny per-byte arithmetic (shift/mask/cvt/FMA)
// beats any memory indirection — b=8 carries a large per-coordinate
// 256-entry float LUT (`lut[d * 256 + code]`), so the score is an honest
// gather: `Σ_d lut[d * 256 + doc_code[d]]`. The dominant cost is the
// gather, which `vgatherdps` (`_mm512_i32gather_ps`) issues 16-wide in a
// single instruction.
//
// Per 16-coordinate chunk:
//   * load 16 doc bytes, zero-extend to i32 lanes (`_mm512_cvtepu8_epi32`);
//   * add the per-position row-base vector `[d*256, (d+1)*256, …]` so lane
//     `j` indexes `lut[(d+j) * 256 + code[d+j]]`;
//   * `_mm512_i32gather_ps(idx, lut_ptr, 4)` gathers all 16 contributions;
//   * accumulate (plain add — the LUT already encodes `q · centre`).
// Four independent accumulators break the add dependency chain, matching
// the b=2/b=4 AVX-512 kernels. Unlike those, b=8 needs no centre-drop
// trick: the asymmetric LUT bakes the per-coordinate query weight in, so
// there is no per-query constant offset to reapply at finalize.
//
// Caller must verify `is_x86_feature_detected!("avx512f") && ..("avx512bw")`
// once. `avx512bw` is gated alongside `avx512f` to match the rest of the
// crate's AVX-512 kernels (which require `avx512dq`) and to keep the byte
// widening (`_mm512_cvtepu8_epi32`) conservatively gated — the F-without-BW
// CPUs (KNL/KNM) are already excluded by the crate's `dq` requirement, so this
// adds no real exclusion. The LUT is the same `dim * 256` f32 layout the scalar
// `scan_b8_to_topk` consumes, so the two paths are score-parity (modulo f32
// summation order, within the crate's 1e-4 cross-backend tolerance).
// -------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn scan_b8_asym_avx512_gather(
    packed: &[u8],
    n: usize,
    dim: usize,
    lut: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    use std::arch::x86_64::*;

    // SAFETY: a `pub(crate) unsafe fn` reachable only via `quant.rs`'s
    // runtime-detected dispatch, which upholds the invariants the raw doc
    // reads (`packed.as_ptr().add(di * dim + base)`), the LUT gather
    // (`_mm512_i32gather_ps` off `lut.as_ptr()`), and the chunk loop depend
    // on:
    //   * `packed.len() == n * dim` (b=8 stores one byte per coordinate),
    //   * `lut.len() == dim * 256` (one 256-entry row per coordinate),
    //   * `dim % 16 == 0` (asserted immediately below) so the 16-lane chunk
    //     loop tiles each doc exactly with no tail.
    // Every gather index `(d + j) * 256 + code` is `< dim * 256` because
    // `d + j < dim` and `code <= 255`, so each gathered f32 is in-bounds.
    // `RankQuant::{new_asymmetric,add}` pack exactly `dim` bytes/doc and the
    // dispatch builds a `dim * 256` LUT, so this holds on every path here.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        // Hard backstop (see `scan_b2_asym_avx2`): mis-dispatch must fail
        // loudly in release, not silently drop the trailing chunk.
        assert_eq!(dim % 16, 0, "b=8 AVX-512 gather path needs dim % 16 == 0");
        assert_eq!(lut.len(), dim * 256, "b=8 LUT must be dim * 256 entries");
        let bytes_per_vec = dim; // one byte per coordinate
        let lut_ptr = lut.as_ptr();

        // Per-position row bases for one 16-lane chunk: lane j contributes
        // `j * 256`. The chunk's coordinate offset `c * 16 * 256` is folded
        // into the doc-byte indices below.
        let lane_row_base = _mm512_setr_epi32(
            0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584,
            3840,
        );
        let chunks_per_vec = bytes_per_vec / 16;

        for di in 0..n {
            let doc = packed.as_ptr().add(di * bytes_per_vec);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            // Round chunks down to a multiple of 4 for the unrolled body;
            // a `dim % 64 != 0` (but `% 16 == 0`) dim leaves a ≤3-chunk tail
            // handled by the single-accumulator loop after.
            let unrolled = chunks_per_vec & !3;

            let mut c = 0usize;
            while c < unrolled {
                macro_rules! step {
                    ($cc:expr, $acc:expr) => {{
                        // Coordinate base for this chunk: `cc * 16 * 256`.
                        let chunk_base = _mm512_set1_epi32(($cc * 16 * 256) as i32);
                        // Load 16 doc bytes, zero-extend to 16 i32 lanes.
                        let bytes = _mm_loadu_si128(doc.add($cc * 16) as *const __m128i);
                        let codes = _mm512_cvtepu8_epi32(bytes);
                        // idx[j] = chunk_base + (j * 256) + code[j]
                        //        = (cc*16 + j) * 256 + code[cc*16 + j]
                        let idx =
                            _mm512_add_epi32(_mm512_add_epi32(chunk_base, lane_row_base), codes);
                        // Gather 16 LUT contributions (scale = 4 bytes/f32).
                        let vals = _mm512_i32gather_ps::<4>(idx, lut_ptr);
                        $acc = _mm512_add_ps($acc, vals);
                    }};
                }
                step!(c, acc0);
                step!(c + 1, acc1);
                step!(c + 2, acc2);
                step!(c + 3, acc3);
                c += 4;
            }

            // Tail: remaining (< 4) chunks fold into acc0.
            while c < chunks_per_vec {
                let chunk_base = _mm512_set1_epi32((c * 16 * 256) as i32);
                let bytes = _mm_loadu_si128(doc.add(c * 16) as *const __m128i);
                let codes = _mm512_cvtepu8_epi32(bytes);
                let idx = _mm512_add_epi32(_mm512_add_epi32(chunk_base, lane_row_base), codes);
                let vals = _mm512_i32gather_ps::<4>(idx, lut_ptr);
                acc0 = _mm512_add_ps(acc0, vals);
                c += 1;
            }

            let s01 = _mm512_add_ps(acc0, acc1);
            let s23 = _mm512_add_ps(acc2, acc3);
            let total = _mm512_add_ps(s01, s23);
            let raw = _mm512_reduce_add_ps(total);
            top.maybe_insert(raw * scale, di);
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod b8_gather_tests {
    use super::{build_b8_asym_lut_into, scan_b8_asym_avx512_gather, scan_b8_to_topk};
    use crate::util::TopK;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Drain a `k`-slot `TopK` into a flat `(score, idx)` vec sorted by the
    /// collector's own composite key, so the two kernels are compared on the
    /// exact tuples a caller would receive.
    fn drain(top: &TopK, k: usize) -> (Vec<f32>, Vec<i64>) {
        let mut scores = vec![f32::NEG_INFINITY; k];
        let mut idxs = vec![-1i64; k];
        top.finalize_into(&mut scores, &mut idxs);
        (scores, idxs)
    }

    fn b8_lut(q_unit: &[f32]) -> Vec<f32> {
        let mut lut = Vec::new();
        build_b8_asym_lut_into(&mut lut, q_unit);
        lut
    }

    /// The AVX-512 `vgatherdps` b=8 kernel must match the scalar LUT
    /// reference within the crate's 1e-4 cross-backend score tolerance,
    /// across the headline embedding dims (all `% 16 == 0`, so the gather
    /// path is actually exercised). 768/1536 are `% 64 == 0` (full
    /// 4-way-unrolled body); to also cover the ≤3-chunk tail path we add
    /// dim=400 (`400 % 16 == 0`, `400 % 64 == 16`).
    #[test]
    fn b8_gather_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("skipping b8 gather parity: no avx512f+avx512bw on this host");
            return;
        }
        for &dim in &[384usize, 400, 768, 1024, 1536] {
            assert_eq!(dim % 16, 0, "test dims must be % 16 for the gather path");
            let n = 64;
            let k = 10;
            let mut rng = ChaCha8Rng::seed_from_u64(0x00B8_0000 + dim as u64);

            // Random doc codes (any byte 0..=255) and a random unit-ish query.
            let packed: Vec<u8> = (0..n * dim).map(|_| rng.random::<u8>()).collect();
            let q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let q_unit: Vec<f32> = q.iter().map(|x| x / qn).collect();
            let scale = 1.0f32 / 137.0; // arbitrary inv_norm-like scale

            let lut = b8_lut(&q_unit);

            let mut top_scalar = TopK::new(k);
            scan_b8_to_topk(&packed, n, dim, &lut, scale, &mut top_scalar);
            let (s_scalar, i_scalar) = drain(&top_scalar, k);

            let mut top_gather = TopK::new(k);
            // SAFETY: avx512f+avx512bw confirmed above; dim % 16 == 0; packed has
            // n*dim bytes and lut has dim*256 entries by construction.
            unsafe {
                scan_b8_asym_avx512_gather(&packed, n, dim, &lut, scale, &mut top_gather);
            }
            let (s_gather, i_gather) = drain(&top_gather, k);

            for slot in 0..k {
                assert!(
                    (s_scalar[slot] - s_gather[slot]).abs() < 1e-4,
                    "dim={dim} slot={slot}: scalar {} vs gather {}",
                    s_scalar[slot],
                    s_gather[slot],
                );
            }
            // With well-separated random scores the top-k id sets agree too.
            assert_eq!(
                i_scalar, i_gather,
                "dim={dim}: top-{k} id ordering diverged between scalar and gather"
            );
        }
    }

    /// The gather kernel's per-doc raw score equals the brute-force
    /// `Σ_d lut[d*256 + code[d]]` (before the `scale` multiply), confirming
    /// the index math `idx[j] = (c*16 + j) * 256 + code` is exact.
    ///
    /// This compares the *unscaled* sum, whose magnitude (~10² for centred
    /// b=8 codes up to ±127.5 over `dim` terms) is far larger than the
    /// `inv_norm`-scaled score a caller sees. The SIMD kernel's 4-way
    /// parallel accumulation rounds in a different order from the strict
    /// sequential brute-force, so the check is *relative* (~1e-5): the
    /// production 1e-4 *absolute* tolerance applies to the small final
    /// scaled scores, which the parity test above covers.
    #[test]
    fn b8_gather_raw_score_is_exact_gather_sum() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            return;
        }
        let dim = 256usize;
        let n = 8;
        let k = n;
        let mut rng = ChaCha8Rng::seed_from_u64(0x00B8_FACE);
        let packed: Vec<u8> = (0..n * dim).map(|_| rng.random::<u8>()).collect();
        let q_unit: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let lut = b8_lut(&q_unit);

        let mut top = TopK::new(k);
        // SAFETY: avx512f+avx512bw confirmed; dim % 16 == 0; shapes match.
        unsafe {
            scan_b8_asym_avx512_gather(&packed, n, dim, &lut, 1.0, &mut top);
        }
        let (scores, idxs) = drain(&top, k);

        // Brute-force reference, indexed by returned doc id.
        let want: Vec<f32> = (0..n)
            .map(|di| {
                let doc = &packed[di * dim..(di + 1) * dim];
                doc.iter()
                    .enumerate()
                    .map(|(d, &code)| lut[d * 256 + code as usize])
                    .sum::<f32>()
            })
            .collect();
        for slot in 0..k {
            let di = idxs[slot] as usize;
            let rel = (scores[slot] - want[di]).abs() / want[di].abs().max(1.0);
            assert!(
                rel < 1e-4,
                "doc {di}: gather {} vs brute {} (rel {rel})",
                scores[slot],
                want[di]
            );
        }
    }

    /// Honest, kernel-isolated micro-benchmark: b=8 scalar LUT vs b=8
    /// AVX-512 gather vs the b=4 AVX-512 asym kernel, on the same N×dim
    /// corpus. `#[ignore]` so it does not run in the default gate — invoke
    /// with:
    ///
    /// ```text
    /// cargo test --release --lib b8_kernel_microbench -- --ignored --nocapture
    /// ```
    ///
    /// It times the inner scan only (LUT build + scan), so the scalar-vs-SIMD
    /// decision is measured directly rather than inferred. Per-iteration
    /// wall time is reported in ms and as ns/doc/dim so the cost is
    /// comparable across widths. Numbers are wall-clock and vary run-to-run;
    /// the parity tests above are the correctness gate.
    #[test]
    #[ignore = "perf micro-bench; run explicitly with --ignored --nocapture --release"]
    fn b8_kernel_microbench() {
        use crate::quant_kernels::{scan_b4_asym_avx512, scan_b8_asym_avx512_gather};
        use std::time::Instant;

        let have_avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512bw"); // b=4 path needs dq, b=8 gather needs bw
        let dim = 1024usize; // % 64 == 0 → valid for both b=4 and b=8 SIMD
        let n = 50_000usize;
        let k = 10usize;
        let iters = 20usize;

        let mut rng = ChaCha8Rng::seed_from_u64(0x00B8_4BE4);
        let q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q_unit: Vec<f32> = q.iter().map(|x| x / qn).collect();
        let scale = 1.0f32 / 137.0;

        // b=8 corpus: one byte per coord.
        let packed8: Vec<u8> = (0..n * dim).map(|_| rng.random::<u8>()).collect();
        // b=4 corpus: two codes per byte → dim/2 bytes per doc.
        let packed4: Vec<u8> = (0..n * dim / 2).map(|_| rng.random::<u8>()).collect();

        let lut8 = b8_lut(&q_unit);

        let bench = |label: &str, mut f: Box<dyn FnMut()>| {
            f(); // warmup
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            let ns_per_doc_dim = per * 1e9 / (n as f64 * dim as f64);
            let gdocs = n as f64 / per / 1e9;
            println!(
                "  {label:<26} {:>8.3} ms/scan  {:>7.3} ns/doc/dim  {:>7.3} Gdoc/s",
                per * 1e3,
                ns_per_doc_dim,
                gdocs,
            );
        };

        println!(
            "\nb=8 asymmetric kernel micro-bench (dim={dim}, n={n}, k={k}, iters={iters}, avx512={have_avx512})"
        );

        {
            let packed8 = packed8.clone();
            let lut8 = lut8.clone();
            bench(
                "b=8 scalar LUT",
                Box::new(move || {
                    let mut top = TopK::new(k);
                    scan_b8_to_topk(&packed8, n, dim, &lut8, scale, &mut top);
                    std::hint::black_box(&top);
                }),
            );
        }

        if have_avx512 {
            let packed8 = packed8.clone();
            let lut8 = lut8.clone();
            bench(
                "b=8 AVX-512 gather",
                Box::new(move || {
                    let mut top = TopK::new(k);
                    // SAFETY: avx512f+avx512bw confirmed; dim % 16 == 0; shapes match.
                    unsafe {
                        scan_b8_asym_avx512_gather(&packed8, n, dim, &lut8, scale, &mut top);
                    }
                    std::hint::black_box(&top);
                }),
            );

            // b=4 AVX-512 asym for cross-width context (raw codes, no LUT;
            // dim % 64 == 0 satisfies its lane invariant).
            let packed4 = packed4.clone();
            let q_unit4 = q_unit.clone();
            bench(
                "b=4 AVX-512 asym (context)",
                Box::new(move || {
                    let mut top = TopK::new(k);
                    // SAFETY: avx512f+dq confirmed; dim % 64 == 0; shapes match.
                    unsafe {
                        scan_b4_asym_avx512(&packed4, n, dim, &q_unit4, scale, &mut top);
                    }
                    std::hint::black_box(&top);
                }),
            );
        } else {
            println!("  (avx512 unavailable — SIMD rows skipped)");
        }
    }
}
