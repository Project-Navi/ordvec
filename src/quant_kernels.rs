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
    let mut lut = vec![0.0f32; dim * n_buckets];
    for d in 0..dim {
        for b in 0..n_buckets {
            lut[d * n_buckets + b] = q_unit[d] * bucket_centre(b as u8, bits);
        }
    }
    match bits {
        1 => scan_b1_to_topk(packed, n, dim, &lut, scale, top),
        2 => scan_b2_to_topk(packed, n, dim, &lut, scale, top),
        4 => scan_b4_to_topk(packed, n, dim, &lut, scale, top),
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
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
