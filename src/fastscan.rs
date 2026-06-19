//! FastScan b=2 scan path ([`RankQuantFastscan`]).
//!
//! An *optional, non-default* alternative to
//! [`RankQuant::search_asymmetric`](crate::quant::RankQuant)
//! at `bits == 2`. It re-blocks the b=2 bucket codes into a block-32,
//! PQ-style nibble layout and scores 32 documents per VPSHUFB lookup
//! against a per-query 8-bit affine-quantised LUT — the classic FAISS
//! FastScan trick. It trades roughly 2× the storage of the single-rate
//! b=2 packing for a lower per-query scan latency at matched recall;
//! the `RankQuant b=2 fastscan` row of `examples/bench_rank` reports
//! the in-repo, reproducible comparison against the single-rate kernel.
//!
//! Cost: `dim / 2` bytes per document (2× the single-rate
//! [`RankQuant`](crate::RankQuant) b=2 packing), and
//! a single-shot `add()` (the block layout's tail padding does not
//! compose with incremental extend).
//!
//! [`RankQuantFastscan`] is a stable, documented *but specialized* public
//! type — not the headline API. The free [`search_asymmetric_fastscan_b2`]
//! entry point stays `pub(crate)`: production callers should reach for
//! [`RankQuant::search_asymmetric`](crate::RankQuant::search_asymmetric),
//! whose AVX-512 → AVX2 → scalar dispatch is the maintained surface. Prefer
//! FastScan only when b=2 scan latency is the binding constraint.
//! This latency path is not part of the constant-weight bitmap overlap
//! calibration theorem.
//!
//! # Provenance
//!
//! This FastScan path consolidates the author's earlier `rank-modes`
//! development (originally a single `rank_index.rs` module): the b=2
//! kernel, the type wrapper with a `checked_mul` overflow guard, input
//! validation, the `k == 0` short-circuit, and independent
//! feature-detection dispatch. Integrated against ordvec's decomposed,
//! hardened `quant`/`util` modules: result buffers are sized through
//! [`result_buffer_len`](crate::util::result_buffer_len) (overflow-
//! safe `nq * k`), queries are normalised through the shared
//! [`l2_normalise`](crate::util::l2_normalise), and `k` is clamped to
//! `n_vectors` exactly as the sibling search methods do.

use rayon::prelude::*;

use crate::rank::{bucket_ranks, rank_transform, rankquant_norm};
use crate::util::{assert_all_finite, l2_normalise, result_buffer_len, TopK};
use crate::SearchResults;

// -------------------------------------------------------------------
// FastScan b=2 (block-32, nibble-LUT, VPSHUFB).
//
// Layout: 32 docs per block, indexed [block][coord_pair][lane32].
// For b=2 we group pairs of coords into one nibble (high=code_a,
// low=code_b). 16-entry LUT per coord-pair, scored 32 docs at a time
// with VPSHUFB on a broadcast __m256i (16-byte LUT replicated into
// each 128-bit lane).
//
// Quantization: all coord-pair LUTs are jointly affine-quantized to
// u8 with a single per-query (offset, scale). Sum of biases is
// tracked once per query and applied at finalize time. Float-domain
// dequant: raw = bias_sum + acc / scale.
//
// Accumulator widening to avoid u8 overflow:
//   - u8 contributions promoted to u16 each pair
//   - flush u16 -> u32 every 256 pairs (safely under u16 wrap)
// For dim=1024 that's pairs=512, one flush mid-stream + final.
// -------------------------------------------------------------------

/// Re-block standard packed b=2 codes into the FastScan block-32
/// nibble layout. `buckets` is the raw bucket array (length `n*dim`,
/// one u8 per coord), NOT the packed bytes — keeps the helper small
/// and the SIMD path indifferent to the source packing.
///
/// Output shape: `n_blocks` contiguous blocks. Each block holds
/// `dim/2` coord-pair groups of 32 bytes. Block layout for block `b`,
/// pair `p`, lane `l`:
///
/// ```text
///   nibble = (bucket[doc=b*32+l, coord=2p] << 2) | bucket[..., 2p+1]
/// ```
///
/// Tail docs (`n % 32 != 0`) are zero-padded to a full block; the
/// scan kernel only emits scores for `n` real docs.
///
/// Internal — call via [`RankQuantFastscan`], which owns the
/// packed bytes and enforces `(n, dim, packed.len())` consistency
/// by construction.
pub(crate) fn pack_fastscan_b2(buckets: &[u8], n: usize, dim: usize) -> Vec<u8> {
    assert_eq!(dim % 2, 0, "fastscan b=2 needs dim % 2 == 0");
    assert_eq!(buckets.len(), n * dim, "buckets must be n*dim");
    let pairs = dim / 2;
    let n_blocks = n.div_ceil(32);
    let bytes_per_block = pairs * 32;
    let mut out = vec![0u8; n_blocks * bytes_per_block];

    for b in 0..n_blocks {
        let block_offset = b * bytes_per_block;
        let doc_base = b * 32;
        let docs_in_block = (n - doc_base).min(32);
        for lane in 0..docs_in_block {
            let row = (doc_base + lane) * dim;
            for p in 0..pairs {
                let a = buckets[row + 2 * p] & 0x3;
                let c = buckets[row + 2 * p + 1] & 0x3;
                let nibble = (a << 2) | c;
                out[block_offset + p * 32 + lane] = nibble;
            }
        }
    }
    out
}

/// Build per-query metadata for a FastScan b=2 scan. Returns the
/// packed u8 LUTs (one 16-byte LUT per coord pair), the global
/// affine offset `bias_sum` and inverse scale `inv_q` such that
/// `raw_score = bias_sum + (u32_accumulator as f32) * inv_q`.
///
/// The single global affine fits all pair-LUTs into the same u8 scale
/// so we can sum quantized values across pairs without per-pair
/// rescaling — the classic FAISS FastScan LUT trick.
///
/// # Precision trade-off (known, intentional)
/// One global affine `[g_min, g_max] → [0, 255]` quantizes *every* coord-pair
/// LUT to the same 8-bit scale. The per-entry error is `O(span / 255)`; when a
/// single coord pair has an outlier score range it widens `span`, so the other
/// pairs lose relative precision. This is the standard FastScan approximation
/// (the same trade-off FAISS makes) and is acceptable for the b=2
/// approximate-scoring / candidate role this path serves — it is a fast
/// pre-ranker, not the exact scorer. Callers needing exact scores use
/// [`RankQuant::search_asymmetric`](crate::RankQuant::search_asymmetric).
fn build_fastscan_b2_query(q: &[f32], dim: usize) -> (Vec<u8>, f32, f32) {
    let pairs = dim / 2;
    // Centres for b=2: bucket b ∈ {0,1,2,3} → centre = b - 1.5
    // ∈ {-1.5, -0.5, +0.5, +1.5}. This matches
    // `crate::rank::bucket_centre(b, 2)` exactly.
    let centres = [-1.5_f32, -0.5, 0.5, 1.5];

    let mut lut_f = vec![0f32; pairs * 16];
    let mut g_min = f32::INFINITY;
    let mut g_max = f32::NEG_INFINITY;
    for p in 0..pairs {
        let qa = q[2 * p];
        let qc = q[2 * p + 1];
        for a in 0..4 {
            for c in 0..4 {
                let nibble = (a << 2) | c;
                let v = qa * centres[a] + qc * centres[c];
                lut_f[p * 16 + nibble] = v;
                if v < g_min {
                    g_min = v;
                }
                if v > g_max {
                    g_max = v;
                }
            }
        }
    }

    // Affine quant: q_u8 = round((v - g_min) * scale), scale = 255 / (g_max - g_min).
    // Per-pair bias = g_min (constant across nibbles). Sum across all pairs:
    //   bias_sum = pairs * g_min
    // Dequant: raw = bias_sum + sum(q_u8) / scale
    let span = (g_max - g_min).max(1e-12);
    let scale = 255.0_f32 / span;
    let inv_q = 1.0_f32 / scale;
    let bias_sum = pairs as f32 * g_min;

    let mut lut_u8 = vec![0u8; pairs * 16];
    for i in 0..pairs * 16 {
        let v = ((lut_f[i] - g_min) * scale).round().clamp(0.0, 255.0) as u8;
        lut_u8[i] = v;
    }
    (lut_u8, bias_sum, inv_q)
}

/// FastScan b=2 AVX-512 kernel. Scores `n` docs from `packed_fs` (built
/// by [`pack_fastscan_b2`]) against the per-query u8 LUTs.
///
/// `bias_sum` and `inv_q` come from [`build_fastscan_b2_query`].
/// `scale` is the per-query inv_norm applied as a final scalar
/// multiplier on the raw float score (matching the asym kernels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
unsafe fn scan_b2_fastscan_avx512(
    packed_fs: &[u8],
    n: usize,
    dim: usize,
    lut_u8: &[u8],
    bias_sum: f32,
    inv_q: f32,
    scale: f32,
    top: &mut TopK,
) {
    use std::arch::x86_64::*;

    // SAFETY: every raw load below is proven in-bounds by invariants the caller
    // (`search_asymmetric_fastscan_b2`) establishes before dispatch:
    //   • `packed_fs.len() == n_blocks * pairs * 32`, asserted with the product
    //     formed via `checked_mul` (overflow → caller panics), where
    //     `n_blocks == n.div_ceil(32)`, `pairs == dim/2`, `bytes_per_block ==
    //     pairs * 32`.
    //   • `lut_u8.len() == pairs * 16` (built by `build_fastscan_b2_query` for
    //     this `dim`).
    // For block `b ∈ 0..n_blocks`, `block_ptr = packed_fs + b*pairs*32`; the
    // per-pair 32-byte load `block_ptr.add(pp*32)` (`pp ∈ p..inner_end ⊆
    // 0..pairs`) reaches at most byte `(b+1)*pairs*32 - 1 ≤ len-1`, and the
    // 16-byte LUT load `lut_u8.add(pp*16)` reaches at most `pairs*16 - 1 =
    // lut_u8.len()-1`. AVX-512 F/BW/DQ are confirmed by the `#[target_feature]`
    // gate plus the caller's runtime `is_x86_feature_detected!`.

    let pairs = dim / 2;
    let bytes_per_block = pairs * 32;
    let n_blocks = n.div_ceil(32);

    // Flush u16 → u32 every FLUSH pairs. Max u8 contribution per pair
    // is 255, so FLUSH × 255 must fit in u16: FLUSH ≤ 257. Pick 256.
    const FLUSH: usize = 256;

    // SAFETY: every raw load/store and AVX-512 intrinsic in this loop is
    // in-bounds and feature-gated per the function-level SAFETY comment above.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        for b in 0..n_blocks {
            let block_ptr = packed_fs.as_ptr().add(b * bytes_per_block);

            // 32-lane u32 accumulators (split across two __m512i, lo/hi 16).
            let mut acc32_lo = _mm512_setzero_si512();
            let mut acc32_hi = _mm512_setzero_si512();

            let mut p = 0usize;
            while p < pairs {
                let chunk = (pairs - p).min(FLUSH);

                // 32-lane u16 accumulator split: each holds 16 u16 values
                // in its low 256 bits.
                let mut acc16_lo = _mm512_setzero_si512(); // lanes 0..16
                let mut acc16_hi = _mm512_setzero_si512(); // lanes 16..32

                let inner_end = p + chunk;
                let inner_chunks_4 = chunk / 4;
                let mut pp = p;

                // Score one coord-pair across all 32 lanes: VPSHUFB the per-pair
                // 16-byte LUT (broadcast into both 128-bit halves) by the packed
                // nibble codes, widen u8 -> u16, accumulate. `pp` / `block_ptr` /
                // `lut_u8` / `acc16_*` are captured by name at each call site.
                // (macro_rules is expanded at compile time, so defining it here
                // has no runtime cost; it keeps the unrolled body in one place and
                // is reused by the remainder loop below.)
                macro_rules! step {
                    ($off:expr) => {{
                        let codes256 =
                            _mm256_loadu_si256(block_ptr.add((pp + $off) * 32) as *const __m256i);
                        let lut128 = _mm_loadu_si128(
                            lut_u8.as_ptr().add((pp + $off) * 16) as *const __m128i
                        );
                        let lut256 = _mm256_broadcastsi128_si256(lut128);
                        let contrib = _mm256_shuffle_epi8(lut256, codes256);
                        let lo128 = _mm256_castsi256_si128(contrib);
                        let hi128 = _mm256_extracti128_si256(contrib, 1);
                        let lo256 = _mm256_cvtepu8_epi16(lo128);
                        let hi256 = _mm256_cvtepu8_epi16(hi128);
                        acc16_lo = _mm512_add_epi16(acc16_lo, _mm512_castsi256_si512(lo256));
                        acc16_hi = _mm512_add_epi16(acc16_hi, _mm512_castsi256_si512(hi256));
                    }};
                }

                // 4-wide unroll, then the remainder one pair at a time.
                for _ in 0..inner_chunks_4 {
                    step!(0);
                    step!(1);
                    step!(2);
                    step!(3);
                    pp += 4;
                }

                while pp < inner_end {
                    step!(0);
                    pp += 1;
                }

                // Widen u16 → u32. Meaningful u16s sit in the low 256 bits.
                let lo256_u16 = _mm512_castsi512_si256(acc16_lo);
                let hi256_u16 = _mm512_castsi512_si256(acc16_hi);
                let lo32 = _mm512_cvtepu16_epi32(lo256_u16);
                let hi32 = _mm512_cvtepu16_epi32(hi256_u16);
                acc32_lo = _mm512_add_epi32(acc32_lo, lo32);
                acc32_hi = _mm512_add_epi32(acc32_hi, hi32);

                p = inner_end;
            }

            let mut tmp_lo = [0u32; 16];
            let mut tmp_hi = [0u32; 16];
            _mm512_storeu_si512(tmp_lo.as_mut_ptr() as *mut _, acc32_lo);
            _mm512_storeu_si512(tmp_hi.as_mut_ptr() as *mut _, acc32_hi);

            let doc_base = b * 32;
            let docs_in_block = (n - doc_base).min(32);
            for lane in 0..docs_in_block {
                let acc = if lane < 16 {
                    tmp_lo[lane]
                } else {
                    tmp_hi[lane - 16]
                };
                let raw = bias_sum + (acc as f32) * inv_q;
                top.maybe_insert(raw * scale, doc_base + lane);
            }
        }
    }
}

/// Scalar reference for [`scan_b2_fastscan_avx512`]. Used for
/// correctness validation and on non-x86 / older x86 targets.
#[allow(clippy::too_many_arguments)] // kernel arity is intrinsic to the packed-scan signature
fn scan_b2_fastscan_scalar(
    packed_fs: &[u8],
    n: usize,
    dim: usize,
    lut_u8: &[u8],
    bias_sum: f32,
    inv_q: f32,
    scale: f32,
    top: &mut TopK,
) {
    let pairs = dim / 2;
    let bytes_per_block = pairs * 32;
    let n_blocks = n.div_ceil(32);
    for b in 0..n_blocks {
        let block_ptr = &packed_fs[b * bytes_per_block..(b + 1) * bytes_per_block];
        let doc_base = b * 32;
        let docs_in_block = (n - doc_base).min(32);
        let mut accs = [0u32; 32];
        for p in 0..pairs {
            for lane in 0..docs_in_block {
                let nibble = block_ptr[p * 32 + lane] as usize;
                accs[lane] += lut_u8[p * 16 + nibble] as u32;
            }
        }
        #[allow(clippy::needless_range_loop)]
        // indexed access is clearer / matches the kernel layout
        for lane in 0..docs_in_block {
            let raw = bias_sum + (accs[lane] as f32) * inv_q;
            top.maybe_insert(raw * scale, doc_base + lane);
        }
    }
}

/// FastScan b=2 search entry point. `packed_fs` was built by
/// [`pack_fastscan_b2`].
///
/// Internal — call via [`RankQuantFastscan::search`], which
/// owns `(dim, n_vectors, packed_fs)` and enforces consistency by
/// construction. The `pub(crate)` visibility + asserts below are
/// defense-in-depth for in-crate callers; the type-level wrapper
/// is the user-facing safe API.
pub(crate) fn search_asymmetric_fastscan_b2(
    packed_fs: &[u8],
    n: usize,
    dim: usize,
    queries: &[f32],
    k: usize,
) -> SearchResults {
    // Validate the contract the unsafe AVX-512 kernel depends on.
    // The kernel computes per-doc / per-pair offsets from (n, dim);
    // every multiplication along that path must not overflow, or a
    // crafted (n, dim) pair could wrap to a value that satisfies the
    // assert below while the kernel reads past `packed_fs`.
    assert!(dim >= 2, "FastScan b=2: dim must be >= 2");
    assert_eq!(
        dim % 2,
        0,
        "FastScan b=2: dim {dim} must be even (pair-encoding)"
    );
    let pairs = dim / 2;
    let n_blocks = n.div_ceil(32);
    let expected_packed = n_blocks
        .checked_mul(pairs)
        .and_then(|x| x.checked_mul(32))
        .unwrap_or_else(|| {
            panic!(
                "FastScan b=2: n={n} dim={dim} packed-length \
                 computation overflows usize"
            )
        });
    assert_eq!(
        packed_fs.len(),
        expected_packed,
        "FastScan b=2: packed_fs.len()={} does not match n={n} dim={dim} (expected {expected_packed})",
        packed_fs.len(),
    );
    assert_eq!(
        queries.len() % dim,
        0,
        "FastScan b=2: queries.len()={} must be a multiple of dim={dim}",
        queries.len(),
    );

    let nq = queries.len() / dim;

    // Clamp `k` to `n_vectors` before it sizes any `vec![_; nq * k]`
    // allocation below; an unclamped `usize::MAX` otherwise aborts the
    // process with `capacity overflow`. Mirrors the sibling search
    // methods in `quant.rs`.
    let k = k.min(n);
    // Result-buffer length `nq * k`, panicking loudly on usize overflow
    // rather than silently sizing a too-small allocation. Same guard
    // the other search paths use.
    let buf_len = result_buffer_len(nq, k);

    // Short-circuit k == 0 (also covers an empty corpus once clamped):
    // `par_chunks_mut(0)` panics. Return a correctly-shaped result with
    // `k == 0`; `buf_len` is 0 here so the buffers are empty, matching
    // the other search methods' early-out.
    if k == 0 {
        return SearchResults {
            scores: vec![0.0; buf_len],
            indices: vec![-1; buf_len],
            nq,
            k,
        };
    }

    let mut scores = vec![f32::NEG_INFINITY; buf_len];
    let mut indices = vec![-1i64; buf_len];

    let centred_norm = rankquant_norm(dim, 2);
    let inv_norm = 1.0_f32 / centred_norm;

    queries
        .par_chunks(dim)
        .zip(scores.par_chunks_mut(k))
        .zip(indices.par_chunks_mut(k))
        .for_each(|((q, out_scores), out_indices)| {
            // Shared L2 normaliser (zeros a degenerate <=1e-12 query),
            // matching `RankQuant::search_asymmetric`.
            let q_unit = l2_normalise(q);

            let (lut_u8, bias_sum, inv_q) = build_fastscan_b2_query(&q_unit, dim);
            let mut top = TopK::new(k);

            // Independent runtime feature detection (per e08506d): the
            // AVX-512 FastScan kernel needs avx512f + avx512bw (VPSHUFB
            // / widening) + avx512dq (32-bit lane ops). Any host missing
            // one falls back to the byte-faithful scalar kernel, which
            // handles every even `dim`.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512dq")
                {
                    scan_b2_fastscan_avx512(
                        packed_fs, n, dim, &lut_u8, bias_sum, inv_q, inv_norm, &mut top,
                    );
                } else {
                    scan_b2_fastscan_scalar(
                        packed_fs, n, dim, &lut_u8, bias_sum, inv_q, inv_norm, &mut top,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            scan_b2_fastscan_scalar(
                packed_fs, n, dim, &lut_u8, bias_sum, inv_q, inv_norm, &mut top,
            );

            top.finalize_into(out_scores, out_indices);
        });

    SearchResults {
        scores,
        indices,
        nq,
        k,
    }
}

// -------------------------------------------------------------------
// RankQuantFastscan: type wrapper around the FastScan b=2 path.
//
// Wraps pack_fastscan_b2 + search_asymmetric_fastscan_b2 in a type
// that owns the packed bytes and enforces (n_vectors, dim,
// packed_fs.len()) consistency by construction. The architectural
// answer to "FastScan API can cause UB from safe Rust": the runtime
// asserts in the underlying pub(crate) functions become defense-in-
// depth rather than the primary safety boundary.
// -------------------------------------------------------------------

/// FastScan b=2 RankQuant index.
///
/// Same retrieval semantics as
/// [`RankQuant::search_asymmetric`](crate::RankQuant::search_asymmetric)
/// at b=2, up to 8-bit LUT quantization noise (the
/// `fastscan_b2_top10_matches_avx512_kernel` test checks the top-10
/// agreement against the single-rate kernel on synthetic data).
/// Single-pass single-index kernel for callers who can't restructure
/// their query path for the two-stage
/// [`Bitmap`](crate::Bitmap) →
/// [`RankQuant::search_asymmetric_subset`](crate::RankQuant::search_asymmetric_subset)
/// pipeline.
///
/// # Storage
///
/// `dim / 2` bytes per document (2× the single-rate `RankQuant`
/// at b=2). The block-32 layout doubles the byte rate in exchange for
/// lower per-query scan latency than the single-rate kernel; the
/// `RankQuant b=2 fastscan` row of `examples/bench_rank` reports the
/// in-repo comparison on the synthetic corpus.
///
/// # v1 limitations
///
/// - **Single-shot `add()`** — the block-32 layout doesn't compose
///   cleanly with incremental extend (tail padding within blocks
///   would interleave with new docs). Subsequent `add()` calls panic;
///   construct a new index for incremental scenarios.
/// - **No `swap_remove`** — the block-32 layout makes byte-exact in-place
///   updates non-trivial (a v2 follow-up). Persistence *is* supported:
///   [`write`](Self::write) / [`load`](Self::load) round-trip via the
///   `.ovfs` format.
///
/// # Concurrency
///
/// `search` takes `&self`; safe to call from multiple threads
/// concurrently.
///
/// # Positioning
///
/// A stable, documented public type, but a **specialized** one: it is the
/// minimum-latency b=2 scan path, not the headline retrieval API. Prefer
/// [`RankQuant`](crate::RankQuant) / [`Bitmap`](crate::Bitmap) / the two-stage
/// flow unless you have measured FastScan to win on your workload.
pub struct RankQuantFastscan {
    dim: usize,
    n_vectors: usize,
    /// Block-32 FastScan layout. Length = `n_blocks * (dim/2) * 32`.
    packed_fs: Vec<u8>,
}

impl std::fmt::Debug for RankQuantFastscan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankQuantFastscan")
            .field("dim", &self.dim)
            .field("n_vectors", &self.n_vectors)
            .finish()
    }
}

impl RankQuantFastscan {
    /// Construct an empty FastScan b=2 index.
    ///
    /// The accepted `dim` domain mirrors
    /// [`RankQuant::new(dim, 2)`](crate::RankQuant::new) exactly, so a
    /// FastScan index and its single-rate sibling agree on which
    /// dimensions are valid:
    ///
    /// - `dim <= u16::MAX` — the rank transform stores ranks as `u16`.
    ///   A larger `dim` would construct here but panic on the first
    ///   [`add`](Self::add), inside
    ///   [`rank_transform`](crate::rank::rank_transform).
    /// - `dim % 4 == 0` — b=2 buckets the rank axis into `2^2 = 4` equal
    ///   bins, so exactly `dim / 4` coordinates land in each bucket and
    ///   the analytical
    ///   [`rankquant_norm`](crate::rank::rankquant_norm) stays exact.
    ///   This subsumes the pair-encoding's `dim % 2 == 0`.
    ///
    /// # Panics
    /// Panics if `dim < 2`, `dim > u16::MAX`, or `dim % 4 != 0`.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2, "FastScan b=2: dim must be >= 2");
        assert!(
            dim <= u16::MAX as usize,
            "FastScan b=2: dim must fit in u16"
        );
        // Mirror `RankQuant::new(dim, 2)`: divisible by 2^bits = 4 so
        // every bucket receives exactly dim/4 ranks (keeps the analytical
        // rankquant_norm exact). dim % 4 == 0 implies the pair-encoding's
        // dim % 2 == 0, so this single check subsumes the old even guard.
        assert_eq!(
            dim % 4,
            0,
            "FastScan b=2: dim {dim} must be divisible by 4 \
             (b=2 constant composition; matches RankQuant::new(dim, 2))"
        );
        Self {
            dim,
            n_vectors: 0,
            packed_fs: Vec::new(),
        }
    }

    /// Add `n = vectors.len() / dim` vectors to the index.
    ///
    /// # Panics
    /// - Panics if `vectors.len()` is not a multiple of `dim`.
    /// - Panics on incremental extend (v1 single-shot limitation —
    ///   see the type's doc comment).
    pub fn add(&mut self, vectors: &[f32]) {
        assert_all_finite(vectors);
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim"
        );
        if n == 0 {
            return;
        }
        assert!(
            self.n_vectors == 0,
            "FastScan v1: incremental add() not supported (block-32 \
             layout has tail-padding semantics that don't compose); \
             construct a new RankQuantFastscan instead"
        );

        let mut buckets = Vec::with_capacity(n * self.dim);
        for d in 0..n {
            let r = rank_transform(&vectors[d * self.dim..(d + 1) * self.dim]);
            let b = bucket_ranks(&r, 2);
            buckets.extend_from_slice(&b);
        }
        self.packed_fs = pack_fastscan_b2(&buckets, n, self.dim);
        self.n_vectors = n;
    }

    /// Run top-`k` asymmetric search.
    ///
    /// Same scoring contract as
    /// [`RankQuant::search_asymmetric`](crate::RankQuant::search_asymmetric)
    /// at b=2, within 8-bit LUT quantization noise.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        assert_all_finite(queries);
        // (dim, n_vectors, packed_fs.len()) consistent by construction.
        search_asymmetric_fastscan_b2(&self.packed_fs, self.n_vectors, self.dim, queries, k)
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
    /// `dim / 2` bytes per document (2× the single-rate `RankQuant`
    /// b=2 packing).
    pub fn bytes_per_vec(&self) -> usize {
        self.dim / 2
    }
    /// Total bytes held by the packed buffer (excludes Vec overhead;
    /// includes per-block tail padding when `n_vectors % 32 != 0`).
    pub fn byte_size(&self) -> usize {
        self.packed_fs.len()
    }

    /// Persist this index to a `.ovfs` file (magic `OVFS`).
    ///
    /// The on-disk form is a 13-byte header (`OVFS` magic, version, `dim`,
    /// `n_vectors`) followed by the opaque block-32 packed FastScan payload.
    /// This is a new ordvec format with no turbovec-era counterpart. Round-trip
    /// is a type-level guarantee: [`Self::load`] reconstructs the same
    /// `(dim, n_vectors)` and packed buffer this writes.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_fastscan(path, self.dim, self.n_vectors, &self.packed_fs)
    }

    /// Persist to any byte writer using the `.ovfs` format.
    pub fn write_to<W: std::io::Write>(&self, writer: W) -> std::io::Result<()> {
        crate::rank_io::write_fastscan_to(writer, self.dim, self.n_vectors, &self.packed_fs)
    }

    /// Load a `.ovfs` FastScan index previously written by [`Self::write`].
    ///
    /// The loader validates the header and that the payload length is exactly
    /// the block-32 size implied by `(dim, n_vectors)` (`dim % 4 == 0`, no
    /// trailing bytes), so the returned index is consistent by construction.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (dim, n_vectors, packed_fs) = crate::rank_io::load_fastscan(path)?;
        Ok(Self {
            dim,
            n_vectors,
            packed_fs,
        })
    }

    /// Load a `.ovfs` FastScan index from any reader that can seek.
    ///
    /// The reader is parsed from its current position through EOF; any trailing
    /// bytes after the declared payload are rejected.
    pub fn read_from<R: std::io::Read + std::io::Seek>(reader: R) -> std::io::Result<Self> {
        let (dim, n_vectors, packed_fs) = crate::rank_io::load_fastscan_from(reader)?;
        Ok(Self {
            dim,
            n_vectors,
            packed_fs,
        })
    }

    /// Load a `.ovfs` FastScan index from an in-memory byte slice.
    pub fn load_from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::read_from(std::io::Cursor::new(bytes))
    }
}
