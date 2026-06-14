//! `B`-bit bucketed-rank index ([`RankQuant`]).
//!
//! Storage is `dim * bits / 8` bytes per document at `bits ∈ {1, 2, 4, 8}`
//! (`b=8` is one byte per coordinate). Symmetric search uses a per-query,
//! per-coord LUT; asymmetric search dispatches AVX-512 → AVX2 → scalar via
//! the kernels in [`crate::quant_kernels`].
//!
//! `b=8` is an evidence/refinement-oriented width: it is supported for
//! asymmetric scoring and code/projection generation at **any** dimension,
//! but symmetric scoring uses the equal-bucket analytical norm and therefore
//! requires `dim % 256 == 0`. For `b ∈ {1, 2, 4}` the existing retrieval
//! modes remain the stable headline surface; `b=8` is an opt-in,
//! explicitly-documented high-precision evidence/refinement surface
//! (e.g. asymmetric quant storage after repair flows, edge-case rerank
//! healing), not a broad retrieval-quant method. It is **not**
//! unstable-experimental. See [`RankQuantCapability`] and
//! [`RankQuant::new_asymmetric`]. Its asymmetric path is a per-coordinate
//! gather against the `dim * 256` LUT: an AVX-512 `vgatherdps` kernel when
//! available (`avx512f` + `dim % 16 == 0`), else the portable scalar LUT.
//!
//! The byte-LUT path ([`search_asymmetric_byte_lut`]) is re-exported
//! `#[doc(hidden)]` (reachable as `ordvec::search_asymmetric_byte_lut`)
//! so `examples/bench_rank.rs` can compare it against the production
//! AVX path on the same data.

use rayon::prelude::*;

use crate::quant_kernels::{
    scan_b1_to_topk, scan_b2_to_topk, scan_b4_to_topk, scan_b8_asym, scan_b8_to_topk,
    scan_via_lut_scalar,
};
#[cfg(target_arch = "x86_64")]
use crate::quant_kernels::{
    scan_b2_asym_avx2, scan_b2_asym_avx512, scan_b4_asym_avx2, scan_b4_asym_avx512,
};
use crate::rank::{
    bucket_centre, bucket_ranks, pack_buckets, rank_to_bucket, rank_transform,
    rankquant_bytes_per_vec, rankquant_norm,
};
use crate::sign_bitmap::SignBitmap;
use crate::util::{assert_all_finite, l2_normalise, result_buffer_len, TopK};
use crate::{validate_candidate_ids, OrdvecError, SearchResults};

fn check_eval_bits(bits: u8) {
    assert!((1..=7).contains(&bits), "bits must be in 1..=7");
}

fn rankquant_eval_norm(dim: usize, bits: u8) -> f32 {
    check_eval_bits(bits);
    assert!(dim >= 2, "dim must be >= 2");
    assert!(dim <= u16::MAX as usize, "dim must fit in u16");
    let mut acc = 0.0f64;
    for rank in 0..dim {
        let b = rank_to_bucket(rank as u16, dim, bits);
        let c = bucket_centre(b, bits) as f64;
        acc += c * c;
    }
    acc.sqrt() as f32
}

fn rankquant_eval_centres(v: &[f32], bits: u8, out: &mut [f32]) {
    debug_assert_eq!(v.len(), out.len());
    let ranks = rank_transform(v);
    for (dst, rank) in out.iter_mut().zip(ranks) {
        let bucket = rank_to_bucket(rank, v.len(), bits);
        *dst = bucket_centre(bucket, bits);
    }
}

fn rankquant_eval_buckets(v: &[f32], bits: u8, out: &mut [u8]) {
    debug_assert_eq!(v.len(), out.len());
    let ranks = rank_transform(v);
    for (dst, rank) in out.iter_mut().zip(ranks) {
        *dst = rank_to_bucket(rank, v.len(), bits);
    }
}

/// Which scoring modes a [`RankQuant`] instance supports.
///
/// The distinction only matters for `b=8`. For `b ∈ {1, 2, 4}` every
/// constructor produces a [`SymmetricAndAsymmetric`](Self::SymmetricAndAsymmetric)
/// instance (the `dim % 2^bits == 0` constructor invariant always holds),
/// so callers never need to branch on this for the headline widths.
///
/// For `b=8` the symmetric analytical L2 norm is exact only when every
/// bucket receives equal occupancy, i.e. `dim % 256 == 0`. When that
/// holds the instance is [`SymmetricAndAsymmetric`](Self::SymmetricAndAsymmetric);
/// otherwise it is [`AsymmetricOnly`](Self::AsymmetricOnly) — code/projection
/// generation, pair-evidence/contingency, and asymmetric (float-query)
/// scoring all work at *any* dim, but the symmetric path
/// ([`RankQuant::search`]) panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankQuantCapability {
    /// Asymmetric (float-query) scoring and code/projection generation
    /// only. Reachable for `b=8` when `dim % 256 != 0`. Symmetric
    /// scoring ([`RankQuant::search`]) panics on these instances.
    AsymmetricOnly,
    /// Full surface: both symmetric and asymmetric scoring. The only
    /// capability for `b ∈ {1, 2, 4}`, and the capability for `b=8` when
    /// `dim % 256 == 0`.
    SymmetricAndAsymmetric,
}

/// `B`-bit RankQuant index.
///
/// Each document is encoded by bucketing its rank vector into
/// `1 << bits` equal-width bins on `[0, dim)` and packing `bits` bits
/// per coordinate. Storage is `dim * bits / 8` bytes per document.
/// Supported bit widths are `1`, `2`, `4`, and `8` (3-bit packing is
/// left for a follow-up; use `2` or `4` in the interim).
///
/// The mean-centred bucket vector has fixed analytical L2 norm
/// `sqrt(dim * (2^(2B) - 1) / 12)` when `dim % (1 << bits) == 0`, so
/// no per-document norms are stored.
///
/// # `b=8` — evidence/refinement width
/// `b=8` is an evidence/refinement-oriented RankQuant width. It is
/// supported for asymmetric scoring and code/projection generation at
/// any dimension; symmetric scoring uses the equal-bucket analytical
/// norm and therefore requires `dim % 256 == 0`. For `b ∈ {1, 2, 4}`,
/// the existing retrieval modes remain the stable headline surface;
/// `b=8` is an opt-in, explicitly-documented high-precision
/// evidence/refinement surface (e.g. asymmetric quant storage after
/// repair flows, edge-case rerank healing), not a broad retrieval-quant
/// method. It is **not** unstable-experimental — it is a stable, core
/// surface — but it is capability-gated: construct an asymmetric-only
/// `b=8` index for non-`256`-aligned dims via [`Self::new_asymmetric`]
/// and check [`Self::symmetric_supported`] before calling
/// [`Self::search`]. See [`RankQuantCapability`].
pub struct RankQuant {
    pub(crate) dim: usize,
    pub(crate) bits: u8,
    pub(crate) n_vectors: usize,
    /// Scoring modes this instance supports — see [`RankQuantCapability`].
    /// Computed once at construction; for `b ∈ {1, 2, 4}` always
    /// [`RankQuantCapability::SymmetricAndAsymmetric`].
    pub(crate) capability: RankQuantCapability,
    /// Row-major packed bucket bytes. `n_vectors * dim * bits / 8` total.
    pub(crate) packed: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoStageCandidatePolicy {
    pub min_candidates: usize,
    pub k_multiplier: usize,
    pub max_candidates: Option<usize>,
}

impl TwoStageCandidatePolicy {
    pub fn candidate_count(&self, k: usize, search_space: usize) -> usize {
        if k == 0 || search_space == 0 {
            return 0;
        }
        let mut count = self.min_candidates.max(k.saturating_mul(self.k_multiplier));
        if let Some(max_candidates) = self.max_candidates {
            count = count.min(max_candidates);
        }
        count.min(search_space)
    }
}

impl Default for TwoStageCandidatePolicy {
    fn default() -> Self {
        Self {
            min_candidates: 256,
            k_multiplier: 32,
            max_candidates: None,
        }
    }
}

/// SIMD dispatch tier for the asymmetric scan kernels.
///
/// Tier selection is gated on *both* runtime CPU features and the
/// kernel lane invariant for the configured `(dim, bits)` — see
/// [`select_simd_tier`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
// `Avx2`/`Avx512` are constructed only by the x86_64 SIMD dispatch.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
enum SimdTier {
    None,
    Avx2,
    Avx512,
}

/// Choose the asymmetric scan tier for `(dim, bits)`.
///
/// Each SIMD kernel carries a lane invariant on `dim`:
///
/// - **AVX-512** (`scan_b{2,4}_asym_avx512`): processes 64 codes per
///   outer iteration (4-way unrolled, 16 codes/chunk), so it requires
///   `dim % 64 == 0`.
/// - **AVX2** (`scan_b2_asym_avx2` / `scan_b4_asym_avx2`): b=2 emits 16
///   codes per 4-byte chunk (`dim % 16 == 0`); b=4 emits 8 codes per
///   4-byte chunk (`dim % 8 == 0`).
///
/// The [`RankQuant::new`] constructor only guarantees
/// `dim % (1 << bits) == 0` and `dim % (8 / bits) == 0`, which is
/// *weaker* than the SIMD invariants (e.g. dim 48 / 80 / 20 are valid
/// constructor dims that violate them). A kernel whose invariant is
/// unmet hits a hard `assert!` and panics in release — the kernels
/// enforce their lane invariant in every build, by design. This
/// selector returns the highest tier whose invariant holds — falling
/// back to [`SimdTier::None`] (scalar LUT, which handles any valid dim)
/// when neither SIMD tier fits, so a constructor-valid-but-SIMD-invalid
/// dim never reaches a kernel that would reject it.
#[inline]
fn select_simd_tier(dim: usize, bits: u8) -> SimdTier {
    // SIMD asymmetric kernels exist only for b ∈ {2, 4}. b=1 (and any
    // future unsupported width) always takes the scalar LUT path, which
    // is also where the byte-LUT bench helper's {2,4}-only restriction
    // is sidestepped: `search_asymmetric` never feeds a b=1 index to a
    // {2,4}-only kernel.
    if !matches!(bits, 2 | 4) {
        return SimdTier::None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq");
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        // AVX-512 first: both supported widths pack 64 codes/outer-iter,
        // so the single invariant is `dim % 64 == 0`.
        if avx512 && dim.is_multiple_of(64) {
            return SimdTier::Avx512;
        }
        // AVX2: per-width lane invariant.
        if avx2 && ((bits == 2 && dim.is_multiple_of(16)) || (bits == 4 && dim.is_multiple_of(8))) {
            return SimdTier::Avx2;
        }
        SimdTier::None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dim, bits);
        SimdTier::None
    }
}

impl RankQuant {
    /// Validate `(dim, bits)` for **code validity** — the precondition for
    /// generating bucket codes, projections, and asymmetric scores.
    ///
    /// Accepts `bits ∈ {1, 2, 4, 8}` and `dim ∈ [2, u16::MAX]`.
    ///
    /// For `b ∈ {1, 2, 4}` this additionally requires `dim % 2^bits == 0`
    /// (the equal-bucket constant-composition invariant): those widths only
    /// expose a full symmetric+asymmetric surface, so code validity and
    /// symmetric-norm validity coincide.
    ///
    /// For `b = 8` it validates **only** that codes pack (`codes_per_byte ==
    /// 1`, so any `dim` works) — it does **not** require `dim % 256 == 0`.
    /// That `dim % 256 == 0` rule is a *symmetric-scoring* precondition, not
    /// a code-validity one, and is checked separately on the symmetric path
    /// (and by [`Self::new`], which constructs a full-capability `b=8`
    /// instance). Use [`Self::new_asymmetric`] for any-`dim` `b=8`.
    pub fn validate_params(dim: usize, bits: u8) -> Result<(), OrdvecError> {
        if !matches!(bits, 1 | 2 | 4 | 8) {
            return Err(OrdvecError::InvalidParameter {
                name: "bits",
                message: "must be 1, 2, 4, or 8".to_string(),
            });
        }
        if dim < 2 {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: "must be >= 2".to_string(),
            });
        }
        if dim > u16::MAX as usize {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: "must fit in u16".to_string(),
            });
        }
        let codes_per_byte = (8 / bits) as usize;
        if !dim.is_multiple_of(codes_per_byte) {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: format!("must be a multiple of {codes_per_byte} for bits = {bits}"),
            });
        }
        // The constant-composition invariant `dim % 2^bits == 0` exists only to
        // make the symmetric analytical L2 norm exact (equal bucket occupancy).
        // For b ∈ {1,2,4} we keep requiring it here (those widths are
        // full-capability by definition), but for b=8 it is a *symmetric*
        // precondition checked elsewhere — code/projection/asymmetric paths
        // never need equal buckets, so a non-256-aligned dim is a valid b=8
        // *code* configuration.
        if bits != 8 {
            let n_buckets = 1usize << bits;
            if !dim.is_multiple_of(n_buckets) {
                return Err(OrdvecError::InvalidParameter {
                    name: "dim",
                    message: format!(
                        "must be divisible by 2^bits = {n_buckets} so every bucket receives exactly dim / 2^bits rank entries"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Construct a full-capability (`SymmetricAndAsymmetric`) index.
    ///
    /// For `b ∈ {1, 2, 4}` this is unchanged: `bits` must be one of those
    /// widths and `dim % 2^bits == 0` (and `dim % (8 / bits) == 0`).
    ///
    /// For `b = 8` this requires `dim % 256 == 0`, which yields the full
    /// symmetric+asymmetric surface. If `dim % 256 != 0` it **panics**
    /// (consistent with this constructor's existing fail-loud style),
    /// directing the caller to [`Self::new_asymmetric`] for an any-`dim`
    /// asymmetric-only `b=8` index. See [`RankQuantCapability`].
    ///
    /// # Panics
    /// Panics if `bits ∉ {1, 2, 4, 8}`, if `dim < 2`, if `dim > u16::MAX`,
    /// if `dim % (8 / bits) != 0`, or — for the equal-bucket symmetric
    /// invariant — if `dim % 2^bits != 0` (`b ∈ {1,2,4}`) / `dim % 256 != 0`
    /// (`b = 8`).
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(matches!(bits, 1 | 2 | 4 | 8), "bits must be 1, 2, 4, or 8");
        assert!(dim >= 2, "dim must be >= 2");
        assert!(dim <= u16::MAX as usize, "dim must fit in u16");
        let codes_per_byte = (8 / bits) as usize;
        assert_eq!(
            dim % codes_per_byte,
            0,
            "dim must be a multiple of {codes_per_byte} for bits = {bits}",
        );
        if bits == 8 {
            // b=8 full-capability requires dim % 256 == 0 (equal bucket
            // occupancy → exact symmetric analytical norm). Fail loud and
            // point at the asymmetric-only constructor so the caller has a
            // non-surprising path for non-aligned dims.
            assert_eq!(
                dim % 256,
                0,
                "RankQuant::new(dim, 8) requires dim % 256 == 0 for symmetric \
                 scoring (equal-bucket analytical norm); dim={dim} is not \
                 256-aligned. Use RankQuant::new_asymmetric(dim, 8) for an \
                 asymmetric-only b=8 index at any dim.",
            );
            return Self {
                dim,
                bits,
                n_vectors: 0,
                capability: RankQuantCapability::SymmetricAndAsymmetric,
                packed: Vec::new(),
            };
        }
        // Audit-safety: require dim divisible by 2^bits so every bucket
        // gets exactly dim / (1 << bits) rank entries per document. This
        // is what makes `rankquant_norm` analytically exact (every doc
        // has identical bucket histogram, identical L2 norm). Common
        // embedding dims (768, 1024, 1536, 3072) all satisfy this for
        // bits in {1, 2, 4}. Without this, the analytical norm becomes
        // approximate and we'd need to store a per-doc inv_norm.
        let n_buckets = 1usize << bits;
        assert_eq!(
            dim % n_buckets,
            0,
            "dim must be divisible by 2^bits = {n_buckets} so every \
             bucket receives exactly dim / 2^bits rank entries; this \
             keeps the analytical rankquant_norm exact per document",
        );
        Self {
            dim,
            bits,
            n_vectors: 0,
            capability: RankQuantCapability::SymmetricAndAsymmetric,
            packed: Vec::new(),
        }
    }

    /// Construct an asymmetric-capable index at **any** valid `dim`.
    ///
    /// This is the non-surprising entry point for `b = 8` at a dimension
    /// that is not `256`-aligned: it produces a
    /// [`RankQuantCapability::AsymmetricOnly`] instance whose
    /// code/projection generation, pair-evidence/contingency, and
    /// asymmetric (float-query) scoring all work, but whose symmetric path
    /// ([`Self::search`]) panics (the equal-bucket analytical norm is not
    /// exact off the `256`-aligned grid). When `dim % 256 == 0`, the `b=8`
    /// instance is upgraded to full [`RankQuantCapability::SymmetricAndAsymmetric`]
    /// (there is no reason to withhold symmetric scoring when it is exact).
    ///
    /// For `b ∈ {1, 2, 4}` this constructs the same full-capability instance
    /// as [`Self::new`] (those widths are always symmetric-capable when their
    /// constructor invariants hold), so it is never *less* capable than
    /// `new` — it is simply the width-agnostic constructor.
    ///
    /// # Panics
    /// Panics if `(dim, bits)` is not a valid **code** configuration —
    /// i.e. `bits ∉ {1, 2, 4, 8}`, `dim < 2`, `dim > u16::MAX`, or
    /// `dim % (8 / bits) != 0`. For `b ∈ {1, 2, 4}` it additionally requires
    /// `dim % 2^bits == 0` (same as [`Self::new`]).
    pub fn new_asymmetric(dim: usize, bits: u8) -> Self {
        // Reuse the code-validity gate (accepts any 256-unaligned dim for b=8,
        // still requires dim % 2^bits for b ∈ {1,2,4}). Convert the structured
        // error into a panic so this constructor matches `new`'s fail-loud style.
        Self::validate_params(dim, bits)
            .unwrap_or_else(|e| panic!("RankQuant::new_asymmetric invalid params: {e}"));
        let capability = Self::capability_for(dim, bits);
        Self {
            dim,
            bits,
            n_vectors: 0,
            capability,
            packed: Vec::new(),
        }
    }

    /// Compute the capability for a code-valid `(dim, bits)` pair.
    ///
    /// `b ∈ {1, 2, 4}` and `256`-aligned `b=8` are full-capability; any
    /// other (i.e. non-`256`-aligned) `b=8` is asymmetric-only.
    #[inline]
    fn capability_for(dim: usize, bits: u8) -> RankQuantCapability {
        if bits == 8 && !dim.is_multiple_of(256) {
            RankQuantCapability::AsymmetricOnly
        } else {
            RankQuantCapability::SymmetricAndAsymmetric
        }
    }

    /// The scoring modes this instance supports — see [`RankQuantCapability`].
    ///
    /// Always [`RankQuantCapability::SymmetricAndAsymmetric`] for
    /// `b ∈ {1, 2, 4}`. For `b=8` it reflects whether `dim % 256 == 0`.
    #[inline]
    pub fn capability(&self) -> RankQuantCapability {
        self.capability
    }

    /// Whether [`Self::search`] (symmetric scoring) is supported on this
    /// instance. `true` for `b ∈ {1, 2, 4}` and for `256`-aligned `b=8`;
    /// `false` for `b=8` at a non-`256`-aligned dim (asymmetric-only).
    ///
    /// Callers should check this before invoking [`Self::search`] on a
    /// `b=8` index built via [`Self::new_asymmetric`].
    #[inline]
    pub fn symmetric_supported(&self) -> bool {
        matches!(self.capability, RankQuantCapability::SymmetricAndAsymmetric)
    }

    /// Fail loud with the exact symmetric-gating message when symmetric
    /// scoring is invoked on an asymmetric-only (`b=8`, non-`256`-aligned)
    /// instance. No-op for symmetric-capable instances.
    #[inline]
    fn assert_symmetric_supported(&self) {
        assert!(
            self.symmetric_supported(),
            "RankQuant b=8 symmetric scoring requires dim % 256 == 0; dim={} supports asymmetric/evidence APIs only.",
            self.dim,
        );
    }

    /// Add documents. Each vector is rank-transformed, bucketed to `bits`
    /// bits/coord, and bit-packed row-major.
    ///
    /// # Panics
    /// Panics if the index would grow beyond `rank_io::MAX_VECTORS` documents
    /// — the supported capacity. Candidate APIs materialise document IDs as
    /// `u32`; `MAX_VECTORS` sits well below `u32::MAX` and matches the on-disk
    /// loader's `n_vectors` ceiling. (Bounds the count, not the byte payload —
    /// see the loaders' separate `MAX_PAYLOAD` cap.) Also panics if the
    /// resulting row-major buffer length would overflow `usize` (reachable only
    /// on 32-bit targets — see `util::checked_new_len`).
    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim",
        );
        assert_all_finite(vectors);
        let bytes_per_vec = rankquant_bytes_per_vec(self.dim, self.bits);
        let new_n = crate::util::checked_new_len(self.n_vectors, n, bytes_per_vec);
        let start = self.packed.len();
        self.packed.resize(start + n * bytes_per_vec, 0);
        let dim = self.dim;
        let bits = self.bits;
        self.packed[start..]
            .par_chunks_mut(bytes_per_vec)
            .zip(vectors.par_chunks(dim))
            .for_each(|(out, v)| {
                let ranks = rank_transform(v);
                let buckets = bucket_ranks(&ranks, bits);
                let packed = pack_buckets(&buckets, bits);
                out.copy_from_slice(&packed);
            });
        self.n_vectors = new_n;
    }

    /// Symmetric search: bucket the query and score against bucketed
    /// docs.
    ///
    /// # Panics
    /// For a `b=8` index built via [`Self::new_asymmetric`] at a
    /// non-`256`-aligned dim (an [`RankQuantCapability::AsymmetricOnly`]
    /// instance), this **panics**: the symmetric analytical norm requires
    /// equal bucket occupancy (`dim % 256 == 0`). Check
    /// [`Self::symmetric_supported`] first, or use [`Self::search_asymmetric`],
    /// which works at any dim. (`b ∈ {1, 2, 4}` and `256`-aligned `b=8`
    /// instances never trip this.) The panic message is:
    /// `RankQuant b=8 symmetric scoring requires dim % 256 == 0; dim={dim}
    /// supports asymmetric/evidence APIs only.`
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        // Symmetric gating: fail loud (with the exact message) for an
        // asymmetric-only b=8 instance before doing any work.
        self.assert_symmetric_supported();
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        assert_all_finite(queries);
        // Clamp the user's `k` to `n_vectors` before it sizes any
        // `vec![_; nq * k]` allocation below. An unclamped `usize::MAX`
        // otherwise aborts the process with `capacity overflow`.
        let k = k.min(self.n_vectors);
        let k_eff = k;
        let buf_len = result_buffer_len(nq, k);
        if k_eff == 0 {
            return SearchResults {
                scores: vec![0.0; buf_len],
                indices: vec![-1; buf_len],
                nq,
                k,
            };
        }
        let dim = self.dim;
        let bits = self.bits;
        let n = self.n_vectors;
        let norm = rankquant_norm(dim, bits);
        let inv_norm_sq = 1.0_f32 / (norm * norm);
        let bytes_per_vec = rankquant_bytes_per_vec(dim, bits);

        let mut scores_flat = vec![0.0f32; buf_len];
        let mut indices_flat = vec![-1i64; buf_len];

        let n_buckets = 1usize << bits;
        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                // Build the per-dim, per-bucket LUT for this query.
                // LUT[d * n_buckets + b] = q_centred[d] * bucket_centre(b).
                let q_ranks = rank_transform(q);
                let mut lut = vec![0.0f32; dim * n_buckets];
                for d in 0..dim {
                    let qb = rank_to_bucket(q_ranks[d], dim, bits);
                    let qc = bucket_centre(qb, bits);
                    for b in 0..n_buckets {
                        lut[d * n_buckets + b] = qc * bucket_centre(b as u8, bits);
                    }
                }
                let mut top = TopK::new(k_eff);
                match bits {
                    1 => scan_b1_to_topk(&self.packed, n, dim, &lut, inv_norm_sq, &mut top),
                    2 => scan_b2_to_topk(&self.packed, n, dim, &lut, inv_norm_sq, &mut top),
                    4 => scan_b4_to_topk(&self.packed, n, dim, &lut, inv_norm_sq, &mut top),
                    8 => scan_b8_to_topk(&self.packed, n, dim, &lut, inv_norm_sq, &mut top),
                    _ => unreachable!(),
                }
                top.finalize_into(out_scores, out_indices);
                let _ = bytes_per_vec; // shape clarity
            });

        SearchResults {
            scores: scores_flat,
            indices: indices_flat,
            nq,
            k,
        }
    }

    /// Asymmetric search: queries stay as raw L2-normalised floats,
    /// documents are B-bit bucket-packed.
    ///
    /// Inner kernel uses a per-query `dim * 2^bits` LUT
    /// (`LUT[d][b] = q_unit[d] * bucket_centre(b)`). The scan unpacks
    /// `8 / bits` codes per byte and accumulates via LUT lookups; the
    /// compiler autovectorises the inner sum.
    ///
    /// Works at **any** valid dim for all supported widths including `b=8`
    /// (the asymmetric path needs no equal-bucket precondition). For `b=8`
    /// the score is a per-coordinate gather `Σ_d lut[d*256 + code[d]]`
    /// against the `dim * 256` LUT: it dispatches to the AVX-512
    /// `vgatherdps` kernel (`scan_b8_asym` → `scan_b8_asym_avx512_gather`)
    /// when `avx512f` is present and `dim % 16 == 0`, else the portable
    /// scalar LUT reference (`scan_b8_to_topk`). Unlike [`Self::search`],
    /// this never panics on an asymmetric-only instance.
    pub fn search_asymmetric(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        assert_all_finite(queries);
        // Clamp `k` to `n_vectors` before sizing any `vec![_; nq * k]`
        // allocation; `usize::MAX` otherwise aborts with capacity
        // overflow.
        let k = k.min(self.n_vectors);
        let k_eff = k;
        let buf_len = result_buffer_len(nq, k);
        if k_eff == 0 {
            return SearchResults {
                scores: vec![0.0; buf_len],
                indices: vec![-1; buf_len],
                nq,
                k,
            };
        }
        let dim = self.dim;
        let bits = self.bits;
        let n = self.n_vectors;
        let norm = rankquant_norm(dim, bits);
        let inv_norm = 1.0_f32 / norm;
        let n_buckets = 1usize << bits;
        let bytes_per_vec = rankquant_bytes_per_vec(dim, bits);

        let mut scores_flat = vec![0.0f32; buf_len];
        let mut indices_flat = vec![-1i64; buf_len];

        // Asymmetric mode: prefer AVX-512 → AVX2 → scalar LUT.
        // Both SIMD paths use the centre-drop trick (raw codes in the
        // hot loop, per-query constant offset re-applied at finalize).
        //
        // CRITICAL: each SIMD kernel carries a *lane invariant* on `dim`
        // (AVX-512 processes 64 codes per outer iter → needs dim % 64;
        // AVX2 b=2 processes 16 codes/chunk → needs dim % 16; AVX2 b=4
        // processes 8 codes/chunk → needs dim % 8). The constructor only
        // guarantees `dim % (1 << bits) == 0` and `dim % (8 / bits) == 0`,
        // so constructor-valid dims like 48 / 80 / 20 can violate the
        // SIMD invariant. Each kernel enforces its lane invariant with a
        // real `assert!` (not a `debug_assert!`), so a mis-dispatch panics
        // loudly in release rather than silently dropping a chunk. The
        // dispatch below must therefore only select a
        // tier whose invariant holds for (dim, bits); otherwise it falls
        // back to the scalar LUT path which handles any valid dim.
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
        let simd_tier = select_simd_tier(dim, bits);

        // The SIMD paths drop the per-lane centre subtract from the hot
        // loop. The query-constant offset is applied inside TopK before
        // eviction, so boundary ties use the same exposed score tuple that
        // callers receive.
        #[cfg(target_arch = "x86_64")]
        let centre = ((1u32 << bits) as f32 - 1.0) / 2.0;

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                let q_unit = l2_normalise(q);
                let mut top = TopK::new(k_eff);

                // b=8 is a per-coordinate gather (`Σ_d lut[d*256 + code[d]]`),
                // not a centre-drop dot product — it routes to its own
                // dispatch (AVX-512 vgatherdps → scalar LUT) and never uses
                // the centre-drop offset (its LUT bakes the centre in).
                if bits == 8 {
                    scan_b8_asym(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                    top.finalize_into(out_scores, out_indices);
                    let _ = bytes_per_vec; // shape clarity
                    return;
                }

                #[cfg(target_arch = "x86_64")]
                let centre_offset = {
                    let q_sum: f32 = q_unit.iter().sum();
                    -centre * q_sum * inv_norm
                };

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    match (simd_tier, bits) {
                        (SimdTier::Avx512, 2) => {
                            top.set_score_offset(centre_offset);
                            scan_b2_asym_avx512(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                        }
                        (SimdTier::Avx512, 4) => {
                            top.set_score_offset(centre_offset);
                            scan_b4_asym_avx512(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                        }
                        (SimdTier::Avx2, 2) => {
                            top.set_score_offset(centre_offset);
                            scan_b2_asym_avx2(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                        }
                        (SimdTier::Avx2, 4) => {
                            top.set_score_offset(centre_offset);
                            scan_b4_asym_avx2(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                        }
                        _ => scan_via_lut_scalar(
                            &self.packed,
                            n,
                            dim,
                            bits,
                            n_buckets,
                            &q_unit,
                            inv_norm,
                            &mut top,
                        ),
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                scan_via_lut_scalar(
                    &self.packed,
                    n,
                    dim,
                    bits,
                    n_buckets,
                    &q_unit,
                    inv_norm,
                    &mut top,
                );

                top.finalize_into(out_scores, out_indices);

                let _ = bytes_per_vec; // shape clarity
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
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn bytes_per_vec(&self) -> usize {
        rankquant_bytes_per_vec(self.dim, self.bits)
    }
    /// Total bytes held by the packed buffer (excludes Vec overhead).
    pub fn byte_size(&self) -> usize {
        self.packed.len()
    }

    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(idx < self.n_vectors, "index out of bounds");
        let last = self.n_vectors - 1;
        let bpv = self.bytes_per_vec();
        if idx != last {
            let src = last * bpv;
            let dst = idx * bpv;
            self.packed.copy_within(src..src + bpv, dst);
        }
        self.packed.truncate(last * bpv);
        self.n_vectors -= 1;
        last
    }

    /// Persist to a `.tvrq` file. Format: 14-byte header + packed bytes.
    ///
    /// # `b=8`
    /// The `.tvrq` on-disk format and its loader currently support only
    /// `bits ∈ {1, 2, 4}`. `b=8` is an in-memory evidence/refinement surface
    /// in this phase; persisting it is a follow-up. To avoid writing a file
    /// that [`Self::load`] would then reject (a silent broken round-trip),
    /// this returns `io::Error` (kind `Unsupported`) for a `b=8` index rather
    /// than emitting an unloadable file.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        if self.bits == 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "RankQuant b=8 persistence is not supported yet (the .tvrq loader \
                 accepts bits ∈ {1, 2, 4}); b=8 is an in-memory evidence surface \
                 in this phase",
            ));
        }
        crate::rank_io::write_rankquant(path, self.bits, self.dim, self.n_vectors, &self.packed)
    }

    /// Load from a `.tvrq` file produced by [`Self::write`].
    ///
    /// Re-runs the same constructor invariants `RankQuant::new`
    /// enforces (`bits ∈ {1, 2, 4}`, `dim % (1 << bits) == 0`,
    /// `dim % (8 / bits) == 0`). Returns `io::Error::InvalidData` on
    /// any violation — never panics on malformed input.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (bits, dim, n_vectors, packed) = crate::rank_io::load_rankquant(path)?;
        // load_rankquant already validates bits ∈ {1,2,4} and bounds
        // dim/n_vectors; we replay the per-type invariants here.
        let n_buckets = 1usize << bits;
        if dim % n_buckets != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TVRQ dim {dim} is not a multiple of 2^bits = {n_buckets}; \
                     constant-composition invariant violated"
                ),
            ));
        }
        let codes_per_byte = (8 / bits) as usize;
        if dim % codes_per_byte != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TVRQ dim {dim} is not a multiple of codes_per_byte = {codes_per_byte}",),
            ));
        }
        // `checked_mul` (not `saturating`): on a 32-bit target the byte count
        // `n_vectors * dim * bits / 8` can overflow `usize`; treat overflow as
        // malformed rather than letting a saturated `usize::MAX` pass as a
        // plausible length. Two steps with distinct messages so a report names
        // which product wrapped (`n_vectors * dim` vs the subsequent `* bits`).
        let nv_dim = n_vectors.checked_mul(dim).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "TVRQ n_vectors * dim overflows usize",
            )
        })?;
        let expected_bytes = nv_dim
            .checked_mul(bits as usize)
            .map(|x| x / 8)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "TVRQ (n_vectors * dim) * bits overflows usize",
                )
            })?;
        if packed.len() != expected_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TVRQ payload length {} does not match expected {expected_bytes}",
                    packed.len(),
                ),
            ));
        }
        // `load_rankquant` only admits bits ∈ {1,2,4} (b=8 is not persistable
        // in this phase — see `write`), and those widths are always
        // full-capability, so the loaded instance is SymmetricAndAsymmetric.
        // `capability_for` keeps that derivation in one place.
        let capability = Self::capability_for(dim, bits);
        Ok(Self {
            dim,
            bits,
            n_vectors,
            capability,
            packed,
        })
    }

    /// Single-query asymmetric scoring restricted to a candidate
    /// subset (e.g., the top-M from a bitmap probe). Returns
    /// `(scores, indices)`: the top-`k` scores and their corresponding
    /// **global** doc IDs (the local candidate positions are mapped back
    /// to global IDs before returning). Results are ordered by score
    /// descending, then global row ID ascending, matching the full-index
    /// search tie policy even when `candidates` is unsorted.
    /// `candidates` may contain duplicate global row IDs. Each candidate entry
    /// is scored independently, so duplicate IDs may produce duplicate returned
    /// global IDs. Callers that require unique hits should deduplicate the
    /// candidate list before calling this method. The candidate list length is
    /// still bounded by `n_vectors`; this keeps duplicate-heavy inputs from
    /// allocating more scratch space than a full-corpus scan.
    ///
    /// Uses the same AVX-512 → AVX2 → scalar dispatch as
    /// [`Self::search_asymmetric`] and the same centre-drop math, just
    /// iterates over the provided candidate list instead of all `n`
    /// documents.
    ///
    /// The candidate docs are gathered into a contiguous scratch buffer
    /// (`m * bytes_per_vec`) before the SIMD scan — negligible for the
    /// intended small/medium candidate sets (`M` ≈ 100–500 from a bitmap
    /// probe), but the copy grows linearly in `M`. For very large `M`
    /// (e.g. misuse via FFI), a full [`Self::search_asymmetric`] may be
    /// cheaper; a gather-free in-place scan is tracked for the FFI work.
    ///
    /// If the candidate list came from [`crate::Bitmap`], this method reranks
    /// that shortlist exactly under RankQuant; it does not itself carry the
    /// bitmap threshold-calibration guarantee.
    pub fn search_asymmetric_subset(
        &self,
        query: &[f32],
        candidates: &[u32],
        k: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        assert_eq!(query.len(), self.dim);
        assert_all_finite(query);
        assert!(
            candidates.len() <= self.n_vectors,
            "search_asymmetric_subset: candidate list length {} exceeds n_vectors {}; deduplicate repeated ids before calling",
            candidates.len(),
            self.n_vectors,
        );
        // Bounds-check candidate ids before the gather below indexes
        // `self.packed[src..src + bpv]` with `src = di * bpv`. An OOB id
        // otherwise surfaces as a cryptic slice-range panic; fail fast
        // with a clear message instead. (The Python FFI validates ids
        // separately, so this assert is the Rust-side backstop.)
        assert!(
            candidates.iter().all(|&di| (di as usize) < self.n_vectors),
            "search_asymmetric_subset: candidate id out of range (n_vectors {})",
            self.n_vectors,
        );
        let dim = self.dim;
        let bits = self.bits;
        let bpv = self.bytes_per_vec();
        let n_buckets = 1usize << bits;
        let m = candidates.len();
        let k_eff = k.min(m);
        if k_eff == 0 {
            return (Vec::new(), Vec::new());
        }

        let norm = rankquant_norm(dim, bits);
        let inv_norm = 1.0_f32 / norm;
        #[cfg(target_arch = "x86_64")]
        let centre = ((1u32 << bits) as f32 - 1.0) / 2.0;

        // L2-normalise the query.
        let q_unit = l2_normalise(query);
        #[cfg(target_arch = "x86_64")]
        let centre_offset = {
            let q_sum: f32 = q_unit.iter().sum();
            -centre * q_sum * inv_norm
        };

        // Pack the candidate docs' bytes into a contiguous buffer so
        // the SIMD kernels can scan them as if they were a small dense
        // sub-index. Cost: m * bpv copy (small for typical m).
        let sub_packed_len = m
            .checked_mul(bpv)
            .expect("search_asymmetric_subset: candidate scratch length overflows usize");
        let mut sub_packed = vec![0u8; sub_packed_len];
        for (i, &di) in candidates.iter().enumerate() {
            let src = (di as usize) * bpv;
            sub_packed[i * bpv..(i + 1) * bpv].copy_from_slice(&self.packed[src..src + bpv]);
        }

        // Dispatch: prefer AVX-512 → AVX2 → scalar LUT. Tier selection
        // is gated on the kernel lane invariant for (dim, bits) via
        // `select_simd_tier` — the same guard `search_asymmetric` uses —
        // so a constructor-valid-but-SIMD-invalid dim (48 / 80 / 20)
        // never reaches a kernel that would drop its tail chunk.
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
        let simd_tier = select_simd_tier(dim, bits);
        let mut top = TopK::new_with_tie_keys(k_eff, candidates);
        // b=8 routes to its own gather dispatch (AVX-512 vgatherdps → scalar
        // LUT), with the centre baked into the LUT (no score-offset trick).
        // The tie keys on `top` still map local scratch positions → global
        // row IDs exactly as for b ∈ {1,2,4}.
        if bits == 8 {
            scan_b8_asym(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
        } else {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                match (simd_tier, bits) {
                    (SimdTier::Avx512, 2) => {
                        top.set_score_offset(centre_offset);
                        scan_b2_asym_avx512(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    }
                    (SimdTier::Avx512, 4) => {
                        top.set_score_offset(centre_offset);
                        scan_b4_asym_avx512(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    }
                    (SimdTier::Avx2, 2) => {
                        top.set_score_offset(centre_offset);
                        scan_b2_asym_avx2(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    }
                    (SimdTier::Avx2, 4) => {
                        top.set_score_offset(centre_offset);
                        scan_b4_asym_avx2(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    }
                    _ => scan_via_lut_scalar(
                        &sub_packed,
                        m,
                        dim,
                        bits,
                        n_buckets,
                        &q_unit,
                        inv_norm,
                        &mut top,
                    ),
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            scan_via_lut_scalar(
                &sub_packed,
                m,
                dim,
                bits,
                n_buckets,
                &q_unit,
                inv_norm,
                &mut top,
            );
        }

        let mut scores = vec![f32::NEG_INFINITY; k_eff];
        let mut local_indices = vec![-1i64; k_eff];
        top.finalize_into(&mut scores, &mut local_indices);
        // Map local → global doc IDs.
        let global_indices: Vec<i64> = local_indices
            .iter()
            .map(|&loc| {
                if loc < 0 {
                    -1
                } else {
                    candidates[loc as usize] as i64
                }
            })
            .collect();
        (scores, global_indices)
    }

    pub fn try_search_with_sign_probe(
        &self,
        sign_probe: &SignBitmap,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<f32>, Vec<i64>), OrdvecError> {
        self.try_search_with_sign_probe_with_policy(
            sign_probe,
            query,
            k,
            TwoStageCandidatePolicy::default(),
        )
    }

    pub fn try_search_with_sign_probe_with_policy(
        &self,
        sign_probe: &SignBitmap,
        query: &[f32],
        k: usize,
        policy: TwoStageCandidatePolicy,
    ) -> Result<(Vec<f32>, Vec<i64>), OrdvecError> {
        if sign_probe.dim() != self.dim {
            return Err(OrdvecError::InvalidParameter {
                name: "sign_probe.dim",
                message: format!("must match RankQuant dim {}", self.dim),
            });
        }
        if sign_probe.len() != self.n_vectors {
            return Err(OrdvecError::InvalidParameter {
                name: "sign_probe.len",
                message: format!("must match RankQuant len {}", self.n_vectors),
            });
        }
        if query.len() != self.dim {
            return Err(OrdvecError::InvalidVectorLength {
                name: "query",
                len: query.len(),
                expected: self.dim,
            });
        }
        validate_finite(query, "query")?;
        let candidate_count = policy.candidate_count(k, self.n_vectors);
        let candidates = sign_probe.top_m_candidates(query, candidate_count);
        validate_candidate_ids(&candidates, self.n_vectors)?;
        Ok(self.search_asymmetric_subset(query, &candidates, k))
    }

    pub fn search_with_sign_probe(
        &self,
        sign_probe: &SignBitmap,
        query: &[f32],
        k: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        self.try_search_with_sign_probe(sign_probe, query, k)
            .expect("search_with_sign_probe validation failed")
    }
}

fn validate_finite(values: &[f32], name: &'static str) -> Result<(), OrdvecError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OrdvecError::InvalidParameter {
            name,
            message: "must contain only finite values".to_string(),
        });
    }
    Ok(())
}

/// Standalone symmetric RankQuant-style eval search for arbitrary bit widths.
///
/// This does **not** use [`RankQuant`] storage and does not change the `.tvrq`
/// packing contract. It rank-transforms `corpus` and `queries`, buckets each
/// rank into `1 << bits` equal-width bins, mean-centres bucket ids, normalises
/// by the analytical norm for that `(dim, bits)`, and returns top-`k` results.
///
/// Intended for research/eval sweeps where non-byte-aligned widths such as
/// `bits = 3` need to be scored without inventing a persistent packed format.
pub fn rankquant_eval_search(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    bits: u8,
    k: usize,
) -> SearchResults {
    check_eval_bits(bits);
    assert!(dim >= 2, "dim must be >= 2");
    assert!(dim <= u16::MAX as usize, "dim must fit in u16");
    let n = corpus.len() / dim;
    let nq = queries.len() / dim;
    assert_eq!(
        corpus.len(),
        n * dim,
        "corpus length must be a multiple of dim"
    );
    assert_eq!(
        queries.len(),
        nq * dim,
        "queries length must be a multiple of dim"
    );
    assert_all_finite(corpus);
    assert_all_finite(queries);

    let k = k.min(n);
    let k_eff = k;
    let buf_len = result_buffer_len(nq, k);
    if nq == 0 || k_eff == 0 {
        return SearchResults {
            scores: vec![0.0; buf_len],
            indices: vec![-1; buf_len],
            nq,
            k,
        };
    }

    let norm = rankquant_eval_norm(dim, bits);
    let inv_norm_sq = 1.0_f32 / (norm * norm);
    let centres: Vec<f32> = (0..(1usize << bits))
        .map(|bucket| bucket_centre(bucket as u8, bits))
        .collect();
    let mut doc_buckets = vec![0u8; n * dim];
    doc_buckets
        .par_chunks_mut(dim)
        .zip(corpus.par_chunks(dim))
        .for_each(|(out, doc)| rankquant_eval_buckets(doc, bits, out));

    let mut scores_flat = vec![0.0f32; buf_len];
    let mut indices_flat = vec![-1i64; buf_len];
    queries
        .par_chunks(dim)
        .zip(scores_flat.par_chunks_mut(k))
        .zip(indices_flat.par_chunks_mut(k))
        .for_each(|((q, out_scores), out_indices)| {
            let mut q_centres = vec![0.0f32; dim];
            rankquant_eval_centres(q, bits, &mut q_centres);
            let mut top = TopK::new(k_eff);
            for (di, doc) in doc_buckets.chunks_exact(dim).enumerate() {
                let acc: f32 = q_centres
                    .iter()
                    .zip(doc)
                    .map(|(q, &bucket)| q * centres[bucket as usize])
                    .sum();
                top.maybe_insert(acc * inv_norm_sq, di);
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

// -------------------------------------------------------------------
// Byte-LUT scoring (asymmetric, B = 2 and B = 4).
//
// Precomputes lut[g][byte] = sum of all per-coordinate contributions
// the byte at position g represents. Inner loop becomes one lookup
// and one add per doc byte: trades arithmetic for memory.
//
// LUT size at D=1024:
//   B=2: 256 groups × 256 entries × 4 B = 256 KiB per query (fits L2)
//   B=4: 512 groups × 256 entries × 4 B = 512 KiB per query (spills L2 a little)
//
// Re-exported `#[doc(hidden)]` for benchmarking. Production callers should reach
// for [`RankQuant::search_asymmetric`] which dispatches to the
// fastest implementation for the current CPU.
// -------------------------------------------------------------------

/// Build the byte-LUT for B=2 asymmetric: `lut[g * 256 + byte]` is the
/// f32 contribution of `doc[g] == byte` to the score, summed across
/// the 4 coordinates packed into that byte.
fn build_byte_lut_b2(q_unit: &[f32]) -> Vec<f32> {
    let dim = q_unit.len();
    debug_assert_eq!(dim % 4, 0);
    let n_groups = dim / 4;
    let mut lut = vec![0.0f32; n_groups * 256];
    for g in 0..n_groups {
        let q0 = q_unit[g * 4];
        let q1 = q_unit[g * 4 + 1];
        let q2 = q_unit[g * 4 + 2];
        let q3 = q_unit[g * 4 + 3];
        for byte in 0u32..256 {
            let c0 = ((byte >> 6) & 3) as f32 - 1.5;
            let c1 = ((byte >> 4) & 3) as f32 - 1.5;
            let c2 = ((byte >> 2) & 3) as f32 - 1.5;
            let c3 = (byte & 3) as f32 - 1.5;
            lut[g * 256 + byte as usize] = q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3;
        }
    }
    lut
}

/// Build the byte-LUT for B=4 asymmetric.
fn build_byte_lut_b4(q_unit: &[f32]) -> Vec<f32> {
    let dim = q_unit.len();
    debug_assert_eq!(dim % 2, 0);
    let n_groups = dim / 2;
    let mut lut = vec![0.0f32; n_groups * 256];
    for g in 0..n_groups {
        let q0 = q_unit[g * 2];
        let q1 = q_unit[g * 2 + 1];
        for byte in 0u32..256 {
            let hi = ((byte >> 4) & 0xF) as f32 - 7.5;
            let lo = (byte & 0xF) as f32 - 7.5;
            lut[g * 256 + byte as usize] = q0 * hi + q1 * lo;
        }
    }
    lut
}

/// Scalar byte-LUT scan for B=2 asymmetric. One add per doc byte.
fn scan_b2_asym_byte_lut(
    packed: &[u8],
    n: usize,
    dim: usize,
    q_unit: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let bytes_per_vec = dim / 4;
    let lut = build_byte_lut_b2(q_unit);
    for di in 0..n {
        let doc = &packed[di * bytes_per_vec..(di + 1) * bytes_per_vec];
        let mut acc = 0.0f32;
        for (g, &byte) in doc.iter().enumerate() {
            acc += lut[g * 256 + byte as usize];
        }
        top.maybe_insert(acc * scale, di);
    }
}

/// Scalar byte-LUT scan for B=4 asymmetric.
fn scan_b4_asym_byte_lut(
    packed: &[u8],
    n: usize,
    dim: usize,
    q_unit: &[f32],
    scale: f32,
    top: &mut TopK,
) {
    let bytes_per_vec = dim / 2;
    let lut = build_byte_lut_b4(q_unit);
    for di in 0..n {
        let doc = &packed[di * bytes_per_vec..(di + 1) * bytes_per_vec];
        let mut acc = 0.0f32;
        for (g, &byte) in doc.iter().enumerate() {
            acc += lut[g * 256 + byte as usize];
        }
        top.maybe_insert(acc * scale, di);
    }
}

/// Bench-only entrypoint for the byte-LUT path. Not used by
/// [`RankQuant::search_asymmetric`] in production (which prefers
/// the AVX2 inline-expand kernel where available). Exposed so the
/// example bench can compare the two empirically on the same data.
///
/// **Bit-width restriction:** the byte-LUT precomputes per-byte
/// contributions for the 4-codes-per-byte (b=2) and 2-codes-per-byte
/// (b=4) packings only. It does **not** support b=1 and will panic on
/// a b=1 index. This is acceptable because it is a benchmarking helper:
/// production callers reach for [`RankQuant::search_asymmetric`],
/// whose dispatch routes b=1 to the scalar LUT path (the SIMD/byte-LUT
/// kernels are only selected for b ∈ {2, 4}). Pass a b ∈ {2, 4} index.
///
/// Returns the raw `Vec<i64>` of doc indices per query, length
/// `queries.len() / dim * k`.
pub fn search_asymmetric_byte_lut(index: &RankQuant, queries: &[f32], k: usize) -> SearchResults {
    let dim = index.dim;
    let bits = index.bits;
    let n = index.n_vectors;
    let nq = queries.len() / dim;
    assert_eq!(queries.len(), nq * dim);
    assert_all_finite(queries);
    // Shadow `k` with the clamp so the clamped value flows into the
    // buffer sizing *and* the `par_chunks_mut(k)` row stride — matching
    // the other search methods. Previously only `k_eff` was clamped
    // while the allocations and chunking used the raw `k`, so a huge
    // `k` (e.g. `usize::MAX`) sized `nq * k` and aborted with capacity
    // overflow. The `result_buffer_len` guard below additionally
    // catches `nq * k` overflowing usize for a large query count.
    let k = k.min(n);
    let k_eff = k;
    let buf_len = result_buffer_len(nq, k);
    if k_eff == 0 {
        // Empty corpus (or k==0): `par_chunks_mut(0)` would panic, and
        // there is nothing to score. Return a correctly-shaped result
        // with `k == 0`, matching the other search methods' early-out.
        return SearchResults {
            scores: vec![0.0; buf_len],
            indices: vec![-1; buf_len],
            nq,
            k,
        };
    }
    let norm = rankquant_norm(dim, bits);
    let inv_norm = 1.0_f32 / norm;
    let mut scores_flat = vec![0.0f32; buf_len];
    let mut indices_flat = vec![-1i64; buf_len];
    queries
        .par_chunks(dim)
        .zip(scores_flat.par_chunks_mut(k))
        .zip(indices_flat.par_chunks_mut(k))
        .for_each(|((q, out_scores), out_indices)| {
            let q_unit = l2_normalise(q);
            let mut top = TopK::new(k_eff);
            match bits {
                2 => scan_b2_asym_byte_lut(&index.packed, n, dim, &q_unit, inv_norm, &mut top),
                4 => scan_b4_asym_byte_lut(&index.packed, n, dim, &q_unit, inv_norm, &mut top),
                _ => panic!("byte-LUT path only supports bits in {{2, 4}}"),
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
