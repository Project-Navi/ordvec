//! `B`-bit bucketed-rank index ([`RankQuantIndex`]).
//!
//! Storage is `dim * bits / 8` bytes per document at `bits ∈ {1, 2, 4}`.
//! Symmetric search uses a per-query, per-coord LUT; asymmetric search
//! dispatches AVX-512 → AVX2 → scalar via the kernels in
//! [`super::quant_kernels`].
//!
//! The byte-LUT path ([`search_asymmetric_byte_lut`]) is exposed
//! publicly via `crate::rank_index::search_asymmetric_byte_lut` so
//! `examples/bench_rank.rs` can compare it against the production
//! AVX path on the same data.

use rayon::prelude::*;

use super::quant_kernels::{
    scan_b1_to_topk, scan_b2_to_topk, scan_b4_to_topk, scan_via_lut_scalar,
};
#[cfg(target_arch = "x86_64")]
use super::quant_kernels::{
    scan_b2_asym_avx2, scan_b2_asym_avx512, scan_b4_asym_avx2, scan_b4_asym_avx512,
};
use super::util::{l2_normalise, TopK};
use crate::rank::{
    bucket_centre, bucket_ranks, pack_buckets, rank_to_bucket, rank_transform,
    rankquant_bytes_per_vec, rankquant_norm,
};
use crate::SearchResults;

/// `B`-bit RankQuant index.
///
/// Each document is encoded by bucketing its rank vector into
/// `1 << bits` equal-width bins on `[0, dim)` and packing `bits` bits
/// per coordinate. Storage is `dim * bits / 8` bytes per document.
/// Supported bit widths are `1`, `2`, and `4` (3-bit packing is left
/// for a follow-up; use `2` or `4` in the interim).
///
/// The mean-centred bucket vector has fixed analytical L2 norm
/// `sqrt(dim * (2^(2B) - 1) / 12)` when `dim % (1 << bits) == 0`, so
/// no per-document norms are stored.
pub struct RankQuantIndex {
    pub(super) dim: usize,
    pub(super) bits: u8,
    pub(super) n_vectors: usize,
    /// Row-major packed bucket bytes. `n_vectors * dim * bits / 8` total.
    pub(super) packed: Vec<u8>,
}

impl RankQuantIndex {
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(matches!(bits, 1 | 2 | 4), "bits must be 1, 2, or 4");
        assert!(dim >= 2, "dim must be >= 2");
        assert!(dim <= u16::MAX as usize, "dim must fit in u16");
        let codes_per_byte = (8 / bits) as usize;
        assert_eq!(
            dim % codes_per_byte,
            0,
            "dim must be a multiple of {codes_per_byte} for bits = {bits}",
        );
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
            packed: Vec::new(),
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim",
        );
        let bytes_per_vec = rankquant_bytes_per_vec(self.dim, self.bits);
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
        self.n_vectors += n;
    }

    /// Symmetric search: bucket the query and score against bucketed
    /// docs.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        let k_eff = k.min(self.n_vectors);
        if k_eff == 0 {
            return SearchResults {
                scores: vec![0.0; nq * k],
                indices: vec![-1; nq * k],
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

        let mut scores_flat = vec![0.0f32; nq * k];
        let mut indices_flat = vec![-1i64; nq * k];

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
    pub fn search_asymmetric(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        let k_eff = k.min(self.n_vectors);
        if k_eff == 0 {
            return SearchResults {
                scores: vec![0.0; nq * k],
                indices: vec![-1; nq * k],
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

        let mut scores_flat = vec![0.0f32; nq * k];
        let mut indices_flat = vec![-1i64; nq * k];

        // Asymmetric mode: prefer AVX-512 → AVX2 → scalar LUT.
        // Both SIMD paths use the centre-drop trick (raw codes in the
        // hot loop, per-query constant offset re-applied at finalize).
        #[derive(Copy, Clone, PartialEq)]
        enum SimdTier {
            None,
            Avx2,
            Avx512,
        }
        #[cfg(target_arch = "x86_64")]
        let simd_tier = if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512dq")
        {
            SimdTier::Avx512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            SimdTier::Avx2
        } else {
            SimdTier::None
        };
        #[cfg(not(target_arch = "x86_64"))]
        let simd_tier = SimdTier::None;

        // For the AVX2 path we drop the per-lane centre subtract from
        // the hot loop and add it back as a per-query constant offset
        // to the top-k scores at finalize time. Ranking is invariant
        // to this constant; absolute scores stay exact.
        let centre = ((1u32 << bits) as f32 - 1.0) / 2.0;

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                let q_unit = l2_normalise(q);
                let mut top = TopK::new(k_eff);
                let mut centre_drop_used = false;

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    match (simd_tier, bits) {
                        (SimdTier::Avx512, 2) => {
                            scan_b2_asym_avx512(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                            centre_drop_used = true;
                        }
                        (SimdTier::Avx512, 4) => {
                            scan_b4_asym_avx512(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                            centre_drop_used = true;
                        }
                        (SimdTier::Avx2, 2) => {
                            scan_b2_asym_avx2(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                            centre_drop_used = true;
                        }
                        (SimdTier::Avx2, 4) => {
                            scan_b4_asym_avx2(&self.packed, n, dim, &q_unit, inv_norm, &mut top);
                            centre_drop_used = true;
                        }
                        _ => scan_via_lut_scalar(
                            &self.packed, n, dim, bits, n_buckets, &q_unit, inv_norm, &mut top,
                        ),
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                scan_via_lut_scalar(
                    &self.packed, n, dim, bits, n_buckets, &q_unit, inv_norm, &mut top,
                );

                top.finalize_into(out_scores, out_indices);

                if centre_drop_used {
                    let q_sum: f32 = q_unit.iter().sum();
                    let offset = -centre * q_sum * inv_norm;
                    for s in out_scores.iter_mut() {
                        if s.is_finite() {
                            *s += offset;
                        }
                    }
                }

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

    /// Single-query asymmetric scoring restricted to a candidate
    /// subset (e.g., the top-M from a bitmap probe). Returns the
    /// top-`k` *candidate* indices (i.e., positions in `candidates`,
    /// not global doc IDs) and their scores. Caller is expected to
    /// map back to global IDs.
    ///
    /// Uses the same AVX-512 → AVX2 → scalar dispatch as
    /// [`Self::search_asymmetric`] and the same centre-drop math, just
    /// iterates over the provided candidate list instead of all `n`
    /// documents. Allocates nothing per-doc.
    /// Persist to a `.tvrq` file. Format: 14-byte header + packed bytes.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_rankquant(
            path, self.bits, self.dim, self.n_vectors, &self.packed,
        )
    }

    /// Load from a `.tvrq` file produced by [`Self::write`].
    ///
    /// Re-runs the same constructor invariants `RankQuantIndex::new`
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
                format!(
                    "TVRQ dim {dim} is not a multiple of codes_per_byte = {codes_per_byte}",
                ),
            ));
        }
        let expected_bytes = n_vectors.saturating_mul(dim).saturating_mul(bits as usize) / 8;
        if packed.len() != expected_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TVRQ payload length {} does not match expected {expected_bytes}",
                    packed.len(),
                ),
            ));
        }
        Ok(Self {
            dim,
            bits,
            n_vectors,
            packed,
        })
    }

    pub fn search_asymmetric_subset(
        &self,
        query: &[f32],
        candidates: &[u32],
        k: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        assert_eq!(query.len(), self.dim);
        let dim = self.dim;
        let bits = self.bits;
        let bpv = self.bytes_per_vec();
        let n_buckets = 1usize << bits;
        let m = candidates.len();
        let k_eff = k.min(m);

        let norm = rankquant_norm(dim, bits);
        let inv_norm = 1.0_f32 / norm;
        let centre = ((1u32 << bits) as f32 - 1.0) / 2.0;

        // L2-normalise the query and gather centre-correction.
        let q_unit = l2_normalise(query);
        let q_sum: f32 = q_unit.iter().sum();
        let centre_offset = -centre * q_sum * inv_norm;

        // Pack the candidate docs' bytes into a contiguous buffer so
        // the SIMD kernels can scan them as if they were a small dense
        // sub-index. Cost: m * bpv copy (small for typical m).
        let mut sub_packed = vec![0u8; m * bpv];
        for (i, &di) in candidates.iter().enumerate() {
            let src = (di as usize) * bpv;
            sub_packed[i * bpv..(i + 1) * bpv]
                .copy_from_slice(&self.packed[src..src + bpv]);
        }

        // Dispatch: prefer AVX-512 → AVX2 → scalar LUT.
        let mut top = TopK::new(k_eff);
        let mut centre_drop_used = false;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let use_avx512 = is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512dq");
            let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            match (use_avx512, use_avx2, bits) {
                (true, _, 2) => {
                    scan_b2_asym_avx512(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    centre_drop_used = true;
                }
                (true, _, 4) => {
                    scan_b4_asym_avx512(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    centre_drop_used = true;
                }
                (false, true, 2) => {
                    scan_b2_asym_avx2(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    centre_drop_used = true;
                }
                (false, true, 4) => {
                    scan_b4_asym_avx2(&sub_packed, m, dim, &q_unit, inv_norm, &mut top);
                    centre_drop_used = true;
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
            &sub_packed, m, dim, bits, n_buckets, &q_unit, inv_norm, &mut top,
        );

        let mut scores = vec![f32::NEG_INFINITY; k_eff];
        let mut local_indices = vec![-1i64; k_eff];
        top.finalize_into(&mut scores, &mut local_indices);
        if centre_drop_used {
            for s in scores.iter_mut() {
                if s.is_finite() {
                    *s += centre_offset;
                }
            }
        }
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
// Exposed publicly for benchmarking. Production callers should reach
// for [`RankQuantIndex::search_asymmetric`] which dispatches to the
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
/// [`RankQuantIndex::search_asymmetric`] in production (which prefers
/// the AVX2 inline-expand kernel where available). Exposed so the
/// example bench can compare the two empirically on the same data.
///
/// Returns the raw `Vec<i64>` of doc indices per query, length
/// `queries.len() / dim * k`.
pub fn search_asymmetric_byte_lut(
    index: &RankQuantIndex,
    queries: &[f32],
    k: usize,
) -> SearchResults {
    let dim = index.dim;
    let bits = index.bits;
    let n = index.n_vectors;
    let nq = queries.len() / dim;
    assert_eq!(queries.len(), nq * dim);
    let k_eff = k.min(n);
    let norm = rankquant_norm(dim, bits);
    let inv_norm = 1.0_f32 / norm;
    let mut scores_flat = vec![0.0f32; nq * k];
    let mut indices_flat = vec![-1i64; nq * k];
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
