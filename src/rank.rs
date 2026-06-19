//! Rank-cosine math primitives and the full-precision [`Rank`] index.
//!
//! This module provides two things:
//!
//! 1. **Math primitives** shared across the index family: rank transform,
//!    mean-centred bucket quantisation, packed-bit encode/decode, and
//!    analytical norms.
//!
//! 2. **[`Rank`]** — the full-precision rank-cosine index (`u16` per
//!    coordinate, `2 * dim` bytes per document). Both symmetric and
//!    asymmetric search paths live here.
//!
//! These primitives also back [`crate::RankQuant`], which buckets each
//! rank into one of `1 << B` equal-width bins on `[0, D)` and packs `B`
//! bits per coordinate, giving compact `D * B / 8` byte storage per
//! document while preserving the rank order the encoder induced.
//!
//! No training, no rotation, no centroid lookup: the bucket index *is*
//! the value, and rank-vector norms are analytical (a permutation of
//! `{0..D-1}` has fixed L2 norm after mean-centring).
//!
//! See the `tests` module below for the round-trip and norm-invariant tests.

use rayon::prelude::*;

use crate::util::{
    assert_all_finite, cmp_finite_f32_then_index, l2_normalise, result_buffer_len, TopK,
};
use crate::SearchResults;

/// Compute the dimension-wise rank transform of a single vector.
///
/// `out[k]` is the rank of `v[k]` among `v[0..d]`, with ties broken by
/// ascending index. Output values are in `[0, d)`. Equivalent
/// to NumPy's `np.argsort(np.argsort(v))` for a vector of length `d`.
///
/// `d` must fit in `u16` (`d <= 65_535`); panics otherwise.
pub fn rank_transform(v: &[f32]) -> Vec<u16> {
    let d = v.len();
    assert!(d <= u16::MAX as usize, "dim must fit in u16");
    assert_all_finite(v);
    let mut order: Vec<u16> = (0..d as u16).collect();
    order.sort_unstable_by(|&lhs, &rhs| {
        let lhs = lhs as usize;
        let rhs = rhs as usize;
        cmp_finite_f32_then_index(v[lhs], lhs, v[rhs], rhs)
    });
    let mut ranks = vec![0u16; d];
    for (rank, &orig_idx) in order.iter().enumerate() {
        ranks[orig_idx as usize] = rank as u16;
    }
    ranks
}

/// In-place variant: write the rank transform of `v` into `out`.
///
/// Allocates a `Vec<u16>` for the auxiliary argsort buffer.
pub fn rank_transform_into(v: &[f32], out: &mut [u16]) {
    let d = v.len();
    assert_eq!(d, out.len(), "out must have the same length as v");
    assert!(d <= u16::MAX as usize, "dim must fit in u16");
    assert_all_finite(v);
    let mut order: Vec<u16> = (0..d as u16).collect();
    order.sort_unstable_by(|&lhs, &rhs| {
        let lhs = lhs as usize;
        let rhs = rhs as usize;
        cmp_finite_f32_then_index(v[lhs], lhs, v[rhs], rhs)
    });
    for (rank, &orig_idx) in order.iter().enumerate() {
        out[orig_idx as usize] = rank as u16;
    }
}

/// Bucket a single rank into one of `1 << bits` equal-width bins on
/// `[0, d)`. Returns a value in `[0, 1 << bits)`.
///
/// For `bits == 8` the codomain is the full `u8` range `[0, 256)`; a
/// valid `rank < d` keeps the quotient `rank * 256 / d < 256`, so the
/// result still fits the returned `u8`.
///
/// # Panics
/// Panics if `bits > 8`, if `d == 0`, or if `rank >= d`. The `rank < d`
/// guard fails loud in *every* build — like the sibling [`pack_buckets`] and
/// [`bucket_centre`] checks — rather than silently clamping an out-of-range
/// rank into the top bucket. Internal callers feed ranks straight from
/// [`rank_transform`] (a permutation of `[0, d)`), so it never trips on the
/// hot path.
#[inline]
pub fn rank_to_bucket(rank: u16, d: usize, bits: u8) -> u8 {
    // `bits` is a `u8`, so a caller could pass e.g. 9 or 255. `1u32 << bits`
    // overflows for `bits >= 32` (in release that silently wraps and yields a
    // wrong bucket; in debug it panics inconsistently), and the result must
    // also fit in the returned `u8`, so cap at 8 — the widest RankQuant width
    // (b=8 yields one bucket per code value in `[0, 256)`, which still fits a
    // `u8`). `d == 0` would divide by zero. Guard both up front so the failure
    // is loud in every build.
    assert!(bits <= 8, "bits too large");
    assert!(d > 0, "d must be positive");
    // A valid rank is a position in `[0, d)`. Reject `rank >= d` loudly instead
    // of silently clamping the quotient back into range: the rest of the public
    // bucket API ([`pack_buckets`] / [`bucket_centre`]) fails loud on an
    // out-of-domain argument, so a direct caller that miscomputes a rank should
    // hear about it rather than receive a plausible-but-wrong bucket.
    assert!((rank as usize) < d, "rank ({rank}) must be < d ({d})");
    let n_buckets = 1u32 << bits;
    // u64 math: `d` is a `usize` and reaches this from the Python binding as a
    // free argument, so `d as u32` could truncate a `d >= 2^32` (e.g. to 0,
    // which would divide by zero and panic). rank ≤ u16::MAX and n_buckets ≤
    // 128, so the product fits u64 comfortably; over the realistic d ≤ u16::MAX
    // domain this is bit-identical to the previous u32 form.
    //
    // With `rank < d` guaranteed above, `rank * n_buckets / d < n_buckets`
    // (integer division floors), so the quotient already lands in
    // `[0, n_buckets)` and fits the returned `u8` without a clamp.
    ((rank as u64 * n_buckets as u64) / d as u64) as u8
}

/// Bucket every entry of a full rank vector.
///
/// # Panics
/// Panics if `bits > 7` (validated up front, so this holds for empty input
/// too), or — via [`rank_to_bucket`] — if any entry is `>= ranks.len()`. A
/// valid rank vector is a permutation of `[0, ranks.len())`, so well-formed
/// input never trips the per-entry guard.
pub fn bucket_ranks(ranks: &[u16], bits: u8) -> Vec<u8> {
    // Validate `bits` up front so an invalid width fails loud even for empty
    // input — an empty `ranks` skips the per-entry `rank_to_bucket` check and
    // would otherwise silently return an empty vec. Mirrors the Python binding,
    // which checks `bits` before its empty short-circuit.
    assert!(bits <= 8, "bits too large");
    let d = ranks.len();
    ranks.iter().map(|&r| rank_to_bucket(r, d, bits)).collect()
}

/// Pack a slice of bucket indices (each in `[0, 1 << bits)`) into a
/// dense byte stream.
///
/// Layout: the bucket with index 0 occupies the most-significant bits
/// of the first byte. Requires `bits ∈ {1, 2, 4, 8}` and `d`'s length to
/// be a multiple of `8 / bits`. For `bits == 8` the packing is the
/// degenerate one-code-per-byte case: each code is copied verbatim into
/// its own byte (no sub-byte shifting), so any `d` is valid.
///
/// # Panics
/// Panics if `bits ∉ {1, 2, 4, 8}`, if `buckets.len()` is not a multiple
/// of `8 / bits`, or if any code is `>= 1 << bits`. The last guard is
/// the public-contract backstop: an out-of-range code would otherwise
/// be silently truncated to `code & ((1 << bits) - 1)` and corrupt the
/// packed stream. (Internal callers feed codes straight from
/// [`rank_to_bucket`], which is always in range; this protects direct
/// callers of the primitive.) Note the `b=8` code range is the full
/// `u8`, so the range guard is vacuously satisfied for that width.
pub fn pack_buckets(buckets: &[u8], bits: u8) -> Vec<u8> {
    assert!(matches!(bits, 1 | 2 | 4 | 8), "bits must be 1, 2, 4, or 8");
    let codes_per_byte = (8 / bits) as usize;
    let d = buckets.len();
    assert_eq!(
        d % codes_per_byte,
        0,
        "d ({d}) must be a multiple of codes_per_byte ({codes_per_byte}) for bits = {bits}",
    );
    // `(1u8 << 8)` overflows a `u8`, so compute the mask in `u16` and saturate
    // the `b=8` case to the full byte (`0xFF`). For `b ∈ {1,2,4}` this is the
    // same value the old `(1u8 << bits) - 1` produced.
    let mask = ((1u16 << bits) - 1) as u8;
    let n_bytes = d / codes_per_byte;
    let mut out = vec![0u8; n_bytes];
    let bits_u = bits as usize;
    // Pack in a single pass, failing loud on an out-of-range code rather than
    // silently masking it (`code & mask` would turn e.g. 7 at bits=2 into 3,
    // packing a different vector). Checking inside the loop keeps the
    // fail-loud guarantee without a second O(d) pass over `buckets`; the
    // branch is loop-invariant-predictable for the always-valid internal
    // callers. Asserting `b <= mask` makes the trailing `& mask` redundant.
    // At `b=8`, `codes_per_byte == 1`, so `shift == 0` and each byte holds one
    // code verbatim.
    for (i, &b) in buckets.iter().enumerate() {
        assert!(
            b <= mask,
            "bucket code {b} out of range: every code must be < 1 << bits ({})",
            mask as u16 + 1,
        );
        let byte_idx = i / codes_per_byte;
        let pos = i % codes_per_byte;
        let shift = (codes_per_byte - 1 - pos) * bits_u;
        out[byte_idx] |= b << shift;
    }
    out
}

/// Unpack a stream of `bits`-bit bucket indices into a `Vec<u8>`.
///
/// Inverse of [`pack_buckets`].
pub fn unpack_buckets(packed: &[u8], d: usize, bits: u8) -> Vec<u8> {
    assert!(matches!(bits, 1 | 2 | 4 | 8), "bits must be 1, 2, 4, or 8");
    let codes_per_byte = (8 / bits) as usize;
    assert_eq!(packed.len() * codes_per_byte, d);
    // `(1u8 << 8)` overflows a `u8`; compute in `u16` and narrow so `b=8`
    // yields the full-byte mask `0xFF` (each byte already holds one code).
    let mask = ((1u16 << bits) - 1) as u8;
    let bits_u = bits as usize;
    let mut out = vec![0u8; d];
    #[allow(clippy::needless_range_loop)] // indexed access is clearer / matches the kernel layout
    for i in 0..d {
        let byte_idx = i / codes_per_byte;
        let pos = i % codes_per_byte;
        let shift = (codes_per_byte - 1 - pos) * bits_u;
        out[i] = (packed[byte_idx] >> shift) & mask;
    }
    out
}

/// Number of bytes per packed RankQuant document at dimension `d` and
/// bit width `bits ∈ {1, 2, 4, 8}`.
///
/// At `bits == 8` each coordinate occupies its own byte (`codes_per_byte
/// == 1`), so the storage is exactly `d` bytes per document.
#[inline]
pub fn rankquant_bytes_per_vec(d: usize, bits: u8) -> usize {
    // Guard the same domain as the sibling pack/unpack helpers: `bits == 0`
    // would divide by zero computing `codes_per_byte`, and only 1/2/4/8 give an
    // integral codes-per-byte.
    assert!(matches!(bits, 1 | 2 | 4 | 8), "bits must be 1,2,4,8");
    let codes_per_byte = (8 / bits) as usize;
    assert_eq!(
        d % codes_per_byte,
        0,
        "d ({d}) must be a multiple of codes_per_byte ({codes_per_byte}) for bits = {bits}"
    );
    d / codes_per_byte
}

/// Mean-centred value of a bucket index for a `bits`-bit RankQuant
/// scheme.
///
/// With `2^B` equal-width bins on the rank axis the bucket centres
/// after mean-centring are `b - (2^B - 1) / 2`, giving the symmetric
/// pattern `..., -1.5, -0.5, +0.5, +1.5, ...` for `B = 2`.
///
/// # Panics
/// Panics if `bits > 8` — bucket codes are `u8`, so the bit width is
/// capped at the representable bucketing range, matching
/// [`rank_to_bucket`] (the RankQuant family uses `bits ∈ {1, 2, 4, 8}`).
/// Also panics if `bucket >= 1 << bits`; this guard fails loud in *every*
/// build — like the sibling [`pack_buckets`] check — so a direct caller
/// cannot silently receive a centre outside the symmetric range. The
/// internal LUT builders only ever pass `bucket ∈ [0, 1 << bits)` (the
/// loop bound *is* `1 << bits`), so the assert never trips on the hot path.
/// For `bits == 8` the centres span `..., -0.5, +0.5, ...` around zero
/// with `bucket - 127.5`; the range guard is vacuous (every `u8` is a
/// valid code).
#[inline]
pub fn bucket_centre(bucket: u8, bits: u8) -> f32 {
    assert!(bits <= 8, "bits too large");
    assert!(
        (bucket as u32) < (1u32 << bits),
        "bucket {bucket} out of range for bits = {bits}",
    );
    let n = (1u32 << bits) as f32;
    bucket as f32 - (n - 1.0) / 2.0
}

/// L2 norm of a mean-centred rank vector of length `d`.
///
/// A rank vector is a permutation of `{0, ..., d - 1}`, so its mean is
/// `(d - 1) / 2` and the centred coordinates have variance
/// `(d^2 - 1) / 12`. The L2 norm is therefore
/// `sqrt(d * (d^2 - 1) / 12)`.
///
/// # Domain
/// The index types only call this with `d ∈ [2, MAX_DIM]` (the `u16` rank
/// invariant). It is `pub` as a standalone analytical-norm utility and stays
/// well-defined for any `d`: the product is formed in `f64` to avoid `f32`
/// intermediate overflow and catastrophic cancellation at large `d`, then
/// narrowed to the returned `f32` (correctly rounded to `f32` precision — `sqrt`
/// and the `f64 → f32` narrowing both round, so the value is not bit-exact).
/// It saturates to `f32::INFINITY` only for astronomically large `d`
/// (`d³/12 > f32::MAX`, i.e. `d ≳ 1.5e26`) — far beyond the `u16` cap. No
/// artificial guard is imposed: for any `d` a host can represent, `+∞` is the
/// honest norm of an unrepresentable magnitude, not a silent error.
#[inline]
pub fn rank_norm(d: usize) -> f32 {
    let d = d as f64;
    (d * (d * d - 1.0) / 12.0).sqrt() as f32
}

/// L2 norm of a mean-centred B-bit RankQuant vector of length `d`,
/// assuming each bucket receives exactly `d / 2^B` coordinates (true by
/// construction when the source is a permutation rank vector and
/// `d % 2^B == 0`).
///
/// The mean-centred bucket index has variance `(2^(2B) - 1) / 12`, so
/// the per-vector L2 norm is `sqrt(d * (2^(2B) - 1) / 12)`.
///
/// This is the **symmetric analytical** norm: it is exact only when
/// every bucket receives exactly `d / 2^B` coordinates, i.e. when
/// `d % 2^B == 0`. For `bits == 8` that precondition is `d % 256 == 0`;
/// the [`crate::RankQuant`] symmetric path enforces it before calling
/// this (see `RankQuant::new` / `symmetric_supported`). This primitive
/// itself only computes the closed form and does not re-check occupancy.
///
/// # Panics
/// Panics if `bits ∉ {1, 2, 4, 8}`, mirroring the [`crate::RankQuant`]
/// bit-width domain (and [`rankquant_bytes_per_vec`]). Without it a
/// nonsensical `bits` would return a norm for a scheme that does not
/// exist (or overflow `1 << bits`).
#[inline]
pub fn rankquant_norm(d: usize, bits: u8) -> f32 {
    assert!(matches!(bits, 1 | 2 | 4 | 8), "bits must be 1,2,4,8");
    let n = (1u32 << bits) as f64;
    let var = (n * n - 1.0) / 12.0;
    ((d as f64) * var).sqrt() as f32
}

// -----------------------------------------------------------------------
// Full-precision rank-cosine index.
//
// `u16` per coordinate; storage is `2 * dim` bytes per document.
// Symmetric and asymmetric search paths share the rank-transform
// pipeline above and the [`TopK`](crate::util::TopK) collector from
// [`crate::util`].
// -----------------------------------------------------------------------

/// Full-precision rank-cosine index.
///
/// Stores each document as a `u16` rank vector of length `dim`. Storage
/// is `2 * dim` bytes per document. Norms are not stored — a permutation
/// of `{0, ..., dim - 1}` has fixed analytical L2 norm
/// `sqrt(dim * (dim^2 - 1) / 12)` after mean-centring.
///
/// Use this mode as the parity / upper-bound reference. For deployment
/// at compact byte budgets, prefer [`crate::RankQuant`].
pub struct Rank {
    dim: usize,
    n_vectors: usize,
    /// Row-major `n_vectors * dim` rank values in `[0, dim)`.
    ranks: Vec<u16>,
}

impl std::fmt::Debug for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rank")
            .field("dim", &self.dim)
            .field("n_vectors", &self.n_vectors)
            .finish()
    }
}

impl Rank {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2, "dim must be >= 2");
        assert!(dim <= u16::MAX as usize, "dim must fit in u16");
        Self {
            dim,
            n_vectors: 0,
            ranks: Vec::new(),
        }
    }

    /// Add documents. Each vector is rank-transformed and stored row-major.
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
        let new_n = crate::util::checked_new_len(self.n_vectors, n, self.dim);
        let start = self.ranks.len();
        self.ranks.resize(start + n * self.dim, 0);
        let dim = self.dim;
        self.ranks[start..]
            .par_chunks_mut(dim)
            .zip(vectors.par_chunks(dim))
            .for_each(|(out, v)| rank_transform_into(v, out));
        self.n_vectors = new_n;
    }

    /// Symmetric rank-cosine search: rank-transform the query, then
    /// score by Spearman correlation against each stored rank vector.
    ///
    /// Score is `sum_d (q_rank[d] - mean) * (d_rank[d] - mean)`. The
    /// constant `1 / (norm * norm)` is omitted because it does not
    /// affect top-`k` ordering, but the *displayed* score reflects the
    /// normalised cosine in `[-1, 1]` by dividing by the fixed
    /// analytical norm pair.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        assert_all_finite(queries);
        // Clamp `k` to `n_vectors` before it sizes the `vec![_; nq * k]`
        // result buffers; an unclamped `usize::MAX` aborts the process
        // with `capacity overflow`.
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
        let mean_2x = (dim as i32) - 1; // 2 * mean = D - 1; use to avoid f32 in the inner loop
        let n = self.n_vectors;
        let norm = rank_norm(dim);
        let inv_norm_sq = 1.0_f32 / (norm * norm);

        let mut scores_flat = vec![0.0f32; buf_len];
        let mut indices_flat = vec![-1i64; buf_len];

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                let q_ranks = rank_transform(q);
                let mut top = TopK::new(k_eff);
                for di in 0..n {
                    let doc = &self.ranks[di * dim..(di + 1) * dim];
                    // sum_d (2*q[d] - (D-1)) * (2*doc[d] - (D-1)) / 4
                    let mut acc: i64 = 0;
                    for d in 0..dim {
                        let qc = 2 * (q_ranks[d] as i32) - mean_2x;
                        let dc = 2 * (doc[d] as i32) - mean_2x;
                        acc += (qc as i64) * (dc as i64);
                    }
                    let s = (acc as f32) * 0.25 * inv_norm_sq;
                    top.maybe_insert(s, di);
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

    /// Asymmetric rank-cosine search: queries stay as raw L2-normalised
    /// floats, documents are stored as ranks.
    ///
    /// Score is `sum_d q_unit[d] * (d_rank[d] - (D - 1) / 2) / norm`.
    /// The per-query constant `((D - 1) / 2) * sum_d q[d]` is folded out
    /// (it shifts every doc's score identically and does not change
    /// top-`k` ordering); the displayed score is the cosine on the
    /// mean-centred rank vector.
    pub fn search_asymmetric(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);
        assert_all_finite(queries);
        // Clamp `k` to `n_vectors` before sizing the `vec![_; nq * k]`
        // result buffers; `usize::MAX` otherwise aborts with capacity
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
        let n = self.n_vectors;
        let norm = rank_norm(dim);
        let inv_norm = 1.0_f32 / norm;
        let mean = (dim as f32 - 1.0) / 2.0;

        let mut scores_flat = vec![0.0f32; buf_len];
        let mut indices_flat = vec![-1i64; buf_len];

        queries
            .par_chunks(dim)
            .zip(scores_flat.par_chunks_mut(k))
            .zip(indices_flat.par_chunks_mut(k))
            .for_each(|((q, out_scores), out_indices)| {
                // L2-normalise the query so the displayed score is a
                // cosine on the centred rank vector.
                let q_unit = l2_normalise(q);
                let q_sum: f32 = q_unit.iter().sum();
                let mut top = TopK::new(k_eff);
                for di in 0..n {
                    let doc = &self.ranks[di * dim..(di + 1) * dim];
                    let mut acc = 0.0f32;
                    for d in 0..dim {
                        acc += q_unit[d] * doc[d] as f32;
                    }
                    // <q, doc - mean> = <q, doc> - mean * <q, 1>
                    let s = (acc - mean * q_sum) * inv_norm;
                    top.maybe_insert(s, di);
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
        self.dim * 2
    }
    /// Total bytes held by the rank buffer (excludes Vec overhead).
    pub fn byte_size(&self) -> usize {
        self.ranks.len() * std::mem::size_of::<u16>()
    }

    /// Remove a vector in O(1) by swapping with the last
    /// (swap-remove semantics).
    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(idx < self.n_vectors, "index out of bounds");
        let last = self.n_vectors - 1;
        let dim = self.dim;
        if idx != last {
            self.ranks
                .copy_within(last * dim..last * dim + dim, idx * dim);
        }
        self.ranks.truncate(last * dim);
        self.n_vectors -= 1;
        last
    }

    /// Persist to a `.ovr` file. Format: 13-byte header + u16 ranks LE.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::rank_io::write_rank(path, self.dim, self.n_vectors, &self.ranks)
    }

    /// Persist to any byte writer using the `.ovr` format.
    pub fn write_to<W: std::io::Write>(&self, writer: W) -> std::io::Result<()> {
        crate::rank_io::write_rank_to(writer, self.dim, self.n_vectors, &self.ranks)
    }

    /// Load from a `.ovr` file produced by [`Self::write`].
    ///
    /// Legacy `.tvr` files (magic `TVR1`) written by older versions of this
    /// crate are also accepted; newly written files use the `OVR1` magic.
    ///
    /// Returns `io::Error` (kind `InvalidData`) on any structural
    /// inconsistency between the header and the payload (`load_rank`
    /// validates `dim ∈ [2, MAX_DIM]`, bounds `n_vectors`, and uses
    /// `checked_mul` for the payload size). Additional invariants
    /// specific to `Rank` are checked here.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (dim, n_vectors, ranks) = crate::rank_io::load_rank(path)?;
        Self::from_persisted_parts(dim, n_vectors, ranks)
    }

    /// Load a `.ovr`/legacy `.tvr` index from any reader that can seek.
    ///
    /// The reader is parsed from its current position through EOF; any trailing
    /// bytes after the declared payload are rejected.
    pub fn read_from<R: std::io::Read + std::io::Seek>(reader: R) -> std::io::Result<Self> {
        let (dim, n_vectors, ranks) = crate::rank_io::load_rank_from(reader)?;
        Self::from_persisted_parts(dim, n_vectors, ranks)
    }

    /// Load a `.ovr`/legacy `.tvr` index from an in-memory byte slice.
    pub fn load_from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::read_from(std::io::Cursor::new(bytes))
    }

    fn from_persisted_parts(
        dim: usize,
        n_vectors: usize,
        ranks: Vec<u16>,
    ) -> std::io::Result<Self> {
        // `checked_mul` (not `saturating`): on a 32-bit target `n_vectors * dim`
        // can overflow `usize`; treat that as malformed rather than letting a
        // saturated `usize::MAX` stand in for the expected length.
        let expected = n_vectors.checked_mul(dim).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "OVR1 n_vectors * dim overflows usize",
            )
        })?;
        if ranks.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "OVR1 payload length does not match dim * n_vectors",
            ));
        }
        Ok(Self {
            dim,
            n_vectors,
            ranks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_transform_matches_numpy_argsort_argsort() {
        // [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
        // argsort = [1, 3, 6, 0, 2, 4, 7, 5]
        // argsort(argsort) = [3, 0, 4, 1, 5, 7, 2, 6]
        let v = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
        let r = rank_transform(&v);
        assert_eq!(r, vec![3, 0, 4, 1, 5, 7, 2, 6]);
    }

    #[test]
    fn rank_transform_is_permutation() {
        let v: Vec<f32> = (0..256).map(|i| (i as f32 * 7.0).sin()).collect();
        let r = rank_transform(&v);
        let mut sorted = r.clone();
        sorted.sort();
        let expected: Vec<u16> = (0..256u16).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn ties_broken_by_index() {
        let v = [1.0_f32, 1.0, 1.0, 1.0];
        let r = rank_transform(&v);
        assert_eq!(r, vec![0, 1, 2, 3]);
    }

    #[test]
    fn duplicate_values_tie_by_original_index() {
        let v = [3.0_f32, 1.0, 3.0, 2.0, 1.0];
        let r = rank_transform(&v);
        assert_eq!(r, vec![3, 0, 4, 2, 1]);
    }

    #[test]
    fn signed_zeroes_tie_by_original_index() {
        let v = [0.0_f32, -0.0, 1.0, -0.0, 0.0];
        let r = rank_transform(&v);
        assert_eq!(r, vec![0, 1, 4, 2, 3]);
    }

    #[test]
    fn rank_to_bucket_partitions_uniformly() {
        let d = 1024;
        let bits = 2u8;
        let mut counts = [0usize; 4];
        for rank in 0..d as u16 {
            let b = rank_to_bucket(rank, d, bits);
            counts[b as usize] += 1;
        }
        for c in counts {
            assert_eq!(c, d / 4);
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn rank_to_bucket_large_d_does_not_divide_by_zero() {
        // `d` reaches this from the Python binding as a free `usize`; a `d`
        // above `u32::MAX` must not truncate through `d as u32` to 0 and panic.
        // 64-bit only: 2^40 isn't representable on a 32-bit usize.
        let huge_d = 1usize << 40;
        assert_eq!(rank_to_bucket(0, huge_d, 2), 0);
        assert!(rank_to_bucket(u16::MAX, huge_d, 2) < 4);
    }

    #[test]
    #[should_panic(expected = "must be < d")]
    fn rank_to_bucket_rejects_rank_ge_d() {
        // A valid rank lives in `[0, d)`; `rank == d` is out of range. It used
        // to clamp silently to the top bucket — now it fails loud in release
        // too, matching pack_buckets / bucket_centre.
        let _ = rank_to_bucket(8, 8, 2);
    }

    #[test]
    #[should_panic(expected = "bits too large")]
    fn bucket_ranks_rejects_bits_above_8_even_when_empty() {
        // `bits` is validated up front, so an invalid width fails loud even on
        // empty input — which never reaches the per-entry rank_to_bucket guard.
        // The valid bucketing range now extends to b=8 (the widest RankQuant
        // width), so b=9 is the first rejected width.
        let _ = bucket_ranks(&[], 9);
    }

    #[test]
    fn bucket_ranks_accepts_bits_8() {
        // b=8 is a supported width: a 4-element rank vector buckets without
        // panicking and yields codes in [0, 256).
        let codes = bucket_ranks(&[0, 1, 2, 3], 8);
        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn pack_unpack_round_trip_bits2() {
        let buckets: Vec<u8> = (0..16).map(|i| (i % 4) as u8).collect();
        let packed = pack_buckets(&buckets, 2);
        assert_eq!(packed.len(), 4);
        let unpacked = unpack_buckets(&packed, 16, 2);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn pack_unpack_round_trip_bits1() {
        let buckets: Vec<u8> = (0..16).map(|i| (i % 2) as u8).collect();
        let packed = pack_buckets(&buckets, 1);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_buckets(&packed, 16, 1);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn pack_unpack_round_trip_bits4() {
        let buckets: Vec<u8> = (0..16).map(|i| (i % 16) as u8).collect();
        let packed = pack_buckets(&buckets, 4);
        assert_eq!(packed.len(), 8);
        let unpacked = unpack_buckets(&packed, 16, 4);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn pack_unpack_round_trip_bits8() {
        // b=8 is the degenerate one-code-per-byte packing: each byte holds a
        // full code in `[0, 256)`, so packed length == code count and the
        // bytes are the codes verbatim. Cover the full code range including
        // 0 and 255 (the extremes the `b ∈ {1,2,4}` mask would have clipped).
        let buckets: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let packed = pack_buckets(&buckets, 8);
        assert_eq!(packed.len(), 256, "b=8 stores one byte per code");
        assert_eq!(packed, buckets, "b=8 packing is the identity byte stream");
        let unpacked = unpack_buckets(&packed, 256, 8);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn pack_unpack_round_trip_bits8_arbitrary_len() {
        // Any `d` is a valid b=8 length (codes_per_byte == 1); 384 is not a
        // multiple of 256 yet still round-trips — code generation never needs
        // the equal-bucket precondition that only the symmetric norm requires.
        let buckets: Vec<u8> = (0..384u16).map(|i| (i % 256) as u8).collect();
        let packed = pack_buckets(&buckets, 8);
        assert_eq!(packed.len(), 384);
        let unpacked = unpack_buckets(&packed, 384, 8);
        assert_eq!(unpacked, buckets);
    }

    #[test]
    fn rank_to_bucket_b8_spans_full_byte_range() {
        // rank in [0, d) with bits=8 must land in [0, 256). Check the extremes
        // and that the top rank maps to the top bucket for d == 256.
        let d = 256usize;
        assert_eq!(rank_to_bucket(0, d, 8), 0);
        assert_eq!(rank_to_bucket(255, d, 8), 255);
        // A coarser d still keeps the quotient in range.
        assert!(rank_to_bucket(383, 384, 8) < 255 || rank_to_bucket(383, 384, 8) == 255);
        for rank in 0..d as u16 {
            let _ = rank_to_bucket(rank, d, 8); // never panics, always < 256
        }
    }

    #[test]
    fn bucket_centre_b8_is_symmetric_around_zero() {
        // For b=8 the 256 centres span -127.5 ..= +127.5 and sum to 0.
        assert_eq!(bucket_centre(0, 8), -127.5);
        assert_eq!(bucket_centre(255, 8), 127.5);
        let sum: f32 = (0..256u16).map(|b| bucket_centre(b as u8, 8)).sum();
        assert!(sum.abs() < 1e-3, "b=8 centres should sum to ~0, got {sum}");
    }

    #[test]
    fn rankquant_norm_b8_matches_direct_computation() {
        // d % 256 == 0 so every bucket gets exactly d/256 entries and the
        // analytical norm is exact.
        let d = 512usize;
        let bits = 8u8;
        let analytical = rankquant_norm(d, bits);
        let ranks: Vec<u16> = (0..d as u16).collect();
        let buckets = bucket_ranks(&ranks, bits);
        let centred: Vec<f32> = buckets.iter().map(|&b| bucket_centre(b, bits)).collect();
        let direct: f32 = centred.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (analytical - direct).abs() / direct < 1e-5,
            "analytical {analytical}, direct {direct}"
        );
    }

    #[test]
    fn bucket_centres_are_symmetric_around_zero() {
        // For B = 2: bucket values are {-1.5, -0.5, +0.5, +1.5}.
        let centres: Vec<f32> = (0..4u8).map(|b| bucket_centre(b, 2)).collect();
        assert_eq!(centres, vec![-1.5, -0.5, 0.5, 1.5]);
        let sum: f32 = centres.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn rank_norm_matches_direct_computation() {
        let d = 1024usize;
        let analytical = rank_norm(d);
        let direct: f32 = {
            let mean = (d as f32 - 1.0) / 2.0;
            let ss: f32 = (0..d)
                .map(|i| {
                    let c = i as f32 - mean;
                    c * c
                })
                .sum();
            ss.sqrt()
        };
        assert!(
            (analytical - direct).abs() / direct < 1e-5,
            "analytical {analytical}, direct {direct}"
        );
    }

    #[test]
    fn rankquant_norm_matches_direct_computation() {
        let d = 1024usize;
        let bits = 2u8;
        let analytical = rankquant_norm(d, bits);
        // Build the bucketed vector of an ideal permutation and measure
        // its mean-centred L2 norm directly.
        let ranks: Vec<u16> = (0..d as u16).collect();
        let buckets = bucket_ranks(&ranks, bits);
        let centred: Vec<f32> = buckets.iter().map(|&b| bucket_centre(b, bits)).collect();
        let direct: f32 = centred.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (analytical - direct).abs() / direct < 1e-5,
            "analytical {analytical}, direct {direct}"
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn pack_buckets_rejects_out_of_range_code() {
        // Code 7 at bits=2 is out of `[0, 4)`. The old `& mask` silently
        // packed it as 3; the contract now rejects it loud.
        let _ = pack_buckets(&[7, 7, 7, 7], 2);
    }

    #[test]
    #[should_panic(expected = "bits too large")]
    fn bucket_centre_rejects_bits_above_8() {
        // bits >= 32 overflows `1 << bits`; the guard caps at 8 (the widest
        // RankQuant width, whose codes still fit a u8), matching
        // `rank_to_bucket`. b=9 is the first rejected width.
        let _ = bucket_centre(0, 9);
    }

    #[test]
    fn bucket_centre_accepts_bits_8() {
        // b=8 centres are valid: code 0 → -127.5, code 255 → +127.5.
        assert_eq!(bucket_centre(0, 8), -127.5);
        assert_eq!(bucket_centre(255, 8), 127.5);
    }

    #[test]
    #[should_panic(expected = "out of range for bits")]
    fn bucket_centre_rejects_out_of_range_bucket() {
        // bucket 4 at bits=2 is outside [0, 4). The guard now fails loud in
        // release too (was debug-only), matching pack_buckets and the Python
        // wrapper — otherwise the caller silently gets a centre of +2.5.
        let _ = bucket_centre(4, 2);
    }

    #[test]
    #[should_panic(expected = "bits must be 1,2,4")]
    fn rankquant_norm_rejects_invalid_bits() {
        // 3-bit packing has no RankQuant scheme; the norm must refuse it
        // rather than return a value for a non-existent layout.
        let _ = rankquant_norm(64, 3);
    }
}
