//! Stateless dense bucket-overlap contingency table for two fixed-length
//! ordinal bucket-code vectors.
//!
//! A [`Contingency`] is the full `nb × nb` co-occurrence table between two
//! bucket-code slices `q` and `d` of equal length `dim`:
//!
//! ```text
//! count(a, b) = #{ j ∈ [0, dim) : q[j] == a  ∧  d[j] == b }
//! ```
//!
//! This is the *stateless dense-code contingency* surface (issue #219): it
//! consumes two `&[u8]` code slices directly — there is no index, no corpus,
//! no persistence, and it is **not** wired into any retrieval path. It is the
//! algebraic object the multi-bucket bilinear score decomposes into (see
//! [`crate::MultiBucketBitmap`]), exposed as a contingency table so callers can
//! compute arbitrary projections (diagonal/band agreement, top-bucket overlap,
//! L1 distance, coarsened tables, the symmetric RankQuant score, and general
//! learned `nb × nb` weight matrices) over a single `O(dim)` histogram pass.
//!
//! Ported to reach behavioural parity with `ordgraph::edge::Contingency`. The
//! `EdgeEvidence` "this is evidence for X" wrapper deliberately stays in
//! ordgraph — ordvec exposes only the substrate primitive.
//!
//! ## Count width
//!
//! Cell counts are stored as `u32`. A single cell holds at most `dim` codes
//! (every coordinate lands in exactly one cell), so the table never overflows
//! while `dim <= u32::MAX`, which the constructor asserts. The reference
//! prototype used `u16` with a `checked_add` that panics past `dim > u16::MAX`;
//! `u32` removes that hazard for the stateless code path, where the caller's
//! `dim` is arbitrary (the crate-wide `u16` *rank* invariant does not bind raw
//! bucket codes), at 2 bytes/cell more than `u16`. For the small `nb` this type
//! targets (`nb` is a per-coordinate bucket count, typically `≤ 16`) the table
//! is `nb² ≤ 256` cells, so the width choice is immaterial to memory.

use crate::OrdvecError;

/// Full bucket-overlap contingency table for two equal-length bucket codes.
///
/// Construct with [`Contingency::new`] from two `&[u8]` code slices that share
/// a length (`dim`) and a bucket count (`nb`). Every accessor and projection
/// reads the cached `nb × nb` table; no rescanning of the input codes occurs
/// after construction.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Contingency {
    buckets: usize,
    /// Row-major `buckets × buckets`. `counts[a * buckets + b]` is the number
    /// of coordinates with query-bucket `a` and doc-bucket `b`.
    counts: Vec<u32>,
}

impl Contingency {
    /// Build the contingency table from two bucket-code slices.
    ///
    /// `query` and `doc` must have the same length (`dim`), and every code must
    /// be a valid bucket id `< nb`. One `O(dim)` histogram pass fills the full
    /// `nb × nb` table.
    ///
    /// # Errors
    /// - [`OrdvecError::InvalidParameter`] if `nb == 0`, or if `dim` exceeds
    ///   `u32::MAX` (a single cell could then exceed `u32`).
    /// - [`OrdvecError::InvalidVectorLength`] if `doc.len() != query.len()`.
    /// - [`OrdvecError::InvalidParameter`] if any code is `>= nb`.
    pub fn new(query: &[u8], doc: &[u8], nb: usize) -> Result<Self, OrdvecError> {
        if nb == 0 {
            return Err(OrdvecError::InvalidParameter {
                name: "nb",
                message: "bucket count must be > 0".to_string(),
            });
        }
        // Codes are `u8`, so a bucket id is in `0..=255` and `nb > 256` is both
        // meaningless and dangerous: `nb` is a caller-supplied `usize`, so a
        // large value would allocate an `nb * nb` table (e.g. `nb = 1 << 20` ⇒
        // a terabyte) and abort the process. Cap it at the u8 domain.
        if nb > u8::MAX as usize + 1 {
            return Err(OrdvecError::InvalidParameter {
                name: "nb",
                message: format!("must be <= 256 (codes are u8); got {nb}"),
            });
        }
        if query.len() != doc.len() {
            return Err(OrdvecError::InvalidVectorLength {
                name: "doc",
                len: doc.len(),
                expected: query.len(),
            });
        }
        if query.len() > u32::MAX as usize {
            return Err(OrdvecError::InvalidParameter {
                name: "dim",
                message: format!("must be <= {} (u32 contingency count cap)", u32::MAX),
            });
        }
        // `nb * nb` is checked: a hostile `nb` near `usize::MAX` would wrap the
        // table-length multiply and silently under-size the allocation.
        let table_len = nb
            .checked_mul(nb)
            .ok_or_else(|| OrdvecError::InvalidParameter {
                name: "nb",
                message: "bucket count squared overflows usize".to_string(),
            })?;
        let mut counts = vec![0u32; table_len];
        build_histogram(query, doc, nb, &mut counts)?;
        Ok(Self {
            buckets: nb,
            counts,
        })
    }

    /// Number of buckets `nb` (the table is `nb × nb`).
    pub fn buckets(&self) -> usize {
        self.buckets
    }

    /// The flat row-major `nb × nb` count table.
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Count of coordinates with query-bucket `a` and doc-bucket `b`.
    ///
    /// # Panics
    /// Panics if `a >= nb` or `b >= nb`.
    pub fn count(&self, query_bucket: usize, doc_bucket: usize) -> u32 {
        assert!(query_bucket < self.buckets, "query_bucket out of range");
        assert!(doc_bucket < self.buckets, "doc_bucket out of range");
        self.counts[query_bucket * self.buckets + doc_bucket]
    }

    /// Sum of row `query_bucket`: how many coordinates the query placed in that
    /// bucket. For fixed-composition codes this is the constant per-bucket
    /// occupancy.
    ///
    /// # Panics
    /// Panics if `query_bucket >= nb`.
    pub fn row_sum(&self, query_bucket: usize) -> u32 {
        assert!(query_bucket < self.buckets, "query_bucket out of range");
        let base = query_bucket * self.buckets;
        self.counts[base..base + self.buckets].iter().sum()
    }

    /// Sum of column `doc_bucket`: how many coordinates the doc placed in that
    /// bucket.
    ///
    /// # Panics
    /// Panics if `doc_bucket >= nb`.
    pub fn column_sum(&self, doc_bucket: usize) -> u32 {
        assert!(doc_bucket < self.buckets, "doc_bucket out of range");
        (0..self.buckets)
            .map(|query_bucket| self.counts[query_bucket * self.buckets + doc_bucket])
            .sum()
    }

    /// Total mass: equals `dim`.
    pub fn total_count(&self) -> u32 {
        self.counts.iter().copied().sum()
    }

    /// Count in the top-bucket cell `(nb − 1, nb − 1)`: coordinates both codes
    /// rank in the highest bucket.
    pub fn top_overlap(&self) -> u32 {
        self.count(self.buckets - 1, self.buckets - 1)
    }

    /// Trace of the table: coordinates assigned to the same bucket by both
    /// codes, summed over all buckets.
    pub fn diagonal_agreement(&self) -> u32 {
        (0..self.buckets)
            .map(|bucket| self.counts[bucket * self.buckets + bucket])
            .sum()
    }

    /// Mass within `radius` of the diagonal: coordinates whose two bucket codes
    /// differ by at most `radius`. `radius = 0` reduces to
    /// [`Self::diagonal_agreement`].
    pub fn band_agreement(&self, radius: usize) -> u32 {
        let mut total = 0u32;
        for qb in 0..self.buckets {
            let base = qb * self.buckets;
            // Iterate only the in-band columns instead of scanning every
            // column with an `abs_diff` filter.
            let start = qb.saturating_sub(radius);
            // `saturating_add`: `radius` is an uncapped public parameter, so a
            // near-`usize::MAX` value must not overflow before the `.min()`.
            let end = qb.saturating_add(radius).min(self.buckets - 1);
            for db in start..=end {
                total += self.counts[base + db];
            }
        }
        total
    }

    /// Mass in the top-right `group_width × group_width` block: coordinates both
    /// codes place in the top `group_width` buckets.
    ///
    /// # Panics
    /// Panics if `group_width == 0` or `group_width > nb`.
    pub fn top_group_overlap(&self, group_width: usize) -> u32 {
        assert!(group_width > 0, "group_width must be > 0");
        assert!(group_width <= self.buckets, "group_width must be <= nb");
        let start = self.buckets - group_width;
        let mut total = 0u32;
        for qb in start..self.buckets {
            let base = qb * self.buckets;
            for db in start..self.buckets {
                total += self.counts[base + db];
            }
        }
        total
    }

    /// Total bucket-index L1 distance: `Σ_{a,b} |a − b| · count(a, b)`. The
    /// integer earth-mover-style cost of moving the doc histogram onto the
    /// query histogram along the bucket axis.
    /// Returns `u64`: the distance-weighted sum `Σ |a−b|·C[a][b]` can reach
    /// `(nb−1)·dim`, which overflows `u32` for accepted inputs (a single count
    /// already fits `u32` only up to `dim ≤ u32::MAX`; the `|a−b|` weight then
    /// scales it past `u32`). `u64` is overflow-free for every constructible
    /// table (`nb ≤ 256`, `dim ≤ u32::MAX` ⇒ max `255 · u32::MAX < u64::MAX`).
    pub fn bucket_l1_distance(&self) -> u64 {
        let mut total = 0u64;
        for qb in 0..self.buckets {
            let base = qb * self.buckets;
            for db in 0..self.buckets {
                total += qb.abs_diff(db) as u64 * u64::from(self.counts[base + db]);
            }
        }
        total
    }

    /// Coarsen the `nb × nb` table into a `groups × groups` table by merging
    /// contiguous equal-width bucket blocks. Preserves total mass.
    ///
    /// # Panics
    /// Panics if `groups == 0`, `groups > nb`, or `nb` is not divisible by
    /// `groups`.
    pub fn coarsened_counts(&self, groups: usize) -> Vec<u32> {
        assert!(groups > 0, "groups must be > 0");
        assert!(groups <= self.buckets, "groups must be <= nb");
        assert!(
            self.buckets.is_multiple_of(groups),
            "bucket count must be divisible by groups"
        );
        let width = self.buckets / groups;
        let mut out = vec![0u32; groups * groups];
        for qb in 0..self.buckets {
            let base = qb * self.buckets;
            let qg = qb / width;
            for db in 0..self.buckets {
                let dg = db / width;
                out[qg * groups + dg] += self.counts[base + db];
            }
        }
        out
    }

    /// Symmetric RankQuant score: `Σ_{a,b} (a − c)(b − c) · count(a, b)` with
    /// `c = (nb − 1) / 2`. Algebraically identical to the per-coordinate
    /// centred product `Σ_j (q[j] − c)(d[j] − c)` — the outer-product weight
    /// matrix just rearranges the same sum.
    pub fn rankquant_symmetric_score(&self) -> f32 {
        let centre = (self.buckets as f32 - 1.0) / 2.0;
        let mut score = 0.0f32;
        for qb in 0..self.buckets {
            let base = qb * self.buckets;
            let qw = qb as f32 - centre;
            // `qw` is invariant over the inner loop: accumulate the row's
            // centred mass first, then scale by `qw` once (nb multiplies
            // instead of nb²).
            let mut row_sum = 0.0f32;
            for db in 0..self.buckets {
                row_sum += (db as f32 - centre) * self.counts[base + db] as f32;
            }
            score += qw * row_sum;
        }
        score
    }

    /// General projection under an arbitrary `nb × nb` weight matrix:
    /// `Σ_{a,b} weights[a * nb + b] · count(a, b)`. This is the learned/custom
    /// weight-matrix entry point — the diagonal, band, and outer-product helpers
    /// above are all special cases of this dense reduction.
    ///
    /// # Panics
    /// Panics if `weights.len() != nb * nb`.
    pub fn project(&self, weights: &[f32]) -> f32 {
        assert_eq!(
            weights.len(),
            self.counts.len(),
            "weights must be an nb * nb matrix",
        );
        weights
            .iter()
            .zip(self.counts.iter())
            .map(|(&w, &c)| w * c as f32)
            .sum()
    }
}

/// Named projections over a [`Contingency`], mirroring
/// `ordgraph::edge::Projection`. Each variant's [`Self::score`] returns the
/// projection as `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// Top-bucket cell count — [`Contingency::top_overlap`].
    TopOverlap,
    /// Top `width`-bucket block mass — [`Contingency::top_group_overlap`].
    TopGroupOverlap { width: usize },
    /// Diagonal trace — [`Contingency::diagonal_agreement`].
    DiagonalAgreement,
    /// Banded mass within `radius` — [`Contingency::band_agreement`].
    BandAgreement { radius: usize },
    /// Bucket-index L1 transport cost — [`Contingency::bucket_l1_distance`].
    BucketL1Distance,
    /// Centred outer-product score — [`Contingency::rankquant_symmetric_score`].
    RankQuantSymmetric,
}

impl Projection {
    /// Evaluate this projection against `contingency`, returning the value as
    /// `f32` (matching the ordgraph projection contract).
    pub fn score(self, contingency: &Contingency) -> f32 {
        match self {
            Self::TopOverlap => contingency.top_overlap() as f32,
            Self::TopGroupOverlap { width } => contingency.top_group_overlap(width) as f32,
            Self::DiagonalAgreement => contingency.diagonal_agreement() as f32,
            Self::BandAgreement { radius } => contingency.band_agreement(radius) as f32,
            Self::BucketL1Distance => contingency.bucket_l1_distance() as f32,
            Self::RankQuantSymmetric => contingency.rankquant_symmetric_score(),
        }
    }
}

/// Fill the row-major `nb × nb` `counts` table from the two code slices.
///
/// Dispatches at runtime to an AVX-512 BW byte-compare kernel when available
/// (the masked-tail popcount-AND discipline mirrors [`crate::Bitmap`]'s
/// scan kernels), otherwise the portable scalar histogram scatter. Both paths
/// validate every code is `< nb` and return [`OrdvecError::InvalidParameter`]
/// on the first out-of-range code.
fn build_histogram(
    query: &[u8],
    doc: &[u8],
    nb: usize,
    counts: &mut [u32],
) -> Result<(), OrdvecError> {
    debug_assert_eq!(query.len(), doc.len());
    debug_assert_eq!(counts.len(), nb * nb);

    // Validate the code range up front so the SIMD kernel can assume every
    // `q[j] < nb` and `d[j] < nb` and index the table without a per-element
    // bounds check (the scalar fallback shares the same validated contract).
    if let Some(bad) = find_out_of_range(query, doc, nb) {
        return Err(OrdvecError::InvalidParameter {
            name: "code",
            message: format!("bucket code {bad} out of range (must be < {nb})"),
        });
    }

    #[cfg(target_arch = "x86_64")]
    let use_avx512 = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vpopcntdq")
        // The byte-compare kernel materialises an nb-wide mask table; cap it to
        // the small bucket counts this type targets so the per-bucket popcount
        // pass stays cheaper than a single scalar scatter.
        && nb <= 16;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512 = false;

    if use_avx512 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            build_histogram_avx512(query, doc, nb, counts);
            return Ok(());
        }
    }

    build_histogram_scalar(query, doc, nb, counts);
    Ok(())
}

/// Return the first code (from either slice) that is `>= nb`, else `None`.
fn find_out_of_range(query: &[u8], doc: &[u8], nb: usize) -> Option<u8> {
    // nb fits a usize; codes are u8, so any code >= nb is out of range. When
    // nb > 255 every u8 code is in range, so the scan can be skipped.
    if nb > u8::MAX as usize {
        return None;
    }
    let cap = nb as u8;
    query.iter().chain(doc.iter()).copied().find(|&c| c >= cap)
}

/// Portable scalar histogram scatter: one `O(dim)` pass, one table increment
/// per coordinate. Assumes every code is `< nb` (validated by the caller).
fn build_histogram_scalar(query: &[u8], doc: &[u8], nb: usize, counts: &mut [u32]) {
    for (&qb, &db) in query.iter().zip(doc.iter()) {
        let idx = qb as usize * nb + db as usize;
        // `+= 1` cannot overflow u32: a cell holds at most `dim <= u32::MAX`
        // coordinates and each coordinate increments exactly one cell.
        counts[idx] += 1;
    }
}

/// AVX-512 BW + VPOPCNTDQ contingency build.
///
/// For each bucket pair `(a, b)` the cell `count(a, b)` is
/// `Σ_words popcount((q_bytes == a) & (d_bytes == b))` — a popcount over the
/// bitwise-AND of two byte-equality masks. This is the exact popcount-AND shape
/// of [`crate::Bitmap`]'s overlap kernel, lifted from precomputed bitmaps to
/// on-the-fly `_mm512_cmpeq_epi8_mask` comparisons, with the trailing
/// `< 64`-byte tail handled under a `bzhi`-style length mask (the masked-tail
/// discipline mirrors the bitmap scan).
///
/// # Safety
/// Requires the `avx512f`, `avx512bw`, and `avx512vpopcntdq` target features,
/// confirmed by the `#[target_feature]` gate plus the caller's runtime
/// `is_x86_feature_detected!`. `query.len() == doc.len()` and
/// `counts.len() == nb * nb` are caller contracts; all loads are masked to the
/// live byte count so no read passes the end of either slice.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn build_histogram_avx512(query: &[u8], doc: &[u8], nb: usize, counts: &mut [u32]) {
    use std::arch::x86_64::*;
    // SAFETY: `query.len() == doc.len() == len` (caller contract), so the two
    // masked 64-byte loads at the same offset cover the same live region; the
    // tail load uses a `(1 << rem) - 1` byte mask so lanes past `len` are never
    // read. `counts.len() == nb * nb` and `a, b < nb <= 16` (dispatch gate)
    // bound every `counts[a * nb + b]` write. AVX-512 F/BW/VPOPCNTDQ are
    // confirmed by `#[target_feature]` + the runtime detection in the caller.
    // The explicit block is required by `#![deny(unsafe_op_in_unsafe_fn)]`.
    unsafe {
        let len = query.len();
        let q_ptr = query.as_ptr();
        let d_ptr = doc.as_ptr();

        // Per-cell 64-bit accumulators, row-major `nb × nb` (nb <= 16 ⇒ <= 256
        // u64). A doc cell holds at most `len` codes; u64 cannot overflow.
        let mut acc = vec![0u64; nb * nb];

        // Bucket-value broadcast vectors, invariant across the 64-byte blocks —
        // precompute once instead of recomputing `set1_epi8` per block per
        // bucket. `nb <= 16` (dispatch gate), so a fixed array of 16 covers it.
        let mut splats = [_mm512_setzero_si512(); 16];
        for (i, s) in splats.iter_mut().enumerate().take(nb) {
            *s = _mm512_set1_epi8(i as i8);
        }

        let mut off = 0usize;
        while off < len {
            let rem = len - off;
            let (q_vec, d_vec) = if rem >= 64 {
                (
                    _mm512_loadu_si512(q_ptr.add(off) as *const __m512i),
                    _mm512_loadu_si512(d_ptr.add(off) as *const __m512i),
                )
            } else {
                // Masked tail: only the low `rem` bytes are loaded; the rest of
                // each register reads as zero and is excluded from every
                // `cmpeq` mask by ANDing with `live` below.
                let load_mask: __mmask64 = (1u64 << rem) - 1;
                (
                    _mm512_maskz_loadu_epi8(load_mask, q_ptr.add(off) as *const i8),
                    _mm512_maskz_loadu_epi8(load_mask, d_ptr.add(off) as *const i8),
                )
            };
            // `live` marks the lanes that hold a real byte in this block.
            let live: __mmask64 = if rem >= 64 {
                u64::MAX
            } else {
                (1u64 << rem) - 1
            };

            // For each query bucket `a`, the lanes where `q == a`; for each doc
            // bucket `b`, the lanes where `d == b`. The cell increment is the
            // popcount of the AND of the two lane masks, restricted to live
            // lanes. This is popcount(maskQ & maskD) — the masked popcount-AND.
            for a in 0..nb {
                let q_eq: __mmask64 = _mm512_cmpeq_epi8_mask(q_vec, splats[a]) & live;
                if q_eq == 0 {
                    continue;
                }
                let row = a * nb;
                for b in 0..nb {
                    let d_eq: __mmask64 = _mm512_cmpeq_epi8_mask(d_vec, splats[b]);
                    acc[row + b] += (q_eq & d_eq).count_ones() as u64;
                }
            }

            off += 64;
        }

        for (cell, &v) in counts.iter_mut().zip(acc.iter()) {
            // Each cell holds at most `len <= u32::MAX` codes (caller-asserted),
            // so the u64 accumulator fits u32.
            *cell = v as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ordgraph::edge parity gate -------------------------------------
    // Every assertion value below is reproduced verbatim from
    // ordgraph-proto/src/edge.rs's tests. The bucket codes are the *already
    // bucketed* outputs ordgraph derives from ranks; here we feed the codes
    // directly (the stateless dense-code contract).

    /// The ordgraph `contingency_counts_bucket_intersections` case:
    /// query = [0,0,1,1,2,2,3,3], doc = [3,3,2,2,1,1,0,0] (reverse), nb = 4.
    #[test]
    fn parity_counts_bucket_intersections() {
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [3u8, 3, 2, 2, 1, 1, 0, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();

        assert_eq!(c.count(0, 3), 2);
        assert_eq!(c.count(3, 0), 2);
        assert_eq!(c.top_overlap(), 0);
        assert_eq!(c.diagonal_agreement(), 0);
    }

    /// The ordgraph `projections_recover_top_diagonal_band_and_rankquant_score`
    /// case: query = [0,0,1,1,2,2,3,3], doc = [0,1,1,2,2,3,3,0], nb = 4.
    #[test]
    fn parity_projections() {
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [0u8, 1, 1, 2, 2, 3, 3, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();

        assert_eq!(Projection::TopOverlap.score(&c), 1.0);
        assert_eq!(Projection::DiagonalAgreement.score(&c), 4.0);
        assert_eq!(Projection::BandAgreement { radius: 1 }.score(&c), 7.0);
        assert_eq!(Projection::TopGroupOverlap { width: 2 }.score(&c), 3.0);
        assert_eq!(Projection::BucketL1Distance.score(&c), 6.0);
        assert_eq!(Projection::RankQuantSymmetric.score(&c), 4.0);

        // Direct-accessor parity (same values, non-enum surface).
        assert_eq!(c.top_overlap(), 1);
        assert_eq!(c.diagonal_agreement(), 4);
        assert_eq!(c.band_agreement(1), 7);
        assert_eq!(c.top_group_overlap(2), 3);
        assert_eq!(c.bucket_l1_distance(), 6);
        assert_eq!(c.rankquant_symmetric_score(), 4.0);
    }

    #[test]
    fn band_agreement_saturates_on_huge_radius() {
        // `radius` is uncapped public input; a near-`usize::MAX` value must not
        // overflow `qb + radius`. It should clamp to the whole table.
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [0u8, 1, 1, 2, 2, 3, 3, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();
        assert_eq!(c.band_agreement(usize::MAX), c.total_count());
    }

    /// `bucket_l1_distance` must not overflow for constructor-accepted inputs.
    /// All mass in the maximum-distance cell at `dim = u32::MAX` (the largest
    /// accepted `dim`) gives `(nb-1)·dim`, which exceeds `u32::MAX`. Built
    /// directly because a `u32::MAX`-length input slice is impractical to
    /// allocate — this is exactly the table such an input would produce.
    #[test]
    fn bucket_l1_distance_does_not_overflow_u32() {
        let nb = 16usize;
        let mut counts = vec![0u32; nb * nb];
        counts[nb - 1] = u32::MAX; // C[0][nb-1]: query bucket 0, doc bucket nb-1
        let c = Contingency {
            buckets: nb,
            counts,
        };
        let expected = (nb as u64 - 1) * u64::from(u32::MAX); // 15 · 4_294_967_295
        assert!(
            expected > u64::from(u32::MAX),
            "fixture must exceed u32 to be a real regression"
        );
        assert_eq!(c.bucket_l1_distance(), expected);
    }

    /// The ordgraph `contingency_has_fixed_row_and_column_margins` case.
    #[test]
    fn parity_fixed_margins() {
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [0u8, 1, 1, 2, 2, 3, 3, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();

        assert_eq!(c.total_count(), 8);
        for bucket in 0..4 {
            assert_eq!(c.row_sum(bucket), 2);
            assert_eq!(c.column_sum(bucket), 2);
        }
    }

    /// The ordgraph `rankquant_symmetric_projection_matches_direct_centered_sum`
    /// case: the table-projected score equals the per-coordinate centred sum.
    #[test]
    fn parity_rankquant_matches_direct_centered_sum() {
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [0u8, 1, 1, 2, 2, 3, 3, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();

        let centre = 1.5f32;
        let direct: f32 = query
            .iter()
            .zip(doc.iter())
            .map(|(&q, &d)| (f32::from(q) - centre) * (f32::from(d) - centre))
            .sum();

        assert_eq!(c.rankquant_symmetric_score(), direct);
    }

    /// The ordgraph `coarsened_counts_preserve_total_mass` case:
    /// coarsened_counts(2) = [3, 1, 1, 3], total = 8.
    #[test]
    fn parity_coarsened_counts() {
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [0u8, 1, 1, 2, 2, 3, 3, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();

        assert_eq!(c.coarsened_counts(2), vec![3, 1, 1, 3]);
        assert_eq!(c.coarsened_counts(2).iter().sum::<u32>(), 8);
    }

    // ---- ordvec-specific surface ----------------------------------------

    /// `project` with a hand-built weight matrix recovers the named
    /// projections it generalises.
    #[test]
    fn project_generalises_named_projections() {
        let query = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let doc = [0u8, 1, 1, 2, 2, 3, 3, 0];
        let c = Contingency::new(&query, &doc, 4).unwrap();
        let nb = 4;

        // Unit-diagonal weights ⇒ diagonal_agreement.
        let mut diag = vec![0.0f32; nb * nb];
        for a in 0..nb {
            diag[a * nb + a] = 1.0;
        }
        assert_eq!(c.project(&diag), c.diagonal_agreement() as f32);

        // Centred outer-product weights ⇒ rankquant_symmetric_score.
        let centre = (nb as f32 - 1.0) / 2.0;
        let mut outer = vec![0.0f32; nb * nb];
        for a in 0..nb {
            for b in 0..nb {
                outer[a * nb + b] = (a as f32 - centre) * (b as f32 - centre);
            }
        }
        assert_eq!(c.project(&outer), c.rankquant_symmetric_score());

        // A fully custom learned matrix: weighted dot product over cells.
        let learned: Vec<f32> = (0..(nb * nb)).map(|i| i as f32 * 0.5).collect();
        let expected: f32 = (0..(nb * nb))
            .map(|i| learned[i] * c.counts()[i] as f32)
            .sum();
        assert_eq!(c.project(&learned), expected);
    }

    /// Constructor input validation.
    #[test]
    fn rejects_mismatched_lengths() {
        let err = Contingency::new(&[0u8, 1], &[0u8, 1, 2], 4).unwrap_err();
        assert!(matches!(err, OrdvecError::InvalidVectorLength { .. }));
    }

    #[test]
    fn rejects_zero_buckets() {
        let err = Contingency::new(&[0u8], &[0u8], 0).unwrap_err();
        assert!(matches!(
            err,
            OrdvecError::InvalidParameter { name: "nb", .. }
        ));
    }

    #[test]
    fn rejects_more_than_256_buckets() {
        // `nb` is a caller-supplied usize; codes are u8, so nb > 256 is rejected
        // before the nb*nb allocation (a large nb would otherwise abort the host).
        let err = Contingency::new(&[0u8], &[0u8], 300).unwrap_err();
        assert!(matches!(
            err,
            OrdvecError::InvalidParameter { name: "nb", .. }
        ));
        // 256 is the boundary and is accepted.
        assert!(Contingency::new(&[0u8], &[0u8], 256).is_ok());
    }

    #[test]
    fn rejects_out_of_range_code() {
        // doc has a code (4) >= nb (4).
        let err = Contingency::new(&[0u8, 1, 2, 3], &[0u8, 1, 2, 4], 4).unwrap_err();
        assert!(matches!(
            err,
            OrdvecError::InvalidParameter { name: "code", .. }
        ));
    }

    /// AVX-512 dispatch is gated on `dim >= 64` (the byte-compare kernel only
    /// fires once there is at least one full 64-byte block plus tail handling).
    /// Build a long, randomised case and check the SIMD-dispatched table
    /// matches an independent scalar histogram, exercising the masked tail.
    #[test]
    fn simd_path_matches_scalar_reference() {
        // Lengths chosen so at least one trips the AVX-512 path with a partial
        // tail (200 = 3*64 + 8) and one is a clean multiple of 64 (256).
        for &len in &[64usize, 200, 256, 1000] {
            for nb in [2usize, 4, 8, 16] {
                let mut seed = 0x9E3779B9u32 ^ (len as u32).wrapping_mul(2654435761);
                let mut next = || {
                    // xorshift32, deterministic, no rng dependency.
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed
                };
                let query: Vec<u8> = (0..len).map(|_| (next() as usize % nb) as u8).collect();
                let doc: Vec<u8> = (0..len).map(|_| (next() as usize % nb) as u8).collect();

                let got = Contingency::new(&query, &doc, nb).unwrap();

                // Independent scalar reference.
                let mut want = vec![0u32; nb * nb];
                for (&q, &d) in query.iter().zip(doc.iter()) {
                    want[q as usize * nb + d as usize] += 1;
                }
                assert_eq!(
                    got.counts(),
                    want.as_slice(),
                    "contingency table mismatch at len={len}, nb={nb}",
                );
                assert_eq!(got.total_count(), len as u32);
            }
        }
    }

    /// nb > 255 forces the scalar path (the SIMD kernel caps at nb <= 16) and
    /// exercises the `find_out_of_range` skip branch (all u8 codes in range).
    #[test]
    fn large_nb_uses_scalar_and_skips_range_scan() {
        // nb = 256 is the max (codes are u8) and still `> 255`, so the
        // out-of-range scan is skipped and the scalar path is taken (nb > 16).
        let query = [0u8, 5, 200, 255];
        let doc = [255u8, 200, 5, 0];
        let c = Contingency::new(&query, &doc, 256).unwrap();
        assert_eq!(c.buckets(), 256);
        assert_eq!(c.count(0, 255), 1);
        assert_eq!(c.count(255, 0), 1);
        assert_eq!(c.count(200, 5), 1);
        assert_eq!(c.count(5, 200), 1);
        assert_eq!(c.total_count(), 4);
    }
}
