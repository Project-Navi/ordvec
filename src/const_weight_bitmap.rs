//! Constant-weight bitmap overlap + the finite constant-weight null (issue #222).
//!
//! This is the *ordinal-kernel* evidence surface built on top of the
//! fixed-composition bucket codes ([`crate::bucket_code::BucketCode`], issue
//! #220). It exposes two literal constant-weight bitmaps derived from a bucket
//! code, their popcount overlap, and the idealized uniform constant-weight
//! *null* that turns an observed overlap into an exact finite tail probability.
//! It carries **no retrieval, graph, or serving concepts** — only the bitmap
//! overlap statistic and its finite combinatorial null.
//!
//! Three pieces model the contract:
//!
//! - [`ConstantWeightBitmap`] — the top-bucket membership bitmap of a bucket
//!   code as a `Vec<bool>`. Bit `j` is set iff coordinate `j` sits in the top
//!   bucket (`buckets - 1`). Its [`ConstantWeightBitmap::overlap`] is the count
//!   of shared set bits — the reference (naive) overlap statistic.
//! - [`PackedConstantWeightBitmap`] — the same membership packed into `u64`
//!   words, with [`PackedConstantWeightBitmap::overlap`] computed by word-level
//!   AND-popcount. The packed overlap routes through the crate's shared
//!   `crate::util::and_popcount` primitive (the same reduction the production
//!   [`crate::Bitmap`] scan kernels use), so a packed scan and the bitmap index
//!   compute overlap with one shared popcount path. It generalises beyond the
//!   top bucket: it can be built from any bucket range or top *group* of
//!   buckets.
//! - [`BitmapNull`] — the idealized uniform constant-weight bitmap null over
//!   all weight-`w` bitmaps in `dim` positions. The fibers of the overlap
//!   statistic partition that space, so [`BitmapNull::fiber_count`] is the
//!   hypergeometric numerator and [`BitmapNull::tail_count`] /
//!   [`BitmapNull::space_size`] give an exact upper-tail probability for an
//!   overlap cutoff.
//!
//! Ported to reach behavioural parity with the `ordgraph` bitmap prototype
//! (`ordgraph-proto/src/bitmap.rs`); the popcount reduction is *not*
//! re-implemented — it delegates to the crate's shared `crate::util`
//! primitive.
//!
//! # Overflow
//! [`choose`] (and therefore [`BitmapNull::space_size`] / `fiber_count` /
//! `tail_count`) accumulates in `u128`. gcd-cancellation keeps the running
//! product minimal, so the representable range is the full set of `(dim, weight)`
//! whose true `C(dim, weight)` fits `u128`. Beyond that the result is not
//! representable and the count **panics (fail-loud)** — in both debug and
//! release — rather than silently wrapping to a wrong value. (This is a
//! deliberate divergence from the reference prototype, which wrapped in release;
//! a public combinatorial that returns a wrong count is unacceptable for an
//! exact null.) The finite null targets the small `dim`/`weight` regime where
//! the exact count is representable; callers near the `u128` ceiling must bound
//! their parameters or pre-check.

use crate::bucket_code::BucketCode;
use crate::util::and_popcount;

/// Constant-weight top-bucket bitmap derived from an ordinal bucket code.
///
/// Bit `j` is `true` iff coordinate `j` of the code is in the top bucket
/// (`buckets - 1`). Under the fixed-composition invariant every bucket holds
/// exactly `dim / buckets` coordinates, so the bitmap has constant weight
/// `dim / buckets` across all codes of the same spec — the property the
/// constant-weight null relies on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantWeightBitmap {
    dim: usize,
    weight: usize,
    bits: Vec<bool>,
}

/// Packed constant-weight bitmap with overlap computed by word-level popcount.
///
/// The membership indicator is packed into `dim.div_ceil(64)` `u64` words.
/// [`Self::overlap`] routes through the crate's shared
/// `crate::util::and_popcount` reduction — the same AND-popcount path the
/// production [`crate::Bitmap`] scan kernels use — so a packed scan and the
/// bitmap index agree bit-for-bit on overlap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedConstantWeightBitmap {
    dim: usize,
    weight: usize,
    words: Vec<u64>,
}

impl ConstantWeightBitmap {
    /// Build the top-bucket membership bitmap of `code`.
    ///
    /// Bit `j` is set iff `code`'s coordinate `j` lands in the top bucket
    /// (`buckets - 1`), via [`BucketCode::top_bitmap`].
    pub fn from_top_bucket(code: &BucketCode) -> Self {
        let bits = code.top_bitmap();
        let weight = bits.iter().filter(|&&bit| bit).count();
        Self {
            dim: bits.len(),
            weight,
            bits,
        }
    }

    /// The bitmap dimension (number of coordinates / bits).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of set bits (constant across codes of the same spec).
    pub fn weight(&self) -> usize {
        self.weight
    }

    /// The raw boolean membership bits.
    pub fn bits(&self) -> &[bool] {
        &self.bits
    }

    /// Count of positions set in **both** bitmaps — the naive shared-set-bit
    /// overlap statistic.
    ///
    /// # Panics
    /// Panics if the two bitmaps have different dimensions (a popcount over
    /// mismatched supports is meaningless), matching the prototype's
    /// fail-loud contract.
    pub fn overlap(&self, other: &Self) -> usize {
        assert_eq!(self.dim, other.dim, "bitmap dimensions must match");
        self.bits
            .iter()
            .zip(&other.bits)
            .filter(|&(lhs, rhs)| *lhs && *rhs)
            .count()
    }
}

impl PackedConstantWeightBitmap {
    /// Pack the membership indicator for the bucket range `[start, end]`.
    ///
    /// Bit `j` is set iff `code`'s coordinate `j` lands in a bucket in the
    /// inclusive range `start_bucket..=end_bucket`.
    ///
    /// # Panics
    /// Panics if `start_bucket > end_bucket`, or if `end_bucket` is outside the
    /// code's bucket domain (`>= buckets`).
    pub fn from_bucket_range(code: &BucketCode, start_bucket: usize, end_bucket: usize) -> Self {
        assert!(start_bucket <= end_bucket, "bucket range must be ordered");
        assert!(
            end_bucket < code.spec().buckets(),
            "bucket range must fit code spec"
        );
        let dim = code.codes().len();
        let mut weight = 0usize;
        let mut words = vec![0u64; dim.div_ceil(64)];
        for (coordinate, &bucket) in code.codes().iter().enumerate() {
            let bucket = bucket as usize;
            if (start_bucket..=end_bucket).contains(&bucket) {
                weight += 1;
                words[coordinate / 64] |= 1u64 << (coordinate % 64);
            }
        }
        Self { dim, weight, words }
    }

    /// Pack the membership indicator for the top `width` buckets.
    ///
    /// Equivalent to [`Self::from_bucket_range`] over `[buckets - width,
    /// buckets - 1]`. `from_top_group(code, 1)` is the packed analogue of
    /// [`ConstantWeightBitmap::from_top_bucket`].
    ///
    /// # Panics
    /// Panics if `width == 0` or `width > buckets`.
    pub fn from_top_group(code: &BucketCode, width: usize) -> Self {
        assert!(width > 0, "top-group width must be positive");
        assert!(
            width <= code.spec().buckets(),
            "top-group width must fit code spec"
        );
        let start = code.spec().buckets() - width;
        Self::from_bucket_range(code, start, code.spec().buckets() - 1)
    }

    /// The bitmap dimension (number of coordinates).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of set bits.
    pub fn weight(&self) -> usize {
        self.weight
    }

    /// The packed `u64` membership words.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Popcount of `self AND other` — the packed overlap statistic.
    ///
    /// Routes through the crate's shared `crate::util::and_popcount`
    /// reduction (scalar `u64::count_ones` over the AND on x86_64, NEON on
    /// aarch64, simd128 on wasm), the same primitive the production
    /// [`crate::Bitmap`] scan kernels use. Equal to
    /// [`ConstantWeightBitmap::overlap`] for the same codes.
    ///
    /// # Panics
    /// Panics if the two bitmaps have different dimensions (their word counts
    /// then differ, which `and_popcount` itself rejects). The explicit `dim`
    /// check fails loud with the bitmap-specific message before the reduction.
    pub fn overlap(&self, other: &Self) -> usize {
        assert_eq!(self.dim, other.dim, "bitmap dimensions must match");
        and_popcount(&self.words, &other.words) as usize
    }
}

/// Overlap profile across a set of top-group widths.
///
/// For each `width` in `widths`, builds the packed top-`width`-group bitmaps of
/// `lhs` and `rhs` and returns their popcount overlap. The result is a vector
/// parallel to `widths`. Both codes must share the same dimension (and, for
/// `from_top_group` to be meaningful, the same spec).
pub fn top_group_overlap_vector(
    lhs: &BucketCode,
    rhs: &BucketCode,
    widths: &[usize],
) -> Vec<usize> {
    widths
        .iter()
        .map(|&width| {
            let lhs_bitmap = PackedConstantWeightBitmap::from_top_group(lhs, width);
            let rhs_bitmap = PackedConstantWeightBitmap::from_top_group(rhs, width);
            lhs_bitmap.overlap(&rhs_bitmap)
        })
        .collect()
}

/// Idealized uniform constant-weight bitmap null.
///
/// Models a uniform distribution over **all** weight-`weight` bitmaps in `dim`
/// positions (there are `C(dim, weight)` of them). The overlap of a random such
/// bitmap with a fixed weight-`weight` bitmap is hypergeometric; this type
/// exposes the exact finite counts:
///
/// - [`Self::space_size`] = `C(dim, weight)` — the total number of bitmaps.
/// - [`Self::fiber_count`] = the number of bitmaps overlapping a fixed one in
///   exactly `overlap` positions (the hypergeometric numerator).
/// - [`Self::tail_count`] = the upper-tail sum `Σ_{o>=threshold} fiber_count(o)`.
///
/// The fibers partition the space, so `Σ_{o=0..=weight} fiber_count(o) ==
/// space_size`, and `tail_count(threshold) / space_size` is the exact upper-tail
/// probability of seeing an overlap `>= threshold` under the null.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitmapNull {
    dim: usize,
    weight: usize,
}

impl BitmapNull {
    /// Build the null over weight-`weight` bitmaps in `dim` positions.
    ///
    /// # Panics
    /// Panics if `dim == 0` or `weight > dim`.
    pub fn new(dim: usize, weight: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(weight <= dim, "weight must be <= dim");
        Self { dim, weight }
    }

    /// The number of positions.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The constant bitmap weight.
    pub fn weight(&self) -> usize {
        self.weight
    }

    /// Total number of weight-`weight` bitmaps: `C(dim, weight)`.
    pub fn space_size(&self) -> u128 {
        choose(self.dim, self.weight)
    }

    /// Number of weight-`weight` bitmaps overlapping a fixed weight-`weight`
    /// bitmap in exactly `overlap` positions.
    ///
    /// This is the hypergeometric numerator
    /// `C(weight, overlap) * C(dim - weight, weight - overlap)`: choose which
    /// `overlap` of the `weight` set bits coincide, then place the remaining
    /// `weight - overlap` set bits among the `dim - weight` zero positions.
    /// Returns `0` for an infeasible `overlap` (more than `weight`, or leaving
    /// more remaining set bits than there are zero positions).
    pub fn fiber_count(&self, overlap: usize) -> u128 {
        if overlap > self.weight {
            return 0;
        }
        let outside = self.weight - overlap;
        if outside > self.dim - self.weight {
            return 0;
        }
        choose(self.weight, overlap)
            .checked_mul(choose(self.dim - self.weight, outside))
            .expect("fiber count overflows u128")
    }

    /// Upper-tail count `Σ_{o>=threshold} fiber_count(o)`.
    ///
    /// `tail_count(0) == space_size` (every bitmap overlaps in `>= 0`
    /// positions), and `tail_count(threshold) == 0` for `threshold > weight`
    /// (no bitmap overlaps a weight-`weight` bitmap in more than `weight`
    /// positions). Monotone non-increasing in `threshold`. Divide by
    /// [`Self::space_size`] for the exact upper-tail probability.
    pub fn tail_count(&self, threshold: usize) -> u128 {
        if threshold == 0 {
            return self.space_size();
        }
        if threshold > self.weight {
            return 0;
        }
        (threshold..=self.weight)
            .map(|overlap| self.fiber_count(overlap))
            .sum()
    }
}

/// Binomial coefficient `C(n, k)` in `u128`.
///
/// Returns `0` for `k > n`. Uses the symmetric `k.min(n - k)` factor count and
/// an exact multiply-then-divide recurrence, with gcd-cancellation of each
/// `(n - i)/(i + 1)` factor to keep the running product as small as possible
/// before each step. The multiply is `checked_mul`: if the true `C(n, k)`
/// exceeds `u128::MAX` this **panics** (fail-loud) rather than silently wrapping
/// to a wrong count. See the module-level Overflow note.
pub fn choose(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc = 1u128;
    for i in 0..k {
        let num = (n - i) as u128;
        let den = (i + 1) as u128;
        // Cancel the shared factor first: this both shrinks the intermediate
        // product (extending the representable range) and keeps the division
        // exact — `den / g` is coprime to `num / g`, and the result `C(n, i+1)`
        // is integral, so `den / g` divides `acc`.
        let g = gcd(num, den);
        acc = (acc / (den / g))
            .checked_mul(num / g)
            .expect("binomial coefficient C(n, k) overflows u128");
    }
    acc
}

/// Greatest common divisor (Euclid), for the exact binomial cancellation above.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket_code::{BucketCode, CompositionSpec};

    /// Build a `dim`-length, 4-bucket code from raw bucket ids.
    fn code(values: &[u8]) -> BucketCode {
        BucketCode::new(
            CompositionSpec::new(values.len(), 4).unwrap(),
            values.to_vec(),
        )
        .unwrap()
    }

    /// Naive shared-set-bit overlap over two packed bitmaps — the independent
    /// reference both `overlap` implementations must match. Replaces the
    /// prototype's `Contingency::top_overlap` cross-check (a retrieval/graph
    /// concept deliberately not ported into ordvec under #222); the prototype's
    /// literal expected values are reproduced verbatim below.
    fn naive_packed_overlap(
        a: &PackedConstantWeightBitmap,
        b: &PackedConstantWeightBitmap,
    ) -> usize {
        a.words()
            .iter()
            .zip(b.words())
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum()
    }

    // ---- ordgraph bitmap parity gate ------------------------------------
    // Assertion values reproduced verbatim from the reference
    // `ordgraph-proto/src/bitmap.rs` #[cfg(test)] module. The prototype
    // cross-checked the overlap against `Contingency::top_overlap`; here the
    // same literal (1, and [1, 3, 8]) is asserted directly and cross-checked
    // against the naive shared-set-bit count instead.

    #[test]
    fn top_bitmap_has_expected_constant_weight() {
        let code = code(&[0, 0, 1, 1, 2, 2, 3, 3]);
        let bitmap = ConstantWeightBitmap::from_top_bucket(&code);

        assert_eq!(bitmap.dim(), 8);
        assert_eq!(bitmap.weight(), 2);
    }

    #[test]
    fn top_overlap_matches_naive_top_top_count() {
        let query = code(&[0, 0, 1, 1, 2, 2, 3, 3]);
        let doc = code(&[0, 1, 1, 2, 2, 3, 3, 0]);
        let query_bitmap = ConstantWeightBitmap::from_top_bucket(&query);
        let doc_bitmap = ConstantWeightBitmap::from_top_bucket(&doc);

        // Prototype literal: top-top overlap is 1.
        assert_eq!(query_bitmap.overlap(&doc_bitmap), 1);
    }

    #[test]
    fn packed_top_overlap_matches_naive_top_top_count() {
        let query = code(&[0, 0, 1, 1, 2, 2, 3, 3]);
        let doc = code(&[0, 1, 1, 2, 2, 3, 3, 0]);
        let query_bitmap = PackedConstantWeightBitmap::from_top_group(&query, 1);
        let doc_bitmap = PackedConstantWeightBitmap::from_top_group(&doc, 1);

        assert_eq!(query_bitmap.dim(), 8);
        assert_eq!(query_bitmap.weight(), 2);
        // Prototype literal: top-top overlap is 1.
        assert_eq!(query_bitmap.overlap(&doc_bitmap), 1);
        assert_eq!(
            query_bitmap.overlap(&doc_bitmap),
            naive_packed_overlap(&query_bitmap, &doc_bitmap)
        );
    }

    #[test]
    fn top_group_overlap_vector_uses_popcount_backed_bitmaps() {
        let query = code(&[0, 0, 1, 1, 2, 2, 3, 3]);
        let doc = code(&[0, 1, 1, 2, 2, 3, 3, 0]);

        // Prototype literal.
        assert_eq!(
            top_group_overlap_vector(&query, &doc, &[1, 2, 4]),
            [1, 3, 8]
        );
    }

    #[test]
    fn bitmap_null_fibers_sum_to_space_size() {
        let null = BitmapNull::new(10, 3);
        let fiber_sum: u128 = (0..=3).map(|overlap| null.fiber_count(overlap)).sum();

        assert_eq!(fiber_sum, choose(10, 3));
        assert_eq!(null.space_size(), choose(10, 3));
    }

    #[test]
    fn bitmap_tail_counts_have_boundary_values_and_are_monotone() {
        let null = BitmapNull::new(10, 3);

        assert_eq!(null.tail_count(0), choose(10, 3));
        assert_eq!(null.tail_count(4), 0);
        assert!(null.tail_count(2) <= null.tail_count(1));
        assert!(null.tail_count(3) <= null.tail_count(2));
    }

    // ---- ordvec-specific correctness surface ----------------------------

    #[test]
    fn null_fibers_partition_space_for_several_params() {
        // The fibers of the overlap statistic partition the whole space, so
        // their counts must sum to C(dim, weight) for every (dim, weight).
        for (dim, weight) in [(8, 2), (10, 3), (16, 4), (20, 5), (32, 8), (5, 0), (5, 5)] {
            let null = BitmapNull::new(dim, weight);
            let fiber_sum: u128 = (0..=weight).map(|o| null.fiber_count(o)).sum();
            assert_eq!(
                fiber_sum,
                null.space_size(),
                "fibers must partition the space for (dim={dim}, weight={weight})"
            );
        }
    }

    #[test]
    fn overlap_parity_const_vs_packed_vs_naive() {
        // The three overlap definitions — bool-bitmap shared-set-bit count,
        // packed AND-popcount (via util::and_popcount), and the standalone
        // naive packed reference — must all agree for the same codes, across
        // every top-group width.
        let query = code(&[0, 0, 1, 1, 2, 2, 3, 3]);
        let doc = code(&[3, 2, 1, 0, 0, 1, 2, 3]);

        for width in 1..=4 {
            let packed_q = PackedConstantWeightBitmap::from_top_group(&query, width);
            let packed_d = PackedConstantWeightBitmap::from_top_group(&doc, width);
            let packed_overlap = packed_q.overlap(&packed_d);
            let naive = naive_packed_overlap(&packed_q, &packed_d);
            assert_eq!(packed_overlap, naive, "packed vs naive at width {width}");

            if width == 1 {
                // Width 1 is exactly the top bucket — the bool bitmap path.
                let const_q = ConstantWeightBitmap::from_top_bucket(&query);
                let const_d = ConstantWeightBitmap::from_top_bucket(&doc);
                assert_eq!(
                    const_q.overlap(&const_d),
                    packed_overlap,
                    "bool vs packed at the top bucket"
                );
            }
        }
    }

    #[test]
    fn packed_overlap_handles_multi_word_dim() {
        // dim = 128 spans two u64 words, exercising the shared and_popcount
        // reduction across word boundaries. A 4-bucket code over 128 coords
        // puts 32 coordinates in the top bucket; overlapping a code with
        // itself yields exactly its weight.
        let values: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        let code = BucketCode::new(CompositionSpec::new(128, 4).unwrap(), values).unwrap();
        let bitmap = PackedConstantWeightBitmap::from_top_group(&code, 1);
        assert_eq!(bitmap.dim(), 128);
        assert_eq!(bitmap.words().len(), 2);
        assert_eq!(bitmap.weight(), 32);
        assert_eq!(bitmap.overlap(&bitmap), 32);
    }

    #[test]
    fn choose_matches_known_small_binomials() {
        assert_eq!(choose(0, 0), 1);
        assert_eq!(choose(5, 0), 1);
        assert_eq!(choose(5, 5), 1);
        assert_eq!(choose(5, 2), 10);
        assert_eq!(choose(10, 3), 120);
        assert_eq!(choose(6, 3), 20);
        assert_eq!(choose(52, 5), 2_598_960);
        // k > n is empty.
        assert_eq!(choose(3, 4), 0);
    }

    #[test]
    fn choose_is_symmetric() {
        for n in 0..=30usize {
            for k in 0..=n {
                assert_eq!(
                    choose(n, k),
                    choose(n, n - k),
                    "C({n},{k}) == C({n},{})",
                    n - k
                );
            }
        }
    }

    #[test]
    fn choose_extends_range_via_gcd_cancellation() {
        // C(128, 64) fits u128 but the naive multiply-then-divide recurrence
        // overflows the intermediate product; gcd-cancellation computes it.
        // Validate via Pascal's identity (no huge literal): C(n,k)=C(n-1,k-1)+C(n-1,k).
        assert_eq!(choose(128, 64), choose(127, 63) + choose(127, 64));
        assert!(choose(128, 64) > 0);
    }

    #[test]
    #[should_panic(expected = "overflows u128")]
    fn choose_panics_fail_loud_on_overflow() {
        // C(300, 150) is far beyond u128::MAX: fail loud, never wrap to a wrong count.
        let _ = choose(300, 150);
    }

    #[test]
    fn fiber_count_zero_outside_feasible_overlap() {
        let null = BitmapNull::new(10, 3);
        // An overlap larger than the weight is impossible.
        assert_eq!(null.fiber_count(4), 0);
        // Exactly the weight: all set bits coincide — there is exactly one such
        // bitmap (the fixed one itself).
        assert_eq!(null.fiber_count(3), 1);
    }

    #[test]
    fn tail_probability_is_well_formed() {
        // tail_count(0) / space_size == 1; the tail at every threshold is a
        // valid fraction of the space.
        let null = BitmapNull::new(16, 4);
        let space = null.space_size();
        assert_eq!(null.tail_count(0), space);
        for threshold in 0..=5 {
            assert!(null.tail_count(threshold) <= space);
        }
    }
}
