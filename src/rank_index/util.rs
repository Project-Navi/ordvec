//! Shared internals for the `rank_index` family.
//!
//! - [`TopK`] is the running top-`k` collector used by every search
//!   path (full ranks, bucketed ranks, bitmap overlap).
//! - [`l2_normalise`] returns the unit-norm copy of a query vector for
//!   the asymmetric scoring path.
//!
//! Both items are `pub(super)` so they are reachable from sibling
//! modules (`index`, `quant`, `bitmap`, `multi_bucket`, `quant_kernels`)
//! but not from outside `crate::rank_index`.

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

pub(super) fn l2_normalise(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        vec![0.0; v.len()]
    } else {
        let inv = 1.0 / norm;
        v.iter().map(|&x| x * inv).collect()
    }
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
pub(super) struct TopK {
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
    pub(super) fn new(k: usize) -> Self {
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
    pub(super) fn maybe_insert(&mut self, score: f32, idx: usize) {
        if self.filled < self.k {
            self.scores[self.filled] = score;
            self.indices[self.filled] = idx as i64;
            self.filled += 1;
            if self.filled == self.k {
                self.recompute_worst();
            }
        } else {
            // Replace the worst kept entry iff the incoming
            // `(score, idx)` is strictly better to keep under the
            // `(score desc, doc_id asc)` order: a higher score, or an
            // equal score with a lower doc_id. doc_ids are unique per
            // scan, so this is a total order — the greedy eviction
            // keeps exactly the top-k set under the composite key.
            let id = idx as i64;
            let better = score > self.worst_val
                || (score == self.worst_val && id < self.worst_idx);
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
    pub(super) fn finalize_into(
        &self,
        out_scores: &mut [f32],
        out_indices: &mut [i64],
    ) {
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
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
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
