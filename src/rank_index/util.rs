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
/// seen so far and the index of the current minimum. `maybe_insert`
/// is O(k) worst-case (k-element scan after each replacement) and the
/// common path — score below current minimum — is O(1). No allocation
/// per document, no full-N partial sort.
pub(super) struct TopK {
    k: usize,
    scores: Vec<f32>,
    indices: Vec<i64>,
    filled: usize,
    min_pos: usize,
    min_val: f32,
}

impl TopK {
    pub(super) fn new(k: usize) -> Self {
        Self {
            k,
            scores: vec![f32::NEG_INFINITY; k],
            indices: vec![-1; k],
            filled: 0,
            min_pos: 0,
            min_val: f32::INFINITY,
        }
    }

    #[inline]
    pub(super) fn maybe_insert(&mut self, score: f32, idx: usize) {
        if self.filled < self.k {
            self.scores[self.filled] = score;
            self.indices[self.filled] = idx as i64;
            self.filled += 1;
            if self.filled == self.k {
                self.recompute_min();
            }
        } else if score > self.min_val {
            self.scores[self.min_pos] = score;
            self.indices[self.min_pos] = idx as i64;
            self.recompute_min();
        }
    }

    fn recompute_min(&mut self) {
        let mut mv = f32::INFINITY;
        let mut mp = 0;
        for i in 0..self.filled {
            let s = self.scores[i];
            if s < mv {
                mv = s;
                mp = i;
            }
        }
        self.min_val = mv;
        self.min_pos = mp;
    }

    /// Drain into `out_scores` / `out_indices` sorted by score
    /// descending. `out_scores.len()` is the user-requested `k`;
    /// positions beyond `self.filled` are left as sentinels.
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
        pairs.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
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
