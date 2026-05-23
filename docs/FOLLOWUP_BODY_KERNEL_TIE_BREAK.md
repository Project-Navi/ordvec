# Follow-up: deterministic tie-breaking for body bitmap candidate selection

`Bitmap::top_m_candidates` and `top_m_candidates_batched`
(in `src/bitmap.rs`) currently partition on
bitmap overlap score alone. Boundary ties are not rare — overlap
scores are small integers (`0..n_top`, e.g. `0..256`), so multiple
docs frequently share the cutoff score, and `select_nth_unstable_by`
may then choose different equal-scored docs at the boundary across
runs or dispatch paths.

**Fix**: add composite-key ordering `(score desc, doc_id asc)` to
both the partition predicate (`select_nth_unstable_by`) and the
post-partition sort (`sort_unstable_by`), so the candidate set at any
given M is fully determined by `(score, doc_id)`.

```rust
let mut cmp = |&a: &u32, &b: &u32| {
    scores[b as usize]
        .cmp(&scores[a as usize])
        .then_with(|| a.cmp(&b))
};
idx.select_nth_unstable_by(m_eff - 1, &mut cmp);
idx[..m_eff].sort_unstable_by(&mut cmp);
```

**Keep it as a standalone change.** Rolling the determinism fix into
an unrelated benchmark or kernel change would muddy attribution — if
recall/latency numbers move, it should be clear whether the kernel
changed or only the tie-break at the candidate-set boundary changed.
The fix is behaviour-preserving on score ordering and only pins the
boundary, so it is safe to land on its own.
