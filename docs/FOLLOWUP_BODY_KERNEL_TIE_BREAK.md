# Resolved: deterministic tie-breaking for body bitmap candidate selection

`Bitmap::top_m_candidates` and `top_m_candidates_batched` now partition and
sort by the composite key `(score desc, doc_id asc)`. Boundary ties are not
rare because overlap scores are small integers (`0..n_top`, e.g. `0..256`), so
the candidate set at the cutoff must be fully determined by score and row ID.

The fixed comparator is:

```rust
let mut cmp = |&a: &u32, &b: &u32| {
    scores[b as usize]
        .cmp(&scores[a as usize])
        .then_with(|| a.cmp(&b))
};
idx.select_nth_unstable_by(m_eff - 1, &mut cmp);
idx[..m_eff].sort_unstable_by(&mut cmp);
```

The broader search-output policy is now tracked in
[`determinism.md`](determinism.md). Future changes to golden row IDs, tie keys,
or duplicate-candidate behavior need an explicit compatibility note.
