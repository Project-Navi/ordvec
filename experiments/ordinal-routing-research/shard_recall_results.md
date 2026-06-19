> ✅ RNG-DESYNC BUG FIXED (was Bug L). build_projs now seeds direction and phase
> RNGs separately and identically across arms, so aligned vs random-offset share
> the SAME R projection directions and differ ONLY in phase — the clean ablation.
> Re-run result: aligned 0.9095 vs random-offset 0.9080 at 16k budget = TIED, so
> "random phase offsets add nothing across R different directions" now holds on a
> controlled comparison. CAVEAT STILL OPEN: the coprime/both arms subdivide
> buckets so the "fair envelope" undersells them (they saturate below high
> budgets); coprimality across R directions remains the wrong geometry — the
> within-axis vernier harness (crt_seam_oracle covers the theory) is the right
> test and is not built here. Numbers below are the post-fix run.

# R-projection shard-recall: does coprime seam-decorrelation help?

Experiment for the training-free routing layer. Five projection arms over a
random-projection ensemble; recall@10 measured against FP32 cosine top-10,
compared at EQUAL candidates-scanned (the only fair axis).

Source: `examples/shard_recall.rs`. Synthetic clustered corpus n=50k, dim=256,
200 queries, k=10, seed=1.

## Fair envelope (max recall@10 at candidates-scanned <= budget)

Post-fix run (Bug L fixed: arms share identical projection directions):

| budget | coprime | aligned | random-offset | both |
|--------|---------|---------|---------------|------|
| 1000   | 0.109   | 0.0885  | 0.0835        | 0.1085 |
| 2000   | 0.1885  | 0.1795  | 0.1840        | 0.1830 |
| 4000   | 0.3370  | 0.3780  | 0.3880        | 0.3290 |
| 8000   | 0.5315  | 0.6425  | 0.6490        | 0.5300 |
| 16000  | 0.5315  | 0.9095  | 0.9080        | 0.5300 |

## Findings

**1. CLEAN RESULT (controlled ablation): random offsets are redundant.**
After the Bug-L fix, `aligned` and `random-offset` share the SAME R projection
directions and identical bucket width — differing ONLY in phase. They are
statistically TIED (0.9095 vs 0.9080 at 16k). Across R *different* random
projection directions, the direction randomness already decorrelates seams;
adding random phase buys nothing. Confirms the literature prediction
(multi-probe LSH / random-rotation decorrelation).

**2. CONFOUND (do NOT read as "coprime is worse"):** the coprime/both arms
subdivide each projection by a distinct prime (W/2..W/53), so their grids are
mostly tiny — they scan few candidates and plateau at ~0.52 due to collapsed
bucket VOLUME, not seam structure. This is a parameterization flaw.

**3. GEOMETRY MISMATCH (the real lesson):** coprime PERIODS are a
within-single-axis stacked-grid (vernier) effect. Across R different random
directions there is only ONE grid per axis, so coprimality of periods across
directions is not even the right test. The vernier idea, to be tested properly,
needs multiple coprime-period grids overlaid on ONE projection direction — a
different harness (TODO).

## Verdict

For the multi-direction routing layer: **plain R random projections with a
single shared grid width are as good as any seam-decorrelation scheme tried
here.** Coprimality adds nothing in this geometry and is only potentially
meaningful in the within-axis stacked-grid setup, which this experiment does
not test. Recommendation: build the oblivious router with R shared-width
random-projection grids; do not invest in coprime periods unless the within-axis
vernier experiment shows a surprise.

Next: (a) within-axis vernier test (coprime grids on one direction);
(b) re-run on real embeddings via --corpus-npy;
(c) add a quantile-k-means control arm to size the train-free vs trained gap.

Reproduce: `cargo run --release --example shard_recall`
