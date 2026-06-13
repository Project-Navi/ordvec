> ⚠️ EXPLORATORY — UNCONTROLLED COMPARISON (adversarial review). The RandomOffset
> arm draws an extra RNG value per projection, desyncing the stream so arms get
> DIFFERENT projection directions — it was never the clean phase-only ablation
> claimed, so the "random offsets are redundant" tie is WITHDRAWN. The "fair
> envelope" is also not fair (coprime arms saturate below high budgets). What
> survives: the retreat — "coprimality across R directions is the wrong geometry;
> needs a within-axis harness." See benchmarks/ADVERSARIAL_REVIEW.md.

# R-projection shard-recall: does coprime seam-decorrelation help?

Experiment for the training-free routing layer. Five projection arms over a
random-projection ensemble; recall@10 measured against FP32 cosine top-10,
compared at EQUAL candidates-scanned (the only fair axis).

Source: `examples/shard_recall.rs`. Synthetic clustered corpus n=50k, dim=256,
200 queries, k=10, seed=1.

## Fair envelope (max recall@10 at candidates-scanned <= budget)

| budget | coprime | aligned | random-offset | both |
|--------|---------|---------|---------------|------|
| 1000   | 0.120   | 0.143   | 0.088         | 0.110 |
| 2000   | 0.214   | 0.239   | 0.218         | 0.207 |
| 4000   | 0.370   | 0.402   | 0.375         | 0.390 |
| 8000   | 0.518   | 0.674   | 0.651         | 0.554 |
| 16000  | 0.518   | 0.895   | 0.883         | 0.554 |

## Findings

**1. CLEAN RESULT (matched granularity): random offsets are redundant.**
`aligned` and `random-offset` use identical bucket width and differ only in
phase — they isolate seam decorrelation exactly. They are statistically TIED
(0.895 vs 0.883 at 16k; aligned marginally ahead). Across R *different* random
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
