# Alternatives considered

This document is a concise record of design alternatives we evaluated for
the rank-mode index types and the batched bitmap kernel, and chose not to
include in this contribution. The intent is to give maintainers visibility
into the search space without requiring them to review the dead branches.

Full code and design specs for each item below live in the paper-repo
archive at `turbovec-arxiv/archive/turbovec-dev-history/`. Anything here
can be brought into scope by a maintainer request; everything here was
removed for scope rather than correctness reasons.

## Trimmed index types

| Name | When | Result | Why excluded | Archive path |
|------|------|--------|--------------|--------------|
| **Count-fold summary tier** (`CountFoldBitmapIndex`) | 2026-05-18 | Null on real embeddings | Pointwise upper bound on bitmap-overlap was too loose to preserve useful candidate recall at the tested α / latency budget; no favourable R@10 / latency Pareto vs the shipped batched-bitmap baseline | `count_fold.rs`, `SPEC_COUNT_FOLD_TIER.md` |

## Trimmed kernel-level alternatives

| Name | When | Status | Why excluded | Archive path |
|------|------|--------|--------------|--------------|
| **Pulp-based bitmap scan** (formerly `pulp-kernel` feature) | 2026-05-15 | Prototype, never landed | The existing hand-rolled AVX-512 VPOPCNTDQ path already retires one popcount per cycle on Zen 5; we never produced a head-to-head benchmark showing a portable-SIMD rewrite improves on it. The `pulp-kernel` cargo feature was removed prior to upstream submission along with the optional `pulp` + `bytemuck` deps it gated. | `bitmap_pulp_prototype/` (WIP commit `3e14277` in branch history) |
| **Harley-Seal CSA popcount aggregation** | 2026-05-18 | Shelved after self-audit | Profile indicated the batched bitmap path was bandwidth-bound, not popcount-retire-bound; expected gain too small for upstream scope | `SPEC_HARLEY_SEAL_CSA.md` |
| **Threshold-seeded bit-slice bound** | 2026-05-18 | Shelved after self-audit | Added per-doc / per-qword control overhead likely exceeded pruning savings at realistic θ on dense embedding distributions | `SPEC_THRESHOLD_SEEDING.md` |

## Trimmed bench-suite modes

The previous `examples/bench_rank.rs` exposed a larger set of `--mode`
flags. We trimmed the ones below because each mode added a knob without
supporting a claim in the public docs. The current set is:
`bitmap`, `batched-two-stage`, `batch-sweep`, `sign-headline`,
`storage-matched`.

| Removed mode | What it ran | Reason |
|--------------|-------------|--------|
| `bitmap-pulp` | `pulp-kernel`-feature variant of the single-query bitmap scan | Prototype; see Pulp-based bitmap scan above. Both the bench mode and the underlying feature flag are gone from the upstream-PR tree. |
| `count-fold-two-stage` | α sweep over the count-fold survivor tier | Null result; see Count-fold summary tier above |
| `--kind {full16x16, diag, top4}` filters of the default-suite multi-bucket b=4 probe | Bilinear bucket-overlap weight schemes used for research-side QPP feature extraction (paper-only, not user-facing) | Not tied to any RANK_MODES.md / README claim; the `MultiBucketBitmapIndex` type itself remains exposed for direct construction by callers who need it |

## What this archive does NOT cover

Items above are the **upstream-scope trims**. The full paper-side
research history (experiment writeups, literature briefs, oracle
features, QPP three-tier work, distillation experiments, etc.) lives
entirely in the paper repository at
`turbovec-arxiv/archive/turbovec-dev-history/`. None of it is needed
to use or maintain `turbovec` — it explains *why* we arrived at the
shipped design, not how the shipped design works.

For an end-user view of the rank-mode types, see
[`RANK_MODES.md`](RANK_MODES.md).
