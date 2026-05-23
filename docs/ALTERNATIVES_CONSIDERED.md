# Alternatives considered

This document is a concise record of design alternatives evaluated for
the rank-mode index types and the batched bitmap kernel, and chosen
not to include. The intent is to give maintainers visibility into the
search space without requiring them to review the dead ends.

Everything listed here was removed for scope rather than correctness
reasons; none of it is present in the current tree.

## Trimmed index types

| Name | When | Result | Why excluded |
|------|------|--------|--------------|
| **Count-fold summary tier** (`CountFoldBitmapIndex`) | 2026-05-18 | Null on real embeddings | Pointwise upper bound on bitmap-overlap was too loose to preserve useful candidate recall at the tested α / latency budget; no favourable R@10 / latency Pareto vs the shipped batched-bitmap baseline |

## Trimmed kernel-level alternatives

| Name | When | Status | Why excluded |
|------|------|--------|--------------|
| **Pulp-based bitmap scan** (formerly a `pulp-kernel` feature) | 2026-05-15 | Prototype, never landed | The existing hand-rolled AVX-512 VPOPCNTDQ path already retires one popcount per cycle on Zen 5; no head-to-head benchmark showed a portable-SIMD rewrite improving on it. The `pulp-kernel` cargo feature was removed along with the optional `pulp` + `bytemuck` deps it gated. |
| **Harley-Seal CSA popcount aggregation** | 2026-05-18 | Shelved after self-audit | Profile indicated the batched bitmap path was bandwidth-bound, not popcount-retire-bound; expected gain too small to justify the complexity |
| **Threshold-seeded bit-slice bound** | 2026-05-18 | Shelved after self-audit | Added per-doc / per-qword control overhead likely exceeded pruning savings at realistic θ on dense embedding distributions |

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
| `--kind {full16x16, diag, top4}` filters of the default-suite multi-bucket b=4 probe | Bilinear bucket-overlap weight schemes used for research-side QPP feature extraction (paper-only, not user-facing) | Not tied to any RANK_MODES.md / README claim; the `MultiBucketBitmap` type itself remains exposed for direct construction by callers who need it |

## Scope note

The items above are the design trims relevant to the rank-mode
contribution. They are recorded here so the shipped design's
boundaries are legible — they explain *why* the current set of types
and kernels looks the way it does, not how the shipped design works.

For an end-user view of the rank-mode types, see
[`RANK_MODES.md`](RANK_MODES.md).
