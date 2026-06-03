# Search Determinism Contract

This document states the compatibility contract for ordvec search output:
scores, ordering, tie handling, backend dispatch, and empty-result shape. It
covers the primitive retrieval surface only. It does not define distributed
merge order, replication, storage manifests, or deployment policy.

## Global Ordering Rule

For public top-k search results, ordvec orders hits by:

1. score descending;
2. row ID ascending when scores compare equal.

The row ID is the internal zero-based insertion row. Subset APIs receive row
IDs from the caller and return the same global row IDs. Duplicate candidate IDs
are scored as duplicate candidate entries and may produce duplicate hits.

`k` is clamped to the search space before result buffers are allocated. A
full-index search space is the number of indexed rows. A subset search space is
the candidate-list length. If the effective `k` is zero, or the search space is
empty, search returns an empty result shape rather than padded sentinel hits.

## Backend Scope

Backend selection must not change the documented ordering rule. Exact integer
popcount primitives are bit-exact across scalar, AVX-512, aarch64 NEON, and
wasm `simd128` implementations. Floating-point RankQuant asymmetric kernels
are checked against the scalar LUT reference with an absolute score tolerance
of `1e-4` and no relative tolerance; intentional changes to that tolerance or
to golden top-k output are compatibility-affecting and must be called out in
the PR and release notes.

Query-level parallelism may change scheduling, but each query is scored and
finalized independently. Batched APIs must match the corresponding single-query
API for the same query rows, modulo the primitive-specific tolerance stated
below. Floating-point comparison tolerances apply only to score equivalence;
the public hit order still follows the global ordering rule above.

## Primitive Contracts

| Surface | Score contract | Tie key | Backend contract |
| --- | --- | --- | --- |
| `Rank::search` | Normalized Spearman-style rank cosine. | Global row ID ascending. | Fixed scalar arithmetic per row; query parallelism does not affect per-query output. |
| `Rank::search_asymmetric` | Float query against stored ranks. | Global row ID ascending. | Fixed scalar arithmetic per row; query parallelism does not affect per-query output. |
| `RankQuant::search` | Symmetric bucketed-rank score. | Global row ID ascending. | Scalar packed-byte LUT path; query parallelism does not affect per-query output. |
| `RankQuant::search_asymmetric` | Float query against stored buckets. | Global row ID ascending. | AVX-512, AVX2, and scalar-LUT dispatch must agree with the scalar reference within the documented test tolerance and preserve top-k order for the golden fixtures. |
| `RankQuant::search_asymmetric_subset` | Same score as `RankQuant::search_asymmetric`, restricted to caller-supplied candidates. | Global row ID ascending, not candidate-list position. Duplicate candidate IDs remain duplicate entries. | Uses the same AVX-512, AVX2, or scalar dispatch as full asymmetric search over a gathered scratch buffer. |
| `Bitmap::search` | Exact `popcount(Q AND D)` as `f32`. | Global row ID ascending. | Popcount scores are integer-exact across scalar and SIMD implementations. |
| `Bitmap::top_m_candidates` | Exact `popcount(Q AND D)` candidate ordering. | Global row ID ascending. | Single-query and batched candidate APIs must return the same ordered candidates. |
| `Bitmap::search_subset` | Exact subset `popcount(Q AND D)` as `f32`. | Global row ID ascending. Duplicate candidate IDs remain duplicate entries. | Subset score kernels must agree with scalar popcount. |
| `SignBitmap::top_m_candidates` | Lowest Hamming distance, equivalently highest sign agreement. | Global row ID ascending. | Single-query and batched candidate APIs must return the same ordered candidates. |
| `SignBitmap::score_all` | Dense sign-agreement counts aligned by row ID. | Not a top-k API. | Popcount scores are integer-exact across scalar and SIMD implementations. |

## FastScan

`RankQuantFastscan` is a hidden, optional b=2 pre-ranker. It is deterministic
for a fixed index, query, and backend dispatch, and its scalar and AVX-512
FastScan kernels operate on the same quantized LUT inputs. It is not
score-equivalent to exact `RankQuant::search_asymmetric`: the global 8-bit LUT
quantization is intentional and can change scores or boundary ordering. Callers
that need exact RankQuant scores should use `RankQuant::search_asymmetric` or
`RankQuant::search_asymmetric_subset`.

## Compatibility Notes

Intentional changes to any of these are compatibility-affecting:

- golden top-k row IDs;
- tie keys or duplicate-candidate behavior;
- empty-result or `k` clamping shape;
- scalar/SIMD score tolerance;
- whether an API is exact or approximate;
- whether a backend is covered by this contract.

Such changes need a compatibility note in the PR and release notes. Performance
changes that preserve the same scores, row ordering, tie keys, and empty-result
shape are not search-contract breaks.

Compatibility note for this contract PR: `RankQuant::search_asymmetric_subset`
now breaks equal-score ties by global row ID instead of local candidate-list
position. That matches full-index search, C ABI hit ordering, Python binding
ordering, and the candidate prefilters. Duplicate candidate IDs are still
scored as duplicate entries and may still produce duplicate hits.
