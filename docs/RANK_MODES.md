# Rank-cosine index modes for turbovec

**Structured synthetic benchmark: RankQuant wins recall and build speed,
loses current query latency.**

> Branch: `nelson/rank-modes`. Status: v1 prototype, scalar kernels.
> Real-embedding rerun required before any of these numbers become
> external claims.

Early structured-synthetic benchmarks suggest a rank-native mode may
be valuable for anisotropic embedding distributions: RankQuant builds
24-62× faster than TurboQuant and preserves substantially more R@10
at matched storage on a low-rank clustered corpus. Query latency is
currently slower because the RankQuant scorer is scalar exact-scan;
the next implementation step is an allocation-free LUT/SIMD kernel
with running top-k (the kernel already maintains running top-k and
uses no per-doc allocations; what remains is the SIMD lowering).
Results should be repeated on real embedding corpora before broad
claims.

This branch adds two new index types alongside `TurboQuantIndex`:

- **`RankIndex`** — stores the dimension-wise rank transform of each
  document as `u16` (`2 * dim` bytes per document).
- **`RankQuantIndex`** — buckets each rank into `1 << bits` equal-width
  bins on `[0, dim)` and packs `bits` bits per coordinate
  (`dim * bits / 8` bytes per document). Supported `bits ∈ {1, 2, 4}`.

Both expose `search` (symmetric: rank-vs-rank, Spearman correlation)
and `search_asymmetric` (FP32 query against rank-stored documents).

The construction has no rotation matrix, no codebook, no Lloyd-Max
training, no per-document norms (the L2 norm of a permutation of
`{0..D-1}` is analytical). Encode is a single `argsort` pass per
vector, with the option to bucket and pack into `B` bits per
coordinate.

## Headline numbers

Bench: `cargo run --release -p turbovec --example bench_rank --
--dim 1024 --n 50000 --queries 200 --clusters 200 --latent 64`

Setup: D=1024, N=50,000 documents, 200 queries, k=10. Low-rank
clustered corpus (200 cluster prototypes, latent_dim=64, projected to
D=1024 with N(0,1) noise = 0.3 for docs, 0.1 for queries). Ground
truth: FP32 brute-force cosine top-10. 32-core Linux x86_64, release
build with `RUSTFLAGS="-l openblas"`.

| mode               | bytes/vec | encode v/s | p50 ms | GiB/s | ns/dim | R@10  |
|--------------------|-----------|------------|--------|-------|--------|-------|
| TurboQuant b=2     | 256       | 44,802     | 0.51   | 23.5  | 0.010  | 0.299 |
| TurboQuant b=4     | 512       | 18,552     | 1.17   | 20.4  | 0.023  | 0.492 |
| RankIndex sym      | 2048      | 1,065,560  | 25.0   | 3.8   | 0.489  | 0.874 |
| RankIndex asym     | 2048      | 1,065,560  | 25.8   | 3.7   | 0.504  | 0.911 |
| RankQuant b=2 sym  | 256       | 1,186,263  | 19.1   | 0.63  | 0.372  | 0.617 |
| **RankQuant b=2 asym** | **256** | **1,186,263** | 18.9 | 0.63  | 0.368  | **0.722** |
| RankQuant b=4 sym  | 512       | 1,142,377  | 19.3   | 1.23  | 0.377  | 0.849 |
| **RankQuant b=4 asym** | **512** | **1,142,377** | 19.5 | 1.22  | 0.381  | **0.889** |
| RankQuant b=1 sym  | 128       | 1,254,733  | 18.4   | 0.32  | 0.359  | 0.407 |
| RankQuant b=1 asym | 128       | 1,254,733  | 18.4   | 0.33  | 0.359  | 0.525 |

## What survived the head-to-head

### Encode throughput: 24-62× faster

TurboQuant b=2 encodes at 44,802 vec/s; RankQuant b=2 at 1,186,263
vec/s — **26.5× faster**. The architectural reason is straightforward:
no rotation matrix multiply, no Lloyd-Max codebook fit, no per-vector
norm storage. Encode is one `argsort` per coordinate + one bucket-pack
pass per document.

### Storage: identical at matched bit width

`bytes_per_vec = dim * bits / 8` for both schemes. The byte budget is
the same lever. The implementation differs in what each byte
*means*: TurboQuant stores a quantised magnitude, RankQuant stores a
bucketed rank.

### Asymmetric beats symmetric consistently

| mode  | sym R@10 | asym R@10 | Δ      |
|-------|---------:|----------:|-------:|
| Rank full | 0.874 | 0.911 | +0.037 |
| RankQuant b=4 | 0.849 | 0.889 | +0.040 |
| RankQuant b=2 | 0.617 | 0.722 | +0.105 |
| RankQuant b=1 | 0.407 | 0.525 | +0.118 |

The asymmetric variant keeps the query side as full FP32 — the
encoder's output is consumed directly, only the document side loses
precision. This is the operating point the paper recommends and it
reproduces here. The advantage grows as document-side precision
shrinks.

### Recall at matched bytes (this corpus only): RankQuant > TurboQuant

| bytes/vec | TurboQuant | RankQuant asym | Δ      |
|-----------|------------|----------------|--------|
| 256       | 0.299      | 0.722          | +0.423 |
| 512       | 0.492      | 0.889          | +0.397 |

**This number is not externally quotable.** TurboQuant's
data-oblivious quantization assumes isotropy on the unit sphere; the
random rotation is the mechanism that pushes anisotropic data toward
an isotropic representation. The low-rank clustered corpus used here
(`latent_dim=64` in `dim=1024`) is deliberately anisotropic to mirror
real-embedding structure, which is the regime where TurboQuant's
oblivious assumption is most strained. The +42 point gap is partly
fair comparison and partly distribution mismatch.

What the gap *does* say credibly: coordinate order survives
anisotropic low-rank structure better than TurboQuant's current
magnitude quantization under this setup. Whether that result
transfers to real learned embeddings is the open question — and the
next benchmark to run.

The rank transform is distribution-agnostic by construction (it only
reads within-vector coordinate order). If real-embedding evaluation
preserves any of this gap, the operational claim is that ordinal
structure is a more recall-faithful target than magnitude structure
for these byte budgets.

## Where TurboQuant still wins

### Query latency: 17-37× faster

| bytes/vec | TurboQuant p50 | RankQuant asym p50 | gap   |
|-----------|----------------|---------------------|-------|
| 256       | 0.51 ms        | 18.9 ms             | 37×   |
| 512       | 1.17 ms        | 19.5 ms             | 17×   |

This is **scalar autovectorisation vs hand-tuned NEON/AVX2**, not an
algorithmic disadvantage of rank-cosine. TurboQuant ships
architecture-specific kernels (NEON intrinsics for ARM,
FAISS-style perm0-interleaved AVX2 for x86) with calibrated 8-bit
LUTs. The scan kernels in `rank_index.rs` are portable Rust with
inline byte unpack — the compiler autovectorises pieces but not the
LUT lookup chain.

To close this gap to ≤2-3× requires:

1. **Per-query 8-bit LUT calibration.** Compute `min`/`max` of the
   per-coordinate LUT, scale to `u8`, scan with SIMD u8 lookups, then
   undo the scale once at the end. This is exactly the TurboQuant
   pattern in `pack.rs` + `search.rs`.
2. **AVX2 / NEON scan kernels.** The 2-bit asymmetric scan maps to
   `_mm256_permutevar8x32_epi32` (or NEON `vqtbl1q_u8`) over packed
   nibbles, with running u16 accumulators and a periodic flush to
   f32. The existing TurboQuant search kernels in `search.rs` are a
   direct template.
3. **SIMD-blocked layout.** Re-use `pack.rs::repack` (or a slimmer
   rank-specific equivalent) so the scan reads contiguous lanes
   across 32-document blocks.

None of these are research questions; they are 1-2 weeks of
implementation work modelled on `search.rs`. The v1 kernel correctness
is verified by `tests/rank_index.rs` against a scalar reference.

## API parity with `TurboQuantIndex`

| capability | TurboQuant | Rank | RankQuant |
|---|---|---|---|
| `new(dim, bits)` | ✓ | `new(dim)` | ✓ |
| `add(&[f32])` | ✓ | ✓ | ✓ |
| `search(&[f32], k)` | ✓ | ✓ symmetric | ✓ symmetric |
| `search_asymmetric(&[f32], k)` | — | ✓ | ✓ |
| `swap_remove(idx)` | ✓ | ✓ | ✓ |
| `len`/`is_empty`/`dim`/`bytes_per_vec`/`byte_size` | ✓ | ✓ | ✓ |
| `write`/`load` | ✓ | ✗ (v1 follow-up) | ✗ (v1 follow-up) |
| `prepare` | ✓ | — (no lazy caches) | — |
| IdMap wrapping | ✓ | ✗ (v1 follow-up) | ✗ (v1 follow-up) |

The missing pieces (`write`/`load`, `IdMapIndex` integration) are
mechanical follow-ups that mirror the existing TurboQuant equivalents
in `io.rs` and `id_map.rs`. They were skipped from v1 to keep the
diff focused on the kernel + benchmark story.

## Test coverage

`cargo test -p turbovec --lib rank::` — 10 unit tests for the
primitives (rank transform vs numpy reference, pack/unpack round-trip
at B=1, 2, 4, bucket-centre symmetry, analytical norms match direct
computation).

`cargo test -p turbovec --test rank_index` — 9 integration tests:

- `rank_index_symmetric_matches_reference` — RankIndex.search matches
  a scalar Spearman implementation on a 256-doc / 128-dim corpus,
  exact top-10 ordering, score agreement to 1e-4.
- `rank_index_asymmetric_matches_reference` — same, for the FP32-vs-
  rank kernel.
- `rankquant_asymmetric_matches_reference_b{1,2,4}` — RankQuant
  asymmetric agrees with the scalar reference at every bit width.
- `rankquant_b2_recovers_planted_neighbour_in_top_10` — 50 queries
  each constructed by adding noise to a known corpus doc; RankQuant-2
  asymmetric recovers the planted doc in top-10 at recall ≥ 0.95.
- `rank_index_recall_at_10_matches_fp32` — rank-cosine and raw FP32
  cosine top-10 sets overlap ≥ 70% on smooth random data at D=128.
- `rank{,quant}_swap_remove_keeps_state_consistent` — `swap_remove`
  is byte-exact across the storage buffer.

## Reproducibility

```bash
git checkout nelson/rank-modes
cargo test -p turbovec --lib rank::                  # unit tests
cargo test -p turbovec --test rank_index              # integration
RUSTFLAGS="-l openblas" cargo run --release -p turbovec --example bench_rank \
    -- --dim 1024 --n 50000 --queries 200
```

The `RUSTFLAGS="-l openblas"` shim is needed on Linux for the
TurboQuant rotation step; rank modes themselves do not depend on
BLAS.

## Next step before upstreaming: real-embedding rerun

The benchmark above uses a synthetic low-rank clustered corpus. The
TurboQuant repo benchmarks on GloVe d=200 and OpenAI d=1536 / d=3072
(see `benchmarks/results/recall_*.json`). The pre-condition for any
externally-quoted claim is reproducing the head-to-head on at least
one of those real-embedding corpora at matched bytes.

The table that decides whether this graduates from prototype to
serious vector-index story:

```text
mode                 bytes/vec   encode v/s   p50 ms   p99 ms   R@10
TurboQuant b=2       256         ?            ?        ?        ?
RankQuant b=2 asym   256         ?            ?        ?        ?
TurboQuant b=4       512         ?            ?        ?        ?
RankQuant b=4 asym   512         ?            ?        ?        ?
```

Until that table exists, the bullets below are *internal review
notes*, not external pitch material.

## Upstreaming rationale (review-internal)

1. **Strict superset of capability.** `TurboQuantIndex` is unchanged
   on this branch; `RankIndex` and `RankQuantIndex` are new types,
   compiled and tested alongside.
2. **Zero new heavy dependencies.** The rank primitives use
   `ordered_float` and `rayon` (already in `Cargo.toml`). No BLAS,
   no codebook training, no rotation matrix.
3. **Storage parity, build-speed advantage, anisotropic-recall
   advantage.** Storage and build speed are unambiguous; recall is
   conditional on real-embedding rerun.
4. **Query latency follow-up is well-scoped.** The SIMD path is
   modelled directly on the existing `search.rs` template; v2 should
   land within a 2-3× factor of TurboQuant at matched bytes (the
   kernel shape — per-coordinate LUT, packed code scan — is the same).
5. **The audit-by-removal rationale.** RankQuant removes
   training, rotation, codebooks, and per-document norms from the
   pipeline. If retrieval survives the removal (which the v1 results
   suggest on the corpora tested), those components were carrying
   less than the dense-quantization literature assumes.
