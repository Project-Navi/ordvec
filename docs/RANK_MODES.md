# Rank-cosine index modes for turbovec

**Real-data headline (Harrier arXiv embeddings, 207k docs, paraphrase
queries, Ryzen 9 9950X):**

At matched 256 B/vec, RankQuant-2 asymmetric beats TurboQuant-2 by
**+3.3 R@10** (0.764 vs 0.731). At matched 512 B/vec, TurboQuant-4
beats RankQuant-4 asymmetric by **+5.2 R@10** (0.895 vs 0.843). Encode
is **26-60× faster** for RankQuant across both.

After AVX-512 lowering with centre-drop, query latency on the same
corpus is **within 1.5× of hand-tuned TurboQuant at 4-bit** and
**within 2.5× at 2-bit**:

| bits | TurboQuant p50 | RankQuant AVX-512 p50 | gap  | Mdocs/s scan |
|------|---------------:|----------------------:|-----:|-------------:|
| 2    | 3.17 ms        | **7.99 ms**           | 2.5× | 26.0         |
| 4    | 5.84 ms        | **9.01 ms**           | 1.5× | 23.1         |

Optimisation chain (b=2 asym p50 on Harrier 207k):

```
scalar LUT                78.9 ms
+ AVX2 inline-expand      10.2 ms   (7.7× lift; replaces per-coord LUT with broadcast-shift-mask-cvt-FMA)
+ centre-drop              9.19 ms  (1.1× lift; raw codes in hot loop, constant added at finalize)
+ AVX-512                  7.99 ms  (1.15× lift; 16-wide FMA, single __m512 per chunk)
```

Centre-drop math: because centred bucket scores differ from raw-code
scores only by a query-constant offset (under the
`dim % (1 << bits) == 0` constraint that fixes every doc's bucket
histogram), the asymmetric kernel can score raw bucket IDs directly
for ranking. The offset is re-applied to the top-k scores at
finalize so the displayed cosines stay exact.

> Branch: `nelson/rank-modes`. Status: v1.1 prototype, AVX-512 scan
> for the asymmetric path with a scalar LUT fallback. Numbers below
> are head-to-head on the paper's exact arXiv corpus (207,695
> Harrier-OSS-v1-0.6B embeddings, 200 paraphrase queries) plus a
> structured synthetic stress-test. The 2-bit point is the operating
> regime where RankQuant beats TurboQuant on recall and dominates on
> build cost; at 4-bit TurboQuant's cosine optimisation wins on recall
> but RankQuant's query latency is within 1.5×.

## Bench environment

| field | value |
|---|---|
| CPU | AMD Ryzen 9 9950X (Zen 5, 16C/32T, full 512-bit AVX-512 datapath) |
| RAM | 128 GB Kingston Fury Beast DDR5-4000 CL29 × 4 DIMMs (capacity-optimised) |
| OS | CachyOS Linux, kernel 7.0.6 |
| Compiler | rustc 1.94.1 (LLVM 21.1.8) |
| Build | `cargo build --release` with `lto = true, codegen-units = 1, opt-level = 3` |
| Governor | `performance` |
| THP | `always` |
| Detected SIMD | sse4.2, avx2, fma, avx512f, avx512bw, avx512vl |
| Latency mode | single-thread per query (rayon parallelises *across* queries; per-query rows measure scan only) |

A two-DIMM DDR5-6000-class system may show shorter absolute latency;
the relative gap to TurboQuant is the load-bearing comparison.

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

## Headline numbers (real data — Harrier arXiv)

Bench:
```bash
RUSTFLAGS="-l openblas" cargo run --release -p turbovec --example bench_rank -- \
  --corpus-npy /path/to/embeddings.npy \
  --queries-npy /path/to/paraphrase_queries.npy \
  --queries 200 --k 10
```

Setup: 207,695-doc arXiv (cs.LO + math.LO + cs.AI) corpus embedded
with `microsoft/Harrier-OSS-v1-0.6B` at D=1024, paraphrase queries
(LLM-generated research-question rewrites from a different model
family). The same artefacts that produced the paper's main-corpus
results. 200 queries × top-10. Ground truth: FP32 brute-force cosine.
32-core Linux x86_64, release build.

Results below are with the AVX-512 asymmetric scan enabled (auto-detected
at runtime; falls through to AVX2 then to a scalar LUT scan). Symmetric
paths and the B=1 asymmetric path remain on the scalar LUT scan.

| mode               | bytes/vec | total MiB | encode v/s  | p50 ms | p99 ms | GiB/s | ns/dim | Mdocs/s | R@10   |
|--------------------|-----------|-----------|-------------|--------|--------|------:|-------:|--------:|--------|
| TurboQuant b=2     | 256       | 50.7      | 47,117      | 3.17   | 3.52   | 15.64 | 0.015  |   65.6  | **0.7305** |
| TurboQuant b=4     | 512       | 101.4     | 19,531      | 5.84   | 6.24   | 16.96 | 0.027  |   35.6  | **0.8945** |
| RankIndex sym      | 2048      | 405.7     | 1,039,409   | 104.1  | 105.6  | 3.81  | 0.489  |    2.0  | 0.8015 |
| RankIndex asym     | 2048      | 405.7     | 1,039,409   | 112.4  | 115.0  | 3.52  | 0.529  |    1.8  | 0.8475 |
| RankQuant b=2 sym  | 256       | 50.7      | 1,225,501   | 78.8   | 79.9   | 0.63  | 0.370  |    2.6  | 0.7130 |
| **RankQuant b=2 asym (AVX-512)** | **256** | **50.7** | **1,225,501** | **7.99** | **8.39** | **6.20** | **0.038** | **26.0** | **0.7635** |
| RankQuant b=4 sym  | 512       | 101.4     | 1,183,268   | 77.2   | 78.3   | 1.28  | 0.363  |    2.7  | 0.7985 |
| **RankQuant b=4 asym (AVX-512)** | **512** | **101.4** | **1,183,268** | **9.01** | **9.69** | **10.99** | **0.042** | **23.1** | **0.8430** |
| RankQuant b=1 asym | 128       | 25.4      | 1,282,106   | 75.7   | 77.4   | 0.33  | 0.356  |    2.7  | 0.6405 |

The kernel does not use a per-coord LUT — `bucket_centre(b) = b - (2^B - 1) / 2`
is one SIMD subtraction (folded out to the per-query offset via
centre-drop), so the inner loop is broadcast → variable-shift → mask
→ cvt → FMA with no LUT memory traffic.

### Reading the real-data table

**At 256 B/vec (2-bit), RankQuant beats TurboQuant on recall by
+3.3 R@10.** This is the paper's primary operating point and the
operating point where RankQuant's design assumption (preserving
coordinate order at compact storage) most directly competes with
TurboQuant's design assumption (preserving cosine geometry at
compact storage). Real-embedding result: ordinal slightly wins.

**At 512 B/vec (4-bit), TurboQuant beats RankQuant on recall by
+5.2 R@10.** At wider codes TurboQuant's per-coordinate magnitude
quantisation captures information the rank bucketing cannot — the
4-bit RankQuant is bucketing into only 16 bins per dim while
4-bit TurboQuant is using 16 Lloyd-Max centroids tuned to minimise
cosine distortion. Real-embedding result: at the wider byte
budget, magnitude wins.

**Encode is 28-59× faster across the board.** TurboQuant b=2:
44,713 vec/s; RankQuant b=2: 1,242,635 vec/s. TurboQuant b=4:
19,721 vec/s; RankQuant b=4: 1,163,978 vec/s. The rank pipeline
has no rotation matmul and no codebook fit; the dominant cost is
`argsort` per vector.

**Query latency is 13-24× slower for RankQuant on this branch.**
TurboQuant achieves 15-17 GiB/s effective scan bandwidth via
hand-tuned NEON/AVX kernels; RankQuant's scalar Rust scan hits
0.63-1.28 GiB/s — a 13-25× implementation gap. After a SIMD
lowering modelled on `search.rs` (calibrated 8-bit LUT,
SIMD-blocked layout, packed-nibble scan), the rank scan should
land within 2-3× of TurboQuant's latency at the same byte budget.

**No claim is currently better than the table above.** The synthetic
stress-test below remains as a method check, not a headline.

## Stress test (low-rank clustered synthetic)

A second bench at smaller scale tests how the methods react to
deliberately anisotropic data. Run:

```bash
cargo run --release -p turbovec --example bench_rank -- \
  --dim 1024 --n 50000 --queries 200 --clusters 200 --latent 64
```

Setup: D=1024, N=50,000 documents, 200 queries, k=10. Low-rank
clustered corpus (200 cluster prototypes, latent_dim=64, projected
to D=1024 with N(0,1) noise = 0.3 for docs, 0.1 for queries).
Ground truth: FP32 brute-force cosine top-10.

| mode               | bytes/vec | encode v/s | p50 ms | GiB/s | ns/dim | R@10  |
|--------------------|-----------|------------|--------|-------|--------|-------|
| TurboQuant b=2     | 256       | 44,802     | 0.51   | 23.5  | 0.010  | 0.299 |
| TurboQuant b=4     | 512       | 18,552     | 1.17   | 20.4  | 0.023  | 0.492 |
| RankIndex sym      | 2048      | 1,065,560  | 25.0   | 3.8   | 0.489  | 0.874 |
| RankIndex asym     | 2048      | 1,065,560  | 25.8   | 3.7   | 0.504  | 0.911 |
| RankQuant b=2 sym  | 256       | 1,186,263  | 19.1   | 0.63  | 0.372  | 0.617 |
| RankQuant b=2 asym | 256       | 1,186,263  | 18.9   | 0.63  | 0.368  | 0.722 |
| RankQuant b=4 sym  | 512       | 1,142,377  | 19.3   | 1.23  | 0.377  | 0.849 |
| RankQuant b=4 asym | 512       | 1,142,377  | 19.5   | 1.22  | 0.381  | 0.889 |
| RankQuant b=1 sym  | 128       | 1,254,733  | 18.4   | 0.32  | 0.359  | 0.407 |
| RankQuant b=1 asym | 128       | 1,254,733  | 18.4   | 0.33  | 0.359  | 0.525 |

The synthetic-vs-real comparison is the load-bearing fact in this
section: at matched bytes, **the +42 R@10 gap on synthetic data
collapses to +3.3 R@10 on real Harrier** at 2-bit, and to *-5.2*
R@10 at 4-bit. The synthetic corpus is anisotropic by construction
(latent_dim=64 in D=1024), which is the regime where TurboQuant's
data-oblivious random rotation is most strained. Real
arXiv-Harrier embeddings have anisotropic structure too, but less
extreme — enough to surface a small RankQuant advantage at the
narrowest byte budget, not enough to make TurboQuant collapse.

## What survived the head-to-head

### Encode throughput: 23-59× faster (real data)

| bench   | bytes/vec | TurboQuant v/s | RankQuant v/s | ratio  |
|---------|-----------|----------------|---------------|--------|
| Harrier | 256       | 44,713         | 1,242,635     | 27.8×  |
| Harrier | 512       | 19,721         | 1,163,978     | 59.0×  |
| Synth   | 256       | 44,802         | 1,186,263     | 26.5×  |
| Synth   | 512       | 18,552         | 1,142,377     | 61.6×  |

The architectural reason is straightforward: no rotation matrix
multiply, no Lloyd-Max codebook fit, no per-vector norm storage.
Encode is one `argsort` per coordinate + one bucket-pack pass per
document. This advantage transfers cleanly between synthetic and
real corpora.

### Storage: identical at matched bit width

`bytes_per_vec = dim * bits / 8` for both schemes. The byte budget is
the same lever. The implementation differs in what each byte
*means*: TurboQuant stores a quantised magnitude, RankQuant stores a
bucketed rank.

### Asymmetric beats symmetric (both corpora)

Harrier:

| mode             | sym R@10 | asym R@10 | Δ      |
|------------------|---------:|----------:|-------:|
| Rank full (2KB)  | 0.802    | 0.848     | +0.046 |
| RankQuant b=4    | 0.799    | 0.843     | +0.045 |
| RankQuant b=2    | 0.713    | 0.764     | +0.051 |
| RankQuant b=1    | 0.504    | 0.641     | +0.137 |

Synthetic:

| mode             | sym R@10 | asym R@10 | Δ      |
|------------------|---------:|----------:|-------:|
| Rank full (2KB)  | 0.874    | 0.911     | +0.037 |
| RankQuant b=4    | 0.849    | 0.889     | +0.040 |
| RankQuant b=2    | 0.617    | 0.722     | +0.105 |
| RankQuant b=1    | 0.407    | 0.525     | +0.118 |

The asymmetric variant keeps the query side as full FP32 — the
encoder's output is consumed directly, only the document side loses
precision. This is the operating point the paper recommends and it
reproduces on both corpora. The advantage grows as document-side
precision shrinks (more information lost on the doc side, more value
in keeping the query rich).

### Real-data recall summary at matched bytes

| bytes/vec | TurboQuant | RankQuant asym | Δ      |
|-----------|-----------:|---------------:|-------:|
| 256       | 0.7305     | 0.7635         | +0.033 |
| 512       | 0.8945     | 0.8430         | -0.052 |

The 2-bit point is the operating regime where RankQuant is
competitive on quality on top of dominating on build cost. The 4-bit
point is where TurboQuant's per-coordinate magnitude quantisation
(16 Lloyd-Max centroids per dim, calibrated to minimise cosine
distortion) pulls ahead of 4-bit rank bucketing (16 equal-width bins
on a permutation axis). This is the honest crossover, not a
synthetic artefact.

## Where TurboQuant still wins

### Query latency: 1.5-2.5× faster after AVX-512 lowering

| corpus  | bytes/vec | TurboQuant p50 | RankQuant AVX-512 p50 | gap  | TQ GiB/s | Rank GiB/s |
|---------|-----------|----------------|------------------------|------|---------:|-----------:|
| Harrier | 256       | 3.17 ms        | 7.99 ms                | 2.5× |    15.64 |       6.20 |
| Harrier | 512       | 5.84 ms        | 9.01 ms                | 1.5× |    16.96 |      10.99 |

**The AVX-512 kernel is an exact packed scan, not an ANN
approximation.** It returns identical top-k to the scalar RankQuant
scorer and agrees within 1e-4 on scores (verified by
`tests/rank_index.rs::rankquant_asymmetric_matches_reference_b{2,4}`).
Exact scan within 1.5× of a tuned quantized baseline is the systems
result this branch was aiming for.

Byte-LUT alternative (head-to-head on the same corpus):

| bytes/vec | inline-expand AVX-512 | scalar byte-LUT | ratio |
|-----------|-----------------------:|----------------:|------:|
| 256       | 7.99 ms                | 19.5 ms         | 2.4×  |
| 512       | 9.01 ms                | 38.2 ms         | 4.2×  |

Same recall, much slower path. Streaming SIMD math beats query-LUT
cache traffic on Zen 5. The byte-LUT scorer stays in the codebase as
a labelled reference path (`turbovec::rank_index::search_asymmetric_byte_lut`)
but is not the production scoring route.

### Remaining headroom

The b=2 path is still decode-bound (6.2 GiB/s effective vs 16+ GiB/s
the platform demonstrably delivers via TurboQuant). Closing the rest
of the gap is, in priority order:

1. **Multi-accumulator b=2 kernel** — break the FMA dependency chain
   by splitting into 2-4 independent accumulators per doc. Cheap to
   implement, likely meaningful on the decode-bound path.
2. **Unroll across docs** — process 2-4 docs per inner iteration so
   the front-end can hide the broadcast/shift/mask latency.
3. **SIMD-blocked layout** — repack into 32-doc tiles like
   `pack.rs::repack`. Improves memory access pattern. Highest
   single-step win but largest restructuring.

None of these are research questions; all of them have a direct
template in `search.rs` for TurboQuant or in the existing
`rank_index.rs` AVX-512 kernel.

The symmetric path is still scalar (lower-priority — asymmetric is
the recommended mode in the paper and wins every recall comparison
here). Symmetric SIMD is a natural follow-up.

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

# Real-data head-to-head on Harrier arXiv embeddings.
# Embedding artefacts are produced by the RankQuant paper's
# arXiv pipeline (turbovec-arxiv repo, microsoft/Harrier-OSS-v1-0.6B).
RUSTFLAGS="-l openblas" cargo run --release -p turbovec --example bench_rank -- \
    --corpus-npy  /path/to/embeddings.npy \
    --queries-npy /path/to/paraphrase_queries.npy \
    --queries 200 --k 10

# Synthetic stress-test (anisotropic low-rank clustered).
RUSTFLAGS="-l openblas" cargo run --release -p turbovec --example bench_rank \
    -- --dim 1024 --n 50000 --queries 200
```

The `RUSTFLAGS="-l openblas"` shim is needed on Linux for the
TurboQuant rotation step; rank modes themselves do not depend on
BLAS. The npy loader is a minimal NumPy v1 reader for `<f4`
little-endian, C-order 2-D arrays; no Python dependency at bench
time.

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
