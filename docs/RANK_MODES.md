# Rank-cosine index modes for turbovec

**Real-data headline (Harrier arXiv embeddings, 207k docs, paraphrase
queries):** at matched 256 B/vec, RankQuant-2 asymmetric beats
TurboQuant-2 by +3.3 R@10 (0.764 vs 0.731); at matched 512 B/vec,
TurboQuant-4 beats RankQuant-4 asymmetric by +5.2 R@10 (0.895 vs
0.843). Encode is 28-59× faster for RankQuant across both. With an
AVX2 inline-expand kernel for the asymmetric scan, query latency is
**within 2.8-3.4× of hand-tuned TurboQuant** (10.5 ms vs 3.1 ms at
2-bit, 16.4 ms vs 5.8 ms at 4-bit on 207k docs).

> Branch: `nelson/rank-modes`. Status: v1 prototype, scalar kernels.
> Numbers below are head-to-head on the paper's exact arXiv corpus
> (207,695 Harrier-OSS-v1-0.6B embeddings, 200 paraphrase queries)
> plus a structured synthetic stress-test. The 2-bit point is the
> operating regime where RankQuant is competitive on quality and
> dominates on build cost; at 4-bit TurboQuant's cosine optimisation
> wins on recall. Query latency follow-up (SIMD scan + 8-bit
> calibrated LUT) is well-scoped against the existing TurboQuant
> kernels.

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

Results below are with AVX2+FMA asymmetric scan enabled (auto-detected
at runtime on x86_64). Scalar-fallback numbers are in parentheses
where they differ.

| mode               | bytes/vec | total MiB | encode v/s  | p50 ms | p99 ms | GiB/s | ns/dim | R@10   |
|--------------------|-----------|-----------|-------------|--------|--------|-------|--------|--------|
| TurboQuant b=2     | 256       | 50.7      | 47,847      | 3.11   | 4.08   | 15.90 | 0.015  | **0.7305** |
| TurboQuant b=4     | 512       | 101.4     | 20,510      | 5.78   | 6.17   | 17.12 | 0.027  | **0.8945** |
| RankIndex sym      | 2048      | 405.7     | 1,059,828   | 105.5  | 110.8  | 3.76  | 0.496  | 0.8015 |
| RankIndex asym     | 2048      | 405.7     | 1,059,828   | 112.3  | 114.7  | 3.53  | 0.528  | 0.8475 |
| RankQuant b=2 sym  | 256       | 50.7      | 1,255,076   | 78.3   | 79.9   | 0.63  | 0.368  | 0.7130 |
| **RankQuant b=2 asym (AVX2)** | **256** | **50.7** | **1,255,076** | **10.5** | **11.0** | **4.73** | **0.049** | **0.7635** |
| RankQuant b=4 sym  | 512       | 101.4     | 1,163,698   | 80.3   | 81.8   | 1.23  | 0.377  | 0.7985 |
| **RankQuant b=4 asym (AVX2)** | **512** | **101.4** | **1,163,698** | **16.4** | **17.0** | **6.05** | **0.077** | **0.8430** |
| RankQuant b=1 asym | 128       | 25.4      | 1,245,716   | 75.9   | 77.6   | 0.33  | 0.357  | 0.6405 |

AVX2 lift, asymmetric path: **b=2 7.5× speedup** (78.9 → 10.5 ms),
**b=4 4.7× speedup** (77.4 → 16.4 ms). The kernel does not use a
per-coord LUT — `bucket_centre(b) = b - (2^B - 1) / 2` is one SIMD
subtraction, so the inner loop is broadcast → variable-shift → mask →
cvt → sub → FMA with no LUT memory traffic.

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

### Query latency: ~3× faster on real data (down from 13-24×)

After AVX2 lowering on the asymmetric path:

| corpus  | bytes/vec | TurboQuant p50 | RankQuant asym p50 | gap   | TQ GiB/s | Rank GiB/s |
|---------|-----------|----------------|---------------------|-------|---------:|-----------:|
| Harrier | 256       | 3.11 ms        | 10.5 ms             | 3.4×  |    15.90 |       4.73 |
| Harrier | 512       | 5.78 ms        | 16.4 ms             | 2.8×  |    17.12 |       6.05 |

TurboQuant ships architecture-specific kernels (NEON for ARM,
FAISS-style perm0-interleaved AVX2 for x86) with calibrated 8-bit
LUTs and SIMD-blocked layout (32 docs at a time). The RankQuant
AVX2 kernel processes one doc at a time and uses f32 accumulators
directly — simpler, well within an order of magnitude of TurboQuant's
hand-tuned path.

To close the remaining 3× to parity / dominance, three avenues
remain (none of them research questions):

1. **Byte-LUT scoring** — precompute `lut4[g][byte] = sum of 4
   per-coord contributions` (256 KiB per query LUT for D=1024, B=2),
   reduce inner loop to `sum_g lut4[g][doc[g]]`. May be
   bandwidth-bound but trivially vectorisable.
2. **AVX-512 path** — Zen 5 (Ryzen 9 9950X is the target box) has a
   full 512-bit datapath; the kernel scales naturally to 16-wide
   FMA. Gated on profiling actually showing the bottleneck is in
   arithmetic, not memory.
3. **SIMD-blocked layout** — process 8-32 docs in parallel per inner
   iteration, mirroring `pack.rs::repack`. Improves memory access
   pattern. Likely the highest single-step win.

The symmetric path is still scalar (lower-priority — asymmetric is
the recommended mode in the paper and wins every recall comparison
here). Symmetric SIMD is a natural follow-up.

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
