# ordvec's ordinal & sign index modes

These index types operate on a *rank* (or *sign*) view of the
embedding rather than on its raw magnitudes:

> RankQuant turns vectors into fixed-mass ordinal sets, so candidate
> generation becomes bitmap overlap instead of low-bit dot product.
> Magnitude quantizers don't have this primitive.

The asymmetric scan ships an AVX-512 path (16-wide FMA, 4-way
multi-accumulator) that auto-detects at runtime and falls through to
AVX2 and then a scalar LUT scan; symmetric paths and the b=1
asymmetric path use the scalar LUT scan. The whole pipeline uses zero
training, zero rotation, zero codebook — the structural prior is what
does the work.

**Self-contained stress test (synthetic clustered corpus).** Every
number in the synthetic tables below is regenerable from this repo
with a single command, no external data and no system dependencies
(ordvec links no BLAS):

```bash
cargo run --release --example bench_rank
```

That runs the head-to-head on a structured synthetic corpus (D=256,
N=30,000, 200 queries, 200 cluster prototypes, latent_dim=64; see
[Stress test](#stress-test-low-rank-clustered-synthetic) for the
exact construction). Results on real embedding corpora are
user-runnable via `--corpus-npy` / `--queries-npy`; the current
arXiv paper-harness result is summarized in the README and the
reproduction shape is described under
[External-corpus results](#external-corpus-results-user-runnable).

The bitmap two-stage path (`Bitmap` candidate gen →
`RankQuant` exact subset rerank) is the operating point that
turns RankQuant from a slow exact scan into a fast two-stage retriever:
the bitmap probe is the cheap candidate generator, and
`search_asymmetric_subset` reruns the exact RankQuant kernel on only
the surviving M candidates. The bench reports this path as its
`TwoStage ...` rows, each annotated with a candidate-recall figure
(`CR` = fraction of exact-RankQuant top-10 indices present in the
bitmap's M-candidate set, averaged over queries — distinct from task
R@10, which is against FP32 brute-force cosine ground truth).

Centre-drop math (why the asymmetric kernel needs no per-coord LUT):
because centred bucket scores differ from raw-code scores only by a
query-constant offset (under the `dim % (1 << bits) == 0` constraint
that fixes every doc's bucket histogram), the asymmetric kernel can
score raw bucket IDs directly for ranking. The offset is re-applied
to the top-k scores at finalize so the displayed cosines stay exact.

## Bench environment

| field | value |
|---|---|
| CPU | AMD Ryzen 9 9950X (Zen 5, 16C/32T, full 512-bit AVX-512 datapath) |
| OS | CachyOS Linux |
| Compiler | rustc 1.95.0 |
| Build | `cargo build --release` with `lto = true, codegen-units = 1, opt-level = 3` |
| Detected SIMD | sse4.2, avx2, fma, avx512f, avx512bw, avx512vl |
| Latency mode | single-thread per query (rayon parallelises *across* queries; per-query rows measure scan only) |

Absolute latencies are machine-specific; the QUALITY columns (R@10,
CR, bytes/vec) are seeded and bit-identical run-to-run, while the
throughput/latency columns are wall-clock and vary across hardware.

The three scored index families are:

- **`Rank`** — stores the dimension-wise rank transform of each
  document as `u16` (`2 * dim` bytes per document).
- **`RankQuant`** — buckets each rank into `1 << bits` equal-width
  bins on `[0, dim)` and packs `bits` bits per coordinate
  (`dim * bits / 8` bytes per document). Supported `bits ∈ {1, 2, 4}`.
- **`Bitmap`** / **`SignBitmap`** — one bit per coordinate
  (`dim / 8` bytes per document); the cheap candidate-gen front end for
  the two-stage path (see [README](../README.md#ordinal-index-family)).

Both `Rank` and `RankQuant` expose `search` (symmetric:
rank-vs-rank, Spearman correlation) and `search_asymmetric` (FP32
query against rank-stored documents).

The construction has no rotation matrix, no codebook, no Lloyd-Max
training, no per-document norms (the L2 norm of a permutation of
`{0..D-1}` is analytical). Encode is a single `argsort` pass per
vector, with the option to bucket and pack into `B` bits per
coordinate.

## Why this works: combinatorics, not geometry

The bitmap two-stage result is not merely a faster scoring kernel —
it is a structural primitive that magnitude-preserving quantization
does not expose. Three properties chain together:

**1. RankQuant is a constant-composition code.** The rank transform
is a permutation of `{0, ..., D-1}`, so under the equal-width bucket
partition every document assigns *exactly the same number of
coordinates to each bucket*: `D / 2^B` coordinates per bucket, for
all docs. For `D=256, B=2` that is 64 coordinates in the top bucket
of every document.

**2. The similarity score decomposes over bucket-overlap counts.**
Let `Q_a` be the set of query coordinates in bucket `a`, and `D_b`
the analogous set for the document. Then asymmetric rank-cosine
re-expresses (up to per-query constants) as a weighted contingency
table of bucket-overlap counts:

```
score(q, d) = Σ_{a,b} w(a, b) · |Q_a ∩ D_b|
```

So RankQuant similarity is a bilinear function of bucket-overlap
counts between two constant-composition partitions, not a dot
product over magnitudes.

**3. The simplest truncation — top-bucket overlap — has a closed-form
null distribution.** For uniformly random fixed-size subsets,
`X = |Q_top ∩ D_top|` is hypergeometric `H(D, n_top, n_top)` with
`E[X] = n_top² / D`. For `D=256, n_top=64` the expected overlap
under the null is exactly **16**. Observed overlaps significantly
above 16 are evidence of shared coordinate salience, with
closed-form p-values from the hypergeometric distribution.

This is what makes the bitmap probe a principled candidate
generator rather than a tunable heuristic. Magnitude quantizers
don't have a hypergeometric null because they don't have fixed
bucket cardinalities — their score distribution depends on the
unknown embedding distribution.

**Checked finite model: symmetry, quotient sufficiency, threshold,
calibration.** The proof chain now has a larger machine-checked middle
than the implementation docs used to claim. In
[`ordvec-formalization`](https://github.com/Fieldnote-Echo/ordvec-formalization),
Lean proves that literal bitmap overlap is the canonical invariant
under query-preserving coordinate relabelings; finite quotient
sufficiency reduces the admission decision to ordered overlap
evidence when the likelihood ratio factors through it; a finite
monotone-likelihood-ratio overlap-tilt model makes an overlap-count
threshold Bayes-optimal among deterministic admission rules; and the
uniform constant-weight bitmap null gives that same threshold event
the exact hypergeometric upper tail. The headline theorem is
`exists_uniformBitmapOverlapTail_finiteBayesRisk_le_and_hypergeomTail`;
the proof path is summarized in the formalization repo's
`docs/proof-spine.md`, with theorem names in `docs/theorem-map.md`.

It is still an *in-model* result. The theorem is about literal
constant-weight bitmaps, finite deterministic admission, and explicit
symmetry / quotient / monotone-overlap assumptions. It does not prove
that real encoders satisfy those assumptions, that the textbook
hypergeometric is every deployment corpus's null, or that ordinal
quotients are representation-complete. Whether real neighbours clear a
cutoff stays empirical, which is what the bench and the paper measure.

The systems consequence is what the bench measures: at a moderate M
the bitmap probe captures most of exact RankQuant's top-10 neighbours,
so the two-stage rerank reproduces near-exact RankQuant R@10 at a
fraction of the full-scan latency. The `bench_rank` run prints this
as its `TwoStage ...` rows with the per-M candidate-recall (`CR`)
figure attached.

## Synthetic stress-test numbers

This is the clean-checkout stress test — regenerated by the default
`bench_rank` run, no external data required:

```bash
cargo run --release --example bench_rank
```

Setup: D=256, N=30,000 documents, 200 queries, k=10. Low-rank
clustered corpus (200 cluster prototypes, latent_dim=64, projected to
D=256 with N(0,1) noise = 0.3 for docs, 0.1 for queries). Ground
truth: FP32 brute-force cosine top-10. The construction is detailed
under [Stress test](#stress-test-low-rank-clustered-synthetic) — it
is a generated Gaussian low-rank fixture, useful for exercising the
rank-mode kernels and their size/latency tradeoffs. Treat its recall
spread as a stress-test result, not the lead retrieval-quality claim;
the current real-embedding arXiv benchmark in the README is the better
guide to retrieval-relevant ordinal behaviour.

Results are with the AVX-512 asymmetric scan enabled where applicable
(auto-detected at runtime; falls through to AVX2 then to a scalar LUT
scan). Symmetric paths and the b=1 asymmetric path use the scalar LUT
scan. R@10 below is the seeded, deterministic quality column; encode
throughput is a representative wall-clock figure (see
[Bench environment](#bench-environment)).

| mode               | bytes/vec | encode v/s | R@10  |
|--------------------|-----------|------------|-------|
| Rank sym           | 512       | 4,559,550  | 0.7825 |
| Rank asym          | 512       | 4,559,550  | 0.8450 |
| RankQuant b=4 sym  | 128       | 5,205,223  | 0.7475 |
| RankQuant b=4 asym | 128       | 5,205,223  | 0.8055 |
| RankQuant b=2 sym  | 64        | 5,251,083  | 0.4660 |
| RankQuant b=2 asym | 64        | 5,251,083  | 0.5715 |
| RankQuant b=2 fastscan | 128   | 283,630    | 0.5700 |
| RankQuant b=1 sym  | 32        | 5,523,695  | 0.2785 |
| RankQuant b=1 asym | 32        | 5,523,695  | 0.3470 |
| Bitmap n_top=64    | 32        | 5,576,810  | 0.2480 |
| SignBitmap probe   | 32        | 19,641,040 | 0.2880 |

(Full latency/bandwidth columns and the machine-readable JSON line are
in [`benchmarks/rank_modes_results.txt`](../benchmarks/rank_modes_results.txt).)

The kernel does not use a per-coord LUT — `bucket_centre(b) = b - (2^B - 1) / 2`
is one SIMD subtraction (folded out to the per-query offset via
centre-drop), so the inner loop is broadcast → variable-shift → mask
→ cvt → FMA with no LUT memory traffic.

### Reading the synthetic table

**Recall favours the asymmetric scan at every bit width.** Keeping the
query side as full FP32 while only the document side loses precision
recovers a consistent R@10 margin over the symmetric (rank-vs-rank)
scan, and the margin widens as document-side precision shrinks.

**Storage is `dim * bits / 8` bytes per document** for the bucketed
modes (`RankQuant`); the single-bit probes (`Bitmap`,
`SignBitmap`) are `dim / 8` bytes. `RankQuantFastscan` is the
exception — its block-32 re-blocking costs `dim / 2` bytes (2× the
single-rate b=2 packing) in exchange for lower per-query scan latency.

**Encode is dominated by `argsort`.** There is no rotation matmul and
no codebook fit to amortise, so encode throughput is high and
data-independent (the dominant per-vector cost is the rank transform).

**Single-query exact-scan latency is the standing weakness.** The
per-query exact RankQuant scan is decode-bound at the narrow byte
budgets; the two-stage path (`Bitmap` → `RankQuant` subset
rerank) is what closes this in practice — it scores only the M bitmap
survivors instead of the full corpus. The candidate-recall vs latency
trade is the bench's two-stage rows; remaining single-query SIMD
headroom is in [Latency characteristics](#latency-characteristics).

## Stress test (low-rank clustered synthetic)

This is the construction behind the [synthetic stress-test
table](#synthetic-stress-test-numbers). The default
`bench_rank` run uses these parameters; the explicit form is:

```bash
cargo run --release --example bench_rank -- \
  --dim 256 --n 30000 --queries 200 --clusters 200 --latent 64
```

Setup: D=256, N=30,000 documents, 200 queries, k=10. Low-rank
clustered corpus (200 cluster prototypes, latent_dim=64, projected
to D=256 with N(0,1) noise = 0.3 for docs, 0.1 for queries).
Ground truth: FP32 brute-force cosine top-10.

The corpus is anisotropic *by construction* (latent_dim=64 in D=256),
but it is still a generated Gaussian fixture. It is useful for
self-contained kernel checks and for stressing the compression modes;
it should not be read as the strongest evidence for the retrieval task.
Real sentence/passage embeddings are anisotropic in task-specific ways,
and the current arXiv source-recovery benchmark is more favorable to
the rank transform than this small synthetic fixture.

## What the head-to-head shows

### Encode throughput

| mode            | bytes/vec | encode v/s |
|-----------------|-----------|------------|
| RankQuant b=2   | 64        | 5,251,083  |
| RankQuant b=4   | 128       | 5,205,223  |
| Rank            | 512       | 4,559,550  |
| SignBitmap      | 32        | 19,641,040 |

The architectural reason is straightforward: no rotation matrix
multiply, no Lloyd-Max codebook fit, no per-vector norm storage.
Encode is one `argsort` per document + one bucket-pack pass. The
numbers above are from the synthetic stress-test run; the per-vector cost
is data-independent.

### Storage

`bytes_per_vec = dim * bits / 8` for the bucketed modes. The
single-bit probes are `dim / 8` bytes; `RankQuantFastscan` is
`dim / 2` bytes (the FastScan space-for-latency trade).

### Asymmetric beats symmetric

Synthetic stress-test run:

| mode             | sym R@10 | asym R@10 | Δ      |
|------------------|---------:|----------:|-------:|
| Rank full (512B) | 0.7825   | 0.8450    | +0.0625 |
| RankQuant b=4    | 0.7475   | 0.8055    | +0.0580 |
| RankQuant b=2    | 0.4660   | 0.5715    | +0.1055 |
| RankQuant b=1    | 0.2785   | 0.3470    | +0.0685 |

The asymmetric variant keeps the query side as full FP32 — the
encoder's output is consumed directly, only the document side loses
precision. This is the recommended mode. The advantage grows as
document-side precision shrinks (more information lost on the doc
side, more value in keeping the query rich).

### Two-stage candidate generation

The bitmap probe is a cheap candidate generator; the exact RankQuant
b=2 kernel reranks only the M survivors. Candidate-recall (`CR`) is the
fraction of the exact-RankQuant top-10 captured in the M-candidate set
(an ANN probe-quality metric, distinct from task R@10):

| mode                        | bytes/vec | CR    | R@10   |
|-----------------------------|-----------|-------|--------|
| TwoStage b=2 M=100          | 96        | 0.976 | 0.5700 |
| TwoStage b=2 M=500          | 96        | 1.000 | 0.5715 |
| TwoStage b=2 M=1000         | 96        | 1.000 | 0.5715 |
| SignTwoStage b=2 M=500      | 96        | 1.000 | 0.5715 |

At M=500 the probe already captures the full exact-RankQuant top-10
(CR = 1.000), so the two-stage rerank matches the full exact b=2 scan
(R@10 = 0.5715) while touching a small fraction of the corpus.

## Latency characteristics

### Single-query exact-scan latency

The per-query exact RankQuant scan is decode-bound at the narrow byte
budgets, so a full single-query exact scan is the slowest route. Two
facts qualify this:

- **The two-stage path is the intended fast route.** Scoring only the
  M bitmap candidates with `search_asymmetric_subset` avoids the
  full-corpus scan entirely. That is the operating point the
  rank-mode README recommends, and where the structural prior pays
  off.
- **The asymmetric AVX-512 kernel is an exact packed scan, not an ANN
  approximation.** It is checked against the scalar RankQuant scorer with
  score tolerances and deterministic golden tie fixtures (see
  [`determinism.md`](determinism.md)); the random reference tests avoid
  overfitting top-k order at near-tolerance boundaries.

The byte-LUT scorer remains in the codebase as a labelled reference
path (`ordvec::search_asymmetric_byte_lut`,
benched as the `RankQuant b=… asym byte-LUT` rows) but is not the
production scoring route — streaming SIMD math beats query-LUT cache
traffic on the hardware tested.

### Remaining headroom

The single-query b=2 exact scan is decode-bound. Further reducing its
latency, in priority order:

1. **Multi-accumulator b=2 kernel** — break the FMA dependency chain
   by splitting into 2-4 independent accumulators per doc. Cheap to
   implement, likely meaningful on the decode-bound path.
2. **Unroll across docs** — process 2-4 docs per inner iteration so
   the front-end can hide the broadcast/shift/mask latency.
3. **SIMD-blocked layout** — repack into 32-doc tiles (the approach
   `RankQuantFastscan` already takes for its b=2 fast path).
   Improves the memory access pattern. Highest single-step win but
   largest restructuring.

None of these are research questions; the AVX-512 kernel in
`src/quant_kernels.rs` (and the block-32 layout in
`src/fastscan.rs`) is a direct template.

The symmetric path is still scalar (lower-priority — asymmetric is
the recommended mode and wins every recall comparison here).
Symmetric SIMD is a natural follow-up.

## External-corpus results (user-runnable)

The synthetic numbers above come from the in-repo generated corpus.
To check the modes on real embeddings, point the same bench at your own
`.npy` arrays:

```bash
cargo run --release --example bench_rank -- \
  --corpus-npy  /path/to/embeddings.npy \
  --queries-npy /path/to/queries.npy \
  --queries 200 --k 10
```

`--corpus-npy` / `--queries-npy` each take a NumPy v1 `.npy` file
holding a 2-D little-endian `float32` (`<f4`), C-order array
(`(n, dim)` for the corpus, `(n_q, dim)` for queries); `--n` and
`--dim` are then taken from the file shapes. The npy loader is a
minimal built-in reader — no Python dependency at bench time, and no
BLAS.

What to expect from real embeddings: dense sentence/passage encoders
often carry retrieval signal in their coordinate order. The current
paper-harness arXiv run (207,695 embeddings, 7,200 source-recovery
queries) has full ordinal rank-cosine within bootstrap noise of dense
exact search and slightly ahead of the tested FAISS HNSW configuration;
RankQuant b=2 asym matches that HNSW configuration within bootstrap
noise at 256 bytes/vector. Run the command above on your target
embeddings to get the number that matters for your deployment — the
arXiv artifact set is not shipped in this crate.

## A null result reported up front

We also tested adding 10 rank-native structural features (per-(q, d)
bitmap-overlap counts, bilinear bucket-pair contingency cells,
query-level concentration broadcast) to a LambdaMART reranker on
both RRF-generated and bitmap-generated candidate sets, over five
seeds. **Null result on both candidate distributions**: the
structural features are scalar projections of information the
LambdaMART baseline already captures via continuous `rank_cos`. We
report it so the obvious follow-up is on record as
tested-and-didn't-land: the right place for rank-native structure is
candidate generation (where the bitmap two-stage above wins), not
LambdaMART feature engineering. The detailed multi-seed numbers for
that experiment appear in the accompanying paper (link TBD).

## API surface

| capability | Rank | RankQuant | Bitmap / SignBitmap |
|---|---|---|---|
| `new(...)` | `new(dim)` | `new(dim, bits)` | `new(dim[, n_top])` |
| `add(&[f32])` | ✓ | ✓ | ✓ |
| `search(&[f32], k)` | ✓ symmetric | ✓ symmetric | — |
| `search_asymmetric(&[f32], k)` | ✓ | ✓ | — |
| `top_m_candidates(&[f32], m)` | — | — | ✓ |
| `search_asymmetric_subset(q, &cands, k)` | — | ✓ | — |
| `swap_remove(idx)` | ✓ | ✓ | ✓ |
| `len`/`is_empty`/`dim`/`bytes_per_vec`/`byte_size` | ✓ | ✓ | ✓ |
| `write`/`load` | ✓ | ✓ | ✓ |

`write`/`load` are implemented for every ordinal-family type (`Rank`,
`RankQuant`, `Bitmap`, `SignBitmap`) with the byte-level
serialisers living in [`src/rank_io.rs`](../src/rank_io.rs) and
[`src/sign_bitmap.rs`](../src/sign_bitmap.rs). `RankQuant`
additionally exposes `search_asymmetric_subset` for scoring a
precomputed candidate set — the rerank half of the two-stage pattern.

`RankQuantFastscan` (re-exported `#[doc(hidden)]`) is an optional
single-pass b=2 fast path; it supports `add`/`search` but not
`swap_remove`/`write`/`load` (see its module docs in
`src/fastscan.rs`). `MultiBucketBitmap` underwrites the
bilinear bucket-overlap decomposition and is reachable only behind the
`experimental` feature.

Search result ordering, backend score-equivalence expectations, tie keys, and
empty-result shapes are specified in [`determinism.md`](determinism.md).

## Test coverage

`cargo test --lib` — unit tests for the primitives in
[`src/rank.rs`](../src/rank.rs) (`mod tests`): rank transform vs numpy
`argsort(argsort)` reference, rank-is-a-permutation, uniform bucket
partitioning, bucket-centre symmetry, pack/unpack round-trips, and
analytical norms.

`cargo test --test index` — the integration suite in
[`tests/index/`](../tests/index/) (`rank.rs`, `quant.rs`,
`bitmap.rs`, `fastscan.rs`, and `multi_bucket.rs` under the
`experimental` feature). Representative cases:

- `rank_index_symmetric_matches_reference` — `Rank::search`
  matches a scalar Spearman implementation on a 256-doc / 128-dim
  corpus, exact top-10 ordering, score agreement to 1e-4.
- `rank_index_asymmetric_matches_reference` — same, for the FP32-vs-
  rank kernel.
- `rankquant_asymmetric_matches_reference_b{1,2,4}` — RankQuant
  asymmetric agrees with the scalar reference at every bit width
  (the AVX-512-vs-scalar exactness check).
- `fastscan_b2_top10_matches_avx512_kernel` — the FastScan b=2 path
  agrees with the single-rate b=2 kernel on top-10 (within 8-bit LUT
  noise).
- `rankquant_b2_recovers_planted_neighbour_in_top_10` — queries
  constructed by adding noise to a known corpus doc; RankQuant-2
  asymmetric recovers the planted doc in top-10.
- `rank_index_recall_at_10_matches_fp32` — rank-cosine and raw FP32
  cosine top-10 sets overlap on smooth random data.
- `rank_index_swap_remove_keeps_state_consistent` /
  `rankquant_swap_remove_keeps_state_consistent` — `swap_remove`
  is byte-exact across the storage buffer.
- `rank_io_loaders_reject_malformed_files_without_panicking` — every
  loader returns `Err` (never panics) on malformed serialised files.

A separate red-team suite (`tests/redteam_*.rs`) covers adversarial
inputs and robustness regressions.

## Reproducibility

```bash
cargo test --lib                                     # unit tests
cargo test --test index                              # integration
cargo test --features experimental                   # + MultiBucket tests

# Headline benchmark (synthetic clustered corpus — no external data,
# no BLAS).
cargo run --release --example bench_rank

# Same bench against your own real-embedding arrays.
cargo run --release --example bench_rank -- \
    --corpus-npy  /path/to/embeddings.npy \
    --queries-npy /path/to/queries.npy \
    --queries 200 --k 10
```

ordvec links no BLAS, so no link shim is needed on any platform. The
npy loader is a minimal NumPy v1 reader for `<f4` little-endian,
C-order 2-D arrays; no Python dependency at bench time. The synthetic
stress-test numbers' quality columns are deterministic given the default
parameters (fixed RNG seed in the example); when benching real arrays,
multi-seed stability is your call.

## Design summary

1. **Additive index family.** `Rank`, `RankQuant`,
   `Bitmap`, and `SignBitmap` are independent types,
   compiled and tested alongside one another.
2. **No heavy dependencies.** The rank primitives use `rayon` plus
   internal finite-`f32` ordering helpers. No BLAS, no codebook
   training, no rotation matrix.
3. **Build-speed advantage.** Encode is fast and data-independent
   because there is no rotation matmul and no codebook fit — the
   per-vector cost is the `argsort`.
4. **Recall is corpus-dependent.** The generated Gaussian fixture is a
   stress test, not the lead quality claim. On the current real arXiv
   embedding task, full ordinal rank-cosine is within bootstrap noise
   of dense exact search, and RankQuant b=2 asym matches the tested
   FAISS HNSW configuration within bootstrap noise. Run the
   external-corpus bench on your data — see above.
5. **The audit-by-removal rationale.** RankQuant removes training,
   rotation, codebooks, and per-document norms from the pipeline. That
   retrieval still works after the removal is the interesting result:
   on the corpora tested, those components were carrying less than the
   dense-quantization literature assumes.
6. **Formal boundary.** The Lean result supports the constant-weight
   bitmap overlap admission model and its idealized null calibration.
   It does not replace real-corpus recall, null-fit, or monotonicity
   checks for a deployed encoder.
