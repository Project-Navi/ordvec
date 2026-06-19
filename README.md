# ordvec

[![CI](https://github.com/Fieldnote-Echo/ordvec/actions/workflows/ci.yml/badge.svg)](https://github.com/Fieldnote-Echo/ordvec/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](#minimum-supported-rust-version)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Fieldnote-Echo/ordvec/badge)](https://scorecard.dev/viewer/?uri=github.com/Fieldnote-Echo/ordvec)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12977/badge)](https://www.bestpractices.dev/projects/12977)
[![codecov](https://codecov.io/gh/Fieldnote-Echo/ordvec/graph/badge.svg)](https://codecov.io/gh/Fieldnote-Echo/ordvec)

[![Crates.io](https://img.shields.io/crates/v/ordvec.svg)](https://crates.io/crates/ordvec)
[![docs.rs](https://docs.rs/ordvec/badge.svg)](https://docs.rs/ordvec)

Training-free ordinal & sign quantization for vector retrieval.

`ordvec` is a small, pure-Rust crate for compressed nearest-neighbour search
that quantizes the **ordinal (rank) and sign structure** of an embedding —
no codebook, no learned rotation, no graph to build.

## What is ordinal retrieval?

Ordinal retrieval is a retrieval family where the index operates on order/sign
quotients of embedding coordinates rather than metric magnitudes, with
training-free compression, quotient-aware candidate generation, and asymmetric
reranking against the original query signal.

Dense retrieval assumes magnitude geometry is the primitive. Ordinal retrieval
asks what survives when embeddings are quotiented down to coordinate order,
signs, and relative salience. Surprisingly, enough retrieval signal survives to
match dense qrel quality at much smaller storage, and those quotients support
quotient-aware candidate-generation rules that are training-free,
append-friendly, and graph-optional.

## Benchmark at a glance

> **ordvec matches dense retrieval quality within BEIR qrel noise at 8–16× smaller
> vector storage — with no training and no graph build — and sub-millisecond
> single-query retrieval on 171K Harrier embeddings. A threaded HNSW graph still
> wins highly-parallel batched serving; ordvec wins the lightweight
> compressed-substrate lane.**

On **trec-covid** (171,332 documents, the public [BEIR](https://github.com/beir-cellar/beir)
benchmark) with **Harrier-Q8** 1024-d embeddings, ordvec's two-stage retrieval
keeps a near-flat per-query cost as the corpus grows, while exact brute-force
(`flat`, identical math to FAISS `IndexFlatIP`) is O(n) — so the speedup
*widens* with scale:

![ordvec speedup over exact search grows with corpus size](https://raw.githubusercontent.com/Fieldnote-Echo/ordvec/main/benchmarks/beir/figures/scaling_curve.png)

- **~100× faster, single query.** At 171K docs, single-query latency: exact
  `flat` 56 ms vs ordvec `Sign→rq2` **0.53 ms** — and the gap grows with the
  corpus (it is ~5× at 1K docs).
- **8–16× smaller.** 256–384 bytes/vector vs 4096 for full float, at
  **nDCG@10 within bootstrap noise of exact** (on trec-covid the ordinal rows
  even edge ahead; see [Benchmarks](#benchmarks)).
- **Reproducible on your machine, one command:**

  ```sh
  make bench-beir-setup     # Python deps + CUDA llama-cpp-python (GGUF Q8 encoder)
  make benchmark-beir       # download BEIR, embed, run all methods, render graphics
  ```

  The figures and result tables in this README were produced by that command on
  public BEIR data: the harness writes the figures and the nDCG/timing summaries,
  the README tables transcribe those outputs, and you can regenerate or verify
  every number yourself (exact latencies vary with hardware and batch size). The
  default run reproduces **scifact + trec-covid**; the harness also supports
  `nfcorpus` and `fiqa`. Latency for every method is measured in **one Rust
  process** (no Python/FFI in the hot path); see the [Benchmarks](#benchmarks)
  section for the single-query, batched-throughput, and threaded views and their
  caveats.

## What's different

Compressed-retrieval paths almost all carry a **fit step**: product
quantization fits a k-means codebook, OPQ adds a learned rotation,
scalar / binary quantizers calibrate to the data distribution, graph indexes
(HNSW) build a navigable graph, and Matryoshka needs a model trained with its
loss. ordvec fits **none** of them — it quantizes the *ordinal and sign*
structure of each vector on its own:

- **Training-free, data-oblivious.** No codebook, no learned rotation, no
  fit step. Encoding is a per-vector rank (or sign) transform — index the
  very first vector with no prior data, and never refit when the corpus
  drifts.
- **Zero system dependencies.** Pure Rust — no BLAS / LAPACK / `ndarray` /
  `faer`. Builds and cross-compiles cleanly, including to `aarch64` and
  `wasm32`.
- **Ordinal + sign quantization.** Compresses the *rank order* of
  coordinates (1/2/4 bits each) and their signs — a different lever from
  the product / scalar / binary quantization most crates use.
- **Predictable footprint.** Exactly `dim * bits / 8` bytes per document —
  known before you see any data (256 B at dim = 1024, 2-bit), with
  `bits ∈ {1, 2, 4}` the size/recall knob. (`b = 8` is an opt-in
  evidence/refinement width — asymmetric scoring at any dim, symmetric only
  when `dim % 256 == 0` — not a broad retrieval mode.)
- **Two-stage retrieval, built in.** A cheap bitmap / sign-popcount
  prefilter feeds an exact rerank — the coarse→fine pipeline ships as
  library primitives. The coarse-scan→exact-rerank pattern, and the
  `RankQuantFastscan` block-32 4-bit LUT path, follow the FAISS FastScan and
  binary-quantization-plus-rescore lineage; ordvec ships them
  batteries-included and dependency-free, not as new techniques.

ordvec is a compressed **flat-scan** substrate (optionally two-stage): small
codes scored by fast SIMD — AVX-512/AVX2 runtime-dispatched on x86_64, baseline
NEON on aarch64, and `simd128` on wasm32. It
is the code-and-scan layer, not a navigable-graph index — but the codes are
small and index-agnostic, so they compose *under* an ANN or sharding layer for
large-scale serving rather than competing with one.

## Ordinal index family

- **`Rank`** — full-precision rank vectors (`u16` per coordinate).
- **`RankQuant`** — ranks bucketed into `1 << bits` equal-width
  bins, `bits` bits per coordinate (`dim * bits / 8` bytes/doc). Both a
  symmetric (Spearman) and asymmetric (float-query LUT) scorer.
- **`Bitmap`** — a top-bucket bitmap per document (one bit per
  coordinate); scoring is `popcount(Q AND D)`, a coarsened rank overlap.
- **`SignBitmap`** — a sign bitmap per document for sign-cosine
  candidate generation, feeding an exact rerank stage.

Two further paths, for callers who need them:

- **`RankQuantFastscan`** — a stable, documented *but specialized* public
  type: an optional b=2 FastScan kernel (block-32 nibble/PQ-LUT, AVX-512 → AVX2
  → scalar dispatch) for absolute-minimum stage-1 scan latency, at 2× the
  RankQuant b=2 footprint (`dim/2` bytes/doc) and 8-bit LUT scoring noise. It
  persists to `.ovfs` (magic `OVFS`). Reach for it only when scan latency at
  b=2 is the binding constraint; the headline retrieval surface is still
  `RankQuant` / `Bitmap` / two-stage.
- **`MultiBucketBitmap`** *(behind `--features experimental`)* — the
  multi-bucket bilinear-overlap probe behind the research-side decomposition;
  an algebraic scaffold, not the top-bucket theorem surface or a production
  path.

## The bitmap prefilter has a checked finite model

The `Bitmap` prefilter scores candidates by `popcount(Q AND D)` over each
document's fixed-size top-bucket set. In the idealized uniform constant-weight
null, two unrelated `n_top`-active bitmaps in `dim` coordinates overlap
**hypergeometrically**, `H(dim, n_top, n_top)`, with expected overlap
`n_top² / dim` (e.g. 16 at `dim = 256`, `n_top = 64`). That makes the null
selectivity of an overlap cutoff closed-form.

The current proof story is stronger than a closed-form null alone. Two pieces
are machine-checked in Lean 4, both `sorry`-free on Lean's standard axiom base
(`propext`, `Classical.choice`, `Quot.sound`):

- the **ordinal invariance** on which the rank transform rests — that a vector's
  sorting permutation is unchanged by any strictly monotone reparametrisation
  of its coordinates — in
  [`takens-formalization`](https://github.com/Project-Navi/takens-formalization)
  (theorem `isOrdinalPatternOf_comp_strictMono`); and
- the **finite constant-weight bitmap admission model** — symmetry makes
  literal overlap the canonical query-preserving invariant, quotient
  sufficiency reduces the decision to that evidence, a finite overlap-tilt
  signal model makes an overlap-count threshold Bayes-optimal among
  deterministic admission rules, and the uniform constant-weight bitmap null
  assigns that same threshold event exactly the hypergeometric upper tail — in
  [`ordvec-formalization`](https://github.com/Fieldnote-Echo/ordvec-formalization)
  (theorem `exists_uniformBitmapOverlapTail_finiteBayesRisk_le_and_hypergeomTail`).

This is an *in-model* result. It proves the rule shape and the idealized finite
null under explicit quotient, symmetry, and monotone-overlap assumptions. It
does not prove that real encoders satisfy those assumptions, that the textbook
hypergeometric is every deployment corpus's null, or that ordinal quotients are
representation-complete. Whether true neighbours clear a cutoff remains an
empirical contract to measure.

Details in [`docs/RANK_MODES.md`](docs/RANK_MODES.md).

## Quickstart

```toml
[dependencies]
ordvec = "0.5"

# Or, to track unreleased `main`, use a git dependency instead:
# ordvec = { git = "https://github.com/Fieldnote-Echo/ordvec" }
```

```rust
use ordvec::RankQuant;

let dim = 1024;
let n_docs = 10_000;
let mut index = RankQuant::new(dim, 2);   // 2 bits/coord → 256 bytes/doc

// `add` takes a flat, row-major buffer of `dim * n_docs` f32s.
// Replace this with your real embeddings.
let doc_embeddings: Vec<f32> = vec![0.0; dim * n_docs];
index.add(&doc_embeddings);

// Asymmetric scan: full-precision queries vs bucketed docs (recommended).
let query_embeddings: Vec<f32> = vec![0.0; dim * 4]; // 4 queries, row-major
let results = index.search_asymmetric(&query_embeddings, 10);

let top_ids = results.indices_for_query(0);     // top-10 doc ids for query 0
let top_scores = results.scores_for_query(0);
```

For the two-stage compressed-scan path (`Bitmap` / `SignBitmap` candidate
generation → `RankQuant` rerank) and the full mode comparison, see
[`docs/RANK_MODES.md`](docs/RANK_MODES.md).

### Caller-owned serial two-stage (DB / runtime integration)

For runtimes that own their own parallelism — an embedded vector DB driving a
bounded thread pool, or a binding releasing the GIL — ordvec exposes a
**no-rayon** serial two-stage path so the *caller* schedules the work, with an
**allocation-free rerank step** (`_into`, on the AVX-512/AVX2 path) for the
steady-state hot loop:

```rust
use ordvec::{RankQuant, SignBitmap, SubsetScratch};
// Shape sketch (not standalone): `rq: RankQuant` and `sign: SignBitmap` are
// built and `add`-ed as in the Quickstart above; `queries` is your flat
// `dim * nq` f32 batch, `m` the shortlist size, `k` the top-k.
// Stage 1 — serial CSR candidate generation (never enters rayon):
let cb = sign.top_m_candidates_batched_serial_csr(&queries, m); // CandidateBatch { offsets, candidates }
// Stage 2 — rerank into CALLER-OWNED buffers with a reusable scratch:
let nq = queries.len() / dim;
let out_k = k.min(rq.len());
let mut scratch = SubsetScratch::new();               // reuse across batches
let mut out_scores = vec![f32::NEG_INFINITY; nq * out_k];
let mut out_indices = vec![-1i64; nq * out_k];
rq.search_asymmetric_subset_batched_serial_into(
    &queries, &cb.offsets, &cb.candidates, k,
    &mut scratch, &mut out_scores, &mut out_indices,
);
```

Contract: candidates are **CSR** (`offsets.len() == nq + 1`; row `qi` is
`candidates[offsets[qi]..offsets[qi+1]]`; rows need **not** be sorted). Output is
**rectangular** `nq * out_k` and **sentinel-padded** (`-1` / `NEG_INFINITY`) for
underfull rows — size both buffers to `nq * k.min(index.len())`. Scores, row ids,
and the deterministic tie policy (`score desc, global row-id asc`) match the
single-query `search_asymmetric_subset`. **Only the `_into` rerank step is
allocation-free** — on the **AVX-512 / AVX2** SIMD path, and only on repeated
calls of the *same* batch shape — reusing the warmed `SubsetScratch` and your
output buffers (no per-row alloc, no whole-buffer preclear). The scalar fallback
(no AVX2, e.g. aarch64) allocates a per-query scoring LUT. Stage 1
(`top_m_candidates_batched_serial_csr`) also allocates a fresh `CandidateBatch`
each call. Neither primitive enters rayon —
partition the query batch and call `_into` once per worker range from your own
pool. A focused decomposition benchmark lives in
[`examples/two_stage_bench.rs`](examples/two_stage_bench.rs).

### Python

The same `Rank` / `RankQuant` / `Bitmap` / `SignBitmap` API is available from
Python — the bindings ship to PyPI as `ordvec`:

```bash
pip install ordvec
```

Wheels target CPython 3.10+ (abi3); to build from source instead, see
[`ordvec-python/`](https://github.com/Fieldnote-Echo/ordvec/tree/main/ordvec-python).
The runtime dependency floor is `numpy>=2.2`.

### Threading / concurrency

`ordvec` supports concurrent read-only/search use. Mutation is exclusive.
The consolidated cross-language ownership and lifetime contract is in
[`docs/bindings-safety.md`](docs/bindings-safety.md).

Python search, candidate-generation, and scoring methods release the GIL and
read NumPy inputs in place. Callers must not mutate query, corpus, candidate,
or scoring input arrays passed to those methods until the call returns.

The C ABI allows concurrent search and info calls on one loaded handle.
`ordvec_index_free` must not race with any other call on the same handle.

The Go wrapper serializes `Close` against `Search` and `Info`; after `Close`,
`Search` and `Info` return `ErrClosed`. Callers must not mutate query or
candidate slices passed to `Search` until the call returns.

## Documentation

- **Design deep-dive & reproducible benchmark tables:**
  [`docs/RANK_MODES.md`](docs/RANK_MODES.md)
- **Design alternatives evaluated and cut:**
  [`docs/ALTERNATIVES_CONSIDERED.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/ALTERNATIVES_CONSIDERED.md)
- **Index-file trust model:**
  [`docs/INDEX_PROVENANCE.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/INDEX_PROVENANCE.md),
  [`docs/determinism.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/determinism.md),
  [`THREAT_MODEL.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/THREAT_MODEL.md)
- **Manifest verifier, C ABI, and Go wrapper:**
  `ordvec-manifest` is versioned and published in lockstep with the core crate
  through its own package gate; use the GitHub checkout for `ordvec-ffi/`,
  `ordvec-go/`, and
  [`docs/c-api.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/c-api.md).
- **Bindings safety and ownership contract:**
  [`docs/bindings-safety.md`](docs/bindings-safety.md)
- **Artifact and platform matrix:**
  [`docs/artifact-platform-matrix.md`](docs/artifact-platform-matrix.md)
- **Pre-1.0 compatibility policy:**
  [`docs/compatibility-policy.md`](docs/compatibility-policy.md) defines the
  stable, experimental, repo-local sidecar, persisted-format, examples/docs,
  MSRV, and release-note review surfaces.
- **Formal proof spine:** [`ordvec-formalization`](https://github.com/Fieldnote-Echo/ordvec-formalization),
  including its [`proof-spine`](https://github.com/Fieldnote-Echo/ordvec-formalization/blob/main/docs/proof-spine.md),
  [`theorem-map`](https://github.com/Fieldnote-Echo/ordvec-formalization/blob/main/docs/theorem-map.md),
  and [`reviewer brief`](https://github.com/Fieldnote-Echo/ordvec-formalization/blob/main/docs/reviewer-brief.md).
- **API docs:** <https://docs.rs/ordvec>, <https://docs.rs/ordvec-manifest>
- **Paper (OrdVec / RankQuant):** _link TBD — see
  [Research collaboration](#research-collaboration)._

## Benchmarks

### BEIR retrieval (public datasets, reproducible)

A fully reproducible harness over standard [BEIR](https://github.com/beir-cellar/beir)
datasets lives in [`benchmarks/beir/`](https://github.com/Fieldnote-Echo/ordvec/tree/main/benchmarks/beir). It embeds the corpus
with **Harrier-Q8** (GGUF `Q8_0` via `llama-cpp-python`, CUDA), then measures
ordvec's methods against two references **in a single Rust process** so the
latency comparison is genuinely apples-to-apples — same machine, batch, and
thread count, no Python/FFI in the hot path:

- **`flat`** — exact inner-product brute force (identical retrieval to FAISS
  `IndexFlatIP`), a pure-Rust SIMD GEMM. *Baseline, not ground truth.*
- **`hnsw`** — pure-Rust HNSW (`hnsw_rs`, M=32, ef_construction=200,
  ef_search=128) — the portable stand-in for the C++ hnswlib.

Reproduce end-to-end (downloads the data, embeds, runs every method, renders the
figures, and emits the summary tables transcribed below):

```sh
make bench-beir-setup      # Python deps + CUDA llama-cpp-python
make benchmark-beir        # quality (nDCG) + scaling sweep + graphics
```

#### Quality — nDCG@10 vs the official BEIR qrels

nDCG@10 is computed against the human-annotated qrels (not against `flat`).
`Δ vs flat` is the paired-bootstrap mean delta; `*` marks a 95% CI that straddles
0 (i.e. within noise of exact). `flat` and the ordvec rows are **deterministic**
(byte-identical run to run); the `hnsw` row is **approximate** — its graph is
built in parallel, so its nDCG and latency vary slightly between runs (≈±0.003
nDCG here, within the same noise band). The numbers below are one representative
run; regenerate your own with `make benchmark-beir`.

| Dataset | Method | Bytes/vec | nDCG@10 | Δ vs flat (95% CI) |
|---|---|--:|--:|---|
| scifact (5,183) | `flat` (exact) | 4096 | 0.7551 | (baseline) |
| | `hnsw` M=32 | 4096 | 0.7554 | +0.0003 * |
| | **ordvec rq4** | **512** | **0.7549** | −0.0003 * |
| | ordvec rq2 | 256 | 0.7471 | −0.0080 * |
| | ordvec sign→rq2 | 384 | 0.7471 | −0.0080 * |
| trec-covid (171,332) | `flat` (exact) | 4096 | 0.7574 | (baseline) |
| | `hnsw` M=32 | 4096 | 0.7555 | −0.0019 * |
| | ordvec rq2 | 256 | 0.7632 | +0.0057 * |
| | **ordvec rq4** | **512** | **0.7636** | +0.0062 * |
| | ordvec sign→rq2 | 384 | 0.7638 | +0.0064 * |

Every ordvec row is within bootstrap noise of exact dense at **8–16× smaller**
storage; on trec-covid the ordinal codes even edge slightly ahead.

#### Latency — three honest views

ordvec never touches the float corpus, so its per-query cost is tiny and grows
slowly with `n`; `flat`'s cost is dominated by streaming the 4096-byte vectors,
which is O(n) and **memory-bandwidth-bound**. That single fact explains all three
views (trec-covid, 171,332 docs, 1024-d):

**1. Single query (batch = 1, 1 thread)** — latency-sensitive serving, where
`flat` cannot amortize its memory traffic:

![single-query latency bars](https://raw.githubusercontent.com/Fieldnote-Echo/ordvec/main/benchmarks/beir/figures/bars_single_thread.png)

`flat` 56 ms → ordvec `sign→rq2` **0.53 ms (≈106×)**, `bitmap→rq2` 0.62 ms (≈91×),
`hnsw` 1.5 ms (37×). The scaling curve [above](#benchmark-at-a-glance) is this
view swept over corpus size — the speedup *grows* with `n`.

**2. Batched throughput (batch = 32, 1 thread)** — when many queries arrive at
once, `flat`'s GEMM amortizes the corpus stream across the batch (56→4 ms),
narrowing the gap: ordvec `sign→rq2`/`bitmap→rq2` stay ≈8–9.5× ahead.

**3. Many cores (batch = 32, 32 threads)** — everything parallelizes and the
field compresses; `hnsw` threads best:

![threaded throughput bars](https://raw.githubusercontent.com/Fieldnote-Echo/ordvec/main/benchmarks/beir/figures/bars_threaded.png)

`hnsw` 4.8× vs `flat`, ordvec `bitmap→rq2` 3.7×, `rq2` 2.5×, `sign→rq2` 2.1×.
**HNSW wins this regime** — by a hair on threaded throughput. The honest
ordvec-vs-HNSW tradeoff, all from this same run (trec-covid, 171,332 docs):

| | HNSW M=32 | ordvec `sign→rq2` |
|---|---|---|
| threaded latency (32 threads, batch 32) | **0.23 ms** ✅ | 0.52 ms |
| single-query latency (batch 1) | 1.52 ms | **0.53 ms** ✅ (~3×) |
| index size / vector | 4096 B + graph | **256–384 B** ✅ (8–16× less) |
| build time, 171K docs | **51.4 s** | **0.26 s** ✅ (training-free) |
| nDCG@10 (trec-covid) | 0.7555 | **0.7638** ✅ |

So even where HNSW edges ahead on threaded latency, ordvec gets there with **no
graph to build** (instant, training-free, and rebuilt for free when the corpus
drifts) and **8–16× less memory** — and it still wins single-query latency and
ties or edges quality. And the two aren't mutually exclusive: ordvec's codes are
index-agnostic, so they compose *under* an HNSW/sharding layer (see
[Scope](#scope)) rather than replacing it.

**Read it honestly:** ordvec's huge latency win is a single-query / low-batch
phenomenon (and grows with corpus size); under large-batch throughput a batched
exact GEMM is a strong baseline and HNSW threads very well. The durable wins are
**compression at iso-quality** and **single-query latency that stays flat as the
corpus grows**. `flat` is a comparison reference, not ground truth; nDCG@10 is
the qrel-based metric. Numbers vary with encoder, dataset, hardware, and batch —
the point is that you can regenerate all of them with `make benchmark-beir`.

### Synthetic stress test

The head-to-head benchmark generates a seeded synthetic corpus in-process, so
the **quality numbers (R@10, candidate-recall, bytes/vec) are deterministic**
and regenerable from a clean checkout with no external corpus file:

```sh
cargo run --release --example bench_rank
```

A few operating points from the committed run
([`benchmarks/rank_modes_results.txt`](benchmarks/rank_modes_results.txt)):

| Mode | bytes/vec | p50 (ms) | Mdocs/s | R@10 |
|------|----------:|---------:|--------:|-----:|
| `Rank` asym (full-precision reference) | 512 | 3.71 | 8 | 0.845 |
| `RankQuant` b=4 asym | 128 | 0.31 | 96 | 0.806 |
| `RankQuant` b=2 asym | 64 | 0.24 | 126 | 0.572 |
| `RankQuant` b=2 FastScan | 128 | 0.09 | 333 | 0.570 |
| Two-stage b=2 (M=500, CR=1.000) | 96 | 0.11 | 275 | 0.572 |

*One representative run on a **synthetic** corpus (dim=256, n=30k, seed=1),
AMD Ryzen 9 9950X (AVX-512), 32 threads, single-thread scan. **R@10 is
deterministic** run-to-run; **throughput/latency vary** with hardware and run.
R@10 is measured against FP32 brute-force cosine on this generated corpus. Treat
it as a small, self-contained kernel and stress-test fixture; the real-embedding
retrieval task above is the better guide to retrieval-relevant behaviour.*

## Scope

ordvec is a **library and substrate**, not a turnkey service: small
ordinal/sign codes, fast SIMD scoring, and a built-in two-stage prefilter —
the code-and-scan layer of a retrieval system. It is not a navigable-graph
index (HNSW) on its own — yet — and not a serving tier at all: ordvec is the
substrate other systems build on, so its small, index-agnostic codes slot
**under** an ANN or sharding layer for large-scale serving rather than
replacing it. Encoding is training-free and data-oblivious by design —
no codebook fit — so you index the first vector with no prior data and never
refit as the corpus grows.

Quality evidence now has two layers: the real-embedding retrieval table above
for the paper claim, and the reproducible synthetic stress test for a
clean-checkout kernel sanity check.

## Security: index-file trust

The on-disk formats (`.ovr` / `.ovrq` / `.ovbm` / `.ovsb` / `.ovfs`; legacy
`.tvr` / `.tvrq` / `.tvbm` / `.tvsb` files still load) carry **no built-in
checksum, MAC, or signature — by design.** The loaders validate *structure*
(magic, version, bounds, exact-length payload) but not *origin*: a
structurally valid file can still be untrusted. If an index file crosses a
trust boundary (network transfer, shared storage), verify it before loading.
`ordvec-manifest` binds an index file to a JSON manifest by SHA-256, header
metadata, row identity, named auxiliary sidecars, and attestation shape checks.
It does not sign artifacts, manage keys, or decide deployment trust policy. No
in-format crypto is shipped because it would add key management the library
can't own. See
[`docs/PERSISTED_FORMAT.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/PERSISTED_FORMAT.md),
[`docs/INDEX_PROVENANCE.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/INDEX_PROVENANCE.md),
and [`THREAT_MODEL.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/THREAT_MODEL.md)
in the full repository.

## Provenance

ordvec was developed within turbovec, factored out into this standalone,
zero-system-dependency crate.
[turbovec](https://github.com/RyanCodrai/turbovec) (MIT, by Ryan Codrai)
is credited as the project it grew within, with thanks; ordvec's
development history is in this repository's git log.

## Acknowledgements

Thanks to Todd Baur ([@toadkicker](https://github.com/toadkicker)) for the
sign-cosine intuition and engineering polish.

Thanks to Mike Singleton ([@singleton2787](https://github.com/singleton2787))
for mathematical assistance and mentorship.

## Research collaboration

ordvec is the reference implementation for an in-progress paper on **ordinal
retrieval** — using the rank and sign structure of embeddings, rather than
their floating-point magnitudes, as the retrieval signal. The repository is
open specifically to grow a group of collaborators, **including potential
named co-authorship where contributions meet the paper's authorship bar** —
a different invitation than "send a PR."
Collaboration we're actively seeking:

- **Real-corpus evaluation** — running the modes against public corpora
  (GloVe, MTEB / BEIR, OpenAI embedding dumps) beyond the synthetic benchmark.
- **Theory** — extending and independently auditing the `sorry`-free Lean
  formalization, especially the finite bitmap proof spine, rank-cosine
  invariants, and empirical diagnostics for when real encoders meet or violate
  the model assumptions.
- **Independent reproduction** — re-running the benchmark on other hardware
  and reporting the numbers.

If that's your area, see
[GOVERNANCE.md](https://github.com/Fieldnote-Echo/ordvec/blob/main/GOVERNANCE.md)
and open an issue or a discussion.

## Contributing

Contributions to the code, the docs, and the paper are all welcome — see
[CONTRIBUTING.md](https://github.com/Fieldnote-Echo/ordvec/blob/main/CONTRIBUTING.md).

## Minimum supported Rust version

ordvec's MSRV is **Rust 1.89** — the release that stabilized the specific
AVX-512 intrinsics the SIMD kernels compile against (it also clears the 1.87
floor from `is_multiple_of`). Because the kernels are built against those
intrinsics, this is a hard compile floor, not just a convenience pin: a
toolchain below 1.89 won't build the crate. Raising the MSRV is treated as a
minor-version change under the
[compatibility policy](docs/compatibility-policy.md). The current feature
stability matrix and downstream embedding notes live in
[`docs/msrv-and-features.md`](docs/msrv-and-features.md).

## License

Licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
