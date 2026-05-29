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

`ordvec` is a small, dependency-light Rust crate for compressed
nearest-neighbour search over high-dimensional embeddings.

## What's different

Compressed-retrieval libraries usually either **fit a codebook to your
data** (product / scalar quantization) or **wrap vectors in a graph**
(HNSW). ordvec does neither — it quantizes the *ordinal* structure of each
vector on its own:

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
  `bits ∈ {1, 2, 4}` the size/recall knob.
- **Two-stage retrieval, built in.** A cheap bitmap / sign-popcount
  prefilter feeds an exact rerank — the coarse→fine pipeline ships as
  library primitives.

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

- **`RankQuantFastscan`** *(`#[doc(hidden)]` — reachable as
  `ordvec::RankQuantFastscan`, but the API is not yet stable)* — an optional
  b=2 FastScan kernel (block-32 PQ-LUT) for absolute-minimum scan latency, at
  2× the RankQuant b=2 footprint (`dim/2` bytes/doc). Surfaced here so
  latency-critical callers know it exists.
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
ordvec = "0.3"

# Or, to track unreleased `main`, use a git dependency instead:
# ordvec = { git = "https://github.com/Fieldnote-Echo/ordvec" }
```

```rust
use ordvec::RankQuant;

let dim = 1024;
let mut index = RankQuant::new(dim, 2);   // 2 bits/coord → 256 bytes/doc

// `add` takes a flat, row-major buffer of `dim * n_docs` f32s.
index.add(&doc_embeddings);               // &[f32], len = dim * n_docs

// Asymmetric scan: full-precision queries vs bucketed docs (recommended).
let results = index.search_asymmetric(&query_embeddings, 10); // len = dim * n_queries

let top_ids = results.indices_for_query(0);     // top-10 doc ids for query 0
let top_scores = results.scores_for_query(0);
```

For the two-stage compressed-scan path (`Bitmap` / `SignBitmap` candidate
generation → `RankQuant` rerank) and the full mode comparison, see
[`docs/RANK_MODES.md`](docs/RANK_MODES.md).

### Python

The same `Rank` / `RankQuant` / `Bitmap` / `SignBitmap` API is available from
Python — the bindings ship to PyPI as `ordvec`:

```bash
pip install ordvec
```

Wheels target CPython 3.10+ (abi3); to build from source instead, see
[`ordvec-python/`](https://github.com/Fieldnote-Echo/ordvec/tree/main/ordvec-python).
The runtime dependency floor is `numpy>=2.2`.

## Documentation

- **Design deep-dive & reproducible benchmark tables:**
  [`docs/RANK_MODES.md`](docs/RANK_MODES.md)
- **Design alternatives evaluated and cut:**
  [`docs/ALTERNATIVES_CONSIDERED.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/ALTERNATIVES_CONSIDERED.md)
- **Index-file trust model:**
  [`docs/INDEX_PROVENANCE.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/INDEX_PROVENANCE.md),
  [`THREAT_MODEL.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/THREAT_MODEL.md)
- **Repo-local manifest verifier, C ABI, and Go wrapper:**
  available from the full GitHub checkout. These sidecars are not part of the
  published core `.crate`; use the GitHub checkout for `ordvec-manifest/`,
  `ordvec-ffi/`, `ordvec-go/`, and
  [`docs/c-api.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/c-api.md).
- **Formal proof spine:** [`ordvec-formalization`](https://github.com/Fieldnote-Echo/ordvec-formalization),
  including its [`proof-spine`](https://github.com/Fieldnote-Echo/ordvec-formalization/blob/main/docs/proof-spine.md),
  [`theorem-map`](https://github.com/Fieldnote-Echo/ordvec-formalization/blob/main/docs/theorem-map.md),
  and [`reviewer brief`](https://github.com/Fieldnote-Echo/ordvec-formalization/blob/main/docs/reviewer-brief.md).
- **API docs:** <https://docs.rs/ordvec>
- **Paper (OrdVec / RankQuant):** _link TBD — see
  [Research collaboration](#research-collaboration)._

## Reproducible benchmark

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
R@10 is measured against FP32 brute-force cosine on this synthetic corpus —
the broader real-corpus evaluation lives in the paper (in progress).*

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

Quality evidence in this repo is the reproducible synthetic benchmark above;
the broader real-corpus evaluation is in the paper (in progress).

## Security: index-file trust

The on-disk formats (`.tvr` / `.tvrq` / `.tvbm` / `.tvsb`) carry **no built-in
checksum, MAC, or signature — by design.** The loaders validate *structure*
(magic, version, bounds, exact-length payload) but not *origin*: a
structurally valid file can still be untrusted. If an index file crosses a
trust boundary (network transfer, shared storage), verify it before loading.
The full GitHub checkout includes a publish=false sidecar CLI,
`ordvec-manifest`, that binds an index file to a JSON manifest by SHA-256,
header metadata, row identity, and attestation shape checks. It does not sign
artifacts, manage keys, or decide deployment trust policy. No in-format crypto
is shipped because it would add key management the library can't own. See
[`docs/INDEX_PROVENANCE.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/INDEX_PROVENANCE.md)
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

If that's your area, see [GOVERNANCE.md](GOVERNANCE.md) and open an issue or a
discussion.

## Contributing

Contributions to the code, the docs, and the paper are all welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Minimum supported Rust version

ordvec's MSRV is **Rust 1.89** — the release that stabilized the specific
AVX-512 intrinsics the SIMD kernels compile against (it also clears the 1.87
floor from `is_multiple_of`). Because the kernels are built against those
intrinsics, this is a hard compile floor, not just a convenience pin: a
toolchain below 1.89 won't build the crate. Raising the MSRV is treated as a
minor-version change.

## License

Licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
