# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance

- **Batched sign candidate generation now streams the corpus once per call.**
  `SignBitmap::top_m_candidates_batched_serial_csr` previously looped the
  single-query path, re-streaming the full sign bitmap per query (the
  documented-naive first cut). The internals now scan the corpus once per call
  in L2-sized doc blocks, score every query of the call against each hot block
  in query tiles via the existing batched kernel, and select per-query top-m
  with bounded `(hamming, doc_id)` min-collectors — bit-identical to a full
  sort by construction, independent of processing order (the key *is* the
  contract's sort key), pinned by an independent oracle suite
  (`tests/tiled_candgen.rs`) across random, tie-heavy, duplicate-run, and edge
  geometries. Per-query corpus traffic drops by the call's query count: at
  1.26M rows × dim=1024, a 2048-query call reads the 161 MB sign sidecar once
  instead of 2048 times. `top_m_candidates` routes through the same core
  (dropping its per-call n-row Hamming materialisation) except at `nq=1`,
  which keeps the dense partition path — the streamed core measured +50–90%
  single-query time at small/medium `n` with `m` in the hundreds (bounded heap
  `O(n log m)` vs `select_nth_unstable_by` `O(n)`), while the dense path is
  parity-or-better at every measured size. The serial contract is preserved
  (no rayon). Together with the collector worst-bound change below, measured
  downstream in a two-stage retrieval stack at 1.26M × 1024: batched search
  throughput 220 → 10.2k queries/s, results bit-identical.
- **Candidate-collector accept test reduced to a cached worst-bound compare.**
  Doc ids visit each per-query heap strictly ascending, so a candidate tying
  the worst kept hamming always loses the `(hamming, doc_id)` tie-break — once
  the collector is full, the accept test is exactly `hamming < worst kept
  hamming`. That bound is now cached in a register-friendly `u32` (`u32::MAX`
  while filling), skipping the heap peek + tuple compare on the ~99.8% reject
  path. Bit-identical by construction; pinned by the tie-heavy and
  duplicate-run oracle suites.
- **Parallel finite-input validation and scratch-based rank encode.**
  `assert_all_finite` paid a full serial pass per add/search batch — measured
  ~0.1 s per GiB, twice per ingest batch counting the caller layer. Scans of
  1M+ floats now split across the rayon pool (4.4× measured).
  `RankQuant::add`'s per-row closure allocated a fresh ranks `Vec` per vector
  inside the parallel loop; it now reuses a per-worker scratch via
  `rank_transform_into`. Measured on a 1.26M × 1024 corpus slice: encode-path
  validation attribution 0.097 s serial scan → 0.022 s parallel, with the
  per-vector allocation churn removed from the hot loop.
- **LUT + parallel constant-composition check on `RankQuant` load.**
  `load_rankquant`'s forged-buffer defense histogrammed every packed code
  serially — 1.29 billion shift/mask ops at 1.26M × 1024, ~1 s of the 1.27 s
  verified open. A 4 KB per-byte bucket-count LUT replaces the per-code inner
  loop and rows validate in parallel; `find_first` keeps the
  lowest-offending-row error contract, with a scalar recheck producing the
  identical message. The security property is unchanged: every row still
  proves uniform composition before the index is usable. Measured verified
  open at 1.26M × 1024: 1.27 s → 0.38 s.

### Changed

- **ordvec-manifest: derived artifact size bounds.** Verification now bounds
  every artifact read by its manifest-declared `file_size_bytes` (the manifest
  itself remains hard-capped at 1 MiB and SHA-256 pins content); manifest
  creation bounds reads by the artifact's observed size. The flat
  `ResourceLimits` byte caps (`max_auxiliary_artifact_bytes`,
  `max_calibration_profile_bytes`, `max_encoder_distortion_profile_bytes`)
  are now explicit opt-in ceilings and default to unbounded — previously the
  64 MiB auxiliary default made legitimate large sign sidecars (>524,288 rows
  at dim=1024) impossible to write with default options.
- **ordvec-manifest: primary artifact reads are now bounded.** The primary
  index artifact is hashed under its declared size (new
  `artifact_file_too_large` reason code); previously this read was unbounded.
  An artifact grown past its declaration now fails fast at the read bound
  instead of surfacing as a digest mismatch after hashing the excess.
- **ordvec-manifest: primary index artifact gains an opt-in ceiling.** New
  `ResourceLimits::max_index_artifact_bytes` (default unbounded) mirrors the
  auxiliary/profile ceilings; the create path also bounds the primary read by
  its observed size. Note: a grown artifact now surfaces as
  `*_file_too_large` (fail-fast) rather than `*_file_size_mismatch`, which
  now indicates truncation only.
- **ordvec-manifest: bounded hashing streams with constant memory.**
  `sha256_file_bounded` no longer materialises the file in memory before
  hashing.

## 0.5.0 - 2026-06-19

### Security

- Hardened the Python binding's GIL-released search, candidate, scoring, and
  `add` paths: NumPy inputs are now copied into Rust-owned buffers before
  `py.detach`, so safe Python code cannot race a detached Rust read by mutating
  the same array from another thread. This intentionally trades zero-copy
  detached reads for race-free copied inputs; large calls may temporarily require
  an additional input-sized buffer.
- Updated release governance to document and audit the two-approver
  `crates-io` / `pypi` GitHub Environment gates: `Fieldnote-Echo` and
  `toadkicker` are listed as required reviewers, self-review is blocked, and a
  30-minute wait timer applies before registry publish jobs can proceed.
- Exposed the calibration-profile byte limit through the `ordvec-manifest`
  Python bindings, including the default constant, `default_resource_limits()`,
  and verifier/create keyword arguments.
- Aligned `.ovfs` / `OVFS` security and provenance docs with the now-public
  `RankQuantFastscan` persistence loader and fuzz target.
- **Hardened `.ovfs` FastScan loading before the format's first stable
  release.** `RankQuantFastscan` now rejects invalid FastScan payload bytes
  (`byte & 0xf0 != 0`), rows that violate b=2 constant composition, and
  nonzero block-tail padding across the path, reader, and byte-slice load APIs.
  Loader fuzzing now runs a safe `search()` after every successful `.ovfs` load,
  and persisted-input tests compare the dispatch path against the scalar
  FastScan reference (AVX-512 under SDE, scalar otherwise).
- **Bounded calibration-profile hashing in `ordvec-manifest`.** Verification now
  applies `max_calibration_profile_bytes` (64 MiB by default, CLI-overridable)
  before hashing calibration profile artifacts, matching the existing bounded
  resource model for encoder-distortion profiles and auxiliary artifacts.
- **Cleared OSV / OpenSSF-Scorecard advisories on the dev-only BEIR benchmark
  tooling** (introduced with the benchmark harness; none reach the published
  `ordvec` crate or the `ordvec` PyPI wheel). The `benchmarks/beir/requirements.txt`
  deps were unpinned, so OSV flagged each against its full historical CVE list;
  they are now lower-bound-pinned at the first patched release (`requests>=2.33.0`,
  `hnswlib>=0.8.0`, `numpy>=1.26`, plus safe floors for the rest). `bincode` 1.x
  (RUSTSEC-2025-0141, *unmaintained* — not a vulnerability) enters only
  transitively via `hnsw_rs` in `benchmarks/beir-bench` and is absent from
  `cargo tree -p ordvec`; it is triaged with a documented `deny.toml` ignore.

### Added

- **Reproducible BEIR benchmark harness** (`make benchmark-beir`; dev-only,
  excluded from the published crate). All latency is measured in a single Rust
  process (`benchmarks/beir-bench`) — ordvec's rank/sign methods against an exact
  inner-product baseline (`flat`, identical retrieval to FAISS `IndexFlatIP`, via
  a pure-Rust SIMD GEMM) and a pure-Rust HNSW (`hnsw_rs`, M=32) — so the
  comparison is apples-to-apples (same machine, batch, thread count, no
  Python/FFI in the hot path). Covers single-query / batched / 32-thread regimes
  and a corpus-size scaling sweep on public BEIR datasets, with the corpus
  embedded by Harrier-Q8 (GGUF `Q8_0` via `llama-cpp-python`, CUDA). The README
  now leads with the resulting scaling curve, latency bars, and nDCG@10 table;
  every figure is regenerated by the harness and the README tables transcribe
  its summary outputs. Replaces the previous private-arXiv real-embedding
  numbers in the README.
- **`RankQuantFastscan` is now a stable, public API** (previously re-exported
  `#[doc(hidden)]`), with `.ovfs` / `OVFS` persistence via
  `RankQuantFastscan::{write,load}` and a ninth `load_fastscan` cargo-fuzz
  target. Metadata-probe support (`probe_index_metadata`) and
  `ordvec-manifest` v1 support for `.ovfs` are deferred to 0.8.0 (#233, #232);
  bind `.ovfs` artifacts with caller-owned checksums or attestations when they
  cross a trust boundary.
- **Caller-owned serial batched/buffered two-stage primitives** (additive):
  `SignBitmap::top_m_candidates_batched_serial_csr`, `CandidateBatch`,
  `SubsetScratch`, `RankQuant::search_asymmetric_subset_batched_serial`, and
  `RankQuant::search_asymmetric_subset_batched_serial_into`. These primitives
  never enter rayon; callers partition query batches and drive the serial
  `_into` primitive from their own scheduler. The serial CSR candidate generator
  is correctness-first in this release; future releases can optimize internals
  behind the same signature.
- `avx512vpop_supported()` (`#[doc(hidden)]`) — reports whether the AVX-512
  VPOPCNTDQ scan kernels are active on the current CPU. The scan dispatch reads
  only this predicate (no per-dimension gate).

### Performance

- **AVX-512 VPOPCNTDQ scan kernels now cover every `dim` (a multiple of 64), not
  just multiples of 512 bits.** Previously the `SignBitmap` and `Bitmap` scan
  kernels took the AVX-512 path only when the per-vector 64-bit word count was a
  multiple of 8 (`dim` a multiple of 512), silently falling back to the scalar
  loop otherwise — so common embedding widths like **768 (BGE) and 384
  (bge-small / MiniLM)** ran the entire stage-1 candidate scan scalar. The
  kernels now process the trailing `(dim / 64) % 8` words with a masked load
  (`_mm512_maskz_loadu_epi64`), so any supported `dim` uses VPOPCNTDQ. Measured
  **~4× faster** stage-1 scan at dim=768 on a Zen5 / AVX-512 host (609 → 153
  µs/query, n=100k; see `examples/bge_kernel_bench`); 1024/1536 unchanged.
  Results are byte-identical to the scalar path — parity tests cover qpv tail
  residues 0..7 plus 384/512/768/1024/1536 for all six SignBitmap/Bitmap scan
  kernels. This is stage-1 scan-kernel throughput, not a whole-pipeline figure.

### Changed

- Updated formalization links and release invariants after the companion
  `ordvec-formalization` repository moved under `Project-Navi`.
- **Clarified BEIR benchmark release claims.** The committed README figures use
  the default method set and do not yet include the newer
  `sign-rq2-threaded` probe row; the docs and plot generator now distinguish
  4096-byte HNSW float-vector storage from implementation-owned graph side
  storage instead of treating the graph as zero.
- **On-disk format magics renamed to `OV*`** (`OVR1` / `OVRQ` / `OVBM` /
  `OVSB`). The loaders still accept the legacy `TV*` magics, so every
  previously-written `.tvr` / `.tvrq` / `.tvbm` / `.tvsb` file continues to load
  unchanged; only the file extensions and magic bytes written by `write()`
  change (#230).
- **Documented the v0.5 `b=8` support boundary.** `b=8` is a stable Rust
  in-memory evidence/refinement width: asymmetric scoring and code/projection
  generation work at any valid dimension, while symmetric `RankQuant::search`
  requires `dim % 256 == 0`. It is not exposed through the Python `RankQuant`
  constructor in v0.5.0, cannot be persisted to `.ovrq`, and each prepared
  asymmetric query/worker owns a `dim * 256` `f32` LUT (about 64 MiB at the
  maximum dimension).
- **Release-hardened the caller-owned serial two-stage primitives** (no API
  change; added in 0.5.0). The trust model is now explicit and tested:
  - Rejection-path regression tests for the full CSR/query/buffer validation set
    on the rerank entry points — overlong row (the guard that bounds the unsafe
    gather), non-monotonic / wrong-final / non-zero-first offsets, non-finite and
    ragged queries, and wrong output-buffer length — so a malformed-but-accepted
    input can never reach the SIMD scan.
  - A counting-allocator test proving `search_asymmetric_subset_batched_serial_into`
    performs **zero heap allocations** in steady state (warmed `SubsetScratch`,
    reused caller buffers, including the scalar LUT scratch) across the rerank
    paths — the strong form of the prior capacity-stability proxy.
  - A focused `two_stage_bench` example decomposing stage-1 candidate-gen /
    single-query rerank loop / batched `_into` / full two-stage at the
    Harrier-1024 shape, with a committed reference capture
    (`benchmarks/two_stage_caller_owned_dim1024.txt`, SYNTHETIC corpus).
  - User-facing docs for the caller-owned / no-rayon / allocation-free contract
    (README + rustdoc examples on the `_into` hot path and the CSR candidate-gen).

### Fixed

- Added a persisted-format registry that drives probe, manifest-coverage, and
  C-ABI load decisions from one table; `.ovfs` now remains explicitly
  known-but-not-probeable/not-manifest-covered, and the C ABI reports it as an
  unsupported format rather than a corrupt index.
- Hid the `SubsetScratch::capacities_for_test` helper behind the non-default
  `test-utils` feature and cleaned stale release-doc comments around FastScan
  and b=8 bucket rustdoc.
- **Made Intel SDE AVX-512 coverage fail closed for release publishes.** Pull
  requests and main pushes may emit a visible warning and skip SDE-dependent
  steps during an Intel mirror outage, but the tag-triggered release workflow
  reruns a fail-closed SDE proof before staging release assets; setup must
  succeed, the AVX-512 CPUID probe must run, and SDE-backed tests must execute
  before publish.
- **Closed manifest verifier path-reopen drift.** Verification and SQLite
  cache-key construction now hash, probe, and validate the canonical path that
  was checked and recorded, rather than reopening the pre-canonical joined path.
- **Marked persisted-format metadata enums non-exhaustive before v0.5 ships.**
  `IndexKind`, `IndexParams`, `ManifestIndexKind`, and `ManifestIndexParams`
  are now future-extensible for later stable formats such as `.ovfs` manifest
  support without forcing downstream exhaustive matches.
- **Corrected FastScan dispatch documentation.** `RankQuantFastscan` dispatches
  AVX-512 when available and otherwise uses its scalar kernel; the AVX2 path is
  part of the exact `RankQuant` asymmetric scorer, not FastScan.
- **`ordvec-manifest` crate and wheel now ship license text.** Both declared
  `MIT OR Apache-2.0` but packaged no `LICENSE-*` files (a pre-0.5.0 defect);
  added `LICENSE-MIT` + `LICENSE-APACHE-2.0` (copied from the workspace root) to
  `ordvec-manifest/` and `ordvec-manifest-python/`, and made the release-publish
  invariant gate require them for the manifest crate. The PyPI canonical-dist
  helper now also inspects the built `ordvec-manifest` wheel and sdist and fails
  the release unless both license files are present in the archive's canonical
  license location (`*.dist-info/licenses/` for the wheel, the archive root for
  the sdist) — closing the regression class at the published-bytes layer, not
  only at `cargo package`.

## 0.4.0 - 2026-06-04

### Added

- Added a `signbitmap_rankquant_twostage` fuzz target and deterministic tests
  for the SignBitmap candidate generation plus RankQuant subset rerank
  pipeline used by downstream retrieval systems.
- Added lockstep `ordvec-manifest` crate publishing to the unified release
  pipeline, including OIDC trusted publishing, pre/post-publish byte-identity
  checks, and release invariants covering both `.crate` artifacts.
- Added a verifier-only `VerifiedLoadPlan` helper to `ordvec-manifest` so Rust
  callers can verify a manifest, retain the typed report, and load from the
  resolved artifact and sidecar paths without re-resolving manifest strings.
- Added named auxiliary artifact verification to `ordvec-manifest`, including
  required/optional sidecar states, path/size/SHA-256 checks, deterministic
  report entries, and SQLite cache invalidation for declared sidecar bytes.

### Documentation

- Documented the `VerifiedLoadPlan` verify-then-load boundary, including the
  fact that returned paths are not immutable file handles and must not be
  treated as TOCTOU protection on mutable storage.
- Documented duplicate-candidate behavior for `RankQuant` subset reranking and
  the repo-local C ABI / Go wrapper.
- Added a pre-1.0 compatibility policy covering stable and experimental Rust
  APIs, Python bindings, the lockstep Manifest crate, repo-local C/Go sidecars,
  primitive persisted formats, examples/docs, MSRV/feature changes, and
  release-note review expectations.

### Fixed

- Hardened Intel SDE setup caching and release-gate handling so transient Intel
  CDN failures no longer leave AVX-512 checks dependent on one live download.

### Security

- Added bounded parser/report defaults to `ordvec-manifest` verification for
  manifest JSON size, row-identity JSONL line length, row count,
  duplicate-tracking memory, auxiliary artifact declaration count and bytes,
  encoder distortion profile artifact bytes, report issue count, and SQLite
  cached report size.

## 0.3.0 - 2026-05-29

### Added

- Added `probe_index_metadata` to inspect persisted `Rank`, `RankQuant`,
  `Bitmap`, and `SignBitmap` headers without allocating payloads.
- Added the lockstep `ordvec-manifest` crate with a strict v1 JSON schema,
  artifact and row-identity verification, attestation shape checks, a CLI, and
  optional SQLite cache/audit support with one active manifest pointer.
- Added optional typed calibration profile references to the v1 manifest
  schema, with path/hash/identity/compatibility verification but no statistical
  computation.
- Added the repo-local, publish=false `ordvec-ffi` crate with the base C ABI
  for loading persisted `RankQuant` and `Bitmap` indexes and running
  synchronous search through opaque handles.
- Added the repo-local `ordvec-go` cgo wrapper over the base C ABI.

### Documentation

- Reframed bitmap-overlap docs around the checked Lean proof spine: query
  symmetry, quotient sufficiency, finite threshold optimality, and idealized
  hypergeometric calibration, while preserving the real-encoder caveats.
- Documented sidecar manifest verification as a pre-load provenance check that
  does not sign, manage keys, call networks, or decide trust policy.

### Fixed

- Hardened Python `add()` input boundaries so attempts to grow an index beyond
  `MAX_VECTORS` raise `ValueError` before crossing into Rust core asserts.
- Corrected Python package dependency wording to the published metadata:
  CPython 3.10+ with `numpy>=2.2`.

### Security

- Hardened the tag-triggered release workflow with exact Linux/aarch64 wheel
  smoke coverage, post-publish PyPI hash verification, reproducible
  release-required fuzz installation, and stricter local release-order
  invariants for OIDC and publish steps.

## [0.2.0] - 2026-05-26

First public release on crates.io / PyPI — the crate was not published before
this. It pairs the OrdVec ontology rebrand (index types drop the `Index` suffix;
the `rank_index` module flattens into the crate root) with the pre-publish
hardening that followed. The `0.1.0` section below records the pre-publish
internal history.

### Added

- **CI fuzz smoke** (`.github/workflows/fuzz.yml`): a bounded cargo-fuzz run on
  every pull request / push to `main` (60s each over `load_rank`,
  `load_rankquant`, and `fastscan_b2`) plus a weekly full sweep over all seven
  targets, so a loader, write→load round-trip, or FastScan-kernel regression
  surfaces in CI between manual campaigns (THREAT-FUZZ-002). cargo-fuzz is
  version-pinned and the actions are SHA-pinned.

### Changed

- **Type renames** — the `Index` suffix was dropped across the ordinal family:
  `RankIndex` → `Rank`, `RankQuantIndex` → `RankQuant`,
  `BitmapIndex` → `Bitmap`, `SignBitmapIndex` → `SignBitmap`,
  `MultiBucketBitmapIndex` → `MultiBucketBitmap`,
  `RankQuantFastscanIndex` → `RankQuantFastscan`.
- **Module flatten** — the `ordvec::rank_index` submodule was flattened into
  the crate root (the types stay re-exported at `ordvec::*`); the test tree
  mirrors this (`tests/rank_index/` → `tests/index/`).
- **`#![deny(unsafe_op_in_unsafe_fn)]` is now enforced crate-wide** (previously
  only in `fastscan.rs`): every unsafe operation in the `bitmap`, `sign_bitmap`,
  `quant_kernels`, and `util` (NEON) SIMD kernels now sits in an explicit
  `unsafe {}` block, keeping the unsafe surface visible to future edits
  (THREAT-SIMD-001).
- **`rank::rank_to_bucket` rejects `rank >= d`** — it now panics (and the Python
  binding raises `ValueError`) instead of silently clamping the result into
  range, matching the fail-loud contract of `pack_buckets` / `bucket_centre`.
  Valid rank vectors (a permutation of `[0, d)`) are unaffected.
- **Python bindings (`ordvec-python`):** raised the floor to **Python 3.10** and
  **numpy 2.2**; the abi3 wheel target moves to `abi3-py310`. Python 3.9 reached
  end-of-life (October 2025) and pytest's CVE-2025-71176 fix dropped 3.9 support.

### Deprecated

Pre-0.2 names retained as deprecated `pub type` aliases at the crate root (in
`src/lib.rs`): `RankIndex`, `RankQuantIndex`, `BitmapIndex`, `SignBitmapIndex`,
`MultiBucketBitmapIndex` (gated `#[cfg(feature = "experimental")]`),
`RankQuantFastscanIndex` (gated `#[doc(hidden)]`). These cover root imports
(`use ordvec::RankIndex;`) only — not the removed module path (below). Remove
these aliases in a future release.

### Removed

- **`ordvec::rank_index::*` module path** — the public `rank_index` module was
  removed by the flatten, so module-path imports
  (`use ordvec::rank_index::RankQuantIndex;`) no longer resolve. Move to the
  crate-root names (`use ordvec::RankQuant;`) or the deprecated root aliases
  above. Done as pre-release cleanup while the crate was unpublished, so the
  published vocabulary is clean for this first release.

### Security

- Remediated **GHSA-6w46-j5rx-g56g** / CVE-2025-71176 (pytest vulnerable tmpdir
  handling) by moving the dev/test toolchain to pytest 9.0.3 on Python ≥3.10.

### Documentation

- **README pre-release lifts:** a caveated operating-point table from the
  committed synthetic benchmark; the Bitmap prefilter's hypergeometric null;
  a "Security: index-file trust" callout; a "Research collaboration" (paper
  co-authorship) section; a "Scope" section; `RankQuantFastscan` (doc-hidden)
  and the `experimental` `MultiBucketBitmap` surfaced with caveats; and a more
  precise MSRV rationale.

## [0.1.0] - 2026-05-22

Initial release. `ordvec` is the training-free ordinal & sign quantization
substrate for vector retrieval. It was developed using the early
[turbovec](https://github.com/RyanCodrai/turbovec) project context as a
rapid-development scaffold, but ordvec's implementation history lives in this
repository. It is data-oblivious (no training, rotation, or codebook), uses
analytical norms, and carries **no system dependencies** — no BLAS, no
`ndarray`, no `faer`.

### Added

- **`RankIndex`** — full-precision rank vectors (`u16` per coordinate,
  `2 * dim` bytes per document).
- **`RankQuantIndex`** — ranks bucketed into `1 << bits` equal-width bins,
  packed at `bits` bits per coordinate (`dim * bits / 8` bytes per document),
  with both symmetric (Spearman) and asymmetric (float-query LUT) scorers.
- **`BitmapIndex`** — a top-bucket bitmap per document (one bit per
  coordinate); scoring is `popcount(Q AND D)`, a coarsened rank overlap.
- **`SignBitmapIndex`** — a sign bitmap per document for sign-cosine candidate
  generation, feeding an exact rerank stage.
- **`RankQuantFastscanIndex`** — an optional FastScan b=2 scan path
  (`#[doc(hidden)]`, reachable as `ordvec::RankQuantFastscanIndex` for callers
  who opt in).
- Runtime-dispatched SIMD kernels (AVX-512 / AVX2, selected via
  `is_x86_feature_detected!`) with a portable scalar fallback. No feature flag
  gates SIMD.
- Serialization/deserialization for the rank and sign-bitmap indices
  (`rank_io`, sign-bitmap persistence).
- `experimental` feature gating `MultiBucketBitmapIndex`, the bilinear
  bucket-overlap research scaffold, off the stable surface.
- Reproducible in-process benchmark (`examples/bench_rank.rs`,
  `cargo run --release --example bench_rank`) generating a seeded synthetic
  corpus, with a committed capture in `benchmarks/rank_modes_results.txt`.
- Red-team regression suites (alpha/beta/gamma/delta) for the rank-mode substrate.

### Notes

- **Minimum supported Rust version (MSRV): 1.89** — the release in which the
  AVX-512 intrinsics this crate relies on were stabilized.
- Dual-licensed under **MIT OR Apache-2.0**.

[0.4.0]: https://github.com/Project-Navi/ordvec/compare/v0.3.0...v0.4.0
[0.2.0]: https://github.com/Project-Navi/ordvec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Project-Navi/ordvec/releases/tag/v0.1.0
