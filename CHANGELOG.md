# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.3.0 - 2026-05-29

### Added

- Added `probe_index_metadata` to inspect persisted `Rank`, `RankQuant`,
  `Bitmap`, and `SignBitmap` headers without allocating payloads.
- Added the repo-local, publish=false `ordvec-manifest` crate with a strict v1
  JSON schema, artifact and row-identity verification, attestation shape
  checks, a CLI, and optional SQLite cache/audit support with one active
  manifest pointer.
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
substrate for vector retrieval, developed within the
[turbovec](https://github.com/RyanCodrai/turbovec) project (MIT, by Ryan
Codrai) and factored out here as a standalone crate. It is data-oblivious (no
training, rotation, or codebook), uses analytical norms, and carries **no
system dependencies** — no BLAS, no `ndarray`, no `faer`.

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

[0.2.0]: https://github.com/Fieldnote-Echo/ordvec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Fieldnote-Echo/ordvec/releases/tag/v0.1.0
