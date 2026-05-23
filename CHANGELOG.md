# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

## [0.2.0] - 2026-05-22

OrdVec ontology rebrand: index types drop the `Index` suffix; the `rank_index`
module is flattened into the crate root. Deprecated `*Index` aliases are
retained for back-compat.

### Changed

- **Type renames** — the `Index` suffix was dropped across the ordinal family:
  `RankIndex` → `Rank`, `RankQuantIndex` → `RankQuant`,
  `BitmapIndex` → `Bitmap`, `SignBitmapIndex` → `SignBitmap`,
  `MultiBucketBitmapIndex` → `MultiBucketBitmap`,
  `RankQuantFastscanIndex` → `RankQuantFastscan`.
- **Module flatten** — the `ordvec::rank_index` submodule was flattened into
  the crate root (the types stay re-exported at `ordvec::*`); the test tree
  mirrors this (`tests/rank_index/` → `tests/index/`).
- **Crate version** bumped to `0.2.0`.

### Deprecated

Pre-0.2 names retained as deprecated `pub use` aliases in `src/lib.rs`:
`RankIndex`, `RankQuantIndex`, `BitmapIndex`, `SignBitmapIndex`,
`MultiBucketBitmapIndex` (gated `#[cfg(feature = "experimental")]`),
`RankQuantFastscanIndex` (gated `#[doc(hidden)]`).
Remove these aliases in a future release.

## [0.1.0] - 2026-05-22

Initial release. `ordvec` is the training-free ordinal & sign quantization
substrate for vector retrieval, extracted as a standalone crate from
[turbovec](https://github.com/RyanCodrai/turbovec). It is data-oblivious (no
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
- Red-team regression suites (alpha/beta/gamma/delta) ported from turbovec.

### Notes

- **Minimum supported Rust version (MSRV): 1.89** — the release in which the
  AVX-512 intrinsics this crate relies on were stabilized.
- Dual-licensed under **MIT OR Apache-2.0**.

[Unreleased]: https://github.com/Fieldnote-Echo/ordvec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Fieldnote-Echo/ordvec/releases/tag/v0.1.0
